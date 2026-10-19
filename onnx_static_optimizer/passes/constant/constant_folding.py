"""Pass 4: Constant Folding —— 两级策略。

折叠"所有输入都能在编译期确定"的节点，产出写入 **initializer**（NPU 编译器
对 initializer 支持更好，也便于 Pass 15 去重）。

两级策略
--------
Level 1  自实现求值器（``ShapeTensorEvaluator``）
    快、可控、不依赖外部库，覆盖 shape 类与常见算术算子。遇到不认识的
    算子返回 None，交给 Level 2。

Level 2  ORT 子图执行
    以某个节点为根、反向抽出**纯常量子图**，单独交给 ORT CPU 执行。
    用来兜底 Level 1 没有 numpy 实现的算子（``Erf``/``Sin``/``Torch`` 等）。

两者的唯一区别是"值从哪来"；候选筛选、体积守卫、落地方式完全一致，
所以合并在同一个 Pass 里，用 ``_fold_locally`` / ``_fold_with_ort`` 分流。

关键纪律
--------
不要把整个模型交给 ORT 跑一遍来做 Constant Folding。你无法控制哪些节点
被折叠，会破坏后续 Pass 的假设，也可能改变数值行为。所以 Level 2 只做
"以节点为根、反向抽出的纯常量子图"，且子图规模有硬上限。

安全边界
--------
* 随机算子（``RandomNormal`` 等）必须排除，否则每次运行结果不同。
* ``If`` / ``Loop`` / ``Scan`` 的 body 是子图，不能当普通节点折叠。
* 结果元素数超限时放弃，避免把巨大数据张量折成 initializer。
"""

from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np
from onnx import ModelProto, helper

from ...graph.graph_editor import add_initializer, remove_node
from ...graph.node_utils import NEVER_FOLD_OPS
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator

CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})

_ORT = None          # None=未探测, False=不可用, module=可用

# _resolve 的返回值状态
_OK = "ok"
_UNAVAILABLE = "unavailable"      # Level 1 算不出来，可以试 Level 2
_TOO_LARGE = "too_large"          # 体积超限，两个 Level 都不该再试
_NOT_CONE = "not_cone"            # 不是纯常量子图，不算失败
_FAILED = "failed"                # ORT 执行失败（同一轮内不再重试）


def ort_module():
    """惰性探测 onnxruntime。未安装时静默降级，不报错。"""
    global _ORT
    if _ORT is None:
        try:
            import onnxruntime as ort
            _ORT = ort
        except Exception:
            _ORT = False
    return _ORT or None


# ----------------------------------------------------------------------
# Level 2 的支撑：子图抽取
# ----------------------------------------------------------------------
def extract_constant_subgraph(graph, root,
                              max_nodes: int) -> Tuple[Optional[List], Optional[List[str]]]:
    """以 root 为根反向收集纯常量子图。

    返回 ``(nodes_in_topo_order, initializer_names)``；不是纯常量子图则返回
    ``(None, None)``。
    """
    needed = []
    seen = set()
    init_names: List[str] = []
    init_seen = set()
    stack = [root]

    while stack:
        node = stack.pop()
        key = tuple(o for o in node.output if o)
        if not key or key in seen:
            continue
        seen.add(key)

        if node.op_type in CONTROL_FLOW_OPS or node.op_type in NEVER_FOLD_OPS:
            return None, None
        if node.op_type == "Constant":
            # Constant 节点没有输入，直接留在子图里
            needed.append(node)
            continue

        needed.append(node)
        if len(needed) > max_nodes:
            return None, None

        for inp in node.input:
            if not inp:
                continue
            if graph.is_initializer(inp):
                if inp not in init_seen:
                    init_seen.add(inp)
                    init_names.append(inp)
                continue
            producer = graph.producer(inp)
            if producer is None:
                return None, None           # 悬空 -> 不是纯常量子图
            stack.append(producer)

    # 按原图顺序过滤，保证拓扑序
    keys = {tuple(o for o in n.output if o) for n in needed}
    ordered = [n for n in graph.graph.node
               if tuple(o for o in n.output if o) in keys]

    ordered = _local_toposort(ordered)
    if ordered is None:
        return None, None
    return ordered, init_names


def _local_toposort(nodes: List) -> Optional[List]:
    """对给定节点子集做拓扑排序，有环返回 None。"""
    produced = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                produced[o] = i

    indeg = [0] * len(nodes)
    succ: List[List[int]] = [[] for _ in nodes]
    for i, n in enumerate(nodes):
        deps = {produced[x] for x in n.input
                if x and x in produced and produced[x] != i}
        indeg[i] = len(deps)
        for d in deps:
            succ[d].append(i)

    queue = deque(i for i in range(len(nodes)) if indeg[i] == 0)
    order: List[int] = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)

    if len(order) != len(nodes):
        return None
    return [nodes[i] for i in order]


def _value_info_for(graph, name: str):
    """构造一个合法的 ValueInfoProto 供子图声明输出。

    rank 已知但具体值未知时，用 dim_param 占位 —— 不能留空 shape，
    否则 onnx.checker 会报 "Field 'shape' of 'type' is required but missing"。
    """
    dtype = graph.get_dtype(name)
    if dtype is None:
        return None

    shape = graph.get_shape(name)
    if shape is not None:
        return helper.make_tensor_value_info(name, dtype, list(shape))

    rank = graph.get_rank(name)
    if rank is None:
        return None
    return helper.make_tensor_value_info(
        name, dtype, [f"d{i}" for i in range(rank)])


def fold_via_ort(graph, node, config, nodes=None,
                 init_names=None) -> Optional[List[np.ndarray]]:
    """抽出纯常量子图并用 ORT CPU 执行。失败一律返回 None。

    ``nodes`` / ``init_names`` 可由调用方预先抽好，避免重复抽取。
    """
    ort = ort_module()
    if ort is None:
        return None

    if nodes is None:
        nodes, init_names = extract_constant_subgraph(
            graph, node, config.ort_max_subgraph_nodes)
    if not nodes:
        return None

    try:
        inputs = []
        for name in init_names:
            init = graph.initializer_map[name]
            inputs.append(helper.make_tensor_value_info(
                name, init.data_type, list(init.dims)))

        outputs = []
        for o in node.output:
            if not o:
                continue
            vi = _value_info_for(graph, o)
            if vi is None:
                return None
            outputs.append(vi)
        if not outputs:
            return None

        sub_graph = helper.make_graph(nodes, "ort_fold", inputs, outputs)
        sub_model = ModelProto()
        sub_model.graph.CopyFrom(sub_graph)
        sub_model.opset_import.extend(
            [helper.make_opsetid(i.domain, i.version)
             for i in graph.model.opset_import])
        # 必须继承 IR version：ORT 会以 "Missing model IR version" 拒绝
        # 一个没有该字段的模型，导致 Level 2 对所有节点静默失败。
        sub_model.ir_version = graph.model.ir_version
        sub_model.producer_name = "onnx-static-optimizer/ort-fold"

        sess = ort.InferenceSession(
            sub_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = {name: graph.get_const(name) for name in init_names}
        return list(sess.run(None, feeds))
    except Exception:
        return None


# ----------------------------------------------------------------------
class ConstantFoldingPass(Pass):
    """两级常量折叠。"""

    name = "04_constant_folding"
    priority = 4
    description = "折叠所有输入可求值的节点：先自实现求值器，再用 ORT 兜底"

    def run(self, graph) -> bool:
        changed = False
        # 同一轮内不要反复对同一个节点起 ORT session。这两个集合只在本
        # 次 run() 内有效 —— 跨轮（PassManager 的收敛循环）重新尝试是
        # 正确的，因为上游可能刚被折成了常量。
        ort_failed: Set[Tuple[str, ...]] = set()
        noted: Set[Tuple[str, ...]] = set()

        while True:
            batch = self._collect(graph, ort_failed, noted)
            if not batch:
                break

            for node, values, level in batch:
                outs = [o for o in node.output if o]
                for name, value in zip(outs, values):
                    add_initializer(graph, name, np.asarray(value))
                remove_node(graph, node)
                self.bump("ort_folded" if level == 2 else "node_folded")

            graph.rebuild()
            changed = True

        return changed

    # ------------------------------------------------------------------
    def _collect(self, graph, ort_failed, noted):
        """扫一遍图，返回 ``[(node, values, level)]``。

        先收集、再统一落地：折叠一个节点可能让下游也可折叠，而边遍历边改图
        会让索引失效。下一轮 while 会重新扫描。
        """
        ev = ShapeTensorEvaluator(
            graph, max_elements=self.config.max_constant_elements)
        batch = []

        for node in list(graph.graph.node):
            if not self._is_candidate(node):
                continue

            outs = [o for o in node.output if o]
            key = tuple(outs)

            values, status = self._fold_locally(ev, node, outs)
            if status == _OK:
                batch.append((node, values, 1))
                continue
            if status == _TOO_LARGE:
                continue

            if not self.config.enable_ort_folding or key in ort_failed:
                continue

            values, status = self._fold_with_ort(graph, node, outs)
            if status == _OK:
                batch.append((node, values, 2))
            elif status == _FAILED:
                ort_failed.add(key)
                self.note("ort_fold_failed")
            elif status == _NOT_CONE:
                if key not in noted:
                    noted.add(key)
                    self.note("skipped_not_constant_cone")

        return batch

    # ------------------------------------------------------------------
    @staticmethod
    def _is_candidate(node) -> bool:
        if node.op_type == "Constant":
            return False
        if node.op_type in NEVER_FOLD_OPS or node.op_type in CONTROL_FLOW_OPS:
            return False
        if not any(node.input):
            return False          # RandomNormal 之类无输入的算子
        return bool([o for o in node.output if o])

    def _fold_locally(self, ev, node, outs):
        """Level 1：自实现求值器。"""
        if not self.config.enable_constant_propagation:
            return None, _UNAVAILABLE

        results = [ev.evaluate(o) for o in outs]
        if any(v is None for v in results):
            return None, _UNAVAILABLE

        max_elements = self.config.max_constant_elements
        if any(int(v.size) > max_elements for v in results):
            self.note("skipped_too_large")
            return None, _TOO_LARGE
        return results, _OK

    def _fold_with_ort(self, graph, node, outs):
        """Level 2：ORT 子图执行兜底。"""
        if ort_module() is None:
            self.note("skipped_no_ort")
            return None, _UNAVAILABLE

        nodes, init_names = extract_constant_subgraph(
            graph, node, self.config.ort_max_subgraph_nodes)
        if not nodes:
            return None, _NOT_CONE

        values = fold_via_ort(graph, node, self.config, nodes, init_names)
        if values is None or len(values) != len(outs):
            return None, _FAILED

        max_elements = self.config.max_constant_elements
        if any(int(np.asarray(v).size) > max_elements for v in values):
            self.note("skipped_too_large")
            return None, _TOO_LARGE
        return values, _OK


__all__ = [
    "ConstantFoldingPass",
    "CONTROL_FLOW_OPS",
    "fold_via_ort",
    "extract_constant_subgraph",
    "ort_module",
]
