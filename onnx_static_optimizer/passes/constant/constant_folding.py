"""Pass 5: Constant Folding (Level 2 —— ONNX Runtime 子图执行)。

在 Pass 4 覆盖不到时，用 ONNX Runtime CPU 执行**小子图**来完成折叠。

为什么需要它
------------
Pass 4 只能折叠自己实现了的算子。像 ``Conv`` 权重 + ``BN`` 这类复杂常量
计算，自己实现成本太高。ORT 是兜底。

**关键纪律（文档反复强调）**

    不要把整个模型交给 ORT 跑一遍来做 Constant Folding。

原因：你无法控制哪些节点被折叠，会破坏后续 Pass 的假设，
也可能改变数值行为。所以这里只做"以某个节点为根、反向抽出的纯常量子图"，
子图规模还有硬上限。
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import onnx
from onnx import ModelProto, helper

from ...graph.graph_editor import add_initializer, remove_node
from ...graph.node_utils import NEVER_FOLD_OPS
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator

CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})

_ORT = None          # None=未探测, False=不可用, module=可用


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
# 子图抽取
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
        # 一个没有该字段的模型，导致本 Pass 对所有节点静默失败。
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
    """ORT 兜底折叠。"""

    name = "05_constant_folding"
    priority = 5
    description = "用 ORT CPU 执行纯常量小子图来完成折叠（Pass 4 的兜底）"

    def run(self, graph) -> bool:
        if not self.config.enable_ort_folding:
            return False
        if ort_module() is None:
            # 没装 ORT：静默降级，Pass 4 的自实现算子仍然可用
            self.note("skipped_no_ort")
            return False

        ev = ShapeTensorEvaluator(
            graph, max_elements=self.config.max_constant_elements)
        max_elements = self.config.max_constant_elements
        changed = False

        for node in list(graph.graph.node):
            if node.op_type == "Constant":
                continue
            if node.op_type in NEVER_FOLD_OPS or node.op_type in CONTROL_FLOW_OPS:
                continue

            outs = [o for o in node.output if o]
            if not outs or not any(node.input):
                continue

            # Pass 4 已经能处理的，交给 Pass 4 —— 避免重复工作
            if all(ev.evaluate(o) is not None for o in outs):
                continue

            # 先抽子图：只有纯常量依赖的才值得起一个 ORT session。
            # 把"根本不该试"和"试了但失败"分开记，报告才有诊断价值。
            nodes, init_names = extract_constant_subgraph(
                graph, node, self.config.ort_max_subgraph_nodes)
            if not nodes:
                self.note("skipped_not_constant_cone")
                continue

            values = fold_via_ort(graph, node, self.config, nodes, init_names)
            if values is None or len(values) != len(outs):
                self.note("ort_fold_failed")
                continue
            if any(int(v.size) > max_elements for v in values):
                self.note("skipped_too_large")
                continue

            for name, value in zip(outs, values):
                add_initializer(graph, name, np.asarray(value))
            remove_node(graph, node)
            graph.rebuild()
            ev = ShapeTensorEvaluator(graph, max_elements=max_elements)
            changed = True
            self.bump("node_folded")

        return changed


__all__ = [
    "ConstantFoldingPass",
    "fold_via_ort",
    "extract_constant_subgraph",
    "ort_module",
]
