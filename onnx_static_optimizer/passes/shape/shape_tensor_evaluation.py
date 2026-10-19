"""Pass 3: Shape Tensor Evaluation —— 把 shape 子图整体替换为 Constant。

与 0.3 的关系
-------------
0.3（``shape/shape_evaluator.py``）是**引擎**，这里是把引擎**接入图重写**的 Pass。
引擎只回答"这个值能不能算出来"，本 Pass 负责"算出来之后怎么改图"。

图模式::

    Shape → Gather → Unsqueeze → Concat → Cast → Reshape
                        ↓ 求值
                  Constant([1, 8400, 80])

候选收集策略
------------
采用文档推荐的**方式 A（保守）**：只从 ``Reshape / Slice / Expand / Tile /
Split / Pad / Resize`` 这类"消费 shape"的算子的 shape 参数输入出发，
反向收集依赖锥。比"遍历所有 int64 value_info"安全得多，且正好命中真实痛点
—— PyTorch 导出的动态 shape 几乎都喂给这几个算子。

关于剪枝
--------
本 Pass 不自己做 orphan 剪枝。替换后上游的 Shape/Gather/Mul 变成死代码，
由 Pass 10 (DCE) 在同一轮收敛循环里清掉。这样避免两份重复的剪枝逻辑。
"""

from typing import Dict, List, Optional, Set, Tuple

from ...graph.graph_editor import replace_node_with_constant
from ...graph.node_utils import NEVER_FOLD_OPS
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator

# 哪些算子的第几个输入是「shape 参数」
SHAPE_CONSUMERS: Dict[str, Tuple[int, ...]] = {
    "Reshape": (1,),
    "Slice": (1, 2, 3, 4),
    "Expand": (1,),
    "Tile": (1,),
    "Split": (1,),
    "Pad": (1,),
    "Resize": (1, 2, 3),
    "Upsample": (1,),
    "TopK": (1,),
    "OneHot": (1,),
    "ConstantOfShape": (0,),
}

# 控制流算子的 body 是子图，不能按普通节点处理
CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})

# 反向锥的规模上限，防止病态图把时间吃光
MAX_CONE = 4096


class ShapeTensorEvaluationPass(Pass):
    """求值 shape 子图，产出 Constant 节点。"""

    name = "03_shape_tensor_evaluation"
    priority = 3
    description = "把求值结果已知的 shape 子图整体替换为 Constant 节点"

    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(
            graph, max_elements=self.config.max_constant_elements)

        changed = False
        for name in self._collect_candidates(graph):
            # 已经是常量 -> 无需处理
            if graph.is_initializer(name):
                continue
            node = graph.producer(name)
            if node is None or node.op_type == "Constant":
                continue
            # 本 Pass 按单个 value 折叠，而 replace_node_with_constant 只会
            # 用 node.output[0] 建 Constant。多输出节点（Split 等）走这条路
            # 会写错名字、并让兄弟输出凭空消失（checker 报 "input 'b' ... is
            # not output of any previous nodes"）。这类节点交给 Pass 4 ——
            # 它对所有输出统一写 initializer。
            if len([o for o in node.output if o]) > 1:
                self.note("skipped_multi_output")
                continue
            # 折叠一个 value 会让它的整条上游变成死代码。那些上游也在候选集里，
            # 如果不跳过，同一个子图会被折叠十几次，产出大量马上被 DCE 删掉的
            # 冗余 Constant。这里只折"还有人用"的那个。
            if not graph.is_graph_output(name) and not graph.consumers(name):
                continue

            value = ev.evaluate(name)
            if value is None:
                continue

            if replace_node_with_constant(graph, node, value):
                graph.rebuild()          # 立刻重建，后续判断用新索引
                changed = True
                self.bump("shape_subgraph_folded")

        return changed

    # ------------------------------------------------------------------
    def _collect_candidates(self, graph) -> List[str]:
        """从 shape 消费者的参数输入出发，反向收集依赖锥。"""
        roots: List[str] = []
        for node in graph.graph.node:
            indices = SHAPE_CONSUMERS.get(node.op_type)
            if not indices:
                continue
            for i in indices:
                if len(node.input) > i and node.input[i]:
                    roots.append(node.input[i])

        cone: List[str] = []
        seen: Set[str] = set()
        stack: List[str] = list(roots)
        # 反向 BFS 收集依赖锥
        while stack:
            name = stack.pop()
            if name in seen or not name:
                continue
            seen.add(name)

            if graph.is_initializer(name):
                continue
            node = graph.producer(name)
            if node is None or node.op_type == "Constant":
                continue
            if node.op_type in NEVER_FOLD_OPS or node.op_type in CONTROL_FLOW_OPS:
                continue

            cone.append(name)
            if len(cone) > MAX_CONE:
                break
            for inp in node.input:
                if inp:
                    stack.append(inp)

        return cone


__all__ = ["ShapeTensorEvaluationPass", "SHAPE_CONSUMERS"]
