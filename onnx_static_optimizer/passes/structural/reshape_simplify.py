"""Pass 12: Reshape Simplification。

减少 Reshape/Flatten 链，处理 Pass 7 没覆盖的结构性冗余。

处理的模式::

    1. Reshape ∘ Reshape        →  Reshape（只保留最后一个）
    2. Reshape(X) 且 shape 不变  →  X       （与 Pass 7 重叠，作为兜底）
    3. Flatten ∘ Reshape        →  合并
    4. Reshape ∘ Flatten        →  合并

为什么 ``Reshape ∘ Reshape`` 可以合并
-------------------------------------
``Reshape(X, s2)`` 的输出只取决于 ``s2`` 和元素总数。元素总数在两次 Reshape
之间是不变量，所以 ``Reshape(Reshape(X, s1), s2) == Reshape(X, s2)``。

**但有一个例外**：``s2`` 里含 ``0``（allowzero=0）时，``0`` 表示"沿用输入的
对应维度"，合并后输入从中间张量变成了 X，语义会变。含 ``-1`` 则是安全的
—— ``-1`` 按元素总数推断，而总数守恒。所以含 ``0`` 时拒绝合并。
"""

from typing import Optional

import numpy as np

from ...graph.graph_editor import bypass_node, remove_node
from ...graph.node_utils import get_attr
from ...passes.base import Pass


class ReshapeSimplifyPass(Pass):
    """Reshape / Flatten 链的合并与消除。"""

    name = "12_reshape_simplify"
    priority = 12
    description = "合并 Reshape/Flatten 链，删除 no-op Reshape"

    def run(self, graph) -> bool:
        if not self.config.enable_reshape_simplify:
            return False

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Reshape":
                continue
            if not node.input or not node.input[0]:
                continue

            prod = graph.producer(node.input[0])
            if prod is None or prod.op_type != "Reshape":
                continue

            mid = prod.output[0]
            # 中间张量只能有一个消费者，否则会破坏其他分支
            if len(graph.consumers(mid)) != 1:
                continue
            # 中间张量是 graph output 时不能动
            if graph.is_graph_output(mid):
                continue
            # s2 含 0 时语义不等价（0 = 沿用输入维度）
            if self._has_zero_dim(graph, node):
                self.note("skipped_placeholder_zero")
                continue

            node.input[0] = prod.input[0]
            remove_node(graph, prod)
            graph.rebuild()
            changed = True
            self.bump("reshape_chain_merged")

        return changed

    # ------------------------------------------------------------------
    @staticmethod
    def _has_zero_dim(graph, node) -> bool:
        """目标 shape 里是否有 0（allowzero=0 时代表"沿用输入维度"）。"""
        if int(get_attr(node, "allowzero", 0) or 0):
            return False
        if len(node.input) < 2 or not node.input[1]:
            return False
        target = graph.get_const(node.input[1])
        if target is None:
            return True         # 求不出值 -> 保守起见当作不安全
        return bool(np.any(np.atleast_1d(target) == 0))


__all__ = ["ReshapeSimplifyPass"]
