"""Pass 6: Shape / Size / Rank Elimination。

把 ``Shape(X)`` / ``Size(X)`` / ``Rank(X)`` 直接替换为常量。

这三种算子本质是 Pass 3 的特例，但因为太常见，单独做成 Pass 便于统计与调试。

图模式::

    Shape(X)  →  Constant([1, 3, 640, 640])
    Size(X)   →  Constant(1228800)
    Rank(X)   →  Constant(4)

前置条件：依赖 Pass 1 + Pass 2 已经把静态 shape 传播到位。
"""

from typing import List, Optional

import numpy as np

from ...graph.graph_editor import replace_node_with_initializer
from ...graph.node_utils import get_attr
from ...passes.base import Pass


class ShapeEliminationPass(Pass):
    """Shape / Size / Rank -> 常量。"""

    name = "06_shape_elimination"
    priority = 6
    description = "把 Shape/Size/Rank 直接替换为常量"

    # 注：ONNX 标准域里没有 Rank 算子（只有 Shape / Size）。
    # 保留是为了兼容非标准导出，实际模型里通常见到的是 Size(Shape(x))。
    HANDLED = frozenset({"Shape", "Size", "Rank"})

    def run(self, graph) -> bool:
        changed = False
        for node in list(graph.graph.node):
            if node.op_type not in self.HANDLED:
                continue

            x = node.input[0] if node.input else ""
            shape = graph.get_shape(x)
            if shape is None:
                # 静态 shape 未知 -> 放弃（绝不猜）
                self.note("skipped_dynamic")
                continue

            value = self._compute(node, shape)
            if value is None:
                continue

            if replace_node_with_initializer(graph, node, value):
                graph.rebuild()
                changed = True
                self.bump(f"eliminated_{node.op_type.lower()}")

        return changed

    # ------------------------------------------------------------------
    @staticmethod
    def _compute(node, shape: List[int]) -> Optional[np.ndarray]:
        if node.op_type == "Shape":
            # opset >= 15 才有 start / end
            start = get_attr(node, "start", 0) or 0
            end = get_attr(node, "end", None)
            return np.array(shape, dtype=np.int64)[start:end]

        if node.op_type == "Size":
            n = 1
            for d in shape:
                n *= int(d)
            return np.array(n, dtype=np.int64)

        if node.op_type == "Rank":
            return np.array(len(shape), dtype=np.int64)

        return None


__all__ = ["ShapeEliminationPass"]
