"""Pass 9: Dynamic Expand Elimination。

静态化 ``Expand`` 的目标 shape；若目标 shape 与输入相同则删除。

图模式::

    Expand(X, dynamic_shape)  →  Expand(X, Constant([1, 80, 6400]))
    Expand(X, X.shape)        →  X            # broadcast 本身已满足

保守策略
--------
``Expand`` 遵循 broadcast 规则：目标 shape 中某维为 1 或等于输入维时不算扩展。
严格判断"是否真的扩展"需要做 broadcast 对齐，第一版只做「完全相等则删除」
这一保守判断 —— 安全且能覆盖相当一部分冗余。
"""

import numpy as np

from ...graph.graph_editor import bypass_node, materialize_input
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator


class ExpandEliminationPass(Pass):
    """Expand 目标 shape 静态化 + no-op 消除。"""

    name = "09_expand_elimination"
    priority = 9
    description = "静态化动态 Expand，并删除目标 shape 与输入相同的 Expand"

    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(
            graph, max_elements=self.config.max_constant_elements)

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Expand":
                continue
            if len(node.input) < 2 or not node.input[1]:
                continue

            data_name, shape_name = node.input[0], node.input[1]
            target = ev.evaluate(shape_name)
            if target is None:
                self.note("skipped_dynamic")
                continue

            static = np.atleast_1d(target).astype(np.int64)

            # --- 静态化 shape 输入 ---
            if not _already_static(graph, shape_name, static):
                materialize_input(graph, node, 1, static)
                graph.rebuild()
                self.bump("shape_materialized")
                changed = True

            # --- 目标 shape 与输入相同 -> 删除 ---
            in_shape = graph.get_shape(data_name)
            if in_shape is not None and list(static) == [int(d) for d in in_shape]:
                if bypass_node(graph, node):
                    graph.rebuild()
                    self.bump("expand_removed")
                    changed = True

        return changed


def _already_static(graph, name: str, value: np.ndarray) -> bool:
    if not graph.is_initializer(name):
        return False
    existing = graph.get_const(name)
    if existing is None or existing.dtype != np.int64:
        return False
    return bool(np.array_equal(existing.ravel(), value.ravel()))


__all__ = ["ExpandEliminationPass"]
