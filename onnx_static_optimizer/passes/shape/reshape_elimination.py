"""Pass 7: Dynamic Reshape Elimination。

把 ``Reshape`` 的 shape 输入从「动态计算」变成「静态常量」，
并在等价时直接删除。

三种情况::

    # 情况 1: shape 输入可求值 -> 内联为静态
    Reshape(X, ShapeTensor)      →  Reshape(X, Constant([1, 8400, 80]))

    # 情况 2: 输入输出 shape 相同 -> 删除 Reshape
    Reshape(X, [1, 8400, 80])    →  X            (X.shape == [1, 8400, 80])

    # 情况 3: shape 里含 0 / -1 等占位符 -> 展开为具体值
    Reshape(X, [0, -1, 80])      →  Reshape(X, [1, 8400, 80])
"""

from typing import List, Optional

import numpy as np

from ...graph.graph_editor import bypass_node, materialize_input
from ...graph.node_utils import get_attr
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator


def resolve_placeholders(target, in_shape: Optional[List[int]],
                         allowzero: bool = False) -> Optional[List[int]]:
    """把 ``0`` / ``-1`` 展开成具体值。无法解析时返回 None。

    * ``0`` 表示「沿用输入的对应维度」（``allowzero=1`` 时是字面量 0）。
    * ``-1`` 表示「该维度自动推断」。
    * 二者都要求 ``in_shape`` 已知。
    """
    t = [int(x) for x in np.atleast_1d(target)]

    if allowzero:
        need_infer = -1 in t
        if not need_infer:
            return t
    elif 0 in t or -1 in t:
        if in_shape is None:
            return None

    out: List[int] = []
    infer_idx = -1
    for i, d in enumerate(t):
        if d == 0 and not allowzero:
            if in_shape is None or i >= len(in_shape):
                return None
            out.append(int(in_shape[i]))
        elif d == -1:
            if infer_idx >= 0:
                return None                 # 多个 -1 无法推断
            infer_idx = i
            out.append(-1)
        else:
            out.append(d)

    if infer_idx >= 0:
        if in_shape is None:
            return None
        total = 1
        for d in in_shape:
            total *= int(d)
        known = 1
        for i, d in enumerate(out):
            if i != infer_idx:
                known *= d
        if known == 0 or total % known != 0:
            return None
        out[infer_idx] = total // known

    # 语义校验：元素总数必须守恒，否则这个 Reshape 本身就是非法的
    if in_shape is not None:
        total_in = 1
        for d in in_shape:
            total_in *= int(d)
        total_out = 1
        for d in out:
            total_out *= int(d)
        if total_in != total_out:
            return None

    return out


class ReshapeEliminationPass(Pass):
    """Reshape 的 shape 输入静态化 + no-op 消除。"""

    name = "07_reshape_elimination"
    priority = 7
    description = "静态化动态 Reshape，并删除 no-op Reshape"

    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(
            graph, max_elements=self.config.max_constant_elements)

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Reshape":
                continue

            data_name = node.input[0]
            shape_name = node.input[1] if len(node.input) > 1 else ""

            target = ev.evaluate(shape_name)
            if target is None:
                continue                    # shape 输入仍动态 -> 本 Pass 无能为力

            allowzero = bool(int(get_attr(node, "allowzero", 0) or 0))
            in_shape = graph.get_shape(data_name)

            resolved = resolve_placeholders(target, in_shape, allowzero)
            if resolved is None:
                self.note("skipped_unresolvable")
                continue

            # --- 静态化：把 shape 输入写成 int64 initializer ---
            # Reshape 的 shape 输入必须是 int64，求值过程中可能被提升成别的类型
            static = np.array(resolved, dtype=np.int64)
            if not _is_static_int64(graph, shape_name, static):
                materialize_input(graph, node, 1, static)
                graph.rebuild()
                self.bump("shape_materialized")
                changed = True

            # --- 情况 2：输入输出 shape 相同 -> 直接删除 ---
            if in_shape is not None and list(resolved) == [int(d) for d in in_shape]:
                if bypass_node(graph, node):
                    graph.rebuild()
                    self.bump("reshape_removed")
                    changed = True

        return changed


def _is_static_int64(graph, name: str, value: np.ndarray) -> bool:
    """shape 输入已经是值相同的 int64 initializer 时无需重复写。"""
    if not graph.is_initializer(name):
        return False
    existing = graph.get_const(name)
    if existing is None:
        return False
    if existing.dtype != np.int64:
        return False
    return bool(np.array_equal(existing.ravel(), value.ravel()))


__all__ = ["ReshapeEliminationPass", "resolve_placeholders"]
