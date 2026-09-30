"""Pass 8: Dynamic Slice Elimination。

静态化 ``Slice`` 的 ``starts/ends/axes/steps``，并在「全量切片」时删除节点。

图模式::

    # 动态 -> 静态
    Slice(X, dynamic_starts, dynamic_ends)   →  Slice(X, [0], [640], axes=[2])

    # 全量切片 -> 删除
    Slice(X, [0], [size(X)], axes=[0])       →  X

注意：ONNX 的 ``starts/ends/axes/steps`` 是**输入**（opset >= 10），
所以"静态化" = 把这些输入替换成 initializer。
"""

from typing import Dict, List, Optional

import numpy as np

from ...graph.graph_editor import bypass_node, materialize_input
from ...graph.node_utils import get_attr, has_attr
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator

# 输入位置 -> 参数名
_PARAM_INDEX = (("starts", 1), ("ends", 2), ("axes", 3), ("steps", 4))


class SliceEliminationPass(Pass):
    """Slice 参数静态化 + 全量切片消除。"""

    name = "08_slice_elimination"
    priority = 8
    description = "静态化动态 Slice，并删除全量切片"

    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(
            graph, max_elements=self.config.max_constant_elements)

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Slice":
                continue

            params, ok = self._resolve_params(graph, node, ev)
            if not ok:
                continue

            # --- 静态化：把动态参数写成 initializer ---
            for key, idx in _PARAM_INDEX:
                value = params.get(key)
                if value is None:
                    continue
                if len(node.input) <= idx or not node.input[idx]:
                    continue
                if graph.is_initializer(node.input[idx]):
                    continue
                dtype = np.int64 if key != "steps" else np.int64
                materialize_input(
                    graph, node, idx,
                    np.atleast_1d(np.asarray(value)).astype(dtype))
                graph.rebuild()
                self.bump(f"{key}_materialized")
                changed = True

            # --- 全量切片 -> 删除 ---
            if self._is_full_slice(graph, node, params):
                if bypass_node(graph, node):
                    graph.rebuild()
                    self.bump("full_slice_removed")
                    changed = True

        return changed

    # ------------------------------------------------------------------
    def _resolve_params(self, graph, node, ev) -> tuple:
        """求值所有参数。starts/ends 求不出来就整体放弃。"""
        params: Dict[str, Optional[np.ndarray]] = {}

        if len(node.input) > 2 and node.input[1] and node.input[2]:
            # opset >= 10：参数是输入
            for key, idx in _PARAM_INDEX:
                if len(node.input) > idx and node.input[idx]:
                    value = ev.evaluate(node.input[idx])
                    if value is None:
                        if key in ("starts", "ends"):
                            self.note("skipped_dynamic")
                            return {}, False    # 核心参数动态 -> 放弃
                        params[key] = None
                    else:
                        params[key] = np.atleast_1d(value).astype(np.int64)
                else:
                    params[key] = None
        else:
            # opset 1：参数是属性，本身已经静态
            params["starts"] = np.array(get_attr(node, "starts"), dtype=np.int64)
            params["ends"] = np.array(get_attr(node, "ends"), dtype=np.int64)
            params["axes"] = (np.array(get_attr(node, "axes"), dtype=np.int64)
                              if has_attr(node, "axes") else None)
            params["steps"] = None

        return params, True

    # ------------------------------------------------------------------
    def _is_full_slice(self, graph, node, params) -> bool:
        """对每个轴检查 start==0、end==dim、step==1。"""
        in_shape = graph.get_shape(node.input[0])
        if in_shape is None:
            return False

        starts = params.get("starts")
        ends = params.get("ends")
        if starts is None or ends is None:
            return False

        rank = len(in_shape)
        axes = params.get("axes")
        if axes is None:
            axes = list(range(len(starts)))
        else:
            axes = [int(a) % rank for a in axes]

        steps = params.get("steps")
        if steps is None:
            steps = [1] * len(starts)

        if len(starts) != len(ends) or len(starts) != len(axes):
            return False
        if len(steps) != len(starts):
            return False

        for s, e, a, st in zip(starts, ends, axes, steps):
            if int(st) != 1:
                return False

            dim = int(in_shape[a])

            s = int(s)
            if s < 0:
                s += dim
            s = max(s, 0)

            # ONNX 会把超出范围的 end 夹到 dim；"切到末尾"常用 INT64_MAX 表示
            e = int(e)
            if e < 0:
                e += dim
            e = min(e, dim)

            if s != 0 or e != dim:
                return False

        return True


__all__ = ["SliceEliminationPass"]
