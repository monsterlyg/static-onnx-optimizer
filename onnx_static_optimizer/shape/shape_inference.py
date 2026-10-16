"""Pass 2: Shape Inference —— 调用 ONNX 官方推导传播已知 shape。

封装官方 API，不做自己实现。

关于合并策略
------------
``infer_shapes`` 返回的是一个**新模型**，它的 value_info 是重新推导的结果。
不能简单地"只在原来没有时才加" —— 导出的模型里 value_info 常常带着
dim_param（例如 ``%1024 = float[N,3,H,W]``），而推导结果里有具体值。
如果保留旧的，``get_shape`` 会一直返回 None，Pass 3 之后全部失效。

所以合并规则是：**取更具体的那个**。
"""

import hashlib
from typing import Optional

import onnx
from onnx import ModelProto, ValueInfoProto

from ..graph.graph_view import GraphView
from ..passes.base import Pass


def run_shape_inference(model: ModelProto,
                        strict: bool = True,
                        data_prop: bool = True) -> bool:
    """就地合并 shape 推导结果。返回是否发生变更。"""
    inferred = _infer(model, strict=strict, data_prop=data_prop) # ModelProto | None
    if inferred is None and strict:
        # strict 模式在部分真实模型上会直接抛错，降级再试一次
        inferred = _infer(model, strict=False, data_prop=data_prop)
    if inferred is None:
        return False

    return _merge_value_info(model, inferred)


def _infer(model: ModelProto, strict: bool,
           data_prop: bool) -> Optional[ModelProto]:
    try:
        return onnx.shape_inference.infer_shapes(
            model,
            check_type=False,
            strict_mode=strict,
            data_prop=data_prop,
        )
    except Exception:
        return None


def _merge_value_info(model: ModelProto,
                      inferred: ModelProto) -> bool:
    """把推导结果合并回原图，冲突时取更具体的一方。"""
    changed = False
    target_graph = model.graph

    existing = {vi.name: vi for vi in target_graph.value_info}

    for vi in inferred.graph.value_info:
        cur = existing.get(vi.name)
        if cur is None:
            target_graph.value_info.append(_copy_vi(vi))
            existing[vi.name] = target_graph.value_info[-1]
            changed = True
        elif _concreteness(vi) > _concreteness(cur):
            cur.type.CopyFrom(vi.type)
            changed = True

    # graph.output 的 shape 也可能被推导得更具体
    inferred_outputs = {o.name: o for o in inferred.graph.output}
    for out in target_graph.output:
        new = inferred_outputs.get(out.name)
        if new is None:
            continue
        if _concreteness(new) > _concreteness(out):
            out.type.CopyFrom(new.type)
            changed = True

    return changed


def _concreteness(vi: ValueInfoProto) -> int:
    """给类型信息打一个"具体程度"分数，用于决定是否覆盖。"""
    if not vi.type.HasField("tensor_type"):
        return 0
    tt = vi.type.tensor_type
    score = 1 if tt.elem_type else 0
    if not tt.HasField("shape"):
        return score
    score += 1
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            score += 2
        else:
            score += 1        # dim_param / unknown 也算有点信息（rank 已知）
    return score


def _copy_vi(vi: ValueInfoProto) -> ValueInfoProto:
    new = ValueInfoProto()
    new.CopyFrom(vi)
    return new


def _graph_signature(graph: GraphView) -> str:
    """图结构的廉价指纹，用于跳过重复的（昂贵的）推导。"""
    h = hashlib.md5()
    for n in graph.graph.node:
        h.update(n.op_type.encode())
        h.update("\x00".join(n.input).encode())
        h.update("\x01".join(n.output).encode())
    for init in graph.graph.initializer:
        h.update(init.name.encode())
    for vi in graph.graph.value_info:
        h.update(vi.name.encode())
    return h.hexdigest()


class ShapeInferencePass(Pass):
    """Pass 2 的 Pass 包装。

    会被反复调用（收敛循环里），所以用一个图指纹跳过无意义的重复推导。
    """

    name = "02_shape_inference"
    priority = 2
    description = "调用 ONNX 官方推导，把已知静态信息传播到全图"

    def __init__(self, config=None):
        super().__init__(config)
        self._last_signature: Optional[str] = None

    def run(self, graph: GraphView) -> bool:
        sig = _graph_signature(graph)
        if sig == self._last_signature:
            return False                    # 图没变，推导结果必然一样

        changed = run_shape_inference(
            graph.model,
            strict=self.config.strict_shape_inference,
            data_prop=self.config.data_prop,
        )
        # 记录的是"推导前"的指纹：只要图不再变，下次就直接跳过
        self._last_signature = _graph_signature(graph) if changed else sig

        if changed:
            self.bump("inference_updated")
        return changed


__all__ = ["run_shape_inference", "ShapeInferencePass"]
