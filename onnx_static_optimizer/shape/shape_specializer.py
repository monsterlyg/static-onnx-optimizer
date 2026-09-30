"""Pass 1: Input Shape Specialization —— 输入形状定形。

把用户指定的具体形状写入 ``graph.input``，消除 ``dim_param``。

这是整条流水线的起点：必须在所有其他 Pass 之前执行，
否则 Shape Inference 拿不到起点，Pass 3 之后的求值全部无从谈起。

支持两种输入形式::

    {"images": [1, 3, 640, 640]}                  # 形式一：完整 shape
    {"images": {"N": 1, "H": 640, "W": 640}}      # 形式二：符号名映射
"""

from typing import Dict, List, Union

from onnx import ModelProto

from ..graph.graph_view import GraphView
from ..passes.base import Pass

ShapeSpec = Union[List[int], Dict[str, int]]


class InputShapeConflict(ValueError):
    """用户给的 shape 与模型里已有的 dim_value 冲突。"""


def specialize_input_shape(model: ModelProto,
                           input_shapes: Dict[str, ShapeSpec],
                           allow_overwrite: bool = False) -> bool:
    """把具体形状写入 graph.input。返回是否发生变更。

    ``allow_overwrite=False`` 时，若目标维与原 dim_value 冲突会抛
    ``InputShapeConflict`` —— 静默覆盖会让用户拿到一个自己没预期的模型。
    """
    if not input_shapes:
        return False

    changed = False
    names = {i.name for i in model.graph.input}

    for name, spec in input_shapes.items():
        if name not in names:
            raise KeyError(f"输入名 {name!r} 不在 graph.input 中；"
                           f"可用输入：{sorted(names)}")

    for inp in model.graph.input:
        spec = input_shapes.get(inp.name)
        if spec is None:
            continue

        tensor_type = inp.type.tensor_type
        shape = tensor_type.shape

        if isinstance(spec, dict):
            target = _resolve_symbolic(shape, spec, inp.name)
        else:
            target = list(spec)

        if len(target) != len(shape.dim):
            raise ValueError(
                f"输入 {inp.name!r} 的 rank 不匹配：模型是 {len(shape.dim)} 维，"
                f"用户给了 {len(target)} 维 {target}"
            )

        # 冲突检查
        for i, (dim, want) in enumerate(zip(shape.dim, target)):
            if dim.HasField("dim_value") and int(dim.dim_value) != int(want):
                if not allow_overwrite:
                    raise InputShapeConflict(
                        f"输入 {inp.name!r} 第 {i} 维已有 dim_value="
                        f"{dim.dim_value}，与用户指定的 {want} 冲突。"
                        f"如确认要覆盖，请设置 allow_overwrite=True。"
                    )

        if _dims_equal(shape, target):
            continue

        # 清空原维度（可能含 dim_param）后重建
        del shape.dim[:]
        for d in target:
            shape.dim.add().dim_value = int(d)
        changed = True

    return changed


def _resolve_symbolic(shape, spec: Dict[str, int], name: str) -> List[int]:
    """形式二：把 dim_param 按名字替换成具体值。"""
    target: List[int] = []
    for i, dim in enumerate(shape.dim):
        if dim.HasField("dim_value"):
            target.append(int(dim.dim_value))
            continue
        param = dim.dim_param
        if param and param in spec:
            target.append(int(spec[param]))
        elif i in spec:
            target.append(int(spec[i]))
        else:
            raise ValueError(
                f"输入 {name!r} 第 {i} 维（dim_param={param!r}）无法从 "
                f"{spec!r} 中确定。请给出完整 shape 或补全符号名映射。"
            )
    return target


def _dims_equal(shape, target) -> bool:
    if len(shape.dim) != len(target):
        return False
    for dim, want in zip(shape.dim, target):
        if not dim.HasField("dim_value") or int(dim.dim_value) != int(want):
            return False
    return True


class InputShapeSpecializationPass(Pass):
    """Pass 1 的 Pass 包装。

    只应在流水线第一轮执行；后续轮次会因为"已无变化"而快速返回。
    """

    name = "01_input_shape_specialization"
    priority = 1
    description = "把用户指定的具体形状写入 graph.input，消除 dim_param"

    def run(self, graph: GraphView) -> bool:
        try:
            changed = specialize_input_shape(
                graph.model, self.config.input_shapes)
        except (KeyError, ValueError) as exc:
            # 定形失败是配置问题，不该被静默吞掉
            raise

        if changed:
            self.bump("specialized_inputs")
        return changed


__all__ = [
    "specialize_input_shape",
    "InputShapeSpecializationPass",
    "InputShapeConflict",
]
