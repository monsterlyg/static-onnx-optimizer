"""节点级工具：属性读写、节点构造。

ONNX 的属性是 proto，Python 侧没有"按名字取值"的语法糖，
所有 Pass 都要用到，所以统一封装。
"""

from typing import Any, List, Optional

import numpy as np
from onnx import AttributeProto, NodeProto, TensorProto, helper, numpy_helper

# 绝对不该被求值/折叠的算子：
# 随机算子输出不确定，有状态/副作用算子折叠会改变语义。
NEVER_FOLD_OPS = frozenset({
    "RandomNormal",
    "RandomUniform",
    "RandomNormalLike",
    "RandomUniformLike",
    "Multinomial",
    "Bernoulli",
    "Dropout",
    "BatchNormalization",  # 推理态可折叠，但训练态不行 —— 交给 Pass 5 判断
})


def get_attr(node: NodeProto, name: str, default: Any = None) -> Any:
    """读取节点属性，不存在返回 default。"""
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def has_attr(node: NodeProto, name: str) -> bool:
    return any(a.name == name for a in node.attribute)


def set_attr(node: NodeProto, name: str, value: Any) -> None:
    """写入节点属性（已存在则覆盖）。

    由 Python 值的类型推断 AttributeProto 的 type，避免手工传 type 出错。
    """
    for i, a in enumerate(node.attribute):
        if a.name == name:
            del node.attribute[i]
            break

    attr = node.attribute.add()
    attr.name = name

    if isinstance(value, bool):
        attr.type = AttributeProto.INT
        attr.i = int(value)
    elif isinstance(value, (int, np.integer)):
        attr.type = AttributeProto.INT
        attr.i = int(value)
    elif isinstance(value, (float, np.floating)):
        attr.type = AttributeProto.FLOAT
        attr.f = float(value)
    elif isinstance(value, (bytes, str)):
        attr.type = AttributeProto.STRING
        attr.s = value.encode() if isinstance(value, str) else value
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.floating):
            attr.type = AttributeProto.FLOATS
            attr.floats.extend([float(x) for x in arr.ravel()])
        else:
            attr.type = AttributeProto.INTS
            attr.ints.extend([int(x) for x in arr.ravel()])
    else:
        raise TypeError(f"不支持的属性类型: {name}={value!r} ({type(value)})")


def remove_attr(node: NodeProto, name: str) -> bool:
    for i, a in enumerate(node.attribute):
        if a.name == name:
            del node.attribute[i]
            return True
    return False


def make_node(op_type: str, inputs: List[str], outputs: List[str],
              name: str = "", **attrs) -> NodeProto:
    """构造节点；attrs 里的 None 会被忽略。"""
    node = helper.make_node(op_type, inputs, outputs, name=name)
    for k, v in attrs.items():
        if v is not None:
            set_attr(node, k, v)
    return node


def make_constant_node(output_name: str, arr: np.ndarray,
                       name: str = "") -> NodeProto:
    """构造一个产出 arr 的 Constant 节点。

    Constant 的 value 属性里张量自带名字，这里统一用 "<output>_const"，
    便于调试时辨认。
    """
    tensor = numpy_helper.from_array(np.asarray(arr), f"{output_name}_const")
    node = helper.make_node(
        "Constant", [], [output_name],
        name=name or f"Constant_{output_name}",
    )
    node.attribute.append(
        helper.make_attribute("value", tensor)
    )
    return node


def constant_node_value(node: NodeProto) -> Optional[np.ndarray]:
    """从 Constant 节点取回它的值。"""
    for a in node.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
        if a.name == "value_float":
            return np.array(a.f, dtype=np.float32)
        if a.name == "value_int":
            return np.array(a.i, dtype=np.int64)
        if a.name == "value_floats":
            return np.array(list(a.floats), dtype=np.float32)
        if a.name == "value_ints":
            return np.array(list(a.ints), dtype=np.int64)
    return None


__all__ = [
    "NEVER_FOLD_OPS",
    "get_attr",
    "has_attr",
    "set_attr",
    "remove_attr",
    "make_node",
    "make_constant_node",
    "constant_node_value",
]
