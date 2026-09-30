"""TensorProto <-> numpy 互转与 dtype 工具。"""

from typing import Optional

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def tensor_to_numpy(tensor: TensorProto) -> np.ndarray:
    """TensorProto -> np.ndarray。

    bfloat16 等 numpy 没有原生对应的类型会退化成 object/float32，
    此时由调用方（求值器）自行决定是否降级。
    """
    return numpy_helper.to_array(tensor)


def numpy_to_tensor(arr: np.ndarray, name: str = "") -> TensorProto:
    """np.ndarray -> TensorProto。"""
    return numpy_helper.from_array(np.asarray(arr), name)


def onnx_dtype_to_numpy(dtype: int):
    """ONNX TensorProto 枚举 -> numpy dtype。不支持的 dtype 抛 ValueError。"""
    return helper.tensor_dtype_to_np_dtype(dtype)


def numpy_dtype_to_onnx(np_dtype) -> int:
    """numpy dtype -> ONNX TensorProto 枚举。"""
    return helper.np_dtype_to_tensor_dtype(np.dtype(np_dtype))


def is_integer_dtype(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def element_count(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def broadcast_shape(*shapes) -> Optional[list]:
    """按 numpy 广播规则求结果 shape，失败返回 None。"""
    try:
        return list(np.broadcast_shapes(*[tuple(s) for s in shapes]))
    except Exception:
        return None


__all__ = [
    "tensor_to_numpy",
    "numpy_to_tensor",
    "onnx_dtype_to_numpy",
    "numpy_dtype_to_onnx",
    "is_integer_dtype",
    "element_count",
    "broadcast_shape",
    "TensorProto",
]
