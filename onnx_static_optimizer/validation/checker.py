"""ONNX 标准检查 + 数值等价验证。

数值等价是最后一道防线：把原模型和优化后的模型都跑一遍 ORT，
比较输出。它能抓住那些"图合法、静态性也通过、但语义被改坏了"的问题。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import ModelProto

from ..graph.tensor_utils import onnx_dtype_to_numpy


def check_model(model: ModelProto) -> Tuple[bool, str]:
    """调用 onnx.checker。"""
    try:
        onnx.checker.check_model(model)
        return True, "PASS"
    except Exception as exc:
        return False, str(exc).strip().splitlines()[0] if str(exc) else repr(exc)


def _ort():
    from ..passes.constant.constant_folding import ort_module
    return ort_module()


def static_input_shapes(model: ModelProto) -> Optional[Dict[str, List[int]]]:
    """取出所有输入的静态 shape；有任何一个非静态就返回 None。"""
    shapes: Dict[str, List[int]] = {}
    for inp in model.graph.input:
        if not inp.type.HasField("tensor_type"):
            return None
        tt = inp.type.tensor_type
        if not tt.HasField("shape"):
            return None
        dims = []
        for d in tt.shape.dim:
            if not d.HasField("dim_value"):
                return None
            dims.append(int(d.dim_value))
        shapes[inp.name] = dims
    return shapes


def make_feeds(model: ModelProto, input_shapes: Dict[str, List[int]],
               seed: int = 0) -> Optional[Dict[str, np.ndarray]]:
    """按输入声明生成随机 feed。"""
    rng = np.random.default_rng(seed)
    feeds: Dict[str, np.ndarray] = {}

    for inp in model.graph.input:
        if inp.name in input_shapes:
            shape = list(input_shapes[inp.name])
        else:
            return None
        if not inp.type.HasField("tensor_type"):
            return None

        elem_type = inp.type.tensor_type.elem_type
        try:
            np_dtype = onnx_dtype_to_numpy(elem_type)
        except Exception:
            return None

        if np.issubdtype(np_dtype, np.floating):
            data = rng.random(shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            data = rng.integers(0, 8, size=shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.bool_):
            data = rng.integers(0, 2, size=shape).astype(np_dtype)
        else:
            return None
        feeds[inp.name] = data

    return feeds


def run_model(model: ModelProto, feeds: Dict[str, np.ndarray]):
    """跑一次 ORT，失败返回 None。"""
    outs, _ = run_model_ex(model, feeds)
    return outs


def run_model_ex(model: ModelProto, feeds: Dict[str, np.ndarray]):
    """跑一次 ORT，返回 ``(outputs, error_message)``。

    把 ORT 的报错原样带出来 —— 数值验证失败时，"哪个模型、为什么失败"
    比一句"执行失败"有用得多。
    """
    ort = _ort()
    if ort is None:
        return None, "onnxruntime 不可用"
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        return list(sess.run(None, feeds)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"


def validate_numerically(original: ModelProto,
                         optimized: ModelProto,
                         input_shapes: Dict[str, List[int]],
                         rtol: float = 1e-4,
                         atol: float = 1e-5,
                         seed: int = 0) -> Tuple[bool, str]:
    """跑两个模型比较输出。"""
    if _ort() is None:
        return False, "onnxruntime 不可用，跳过数值验证"

    feeds = make_feeds(original, input_shapes, seed=seed)
    if feeds is None:
        return False, "无法构造输入 feed（输入 shape 未完全静态）"

    ref, ref_err = run_model_ex(original, feeds)
    if ref is None:
        return False, f"原模型 ORT 执行失败：{ref_err}"

    opt, opt_err = run_model_ex(optimized, feeds)
    if opt is None:
        return False, f"优化后模型 ORT 执行失败：{opt_err}"

    if len(ref) != len(opt):
        return False, f"输出个数不一致：原 {len(ref)} vs 优化后 {len(opt)}"

    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, opt)):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return False, f"输出 {i} shape 不一致：{a.shape} vs {b.shape}"
        a64 = a.astype(np.float64, copy=False)
        b64 = b.astype(np.float64, copy=False)
        diff = np.abs(a64 - b64)
        scale = np.maximum(np.abs(a64), np.abs(b64))
        # 逐元素容差，等价于 np.allclose 的判据
        ok = diff <= (atol + rtol * scale)
        if not np.all(ok):
            bad = int(diff.size - np.count_nonzero(ok))
            return False, (f"输出 {i} 数值不一致：{bad}/{diff.size} 个元素超差，"
                           f"最大绝对误差 {float(diff.max()):.3e}")
        if diff.size:
            worst = max(worst, float(diff.max()))

    return True, f"PASS（最大绝对误差 {worst:.3e}）"


__all__ = [
    "check_model",
    "validate_numerically",
    "static_input_shapes",
    "make_feeds",
    "run_model",
    "run_model_ex",
]
