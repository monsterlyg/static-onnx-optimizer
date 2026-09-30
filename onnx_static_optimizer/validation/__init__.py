"""验证体系：ONNX 标准检查、静态性验证、数值等价验证。"""

from .checker import (
    check_model,
    make_feeds,
    run_model,
    run_model_ex,
    static_input_shapes,
    validate_numerically,
)
from .static_validator import StaticValidationReport, validate

__all__ = [
    "check_model",
    "make_feeds",
    "run_model",
    "run_model_ex",
    "static_input_shapes",
    "validate_numerically",
    "StaticValidationReport",
    "validate",
]
