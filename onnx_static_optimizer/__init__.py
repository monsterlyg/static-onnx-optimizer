"""ONNX 动转静优化器。

将一个带动态维度的 ONNX 模型转换为 NPU 编译器友好的静态模型。

    from onnx_static_optimizer import optimize, OptimizerConfig

    result = optimize("yolov8n.onnx", {"images": [1, 3, 640, 640]})
    result.save("yolov8n_static.onnx")
"""

from .config import OptimizerConfig
from .optimizer import OptimizationResult, optimize
from .graph.graph_view import GraphView

__all__ = [
    "OptimizerConfig",
    "OptimizationResult",
    "optimize",
    "GraphView",
]

__version__ = "0.1.0"
