"""Pass 基类、PassManager 与 15 个 Pass 的导出。"""

from .base import Pass, PassManager

from .shape import (
    ExpandEliminationPass,
    ReshapeEliminationPass,
    ShapeEliminationPass,
    ShapeTensorEvaluationPass,
    SliceEliminationPass,
)
from .constant import ConstantFoldingPass, ConstantPropagationPass
from .structural import (
    CastSimplifyPass,
    IdentityEliminationPass,
    ReshapeSimplifyPass,
    TransposeSimplifyPass,
)
from .cleanup import (
    DCEPass,
    InitializerCleanupPass,
    UnusedValueInfoCleanupPass,
)

__all__ = [
    "Pass",
    "PassManager",
    # 动转静核心
    "ShapeTensorEvaluationPass",
    "ConstantPropagationPass",
    "ConstantFoldingPass",
    "ShapeEliminationPass",
    "ReshapeEliminationPass",
    "SliceEliminationPass",
    "ExpandEliminationPass",
    # 结构简化
    "IdentityEliminationPass",
    "ReshapeSimplifyPass",
    "TransposeSimplifyPass",
    "CastSimplifyPass",
    # 清理
    "DCEPass",
    "InitializerCleanupPass",
    "UnusedValueInfoCleanupPass",
]
