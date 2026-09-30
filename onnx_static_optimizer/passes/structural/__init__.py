"""P2：结构简化 Pass。"""

from .identity_elimination import IdentityEliminationPass
from .reshape_simplify import ReshapeSimplifyPass
from .transpose_simplify import TransposeSimplifyPass
from .cast_simplify import CastSimplifyPass

__all__ = [
    "IdentityEliminationPass",
    "ReshapeSimplifyPass",
    "TransposeSimplifyPass",
    "CastSimplifyPass",
]
