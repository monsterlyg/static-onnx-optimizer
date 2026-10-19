"""P0：动转静核心 Pass（shape 相关）。"""

from .eliminate_shape import ShapeEliminationPass
from .reshape_elimination import ReshapeEliminationPass
from .slice_elimination import SliceEliminationPass
from .expand_elimination import ExpandEliminationPass

__all__ = [
    "ShapeEliminationPass",
    "ReshapeEliminationPass",
    "SliceEliminationPass",
    "ExpandEliminationPass",
]
