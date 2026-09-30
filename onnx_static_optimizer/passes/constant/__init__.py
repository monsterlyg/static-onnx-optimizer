"""P1：常量相关 Pass。"""

from .constant_propagation import ConstantPropagationPass
from .constant_folding import ConstantFoldingPass

__all__ = ["ConstantPropagationPass", "ConstantFoldingPass"]
