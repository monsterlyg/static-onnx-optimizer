"""P1：常量相关 Pass。

原 ``constant_propagation`` 与 ``constant_folding`` 已合并为单个两级折叠 Pass
（求值器优先，ORT 兜底）。``ConstantPropagationPass`` 保留为别名，避免外部
导入失效。
"""

from .constant_folding import ConstantFoldingPass

ConstantPropagationPass = ConstantFoldingPass

__all__ = ["ConstantFoldingPass", "ConstantPropagationPass"]
