"""P5：清理 Pass。"""

from .dce import DCEPass
from .unused_initializer import InitializerCleanupPass
from .unused_value_info import UnusedValueInfoCleanupPass

__all__ = ["DCEPass", "InitializerCleanupPass", "UnusedValueInfoCleanupPass"]
