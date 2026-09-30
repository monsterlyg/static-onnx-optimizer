"""Pass 11: Identity Elimination。

``A → Identity → B`` 改为 ``A → B``。

最简单的 Pass，但一定要做 —— PyTorch 导出经常产生 Identity。

例外处理
--------
如果 ``Identity`` 的输出是 graph output，而输入也是别的东西，
改名时要小心别让两个 output 撞名。``bypass_node`` 已经在撞名时放弃改写，
保证不会产生非法图。
"""

from ...graph.graph_editor import bypass_node
from ...passes.base import Pass


class IdentityEliminationPass(Pass):
    """删除纯拷贝的 Identity 节点。"""

    name = "11_identity_elimination"
    priority = 11
    description = "A → Identity → B 改为 A → B"

    def run(self, graph) -> bool:
        if not self.config.enable_identity_elimination:
            return False

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Identity":
                continue
            if not node.input or not node.input[0]:
                continue
            if bypass_node(graph, node):
                graph.rebuild()
                changed = True
                self.bump("identity_removed")

        return changed


__all__ = ["IdentityEliminationPass"]
