"""Pass 10: Dead Code Elimination (DCE)。

删除所有「无法到达 graph output」的节点。

为什么不能写成 ``if len(node.output) == 0: delete``
--------------------------------------------------
那样只能删"无输出"的节点，而真正的死代码是"**有输出但没人用**"的节点。
必须做反向可达性分析。

子图注意
--------
``If`` / ``Loop`` / ``Scan`` 的 body 可以引用**外层**的 value。如果一个外层
节点的输出只被子图引用，它在主图里"看起来"没人用，但删掉会让子图失去输入。
所以子图引用到的名字必须一并视为 live。
"""

from typing import List, Set

from ...passes.base import Pass


class DCEPass(Pass):
    """从 graph output 反向可达性分析，删除不可达节点。"""

    name = "10_dce"
    priority = 10
    description = "删除所有无法到达 graph output 的节点"

    def run(self, graph) -> bool:
        if not self.config.enable_dce:
            return False

        live: Set[str] = set(graph.graph_output_names)
        stack: List[str] = list(live)

        # 子图引用的外层 value 也算 live
        for node in graph.subgraph_nodes():
            for inp in node.input:
                if inp and inp not in live:
                    live.add(inp)
                    stack.append(inp)

        # 反向 BFS：从 output 出发，沿 producer 往上走
        while stack:
            name = stack.pop()
            node = graph.producer(name)
            if node is None:
                continue                    # input / initializer，到头了
            for inp in node.input:
                if inp and inp not in live:
                    live.add(inp)
                    stack.append(inp)

        # 删除所有输出都不在 live 里的节点
        dead: List[int] = []
        for i, node in enumerate(graph.graph.node):
            outs = [o for o in node.output if o]
            if not outs:
                continue                    # 无输出的节点不碰
            if not any(o in live for o in outs):
                dead.append(i)

        for i in reversed(dead):            # 逆序删除，避免位移影响
            del graph.graph.node[i]

        if dead:
            self.bump("node_removed", len(dead))
        return bool(dead)


__all__ = ["DCEPass"]
