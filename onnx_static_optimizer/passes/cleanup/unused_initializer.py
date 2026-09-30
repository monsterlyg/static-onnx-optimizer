"""Pass 15: Initializer Cleanup。

删除没有任何消费者引用的 initializer，减小模型体积。

子图注意
--------
``If`` / ``Loop`` / ``Scan`` 的 body 里也引用 initializer，扫描时必须**递归
进子图**收集，否则会误删 —— 这类 bug 只在跑带控制流的模型时才会暴露。

该 Pass 通常放最后，因为前面的折叠会不断产生新常量、淘汰旧常量。
"""

from typing import List, Set

from ...passes.base import Pass


class InitializerCleanupPass(Pass):
    """删除未被引用的 initializer。"""

    name = "15_initializer_cleanup"
    priority = 15
    description = "删除没有任何消费者引用的 initializer"

    def run(self, graph) -> bool:
        if not self.config.enable_initializer_cleanup:
            return False

        used: Set[str] = set()

        # 主图节点
        for node in graph.graph.node:
            used.update(i for i in node.input if i)
        # 子图节点（Loop / If body）
        for node in graph.subgraph_nodes():
            used.update(i for i in node.input if i)
        # graph output 直接引用 initializer 的情况
        used.update(graph.graph_output_names)

        dead: List[int] = []
        for i, init in enumerate(graph.graph.initializer):
            if init.name not in used:
                dead.append(i)

        for i in reversed(dead):
            del graph.graph.initializer[i]

        if dead:
            self.bump("initializer_removed", len(dead))
        return bool(dead)


__all__ = ["InitializerCleanupPass"]
