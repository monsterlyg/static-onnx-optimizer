"""Pass 16: Unused ValueInfo Cleanup。

删除指向「已经不存在的 value」的 value_info 条目。

为什么 MVP 里必须有它
---------------------
前 15 个 Pass 会大量改名、bypass、折叠节点，于是 value_info 里会残留一批
**悬空条目** —— 它们引用的 value 已经没有任何生产者了（例如被 bypass 掉的
``Identity`` 的输出、被抵消掉的 ``Transpose`` 的中间张量）。

这些条目对执行无害，但会让 StaticValidator 误报"还有动态维度"：
它看到 ``%tr1 = float[A,B]`` 就认为没静态化，而实际上 ``tr1`` 这个名字
早就不在图里了。

**工具不能把"自己制造的残留"报成"用户的问题"**，所以必须清掉。

只删悬空条目，不删有效条目的原因
--------------------------------
value_info 里的中间张量 shape 对 NPU 编译器是有用信息（内存排布要用），
所以不做"只保留 input/output"的激进清理，只删真正无主的。
"""

from typing import List, Set

from ...passes.base import Pass


def _existing_names(graph) -> Set[str]:
    """图中真实存在的 value 名字集合。"""
    names: Set[str] = set(graph.producer_map)
    names |= set(graph.initializer_map)
    names |= {i.name for i in graph.graph.input}
    names |= {o.name for o in graph.graph.output}
    return names


class UnusedValueInfoCleanupPass(Pass):
    """删除悬空的 value_info 条目。"""

    name = "16_unused_value_info"
    priority = 16
    description = "删除引用了已不存在 value 的 value_info 条目"

    def run(self, graph) -> bool:
        existing = _existing_names(graph)

        dead: List[int] = []
        for i, vi in enumerate(graph.graph.value_info):
            if vi.name not in existing:
                dead.append(i)

        for i in reversed(dead):
            del graph.graph.value_info[i]

        if dead:
            self.bump("value_info_removed", len(dead))
        return bool(dead)


__all__ = ["UnusedValueInfoCleanupPass"]
