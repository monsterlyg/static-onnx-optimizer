"""Pass 13: Transpose Simplification。

消除或合并 Transpose 链::

    1. Transpose(perm=identity)           →  删除
    2. Transpose(p) ∘ Transpose(inv(p))   →  删除
    3. Transpose(p) ∘ Transpose(q)        →  Transpose(q ∘ p)   （合并）

perm 合成的推导
---------------
``Transpose(p)`` 的语义是 ``out[i] = in[p[i]]``。
若有 ``A --q--> B --p--> C``，则::

    C[i] = B[p[i]] = A[q[p[i]]]

所以合成后的 perm 是 ``[q[i] for i in p]``。

默认 perm 是**逆序**（``[n-1, ..., 0]``），不是 identity —— 别搞错。
"""

from typing import List, Optional

from ...graph.graph_editor import bypass_node, remove_node
from ...graph.node_utils import get_attr, set_attr
from ...passes.base import Pass


def compose(p: List[int], q: List[int]) -> List[int]:
    """先做 q 再做 p，等价于 ``perm = [q[i] for i in p]``。"""
    return [q[i] for i in p]


class TransposeSimplifyPass(Pass):
    """Transpose 的消除与合并。"""

    name = "13_transpose_simplify"
    priority = 13
    description = "消除 identity Transpose，合并/抵消 Transpose 链"

    def run(self, graph) -> bool:
        if not self.config.enable_transpose_simplify:
            return False

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Transpose":
                continue
            if not node.input or not node.input[0]:
                continue

            perm = self._perm_of(graph, node)
            if perm is None:
                continue

            # --- 模式 1：identity perm -> 删除 ---
            if perm == list(range(len(perm))):
                if bypass_node(graph, node):
                    graph.rebuild()
                    changed = True
                    self.bump("identity_transpose_removed")
                continue

            # --- 模式 2/3：与前一个 Transpose 合并 ---
            prod = graph.producer(node.input[0])
            if prod is None or prod.op_type != "Transpose":
                continue

            mid = prod.output[0]
            if len(graph.consumers(mid)) != 1 or graph.is_graph_output(mid):
                continue

            p_prev = self._perm_of(graph, prod)
            if p_prev is None:
                continue

            merged = compose(perm, p_prev)

            if merged == list(range(len(merged))):
                # 互相抵消 -> 两个都删
                if bypass_node(graph, prod):
                    graph.rebuild()
                    bypass_node(graph, node)
                    graph.rebuild()
                    changed = True
                    self.bump("transpose_pair_cancelled")
            else:
                # 合并成一个
                set_attr(node, "perm", merged)
                node.input[0] = prod.input[0]
                remove_node(graph, prod)
                graph.rebuild()
                changed = True
                self.bump("transpose_chain_merged")

        return changed

    # ------------------------------------------------------------------
    @staticmethod
    def _perm_of(graph, node) -> Optional[List[int]]:
        """取节点的 perm；缺省是逆序，需要 rank 才能确定。"""
        perm = get_attr(node, "perm")
        if perm is not None:
            return [int(p) for p in perm]

        rank = graph.get_rank(node.input[0])
        if rank is None:
            return None
        return list(range(rank - 1, -1, -1))


__all__ = ["TransposeSimplifyPass", "compose"]
