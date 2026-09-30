"""Pass 14: Cast Simplification。

消除/合并多余的 Cast，这对 NPU 尤其重要（NPU 对 dtype 敏感）。

处理的模式::

    1. Cast(t1→t2) ∘ Cast(t2→t2)   →  删除后一个（它是 no-op）
    2. Cast(t1→t2) ∘ Cast(t2→t1)   →  两个都删（需开关，会损失精度）
    3. Cast(t1→t2) ∘ Cast(t2→t3)   →  合并为 Cast(t1→t3)

关于模式 2 的谨慎
-----------------
虽然类型回到原点，但中间经过 t2 已经损失精度（典型是 FP32→FP16→FP32），
删除会**改变数值结果**。是否允许取决于部署目标（NPU 常允许，因为最终也是
FP16）。所以由 ``allow_fp16_roundtrip_removal`` 控制，**默认关闭**。

（注：pass_implementation.md 里模式 1 的伪代码与其注释不一致 ——
注释说"删后一个"，代码却删了前一个。前一个是被下游依赖的，
删掉会让图失去生产者。这里按注释的正确语义实现。）
"""

from typing import Optional

from ...graph.graph_editor import bypass_node, remove_node
from ...graph.node_utils import get_attr
from ...passes.base import Pass


class CastSimplifyPass(Pass):
    """Cast 链的消除与合并。"""

    name = "14_cast_simplify"
    priority = 14
    description = "消除/合并多余的 Cast"

    def run(self, graph) -> bool:
        if not self.config.enable_cast_simplify:
            return False

        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Cast":
                continue
            if not node.input or not node.input[0]:
                continue

            dst = get_attr(node, "to")
            if dst is None:
                continue

            prod = graph.producer(node.input[0])
            if prod is None or prod.op_type != "Cast":
                continue

            mid = prod.output[0]
            if len(graph.consumers(mid)) != 1 or graph.is_graph_output(mid):
                continue

            src = get_attr(prod, "to")
            if src is None:
                continue

            if src == dst:
                # 模式 1：第二段是 no-op，只删它。prod 仍被需要，不能删。
                if bypass_node(graph, node):
                    graph.rebuild()
                    changed = True
                    self.bump("noop_cast_removed")
                continue

            orig = graph.get_dtype(prod.input[0])
            if orig is not None and dst == orig:
                # 模式 2：类型回到原点（往返 Cast）
                if not self.config.allow_fp16_roundtrip_removal:
                    self.note("skipped_roundtrip_precision")
                    continue
                if bypass_node(graph, prod):
                    graph.rebuild()
                    bypass_node(graph, node)
                    graph.rebuild()
                    changed = True
                    self.bump("roundtrip_cast_removed")
                continue

            # 模式 3：合并为一次 Cast(t1 -> dst)
            node.input[0] = prod.input[0]
            remove_node(graph, prod)
            graph.rebuild()
            changed = True
            self.bump("cast_chain_merged")

        return changed


__all__ = ["CastSimplifyPass"]
