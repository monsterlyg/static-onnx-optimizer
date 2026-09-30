"""Pass 4: Constant Propagation。

把「所有输入都是常量」的节点在编译期算出来。

与 Pass 3 的区别
----------------
* Pass 3 专攻 **shape 子图**：输入可能非常量，靠 ``graph.get_shape`` 求值。
* Pass 4 处理**纯数据常量**：输入全是常量，直接算。

输出形式
--------
折叠结果写入 **initializer** 而非新建 Constant 节点 —— NPU 编译器对
initializer 支持更好，且便于 Pass 15 去重。

安全边界
--------
* 随机算子（RandomNormal 等）必须排除，否则每次运行结果不同。
* ``If`` / ``Loop`` / ``Scan`` 的 body 是子图，不能当普通节点折叠。
* 结果元素数超限时放弃，避免把巨大数据张量折成 initializer。
"""

from typing import List, Tuple

from ...graph.graph_editor import add_initializer, remove_node
from ...graph.node_utils import NEVER_FOLD_OPS
from ...passes.base import Pass
from ...shape.shape_evaluator import ShapeTensorEvaluator

CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})


class ConstantPropagationPass(Pass):
    """全常量输入的节点 -> initializer。"""

    name = "04_constant_propagation"
    priority = 4
    description = "把所有输入都是常量的节点在编译期算出来"

    def run(self, graph) -> bool:
        if not self.config.enable_constant_propagation:
            return False

        max_elements = self.config.max_constant_elements
        changed = False

        # 反复扫描直到本轮无新折叠 —— 折叠一个可能让下游也可折叠。
        # 用"先收集、再统一折叠"的批处理，避免边遍历边改图带来的 O(N²) 重建。
        while True:
            ev = ShapeTensorEvaluator(graph, max_elements=max_elements)
            batch: List[Tuple[object, List, List]] = []

            for node in list(graph.graph.node):
                if node.op_type == "Constant":
                    continue
                if node.op_type in NEVER_FOLD_OPS:
                    continue
                if node.op_type in CONTROL_FLOW_OPS:
                    continue
                if not any(node.input):
                    continue

                outs = [o for o in node.output if o]
                if not outs:
                    continue

                # 所有输入必须可求值
                values = [ev.evaluate(i) if i else None for i in node.input]
                if any(v is None for v in values):
                    continue

                # 所有输出必须可求值
                results = [ev.evaluate(o) for o in outs]
                if any(v is None for v in results):
                    continue

                # 体积守卫
                if any(int(v.size) > max_elements for v in results):
                    self.note("skipped_too_large")
                    continue

                batch.append((node, outs, results))

            if not batch:
                break

            for node, outs, results in batch:
                for name, value in zip(outs, results):
                    add_initializer(graph, name, value)
                remove_node(graph, node)
                self.bump("node_folded")

            graph.rebuild()
            changed = True

        return changed


__all__ = ["ConstantPropagationPass", "CONTROL_FLOW_OPS"]
