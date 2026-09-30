"""Static Validator —— 静态性验证（核心）。

**ONNX checker 通过 ≠ 静态化**。checker 只检查图是否合法，不检查
shape 是否静态。所以需要这个独立的验证器。

同时区分两种动态（文档 §8.2）::

    Shape-Dependent Dynamic    ✓ 可静态化
        Shape(X) → Gather → Concat → Reshape

    Data-Dependent Dynamic     ✗ 无法静态化
        NonZero(X) / TopK(X, dynamic_k) / Where(condition) / NMS

后者输出数量取决于实际数据，必须在报告里明确列出 —— 让用户知道
"这个模型没法完全静态化"是数据本身的性质，不是工具没做好。
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..graph.node_utils import get_attr

# 输出形状取决于数据的算子：无论怎么优化都静态化不了
ALWAYS_DATA_DEPENDENT = frozenset({
    "NonZero",
    "NonMaxSuppression",
    "Unique",
})

# 条件性 data-dependent：只有某个输入是动态时才算
CONDITIONALLY_DATA_DEPENDENT = {
    "TopK": (1,),        # K 动态 -> 输出个数动态
    "Where": (0,),       # condition 动态 -> 输出个数不变，但内容是数据相关
}


@dataclass
class StaticValidationReport:
    """静态性检查结果。"""

    dynamic_inputs: List[str] = field(default_factory=list)
    dynamic_outputs: List[str] = field(default_factory=list)
    dynamic_value_info: List[str] = field(default_factory=list)
    dynamic_reshape: List[str] = field(default_factory=list)
    dynamic_slice: List[str] = field(default_factory=list)
    dynamic_expand: List[str] = field(default_factory=list)
    shape_nodes: List[str] = field(default_factory=list)
    size_nodes: List[str] = field(default_factory=list)
    rank_nodes: List[str] = field(default_factory=list)
    data_dependent_ops: List[str] = field(default_factory=list)

    @property
    def static_issues(self) -> List[str]:
        """可静态化但尚未静态化的问题项（工具的责任）。"""
        issues = []
        if self.dynamic_inputs:
            issues.append(f"dynamic inputs: {len(self.dynamic_inputs)}")
        if self.dynamic_outputs:
            issues.append(f"dynamic outputs: {len(self.dynamic_outputs)}")
        if self.dynamic_value_info:
            issues.append(f"dynamic value_info: {len(self.dynamic_value_info)}")
        if self.dynamic_reshape:
            issues.append(f"dynamic Reshape: {len(self.dynamic_reshape)}")
        if self.dynamic_slice:
            issues.append(f"dynamic Slice: {len(self.dynamic_slice)}")
        if self.dynamic_expand:
            issues.append(f"dynamic Expand: {len(self.dynamic_expand)}")
        if self.shape_nodes:
            issues.append(f"Shape nodes: {len(self.shape_nodes)}")
        if self.size_nodes:
            issues.append(f"Size nodes: {len(self.size_nodes)}")
        if self.rank_nodes:
            issues.append(f"Rank nodes: {len(self.rank_nodes)}")
        return issues

    @property
    def is_static(self) -> bool:
        return not self.static_issues

    def format(self) -> str:
        lines = ["Dynamic Shape", "-------------"]

        def row(label: str, names: List[str], warn_ok: bool = True):
            # 用纯 ASCII 标记 —— Windows 控制台默认 GBK，打不出 ✓/✗
            mark = "ok" if not names else "!!"
            lines.append(f"{label:<20} {len(names):>3}  {mark}")
            for n in names[:5]:
                lines.append(f"    - {n}")
            if len(names) > 5:
                lines.append(f"    ... 另有 {len(names) - 5} 个")

        row("Dynamic inputs:", self.dynamic_inputs)
        row("Dynamic outputs:", self.dynamic_outputs)
        row("Dynamic ValueInfo:", self.dynamic_value_info)
        row("Shape nodes:", self.shape_nodes)
        row("Size nodes:", self.size_nodes)
        row("Rank nodes:", self.rank_nodes)
        row("Dynamic Reshape:", self.dynamic_reshape)
        row("Dynamic Slice:", self.dynamic_slice)
        row("Dynamic Expand:", self.dynamic_expand)

        if self.data_dependent_ops:
            lines.append("")
            lines.append("Data-Dependent (无法静态化，需要用户处理)")
            lines.append("----------------------------------------")
            for n in self.data_dependent_ops:
                lines.append(f"    - {n}")

        return "\n".join(lines)


def validate(model, graph=None) -> StaticValidationReport:
    """对模型做静态性检查。"""
    from ..graph.graph_view import GraphView
    if graph is None:
        graph = GraphView(model)

    report = StaticValidationReport()

    # --- 输入 / 输出 ---
    for inp in model.graph.input:
        if _has_dynamic_dim(inp):
            report.dynamic_inputs.append(inp.name)
    for out in model.graph.output:
        if _has_dynamic_dim(out):
            report.dynamic_outputs.append(out.name)

    # --- value_info 里残留的 dim_param ---
    # 只看"真实存在"的 value：前序 Pass 的改名/bypass/折叠会留下悬空条目，
    # 它们引用的 value 已经不在图里，不该算作动态项。
    existing = set(graph.producer_map) | set(graph.initializer_map)
    existing |= {i.name for i in model.graph.input}
    existing |= {o.name for o in model.graph.output}

    for vi in model.graph.value_info:
        if vi.name not in existing:
            continue
        if _has_dynamic_dim(vi):
            report.dynamic_value_info.append(vi.name)

    # --- 算子级检查（主图 + 子图）---
    for node in graph.all_nodes():
        op = node.op_type

        if op == "Shape":
            report.shape_nodes.append(_label(node))
        elif op == "Size":
            report.size_nodes.append(_label(node))
        elif op == "Rank":
            report.rank_nodes.append(_label(node))
        elif op == "Reshape":
            if len(node.input) > 1 and node.input[1] and not graph.is_constant(node.input[1]):
                report.dynamic_reshape.append(_label(node))
        elif op == "Slice":
            if _slice_is_dynamic(graph, node):
                report.dynamic_slice.append(_label(node))
        elif op == "Expand":
            if len(node.input) > 1 and node.input[1] and not graph.is_constant(node.input[1]):
                report.dynamic_expand.append(_label(node))

        # --- data-dependent ---
        if op in ALWAYS_DATA_DEPENDENT:
            report.data_dependent_ops.append(_label(node))
        elif op in CONDITIONALLY_DATA_DEPENDENT:
            for idx in CONDITIONALLY_DATA_DEPENDENT[op]:
                if len(node.input) > idx and node.input[idx]:
                    if not graph.is_constant(node.input[idx]):
                        report.data_dependent_ops.append(_label(node))
                        break

    return report


# ----------------------------------------------------------------------
def _has_dynamic_dim(value_info) -> bool:
    if not value_info.type.HasField("tensor_type"):
        return True
    tt = value_info.type.tensor_type
    if not tt.HasField("shape"):
        return True
    for d in tt.shape.dim:
        if not d.HasField("dim_value"):
            return True
    return False


def _slice_is_dynamic(graph, node) -> bool:
    # opset >= 10：starts/ends 是输入
    if len(node.input) > 2 and node.input[1] and node.input[2]:
        for i in (1, 2):
            if node.input[i] and not graph.is_constant(node.input[i]):
                return True
        return False
    # opset 1：starts/ends 是属性，天然静态
    return get_attr(node, "starts") is None or get_attr(node, "ends") is None


def _label(node) -> str:
    name = node.name or (node.output[0] if node.output else "?")
    return f"{node.op_type}({name})"


__all__ = ["StaticValidationReport", "validate"]
