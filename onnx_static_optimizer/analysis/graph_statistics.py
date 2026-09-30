"""图统计与动态性分析。"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from ..graph.graph_view import GraphView


@dataclass
class GraphStatistics:
    nodes: int = 0
    initializers: int = 0
    value_info: int = 0
    inputs: int = 0
    outputs: int = 0
    op_histogram: Dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        lines = ["Graph Statistics", "----------------"]
        lines.append(f"Nodes:        {self.nodes}")
        lines.append(f"Initializers: {self.initializers}")
        lines.append(f"ValueInfo:    {self.value_info}")
        if self.op_histogram:
            top = sorted(self.op_histogram.items(),
                         key=lambda kv: -kv[1])[:8]
            lines.append("Top ops:      " +
                         ", ".join(f"{k}x{v}" for k, v in top))
        return "\n".join(lines)


def analyze(model) -> GraphStatistics:
    graph = GraphView(model)
    hist = Counter()
    for node in graph.graph.node:
        hist[node.op_type] += 1
    return GraphStatistics(
        nodes=len(graph.graph.node),
        initializers=len(graph.graph.initializer),
        value_info=len(graph.graph.value_info),
        inputs=len(graph.graph.input),
        outputs=len(graph.graph.output),
        op_histogram=dict(hist),
    )


def dynamic_op_names(model) -> List[str]:
    """列出所有可能与动态 shape 相关的算子类型（用于快速摸底）。"""
    interesting = {
        "Shape", "Size", "Rank", "Reshape", "Slice", "Expand", "Tile",
        "Split", "Pad", "Resize", "Upsample", "Concat", "Gather", "Cast",
        "NonZero", "TopK", "Where", "NonMaxSuppression",
    }
    graph = GraphView(model)
    found = {n.op_type for n in graph.graph.node if n.op_type in interesting}
    return sorted(found)


__all__ = ["GraphStatistics", "analyze", "dynamic_op_names"]
