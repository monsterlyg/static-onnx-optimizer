"""GraphView —— 图视图。

ONNX 的 graph.node 是扁平列表，直接遍历做「找生产者 / 找消费者」是 O(N²)。
所有 Pass 都要这两类查询，所以在这里一次性建好索引。

关键约定
--------
任何 Pass 修改图之后都必须调用 ``graph.rebuild()``（PassManager 会自动调用），
否则后续查询用的是脏索引。
"""

from typing import Dict, Iterator, List, Optional, Set

import numpy as np
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto

from .node_utils import constant_node_value
from .tensor_utils import tensor_to_numpy


class GraphView:
    """对 ModelProto.graph 的只读视图 + 索引。"""

    def __init__(self, model: ModelProto):
        self.model: ModelProto = model
        self.graph: GraphProto = model.graph
        # 由 optimizer 依据 config 设置：True 时 bypass 不重命名 graph output
        self.preserve_output_names = False
        self.rebuild()

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        """O(N) 重建全部索引。任何图修改后必须调用。"""
        self.producer_map: Dict[str, NodeProto] = {}
        self.consumer_map: Dict[str, List[NodeProto]] = {}
        self.initializer_map: Dict[str, TensorProto] = {}
        self.value_info_map: Dict[str, ValueInfoProto] = {}

        for init in self.graph.initializer:
            self.initializer_map[init.name] = init

        # graph.input / graph.output 的 shape 也存放在 ValueInfoProto 里，
        # 但不在 graph.value_info 中。漏掉它们会导致 Shape(graph_input) 永远查不到。
        for vi in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output):
            self.value_info_map[vi.name] = vi

        for node in self.graph.node:
            for out in node.output:
                if out:                       # 允许空输出名（可选输出）
                    self.producer_map[out] = node
            for inp in node.input:
                if inp:
                    self.consumer_map.setdefault(inp, []).append(node)

        # Constant 节点的值缓存，避免每次 rebuild 都反序列化大张量
        self._const_cache: Dict[str, Optional[np.ndarray]] = {}

    # 兼容别名：文档中两种写法都出现过
    _rebuild = rebuild

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def producer(self, name: str) -> Optional[NodeProto]:
        return self.producer_map.get(name)

    def consumers(self, name: str) -> List[NodeProto]:
        return self.consumer_map.get(name, [])

    def is_initializer(self, name: str) -> bool:
        return name in self.initializer_map

    def is_graph_input(self, name: str) -> bool:
        return any(i.name == name for i in self.graph.input)

    def is_graph_output(self, name: str) -> bool:
        return any(o.name == name for o in self.graph.output)

    @property
    def graph_output_names(self) -> Set[str]:
        return {o.name for o in self.graph.output}

    def node_count(self) -> int:
        return len(self.graph.node)

    # ------------------------------------------------------------------
    # 类型 / 形状
    # ------------------------------------------------------------------
    def get_value_info(self, name: str) -> Optional[ValueInfoProto]:
        return self.value_info_map.get(name)

    def get_shape(self, name: str) -> Optional[List[int]]:
        """返回静态 shape；任何一维不是 dim_value 就返回 None。

        这是 Shape/Size/Rank 求值的命根子：返回 None 表示"静态未知"，
        求值器会据此放弃，绝不会猜。
        """
        if name in self.initializer_map:
            return list(self.initializer_map[name].dims)

        vi = self.value_info_map.get(name)
        if vi is None or not vi.type.HasField("tensor_type"):
            return None
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None

        dims: List[int] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                return None          # dim_param 或 unknown -> 静态未知
        return dims

    def get_shape_partial(self, name: str) -> Optional[List[Optional[int]]]:
        """返回部分静态 shape，未知维为 None（用于诊断信息，不用于求值）。"""
        if name in self.initializer_map:
            return list(self.initializer_map[name].dims)

        vi = self.value_info_map.get(name)
        if vi is None or not vi.type.HasField("tensor_type"):
            return None
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None

        dims: List[Optional[int]] = []
        for d in tt.shape.dim:
            dims.append(int(d.dim_value) if d.HasField("dim_value") else None)
        return dims

    def get_rank(self, name: str) -> Optional[int]:
        s = self.get_shape_partial(name)
        return None if s is None else len(s)

    def get_dtype(self, name: str) -> Optional[int]:
        """返回 ONNX TensorProto 的 elem_type 枚举。"""
        if name in self.initializer_map:
            return self.initializer_map[name].data_type

        vi = self.value_info_map.get(name)
        if vi is None or not vi.type.HasField("tensor_type"):
            return None
        return vi.type.tensor_type.elem_type

    # ------------------------------------------------------------------
    # 常量
    # ------------------------------------------------------------------
    def get_const(self, name: str) -> Optional[np.ndarray]:
        """返回编译期已知的值，否则 None。

        覆盖两条路径：initializer，以及 Constant 节点。
        """
        if name in self._const_cache:
            return self._const_cache[name]

        value: Optional[np.ndarray] = None
        if name in self.initializer_map:
            try:
                value = tensor_to_numpy(self.initializer_map[name])
            except Exception:
                value = None
        else:
            node = self.producer(name)
            if node is not None and node.op_type == "Constant":
                try:
                    value = constant_node_value(node)
                except Exception:
                    value = None

        self._const_cache[name] = value
        return value

    def is_constant(self, name: str) -> bool:
        return self.get_const(name) is not None

    # ------------------------------------------------------------------
    # 子图遍历（If / Loop / Scan 的 body）
    # ------------------------------------------------------------------
    def iter_subgraphs(self) -> Iterator[GraphProto]:
        """递归产出所有子图（含嵌套）。"""
        yield from _iter_subgraphs(self.graph)

    def subgraph_nodes(self) -> Iterator[NodeProto]:
        """只产出**子图**（If / Loop / Scan 的 body）里的节点。

        用途：检查"子图引用了哪些外层 value"。若这里包含主图节点，
        DCE 会把所有节点都当成 live，等于什么都没删。
        """
        for g in _iter_subgraphs(self.graph):
            for n in g.node:
                yield n

    def all_nodes(self) -> Iterator[NodeProto]:
        """产出主图 + 所有子图的节点。用于全局分析（如静态性检查）。"""
        for n in self.graph.node:
            yield n
        yield from self.subgraph_nodes()


def _iter_subgraphs(graph: GraphProto) -> Iterator[GraphProto]:
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == 5:              # AttributeProto.GRAPH
                yield attr.g
                yield from _iter_subgraphs(attr.g)
            elif attr.type == 10:           # AttributeProto.GRAPHS
                for g in attr.graphs:
                    yield g
                    yield from _iter_subgraphs(g)


__all__ = ["GraphView"]
