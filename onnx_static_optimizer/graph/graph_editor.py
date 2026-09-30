"""图编辑工具。

设计要点
--------
1. **节点用"输出名"识别，不用对象身份。**
   protobuf 的 repeated composite container 在每次访问时可能返回新的
   Python 包装对象，用 ``is`` 或 ``list.index`` 匹配都不可靠。
   ONNX 图是 SSA 的，输出名集合唯一标识一个节点。

2. **删除用「按索引逆序 del」，不用 ``del [:] + extend``。**
   ``del graph.node[:]`` 会让底层消息失效，再把旧引用 extend 回去
   是未定义行为。逐索引删除是已验证安全的写法。

3. **插入 Constant 必须插在原节点位置**，不能 append，
   否则破坏拓扑序，onnx.checker 会报 "must be topologically sorted"。
"""

from typing import Iterable, List, Optional

import numpy as np
from onnx import GraphProto, ModelProto, NodeProto, TensorProto

from .node_utils import make_constant_node, make_node
from .tensor_utils import numpy_to_tensor


# ----------------------------------------------------------------------
# 定位
# ----------------------------------------------------------------------
def find_node_index(graph, node: NodeProto) -> int:
    """按输出名定位节点索引，找不到返回 -1。"""
    outs = tuple(o for o in node.output if o)
    if not outs:
        return -1
    for i, n in enumerate(graph.graph.node):
        if tuple(o for o in n.output if o) == outs:
            return i
    return -1


def find_node_by_output(graph, output_name: str) -> Optional[NodeProto]:
    return graph.producer(output_name)


# ----------------------------------------------------------------------
# 删除
# ----------------------------------------------------------------------
def remove_node(graph, node: NodeProto) -> bool:
    idx = find_node_index(graph, node)
    if idx < 0:
        return False
    del graph.graph.node[idx]
    return True


def remove_node_by_output(graph, output_name: str) -> bool:
    for i, n in enumerate(graph.graph.node):
        if output_name in n.output:
            del graph.graph.node[i]
            return True
    return False


def remove_nodes(graph, nodes: Iterable[NodeProto]) -> int:
    """批量删除。索引逆序删除，避免位移影响。"""
    indices = set()
    for n in nodes:
        idx = find_node_index(graph, n)
        if idx >= 0:
            indices.add(idx)
    for idx in sorted(indices, reverse=True):
        del graph.graph.node[idx]
    return len(indices)


# ----------------------------------------------------------------------
# 替换
# ----------------------------------------------------------------------
def replace_node_with_initializer(graph, node: NodeProto,
                                  value: np.ndarray) -> bool:
    """把节点的输出改为一个同名 initializer，并删除该节点。

    下游无需改线（名字沿用），且 initializer 是叶子，不涉及拓扑序。
    """
    out_name = node.output[0]
    add_initializer(graph, out_name, value)
    return remove_node(graph, node)


def replace_node_with_constant(graph, node: NodeProto,
                               value: np.ndarray) -> bool:
    """把节点就地替换成一个 Constant 节点（保持拓扑序）。"""
    out_name = node.output[0]
    idx = find_node_index(graph, node)
    if idx < 0:
        return False
    new_node = make_constant_node(out_name, value)
    del graph.graph.node[idx]
    graph.graph.node.insert(idx, new_node)
    return True


def add_initializer(graph, name: str, value: np.ndarray) -> TensorProto:
    """新增/覆盖一个 initializer。"""
    if name in graph.initializer_map:
        for i, init in enumerate(graph.graph.initializer):
            if init.name == name:
                del graph.graph.initializer[i]
                break
    tensor = numpy_to_tensor(np.asarray(value), name)
    graph.graph.initializer.append(tensor)
    return tensor


def materialize_input(graph, node: NodeProto, index: int,
                      value: np.ndarray, name_hint: str = "") -> str:
    """把节点的第 index 个输入替换成一个新建的 initializer。

    用于 Slice/Expand 等"参数是输入而非属性"的算子静态化。
    返回新建的 initializer 名字。
    """
    base = name_hint or f"{node.output[0]}_{node.op_type}_in{index}"
    name = unique_name(graph, base)
    add_initializer(graph, name, value)
    while len(node.input) <= index:
        node.input.append("")
    node.input[index] = name
    return name


# ----------------------------------------------------------------------
# 重连（bypass）
# ----------------------------------------------------------------------
def bypass_node(graph, node: NodeProto, src_index: int = 0,
                dst_index: int = 0) -> bool:
    """把 ``node.output[dst_index]`` 的消费者改接 ``node.input[src_index]``，
    然后删除 node。

    若输出名是 graph output，则把 graph output 改名（撞名时放弃 bypass）。
    """
    if len(node.input) <= src_index or not node.input[src_index]:
        return False
    if len(node.output) <= dst_index or not node.output[dst_index]:
        return False

    src_name = node.input[src_index]
    out_name = node.output[dst_index]

    # 1. 重连消费者
    for consumer in graph.consumers(out_name):
        for i, inp in enumerate(consumer.input):
            if inp == out_name:
                consumer.input[i] = src_name

    # 2. graph output 改名
    if graph.is_graph_output(out_name):
        if getattr(graph, "preserve_output_names", False):
            # 用户要求保持输出名不变 -> 放弃 bypass，保留该节点
            return False
        if graph.is_graph_output(src_name):
            # 两个名字都是 graph output，改名会造成重复 —— 放弃
            return False
        if not rename_graph_output(graph, out_name, src_name):
            return False

    # 3. 删除节点
    return remove_node(graph, node)


def rename_graph_output(graph, old_name: str, new_name: str) -> bool:
    for o in graph.graph.output:
        if o.name == old_name:
            o.name = new_name
            return True
    return False


# ----------------------------------------------------------------------
# 名字
# ----------------------------------------------------------------------
def unique_name(graph, base: str) -> str:
    """生成一个图内未被占用的名字。"""
    taken = set(graph.producer_map) | set(graph.initializer_map)
    taken |= {i.name for i in graph.graph.input}
    taken |= {o.name for o in graph.graph.output}
    taken |= set(graph.consumer_map)

    if base not in taken:
        return base
    i = 1
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


# ----------------------------------------------------------------------
# 拓扑排序
# ----------------------------------------------------------------------
def topological_sort(graph) -> bool:
    """Kahn 算法重排节点。有环则保持原序并返回 False。"""
    nodes = list(graph.graph.node)
    if not nodes:
        return True

    produced = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                produced[o] = i

    indeg = [0] * len(nodes)
    succ: List[List[int]] = [[] for _ in nodes]
    for i, n in enumerate(nodes):
        deps = set()
        for inp in n.input:
            if inp and inp in produced and produced[inp] != i:
                deps.add(produced[inp])
        indeg[i] = len(deps)
        for d in deps:
            succ[d].append(i)

    from collections import deque
    queue = deque(i for i in range(len(nodes)) if indeg[i] == 0)
    order: List[int] = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)

    if len(order) != len(nodes):
        return False            # 有环（控制流子图），保持原序

    if order == list(range(len(nodes))):
        return True             # 已是有序，不折腾

    copies = []
    for i in order:
        p = NodeProto()
        p.CopyFrom(nodes[i])
        copies.append(p)

    del graph.graph.node[:]
    graph.graph.node.extend(copies)
    return True


__all__ = [
    "find_node_index",
    "find_node_by_output",
    "remove_node",
    "remove_node_by_output",
    "remove_nodes",
    "replace_node_with_initializer",
    "replace_node_with_constant",
    "add_initializer",
    "materialize_input",
    "bypass_node",
    "rename_graph_output",
    "unique_name",
    "topological_sort",
]
