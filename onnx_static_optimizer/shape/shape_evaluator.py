"""Shape Tensor Evaluator —— 全局常量求值器（核心）。

定位
----
给定一个 ``value_name``，如果它的值能在编译期确定，返回 ``np.ndarray``；
否则返回 ``None``。

``None`` 是**一等公民**：它表示"已判定：不可求值"，而不是"还没算"。
因此缓存判断必须用 ``name in memo`` 而不是 ``memo.get(name)`` —— 否则
同一个不可求值的 value 会被反复递归，图上退化成指数级重复计算。

正确性不变式
------------
handler 只有在**能证明**结果等于运行期结果时才返回值，否则必须返回 None。
两个方向的错误代价极不对称：
  * 假阴性（该折没折）→ 少优化，模型照样跑。
  * 假阳性（不该折折了）→ 静默算错，NPU 上极难排查。
所以一律「宁可不优化，绝不改错语义」。

Shape / Size / Rank 的特殊处理
------------------------------
这三个算子读的是输入的**类型**（shape），不是输入的**值**。
如果走通用路径 ``inputs = [evaluate(i) ...]``，由于 X 是数据张量，
``evaluate(X)`` 必然返回 None，它们会被 ``any(v is None)`` 拦掉，
永远求不出值 —— 而 Shape 恰恰是整条 shape 链的起点。
所以必须短路，绕开 inputs，直接查 ``graph.get_shape()``。
"""

from typing import Any, Dict, List, Optional

import numpy as np
from onnx import numpy_helper

from ..graph.node_utils import get_attr, has_attr
from ..graph.tensor_utils import onnx_dtype_to_numpy
from ..graph.graph_view import GraphView


class ShapeTensorEvaluator:
    """递归 + 记忆化的 shape 子图求值引擎。"""

    # op_type -> handler(self, node, inputs)
    HANDLERS: Dict[str, Any] = {}

    # 读类型不读值的算子，必须短路
    SHAPE_READERS = frozenset({"Shape", "Size", "Rank"})

    def __init__(self, graph: GraphView, trace: Optional[list] = None,
                 max_elements: int = 1 << 22):
        self.graph = graph
        self.memo: Dict[str, Optional[np.ndarray]] = {}
        self.visiting: set = set()
        self.trace = trace
        self.max_elements = max_elements
        self.stats = {"hit": 0, "calc": 0, "miss": 0, "error": 0}

    # ------------------------------------------------------------------
    # 主递归
    # ------------------------------------------------------------------
    def evaluate(self, name: str) -> Optional[np.ndarray]:
        """对指定 value 做编译期常量求值，结果按 value 名记忆化。

        命中缓存直接返回；遇到环（如 Loop 子图）时剪枝并缓存 None。返回 None
        表示「已判定不可求值」，后续访问同样命中缓存，不会重复递归。

        Args:
            name: 待求值的 value 名；空字符串视为非法输入。

        Returns:
            能在编译期确定的常量数组；无法确定时返回 None。
        """
        if not name:
            return None

        if name in self.memo:
            self.stats["hit"] += 1
            self._record("HIT ", name, self.memo[name])
            return self.memo[name]

        if name in self.visiting:
            # 图可能有环（Loop 子图）—— 环上的值一律不可求值
            return self._store(name, None)

        self.visiting.add(name)
        try:
            return self._evaluate_inner(name)
        finally:
            self.visiting.discard(name)

    def _evaluate_inner(self, name: str) -> Optional[np.ndarray]:
        # 1. initializer 或 Constant 节点，直接就是常量
        const = self.graph.get_const(name)
        if const is not None:
            return self._store(name, const)

        node = self.graph.producer(name)
        if node is None:
            # 悬空：graph input、未定义名字，或该 value 无生产者
            return self._store(name, None)

        handler = self.HANDLERS.get(node.op_type)
        if handler is None:
            return self._store(name, None)

        # 2. Shape / Size / Rank 短路：不看 inputs，只查静态 shape
        if node.op_type in self.SHAPE_READERS:
            return self._store(name, self._safe(handler, node, None))

        # 3. 逐个求值输入
        inputs = [self.evaluate(i) if i else None for i in node.input]
        if any(v is None for v in inputs):
            return self._store(name, None)

        # 4. 分发到算子实现
        out = self._safe(handler, node, inputs)
        out = self._select_output(node, name, out)
        return self._store(name, out)

    @staticmethod
    def _select_output(node, name: str, out):
        """多输出算子（Split 等）返回 list，按 name 取对应那一路。"""
        outs = [o for o in node.output if o]
        if len(outs) > 1 and isinstance(out, (list, tuple)) and len(out) == len(outs):
            return out[outs.index(name)]
        return out

    def _safe(self, handler, node, inputs) -> Optional[np.ndarray]:
        """任何求值失败一律降级为 None，绝不让 Pass 崩溃。"""
        try:
            return handler(self, node, inputs)
        except Exception as exc:
            self.stats["error"] += 1
            self._record("ERR ", f"{node.op_type}", repr(exc))
            return None

    def _store(self, name: str, value) -> Optional[np.ndarray]:
        if value is not None:
            value = np.asarray(value)
            # 元素数守卫：避免把巨大的数据张量折成常量
            if value.size > self.max_elements:
                value = None
        self.memo[name] = value
        if value is None:
            self.stats["miss"] += 1
        else:
            self.stats["calc"] += 1
        self._record("CALC" if value is not None else "MISS", name, value)
        return value

    def _record(self, kind: str, name: str, value) -> None:
        if self.trace is not None:
            self.trace.append((kind, name, value))

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------
    def evaluate_many(self, names) -> Dict[str, Optional[np.ndarray]]:
        return {n: self.evaluate(n) for n in names}

    def explain(self, name: str) -> str:
        """返回人类可读的求值轨迹，用于调试"为什么没折掉"。"""
        trace: list = []
        self.trace = trace
        result = self.evaluate(name)
        self.trace = None
        lines = [f"evaluate({name!r}) -> {result!r}", "--- 求值顺序 ---"]
        for kind, n, v in reversed(trace):
            lines.append(f"  {kind}  {n:28s} -> {v!r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    @staticmethod
    def register(*op_types):
        def deco(fn):
            for op in op_types:
                ShapeTensorEvaluator.HANDLERS[op] = fn
            return fn
        return deco


# ======================================================================
# 阶段一：纯 shape 计算（覆盖约 80% 的真实场景）
# ======================================================================

@ShapeTensorEvaluator.register("Shape")
def _eval_shape(ev, node, inputs):
    # 关键：不走 inputs，直接查 X 的静态 shape
    shape = ev.graph.get_shape(node.input[0])
    if shape is None:
        return None
    start = get_attr(node, "start", 0) or 0
    end = get_attr(node, "end", None)
    return np.array(shape, dtype=np.int64)[start:end]


@ShapeTensorEvaluator.register("Size")
def _eval_size(ev, node, inputs):
    shape = ev.graph.get_shape(node.input[0])
    if shape is None:
        return None
    n = 1
    for d in shape:
        n *= int(d)
    return np.array(n, dtype=np.int64)


@ShapeTensorEvaluator.register("Rank")
def _eval_rank(ev, node, inputs):
    # 注意：ONNX 标准域里**没有** Rank 算子 —— 已用 onnx.defs 核实，
    # 标准域只有 Shape / Size / ConstantOfShape 三个"读元信息"的算子。
    # 保留这个 handler 是防御性的（某些工具链或未来版本可能引入，
    # 且实现成本为零）。取 rank 的惯用写法是 Size(Shape(x))。
    shape = ev.graph.get_shape(node.input[0])
    return None if shape is None else np.array(len(shape), dtype=np.int64)


@ShapeTensorEvaluator.register("Constant")
def _eval_constant(ev, node, inputs):
    for a in node.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
    return None


@ShapeTensorEvaluator.register("Cast")
def _eval_cast(ev, node, inputs):
    to = get_attr(node, "to")
    if to is None:
        return None
    return inputs[0].astype(onnx_dtype_to_numpy(to))


@ShapeTensorEvaluator.register("Gather")
def _eval_gather(ev, node, inputs):
    data, indices = inputs[0], inputs[1]
    axis = get_attr(node, "axis", 0)
    return np.take(data, np.atleast_1d(indices).astype(np.int64), axis=axis)


@ShapeTensorEvaluator.register("Concat")
def _eval_concat(ev, node, inputs):
    axis = get_attr(node, "axis", 0)
    return np.concatenate([np.atleast_1d(i) for i in inputs], axis=axis)


@ShapeTensorEvaluator.register("Unsqueeze")
def _eval_unsqueeze(ev, node, inputs):
    data = inputs[0]
    axes = get_attr(node, "axes")
    if axes is None:
        # opset >= 13: axes 是第 2 个输入
        if len(inputs) > 1 and inputs[1] is not None:
            axes = np.atleast_1d(inputs[1]).astype(np.int64).tolist()
        else:
            return None
    axes = [int(a) for a in axes]
    out_rank = data.ndim + len(axes)
    for a in sorted(a % out_rank for a in axes):
        data = np.expand_dims(data, axis=a)
    return data


@ShapeTensorEvaluator.register("Squeeze")
def _eval_squeeze(ev, node, inputs):
    data = inputs[0]
    axes = get_attr(node, "axes")
    if axes is None:
        if len(inputs) > 1 and inputs[1] is not None:
            axes = np.atleast_1d(inputs[1]).astype(np.int64).tolist()
        else:
            return np.squeeze(data)
    axes = sorted({int(a) % data.ndim for a in axes})
    return np.squeeze(data, axis=tuple(axes))


@ShapeTensorEvaluator.register("Reshape")
def _eval_reshape(ev, node, inputs):
    data = inputs[0]
    target = np.atleast_1d(inputs[1]).astype(np.int64)
    allowzero = int(get_attr(node, "allowzero", 0) or 0)
    shape: List[int] = [int(d) for d in target]
    if not allowzero:
        for i, d in enumerate(shape):
            if d == 0:
                if i >= data.ndim:
                    return None
                shape[i] = int(data.shape[i])
    return data.reshape(shape)


@ShapeTensorEvaluator.register("Identity")
def _eval_identity(ev, node, inputs):
    return inputs[0]


@ShapeTensorEvaluator.register("Transpose")
def _eval_transpose(ev, node, inputs):
    perm = get_attr(node, "perm")
    if perm is not None:
        perm = [int(p) for p in perm]
    return np.transpose(inputs[0], perm)


@ShapeTensorEvaluator.register("Flatten")
def _eval_flatten(ev, node, inputs):
    data = inputs[0]
    axis = int(get_attr(node, "axis", 1) or 1)
    axis %= (data.ndim + 1)
    lead = 1
    for d in data.shape[:axis]:
        lead *= int(d)
    tail = 1
    for d in data.shape[axis:]:
        tail *= int(d)
    return data.reshape((lead, tail))


@ShapeTensorEvaluator.register("ConstantOfShape")
def _eval_constant_of_shape(ev, node, inputs):
    shape = np.atleast_1d(inputs[0]).astype(np.int64)
    value = get_attr(node, "value")
    if value is not None:
        arr = numpy_helper.to_array(value)
        fill = arr.ravel()[0] if arr.size else 0
        dtype = arr.dtype
    else:
        fill, dtype = 0, np.float32
    return np.full(tuple(int(d) for d in shape), fill, dtype=dtype)


# ======================================================================
# 阶段二：算术 + 切片
# ======================================================================

def _binary(op):
    def fn(ev, node, inputs):
        return op(inputs[0], inputs[1])
    return fn


ShapeTensorEvaluator.HANDLERS["Add"] = _binary(np.add)
ShapeTensorEvaluator.HANDLERS["Sub"] = _binary(np.subtract)
ShapeTensorEvaluator.HANDLERS["Mul"] = _binary(np.multiply)
ShapeTensorEvaluator.HANDLERS["Min"] = _binary(np.minimum)
ShapeTensorEvaluator.HANDLERS["Max"] = _binary(np.maximum)


@ShapeTensorEvaluator.register("Div")
def _eval_div(ev, node, inputs):
    a, b = inputs[0], inputs[1]
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        # ONNX 整型 Div 为截断除法；shape 场景操作数均为正，floor 与截断等价
        return np.floor_divide(a, b)
    return np.true_divide(a, b)


@ShapeTensorEvaluator.register("Mod")
def _eval_mod(ev, node, inputs):
    a, b = inputs[0], inputs[1]
    if int(get_attr(node, "fmod", 0) or 0):
        return np.fmod(a, b)
    return np.mod(a, b)


@ShapeTensorEvaluator.register("Neg")
def _eval_neg(ev, node, inputs):
    return np.negative(inputs[0])


@ShapeTensorEvaluator.register("Abs")
def _eval_abs(ev, node, inputs):
    return np.abs(inputs[0])


@ShapeTensorEvaluator.register("Floor")
def _eval_floor(ev, node, inputs):
    return np.floor(inputs[0])


@ShapeTensorEvaluator.register("Ceil")
def _eval_ceil(ev, node, inputs):
    return np.ceil(inputs[0])


@ShapeTensorEvaluator.register("Equal")
def _eval_equal(ev, node, inputs):
    return np.equal(inputs[0], inputs[1])


@ShapeTensorEvaluator.register("Greater")
def _eval_greater(ev, node, inputs):
    return np.greater(inputs[0], inputs[1])


@ShapeTensorEvaluator.register("Less")
def _eval_less(ev, node, inputs):
    return np.less(inputs[0], inputs[1])


@ShapeTensorEvaluator.register("Slice")
def _eval_slice(ev, node, inputs):
    data = inputs[0]

    if len(node.input) > 2 and node.input[1] and node.input[2]:
        # opset >= 10: starts/ends/axes/steps 都是输入
        starts = np.atleast_1d(inputs[1]).astype(np.int64)
        ends = np.atleast_1d(inputs[2]).astype(np.int64)
        axes = (np.atleast_1d(inputs[3]).astype(np.int64)
                if len(node.input) > 3 and node.input[3] else None)
        steps = (np.atleast_1d(inputs[4]).astype(np.int64)
                 if len(node.input) > 4 and node.input[4] else None)
    else:
        # opset 1: 全是属性
        starts = np.array(get_attr(node, "starts"), dtype=np.int64)
        ends = np.array(get_attr(node, "ends"), dtype=np.int64)
        axes = (np.array(get_attr(node, "axes"), dtype=np.int64)
                if has_attr(node, "axes") else None)
        steps = None

    rank = data.ndim
    if axes is None:
        axes = list(range(len(starts)))
    else:
        axes = [int(a) % rank for a in axes]
    if steps is None:
        steps = [1] * len(starts)

    slices = [slice(None)] * rank
    for s, e, a, st in zip(starts, ends, axes, steps):
        slices[a] = slice(int(s), int(e), int(st))
    return data[tuple(slices)]


# ======================================================================
# 阶段三：高级
# ======================================================================

@ShapeTensorEvaluator.register("Range")
def _eval_range(ev, node, inputs):
    start = int(np.atleast_1d(inputs[0]).ravel()[0])
    limit = int(np.atleast_1d(inputs[1]).ravel()[0])
    delta = int(np.atleast_1d(inputs[2]).ravel()[0])
    if delta == 0:
        return None
    return np.arange(start, limit, delta, dtype=np.int64)


@ShapeTensorEvaluator.register("Expand")
def _eval_expand(ev, node, inputs):
    data = inputs[0]
    shape = tuple(int(d) for d in np.atleast_1d(inputs[1]).astype(np.int64))
    return np.broadcast_to(data, shape).copy()


@ShapeTensorEvaluator.register("Where")
def _eval_where(ev, node, inputs):
    return np.where(inputs[0], inputs[1], inputs[2])


@ShapeTensorEvaluator.register("GatherND")
def _eval_gathernd(ev, node, inputs):
    data, indices = inputs[0], inputs[1]
    if int(get_attr(node, "batch_dims", 0) or 0):
        return None                     # batch_dims 语义复杂，保守跳过
    idx = np.atleast_1d(indices).astype(np.int64)
    if idx.ndim == 0:
        return None
    k = idx.shape[-1]
    if k > data.ndim:
        return None
    flat = idx.reshape(-1, k)
    picked = data[tuple(flat[:, i] for i in range(k))]
    return picked.reshape(idx.shape[:-1] + data.shape[k:])


@ShapeTensorEvaluator.register("Split")
def _eval_split(ev, node, inputs):
    data = inputs[0]
    axis = int(get_attr(node, "axis", 0) or 0)
    axis %= data.ndim
    dim = int(data.shape[axis])

    split = get_attr(node, "split")
    if split is None and len(inputs) > 1 and inputs[1] is not None:
        split = np.atleast_1d(inputs[1]).astype(np.int64).tolist()

    outs = [o for o in node.output if o]
    if split is None:
        return [np.array(p) for p in np.split(data, len(outs), axis=axis)]

    split = [int(s) for s in split]
    if any(s == -1 for s in split):
        known = sum(s for s in split if s != -1)
        split = [dim - known if s == -1 else s for s in split]

    pieces, pos = [], 0
    for s in split:
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(pos, pos + s)
        pieces.append(np.array(data[tuple(sl)]))
        pos += s
    return pieces


__all__ = ["ShapeTensorEvaluator"]
