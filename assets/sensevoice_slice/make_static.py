#!/usr/bin/env python3
"""
Turn the sherpa-onnx SenseVoice ONNX export into a *fully static*,
RKNN-friendly graph.

Same recipe as ``ONNX动态改静态实战指南.md`` (NeMo FastConformer-CTC case),
applied to a FunASR SenseVoice-Small export.

The original export has four inputs::

    x          float32 [N, T, 560]
    x_length   int32   [N]
    language   int32   [N]
    text_norm  int32   [N]

and is dynamic-shape.  ``x_length`` is *only* consumed by the padding-mask
computation (``Add(x_length, 4)`` -> ``Range`` -> ``Less`` -> ``Where``), so once
``x`` has a fixed shape the mask degenerates to "everything is valid" and the
whole ``Shape/Gather/Range/Less/Equal/Where/...`` bookkeeping can be constant
folded away.  ``x_length`` therefore disappears, leaving three inputs::

    x          float32 [1, T, 560]     (T fixed, e.g. 100)
    language   int32   [1]
    text_norm  int32   [1]

What this script does
---------------------
  1. Fixes ``x`` to ``[1, T, 560]`` and ``language``/``text_norm`` to ``[1]``.
  2. Replaces the ``x_length`` input with the constant ``T``.
     (The encoder length is ``x_length + 4`` -- the 4 prompt tokens that get
     prepended in ``/Concat_2`` -- so ``x_length == T`` makes every frame valid
     and every mask a no-op.)
  3. Constant-folds every node that does not depend on ``x``/``language``/
     ``text_norm``.
  4. Simplifies the now-constant mask *applications*:
        Where(cond, X, Y) with cond constant -> X or Y
        Mul(x, 1) -> x      Add(x, 0) -> x      Div(x, 1) -> x
  5. Dead-code elimination + shape inference + negative-axis normalisation.

Usage
-----
    python3 make_static.py                    # T=100
    python3 make_static.py --frames 200
    python3 make_static.py -i model.onnx -o model.static.onnx
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

# Ops whose output depends only on the *shape* of the input, never on its
# values.  They must not be treated as "carrying" data dependence, otherwise
# the whole Shape/Gather/Reshape machinery would be considered live.
SHAPE_ONLY_OPS = {"Shape", "Size"}

UNFRIENDLY_OPS = (
    "Shape", "Gather", "Where", "ConstantOfShape", "Range", "Equal", "Less",
    "Expand", "Tile", "And", "Not", "Or", "Xor", "NonZero", "ScatterND",
    "Slice",
)

# The data input, its rank-3 shape, the length input and the two index inputs.
DATA_INPUT = "x"
LENGTH_INPUT = "x_length"
INDEX_INPUTS = ("language", "text_norm")
FEATURE_DIM = 560


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def tensor_shape(vi):
    """Return a shape list, using None for unknown/dynamic dims."""
    dims = vi.type.tensor_type.shape.dim
    if len(dims) == 0:
        return None
    out = []
    for d in dims:
        if d.HasField("dim_value") and d.dim_value > 0:
            out.append(d.dim_value)
        else:
            out.append(None)
    return out


def set_input_shape(graph, name, shape):
    for coll in (graph.input, graph.value_info):
        for vi in coll:
            if vi.name != name:
                continue
            dims = vi.type.tensor_type.shape.dim
            while len(dims) > 0:
                del dims[0]
            for value in shape:
                d = dims.add()
                d.dim_value = value


def clear_value_info(graph):
    while len(graph.value_info) > 0:
        del graph.value_info[0]


def op_histogram(graph):
    return Counter(n.op_type for n in graph.node)


def count_dynamic_dims(graph):
    n = 0
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_param or not d.HasField("dim_value"):
                    n += 1
    return n


def elem_type_of(graph, name):
    for vi in graph.input:
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    return None


# ---------------------------------------------------------------------------
# step 1+2: static inputs
# ---------------------------------------------------------------------------

def build_static_inputs(model, frames, keep_mask=False):
    graph = model.graph

    # The SenseVoice export is already time-major (x is [N, T, 560]), so unlike
    # the FastConformer case there is no head Transpose to remove.  Only drop
    # one if it really exists and really is the [0, 2, 1] transpose.
    for node in list(graph.node):
        if node.op_type == "Transpose" and DATA_INPUT in node.input:
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
            if perm != [0, 2, 1]:
                raise RuntimeError("unexpected head Transpose perm: %s" % perm)
            transposed = node.output[0]
            for other in graph.node:
                for i, name in enumerate(other.input):
                    if name == transposed:
                        other.input[i] = DATA_INPUT
            graph.node.remove(node)
            log("  removed head Transpose '%s' (perm %s)" % (node.name, perm))

    set_input_shape(graph, DATA_INPUT, [1, frames, FEATURE_DIM])
    log("  %s fixed to [1, %d, %d]" % (DATA_INPUT, frames, FEATURE_DIM))

    for name in INDEX_INPUTS:
        set_input_shape(graph, name, [1])
        log("  %s fixed to [1]" % name)

    set_input_shape(graph, LENGTH_INPUT, [1])
    if keep_mask:
        # Keep the length as a runtime scalar so the padding mask stays alive.
        # rebuild_mask() rewrites the Range/Less/Where machinery into
        # NPU-friendly arithmetic (Add/Cast/Reshape/Sub/Clip + Add/Mul).
        log("  '%s' kept as a scalar input [1] (padding mask kept alive)"
            % LENGTH_INPUT)
        return model

    # x_length -> constant T
    length_dtype = elem_type_of(graph, LENGTH_INPUT) or onnx.TensorProto.INT32
    for vi in list(graph.input):
        if vi.name == LENGTH_INPUT:
            graph.input.remove(vi)
    if not any(i.name == LENGTH_INPUT for i in graph.initializer):
        # np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[length_dtype]
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(length_dtype)
        graph.initializer.append(
            numpy_helper.from_array(np.array([frames], dtype=np_dtype),
                                    name=LENGTH_INPUT))
    log("  '%s' input -> constant %d (padding masks become no-ops)"
        % (LENGTH_INPUT, frames))
    return model


# ---------------------------------------------------------------------------
# step 1b (optional): keep the padding mask, rebuilt with NPU-friendly ops
# ---------------------------------------------------------------------------

# Finite stand-in for the -inf the exporter puts on padded attention keys.
# Must stay well inside fp16 range (|x| < 65504) so the RKNN fp16 path is safe.
MASK_FILL = 1.0e4


def _toposort(graph):
    """Sort graph.node topologically.

    Constant folding and DCE both make a single forward/backward pass over
    ``graph.node`` and therefore assume it is topologically ordered.  The graph
    surgery in rebuild_mask() inserts nodes and rewires early ones, which breaks
    that order, so restore it here.
    """
    nodes = list(graph.node)
    n = len(nodes)
    producer = {}
    for k, nd in enumerate(nodes):
        for o in nd.output:
            if o:
                producer[o] = k
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for k, nd in enumerate(nodes):
        seen = set()
        for i in nd.input:
            pk = producer.get(i)
            if pk is not None and pk != k and pk not in seen:
                seen.add(pk)
                indeg[k] += 1
                succ[pk].append(k)
    queue = [k for k in range(n) if indeg[k] == 0]
    order = []
    head = 0
    while head < len(queue):
        k = queue[head]
        head += 1
        order.append(k)
        for m in succ[k]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(order) != n:                      # cycle -> keep the rest in place
        placed = set(order)
        order += [k for k in range(n) if k not in placed]
    del graph.node[:]
    graph.node.extend(nodes[k] for k in order)
    return graph


def _constant_value(graph, name, by_out, init):
    """Value of a tensor that is either an initializer or a Constant node."""
    if name in init:
        return numpy_helper.to_array(init[name])
    n = by_out.get(name)
    if n is not None and n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    return None


def rebuild_mask(model, frames):
    """Rebuild the padding mask from a scalar length using NPU-friendly ops.

    The FunASR export computes the mask with a ``Range``/``Less`` and applies
    it with ``Equal``/``Where``::

        len4   = Add(x_length, 4)                       # +4 prompt frames
        arange = Range(0, T+4, 1)
        valid  = Less(arange, len4)                     # True = keep   [1, L]
        scores = Where(Equal(valid, 0), -inf, scores)   # -> Softmax
        prob   = Where(Equal(valid, 0), 0,    prob)     # softmax output
        v      = Mul(v, valid)                          # conv / value masking

    ``Range``/``Less``/``Equal``/``Where`` are the ops RKNN cannot handle well,
    so this rewrites them into pure arithmetic on a compile-time ``arange``
    constant::

        valid  = Clip(Cast(x_length + 4) - arange, 0, 1)          # [1, L]
        scores = Add(scores, (1 - valid) * -MASK_FILL)
        prob   = Mul(prob,   valid)
        v      = Mul(v,      valid)   (unchanged, just fed the new valid)

    The mask is a *key* mask (last dim), so ``[1,1,1,L]`` broadcasts over the
    ``[1,H,L,L]`` attention tensors exactly like the original ``[1,1,1,L]``
    condition did.
    """
    graph = model.graph
    by_out = {o: n for n in graph.node for o in n.output if o}
    init = {i.name: i for i in graph.initializer}
    L = frames + 4

    # locate the single mask Less, then walk its Cast chain to the float mask
    less = [n for n in graph.node if n.op_type == "Less"]
    if len(less) != 1:
        raise RuntimeError("expected exactly one Less (padding mask), "
                           "found %d" % len(less))
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    cur, last_cast = less[0].output[0], None
    while True:
        casts = [c for c in consumers.get(cur, []) if c.op_type == "Cast"]
        if len(casts) != 1:
            break
        last_cast, cur = casts[0], casts[0].output[0]
    if last_cast is None:
        raise RuntimeError("no Cast node after the mask Less")
    valid_name = last_cast.output[0]

    # --- new prefix: valid = Clip(len4 - arange, 0, 1) ---------------------
    def init_of(value, dtype):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype))

    for nm, val, dt in [
            ("mask_arange", np.arange(L).reshape(1, L), np.float32),
            ("mask_four", np.array(4), np.int32),
            ("mask_len4r_shape", np.array([1, 1]), np.int64),
            # Clip min/max must be 0-D scalars: onnxruntime (used by
            # rknn-toolkit2's constant folder) rejects a [1] shape with
            # "min->Shape().IsScalar() was false".
            ("mask_zero", 0.0, np.float32),
            ("mask_one", 1.0, np.float32),
            ("mask_valid111_shape", np.array([1, 1, 1, L]), np.int64),
            ("mask_negbig", np.array([-MASK_FILL]), np.float32)]:
        ini = init_of(val, dt)
        ini.name = nm
        graph.initializer.append(ini)

    # turn the last Cast into Clip(mask_raw, 0, 1) -- keeps its output name
    last_cast.op_type = "Clip"
    del last_cast.attribute[:]
    last_cast.input[:] = ["mask_raw", "mask_zero", "mask_one"]
    last_cast.name = "mask_valid"

    graph.node.extend([
        helper.make_node("Add", [LENGTH_INPUT, "mask_four"],
                         ["mask_len4"], name="mask_len4"),
        helper.make_node("Cast", ["mask_len4"], ["mask_len4f"],
                         name="mask_len4f", to=onnx.TensorProto.FLOAT),
        helper.make_node("Reshape", ["mask_len4f", "mask_len4r_shape"],
                         ["mask_len4r"], name="mask_len4r"),
        helper.make_node("Sub", ["mask_len4r", "mask_arange"],
                         ["mask_raw"], name="mask_raw"),
        helper.make_node("Reshape", [valid_name, "mask_valid111_shape"],
                         ["mask_valid111"], name="mask_valid111"),
        helper.make_node("Sub", ["mask_one", "mask_valid111"],
                         ["mask_pad"], name="mask_pad"),
        helper.make_node("Mul", ["mask_pad", "mask_negbig"],
                         ["mask_pad_neg"], name="mask_pad_neg"),
    ])

    # --- rewrite the Where(pad, fill, data) applications -------------------
    rewrote = Counter()
    for node in graph.node:
        if node.op_type != "Where":
            continue
        cond, fill_name, data = node.input
        fill = _constant_value(graph, fill_name, by_out, init)
        if fill is None or fill.size == 0:
            continue
        fv = float(np.asarray(fill).reshape(-1)[0])
        if fv <= -1e30:                       # -inf -> additive key mask
            node.op_type = "Add"
            node.input[:] = [data, "mask_pad_neg"]
            rewrote["Add(scores,pad*-BIG)"] += 1
        elif fv == 0.0:                       # 0 -> multiplicative mask
            node.op_type = "Mul"
            node.input[:] = [data, "mask_valid111"]
            rewrote["Mul(prob,valid)"] += 1
    log("  rewrote mask applications: %s"
        % ", ".join("%s x%d" % kv for kv in sorted(rewrote.items())))
    if sum(rewrote.values()) != 140:
        log("  WARNING: expected 140 mask Where nodes, rewrote %d"
            % sum(rewrote.values()))
    _toposort(graph)
    return model


# ---------------------------------------------------------------------------
# step 3: constant folding
# ---------------------------------------------------------------------------

def evaluate_all(model, feeds):
    """Run the graph once and return every intermediate tensor value."""
    graph = model.graph
    known = {i.name for i in graph.initializer}
    known |= {i.name for i in graph.input}
    known |= {o.name for o in graph.output}
    added = []
    # append 是为了骗 ORT 把所有中间值算出来并返回；
    # pop 是为了骗完就还原，别污染内存里的 model。 
    # 中间靠 SerializeToString() 这个"拷贝点"把两者隔开—
    # 序列化必须在 pop 之前，pop 不影响已经生成的 blob。
    for node in graph.node:
        for name in node.output:
            if name and name not in known:
                known.add(name)
                added.append(name)
                graph.output.append(helper.ValueInfoProto(name=name))

    blob = model.SerializeToString()
    for _ in added:
        graph.output.pop()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3
    sess = ort.InferenceSession(blob, opts, providers=["CPUExecutionProvider"])
    values = sess.run(None, feeds)
    return {o.name: v for o, v in zip(sess.get_outputs(), values)}


def fold_constants(model, values):
    """Replace every node that cannot see a data input with an initializer."""
    graph = model.graph
    graph_outputs = {o.name for o in graph.output}

    dependent = {i.name for i in graph.input}  # 起点：真正吃运行时数据的输入
    kept = []
    folded = 0
    for node in graph.node:
        data_dependent = any(name in dependent for name in node.input if name)
        outputs = [o for o in node.output if o]

        if (not data_dependent # 语义上确实是常量
                and all(o in values for o in outputs) # 有实测值可固化
                and not any(o in graph_outputs for o in outputs)): # 不把图输出冻成常量
            for name in outputs:
                graph.initializer.append(
                    numpy_helper.from_array(np.asarray(values[name]), name=name))
            folded += 1
            continue
        # 如果node的所有输出可被折叠进initializer里，这个node不会被keep
        if data_dependent and node.op_type not in SHAPE_ONLY_OPS:
            dependent.update(outputs)  # 只有携带数据的节点才把依赖传下去
        kept.append(node)

    del graph.node[:]
    graph.node.extend(kept)
    log("  folded %d constant-only node(s); %d remain"
        % (folded, len(graph.node)))
    return model


# Ops whose `axis` attribute indexes input[0] and therefore accepts a negative
# value.  RKNN folds Transpose through these ops by indexing the perm with the
# raw axis, so a lingering -1 makes it raise "ValueError: -1 is not in list".
AXIS_ATTR_OPS = {
    "Softmax", "LogSoftmax", "Hardmax", "Concat", "Split", "Gather",
    "GatherElements", "Flatten", "ArgMax", "ArgMin", "CumSum",
    "LayerNormalization", "InstanceNormalization", "BatchNormalization",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL2", "LpNormalization", "Compress", "TopK",
}


def normalize_axes(model, shapes):
    """Rewrite negative `axis`/`axes` attributes to their positive equivalent."""
    n_fixed = 0
    for node in model.graph.node:
        if node.op_type not in AXIS_ATTR_OPS or not node.input:
            continue
        rank = shapes.get(node.input[0])
        if not rank or any(d is None for d in rank):
            continue
        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                attr.i += len(rank)
                n_fixed += 1
            elif attr.name == "axes" and attr.type == onnx.AttributeProto.INTS:
                for i, value in enumerate(attr.ints):
                    if value < 0:
                        attr.ints[i] = value + len(rank)
                        n_fixed += 1
    log("  normalized %d negative axis attribute(s)" % n_fixed)
    return model


# ---------------------------------------------------------------------------
# step 4: simplify no-op mask applications
# ---------------------------------------------------------------------------

def broadcast_shape(a, b):
    """Shape of broadcast(a, b), or None if it cannot be determined."""
    if a is None or b is None:
        return None
    if any(d is None for d in a) or any(d is None for d in b):
        return None
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + list(a)
    b = [1] * (n - len(b)) + list(b)
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            return None
    return out


def simplify_mask_ops(model, shapes):
    graph = model.graph
    by_name = {i.name: i for i in graph.initializer}
    cache = {}

    def const(name):
        if name not in cache:
            init = by_name.get(name)
            cache[name] = numpy_helper.to_array(init) if init is not None else None
        return cache[name]

    alias = {}

    # resolve 是"别名链寻根"：
    # 把消费者对某个中间别名的引用，改写成对最终真实张量的引用，
    # 从而保证删掉被别名的节点后，图里没有任何悬空引用。
    # guard 只是防止意外成环时死循环。
    def resolve(name):
        guard = set()
        while name in alias and name not in guard:
            guard.add(name)
            name = alias[name]
        return name

    def is_fill(name, value):
        arr = const(name)
        if arr is None or arr.dtype.kind not in "fc" or arr.size == 0:
            return None
        if arr.size == 1:
            return float(arr.reshape(-1)[0]) == value
        return bool(np.all(arr == value))

    # 待删节点按 node.name 记录。protobuf 的 repeated field 每次遍历都会生成新的
    # Python wrapper 对象，id(node) 在两次遍历之间并不稳定（实测同一元素地址会变），
    # 所以不能用 id 当键——否则 "id(n) not in removed" 恒为真，一个节点都删不掉。
    # 名字为空或重名的模型退回按索引匹配，避免误删同名节点。
    names = [n.name for n in graph.node]
    use_names = all(names) and len(set(names)) == len(names)
    if not use_names:
        log("  WARNING: node names are empty or duplicated, "
            "falling back to index-based removal")
    removed_names = set()
    removed_idx = set()

    def mark_removed(idx, node):
        if use_names:
            removed_names.add(node.name)
        else:
            removed_idx.add(idx)

    def is_removed(idx, node):
        return node.name in removed_names if use_names else idx in removed_idx

    stats = Counter()

    for idx, node in enumerate(graph.node):
        if node.op_type == "Where":
            cond = const(node.input[0])
            if cond is None or cond.dtype != np.bool_:
                continue
            if cond.all():
                pick, which = node.input[1], "true"
            elif not cond.any():
                pick, which = node.input[2], "false"
            else:
                continue
            alias[node.output[0]] = resolve(pick)
            mark_removed(idx, node)
            stats["Where->%s" % which] += 1
            continue

        if node.op_type in ("Mul", "Add", "Div"):
            fill = 1.0 if node.op_type in ("Mul", "Div") else 0.0
            pairs = ((1, 0),) if node.op_type == "Div" else ((0, 1), (1, 0))
            for keep_i, fill_i in pairs:
                keep, other = node.input[keep_i], node.input[fill_i]
                if not is_fill(other, fill):
                    continue
                if broadcast_shape(shapes.get(keep),
                                   list(const(other).shape)) != shapes.get(keep):
                    continue
                alias[node.output[0]] = resolve(keep)
                mark_removed(idx, node)
                stats["%s-by-%g" % (node.op_type, fill)] += 1
                break

    if not stats:
        log("  no no-op mask ops left")
        return model, stats

    for idx, node in enumerate(graph.node):
        if is_removed(idx, node):
            continue
        for i, name in enumerate(node.input):
            resolved = resolve(name)
            if resolved != name:
                node.input[i] = resolved

    for out in graph.output:
        if out.name in alias:
            graph.node.append(
                helper.make_node("Identity", [resolve(out.name)], [out.name],
                                 name=out.name + "_identity"))

    n_before = len(graph.node)
    kept = [n for idx, n in enumerate(graph.node) if not is_removed(idx, n)]
    del graph.node[:]
    graph.node.extend(kept)
    log("  simplified: %s (removed %d node(s))"
        % (", ".join("%s x%d" % kv for kv in sorted(stats.items())),
           n_before - len(kept)))
    return model, stats


# ---------------------------------------------------------------------------
# step 5: dead code elimination
# ---------------------------------------------------------------------------

def eliminate_dead_code(model):
    graph = model.graph
    needed = {o.name for o in graph.output}
    keep = []
    for node in reversed(graph.node):
        if any(o in needed for o in node.output if o):
            keep.append(node)
            needed.update(i for i in node.input if i)
    keep.reverse()
    n_removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)

    used = {i.name for i in graph.input}
    used |= {o.name for o in graph.output}
    for node in graph.node:
        used.update(i for i in node.input if i)
    keep_init = [i for i in graph.initializer if i.name in used]
    n_init = len(graph.initializer) - len(keep_init)
    del graph.initializer[:]
    graph.initializer.extend(keep_init)

    log("  DCE removed %d node(s) and %d unused initializer(s)" % (n_removed, n_init))
    return model


def shape_map(model):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for coll in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info):
        for vi in coll:
            shapes[vi.name] = tensor_shape(vi)
    for init in inferred.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(original_path, new_path, frames, trials=3, keep_mask=False):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    ref = ort.InferenceSession(original_path, opts, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(new_path, opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(1234)
    worst = 0.0
    for t in range(trials):
        feats = rng.standard_normal((1, frames, FEATURE_DIM)).astype(np.float32)
        lang = np.array([t % 3], dtype=np.int32)
        norm = np.array([t % 2], dtype=np.int32)

        want = ref.run(None, {DATA_INPUT: feats,
                              LENGTH_INPUT: np.array([frames], dtype=np.int32),
                              "language": lang, "text_norm": norm})[0]
        new_feeds = {DATA_INPUT: feats, "language": lang, "text_norm": norm}
        if keep_mask:
            new_feeds[LENGTH_INPUT] = np.array([frames], dtype=np.int32)
        got = new.run(None, new_feeds)[0]

        if want.shape != got.shape:
            log("  trial %d: SHAPE MISMATCH ref%s vs new%s" % (t, want.shape, got.shape))
            return False
        diff = float(np.max(np.abs(want - got)))
        rel = diff / (float(np.max(np.abs(want))) + 1e-9)
        worst = max(worst, rel)
        log("  trial %d: shape=%s  max|abs diff|=%.3e  max rel diff=%.3e"
            % (t, want.shape, diff, rel))

    ok = worst < 1e-4
    log("  %s (worst relative diff %.3e)" % ("PASS" if ok else "FAIL", worst))
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=os.path.join(here, "model.onnx"))
    ap.add_argument("-o", "--output", default=os.path.join(here, "model.static.onnx"))
    ap.add_argument("--frames", type=int, default=100,
                    help="fixed number of time frames T (default 100)")
    ap.add_argument("--keep-mask", action="store_true",
                    help="keep x_length as a scalar input and rebuild the "
                         "padding mask with NPU-friendly ops (Add/Sub/Clip/Mul) "
                         "instead of folding it away")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    log("=" * 72)
    log("SenseVoice-Small -> static / RKNN-friendly ONNX")
    log("  input : %s" % args.input)
    log("  output: %s" % args.output)
    log("  frames: %d" % args.frames)
    log("=" * 72)

    model = onnx.load(args.input)
    log("[1/5] static inputs")
    model = build_static_inputs(model, args.frames, keep_mask=args.keep_mask)

    if args.keep_mask:
        log("[1b/5] rebuilding padding mask (arange + Clip + Add/Mul)")
        model = rebuild_mask(model, args.frames)

    log("[2/5] evaluating graph with fixed input")
    clear_value_info(model.graph)
    feeds = {
        DATA_INPUT: np.zeros((1, args.frames, FEATURE_DIM), dtype=np.float32),
        "language": np.array([0], dtype=np.int32),
        "text_norm": np.array([0], dtype=np.int32),
    }
    if args.keep_mask:
        feeds[LENGTH_INPUT] = np.array([args.frames], dtype=np.int32)
    probe = evaluate_all(model, feeds)
    log("  captured %d intermediate tensors" % len(probe))

    log("[3/5] constant folding")
    model = fold_constants(model, probe)
    del probe

    log("[4/5] simplifying no-op mask ops")
    model, _ = simplify_mask_ops(model, shape_map(model))

    log("[5/5] dead-code elimination + shape inference")
    model = eliminate_dead_code(model)
    clear_value_info(model.graph)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)

    seen = set()
    for vi in list(model.graph.value_info):
        if vi.name in seen:
            model.graph.value_info.remove(vi)
        else:
            seen.add(vi.name)

    model = normalize_axes(model, shape_map(model))
    onnx.checker.check_model(model)
    onnx.save(model, args.output)

    log("")
    log("=" * 72)
    log("inputs : %s" % [(i.name, tensor_shape(i)) for i in model.graph.input])
    log("outputs: %s" % [(o.name, tensor_shape(o)) for o in model.graph.output])
    log("dynamic dims remaining: %d" % count_dynamic_dims(model.graph))
    log("nodes: %d" % len(model.graph.node))
    log("ops  : %s" % ", ".join("%s:%d" % kv for kv in op_histogram(model.graph).most_common()))
    leftover = [op for op in UNFRIENDLY_OPS
                if any(n.op_type == op for n in model.graph.node)]
    log("unfriendly ops still present: %s" % (leftover if leftover else "none"))
    log("size  : %.1f MB" % (os.path.getsize(args.output) / 1e6))
    log("=" * 72)

    if not args.no_verify:
        log("verifying against the original model")
        ok = verify(args.input, args.output, args.frames,
                    keep_mask=args.keep_mask)
        if not ok:
            sys.exit(1)
    log("done -> %s" % args.output)


if __name__ == "__main__":
    main()
