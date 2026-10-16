#!/usr/bin/env python3
"""从 SenseVoice 大模型里切出一小块**可运行**的子图，方便用 netron 对比
fold_constants / simplify_mask_ops / eliminate_dead_code 三步的效果。

切片内容（都是模型里真实的节点）：
  * head：x + prompt embedding + Tile/Expand/ConstantOfShape + Concat_2
  * mask 构造链：x_length(+4) / Shape(Concat_2) / Range / Less / Cast / Unsqueeze
  * encoders0.0 的 mask 应用：Unsqueeze_7 / Equal / Cast / Cast_1
                              + Where(cond, -inf, scores) / Where_1(cond, 0, softmax)

两处"切边"（原节点在切片外）提升为图输入：
  /encoder/encoders0.0/self_attn/MatMul_output_0
  /encoder/encoders0.0/self_attn/Softmax_output_0

产出（都在本目录下）：
  slice_0_extracted.onnx     刚切出来的样子（x_length 还是图输入）
  slice_1_inputs_fixed.onnx  固定形状 + x_length 降级为常量（= fold 的输入）
  slice_2_after_fold.onnx    fold_constants 之后
  slice_3_after_simplify.onnx simplify_mask_ops 之后
  slice_4_after_dce.onnx     eliminate_dead_code 之后

用法：
  python3 make_slice.py            # 默认 T=8
  python3 make_slice.py --frames 16
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_static as M

SRC = os.path.join(HERE, "model.onnx")

# 以这些节点为种子做反向闭包（它们产出的东西就是我们想看的）
SEED_NODES = [
    "/encoder/Unsqueeze_1",                              # mask 张量本身
    "/encoder/encoders0.0/self_attn/Unsqueeze_7",        # 每个 layer 把 mask 变形
    "/encoder/encoders0.0/self_attn/Equal",
    "/encoder/encoders0.0/self_attn/Cast",
    "/encoder/encoders0.0/self_attn/Cast_1",
    "/encoder/encoders0.0/self_attn/Where",              # mask 应用 1：Where(cond,-inf,scores)
    "/encoder/encoders0.0/self_attn/Where_1",            # mask 应用 2：Where(cond,0,softmax)
    "/encoder/encoders0.0/Mul",                          # 真实存在的 Mul(x, 1) 空操作
]

# 切边：由切片外的节点产生，提升成图输入
CUT_TENSORS = [
    "/encoder/encoders0.0/self_attn/MatMul_output_0",
    "/encoder/encoders0.0/self_attn/Softmax_output_0",
]

# 切片对外的输出（方便在 netron 里一眼找到）
SLICE_OUTPUTS = [
    "/Concat_2_output_0",                                # head 输出 [1, T+4, 560]
    "/encoder/Unsqueeze_1_output_0",                     # mask
    "/encoder/encoders0.0/self_attn/Where_output_0",     # mask 应用 1 的结果
    "/encoder/encoders0.0/self_attn/Where_1_output_0",   # mask 应用 2 的结果
    "/encoder/encoders0.0/Mul_output_0",                 # Mul(x,1) 的结果
]


def set_static_input_shapes(model, frames):
    want = {"x": [1, frames, M.FEATURE_DIM], "language": [1], "text_norm": [1],
            "x_length": [1]}
    for vi in model.graph.input:
        if vi.name in want:
            dims = vi.type.tensor_type.shape.dim
            del dims[:]
            for v in want[vi.name]:
                dims.add().dim_value = v


def probe_shapes(frames, tensors):
    """跑一次原始模型，拿到这些张量的真实 shape/dtype（shape inference 在这张图上不可靠）。"""
    m = onnx.load(SRC)
    set_static_input_shapes(m, frames)
    for t in tensors:
        m.graph.output.append(helper.ValueInfoProto(name=t))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    feeds = {"x": np.zeros((1, frames, M.FEATURE_DIM), np.float32),
             "x_length": np.array([frames], np.int32),
             "language": np.array([0], np.int32),
             "text_norm": np.array([0], np.int32)}
    outs = sess.run(tensors, feeds)
    info = {}
    for t, v in zip(tensors, outs):
        info[t] = (list(v.shape), ort_dtype_to_onnx(v.dtype))
    return info


def ort_dtype_to_onnx(dt):
    return {"float32": TensorProto.FLOAT, "int32": TensorProto.INT32,
            "int64": TensorProto.INT64, "bool": TensorProto.BOOL}[str(dt)]


def make_feeds(model):
    """按切片实际的图输入构造全零 feeds。"""
    np_dt = {TensorProto.FLOAT: np.float32, TensorProto.INT32: np.int32,
             TensorProto.INT64: np.int64, TensorProto.BOOL: np.bool_}
    feeds = {}
    for vi in model.graph.input:
        dt = np_dt.get(vi.type.tensor_type.elem_type, np.float32)
        shape = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
                 for d in vi.type.tensor_type.shape.dim]
        feeds[vi.name] = np.zeros(shape, dt)
    return feeds


def build_slice(frames, info):
    m = onnx.load(SRC)
    set_static_input_shapes(m, frames)
    g = m.graph

    by_name = {n.name: n for n in g.node}
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n

    keep = set()

    def add_backward(node_name):
        """node_name 是**节点名**；沿输入张量找生产者节点，递归加入。"""
        stack = [node_name]
        while stack:
            nm = stack.pop()
            if nm in keep or nm not in by_name:
                continue
            keep.add(nm)
            for t in by_name[nm].input:
                prod = by_out.get(t) if t else None
                if prod is not None:
                    stack.append(prod.name)

    for nm in SEED_NODES:
        if nm not in by_name:
            raise SystemExit("seed node not found: %s" % nm)
        add_backward(nm)

    nodes = [n for n in g.node if n.name in keep]           # 保持原拓扑序
    produced = {o for n in nodes for o in n.output}
    used = {i for n in nodes for i in n.input if i}

    inputs = []
    for vi in g.input:
        if vi.name in used:
            inputs.append(helper.make_tensor_value_info(
                vi.name, vi.type.tensor_type.elem_type,
                [d.dim_value for d in vi.type.tensor_type.shape.dim]))
    for t in CUT_TENSORS:
        if t not in produced:
            shape, dt = info[t]
            inputs.append(helper.make_tensor_value_info(t, dt, shape))
            print("  cut edge -> input: %s %s" % (t, shape))

    outputs = []
    for t in SLICE_OUTPUTS:
        if t in produced:
            shape, dt = info[t]
            outputs.append(helper.make_tensor_value_info(t, dt, shape))

    inits = [i for i in g.initializer if i.name in used]
    slice_graph = helper.make_graph(nodes, "sensevoice_slice", inputs, outputs,
                                    initializer=inits)
    sliced = helper.make_model(slice_graph, opset_imports=list(m.opset_import))
    sliced.ir_version = m.ir_version
    sliced.producer_name = "make_slice.py"
    onnx.checker.check_model(sliced)
    print("slice: %d nodes, %d initializers, %d inputs, %d outputs"
          % (len(nodes), len(inits), len(inputs), len(outputs)))
    return sliced


def save(model, name):
    path = os.path.join(WORK, name)
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print("  (checker warning on %s: %s)" % (name, e))
    onnx.save(model, path)
    print("  saved %-28s %d nodes, %d initializers"
          % (name, len(model.graph.node), len(model.graph.initializer)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    T = args.frames

    print("[0] 探测切边形状 (T=%d)" % T)
    info = probe_shapes(T, CUT_TENSORS + SLICE_OUTPUTS)
    for k, v in info.items():
        print("    %-56s %s" % (k, v[0]))

    print("[1] 切子图")
    sliced = build_slice(T, info)
    save(sliced, "slice_0_extracted.onnx")

    print("[2] 固定输入形状 + x_length -> 常量 (= fold_constants 的输入)")
    model = onnx.load(os.path.join(WORK, "slice_0_extracted.onnx"))
    model = M.build_static_inputs(model, T)
    M.clear_value_info(model.graph)
    save(model, "slice_1_inputs_fixed.onnx")

    feeds = make_feeds(model)
    print("    feeds: %s" % {k: (list(v.shape), str(v.dtype)) for k, v in feeds.items()})

    print("[3] fold_constants")
    probe = M.evaluate_all(model, feeds)
    print("    captured %d tensors" % len(probe))
    model = M.fold_constants(model, probe)
    save(model, "slice_2_after_fold.onnx")

    print("[4] simplify_mask_ops")
    model, _ = M.simplify_mask_ops(model, M.shape_map(model))
    save(model, "slice_3_after_simplify.onnx")

    print("[5] eliminate_dead_code")
    model = M.eliminate_dead_code(model)
    save(model, "slice_4_after_dce.onnx")
    print("done")


if __name__ == "__main__":
    main()
