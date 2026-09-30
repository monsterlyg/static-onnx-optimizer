"""构造测试用的 ONNX 模型。

不依赖任何真实模型文件，全部程序化生成，保证测试可复现。
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def _t(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def build_yolo_like(n=1, h=8, w=8, opset=13):
    """YOLOv8 风格的动态 shape 子图 + 结构冗余。

    覆盖的 Pass：

    * Pass 3  求值 ``Shape → Gather → Mul → Concat`` 与 ``Shape → Gather → Div``
    * Pass 6  ``Shape / Size / Rank`` -> 常量
    * Pass 7  ``Reshape`` 的 ``[-1, h*w]`` 静态化
    * Pass 8  ``Slice`` 的动态 ``ends`` 静态化
    * Pass 9  ``Expand`` 目标 shape 静态化 + no-op 删除
    * Pass 11 ``Identity`` 删除
    * Pass 13 ``Transpose`` 抵消
    * Pass 14 ``Cast`` 合并
    * Pass 10 上游孤儿节点清理
    """
    rng = np.random.default_rng(0)
    nodes = []
    inits = [
        _t("conv_w", (rng.standard_normal((8, 3, 3, 3)) * 0.1).astype(np.float32)),
        _t("conv_b", np.zeros(8, dtype=np.float32)),
        _t("idx0", np.array([0], dtype=np.int64)),
        _t("idx1", np.array([1], dtype=np.int64)),
        _t("idx2", np.array([2], dtype=np.int64)),
        _t("idx3", np.array([3], dtype=np.int64)),
        _t("neg1", np.array([-1], dtype=np.int64)),
        _t("two", np.array([2], dtype=np.int64)),
        _t("ax0", np.array([0], dtype=np.int64)),
    ]

    def node(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))

    # --- 数据通路（绝不该被折叠）---
    # pads=1 保持 8x8，否则 3x3 Conv 会把空间维缩到 6x6，后面的 Reshape 就非法了
    node("Conv", ["images", "conv_w", "conv_b"], ["c1"], pads=[1, 1, 1, 1])
    node("Relu", ["c1"], ["r1"])                       # [1, 8, 8, 8]

    # --- 动态 shape 子图 1：newshape = [-1, H*W] ---
    node("Shape", ["images"], ["sh0"])                 # [1, 3, 8, 8]
    node("Gather", ["sh0", "idx2"], ["h"], axis=0)     # [8]
    node("Gather", ["sh0", "idx3"], ["w"], axis=0)     # [8]
    node("Mul", ["h", "w"], ["hw"])                    # [64]
    node("Concat", ["neg1", "hw"], ["newshape"], axis=0)
    node("Reshape", ["r1", "newshape"], ["p"])         # [8, 64]

    # --- 动态 shape 子图 2：ends = dim(p, 1) / 2 ---
    node("Shape", ["p"], ["sh1"])
    node("Gather", ["sh1", "idx1"], ["k"], axis=0)     # [64]
    node("Div", ["k", "two"], ["half"])                # [32]
    node("Slice", ["p", "idx0", "half", "idx1"], ["sl"])   # [8, 32]

    # --- 动态 shape 子图 3：Expand 到自身 shape（no-op）---
    node("Shape", ["sl"], ["sh2"])
    node("Gather", ["sh2", "idx0"], ["d0"], axis=0)
    node("Gather", ["sh2", "idx1"], ["d1"], axis=0)
    node("Concat", ["d0", "d1"], ["tgt"], axis=0)
    node("Expand", ["sl", "tgt"], ["ex"])              # [8, 32]

    # --- 结构冗余 ---
    node("Identity", ["ex"], ["id1"])
    node("Transpose", ["id1"], ["tr1"], perm=[1, 0])   # [32, 8]
    node("Transpose", ["tr1"], ["tr2"], perm=[1, 0])   # [8, 32] 互相抵消

    # --- Size / Shape+Size -> 常量，再拼成 int64 输出 ---
    # 注：ONNX 标准域里没有 Rank 算子（已用 onnx.defs 核实）。
    # "取 rank" 的惯用写法是 Size(Shape(x)) —— Shape 给出 [1,8,8,8]，
    # 对它取 Size 就得到 4。
    node("Size", ["r1"], ["sz"])                       # 512
    node("Shape", ["r1"], ["sh3"])                     # [1, 8, 8, 8]
    node("Size", ["sh3"], ["rk"])                      # 4  == Rank(r1)
    node("Unsqueeze", ["sz", "ax0"], ["szu"])
    node("Unsqueeze", ["rk", "ax0"], ["rku"])
    node("Concat", ["szu", "rku"], ["meta_raw"], axis=0)
    node("Cast", ["meta_raw"], ["mi"], to=TensorProto.INT32)
    node("Cast", ["mi", ], ["out_meta"], to=TensorProto.DOUBLE)   # 合并成一次 Cast

    inputs = [helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["N", 3, "H", "W"])]
    outputs = [
        helper.make_tensor_value_info("tr2", TensorProto.FLOAT, ["A", "B"]),
        helper.make_tensor_value_info("out_meta", TensorProto.DOUBLE, ["M"]),
    ]

    graph = helper.make_graph(nodes, "yolo_like", inputs, outputs,
                              initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def build_static_model(opset=13):
    """已经是静态的模型 —— 优化器应当几乎不做任何改动（幂等性基线）。"""
    rng = np.random.default_rng(1)
    inits = [
        _t("conv_w", (rng.standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)),
        _t("conv_b", np.zeros(4, dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("Conv", ["images", "conv_w", "conv_b"], ["c1"],
                         pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["out"]),
    ]
    inputs = [helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 8, 8])]
    outputs = [helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [1, 4, 8, 8])]
    graph = helper.make_graph(nodes, "static", inputs, outputs, initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def build_data_dependent_model(opset=13):
    """含 NonZero —— 输出个数取决于数据，**无法**静态化。

    用于验证 StaticValidator 会把它明确报告出来，而不是假装成功。
    """
    nodes = [
        helper.make_node("NonZero", ["images"], ["nz"]),
        helper.make_node("Identity", ["nz"], ["out"]),
    ]
    inputs = [helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["N", 3])]
    # NonZero 输出是 [rank, 非零元素个数]，K 是数据相关的
    outputs = [helper.make_tensor_value_info(
        "out", TensorProto.INT64, ["R", "K"])]
    graph = helper.make_graph(nodes, "data_dependent", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def build_shape_start_end_model(opset=15):
    """测试 opset 15+ 的 ``Shape(start/end)`` —— 只取 H/W 两维。"""
    nodes = [
        helper.make_node("Shape", ["images"], ["hw"], start=2, end=4),
        helper.make_node("Cast", ["hw"], ["out"], to=TensorProto.FLOAT),
    ]
    inputs = [helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["N", 3, "H", "W"])]
    outputs = [helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, ["A", "B"])]
    graph = helper.make_graph(nodes, "shape_start_end", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


__all__ = [
    "build_yolo_like",
    "build_static_model",
    "build_data_dependent_model",
    "build_shape_start_end_model",
]
