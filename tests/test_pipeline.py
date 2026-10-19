"""端到端测试。

不依赖 pytest，直接 ``python tests/test_pipeline.py`` 即可运行。

覆盖：
  1. YOLOv8 风格动态模型的完整流水线（静态性 / 数值等价 / 各 Pass 生效）
  2. 幂等性（对输出再跑一次应当零变更）
  3. 已静态模型不被破坏
  4. data-dependent 动态被明确报告
  5. opset 15+ 的 Shape(start/end)
  6. 输入 shape 冲突报错
  7. ShapeTensorEvaluator 的算子语义单测
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnx

from onnx_static_optimizer import OptimizerConfig, optimize
from onnx_static_optimizer.analysis.graph_statistics import analyze
from onnx_static_optimizer.graph.graph_view import GraphView
from onnx_static_optimizer.shape.shape_evaluator import ShapeTensorEvaluator
from onnx_static_optimizer.validation.static_validator import validate

from models import (
    build_data_dependent_model,
    build_shape_start_end_model,
    build_static_model,
    build_yolo_like,
)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def op_counts(model):
    graph = GraphView(model)
    counts = {}
    for n in graph.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return counts


def evaluated(ev, name):
    """求值并断言一定成功，返回 ndarray。

    ``ShapeTensorEvaluator.evaluate`` 的返回类型是 ``Optional[np.ndarray]``
    （不可求值返回 None，这是刻意设计），因此不能直接对结果取成员。
    这里显式断言，既消除 reportOptionalMemberAccess，也让"本该能求值却
    返回 None"的失败信息更清晰。
    """
    value = ev.evaluate(name)
    assert value is not None, f"{name} 应当可求值，实际返回 None"
    return value


# ======================================================================
@test
def test_yolo_like_full_pipeline():
    model = build_yolo_like()
    result = optimize(model, {"images": [1, 3, 8, 8]})

    assert result.converged, (
        f"未收敛，仍在报变更的 Pass：{result.stalled_passes}")

    # --- 图合法性 ---
    assert result.checker_ok, f"onnx.checker 失败：{result.checker_message}"

    # --- 静态性 ---
    assert result.is_static, (
        "仍有动态项：\n" + result.format())

    # --- 数值等价 ---
    assert result.numerical_ok is True, (
        f"数值不等价：{result.numerical_message}")

    # --- 各 Pass 确实生效 ---
    counts = op_counts(result.model)
    for op in ("Shape", "Size", "Rank"):
        assert op not in counts, f"{op} 未被消除"
    assert "Identity" not in counts, "Identity 未被消除"
    assert "Transpose" not in counts, "Transpose 对被抵消"
    assert "Expand" not in counts, "no-op Expand 未被删除"

    detail = result.detail_summary()
    assert detail.get("identity_removed", 0) >= 1, detail
    assert detail.get("transpose_pair_cancelled", 0) >= 1, detail
    # Shape/Size 具体由 Pass 6 还是 Pass 4 消除，取决于它们谁先看到机会
    # （Pass 4 在 Pass 6 之前跑）。这里只断言"确实被消除了"，
    # 真正的验收是上面 op_counts 里已经没有 Shape/Size/Rank。
    assert detail.get("eliminated_shape", 0) >= 1, detail
    assert detail.get("eliminated_size", 0) >= 1, detail
    # 原 Pass 3 已移除，shape 子图统一由 Pass 4 折叠。
    assert detail.get("node_folded", 0) >= 1, detail
    assert detail.get("expand_removed", 0) >= 1, detail

    # --- 节点数确实下降了 ---
    assert result.nodes_after < result.nodes_before, (
        f"{result.nodes_before} -> {result.nodes_after}")

    # --- 输入已定形，输出已解析 ---
    assert result.input_shapes_after["images"] == "[1,3,8,8]"
    out_shapes = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim]
                  for o in result.model.graph.output}
    for name, dims in out_shapes.items():
        assert dims, f"输出 {name} 仍是动态 shape"

    # --- 数据通路绝不能被折叠 ---
    assert "Conv" in counts and "Relu" in counts, "数据通路被误删"

    print(f"  nodes: {result.nodes_before} -> {result.nodes_after}")
    print(f"  iterations: {result.iterations}, converged={result.converged}")
    print(f"  {result.numerical_message}")


@test
def test_idempotent():
    first = optimize(build_yolo_like(), {"images": [1, 3, 8, 8]})
    second = optimize(first.model, {"images": [1, 3, 8, 8]})

    total = sum(st["changes"] for st in second.pass_stats)
    assert total == 0, (
        "第二次优化仍有变更（Pass 不幂等）："
        f"{[st for st in second.pass_stats if st['changes']]}")
    assert second.nodes_after == first.nodes_after
    print(f"  second run changes: {total} (nodes={second.nodes_after})")


@test
def test_static_model_untouched():
    model = build_static_model()
    result = optimize(model, {})
    assert result.checker_ok, result.checker_message
    assert result.is_static
    assert result.numerical_ok is True, result.numerical_message
    assert result.nodes_after == result.nodes_before, (
        f"静态模型的节点数被改动：{result.nodes_before} -> {result.nodes_after}")
    print(f"  nodes unchanged: {result.nodes_after}")


@test
def test_data_dependent_is_reported():
    model = build_data_dependent_model()
    result = optimize(model, {"images": [1, 3]},
                      config=OptimizerConfig(validate_numerically=False))

    assert not result.is_static, "含 NonZero 的模型不该被判为静态"
    assert result.validation.data_dependent_ops, (
        "NonZero 必须被报告为 data-dependent")
    assert any("NonZero" in s for s in result.validation.data_dependent_ops)
    print(f"  reported: {result.validation.data_dependent_ops}")


@test
def test_shape_start_end():
    model = build_shape_start_end_model()
    result = optimize(model, {"images": [1, 3, 8, 8]})
    assert result.checker_ok, result.checker_message
    assert result.is_static, result.format()
    assert result.numerical_ok is True, result.numerical_message
    # Shape(start=2, end=4) -> [8, 8]
    counts = op_counts(result.model)
    assert "Shape" not in counts
    print(f"  nodes: {result.nodes_before} -> {result.nodes_after}")


@test
def test_input_shape_conflict_raises():
    model = build_static_model()          # images 已经是 [1,3,8,8]
    try:
        optimize(model, {"images": [1, 3, 16, 16]})
    except ValueError as exc:
        assert "冲突" in str(exc)
        print(f"  正确报错：{str(exc)[:60]}...")
        return
    raise AssertionError("shape 冲突时应当报错，而不是静默覆盖")


@test
def test_unknown_input_name_raises():
    try:
        optimize(build_static_model(), {"not_an_input": [1, 3, 8, 8]})
    except KeyError as exc:
        print(f"  正确报错：{exc}")
        return
    raise AssertionError("输入名不存在时应当报错")


# ======================================================================
# 求值器算子语义单测
# ======================================================================
@test
def test_evaluator_op_semantics():
    from onnx import TensorProto, helper, numpy_helper

    def t(name, arr):
        return numpy_helper.from_array(np.asarray(arr), name)

    inits = [
        t("shape_in", np.array([2, 3, 4, 5], dtype=np.int64)),
        t("idx", np.array([2, 3], dtype=np.int64)),
        t("ax0", np.array([0], dtype=np.int64)),
        t("neg1", np.array([-1], dtype=np.int64)),
        t("two", np.array([2], dtype=np.int64)),
        t("starts", np.array([1], dtype=np.int64)),
        t("ends", np.array([3], dtype=np.int64)),
        t("axes", np.array([0], dtype=np.int64)),
        t("squeeze_ax", np.array([0], dtype=np.int64)),
    ]
    nodes = [
        # Gather(axis=0, indices=[2,3]) -> [4, 5]
        helper.make_node("Gather", ["shape_in", "idx"], ["g"], axis=0),
        # Mul([4,5],[2]) -> [8,10]
        helper.make_node("Mul", ["g", "two"], ["m"]),
        # Concat([-1], [8,10]) -> [-1,8,10]
        helper.make_node("Concat", ["neg1", "m"], ["c"], axis=0),
        # Unsqueeze([-1,8,10], axes=[0]) -> [[-1,8,10]]
        helper.make_node("Unsqueeze", ["c", "ax0"], ["u"]),
        # Squeeze back
        helper.make_node("Squeeze", ["u", "squeeze_ax"], ["s"]),
        # Slice([2,3,4,5], starts=[1], ends=[3], axes=[0]) -> [3,4]
        helper.make_node("Slice", ["shape_in", "starts", "ends", "axes"], ["sl"]),
        # Cast to float
        helper.make_node("Cast", ["s"], ["cf"], to=TensorProto.FLOAT),
        # Div int: 10 / 3
        helper.make_node("Div", ["m", "two"], ["dv"]),
    ]
    graph = helper.make_graph(
        nodes, "eval_unit",
        [], [helper.make_tensor_value_info("cf", TensorProto.FLOAT, [3]),
             helper.make_tensor_value_info("sl", TensorProto.INT64, [2]),
             helper.make_tensor_value_info("dv", TensorProto.INT64, [2])],
        initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    ev = ShapeTensorEvaluator(GraphView(model))
    assert evaluated(ev, "g").tolist() == [4, 5]
    assert evaluated(ev, "m").tolist() == [8, 10]
    assert evaluated(ev, "c").tolist() == [-1, 8, 10]
    assert evaluated(ev, "u").shape == (1, 3)
    assert evaluated(ev, "s").tolist() == [-1, 8, 10]
    assert evaluated(ev, "sl").tolist() == [3, 4]
    assert evaluated(ev, "cf").tolist() == [-1.0, 8.0, 10.0]
    assert evaluated(ev, "dv").tolist() == [4, 5]
    # 不存在的名字 -> None，且要缓存住
    assert ev.evaluate("does_not_exist") is None
    assert "does_not_exist" in ev.memo
    print(f"  stats: {ev.stats}")


@test
def test_evaluator_shape_of_non_constant():
    """Shape 的输入是数据张量（非常量），但 shape 已知 -> 必须能求值。"""
    from onnx import TensorProto, helper

    nodes = [helper.make_node("Shape", ["x"], ["s"])]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])]
    outputs = [helper.make_tensor_value_info("s", TensorProto.INT64, [4])]
    graph = helper.make_graph(nodes, "shape_of_input", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    ev = ShapeTensorEvaluator(GraphView(model))
    assert evaluated(ev, "s").tolist() == [1, 3, 8, 8]
    # 而数据张量本身必须求不出来
    assert ev.evaluate("x") is None
    print("  Shape(非 const) 求值正确；x 本身正确返回 None")


@test
def test_evaluator_cycle_guard():
    """成环时不能无限递归。"""
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Add", ["b", "one"], ["a"]),
        helper.make_node("Add", ["a", "one"], ["b"]),
    ]
    graph = helper.make_graph(
        nodes, "cycle", [], [
            helper.make_tensor_value_info("a", TensorProto.INT64, [1])],
        initializer=[helper.make_tensor(
            "one", TensorProto.INT64, [1], [1])])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    ev = ShapeTensorEvaluator(GraphView(model))
    assert ev.evaluate("a") is None       # 有环 -> 不可求值，且不挂
    print("  环上安全返回 None")


# ======================================================================
# 回归：多输出节点的**兄弟输出**不能被折叠吃掉
# ======================================================================
@test
def test_multi_output_producer_stays_valid():
    """Split 的两个输出各走一条路，折叠时不能只换掉 output[0]。

    ``replace_node_with_constant`` 只认 ``node.output[0]``。若按单个 value
    折叠 ``output[1]``，会把值写到 ``output[0]`` 名下、兄弟输出直接消失，
    onnx.checker 报 "input 'b' ... is not output of any previous nodes"。
    这是原 Pass 3 的 bug；Pass 3 已移除，本用例继续守 Pass 4 的行为。
    """
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Shape", ["images"], ["s"]),
        helper.make_node("Split", ["s"], ["a", "b"], axis=0),   # a=[1], b=[3,8,8]
        helper.make_node("Cast", ["a"], ["af"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["af", "one"], ["act"]),        # a 这条必须活着
        helper.make_node("Reshape", ["y", "b"], ["r"]),         # b 是 shape 参数
    ]
    g = helper.make_graph(
        nodes, "multi_out",
        [helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, ["N", 3, "H", "W"]),
         helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("act", TensorProto.FLOAT, ["A"]),
         helper.make_tensor_value_info("r", TensorProto.FLOAT, ["X", "Y"])],
        initializer=[helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])])
    model = helper.make_model(
        g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    result = optimize(model, {"images": [1, 3, 8, 8]})
    assert result.checker_ok, result.checker_message
    assert result.numerical_ok is True, result.numerical_message
    # b 必须仍然有生产者（要么是 Split，要么已被合法地折成 initializer/Constant）
    gv = GraphView(result.model)
    producer = gv.producer("b")
    assert producer is not None or "b" in gv.initializer_map, (
        f"Split 的兄弟输出 b 凭空消失了；nodes={[n.op_type for n in gv.graph.node]}")
    print(f"  nodes: {result.nodes_before} -> {result.nodes_after}, "
          f"checker/numeric 均通过")


# ======================================================================
# 回归：Pass 5 的 ORT 子模型必须有 ir_version
# ======================================================================
@test
def test_pass5_folds_ops_without_handler():
    """评估器没有 handler 的算子，Pass 5 必须能用 ORT 折掉。

    历史 bug：``fold_via_ort`` 构造子模型时漏了 ``ir_version``，ORT 报
    "Missing model IR version"，于是每一次折叠都静默失败、Pass 5 形同虚设。
    """
    from onnx import TensorProto, helper

    from onnx_static_optimizer.config import OptimizerConfig
    from onnx_static_optimizer.passes.constant.constant_folding import (
        ConstantFoldingPass,
    )

    # Erf 不在 ShapeTensorEvaluator.HANDLERS 里，故 Pass 4 无能为力。
    nodes = [
        helper.make_node("MatMul", ["w", "w2"], ["mm"]),
        helper.make_node("Erf", ["mm"], ["er"]),
        helper.make_node("Relu", ["x"], ["y"]),
    ]
    inits = [
        helper.make_tensor("w", TensorProto.FLOAT, [3, 3],
                           list(np.arange(9, dtype=np.float32) / 9)),
        helper.make_tensor("w2", TensorProto.FLOAT, [3, 3],
                           list(np.ones((3, 3), dtype=np.float32) / 3)),
    ]
    g = helper.make_graph(
        nodes, "ort_fold",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("er", TensorProto.FLOAT, [3, 3])],
        initializer=inits)
    model = helper.make_model(
        g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    from onnx_static_optimizer.passes.constant.constant_folding import (
        ort_module,
    )
    if ort_module() is None:
        print("  skipped: onnxruntime 未安装")
        return

    gv = GraphView(model)
    p = ConstantFoldingPass(OptimizerConfig())
    assert p.run(gv), "Pass 5 应当折掉 Erf 子图，实际没有变更"
    assert p.detail.get("node_folded", 0) >= 1, p.stats()
    assert "Erf" not in op_counts(model), "Erf 未被折叠"
    assert "er" in {i.name for i in model.graph.initializer}
    print(f"  folded: {p.detail}, nodes={[n.op_type for n in model.graph.node]}")



# ======================================================================
def main():
    failed = []
    for fn in TESTS:
        print("=" * 62)
        print(fn.__name__)
        print("=" * 62)
        try:
            fn()
            print("PASS\n")
        except Exception as exc:
            traceback.print_exc()
            failed.append((fn.__name__, exc))
            print("FAIL\n")

    print("=" * 62)
    if failed:
        print(f"{len(failed)}/{len(TESTS)} FAILED:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        return 1
    print(f"ALL {len(TESTS)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
