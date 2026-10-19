"""主入口：流水线组装与执行。

流水线顺序（pass_implementation.md「附：执行顺序与收敛」）::

    第一轮（动转静）:  P1  P2  P3  P4  P5  P6  P7  P8  P9
    第二轮（图简化）:  P11 P12 P13 P14
    第三轮（清理）:    P10 P15

为什么要循环
------------
单个 Pass 常常会「解锁」上游的新机会::

    P6 消除 Shape(X)  -> X 的链路变死代码      -> 需要 P10 DCE
    P3 折叠 shape 子图 -> 产生新 Constant      -> 需要 P4 再次传播
    P7 删除 Reshape    -> 上游 Transpose 可能变成 identity -> 需要 P13

所以用 ``run_until_converged``（MAX_ITER=10）反复跑整条链，直到某一轮零变更。
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from onnx import ModelProto

from .analysis.graph_statistics import analyze
from .config import OptimizerConfig, ShapeSpec
from .graph.graph_editor import topological_sort
from .graph.graph_view import GraphView
from .passes.base import Pass, PassManager
from .passes.cleanup import (
    DCEPass,
    InitializerCleanupPass,
    UnusedValueInfoCleanupPass,
)
from .passes.constant import ConstantFoldingPass, ConstantPropagationPass
from .passes.shape import (
    ExpandEliminationPass,
    ReshapeEliminationPass,
    ShapeEliminationPass,
    ShapeTensorEvaluationPass,
    SliceEliminationPass,
)
from .passes.structural import (
    CastSimplifyPass,
    IdentityEliminationPass,
    ReshapeSimplifyPass,
    TransposeSimplifyPass,
)
from .shape.shape_inference import ShapeInferencePass, run_shape_inference
from .shape.shape_specializer import resolve_shape_spec, specialize_input_shape
from .validation.checker import (
    check_model,
    static_input_shapes,
    validate_numerically,
)
from .validation.static_validator import StaticValidationReport, validate


def build_pipeline(config: OptimizerConfig) -> List[Pass]:
    """组装 Pass 流水线。

    注意：Pass 1（Input Shape Specialization）**不在**这里 ——
    它单独在循环外执行，这样配置错误（输入名写错、shape 冲突）能直接抛给
    用户，而不是被 PassManager 的异常保护吞掉。
    """
    return [
        # --- 第一轮：动转静 ---
        ShapeInferencePass(config),
        ShapeTensorEvaluationPass(config),
        ConstantPropagationPass(config),
        ConstantFoldingPass(config),
        ShapeEliminationPass(config),
        ReshapeEliminationPass(config),
        SliceEliminationPass(config),
        ExpandEliminationPass(config),
        # --- 第二轮：图简化 ---
        IdentityEliminationPass(config),
        ReshapeSimplifyPass(config),
        TransposeSimplifyPass(config),
        CastSimplifyPass(config),
        # --- 第三轮：清理 ---
        DCEPass(config),
        InitializerCleanupPass(config),
        # 前两步会留下悬空的 value_info 条目，必须清掉，
        # 否则 StaticValidator 会把"自己制造的残留"报成动态项。
        UnusedValueInfoCleanupPass(config),
    ]


# ----------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """优化结果。"""

    model: ModelProto
    original_model: ModelProto
    config: OptimizerConfig
    iterations: int = 0
    converged: bool = False
    pass_stats: List[dict] = field(default_factory=list)
    stalled_passes: List[str] = field(default_factory=list)
    # optimize() 总会填充该字段；这里给一个空报告作为默认值，
    # 使类型保持非 Optional，调用方无需再做 None 判空。
    validation: StaticValidationReport = field(
        default_factory=StaticValidationReport)
    checker_ok: bool = False
    checker_message: str = ""
    numerical_ok: Optional[bool] = None
    numerical_message: str = ""
    nodes_before: int = 0
    nodes_after: int = 0
    initializers_before: int = 0
    initializers_after: int = 0
    input_shapes_before: Dict[str, str] = field(default_factory=dict)
    input_shapes_after: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import onnx
        onnx.save(self.model, path)

    @property
    def is_static(self) -> bool:
        return self.validation.is_static

    def format(self) -> str:
        """返回验证报告的格式化字符串。"""
        return self.validation.format()

    def detail_summary(self) -> Dict[str, int]:
        """聚合各 Pass 的**真实变更**计数。"""
        agg: Dict[str, int] = {}
        for st in self.pass_stats:
            for k, v in st.get("detail", {}).items():
                agg[k] = agg.get(k, 0) + v
        return agg

    def notes_summary(self) -> Dict[str, int]:
        """聚合各 Pass 的**诊断计数**（跳过 / 失败），不计入变更。"""
        agg: Dict[str, int] = {}
        for st in self.pass_stats:
            for k, v in st.get("notes", {}).items():
                agg[k] = agg.get(k, 0) + v
        return agg

    # ------------------------------------------------------------------
    def report(self) -> str:
        L = "=" * 52
        lines = [L, "ONNX Static Optimization Report", L, ""]

        names = [i.name for i in self.original_model.graph.input]
        lines.append(f"Inputs: {names}")
        for name in names:
            before = self.input_shapes_before.get(name, "?")
            after = self.input_shapes_after.get(name, "?")
            mark = "->" if before != after else "=="
            lines.append(f"  {name}: {before} {mark} {after}")
        lines.append(f"Iterations: {self.iterations} "
                     f"({'converged' if self.converged else 'NOT converged'})")
        lines.append("")

        # --- 图规模 ---
        lines.append("Graph Statistics")
        lines.append("-" * 16)
        lines.append(f"Nodes:        {self.nodes_before} -> {self.nodes_after}"
                     f"  ({self.nodes_after - self.nodes_before:+d})")
        lines.append(f"Initializers: {self.initializers_before} -> "
                     f"{self.initializers_after}"
                     f"  ({self.initializers_after - self.initializers_before:+d})")
        lines.append("")

        # --- 静态性 ---
        lines.append(self.validation.format())
        lines.append("")

        # --- 简化统计 ---
        detail = self.detail_summary()
        if detail:
            lines.append("Simplification")
            lines.append("-" * 14)
            for k, v in sorted(detail.items(), key=lambda kv: -kv[1]):
                lines.append(f"{k:<32} {v}")
            lines.append("")

        # --- 跳过的项（诊断用，不是变更）---
        notes = self.notes_summary()
        if notes:
            lines.append("Skipped / Diagnostics")
            lines.append("-" * 20)
            for k, v in sorted(notes.items(), key=lambda kv: -kv[1]):
                lines.append(f"{k:<32} {v}")
            lines.append("")

        # --- 验证 ---
        lines.append("Validation")
        lines.append("-" * 10)
        lines.append(f"{'ONNX checker:':<21}"
                     f"{'PASS' if self.checker_ok else 'FAIL'}")
        lines.append(f"{'Static shape:':<21}"
                     f"{'PASS' if self.is_static else 'FAIL'}")
        if self.numerical_ok is None:
            lines.append(f"{'Numerical equiv:':<21}SKIPPED  "
                         f"({self.numerical_message})")
        else:
            lines.append(f"{'Numerical equiv:':<21}"
                         f"{'PASS' if self.numerical_ok else 'FAIL'}  "
                         f"({self.numerical_message})")

        if not self.checker_ok:
            lines.append("")
            lines.append(f"checker error: {self.checker_message}")
        if not self.converged:
            lines.append("")
            lines.append(f"WARNING: 未在 {self.config.max_iterations} 轮内收敛。")
            lines.append(f"  仍在报变更的 Pass（不幂等嫌疑）：{self.stalled_passes}")
        lines.append(L)
        return "\n".join(lines)


# ----------------------------------------------------------------------
def _clone(model: ModelProto) -> ModelProto:
    new = ModelProto()
    new.CopyFrom(model)
    return new


def _sync_output_types(graph) -> bool:
    """把 graph.output 声明的类型同步成实际已知的类型。

    折叠后 graph output 可能已经变成 initializer（例如一个纯常量输出），
    但 graph.output 里声明的还是导出时的符号 shape。不同步的话
    StaticValidator 会误报"dynamic output"。
    """
    changed = False
    for out in graph.graph.output:
        init = graph.initializer_map.get(out.name)
        if init is None:
            continue
        tt = out.type.tensor_type
        if tt.elem_type != init.data_type:
            tt.elem_type = init.data_type
            changed = True
        dims = list(init.dims)
        current = [int(d.dim_value) for d in tt.shape.dim
                   if d.HasField("dim_value")]
        if current == dims and len(current) == len(tt.shape.dim):
            continue
        del tt.shape.dim[:]
        for d in dims:
            tt.shape.dim.add().dim_value = int(d)
        changed = True
    return changed


def _describe_input_shape(model) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for inp in model.graph.input:
        if not inp.type.HasField("tensor_type"):
            out[inp.name] = "?"
            continue
        tt = inp.type.tensor_type
        if not tt.HasField("shape"):
            out[inp.name] = "?"
            continue
        dims = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(str(int(d.dim_value)))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        out[inp.name] = "[" + ",".join(dims) + "]"
    return out


def _resolve_config_shapes(model,
                           specs: Dict[str, ShapeSpec]) -> Dict[str, List[int]]:
    """把配置里的 ``ShapeSpec`` 解析成 ``{输入名: [具体维度]}``。

    ``static_input_shapes`` 拿不到完整静态 shape 时的兜底：符号名映射
    按原模型声明的维度解析，保证交给数值验证的一定是 ``List[int]``
    （``make_feeds`` 会 ``list(input_shapes[name])``，字典形式会被误读）。
    解析不出来的输入直接跳过，由 ``make_feeds`` 报"输入 shape 未完全静态"。
    """
    shapes: Dict[str, List[int]] = {}
    for inp in model.graph.input:
        spec = specs.get(inp.name)
        if spec is None or not inp.type.HasField("tensor_type"):
            continue
        try:
            shapes[inp.name] = resolve_shape_spec(
                inp.type.tensor_type.shape, spec, inp.name)
        except (ValueError, KeyError):
            continue
    return shapes


def optimize(model: Union[str, ModelProto],
             input_shapes: Optional[Dict[str, ShapeSpec]] = None,
             config: Optional[OptimizerConfig] = None) -> OptimizationResult:
    """把动态 ONNX 模型静态化。

    Parameters
    ----------
    model
        输入模型：文件路径或 ``ModelProto``。**不会被就地修改。**
    input_shapes
        用户指定的输入形状，如 ``{"images": [1, 3, 640, 640]}``。
        会覆盖 ``config.input_shapes``。
    config
        优化器配置。为 None 时用默认配置。
    """
    if isinstance(model, str):
        import onnx
        source = onnx.load(model)
    else:
        source = model

    config = config or OptimizerConfig()
    if input_shapes:
        config.input_shapes = dict(input_shapes)

    original = _clone(source)
    working = _clone(source)

    result = OptimizationResult(
        model=working,
        original_model=original,
        config=config,
        input_shapes_before=_describe_input_shape(source),
    )

    graph = GraphView(working)
    graph.preserve_output_names = config.preserve_graph_output_names
    stats_before = analyze(working)

    # ------------------------------------------------------------------
    # Pass 1 单独执行：配置错误必须让用户看到，不能被 PassManager 吞掉
    # ------------------------------------------------------------------
    specialize_input_shape(working, config.input_shapes)
    graph.rebuild()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    manager = PassManager(build_pipeline(config),
                          max_iter=config.max_iterations)
    # manager._verbose = config.verbose
    manager.run_until_converged(graph)

    result.iterations = manager.iterations
    result.converged = manager.converged
    result.pass_stats = manager.stats()
    result.stalled_passes = manager.stalled_passes()

    # ------------------------------------------------------------------
    # 收尾：拓扑排序 + 最后一次 shape 推导 + 同步输出声明
    # ------------------------------------------------------------------
    topological_sort(graph)
    run_shape_inference(working, strict=False, data_prop=True)
    graph.rebuild()
    _sync_output_types(graph)
    graph.rebuild()

    stats_after = analyze(working)
    result.nodes_before = stats_before.nodes
    result.nodes_after = stats_after.nodes
    result.initializers_before = stats_before.initializers
    result.initializers_after = stats_after.initializers
    result.input_shapes_after = _describe_input_shape(working)

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    result.validation = validate(working, graph)
    result.checker_ok, result.checker_message = check_model(working)

    if config.validate_numerically:
        shapes = static_input_shapes(working) or _resolve_config_shapes(
            original, config.input_shapes)
        try:
            ok, msg = validate_numerically(
                original, working, shapes,
                rtol=config.numeric_rtol, atol=config.numeric_atol,
                seed=config.seed,
            )
            result.numerical_ok, result.numerical_message = ok, msg
        except Exception as exc:
            result.numerical_ok = None
            result.numerical_message = f"验证异常：{exc}"
    else:
        result.numerical_ok = None
        result.numerical_message = "已按配置关闭"

    return result


__all__ = ["optimize", "build_pipeline", "OptimizationResult"]
