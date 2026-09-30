# ONNX 动转静优化器设计文档

## 一、项目背景与目标

### 1.1 问题背景

从 PyTorch 导出的 ONNX 模型通常带有大量动态维度（如 `[N, 3, H, W]`），而 RKNN、地平线、算能等 NPU 编译器要求输入必须是完全静态的模型。这导致开发者不得不手动处理复杂的 shape 推导、动态算子消除等工作。

### 1.2 核心目标

构建一个纯 Python 的 ONNX 静态化工具，将动态 ONNX 模型转换为 NPU 友好的静态 ONNX 模型：

```
输入: Dynamic ONNX  [N, 3, H, W]，或者三维输入，例如：[N, C, 80]
      用户指定: N=1, H=640, W=640
输出: Static ONNX   [1, 3, 640, 640]
```

### 1.3 与 ONNX Runtime 优化的区别

ONNX Runtime 的 Graph Optimization 主要目标是**推理性能优化**（包括 Constant Folding、冗余节点消除、Conv/BN 融合、Layout Optimization）。而本项目的目标是**Static Shape Specialization / Shape Computation Elimination**，核心是将动态 shape "钉死"，让 NPU 编译器能正确解析和优化。

## 二、整体架构

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Hardware Layer                           │
│              RKNN / Horizon / Sophgo / Generic              │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                   Graph Optimization Layer                  │
│              Fusion / Simplification / DCE                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                    Staticization Layer  ⭐ 核心创新点        │
│                                                           │
│    Shape Specialization → Shape Evaluation →               │
│    Constant Folding → Dynamic Operator Elimination         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Pass 流水线

```
                    Dynamic ONNX
                         │
                         ▼
              ┌─────────────────────┐
              │ 1. Model Validation │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 2. Shape Spec.      │  ← 指定输入具体形状
              │ 3. Shape Inference  │  ← 传播已知 shape
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 4. Shape Tensor     │  ← 核心：求值 shape 子图
              │     Evaluation      │
              │ 5. Constant Prop.   │  ← 常量传播
              │ 6. Constant Folding │  ← 常量折叠
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 7. Dynamic Op       │  ← Reshape/Slice/Expand 静态化
              │     Elimination     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 8. Structural       │
              │     Simplification  │  ← Identity/Transpose/Cast 简化
              │ 9. Arithmetic       │
              │     Simplification  │  ← Add 0, Mul 1 等消除
              │ 10. Control Flow    │  ← If/Loop 常量化
              │     Simplification  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 11. DCE + Cleanup   │  ← 死代码消除
              │ 12. Operator Fusion │  ← Conv+BN, MatMul+Add
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 13. Final Shape     │
              │     Inference       │
              │ 14. Static          │  ← 验证是否真正静态化
              │     Validation      │
              └──────────┬──────────┘
                         ▼
                    Static ONNX
```

## 三、核心设计理念

### 3.1 先静态化，再性能优化

不要一上来就做 Conv-BN Fusion、Layout Optimization。优先确保 NPU 编译器能接受这个图。性能优化是后续阶段的事。

### 3.2 Shape Graph 和 Data Graph 分开处理

ONNX 图实际上包含两类计算：

```
                    ONNX Graph
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     Data Graph                    Shape Graph
          │                             │
       Conv                          Shape
       MatMul                        Gather
       Relu                          Concat
       Add                           Cast
       ...                           Reshape
```

优化器应重点识别哪些节点是在计算 Tensor 数据，哪些节点只是在计算 Tensor 的 shape。然后优先把 Shape Graph 求值。

### 3.3 能证明安全才删除

每个 Simplification Pass 都应该有：
- **Pattern**: 匹配的子图模式
- **Precondition**: 前提条件
- **Rewrite**: 安全转换规则

例如 `Mul(X, 1)` 看起来可以删除，但需要考虑 dtype、NaN、Inf、broadcasting、FP16 等情况。

### 3.4 Pass 幂等性

执行 `Pass(); Pass(); Pass()` 最终应该 `Graph1 == Graph2`。例如 `Transpose ∘ Transpose` 第一次消掉后，第二次不能再修改。

### 3.5 循环执行至收敛

```
ShapeInference
      ↓
ConstantFolding
      ↓
DCE
      ↓
ShapeInference
      ↓
ConstantFolding
      ↓
DCE
      ↓
   ...
直到 graph_hash_before == graph_hash_after
```

## 四、Pass 详细分类

### 4.1 P0: Shape 核心（动转静关键）

#### 4.1.1 Input Shape Specialization

用户指定具体输入形状：

```python
input_shapes = {
    "images": [1, 3, 640, 640]
}

# 将 dim_param / unknown 转为具体 dim_value
# [N, 3, H, W] → [1, 3, 640, 640]
```

#### 4.1.2 Shape Tensor Evaluator（最重要）

这是整个系统的核心。专门用于求值 shape 计算子图：

```
原始图:
    Shape(X)
       │
       ▼
    Gather(axis=0)
       │
       ▼
    Unsqueeze
       │
       ▼
    Concat
       │
       ▼
    Reshape

求值后:
    Constant([1, 3, 640, 640])
```

需要支持的算子分为三阶段：

**第一阶段（基础）**：
- Constant, Shape, Size, Rank
- Cast, Gather, Concat
- Unsqueeze, Squeeze, Reshape

**第二阶段（算术）**：
- Slice, Add, Sub, Mul, Div
- Mod, Min, Max

**第三阶段（高级）**：
- Range, Expand, Where
- GatherND, Split

#### 4.1.3 Dynamic Operator Elimination

| 算子类型 | 转换示例 |
|---------|---------|
| Dynamic Reshape | `Reshape(X, [0,-1,80])` → `Reshape(X, [1,8400,80])` |
| Dynamic Slice | `Slice(X, starts=dynamic)` → `Slice(X, starts=[0,0,0])` |
| Dynamic Expand | `Expand(X, target_shape)` → 若相等则删除 |
| Shape/Size/Rank | 已知 shape 时 → Constant |

### 4.2 P1: Constant 相关

#### 4.2.1 Constant Propagation

```
Constant(3) → Add → Constant(4)
                   ↓
              Constant(7)
```

#### 4.2.2 Constant Folding

采用两级策略：
- **Level 1**: 自己实现 shape operators 的常量求值
- **Level 2**: 对于复杂计算，使用 ONNX Runtime CPU 执行子图

注意：不要把整个模型交给 ORT 跑一遍来做 Constant Folding，需要控制哪些节点可以折叠。

### 4.3 P2: 结构简化

#### 4.3.1 Identity Elimination

```
A → Identity → B  →  A → B
```

#### 4.3.2 Reshape Simplification

- `Reshape ∘ Reshape` → 保留最后一个
- `Reshape(X)` 输入输出 shape 相同 → 删除
- `Flatten ∘ Reshape` 合并

#### 4.3.3 Transpose Simplification

- `Transpose(identity_perm)` → 删除
- `Transpose(A) ∘ Transpose(inverse(A))` → 删除
- 连续的 Transpose 合并

#### 4.3.4 Cast Simplification

- `Cast(A) ∘ Cast(A)` → 合并为一个 Cast
- `Cast(FP32→FP16) ∘ Cast(FP16→FP32)` → 删除

#### 4.3.5 Squeeze/Unsqueeze

- `Unsqueeze ∘ Squeeze` → Identity 或删除
- 多个连续操作合并

#### 4.3.6 Arithmetic Simplification

- `Add(X, 0)` → `X`
- `Sub(X, 0)` → `X`
- `Mul(X, 1)` → `X`
- `Div(X, 1)` → `X`

注意：`Mul(X, 0)` 需要处理 NaN/Inf 语义，不建议第一版实现。

### 4.4 P3: 算子融合

#### 4.4.1 Conv + BN Fusion

```
Conv → BatchNormalization  →  Conv'
```

融合公式：
```
W' = W * (γ / sqrt(σ² + ε))
b' = β + (b - μ) * (γ / sqrt(σ² + ε))
```

#### 4.4.2 MatMul + Add → Gemm

```
MatMul → Add(bias)  →  Gemm
```

注意需要考虑目标 NPU 对 Gemm 的支持情况。

### 4.5 P4: 控制流简化

#### 4.5.1 If Simplification

```
If(condition=Constant(True))  →  then_branch
If(condition=Constant(False)) →  else_branch
```

#### 4.5.2 Loop Unrolling

```
Loop(trip_count=Constant(3))  →  Body × 3
```

### 4.6 P5: 清理

#### 4.6.1 Dead Code Elimination (DCE)

从 `graph.output` 反向遍历，删除没有到达输出的节点。

#### 4.6.2 Initializer Cleanup

删除没有被任何节点使用的 initializer。

#### 4.6.3 ValueInfo Cleanup

最终只保留 `graph.input`、`graph.output` 和必需的 value_info。

## 五、项目结构

```
onnx_static_optimizer/
├── __init__.py
├── optimizer.py              # 主入口，Pass 管理器
├── config.py                 # 配置
│
├── graph/
│   ├── graph_view.py         # 图视图，封装节点操作
│   ├── graph_editor.py       # 图编辑工具
│   ├── node_utils.py         # 节点工具函数
│   ├── tensor_utils.py       # Tensor 工具
│   └── initializer_utils.py  # Initializer 工具
│
├── shape/
│   ├── shape_specializer.py  # 输入形状定形
│   ├── shape_inference.py    # Shape Inference
│   ├── shape_evaluator.py    # ⭐ Shape Tensor 求值器
│   ├── shape_analyzer.py     # 动态分析
│   └── symbolic_shape.py     # 符号形状
│
├── passes/
│   ├── base.py               # Pass 基类
│   │
│   ├── constant/
│   │   ├── constant_propagation.py
│   │   ├── constant_folding.py
│   │   └── initializer_dedup.py
│   │
│   ├── shape/
│   │   ├── eliminate_shape.py
│   │   ├── eliminate_size.py
│   │   ├── eliminate_rank.py
│   │   ├── reshape_simplify.py
│   │   ├── slice_simplify.py
│   │   ├── gather_simplify.py
│   │   ├── concat_simplify.py
│   │   └── expand_simplify.py
│   │
│   ├── structural/
│   │   ├── identity_elimination.py
│   │   ├── transpose_simplify.py
│   │   ├── squeeze_simplify.py
│   │   ├── unsqueeze_simplify.py
│   │   └── cast_simplify.py
│   │
│   ├── arithmetic/
│   │   ├── arithmetic_simplify.py
│   │   └── broadcast_simplify.py
│   │
│   ├── control_flow/
│   │   ├── if_simplify.py
│   │   ├── loop_unroll.py
│   │   └── scan_simplify.py
│   │
│   ├── fusion/
│   │   ├── conv_bn.py
│   │   ├── conv_add.py
│   │   └── matmul_add.py
│   │
│   └── cleanup/
│       ├── dce.py
│       ├── unused_initializer.py
│       └── unused_value_info.py
│
├── analysis/
│   ├── dynamic_analyzer.py   # 动态分析器
│   ├── graph_statistics.py   # 图统计
│   └── dependency.py         # 依赖分析
│
├── validation/
│   ├── checker.py            # ONNX 标准检查
│   ├── static_validator.py   # ⭐ 静态性验证
│   └── numerical_validator.py # 数值等价验证
│
└── cli.py                    # 命令行工具
```

## 六、核心数据结构

### 6.1 GraphView

封装 ONNX 图操作，避免直接操作 `model.graph.node`：

```python
class GraphView:
    nodes              # 节点列表
    inputs             # 输入列表
    outputs            # 输出列表
    initializers       # 初始化器
    value_info         # ValueInfo

    producer_map       # output_name → node 映射
    consumer_map       # input_name → [nodes] 映射

    def producer(self, name): ...      # 获取生产者
    def consumers(self, name): ...     # 获取消费者列表
    def is_constant(self, name): ...   # 是否为常量
    def get_shape(self, name): ...     # 获取 shape
```

### 6.2 Pass 基类

```python
class Pass:
    name: str

    def run(self, graph: GraphView) -> bool:
        """返回是否发生变更"""
        ...

    def changed(self) -> bool:
        ...

    def statistics(self) -> dict:
        ...
```

### 6.3 PassManager

```python
class PassManager:
    def add(self, pass_: Pass): ...

    def run(self, graph: GraphView) -> bool:
        """执行所有 pass，返回是否发生变更"""

    def run_until_converged(self, graph: GraphView, max_iter=10):
        """循环执行直到图不再变化"""
```

## 七、推荐 Pipeline

### 7.1 Phase A: 动转静核心（必须）

```
01 InputShapeSpecialization     # 指定输入形状 [1,3,640,640]
02 ShapeInference               # 传播已知 shape
03 ShapeTensorEvaluation        # ⭐ 求值 shape 子图
04 ConstantPropagation          # 常量传播
05 ConstantFolding              # 常量折叠
06 ShapeElimination             # Shape/Size/Rank → Constant
07 DynamicReshapeElimination    # 动态 Reshape 静态化
08 DynamicSliceElimination      # 动态 Slice 静态化
09 DynamicExpandElimination     # 动态 Expand 静态化
```

### 7.2 Phase B: 图简化

```
10 ReshapeSimplification        # Reshape 链简化
11 TransposeSimplification      # Transpose 链简化
12 CastSimplification           # Cast 链简化
13 SqueezeSimplification        # Squeeze/Unsqueeze 简化
14 IdentityElimination          # Identity 删除
15 ArithmeticSimplification     # Add 0, Mul 1 等
16 ControlFlowSimplification    # If/Loop 常量化
```

### 7.3 Phase C: 清理与验证

```
17 DCE                          # 死代码消除
18 InitializerCleanup           # 清理未使用 initializer
19 ValueInfoCleanup             # 清理未使用 value_info
20 ConvBNFusion                 # 算子融合
21 FinalShapeInference          # 最终 shape 推导
22 StaticValidation             # ⭐ 静态性验证
23 NumericalValidation          # 数值等价验证
```

### 7.4 迭代执行

建议设置 `MAX_ITER = 10`，循环执行直到收敛：

```
for i in range(MAX_ITER):
    changed = False
    for p in passes:
        if p.run(graph):
            changed = True
    if not changed:
        break
```

## 八、验证体系

### 8.1 Static Validator（核心）

必须检查：

| 检查项 | 说明 |
|-------|------|
| Dynamic inputs | 所有输入是否有明确 dim_value |
| Dynamic outputs | 所有输出是否有明确 dim_value |
| Dynamic ValueInfo | 是否还有 dim_param |
| Dynamic Reshape | shape input 是否静态 |
| Dynamic Slice | starts/ends/axes/steps 是否静态 |
| Dynamic Expand | shape 是否静态 |
| Shape nodes | 是否还有 Shape operator |
| Size nodes | 是否还有 Size operator |
| Rank nodes | 是否还有 Rank operator |

### 8.2 区分两种动态

```
Shape-Dependent Dynamic    ✓ 可静态化
    Shape(X) → Gather → Concat → Reshape

Data-Dependent Dynamic     ✗ 无法静态化
    NonZero(X)
    TopK(X, dynamic_k)
    Where(condition)
    NonMaxSuppression
```

后者输出数量取决于实际数据，必须在 validator 中明确报告。

### 8.3 数值等价验证

```
Original ONNX  ──ORT──→  reference_output
                    │
Optimized ONNX ──ORT──→  optimized_output
                    │
                    ▼
            np.allclose(ref, opt, rtol=1e-4, atol=1e-5)
```

## 九、最终报告示例

```
==================================================
ONNX Static Optimization Report
==================================================

Model: yolov8.onnx
Input: images [N,3,H,W] → [1,3,640,640]

Graph Statistics
----------------
Nodes:       412 → 286  (-126)
Initializers: 238 → 214  (-24)

Dynamic Shape
-------------
Dynamic inputs:      0  ✓
Dynamic outputs:     0  ✓
Dynamic ValueInfo:   0  ✓
Shape nodes:         0  ✓
Size nodes:          0  ✓
Rank nodes:          0  ✓
Dynamic Reshape:     0  ✓
Dynamic Slice:       0  ✓
Dynamic Expand:      0  ✓

Simplification
--------------
Constant folded:     94
Identity removed:    17
Reshape removed:     11
Transpose removed:    4
Cast removed:         8
DCE nodes:           73

Validation
----------
ONNX checker:        PASS
Static shape:        PASS
ORT inference:       PASS
Numerical equiv:     PASS
==================================================
```

## 十、最值得优先实现的 Pass

如果分阶段实施，第一版 MVP 建议按此优先级实现：

| 优先级 | Pass | 价值 |
|-------|------|------|
| ★★★★★ | Input Shape Specialization | 动转静起点 |
| ★★★★★ | Shape Inference | 传播已知 shape |
| ★★★★★ | Shape Tensor Evaluator | 核心求值器 |
| ★★★★★ | Constant Propagation | 常量传播 |
| ★★★★★ | Constant Folding | 常量折叠 |
| ★★★★★ | Shape/Size/Rank Elimination | 消除动态算子 |
| ★★★★★ | Dynamic Reshape Elimination | 消除动态 Reshape |
| ★★★★★ | Dynamic Slice Elimination | 消除动态 Slice |
| ★★★★★ | Dynamic Expand Elimination | 消除动态 Expand |
| ★★★★★ | DCE | 死代码消除 |
| ★★★★☆ | Identity Elimination | 结构简化 |
| ★★★★☆ | Reshape Simplification | 结构简化 |
| ★★★★☆ | Transpose Simplification | 结构简化 |
| ★★★★☆ | Cast Simplification | 结构简化 |
| ★★★★☆ | Initializer Cleanup | 清理优化 |

先做好这 15 个 Pass，已经能解决大部分"动态 ONNX → NPU 静态 ONNX"的问题。

## 十一、设计决策总结

1. **先静态化后优化**：确保 NPU 编译器能接受，再考虑性能
2. **Shape Tensor Evaluator 是核心**：实现可靠的 shape 子图求值器
3. **Pass 分优先级执行**：P0（动转静）→ P1（常量）→ P2（简化）→ P3（融合）
4. **幂等且可迭代**：每次变更后重新触发 ShapeInference，逐步收敛
5. **严格验证**：ONNX checker 通过 ≠ 静态化，需单独验证
6. **区分两种动态**：Shape-Dependent 可静态化，Data-Dependent 不可
