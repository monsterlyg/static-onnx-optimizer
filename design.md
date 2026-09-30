输入一个 ONNX 动态模型 → 在 Python 中尽可能把它“定形（shape specialization）”→ 消除动态 Shape 计算 → 做常量传播/折叠和死代码删除 → 尽可能把动态算子改写成静态算子 → 最终输出适合 RKNN、算能、地平线等 NPU 编译器的静态 ONNX

那么思路和单纯的“ONNX Runtime 性能优化”其实不完全一样。

ONNX Runtime 的官方 Graph Optimization 主要是为了推理性能，包括 Constant Folding、冗余节点消除、Conv/BN 等融合以及 Layout Optimization；而你真正需要的是更偏向 Static Shape Specialization / Graph Canonicalization / Shape Computation Elimination 的一套 pipeline。

一、我建议你把优化 Pass 分成 7 大类

你现在想到的：

Shape Inference
Constant Folding
DCE

是非常核心的，但还不够。

我会把完整体系设计成：

                    Dynamic ONNX
                         │
                         ▼
              ┌─────────────────────┐
              │ 1. Shape Specialize │
              │ 2. Shape Inference  │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ 3. Constant Prop.  │
              │ 4. Constant Folding│
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ 5. Shape Graph      │
              │    Simplification   │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ 6. Structural        │
              │    Simplification   │
              │    + DCE             │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ 7. Operator Fusion  │
              │    / Canonicalize   │
              └─────────┬───────────┘
                        │
                        ▼
                  Static ONNX

其中真正对你这个目标最重要的其实是 1、3、4、5、6。

二、第一类：Shape Specialization —— 最重要

这是你和普通 ONNX Runtime 优化最大的区别。

假设原模型：

input
shape = [N, 3, H, W]

你实际上部署时明确知道：

N = 1
H = 640
W = 640

那么第一步不是简单：

onnx.shape_inference.infer_shapes()

而应该主动把：

[N, 3, H, W]

变成：

[1, 3, 640, 640]

也就是：

把动态输入实例化成具体 shape。

这一步可以称为：

Static Shape Specialization

例如：

input_shape = {
    "images": [1, 3, 640, 640]
}

然后：

Shape(input)
↓
[1, 3, 640, 640]

进一步：

Shape
Gather
Unsqueeze
Concat
Mul
Div
Sub
...

都可能被计算掉。

三、Shape Inference

这一步你已经知道。

import onnx

model = onnx.load("model.onnx")

model = onnx.shape_inference.infer_shapes(
    model,
    check_type=True,
    strict_mode=True,
    data_prop=True
)

ONNX 官方的 shape inference 可以传播静态 shape 和 symbolic shape，但它并不是万能的；尤其是动态 Reshape、依赖运行时数据的 shape computation 等情况，可能无法继续推导。

所以：

Shape inference ≠ Dynamic → Static

这一点非常重要。

它更多是：

已有信息
   ↓
传播 shape 信息

而不是：

动态模型
   ↓
自动猜出你的部署 shape
四、第二类：Constant Propagation

这是我非常建议你单独考虑的。

例如：

Constant
   ↓
Shape
   ↓
Gather
   ↓
Unsqueeze
   ↓
Concat

实际上这些全部可以在编译期计算。

例如：

Shape(input)
        ↓
[1,3,640,640]
        ↓
Gather(axis=2)
        ↓
640

最终：

640

而不是让 NPU 运行时再算。

五、第三类：Constant Folding

你已经提到了。

但这里要注意：

Constant Folding 不只是权重计算

很多人理解成：

Conv weight
+
bias
→
某个常量

实际上对于动态模型，Constant Folding 最重要的用途之一是：

折叠 Shape Tensor 计算。

例如：

Shape
 ↓
Gather
 ↓
Cast
 ↓
Mul
 ↓
Concat
 ↓
Reshape

如果输入 shape 已经静态化：

Shape
 ↓
[1,3,640,640]

那么整条链可能直接变成：

Constant([1,3,640,640])

这对 RKNN / Horizon / 算能这类编译器特别重要。

六、第四类：Shape Graph Simplification

这个我认为是你这个项目里最值得自己实现的一类 Pass。

很多动态 ONNX 的真正问题不是模型本身动态，而是：

图里面存在大量“计算 shape 的图”。

比如：

Shape
 ↓
Gather
 ↓
Slice
 ↓
Concat
 ↓
Unsqueeze
 ↓
Cast
 ↓
Mul
 ↓
Div
 ↓
Reshape

最终只是为了得到：

[1, 8400, 80]

这种东西如果部署平台支持不好，就非常麻烦。

你可以专门做：

Shape-subgraph folding

目标：

Shape
Gather
Unsqueeze
Concat
Cast
...

↓

Constant
七、第五类：Dynamic Reshape → Static Reshape

这个特别重要。

比如：

Reshape
  │
  └── shape = [0, -1, 80]

如果经过 shape specialization 后已经知道：

input = [1, 8400, 80]

那么：

Reshape(shape=[0,-1,80])

可以直接变成：

Reshape(shape=[1,8400,80])

进一步，如果输入输出 shape 本来就一样：

Reshape

甚至可以删除。

八、第六类：Eliminate Shape / Size / Rank

这是动态模型里非常重要的一组。

例如：

Shape
Size
Rank

如果输入已经静态：

Shape(X)
→ Constant([1,3,640,640])
Size(X)
→ Constant(1228800)
Rank(X)
→ Constant(4)

这些都应该尽量消掉。

因此可以专门做：

EliminateShape
EliminateSize
EliminateRank

实际上这类 Pass 对动态模型的“动转静”价值非常高。

九、第七类：Eliminate Dynamic Slice

例如：

Shape
 ↓
Gather
 ↓
Slice

动态模型经常有：

starts
ends
axes
steps

这些参数实际上可能全部是常量。

那么：

Dynamic Slice

可以变成：

Static Slice

甚至进一步：

Slice(X, 0, 640)

如果恰好覆盖整个 tensor：

Slice(X, 0, size(X))

那么：

Slice

直接删除。

ONNX Runtime 自身也包含 Slice/Identity/Unsqueeze/Dropout 等冗余节点消除。

十、Unsqueeze / Squeeze 消除

这是非常常见的。

例如：

Unsqueeze(axis=1)
Squeeze(axis=1)

直接：

Identity

进一步：

Identity

删除。

或者：

Unsqueeze
 ↓
Concat
 ↓
Reshape

如果整个 shape 已经静态，可以一起折叠。

十一、Transpose 简化

这个也非常值得做。

例如：

Transpose
 ↓
Transpose

如果：

perm1 = inverse(perm2)

那么：

Transpose
Transpose

直接消掉。

例如：

Transpose([0,2,3,1])
Transpose([0,3,1,2])

↓

Identity

另外：

Transpose
 ↓
Transpose
 ↓
Transpose

也可以合并成一个。

最终：

Transpose

如果 perm 是 identity：

[0,1,2,3]

直接删除。

十二、Reshape 简化

建议单独做一个 ReshapeSimplifier。

可以处理：

1. Reshape → Reshape
Reshape(A)
 ↓
Reshape(B)

如果中间没有实际意义：

Reshape(B)
2. Reshape → Identity

输入输出 shape 一样：

Reshape

删除。

3. Reshape → Flatten

某些情况下：

Flatten

可以转换成静态：

Reshape
4. Flatten → Reshape

可以进一步合并。

十三、Concat / Split 静态化

比如：

Split

的 split 参数动态：

Shape
 ↓
Gather
 ↓
...
 ↓
Split

如果 shape 已知：

Split
split = [320, 320]

就可以把它固定。

类似：

Concat
Split
Slice
Gather
GatherElements

都值得做 staticization。

十四、Gather / GatherND 简化

例如：

Gather(
    Constant([1,3,640,640]),
    Constant(2)
)

直接：

Constant(640)

这是 Constant Folding 的一个重要组成部分。

但是如果你自己实现 Pass，我建议不要完全依赖 ORT。

可以实现一个：

Tensor Constant Evaluator

支持：

Add
Sub
Mul
Div
Mod
Concat
Gather
Unsqueeze
Squeeze
Shape
Size
Reshape
Slice
Cast
Expand
Range

这样很多 dynamic shape graph 都可以直接折叠。

十五、Expand / Tile 静态化

比如：

Expand(X, dynamic_shape)

经过 specialization 后：

Expand(X, [1,80,6400])

然后判断：

是否实际需要 Expand
是否 broadcast 本身已经满足
是否可以删除

类似：

Tile
Expand
Broadcast

都值得检查。

十六、Broadcast Simplification

例如：

Add

两边：

[1,80,1]
[1,80,6400]

如果其中一个实际上只是 scalar/bias：

Constant

那么可以进一步触发：

Add + Constant

相关融合。

ORT 本身就有不少这种语义保持型融合，例如 Conv+Add、Conv+Mul、Conv+BN。

十七、Cast Elimination

这个对 NPU 非常重要。

比如：

Cast FP32 → FP16
 ↓
Cast FP16 → FP32

如果中间没有必要：

删除

或者：

Cast
 ↓
Cast

可以合并。

还可以处理：

Constant FP32

如果后续强制要求：

FP16

直接把 initializer 转成 FP16，而不是：

Constant FP32
 ↓
Cast
 ↓
Operator
十八、Constant Deduplication

这个经常被忽略。

例如模型中：

Constant A = [1,3,640,640]
Constant B = [1,3,640,640]
Constant C = [1,3,640,640]

可以合并成：

Constant A
   ↑
 ┌─┼─┐
 B C ...

更准确地说，是：

initializer deduplication

对于大模型可以明显降低模型体积。

十九、Dead Code Elimination

你已经提到。

但我建议你实现成至少三种：

1. Dead Node Elimination

没有 downstream consumer：

Node
 ↓
nothing

删除。

2. Dead Initializer Elimination

：

initializer

没有任何节点使用：

删除
3. Unused ValueInfo Elimination

：

value_info

但已经没有对应 tensor：

删除
二十、Identity Elimination

例如：

Identity

如果没有特殊语义：

A → Identity → B

直接：

A → B

非常简单，但是一定值得做。

二十一、No-op Arithmetic Elimination

这一类也很有用。

例如：

Add(X, 0)

→

X
Sub(X, 0)

→

X
Mul(X, 1)

→

X
Div(X, 1)

→

X

以及：

Concat(X)

→

X

等等。

但这里要注意 NaN / Inf / dtype / broadcasting / floating-point semantics，不能简单粗暴地删。

二十二、Operator Fusion

这一类属于：

性能优化

而不是严格意义上的：

动转静

但是为了 NPU 部署，非常值得做。

例如：

Conv
 +
BatchNorm

↓

Conv

把 BN 参数融合到 Conv：

W'
b'

同理：

Conv + Add
Conv + Mul
MatMul + Add

等。

ORT 的 Basic/Extended Graph Optimization 中已经包含大量这样的融合。

二十三、特别推荐：BN Folding

如果你的目标包括：

RKNN
Horizon
Sophgo
Ascend
NPU

我会把：

Conv + BN

融合单独列为一个 Pass。

公式：

$$ W' = W \frac{\gamma}{\sqrt{\sigma^2+\epsilon}} $$ $$ b' = \beta + (b-\mu) \frac{\gamma}{\sqrt{\sigma^2+\epsilon}} $$

最终：

Conv
 ↓
BN

变成：

Conv'
二十四、还有一个非常重要的：If / Loop / Scan 静态化

这个可能比你想象的重要。

例如：

If

条件是：

Constant(True)

那么：

If
├── then_branch
└── else_branch

直接变：

then_branch

同样：

If(Constant(False))

直接：

else_branch

如果：

Loop

循环次数是常量：

Loop(trip_count=3)

理论上可以展开：

Loop

↓

Body
Body
Body

这叫：

Loop unrolling

对于 NPU 编译器，有时候非常有价值。

二十五、NonZero / Where / Shape-dependent control flow

这类是动转静最难的部分。

比如：

NonZero

输出 shape 本身依赖运行时数据。

这种：

Data-dependent shape

和：

Shape-dependent shape

一定要区分。

Shape-dependent

例如：

Shape(X)

只要 X shape 已知：

可以静态化
Data-dependent

例如：

NonZero(X)

它到底有多少个非零元素？

必须运行时才能知道。

这种：

Dynamic output cardinality

通常不能简单通过 shape inference 解决。

这也是 ONNX 官方 Shape Inference 本身的边界之一。

二十六、因此我建议你建立一个“动态算子审计”

最终输出：

Static Shape Report

比如：

Input:
    images        [1,3,640,640]

Dynamic tensors:
    0

Dynamic dimensions:
    0

Dynamic operators:
    0

Shape operators:
    Shape       0
    Size        0
    Rank        0

Dynamic Reshape:
    0

Dynamic Slice:
    0

Dynamic Expand:
    0

Control Flow:
    If          0
    Loop        0
    Scan        0

Unused nodes:
    0

Unused initializers:
    0

这个东西对于你后面接 RKNN / 地平线特别有价值。

二十七、我反而建议你不要一上来把所有 ORT Optimization Level 全打开

因为你现在的目标不是：

ONNX Runtime CPU/GPU 跑得最快

而是：

让第三方 NPU Compiler 最容易吃进去。

这两者并不完全一致。

例如 ORT：

Layout Optimization

可能为了 ORT kernel 做：

NCHW → NHWC

但你的 RKNN/Horizon 编译器可能更喜欢另外一种布局。

所以我建议：

Phase A
Dynamic → Static

然后：

Phase B
Canonicalization

最后才：

Phase C
Hardware-oriented fusion

不要反过来。

二十八、我会给你的 Python 优化器设计成这样

如果你准备自己做一个库，我建议 API 类似：

optimizer = ONNXStaticOptimizer(
    input_shapes={
        "images": [1, 3, 640, 640]
    }
)

optimizer.load("model.onnx")

optimizer.run([
    "shape_specialization",
    "shape_inference",
    "constant_propagation",
    "constant_folding",
    "shape_simplification",
    "reshape_simplification",
    "slice_simplification",
    "transpose_simplification",
    "cast_simplification",
    "broadcast_simplification",
    "control_flow_simplification",
    "identity_elimination",
    "dead_code_elimination",
])

optimizer.validate()

optimizer.save("model_static.onnx")

然后再提供：

optimizer.optimize_for("rknn")

或者：

optimizer.optimize_for("horizon")

这样以后就能加入：

rknn_passes
horizon_passes
axera_passes
sophgo_passes
二十九、推荐你最终形成这样的 Pass Pipeline

如果是我来设计，我会优先做这个顺序：

                    ONNX
                     │
                     ▼
        ┌────────────────────────┐
        │ 0. Model Validation    │
        │    checker / opset     │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 1. Input Shape         │
        │    Specialization      │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 2. Shape Inference     │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 3. Constant Propagation│
        │    Constant Folding    │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 4. Shape Graph         │
        │    Simplification      │
        │                        │
        │ Shape / Size / Rank    │
        │ Gather / Slice         │
        │ Concat / Cast          │
        │ Reshape                │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 5. Structural          │
        │    Simplification      │
        │                        │
        │ Identity               │
        │ Reshape                │
        │ Transpose              │
        │ Squeeze/Unsqueeze      │
        │ Cast                   │
        │ Slice                  │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 6. Arithmetic          │
        │    Simplification      │
        │                        │
        │ Add 0                  │
        │ Mul 1                  │
        │ Div 1                  │
        │ etc.                   │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 7. Control Flow        │
        │    Simplification      │
        │                        │
        │ If / Loop / Scan       │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 8. DCE                 │
        │ 9. Initializer Cleanup │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 10. Operator Fusion    │
        │     Conv+BN             │
        │     Conv+Add            │
        │     etc.                │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 11. Final Shape        │
        │     Inference          │
        └───────────┬────────────┘
                    ↓
        ┌────────────────────────┐
        │ 12. Static Validation  │
        └────────────────────────┘
                    ↓
              static.onnx

这个顺序比单纯 onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL 更适合你的目标。

因为 ORT 的优化重点是推理性能和通用图优化，而 ONNX 的 shape annotation 本身就是为了帮助静态内存规划、内存复用等运行时优化；你的目标实际上是在更早阶段把这些静态信息“钉死”。

三十、最关键的一点：你真正应该做的是“Shape Tensor Evaluator”

如果你准备自己实现，我最建议投入精力的并不是再写一个 DCE。

而是：

做一个可靠的 Constant/Shape Tensor Evaluator。

专门处理：

Shape
Size
Rank
Gather
GatherND
Slice
Concat
Split
Unsqueeze
Squeeze
Reshape
Cast
Add
Sub
Mul
Div
Mod
Floor
Ceil
Min
Max
Where
Expand
Range

然后建立：

Dynamic Shape Graph
        ↓
Shape Tensor Evaluation
        ↓
Concrete Shape
        ↓
Constant
        ↓
Constant Folding
        ↓
DCE

这会直接解决大量：

PyTorch → ONNX

导出后产生的动态 shape 垃圾图。

而且 ONNX Runtime 自己也明确指出，动态 axis 会阻碍部分优化，因为 shape inference 无法获得精确 shape。

如果你现在准备真正开始写这个 Python 库

我建议下一步不要直接写代码，而是先把 Pass 列表 + 每个 Pass 的触发条件 + 输入/输出图模式 + Pass 顺序 + 哪些能直接调用 ONNX/ORT、哪些必须自己实现 定下来。

我可以直接给你整理成一份类似 “ONNX 动转静优化器技术设计文档”，包括大约 40～60 个具体 Pass，并标明：

onnx 原生 API 能不能做
onnxruntime 能不能做
是否需要自己遍历 GraphProto
Pass 的前置条件
Pass 的后置条件
是否会改变模型语义
对 RKNN / 地平线 / 算能的价值
哪些 Pass 可以循环执行直到收敛
最终如何自动判断 “这个 ONNX 已经真正静态化”

这个会比单纯列优化名更适合你后面直接实现。

ONNX 动转静与 NPU 部署优化器设计文档

版本：v1.0
语言：Python 3.x
核心依赖：ONNX、ONNX Runtime、NumPy
目标平台：RKNN、地平线、算能等嵌入式 NPU

1. 项目目标
   1.1 核心目标

构建一个纯 Python 的 ONNX Graph Optimizer：

Dynamic ONNX
     │
     ▼
Input Shape Specialization
     │
     ▼
Shape Inference
     │
     ▼
Shape Tensor Evaluation
     │
     ▼
Constant Propagation / Folding
     │
     ▼
Dynamic Operator Elimination
     │
     ▼
Graph Simplification
     │
     ▼
NPU-friendly Canonicalization
     │
     ▼
Static ONNX

最终希望做到：

输入：
[N, 3, H, W]

指定：
N=1
H=640
W=640

输出：
[1,3,640,640]

并且尽可能消除：

Shape
Size
Rank
Gather
Concat
Unsqueeze
Squeeze
Cast
Reshape
Expand
Slice

等由动态 shape 引入的辅助计算。

2. 设计原则

整个项目建议遵循 6 个原则。

原则 1：先静态化，再做性能优化

不要一开始就：

Conv-BN Fusion
Transpose Fusion
Layout Optimization

优先：

Dynamic → Static

因为 NPU 部署首先要解决的是编译器能不能接受这个图。

原则 2：Shape Graph 和 Data Graph 分开处理

这是整个项目最重要的设计。

ONNX 图实际上可以理解为：

                 ONNX Graph
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
     Data Graph             Shape Graph
          │                     │
       Conv                  Shape
       MatMul                Gather
       Relu                  Concat
       Add                   Cast
       ...                   Reshape

你的优化器应该重点识别：

哪些节点是在计算 Tensor 数据，哪些节点只是在计算 Tensor 的 shape。

然后优先把 Shape Graph 求值。

原则 3：能证明安全才删除

例如：

Mul(X, 1)

看起来可以删除，但需要考虑：

dtype
NaN
Inf
broadcasting
FP16
INT8

所以每一个 Simplification Pass 都应该有：

Pattern
+
Precondition
+
Rewrite

而不是简单：

if node.op_type == "Mul":
    delete(node)
原则 4：Pass 必须幂等

执行：

Pass()
Pass()
Pass()

最终应该：

Graph1 == Graph2

例如：

Transpose
Transpose

第一次消掉。

第二次不能继续修改。

原则 5：Pass 可以重复执行

推荐：

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

直到：

graph_hash_before == graph_hash_after

即达到 fixed point。

原则 6：最终必须做 Static Validation

不能因为：

onnx.checker.check_model()

通过了，就认为已经静态化。

需要自己检查：

dynamic input
dynamic output
dynamic ValueInfo
dynamic Reshape
dynamic Slice
dynamic Expand
dynamic control flow
dynamic shape tensor
3. 总体架构

推荐项目结构：

onnx_static_optimizer/
│
├── __init__.py
│
├── optimizer.py
│
├── config.py
│
├── graph/
│   ├── graph_view.py
│   ├── graph_editor.py
│   ├── node_utils.py
│   ├── tensor_utils.py
│   └── initializer_utils.py
│
├── shape/
│   ├── shape_specializer.py
│   ├── shape_inference.py
│   ├── shape_evaluator.py
│   ├── shape_analyzer.py
│   └── symbolic_shape.py
│
├── passes/
│   ├── base.py
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
│   ├── dynamic_analyzer.py
│   ├── graph_statistics.py
│   └── dependency.py
│
├── validation/
│   ├── checker.py
│   ├── static_validator.py
│   └── numerical_validator.py
│
└── cli.py
4. 核心数据结构

建议不要直接大量操作：

model.graph.node

而是封装一个：

GraphView

维护：

class GraphView:

    nodes
    inputs
    outputs
    initializers
    value_info

    producer_map
    consumer_map

    initializer_map
    value_info_map

这样可以快速实现：

producer = graph.producer("x")
consumers = graph.consumers("x")

而不是每次：

for node in graph.node:
    ...
5. Pass 基类

建议所有 Pass 统一接口：

class Pass:

    name: str

    def run(self, graph):
        ...

    def changed(self):
        ...

    def statistics(self):
        ...

例如：

class IdentityElimination(Pass):

    name = "identity_elimination"

    def run(self, graph):

        changed = False

        for node in graph.nodes:

            if node.op_type != "Identity":
                continue

            ...

        return changed

最终：

result = optimizer.run()

能够得到：

Pass Statistics
────────────────────────
shape_specialization      12 changes
shape_inference           0 changes
constant_folding          84 changes
reshape_simplification    17 changes
transpose_simplification   3 changes
DCE                       105 changes
6. Pass 优先级

我建议把 Pass 分成 5 个等级。

Level	类型	目的
P0	Shape	动转静核心
P1	Constant	编译期计算
P2	Structural	图清理
P3	Fusion	性能优化
P4	Hardware	NPU 特化
7. P0：Input Shape Specialization

这是第一优先级。

输入：

images: [N, 3, H, W]

用户：

{
    "images": [1, 3, 640, 640]
}

变成：

images: [1,3,640,640]

需要处理：

dim_value
dim_param
unknown dimension

例如：

def specialize_input_shape(
    model,
    input_shapes
):
    ...

必须支持：

[1, 3, 640, 640]

以及：

{"N": 1, "H": 640, "W": 640}
8. P0：Shape Inference

调用：

onnx.shape_inference.infer_shapes()

但建议封装：

ShapeInferencePass

并支持：

before
after
strict
data_prop
9. P0：Shape Tensor Evaluator

这是整个系统的核心。

建议：

class ShapeTensorEvaluator:
    def evaluate(self, value_name):
        ...

输入：

Shape(input)

输出：

np.array([1,3,640,640])

需要支持：

第一阶段
Constant
Shape
Size
Rank
Cast
Gather
Concat
Unsqueeze
Squeeze
Reshape
第二阶段
Slice
Add
Sub
Mul
Div
Mod
Min
Max
第三阶段
Range
Expand
Where
GatherND
Split
10. Shape Tensor 求值示例

原图：

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

如果：

X = [1,3,640,640]

Evaluator：

Shape(X)
→ [1,3,640,640]

Gather(..., 0)
→ 1

Unsqueeze
→ [1]

Concat
→ [1,...]

最后：

Constant([1,...])

整个 shape subgraph 删除。

11. P1：Constant Propagation

定义：

ConstantPropagationPass

负责：

Constant → Operator → Constant

例如：

Constant(3)
Constant(4)
   ↓
 Add

变成：

Constant(7)
12. P1：Constant Folding

这里建议不要自己实现所有 ONNX operator。

可以采用两级策略：

Level 1
    自己实现 shape operators

Level 2
    ONNX Runtime CPU execution

例如：

Constant
 ↓
Add
 ↓
Mul
 ↓
Reshape

可以尝试用 ORT 执行这个小子图。

但是：

不要把整个模型交给 ORT 跑一遍来做 Constant Folding。

因为你需要控制哪些节点可以折叠。

13. P0：Shape / Size / Rank Elimination

目标：

Shape(X)

如果 X 已知：

[1,3,640,640]

直接：

Constant([1,3,640,640])

同理：

Size(X)
→ Constant(1228800)
Rank(X)
→ Constant(4)
14. P0：Dynamic Reshape Elimination

例如：

Reshape(X, ShapeTensor)

如果：

ShapeTensor = [1,8400,80]

变成：

Reshape(X, [1,8400,80])

如果最终输入输出 shape 相同：

Reshape

删除。

15. P0：Dynamic Slice Elimination

处理：

starts
ends
axes
steps

全部静态化。

例如：

Slice(
    X,
    starts=[0],
    ends=[640],
    axes=[2]
)

如果整个 slice 覆盖 X：

Slice(X)

删除。

16. P0：Dynamic Expand Elimination

例如：

Expand(
    X,
    ShapeTensor
)

如果目标 shape 已知：

Expand(X, [1,80,6400])

继续判断：

X.shape == target.shape

如果相等：

Expand

删除。

17. P1：Gather Simplification

例如：

Gather(
    Constant([1,3,640,640]),
    Constant(2)
)

直接：

Constant(640)

这是 Shape Tensor Evaluator 的基础功能。

18. P1：Concat Simplification

例如：

Concat(
    Constant([1]),
    Constant([640]),
    Constant([640])
)

→

Constant([1,640,640])
19. P2：Reshape Simplification

处理：

Reshape → Reshape
Reshape → Identity
Flatten → Reshape
Reshape → Flatten

目标是减少：

Reshape
Flatten

数量。

20. P2：Transpose Simplification

处理：

Transpose(identity)

删除。

以及：

Transpose(A)
Transpose(inverse(A))

删除。

进一步：

Transpose(A)
Transpose(B)

合并成：

Transpose(B ∘ A)
21. P2：Squeeze / Unsqueeze

处理：

Unsqueeze → Squeeze

以及：

Squeeze → Unsqueeze

条件满足时消除。

22. P2：Cast Simplification

例如：

Cast(FP32→FP16)
Cast(FP16→FP32)

如果中间没有依赖：

删除

以及：

Cast
Cast

可以合并。

23. P2：Identity Elimination

最简单：

X → Identity → Y

改成：

X → Y

同时处理：

Identity

相关 initializer/value_info。

24. P2：Arithmetic Simplification

第一阶段支持：

Add(X, 0)
Sub(X, 0)
Mul(X, 1)
Div(X, 1)

以及：

Mul(X, 0)

但 Mul(X,0) 不建议第一版直接做，需要处理 NaN/Inf 语义。

25. P2：Broadcast Simplification

识别：

scalar + tensor
scalar * tensor

以及：

[1,1,1,C]
+
[1,H,W,C]

这类明确 broadcast。

目标不是一定删除，而是：

让 broadcast shape 明确。

26. P2：DCE

DCE 应该从：

graph.output

反向遍历：

output
 ↓
producer
 ↓
producer's inputs
 ↓
...

最终：

没有到达 output 的节点

删除。

这比：

if len(node.output) == 0:

更正确。

27. P2：Initializer Cleanup

删除：

unused initializer

例如：

W1
W2
W3

但：

W2

已经没有任何 consumer。

删除。

28. P2：ValueInfo Cleanup

最终只保留：

graph.input
graph.output
required value_info

避免留下大量：

dim_param

造成下游工具误判。

29. P3：Conv + BN Fusion

这是第一批硬件无关融合。

Conv
 ↓
BatchNormalization

↓

Conv'

注意必须满足：

Conv output
    ↓
BN

且 Conv 输出没有其他 consumer。

30. P3：MatMul + Add

典型：

MatMul
 ↓
Add(bias)

变成：

Gemm

但这里需要考虑目标 NPU 对 Gemm 的支持情况。

因此建议：

generic profile
rknn profile
horizon profile

分别决定是否融合。

31. P4：NPU Profile

最终建议不要把硬件规则写死。

设计：

TargetProfile

例如：

RKNNProfile
HorizonProfile
SophgoProfile
GenericProfile

包含：

supported_ops
preferred_layout
forbidden_ops
preferred_ops
dynamic_shape_policy
fusion_policy
dtype_policy

例如：

RKNNProfile.dynamic_shape_policy = "forbid"

最终 validator：

Dynamic dimension detected
→ ERROR
32. Pass Scheduler

建议设计成：

class PassManager:

    def add(self, pass_):
        ...

    def run(self, graph):
        ...

支持：

manager.run_until_converged()

例如：

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

直到：

no_change
33. 推荐实际 Pipeline

第一版我建议：

01 InputShapeSpecialization

02 ShapeInference

03 ShapeTensorEvaluation

04 ConstantPropagation

05 ConstantFolding

06 ShapeElimination
   ├─ Shape
   ├─ Size
   └─ Rank

07 ReshapeSimplification

08 SliceSimplification

09 GatherSimplification

10 ConcatSimplification

11 ExpandSimplification

12 TransposeSimplification

13 SqueezeSimplification

14 UnsqueezeSimplification

15 CastSimplification

16 IdentityElimination

17 ArithmeticSimplification

18 DCE

19 InitializerCleanup

20 ValueInfoCleanup

21 ShapeInference

22 ConstantFolding

23 DCE

24 ConvBNFusion

25 FinalShapeInference

26 StaticValidation

其中：

01～23

是你的核心动转静引擎。

24

开始进入性能优化。

34. Fixed Point

建议：

MAX_ITER = 10

for i in range(MAX_ITER):

    changed = False

    for p in passes:
        if p.run(graph):
            changed = True

    if not changed:
        break

最终记录：

Converged after 3 iterations

如果：

10 iterations

还没有收敛：

WARNING
35. Static Validator

这是整个项目非常重要的一部分。

定义：

StaticModelValidator

至少检查：

Input
所有输入是否有明确 dim_value
Output
所有输出是否有明确 dim_value
ValueInfo
是否还有 dim_param
Reshape
shape input 是否静态
Slice
starts/ends/axes/steps 是否静态
Expand
shape 是否静态
Shape
是否还有 Shape operator
Size
是否还有 Size operator
Rank
是否还有 Rank operator
36. 最终输出报告

例如：

==================================================
ONNX Static Optimization Report
===============================

Model:
    yolov8.onnx

Input:
    images
    Original: [N,3,H,W]
    Final:    [1,3,640,640]

---

Graph Statistics
----------------

Nodes:
    Before:  412
    After:   286
    Removed: 126

Initializers:
    Before:  238
    After:   214

---

Dynamic Shape
-------------

Dynamic inputs:       0
Dynamic outputs:      0
Dynamic ValueInfo:    0

Shape nodes:          0
Size nodes:           0
Rank nodes:           0

Dynamic Reshape:      0
Dynamic Slice:        0
Dynamic Expand:       0

---

Simplification
--------------

Constant folded:      94
Identity removed:     17
Reshape removed:      11
Transpose removed:     4
Cast removed:          8
DCE nodes:            73

---

Validation
----------

ONNX checker:         PASS
Static shape:         PASS
ORT inference:        PASS
Numerical equivalence: PASS
===========================

这个报告对于你以后定位：

“为什么 RKNN 转换失败？”

会非常有价值。

37. 数值等价验证

不能只检查：

onnx.checker.check_model()

还需要：

Original ONNX
      │
      ├── ORT
      │
      ↓
reference output

Optimized ONNX
      │
      ├── ORT
      │
      ↓
optimized output

比较：

np.allclose(
    ref,
    optimized,
    rtol=1e-4,
    atol=1e-5
)

但是 FP16/INT8 情况下阈值应该由 profile 决定。

38. Regression Test

建议准备一个：

tests/models/

目录。

至少包含：

dynamic_reshape.onnx
dynamic_slice.onnx
dynamic_gather.onnx
dynamic_expand.onnx
shape_concat.onnx
shape_arithmetic.onnx
transpose_chain.onnx
reshape_chain.onnx
conv_bn.onnx
if_constant.onnx
loop_constant.onnx

每个模型保存：

input
expected_output
expected_static_shape
expected_node_count
39. 最值得优先实现的 15 个 Pass

如果你不想一开始就做 40～60 个，我建议按照这个优先级：

优先级	Pass
★★★★★	Input Shape Specialization
★★★★★	Shape Inference
★★★★★	Shape Tensor Evaluator
★★★★★	Constant Propagation
★★★★★	Constant Folding
★★★★★	Shape Elimination
★★★★★	Dynamic Reshape → Static
★★★★★	Dynamic Slice → Static
★★★★★	Dynamic Expand → Static
★★★★★	DCE
★★★★☆	Identity Elimination
★★★★☆	Reshape Simplification
★★★★☆	Transpose Simplification
★★★★☆	Cast Simplification
★★★★☆	Initializer Cleanup

先把这 15 个做好，已经能解决相当大一部分“动态 ONNX → NPU 静态 ONNX”的问题。

40. 一个非常关键的边界

你这个项目最好明确区分：

Shape Dynamic

和：

Data Dynamic

例如：

可以静态化
Shape(X)
Gather(Shape(X))
Reshape(X, Shape(X))
Slice(X, dynamic_shape)

因为这些本质上是：

shape-dependent
通常不能直接静态化
NonZero(X)
TopK(X, dynamic_k)
Where(condition)
NonMaxSuppression

如果输出数量取决于：

实际数据

那就是：

data-dependent dynamic

不能仅靠：

shape inference
+
constant folding

解决。

这应该在你的 Validator 中明确报告：

ERROR:
Data-dependent dynamic operator detected

Operator:
NonZero

Reason:
Output cardinality depends on runtime data.

而不是把它误判为普通动态 shape。

41. 最终架构可以抽象成三层

我非常建议最终做成：

┌───────────────────────────────────────┐
│             Hardware Layer            │
│                                       │
│ RKNN / Horizon / Sophgo / Generic     │
└───────────────────▲───────────────────┘
                    │
┌───────────────────┴───────────────────┐
│          Graph Optimization Layer     │
│                                       │
│ Fusion / Simplification / DCE         │
└───────────────────▲───────────────────┘
                    │
┌───────────────────┴───────────────────┐
│             Staticization Layer       │
│                                       │
│ Shape Specialization                  │
│ Shape Evaluation                      │
│ Constant Folding                      │
│ Dynamic → Static                      │
└───────────────────────────────────────┘

其中真正属于你这个项目核心创新/核心价值的是：

Staticization Layer，而不是普通的 Graph Optimization Layer。

42. 推荐的 MVP

如果现在马上开始写，我建议第一版只实现：

InputShapeSpecialization
        ↓
ShapeInference
        ↓
ShapeTensorEvaluator
        ↓
ConstantPropagation
        ↓
ConstantFolding
        ↓
Shape/Size/Rank Elimination
        ↓
DynamicReshapeElimination
        ↓
DynamicSliceElimination
        ↓
DynamicExpandElimination
        ↓
IdentityElimination
        ↓
ReshapeSimplification
        ↓
TransposeSimplification
        ↓
DCE
        ↓
InitializerCleanup
        ↓
FinalShapeInference
        ↓
StaticValidator

不要第一版就做复杂的 Conv/BN、Layout、Quantization、Loop Unroll。

先把这个闭环跑通：

                  ┌───────────────┐
Dynamic ONNX ────►│ Staticizer    │
                  └───────┬───────┘
                          │
                          ▼
                   Static ONNX
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
            ORT                    NPU Compiler
        numerical test        RKNN/Horizon/Sophgo

等这个基础框架稳定后，再把 ORT 的优化作为 P3/P4 后端优化接进去，而不要把 ORT ORT_ENABLE_ALL 当成整个“动转静”方案。