# ONNX Static Optimizer

把带动态维度的 ONNX 模型转换为 **NPU 编译器友好的静态模型**。

```
输入: Dynamic ONNX   images [N, 3, H, W]
      用户指定:        N=1, H=640, W=640
输出: Static ONNX    images [1, 3, 640, 640]，且图里不再有任何"运行时算 shape"的节点
```

RKNN / 地平线 / 算能等 NPU 编译器要求每个张量的尺寸在**编译期**完全确定。
PyTorch 导出动态模型时会把"自己算一遍尺寸"的逻辑写进图里（`Shape` → `Gather`
→ `Concat` → `Reshape`），编译器解不开，于是整个模型编译失败。

本工具在 Python 侧把这些 shape 子图**提前算掉、烧成常量**，让编译器能正常排布内存。

---

## 安装

```bash
pip install onnx numpy onnxruntime
```

`onnxruntime` 是可选的：没装时 Pass 5（ORT 子图折叠）会自动跳过，
自实现的算子折叠仍然可用。

## 快速开始

### Python API

```python
from onnx_static_optimizer import optimize

result = optimize("yolov8n.onnx", {"images": [1, 3, 640, 640]})
print(result.report())

if result.is_static:
    result.save("yolov8n_static.onnx")
```

### 命令行

```bash
# 直接给完整 shape
python -m onnx_static_optimizer.cli yolov8n.onnx \
    -i images=1,3,640,640 -o yolov8n_static.onnx

# 符号名映射形式（对应导出时的 dim_param）
python -m onnx_static_optimizer.cli model.onnx -i images=N:1,H:640,W:640

# 只看报告不写文件
python -m onnx_static_optimizer.cli model.onnx -i images=1,3,640,640 --dry-run
```

退出码：`0` 全部通过；`3` 仍有动态项；`4` ONNX checker 失败；`5` 数值不等价。

### 输出报告

```
Inputs: ['images']
  images: [N,3,H,W] -> [1,3,8,8]
Iterations: 4 (converged)

Graph Statistics
----------------
Nodes:        28 -> 4  (-24)
Initializers: 9 -> 7  (-2)

Dynamic Shape
-------------
Dynamic inputs:        0  ok
Dynamic outputs:       0  ok
Dynamic ValueInfo:     0  ok
Shape nodes:           0  ok
Size nodes:            0  ok
Dynamic Reshape:       0  ok
Dynamic Slice:         0  ok
Dynamic Expand:        0  ok

Validation
----------
ONNX checker:        PASS
Static shape:        PASS
Numerical equiv:     PASS  (PASS（最大绝对误差 0.000e+00）)
```

---

## 流水线

按 `pass_implementation.md` 的顺序执行，并循环至收敛（`max_iterations=10`）：

**第一轮 · 动转静**

| #  | Pass                       | 作用                                      |
| -- | -------------------------- | ----------------------------------------- |
| 01 | Input Shape Specialization | 把用户指定的具体形状写入`graph.input`   |
| 02 | Shape Inference            | 调用 ONNX 官方推导传播已知 shape          |
| 03 | Shape Tensor Evaluation    | 求值 shape 子图，替换为`Constant`       |
| 04 | Constant Propagation       | 折叠所有输入都是常量的节点                |
| 05 | Constant Folding           | ORT 子图执行兜底（Pass 4 覆盖不到的算子） |
| 06 | Shape Elimination          | `Shape` / `Size` → 常量              |
| 07 | Reshape Elimination        | 静态化动态`Reshape`，删除 no-op         |
| 08 | Slice Elimination          | 静态化动态`Slice`，删除全量切片         |
| 09 | Expand Elimination         | 静态化动态`Expand`，删除 no-op          |

**第二轮 · 图简化**

| #  | Pass                     | 作用                     |
| -- | ------------------------ | ------------------------ |
| 11 | Identity Elimination     | 删除纯拷贝               |
| 12 | Reshape Simplification   | 合并`Reshape` 链       |
| 13 | Transpose Simplification | 消除 / 合并`Transpose` |
| 14 | Cast Simplification      | 消除 / 合并`Cast`      |

**第三轮 · 清理**

| #  | Pass                     | 作用                       |
| -- | ------------------------ | -------------------------- |
| 10 | DCE                      | 反向可达性分析，删除死代码 |
| 15 | Initializer Cleanup      | 删除未引用的 initializer   |
| 16 | Unused ValueInfo Cleanup | 删除悬空的 value_info 条目 |

> **为什么要循环**：单个 Pass 常常会解锁上游的新机会。例如 `Pass 6` 消掉
> `Shape(X)` 后 `X` 的链路变成死代码，需要 `Pass 10` 清掉；`Pass 3` 折叠
> shape 子图产生新 `Constant`，需要 `Pass 4` 再传播一轮。

---

## 核心：Shape Tensor Evaluator

`shape/shape_evaluator.py` 是整个系统的核心，不是一个 Pass 而是一个被复用的**求值引擎**。
给定一个 `value_name`，如果它的值能在编译期确定就返回 `np.ndarray`，否则返回 `None`。

三个关键设计：

1. **`None` 是一等公民。** 它表示"已判定：不可求值"，而不是"还没算"。
   所以缓存判断用 `name in memo` 而不是 `memo.get(name)` —— 否则同一个不可求值的
   value 会被反复递归，图上退化成指数级重复计算。
2. **`Shape` / `Size` 必须短路。** 它们读的是输入的**类型**（shape）而不是**值**。
   如果走通用路径，由于数据张量本身求不出值，`Shape` 会被 `any(v is None)` 拦掉，
   永远返回 `None` —— 而 `Shape` 恰恰是整条 shape 链的起点。
3. **求值失败一律降级为 `None`，绝不猜。** 两个方向的错误代价极不对称：
   假阴性只是少优化一点，假阳性会让模型**静默算错**，在 NPU 上极难排查。

算子按三阶段落地：纯 shape 计算 → 算术/切片 → 高级算子，见 `HANDLERS` 注册表。

---

## 测试

```bash
python tests/test_pipeline.py
```

10 个测试，覆盖：完整流水线、幂等性、已静态模型不被破坏、data-dependent 报告、
`Shape(start/end)`、输入 shape 冲突报错、以及求值器的算子语义 / 环保护单测。

单个 Pass 的调试入口（按编号或名字挑一个 Pass，看它到底改了什么）：

```bash
python tests/test_pass.py --list             # 列出全部 Pass
python tests/test_pass.py 06                 # 跑 Pass 6（默认先跑一遍前置 Pass）
python tests/test_pass.py 11 --repeat 2      # 跑两遍，验证幂等
python tests/test_pass.py 06 --model m.onnx -i images=1,3,640,640
```

---

## 已知边界

**`Rank` 不是 ONNX 算子。** 用 `onnx.defs` 核实过，标准域里只有 `Shape`、
`Size`、`ConstantOfShape` 三个"读元信息"的算子，没有 `Rank`。
设计文档里多处提到的 `Rank` 是笔误 —— 取 rank 的惯用写法是 `Size(Shape(x))`。
代码里保留了 `Rank` 的处理分支（防御性，成本为零），但真实模型里见不到。

**两种动态要分清。** 不是所有"动态"都能救：

- **Shape-Dependent**（可静态化）：`Shape` → `Gather` → `Concat` → `Reshape`
- **Data-Dependent**（无法静态化）：`NonZero`、`NMS`、动态 k 的 `TopK`、
  动态 condition 的 `Where` —— 输出个数取决于实际数据

后者会在报告的 `Data-Dependent` 段落里明确列出，这是数据本身的性质，
不是工具没做好。

**输出名可能被改写。** bypass 一个产出 graph output 的节点时，默认把 output
改名成上游名字（避免留下无用的 Identity）。若下游按名字取输出，设置
`OptimizerConfig(preserve_graph_output_names=True)` 来保持名字不变。

**往返 Cast 默认不删。** `FP32→FP16→FP32` 虽然类型回到原点，但中间已经损失精度，
删除会改变数值结果。由 `allow_fp16_roundtrip_removal` 控制，默认关闭。

---

## 项目结构

```
onnx_static_optimizer/
├── optimizer.py              # 主入口，流水线组装
├── config.py                 # 配置（所有激进优化都有开关）
├── cli.py                    # 命令行
├── graph/
│   ├── graph_view.py         # 图视图 + 索引 + get_shape/get_const
│   ├── graph_editor.py       # bypass / 替换 / 拓扑排序
│   ├── node_utils.py         # 属性读写
│   └── tensor_utils.py       # TensorProto <-> numpy
├── shape/
│   ├── shape_evaluator.py    # ⭐ 求值引擎
│   ├── shape_specializer.py  # Pass 1
│   └── shape_inference.py    # Pass 2
├── passes/
│   ├── base.py               # Pass / PassManager
│   ├── shape/                # Pass 3, 6, 7, 8, 9
│   ├── constant/             # Pass 4, 5
│   ├── structural/           # Pass 11-14
│   └── cleanup/              # Pass 10, 15, 16
├── analysis/                 # 图统计
└── validation/               # ONNX checker / 静态性 / 数值等价
```
