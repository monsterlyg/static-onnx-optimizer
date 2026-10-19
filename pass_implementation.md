# ONNX 动转静优化器 —— 第一版 15 个 Pass 实现思路

> 本文档聚焦「怎么写」，为每个 Pass 给出：目标、图模式、前置条件、实现步骤、伪代码与注意事项。
> 配套阅读：《design_summary.md》（整体设计）。

---

## 零、公共基础设施

在写任何 Pass 之前，先搭好三块地基。**这三个是 15 个 Pass 的共同依赖，不先做会反复返工。**

### 0.1 GraphView —— 图视图

ONNX 的 `graph.node` 是一个扁平列表，直接遍历做「找生产者 / 找消费者」是 O(N²)。所有 Pass 都需要这两类查询，所以先封装：

```python
class GraphView:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        self._rebuild()

    def _rebuild(self):
        # 一次 O(N) 扫描建立索引
        self.producer_map = {}   # output_name -> node
        self.consumer_map = {}   # input_name  -> [node, ...]
        self.initializer_map = {}  # name -> TensorProto
        self.value_info_map = {}   # name -> ValueInfoProto
        for init in self.graph.initializer:
            self.initializer_map[init.name] = init
        for node in self.graph.node:
            for out in node.output:
                if out:                       # 允许空输出名（可选输出）
                    self.producer_map[out] = node
            for inp in node.input:
                if inp:
                    self.consumer_map.setdefault(inp, []).append(node)

    # 查询
    def producer(self, name):  return self.producer_map.get(name)
    def consumers(self, name): return self.consumer_map.get(name, [])
    def is_initializer(self, name): return name in self.initializer_map

    # 求值器接口（见 0.3）
    def get_const(self, name) -> "np.ndarray | None": ...

    def get_shape(self, name) -> "list[int] | None": ...
```

**关键约定**：

- 任何 Pass 修改图后，必须调用 `graph._rebuild()`（或提供 `invalidate()`）重建索引。
- Pass 之间不共享 `GraphView` 的缓存，保证幂等。

### 0.2 Pass 基类与 PassManager

```python
class Pass:
    name = "base"
    priority = 0            # 便于排序，见下
    def run(self, graph: GraphView) -> bool:
        """返回 True 表示图发生了变化。必须幂等。"""
        raise NotImplementedError
    def stats(self) -> dict:
        return {"name": self.name, "changes": self._changes}

class PassManager:
    def __init__(self, passes):
        self.passes = passes
    def run_once(self, graph) -> bool:
        changed = False
        for p in self.passes:
            if p.run(graph):
                changed = True
                graph._rebuild()      # 每个 pass 后重建，避免脏索引
        return changed
    def run_until_converged(self, graph, max_iter=10) -> int:
        for i in range(max_iter):
            if not self.run_once(graph):
                return i
        return max_iter   # 达到上限仍未收敛 -> WARNING
```

**约定**：

- 每个 Pass 自己记录 `_changes` 计数，便于输出报告。
- `run()` 返回的 bool 是「是否变更」，不是「是否成功」。收敛判定依赖它。

### 0.3 Shape Tensor Evaluator —— 全局常量求值器（**核心，先做**）

**为什么先做它**：Pass 4/5/6/7/8/9 全部依赖它。它不是一个 Pass，而是一个被复用的**求值引擎**。

**定位**：给定一个 `value_name`，如果它的值可以在编译期确定，就返回一个 `np.ndarray`；否则返回 `None`。

**实现思路 —— 递归 + 记忆化**：

```python
class ShapeTensorEvaluator:
    def __init__(self, graph: GraphView):
        self.graph = graph
        self.memo = {}        # value_name -> np.ndarray | None

    def evaluate(self, name) -> "np.ndarray | None":
        if name in self.memo:
            return self.memo[name]

        # 1. initializer 直接就是常量
        if self.graph.is_initializer(name):
            return self._store(name, tensor_to_numpy(self.graph.initializer_map[name]))

        # 2. 该 value 是 graph input 且已知静态 shape -> 它本身不是常量，
        #    但 Shape(input) 可以由 get_shape 得到，见 _eval_node

        # 3. 找生产者节点，递归求值其输入
        node = self.graph.producer(name)
        if node is None:
            return self._store(name, None)      # 悬空，无法求值

        # 4. 逐个求值输入
        inputs = [self.evaluate(i) for i in node.input if i]
        if any(v is None for v in inputs):
            return self._store(name, None)      # 输入不全为常量 -> 不可求值

        # 5. 分发到算子实现
        out = self._eval_node(node, inputs)
        return self._store(name, out)

    def _eval_node(self, node, inputs):
        handler = self.HANDLERS.get(node.op_type)
        if handler is None:
            return None
        try:
            return handler(self, node, inputs)
        except Exception:
            return None     # 求值失败一律降级为 None，绝不让 Pass 崩溃

    def _store(self, name, value):
        self.memo[name] = value
        return value
```

**注意 —— `Shape` 算子的特殊处理**：`Shape(X)` 的输入 `X` 通常是**数据张量（非常量）**，但它的 **shape 已知**。所以 `Shape` 的 handler 不能依赖 `inputs[0]` 是常量，而要用 `self.graph.get_shape(X)`：

```python
def _eval_shape(self, node, inputs):
    # 注意：不走 inputs，直接查 X 的静态 shape
    x_name = node.input[0]
    shape = self.graph.get_shape(x_name)
    if shape is None:
        return None
    # opset >= 15: Shape 有 start/end 属性
    start = get_attr(node, "start", 0)
    end = get_attr(node, "end", None)
    arr = np.array(shape, dtype=np.int64)[start:end]
    return arr

def _eval_size(self, node, inputs):
    shape = self.graph.get_shape(node.input[0])
    return None if shape is None else np.array(np.prod(shape), dtype=np.int64)

def _eval_rank(self, node, inputs):
    shape = self.graph.get_shape(node.input[0])
    return None if shape is None else np.array(len(shape), dtype=np.int64)
```

**算子实现分三阶段落地**（先做第一阶段就能覆盖绝大多数 shape 子图）：

| 阶段 | 算子                                                                               | 说明                    |
| ---- | ---------------------------------------------------------------------------------- | ----------------------- |
| 一   | `Constant, Shape, Size, Rank, Cast, Gather, Concat, Unsqueeze, Squeeze, Reshape` | 纯 shape 计算，覆盖 80% |
| 二   | `Slice, Add, Sub, Mul, Div, Mod, Min, Max`                                       | 算术 + 切片             |
| 三   | `Range, Expand, Where, GatherND, Split`                                          | 高级                    |

**实现要点**：

- 用 NumPy 实现，语义对齐 ONNX 算子定义（尤其 `Gather` 的 axis、`Unsqueeze` 的 axes）。
- `Cast` 只需支持 shape 计算用到的整型/浮点互转。
- 所有 handler 签名统一为 `(evaluator, node, inputs) -> np.ndarray | None`。
- **递归深度保护**：图可能有环（Loop 子图），加一个 visiting 集合防死循环。

---

## Pass 1: Input Shape Specialization

**目标**：把用户指定的具体形状写入 `graph.input`，消除 `dim_param`。

**图模式**：不涉及节点，只改 `graph.input[*].type.tensor_type.shape`。

**实现步骤**：

```python
def specialize_input_shape(model, input_shapes: dict) -> bool:
    changed = False
    for inp in model.graph.input:
        if inp.name not in input_shapes:
            continue
        target = input_shapes[inp.name]         # 如 [1, 3, 640, 640]
        shape = inp.type.tensor_type.shape
        # 清空原维度（可能含 dim_param）
        del shape.dim[:]
        for d in target:
            dim = shape.dim.add()
            if isinstance(d, int):
                dim.dim_value = d
            else:
                dim.dim_param = d               # 保留符号名（少见）
        changed = True
    return changed
```

**两种输入形式都要支持**：

```python
# 形式一：直接给完整 shape
{"images": [1, 3, 640, 640]}

# 形式二：符号名映射（用于从 dim_param 推导）
{"images": {"N": 1, "H": 640, "W": 640}}
```

形式二需要先把 `dim_param` 读出来，再按名字替换。**建议第一版只支持形式一**，形式二留作扩展。

**注意事项**：

- 必须在**所有其他 Pass 之前**执行，否则 Shape Inference 拿不到起点。
- 如果用户给的 shape 与原 `dim_value` 冲突，应当报错而非静默覆盖。

---

## Pass 2: Shape Inference

**目标**：调用 ONNX 官方推导，把已知静态信息传播到全图。

**实现思路**：封装官方 API，不做自己实现。

```python
import onnx

def shape_inference(model, strict=True, data_prop=True) -> bool:
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model,
            check_type=True,
            strict_mode=strict,
            data_prop=data_prop,     # 关键：让 Shape/Gather 等也能被推导
        )
    except Exception:
        return False

    changed = _merge_value_info(model.graph, inferred.graph)
    return changed
```

**`_merge_value_info` 做什么**：`infer_shapes` 返回的是一个**新模型**，它的 `value_info` 包含推导结果。需要把新增的 value_info 合并回原图（按 name 去重，只在原来没有时才加）。

**注意事项**：

- `data_prop=True` 很重要：它让 ONNX 尝试推导 Shape 类算子的输出值。
- 官方 Shape Inference **不是万能的**：动态 Reshape、data-dependent 计算推不出来。这正是需要 Pass 4 的原因。
- **Shape Inference ≠ 动转静**。它只传播已有信息，不会"猜"出部署 shape。
- 该 Pass 会被**反复调用**（收敛循环里），所以要能容忍"无变化"并快速返回。

---

## Pass 3: Shape Tensor Evaluator 落地

> **已移除（v0.2）。** 本 Pass 折叠出的常量值与 Pass 4 完全相同，但写成
> `Constant` 节点而非 initializer。实测关掉它：最终常量集合逐个数值相等，
> 节点数更少（106 → 96），耗时更低。shape 子图统一由 Pass 4 的求值器处理。
> 本节保留作为设计演进记录。

**目标**：把「求值结果已知的 shape 子图」整体替换为一个 `Constant` 节点。

**与 0.3 的关系**：0.3 是引擎，这里是把引擎**接入图重写**的 Pass。

**图模式**：

```
Shape → Gather → Unsqueeze → Concat → Cast → Reshape
                    ↓ 求值
              Constant([1, 8400, 80])
```

**实现步骤**：

```python
class ShapeTensorEvaluationPass(Pass):
    name = "shape_tensor_evaluation"

    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(graph)
        changed = False

        # 1. 收集所有"疑似 shape 子图"的输出 value
        candidates = self._collect_shape_values(graph)

        for name in candidates:
            val = ev.evaluate(name)
            if val is None:
                continue
            # 2. 已有同名 initializer/Constant -> 跳过
            if graph.is_initializer(name):
                continue
            # 3. 替换：加一个 Constant 节点产出 name，并删掉原来的生产子图
            self._replace_with_constant(graph, name, val)
            changed = True

        if changed:
            self._prune_orphans(graph)   # 删除被孤立的 shape 计算节点
        return changed
```

**`_collect_shape_values` 怎么找候选**：

- 方式 A（保守）：只从 `Reshape` / `Slice` / `Expand` / `Tile` / `Split` / `Pad` 的**shape 参数输入**出发，反向收集。
- 方式 B（激进）：遍历所有 value_info，凡是 dtype 为 int64 且 rank ≤ 1 的都尝试求值。

**建议第一版用方式 A**，更安全，且正好命中真实痛点（PyTorch 导出的动态 shape 几乎都喂给这几个算子）。

**`_replace_with_constant` 的关键**：

- 新建 `Constant` 节点：`value = numpy_to_tensor(val)`。
- 用新节点的 output 名字**沿用原名** `name`，这样下游无需改线。
- 删除原来的生产者节点，但要**递归检查**：如果它的上游只服务于它，一并删除（这就是 `_prune_orphans`）。

**注意事项**：

- 替换必须保证**语义等价**：求值结果与原运行期结果一致。
- 如果 `name` 同时是 graph output，要谨慎（少见）。
- 删除节点时注意**多输出节点**：只有当某个 output 无消费者时才能删整个节点。

---

## Pass 4: Constant Propagation

**目标**：把「所有输入都是常量」的节点在编译期算出来。

**覆盖范围**：既处理纯数据常量（输入全是 initializer），也处理 shape 子图
（输入是数据张量、但 `Shape`/`Size` 只读它的类型，故仍可求值）。这两类在
求值器眼里没有区别 —— 判据统一为「所有输出都能求值」。

**实现思路**：复用 `ShapeTensorEvaluator`，但遍历对象是**所有节点**：

```python
class ConstantPropagationPass(Pass):
    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(graph)
        changed = False
        # 反复扫描直到本轮无新折叠（因为折叠一个可能让上游也可折叠）
        while True:
            round_changed = False
            for node in list(graph.graph.node):
                if node.op_type == "Constant":
                    continue
                if not node.input:
                    continue
                if all(graph.is_initializer(i) or ev.evaluate(i) is not None
                       for i in node.input if i):
                    # 所有输入可求值 -> 尝试折叠
                    outs = [ev.evaluate(o) for o in node.output if o]
                    if any(o is None for o in outs):
                        continue
                    self._fold(graph, node, outs)
                    round_changed = True
            changed |= round_changed
            if not round_changed:
                break
        return changed
```

**`_fold` 的实现**：把节点替换成 `Constant`（或直接写入 initializer）。**推荐写入 `initializer`** 而非新建 `Constant` 节点——NPU 编译器对 initializer 支持更好，且便于后续 `InitializerCleanup` 去重。

**注意事项**：

- **不是所有算子都能折叠**：随机算子（`RandomNormal`）、有副作用的算子必须排除。
- **控制流子图**：`If`/`Loop` 的 body 是子图，不能当普通节点折叠。
- 折叠后要立刻 `_rebuild()`，否则后续判断用的是旧索引。

---

## Pass 5: Constant Folding

**目标**：在 Pass 4 覆盖不到时，用 ONNX Runtime CPU 执行**小子图**来完成折叠。

**为什么需要它**：Pass 4 只能折叠自己实现了的算子。像 `Conv` 权重 + `BN` 这类复杂的常量计算，自己实现成本太高。

**两级策略**：

```
Level 1: 自实现算子（Pass 4）          —— 快、可控、覆盖 shape 类
Level 2: ORT 子图执行                  —— 兜底、覆盖复杂算子
```

**Level 2 实现思路 —— 抽子图 + 单独跑**：

```python
def fold_via_ort(graph, node) -> "np.ndarray | None":
    # 1. 以 node 为根，反向收集所有依赖（只含常量/initializer 可达的部分）
    subgraph = extract_constant_subgraph(graph, node)
    # 2. 把收集到的 initializer 作为 input，node 的 output 作为 output
    sub_model = build_model(subgraph)
    # 3. ORT 只跑这一小块
    sess = ort.InferenceSession(sub_model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    feeds = {init.name: to_numpy(init) for init in subgraph.initializers}
    return sess.run(None, feeds)
```

**关键纪律（文档中反复强调）**：

> **不要把整个模型交给 ORT 跑一遍来做 Constant Folding。**

原因：你无法控制哪些节点被折叠，会破坏后续 Pass 的假设，也可能改变数值行为。

**注意事项**：

- 抽子图时要**限定范围**：只允许纯常量依赖，遇到非常量输入就停止（该子图不可折叠）。
- ORT 执行失败要静默降级，不能让整个优化流程崩溃。
- 建议给 Level 2 加**开关**（默认开，出问题可关）。

---

## Pass 6: Shape / Size / Rank Elimination

**目标**：把 `Shape(X)` / `Size(X)` / `Rank(X)` 直接替换为常量。

**图模式**：

```
Shape(X)  →  Constant([1, 3, 640, 640])
Size(X)   →  Constant(1228800)
Rank(X)   →  Constant(4)
```

**实现思路**：这三种算子本质是 Pass 4 折叠的特例，但因为太常见，单独做成 Pass 便于统计与调试。

```python
class ShapeEliminationPass(Pass):
    name = "shape_elimination"
    HANDLED = {"Shape", "Size", "Rank"}

    def run(self, graph) -> bool:
        changed = False
        for node in list(graph.graph.node):
            if node.op_type not in self.HANDLED:
                continue
            x = node.input[0]
            shape = graph.get_shape(x)
            if shape is None:
                continue
            val = self._compute(node, shape)
            self._replace_with_initializer(graph, node.output[0], val)
            changed = True
        return changed

    def _compute(self, node, shape):
        if node.op_type == "Shape":
            return np.array(shape, dtype=np.int64)
        if node.op_type == "Size":
            return np.array(int(np.prod(shape)), dtype=np.int64)
        if node.op_type == "Rank":
            return np.array(len(shape), dtype=np.int64)
```

**注意事项**：

- **依赖 `graph.get_shape(x)` 的可靠性**：这依赖 Pass 1 + Pass 2 已经把 shape 传播到位。所以顺序上必须在它们之后。
- `Shape` 有 `start`/`end` 属性（opset 15+），要处理切片。
- 替换后，原来喂给 `Shape` 的那条链路可能变成死代码 → 交给 DCE 收尾。

---

## Pass 7: Dynamic Reshape Elimination

**目标**：把 `Reshape` 的 shape 输入从「动态计算」变成「静态常量」，或在等价时直接删除。

**三种情况**：

```python
# 情况 1: shape 输入可求值 -> 内联为静态
Reshape(X, ShapeTensor)         # ShapeTensor 是动态子图
    →  Reshape(X, Constant([1,8400,80]))

# 情况 2: 输入输出 shape 相同 -> 删除 Reshape
Reshape(X, [1,8400,80])         # X.shape == [1,8400,80]
    →  X

# 情况 3: shape 里含 0 / -1 等占位符 -> 展开为具体值
Reshape(X, [0, -1, 80])         # 已知 X=[1,8400,80]
    →  Reshape(X, [1,8400,80])
```

**实现思路**：

```python
class ReshapeEliminationPass(Pass):
    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(graph)
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Reshape":
                continue
            data, shape_in = node.input[0], node.input[1]

            # --- 情况 1/3: shape 输入可求值 ---
            target = ev.evaluate(shape_in)
            if target is not None:
                in_shape = graph.get_shape(data)
                resolved = resolve_placeholders(target, in_shape)  # 处理 0 / -1
                self._set_static_shape(graph, node, resolved)
                changed = True
                # 落到情况 2 的判断
                if resolved == in_shape:
                    self._bypass(graph, node)     # 直接重连，删除 Reshape
            # --- 情况 2: 无法求值但 shape 已知 ---
            elif graph.get_shape(data) == graph.get_shape(node.output[0]):
                self._bypass(graph, node)
                changed = True
        return changed
```

**`resolve_placeholders` 的关键逻辑**：

- `0` 表示「沿用输入的对应维度」（ONNX Reshape 语义）。
- `-1` 表示「该维度自动推断」。
- 二者都要求 `in_shape` 已知，否则无法解析 → 跳过。

**`_bypass` 的实现**（通用工具，多处复用）：

```python
def _bypass(graph, node):
    """把 producer(data) 直接连到 consumers(node.output)"""
    data_name = node.input[0]
    out_name = node.output[0]
    for consumer in graph.consumers(out_name):
        for i, inp in enumerate(consumer.input):
            if inp == out_name:
                consumer.input[i] = data_name
    # 若 out_name 是 graph output，需改 output 名
    _rename_graph_output(graph, out_name, data_name)
    _remove_node(graph, node)
```

**注意事项**：

- 改完必须重建索引，因为 `consumers` 查询依赖它。
- 若 `out_name` 是 graph output，不能简单删除，要把它从 output 列表里换名。

---

## Pass 8: Dynamic Slice Elimination

**目标**：静态化 `Slice` 的 `starts/ends/axes/steps`，并在「全量切片」时删除节点。

**图模式**：

```python
# 动态 -> 静态
Slice(X, dynamic_starts, dynamic_ends)   →  Slice(X, [0], [640], axes=[2])

# 全量切片 -> 删除
Slice(X, [0], [size(X)], axes=[0])       →  X
```

**实现思路**：

```python
class SliceEliminationPass(Pass):
    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(graph)
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Slice":
                continue
            # 1. 求值所有参数输入
            params = {}
            for key, idx in [("starts",1), ("ends",2), ("axes",3), ("steps",4)]:
                if len(node.input) > idx and node.input[idx]:
                    v = ev.evaluate(node.input[idx])
                    if v is None and key in ("starts", "ends"):
                        break        # 核心参数不可求值 -> 放弃
                    params[key] = v
            else:
                # 2. 全部求值成功 -> 写入 initializer，静态化
                self._materialize_params(graph, node, params)
                changed = True

                # 3. 判断是否为全量切片
                if self._is_full_slice(graph, node, params):
                    self._bypass(graph, node)
        return changed
```

**`_is_full_slice` 判断逻辑**：对每个轴，检查 `start==0`、`end==shape[axis]`、`step==1`。全部满足则是 no-op。

**注意事项**：

- ONNX `Slice` 的 `starts/ends/axes/steps` 是**输入**（不是属性），所以静态化 = 把这些输入替换成 initializer。
- `steps` 缺省为 1；`axes` 缺省为 `[0..len(starts)-1]`。
- 负索引语义（`ends=-1` 表示倒数第一个）要正确处理，别误判为全量。

---

## Pass 9: Dynamic Expand Elimination

**目标**：静态化 `Expand` 的目标 shape；若目标 shape 与输入相同则删除。

**图模式**：

```python
Expand(X, dynamic_shape)  →  Expand(X, Constant([1,80,6400]))
Expand(X, X.shape)        →  X            # broadcast 本身已满足
```

**实现思路**：

```python
class ExpandEliminationPass(Pass):
    def run(self, graph) -> bool:
        ev = ShapeTensorEvaluator(graph)
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Expand":
                continue
            data, shape_in = node.input[0], node.input[1]
            target = ev.evaluate(shape_in)
            if target is None:
                continue
            # 1. 静态化 shape 输入
            self._materialize(graph, node, 1, target)
            changed = True
            # 2. 若 broadcast 后 shape 不变 -> 删除
            in_shape = graph.get_shape(data)
            if in_shape is not None and list(target) == list(in_shape):
                self._bypass(graph, node)
        return changed
```

**注意事项**：

- `Expand` 遵循 broadcast 规则：目标 shape 中某维为 1 或等于输入维时不算扩展。严格判断「是否真的扩展」需要做 broadcast 对齐。
- 第一版可只做「完全相等则删除」这一保守判断，安全且能覆盖相当一部分冗余。

---

## Pass 10: Dead Code Elimination (DCE)

**目标**：删除所有「无法到达 graph output」的节点。

**实现思路 —— 从 output 反向可达性分析**：

```python
class DCEPass(Pass):
    def run(self, graph) -> bool:
        # 1. 收集所有 graph output 名
        live = set(o.name for o in graph.graph.output)

        # 2. 反向 BFS：从 output 出发，沿 producer 往上走
        stack = list(live)
        while stack:
            name = stack.pop()
            node = graph.producer(name)
            if node is None:
                continue          # 是 input 或 initializer
            for inp in node.input:
                if inp and inp not in live:
                    live.add(inp)
                    stack.append(inp)

        # 3. 删除所有 output 都不在 live 集合里的节点
        kept = []
        removed = 0
        for node in graph.graph.node:
            if any(o in live for o in node.output if o):
                kept.append(node)
            else:
                removed += 1
        del graph.graph.node[:]
        graph.graph.node.extend(kept)
        return removed > 0
```

**为什么不能写成 `if len(node.output) == 0: delete`**：那样只能删「无输出」的节点，而真正的死代码是「有输出但没人用」的节点。必须做反向可达性分析。

**注意事项**：

- 这是**唯一一个**会大幅减少节点数的 Pass，也是收敛循环里的常客。
- 必须在 `_bypass` / `_replace` 类操作之后运行，才能把留下的孤儿节点清干净。
- 建议 DCE 同时清理**孤立 initializer**（或交给 Pass 15）。

---

## Pass 11: Identity Elimination

**目标**：`A → Identity → B` 改为 `A → B`。

**实现思路**：最简单的 Pass，但一定要做——PyTorch 导出经常产生 Identity。

```python
class IdentityEliminationPass(Pass):
    name = "identity_elimination"

    def run(self, graph) -> bool:
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Identity":
                continue
            self._bypass(graph, node)      # 复用 Pass 7 的工具
            changed = True
        return changed
```

**注意事项**：

- **例外**：如果 `Identity` 的输出是 **graph output**，且输入也同时是别的东西，改名时要小心，别让两个 output 撞名。
- 某些框架用 Identity 标记「这是输出」，删除前确认语义（一般 ONNX 语义里 Identity 就是纯拷贝，可删）。

---

## Pass 12: Reshape Simplification

**目标**：减少 Reshape/Flatten 链，处理 Pass 7 没覆盖的结构性冗余。

**处理的模式**：

```
1. Reshape ∘ Reshape        →  Reshape(只保留最后一个)
2. Reshape(X) 且 shape 不变  →  X          （与 Pass 7 重叠，作为兜底）
3. Flatten ∘ Reshape        →  合并
4. Reshape ∘ Flatten        →  合并
```

**实现思路 —— 相邻同类节点合并**：

```python
class ReshapeSimplifyPass(Pass):
    def run(self, graph) -> bool:
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Reshape":
                continue
            prod = graph.producer(node.input[0])
            if prod is None or prod.op_type != "Reshape":
                continue
            # 中间 tensor 只被 node 消费，才可安全合并
            if len(graph.consumers(prod.output[0])) != 1:
                continue
            # 让 node 直接吃 prod 的输入
            node.input[0] = prod.input[0]
            self._try_remove(graph, prod)
            changed = True
        return changed
```

**注意事项**：

- 合并前必须确认**中间 tensor 只有一个消费者**，否则会破坏其他分支。
- `Flatten` 的 `axis` 属性要换算成等价的 `Reshape` shape，别直接当成 Reshape 处理。

---

## Pass 13: Transpose Simplification

**目标**：消除或合并 Transpose 链。

**处理的模式**：

```
1. Transpose(perm=identity)           →  删除
2. Transpose(p) ∘ Transpose(inv(p))   →  删除
3. Transpose(p) ∘ Transpose(q)        →  Transpose(q ∘ p)   （合并）
```

**实现思路 —— perm 合成**：

```python
def compose(p, q):
    """先做 q 再做 p，等价于 perm = [q[i] for i in p]"""
    return [q[i] for i in p]

class TransposeSimplifyPass(Pass):
    def run(self, graph) -> bool:
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Transpose":
                continue
            perm = get_attr(node, "perm", None)
            if perm is not None and perm == list(range(len(perm))):
                self._bypass(graph, node)      # identity -> 删除
                changed = True
                continue
            prod = graph.producer(node.input[0])
            if prod is None or prod.op_type != "Transpose":
                continue
            if len(graph.consumers(prod.output[0])) != 1:
                continue
            p = get_attr(prod, "perm", None)
            if p is None:
                continue
            merged = compose(perm, p)
            if merged == list(range(len(merged))):
                self._bypass(graph, node)      # 抵消 -> 删除两个
                self._try_remove(graph, prod)
            else:
                set_attr(node, "perm", merged) # 合并成一个
                node.input[0] = prod.input[0]
                self._try_remove(graph, prod)
            changed = True
        return changed
```

**注意事项**：

- `perm` 缺省是**逆序**（`[n-1,...,0]`），不是 identity！别搞错默认值。
- `compose` 的顺序容易写反，建议写单测验证：`Transpose(p)∘Transpose(inv(p)) == identity`。
- 三连 Transpose 也应能通过反复执行合并成一个（幂等性在这里体现）。

---

## Pass 14: Cast Simplification

**目标**：消除/合并多余的 Cast，这对 NPU 尤其重要（NPU 对 dtype 敏感）。

**处理的模式**：

```
1. Cast(t1→t2) ∘ Cast(t2→t1)   →  删除（若中间无其他消费者）
2. Cast ∘ Cast                  →  合并为一次 Cast
3. Constant(FP32) + Cast→FP16   →  直接把 initializer 转成 FP16
```

**实现思路**：

```python
class CastSimplifyPass(Pass):
    def run(self, graph) -> bool:
        changed = False
        for node in list(graph.graph.node):
            if node.op_type != "Cast":
                continue
            prod = graph.producer(node.input[0])
            if prod is None or prod.op_type != "Cast":
                continue
            if len(graph.consumers(prod.output[0])) != 1:
                continue
            src = get_attr(prod, "to")
            dst = get_attr(node, "to")
            if src == dst:
                self._bypass(graph, node)       # 同一个 cast 两次 -> 删后一个
                self._try_remove(graph, prod)
            elif src == _orig_dtype(graph, prod.input[0]):
                # 类型回到原始 -> 两个都可删（需确认无精度损失顾虑）
                self._bypass(graph, node)
                self._try_remove(graph, prod)
            else:
                # 合并为一个 Cast
                node.input[0] = prod.input[0]
                self._try_remove(graph, prod)
            changed = True
        return changed
```

**注意事项**：

- **FP32→FP16→FP32 的删除要谨慎**：虽然类型回到原点，但中间经过 FP16 已经损失精度，删除会**改变数值结果**。是否允许取决于 `profile`（NPU 部署常允许，因为最终也是 FP16）。
- 建议把这个行为做成**配置开关**，默认保守（只在 `src == dst` 时合并）。
- 类型用 ONNX 的 `TensorProto` 枚举（`FLOAT=1`, `FLOAT16=10`, `INT64=7` 等）。

---

## Pass 15: Initializer Cleanup

**目标**：删除没有任何消费者引用的 initializer，减小模型体积。

**实现思路**：

```python
class InitializerCleanupPass(Pass):
    def run(self, graph) -> bool:
        used = set()
        for node in graph.graph.node:
            used.update(i for i in node.input if i)
        # graph output 若直接引用 initializer，也要保留
        used.update(o.name for o in graph.graph.output)

        kept = [init for init in graph.graph.initializer if init.name in used]
        removed = len(graph.graph.initializer) - len(kept)
        if removed:
            del graph.graph.initializer[:]
            graph.graph.initializer.extend(kept)
        return removed > 0
```

**注意事项**：

- **子图**：`If`/`Loop`/`Scan` 的 body 里也引用 initializer，扫描时要**递归进子图**收集，否则会误删。
- 可以顺带做 **initializer 去重**（多个同名内容的常量合并），这是额外的体积优化，非必须。
- 该 Pass 通常放最后，因为前面的折叠会不断产生新常量、淘汰旧常量。

---

## 附：执行顺序与收敛

### 推荐顺序

```
第一轮（动转静）:
  P1  InputShapeSpecialization
  P2  ShapeInference
  P4  ConstantPropagation
  P5  ConstantFolding
  P6  ShapeElimination
  P7  ReshapeElimination
  P8  SliceElimination
  P9  ExpandElimination

第二轮（图简化）:
  P11 IdentityElimination
  P12 ReshapeSimplify
  P13 TransposeSimplify
  P14 CastSimplify

第三轮（清理）:
  P10 DCE
  P15 InitializerCleanup
```

### 为什么要循环

单个 Pass 常常会「解锁」上游的新机会。例如：

```
P6 消除 Shape(X) → X 的链路变死代码 → 需要 P10 DCE
P4 折叠 shape 子图 → 产生新常量 → 下一轮 P4 再传播一轮
P7 删除 Reshape   → 上游 Transpose 可能变成 identity → 需要 P13
```

所以用 `run_until_converged`（`MAX_ITER=10`）反复跑整条链，直到某一轮零变更。

### 收敛异常处理

如果 10 轮还没收敛，说明有 Pass **不幂等**（每次跑都报 changed）。此时应：

1. 输出各 Pass 的 `changes` 统计，定位是哪个 Pass 在空转；
2. 检查该 Pass 的 `run()` 是否错误地在「无实际修改」时返回了 `True`。

---

## 附：每个 Pass 的通用检查清单

写每个 Pass 时对照：

- [ ] **幂等**：连续跑两次，第二次必须返回 `False`。
- [ ] **安全**：改图前检查消费者数量、graph output 引用。
- [ ] **索引**：改图后重建 `GraphView` 索引。
- [ ] **统计**：记录 `changes` 数，供最终报告。
- [ ] **异常**：求值/解析失败一律跳过（返回 False），不抛异常中断全流程。
- [ ] **子图**：涉及 initializer/value_info 遍历时，考虑 `If`/`Loop` 子图。
