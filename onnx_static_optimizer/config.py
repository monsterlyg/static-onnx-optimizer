"""优化器配置。

所有"可能改变数值行为"的激进优化都必须由配置开关控制，默认取保守值。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OptimizerConfig:
    """优化器配置。

    字段按流水线顺序排列，与 pass_implementation.md 中的 Pass 编号对应。
    """

    # ---------------- Pass 1: Input Shape Specialization ----------------
    # {"images": [1, 3, 640, 640]}
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)

    # ---------------- Pass 2: Shape Inference ----------------
    strict_shape_inference: bool = True # 如果推理过程中遇到任何错误，会直接抛出异常，而不是静默停止
    # 是 Pass 3 之后一切工作的起点。
    data_prop: bool = True # 对少数特定算子启用数据传播，尝试利用 initializer 中的常量数据来计算形状

    # ---------------- Pass 3 / 4: 求值器 ----------------
    enable_shape_tensor_evaluation: bool = True
    enable_constant_propagation: bool = True

    # ---------------- Pass 5: ORT 子图折叠 ----------------
    enable_ort_folding: bool = True
    # 单次折叠允许的最大子图节点数 / 元素总数，防止误抽大子图
    ort_max_subgraph_nodes: int = 64
    ort_max_elements: int = 1 << 22  # 4M elements

    # 单个常量折叠结果的最大元素数。超过则放弃折叠 —— 把巨大的数据张量
    # 折成 initializer 只会让模型变大，且会破坏后续 Pass 的假设。
    max_constant_elements: int = 1 << 20  # 1M elements

    # ---------------- Pass 6-9: 动态算子静态化 ----------------
    enable_shape_elimination: bool = True
    enable_reshape_elimination: bool = True
    enable_slice_elimination: bool = True
    enable_expand_elimination: bool = True

    # ---------------- Pass 11-14: 结构简化 ----------------
    enable_identity_elimination: bool = True
    enable_reshape_simplify: bool = True
    enable_transpose_simplify: bool = True
    enable_cast_simplify: bool = True

    # FP32→FP16→FP32 的往返 Cast 删除会改变数值结果（中间已损失精度）。
    # NPU 部署常允许，但默认关闭，保守优先。
    allow_fp16_roundtrip_removal: bool = False

    # bypass 一个产出 graph output 的节点时，默认把 output 改名成上游名字
    # （pass_implementation.md 的做法）。开启此项则放弃 bypass、保留该节点，
    # 以保持输出名不变 —— 下游按名字取输出的部署代码不受影响。
    preserve_graph_output_names: bool = False

    # ---------------- Pass 10 / 15: 清理 ----------------
    enable_dce: bool = True
    enable_initializer_cleanup: bool = True

    # ---------------- 流水线 ----------------
    max_iterations: int = 10

    # ---------------- 验证 ----------------
    validate_numerically: bool = True
    numeric_rtol: float = 1e-4
    numeric_atol: float = 1e-5

    verbose: bool = False
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if k != "input_shapes"
        }
