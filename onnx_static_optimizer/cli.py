"""命令行工具。

用法::

    # 指定输入形状
    python -m onnx_static_optimizer.cli yolov8n.onnx \
        -i images=1,3,640,640 -o yolov8n_static.onnx

    # 符号名映射形式
    python -m onnx_static_optimizer.cli model.onnx -i images=N:1,H:640,W:640

    # 只看不写
    python -m onnx_static_optimizer.cli model.onnx -i images=1,3,640,640 --dry-run
"""

import argparse
import sys
from typing import Dict

from .analysis.graph_statistics import analyze, dynamic_op_names
from .config import OptimizerConfig, ShapeSpec
from .optimizer import optimize


def parse_input_shapes(items) -> Dict[str, ShapeSpec]:
    """解析 ``-i`` 参数。

    ``images=1,3,640,640``    -> [1, 3, 640, 640]
    ``images=N:1,H:640,W:640`` -> {"N": 1, "H": 640, "W": 640}
    """
    shapes: Dict[str, ShapeSpec] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"输入形状格式错误：{item!r}（缺少 '='）")
        name, spec = item.split("=", 1)
        name = name.strip()

        if ":" in spec:
            mapping: Dict[str, int] = {}
            for part in spec.split(","):
                if ":" not in part:
                    raise ValueError(f"符号名映射格式错误：{part!r}")
                key, value = part.split(":", 1)
                mapping[key.strip()] = int(value)
            shapes[name] = mapping
        else:
            shapes[name] = [int(x) for x in spec.split(",") if x.strip()]

    return shapes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="onnx-static-optimizer",
        description="把带动态维度的 ONNX 模型静态化，供 NPU 编译器使用。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("model", help="输入 ONNX 模型路径")
    p.add_argument("-o", "--output", help="输出 ONNX 模型路径")
    p.add_argument("-i", "--input-shape", action="append", metavar="NAME=SHAPE",
                   help="输入形状，可重复。如 images=1,3,640,640")
    p.add_argument("--max-iter", type=int, default=10,
                   help="收敛循环上限（默认 10）")
    p.add_argument("--no-ort-fold", action="store_true",
                   help="关闭 Pass 4 的 ORT 兜底（仅用自实现求值器折叠）")
    p.add_argument("--allow-fp16-roundtrip", action="store_true",
                   help="允许删除 FP32→FP16→FP32 的往返 Cast（会改变数值）")
    p.add_argument("--allow-overwrite", action="store_true",
                   help="允许用户 shape 覆盖模型里已有的 dim_value")
    p.add_argument("--no-numeric-check", action="store_true",
                   help="跳过数值等价验证")
    p.add_argument("--dry-run", action="store_true",
                   help="只分析并打印报告，不写文件")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    input_shapes = None
    try:
        input_shapes = parse_input_shapes(args.input_shape)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    import onnx
    source = onnx.load(args.model)

    # 先给用户看一眼"这个模型里有什么动态算子"
    print(f"Model: {args.model}")
    print(f"Dynamic-related ops: {dynamic_op_names(source)}")
    print(analyze(source).format())
    print()

    config = OptimizerConfig(
        max_iterations=args.max_iter,
        enable_ort_folding=not args.no_ort_fold,
        allow_fp16_roundtrip_removal=args.allow_fp16_roundtrip,
        validate_numerically=not args.no_numeric_check,
        verbose=args.verbose,
    )

    try:
        result = optimize(source, input_shapes=input_shapes, config=config)
    except (KeyError, ValueError) as exc:
        print(f"优化失败：{exc}", file=sys.stderr)
        return 1

    print(result.report())

    if not args.dry_run:
        if not args.output:
            print("未指定 -o，跳过写文件（等价于 --dry-run）", file=sys.stderr)
        else:
            result.save(args.output)
            print(f"\nSaved: {args.output}")

    # 退出码：静态化成功且验证通过才为 0
    if not result.is_static:
        return 3
    if not result.checker_ok:
        return 4
    if result.numerical_ok is False:
        return 5
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
