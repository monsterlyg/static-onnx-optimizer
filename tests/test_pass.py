"""单个 Pass 的调试入口。

按编号或名字挑一个 Pass，跑在示例模型（或指定模型）上，看它到底改了什么。

用法::

    python tests/test_pass.py --list                  # 列出全部 Pass
    python tests/test_pass.py 06                      # Pass 6：Shape/Size -> 常量
    python tests/test_pass.py shape_elimination       # 也可以按名字片段
    python tests/test_pass.py 11 --repeat 2           # 跑两遍，验证幂等（第二遍应 False）
    python tests/test_pass.py 06 --no-prefix          # 不跑前置 Pass，直接上
    python tests/test_pass.py --all                   # 逐个跑一遍全部 Pass
    python tests/test_pass.py 11 --demo static
    python tests/test_pass.py 06 -m yolov8n.onnx -i images=1,3,640,640 --save out.onnx

为什么要跑"前置 Pass"
--------------------
单个 Pass 很少能直接动手：Pass 6 要消 ``Shape``，得先有 Pass 2 推出来的
value_info；Pass 7 要静态化 ``Reshape``，得先有 Pass 3/4 把 newshape 折成常量。
所以默认先按编号把更小的 Pass 各跑一遍（``--no-prefix`` 关掉），
这样看到的是"这个 Pass 在真实流水线里能吃到什么、又做了什么"。

每个目标 Pass 都重新构造一份 Pass 实例和一份模型副本 —— Pass 内部可能带状态
（例如 ``ShapeInferencePass`` 缓存了图指纹），跨目标复用会互相串味。
"""

import argparse
import copy
import os
import sys
import traceback
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx

from onnx_static_optimizer import OptimizerConfig
from onnx_static_optimizer.cli import parse_input_shapes
from onnx_static_optimizer.graph.graph_view import GraphView
from onnx_static_optimizer.optimizer import build_pipeline
from onnx_static_optimizer.passes.base import Pass
from onnx_static_optimizer.shape.shape_specializer import InputShapeSpecializationPass
from onnx_static_optimizer.validation.static_validator import validate

from models import (
    build_data_dependent_model,
    build_shape_start_end_model,
    build_static_model,
    build_yolo_like,
)

# 内置示例模型：名字 -> (构造函数, 默认定形目标)。默认 yolo 覆盖的 Pass 最多。
DEMOS = {
    "yolo": (build_yolo_like, {"images": [1, 3, 8, 8]}),
    "static": (build_static_model, {"images": [1, 3, 8, 8]}),
    "shape-start-end": (build_shape_start_end_model, {"images": [1, 3, 8, 8]}),
    "data-dependent": (build_data_dependent_model, {"images": [1, 3]}),
}


# ======================================================================
# Pass 注册表
# ======================================================================
def build_all_passes(config: OptimizerConfig) -> List[Pass]:
    """全部 Pass，按 priority 排序。

    Pass 1 在收敛循环外单独执行，不在 ``build_pipeline`` 里，这里补上 ——
    否则 ``test_pass.py 01`` 找不到目标。
    """
    passes = [InputShapeSpecializationPass(config)] + build_pipeline(config)
    return sorted(passes, key=lambda p: p.priority)


def resolve_passes(token: str, passes: List[Pass]) -> List[Pass]:
    """把 ``06`` / ``6`` / ``shape_elimination`` 解析成 Pass 列表。"""
    token = token.strip()

    if token.isdigit():
        hit = [p for p in passes if p.priority == int(token)]
        if hit:
            return hit
        raise SystemExit(f"没有编号为 {token} 的 Pass（--list 查看全部）")

    exact = [p for p in passes if p.name == token]
    if exact:
        return exact

    fuzzy = [p for p in passes if token.lower() in p.name.lower()]
    if len(fuzzy) == 1:
        return fuzzy
    if not fuzzy:
        raise SystemExit(f"没有名字含 {token!r} 的 Pass（--list 查看全部）")
    raise SystemExit(f"{token!r} 匹配到多个 Pass："
                     f"{', '.join(p.name for p in fuzzy)}")


# ======================================================================
# 跑 Pass
# ======================================================================
def run_pass_once(p: Pass, model) -> dict:
    """在 model 上跑一次 Pass，返回统计。

    异常**不吞** —— 调试入口就是要看到 traceback，这与 PassManager 的
    "单 Pass 出错不中断流水线"策略正好相反。
    """
    graph = GraphView(model)
    p.reset_stats()
    changed = bool(p.run(graph))
    if changed:
        graph.rebuild()          # 改完必须重建索引，否则后续查询拿到脏数据

    stats = dict(p.stats())
    stats["changed"] = changed
    return stats


def summary(model) -> dict:
    """图规模 + 算子计数。

    规模三项都要看：Pass 15/16 只动 initializer 和 value_info，不看这两项
    会显示成"什么都没做"。
    """
    s = {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "ops": {},
    }
    for node in GraphView(model).graph.node:
        s["ops"][node.op_type] = s["ops"].get(node.op_type, 0) + 1
    return s


def format_summary(s: dict) -> str:
    head = f"nodes={s['nodes']} init={s['initializers']} vi={s['value_info']}"
    ops = sorted(s["ops"].items(), key=lambda kv: (-kv[1], kv[0]))
    detail = "  ".join(f"{op}={n}" for op, n in ops) or "(空图)"
    return f"{head}   ops: {detail}"


def format_diff(before: dict, after: dict) -> str:
    parts = []
    for key, label in (("nodes", "nodes"), ("initializers", "init"),
                       ("value_info", "vi")):
        if before[key] != after[key]:
            parts.append(f"{label} {before[key]}->{after[key]}")

    ops = [f"{op} {before['ops'].get(op, 0)}->{after['ops'].get(op, 0)}"
           for op in sorted(set(before["ops"]) | set(after["ops"]))
           if before["ops"].get(op, 0) != after["ops"].get(op, 0)]
    if ops:
        parts.append("  ".join(ops))
    return "  |  ".join(parts) if parts else "(无变化)"


def run_target(args, priority: int, source, config: OptimizerConfig
               ) -> Tuple[str, bool]:
    """跑一个 Pass（含前置），打印它改了什么。返回 (名字, 是否无异常)。"""
    # 全新的 Pass 实例 + 全新的模型副本：避免跨目标串状态
    passes = build_all_passes(config)
    target = next(p for p in passes if p.priority == priority)
    model = copy.deepcopy(source)

    prefix = ([] if args.no_prefix
              else [p for p in passes if p.priority < target.priority])

    print("=" * 64)
    print(f"[{target.priority:02d}] {target.name}   (priority={target.priority})")
    print(f"  {target.description}")
    print("-" * 64)
    print(f"prefix : {len(prefix)} pass(es)"
          + (f" -> {[p.name for p in prefix]}" if args.verbose else ""))

    try:
        for p in prefix:
            run_pass_once(p, model)
        before = summary(model)
        print(f"before : {format_summary(before)}")

        stats_list = [run_pass_once(target, model) for _ in range(args.repeat)]
        after = summary(model)
    except Exception:
        traceback.print_exc()
        print(f"FAIL   : {target.name} 抛异常")
        return target.name, False

    for i, st in enumerate(stats_list):
        tag = f"run#{i + 1}" if args.repeat > 1 else "run   "
        print(f"{tag} : changed={st['changed']}  changes={st['changes']}")
        if st["detail"]:
            print(f"         detail: {st['detail']}")
        if st["notes"]:
            print(f"         notes : {st['notes']}")

    print(f"after  : {format_summary(after)}")
    print(f"diff   : {format_diff(before, after)}")

    report = validate(model)
    if report.is_static:
        print("static : PASS")
    else:
        print(f"static : ISSUES -> {', '.join(report.static_issues)}")
    if report.data_dependent_ops:
        print(f"data-dep: {report.data_dependent_ops}")

    if args.save:
        onnx.save(model, args.save)
        print(f"saved  : {args.save}")
    return target.name, True


# ======================================================================
# 输入
# ======================================================================
def load_source(args):
    """取一份原始模型：``--model`` 优先，否则用内置示例。"""
    if args.model:
        return onnx.load(args.model)
    builder, _ = DEMOS[args.demo]
    return builder()


def resolve_input_shapes(args) -> Dict[str, list]:
    """定形目标：``-i`` 优先，否则用示例模型自带的默认值。"""
    if args.input_shape:
        return parse_input_shapes(args.input_shape)
    if args.model:
        raise SystemExit("用 --model 时必须用 -i 给出目标输入形状，"
                         "例如 -i images=1,3,640,640")
    return dict(DEMOS[args.demo][1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_pass",
        description="单个 Pass 的调试入口：挑一个 Pass 跑在示例模型上，看它改了什么。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pass_name", nargs="?",
                   help="要测的 Pass：编号（06）或名字片段（shape_elimination）")
    p.add_argument("-l", "--list", action="store_true",
                   help="列出全部 Pass 后退出")
    p.add_argument("-a", "--all", action="store_true",
                   help="逐个跑一遍全部 Pass")
    p.add_argument("-m", "--model", metavar="PATH",
                   help="输入 ONNX 模型路径；省略则用内置示例模型")
    p.add_argument("--demo", choices=sorted(DEMOS), default="yolo",
                   help="内置示例模型（默认 yolo，覆盖的 Pass 最多）")
    p.add_argument("-i", "--input-shape", action="append", metavar="NAME=SHAPE",
                   help="输入形状，可重复。如 images=1,3,640,640 或 "
                        "images=N:1,H:640,W:640")
    p.add_argument("--no-prefix", action="store_true",
                   help="不跑前置 Pass（默认会先按编号跑一遍更小的 Pass）")
    p.add_argument("--repeat", type=int, default=1, metavar="N",
                   help="重复跑目标 Pass N 次（>1 可验证幂等，第二遍应 changed=False）")
    p.add_argument("--save", metavar="PATH", help="把跑完的模型写出来")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.pass_name = '2' # shape inference pass
    args.input_shape = ["speech=1,80,400"]  # Must be a list for parse_input_shapes
    args.model = R"D:\study_and_work\Honghe\onnx-static-optimizer\assets\fsmn-vad\model.onnx"

    if args.list:
        passes = build_all_passes(OptimizerConfig())
        print(f"可用的 Pass（共 {len(passes)} 个，按 priority 排序）:")
        for p in passes:
            print(f"  {p.priority:>2}  {p.name:<30} {p.description}")
        return 0

    if args.repeat < 1:
        build_parser().error("--repeat 必须 >= 1")
    if args.all and args.save:
        build_parser().error("--all 与 --save 不能同时用：每个 Pass 都会覆盖同一个文件")

    config = OptimizerConfig(
        input_shapes=resolve_input_shapes(args),
        validate_numerically=False,      # 单 Pass 调试不做端到端数值验证
        verbose=args.verbose,
    )
    passes = build_all_passes(config)

    if args.all:
        targets = [p.priority for p in passes]
    elif args.pass_name:
        targets = [p.priority for p in resolve_passes(args.pass_name, passes)]
    else:
        build_parser().error("请给出要测的 Pass（编号或名字），或用 --list 查看全部")

    source = load_source(args)
    print(f"source : {args.model or f'<demo:{args.demo}>'}"
          f"   input_shapes={config.input_shapes}")
    print(f"target : {len(targets)} pass(es)")

    results = [run_target(args, prio, source, config) for prio in targets]

    print("=" * 64)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"{len(failed)}/{len(results)} FAILED: {failed}")
        return 1
    print(f"OK: {len(results)} pass(es) 跑完，无异常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
