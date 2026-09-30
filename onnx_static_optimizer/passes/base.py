"""Pass 基类与 PassManager。"""

from typing import Dict, Iterable, List, Optional

from ..config import OptimizerConfig


class Pass:
    """所有 Pass 的基类。

    契约
    ----
    * ``run(graph) -> bool``：返回**是否发生变更**（不是是否成功）。
    * **幂等**：连续跑两次，第二次必须返回 False。
    * 改图后不自己 rebuild，由 PassManager 统一调用。
    * 任何解析/求值失败一律跳过，不抛异常中断全流程。
    """

    name = "base"
    priority = 0
    description = ""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self._changes = 0
        self._detail: Dict[str, int] = {}
        self._notes: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def reset_stats(self) -> None:
        self._changes = 0
        self._detail = {}
        self._notes = {}

    def bump(self, key: str = "", n: int = 1) -> None:
        """记一笔**真实变更**（会推进收敛判定）。"""
        self._changes += n
        if key:
            self._detail[key] = self._detail.get(key, 0) + n

    def note(self, key: str, n: int = 1) -> None:
        """记一笔**诊断计数**（跳过、失败等）—— 不计入 changes。

        这点很重要：把"因为求不出来而跳过"也当成变更，会让 Pass 每轮都
        报 changed，收敛循环永远跑满上限。
        """
        self._notes[key] = self._notes.get(key, 0) + n

    @property
    def changes(self) -> int:
        return self._changes

    @property
    def detail(self) -> Dict[str, int]:
        """本轮的真实变更细项。"""
        return dict(self._detail)

    @property
    def notes(self) -> Dict[str, int]:
        """本轮的诊断计数（跳过 / 失败）。"""
        return dict(self._notes)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "changes": self._changes,
            "detail": self.detail,
            "notes": self.notes,
        }

    # ------------------------------------------------------------------
    def run(self, graph) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:                      # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"


class PassManager:
    """按顺序执行 Pass，并循环至收敛。"""

    def __init__(self, passes: Optional[Iterable[Pass]] = None,
                 max_iter: int = 10):
        self.passes: List[Pass] = list(passes or [])
        self.max_iter = max_iter
        self.iterations = 0
        self.converged = False
        self.history: List[Dict[str, int]] = []
        # 跨轮次累计 —— 每轮 run() 会 reset_stats()，只看最后一轮的话
        # 拿到的是"收敛那一轮"的空统计（那一轮恰好什么都没改）。
        self.total_changes: Dict[str, int] = {}
        self.total_detail: Dict[str, Dict[str, int]] = {}
        self.total_notes: Dict[str, Dict[str, int]] = {}

    def add(self, p: Pass) -> "PassManager":
        self.passes.append(p)
        return self

    # ------------------------------------------------------------------
    def run_once(self, graph) -> bool:
        """执行一遍所有 Pass，返回是否有任何变更。"""
        changed = False
        for p in self.passes:
            p.reset_stats()
            try:
                if p.run(graph):
                    changed = True
                    graph.rebuild()     # 每个 pass 后重建，避免脏索引
            except Exception as exc:    # 单个 Pass 出错不该中断整条流水线
                p.note(f"error:{type(exc).__name__}")
                if getattr(self, "_verbose", False):
                    import traceback
                    traceback.print_exc()

            self._accumulate(p)
        return changed

    def _accumulate(self, p: Pass) -> None:
        if p.changes:
            self.total_changes[p.name] = (
                self.total_changes.get(p.name, 0) + p.changes)
        for store, values in ((self.total_detail, p.detail),
                              (self.total_notes, p.notes)):
            if not values:
                continue
            agg = store.setdefault(p.name, {})
            for k, v in values.items():
                agg[k] = agg.get(k, 0) + v

    def run_until_converged(self, graph) -> int:
        """循环执行直到某一轮零变更，返回执行的轮数。"""
        self.converged = False
        for i in range(self.max_iter):
            self.iterations = i + 1
            changed = self.run_once(graph)
            self.history.append({
                p.name: p.changes for p in self.passes if p.changes
            })
            if not changed:
                self.converged = True
                return self.iterations

        # 达到上限仍未收敛 -> 说明有 Pass 不幂等
        return self.iterations

    def stats(self) -> List[dict]:
        """跨轮次累计的统计。"""
        return [
            {
                "name": p.name,
                "changes": self.total_changes.get(p.name, 0),
                "detail": self.total_detail.get(p.name, {}),
                "notes": self.total_notes.get(p.name, {}),
            }
            for p in self.passes
        ]

    def stalled_passes(self) -> List[str]:
        """返回最后一轮仍在报变更的 Pass —— 不幂等的嫌疑对象。"""
        if not self.history:
            return []
        return list(self.history[-1].keys())


__all__ = ["Pass", "PassManager"]
