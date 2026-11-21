r"""InnerAgent：只吃公开反馈的策略改写器（真正的从反馈学习）。

"自我改进"确实发生在这里：每一代从公开历史里取出公开分数最高的策略作为锚点，
围绕它做步长衰减的局部扰动（高斯随机游走）。因为奖励沿隐藏真值方向单调，
best 会逐代爬升。它拿不到 ScoreVector、拿不到私集、拿不到 gate 内部状态、
更拿不到隐藏真值 w\*；observe() 的类型断言 + PublicHistory 不含 private 字段
把这条钉死。无内部状态机——锚点每代从 history 重算，杜绝时序 bug。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .barriers import assert_public_only
from .core import HeuristicPolicy, PublicHistory


@dataclass
class InnerAgent:
    seed: int = 7
    step: float = 0.5
    decay: float = 0.94
    min_step: float = 0.04

    def observe(self, history: object) -> str:
        assert_public_only(history)
        return history.summary()

    def propose(self, history: PublicHistory) -> HeuristicPolicy:
        assert_public_only(history)
        if not history.entries:
            return HeuristicPolicy((1.0, 0.0, 0.0), name="inner-seed")
        best = max(history.entries, key=lambda e: e.public_mean)
        center = list(best.weights)
        step = max(self.step * (self.decay ** len(history.entries)), self.min_step)
        rng = random.Random(f"{self.seed}:{len(history.entries)}")
        cand = tuple(round(c + rng.gauss(0.0, step), 4) for c in center)
        return HeuristicPolicy(cand, name=f"inner-gen{len(history.entries)}")


def baseline_policy() -> HeuristicPolicy:
    """固定的外部基线，供 paired gate 参考；不属于 InnerAgent 的搜索空间。"""
    return HeuristicPolicy((1.0, 0.0, 0.0), name="baseline-size-desc")
