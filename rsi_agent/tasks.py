r"""上下文决策任务：一个干净的、非循环的自我改进试验床。
每个 episode 给 n_actions 个候选，每个候选有 3 维特征 x（随机生成）。
策略用权重 w 给候选打分 argmax(w·x)，奖励由**独立的隐藏真值 w\*** 决定：
    reward = x_selected · w\* + deterministic_noise(episode)
关键（对照 P2 循环论证的翻版）：
- 标签来自隐藏 w\* 与 x 的内积，w\* 从不出现在特征里，策略只靠公开反馈搜索；
- public 与 private 共享 w\*（同一规律）但用不同 episode 样本（真 held-out）；
- noise 由 episode 种子确定，重复评测稳定，实例间有真实难度差异。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .core import HeuristicPolicy, ScoreVector

TRUE_W = (0.9, -0.6, 1.2)  # 隐藏真值；baseline 的 (1,0,0) 只是次优


@dataclass(frozen=True)
class ContextualTask:
    seed: int
    n_episodes: int = 128
    n_actions: int = 6
    true_w: tuple[float, float, float] = TRUE_W

    def episodes(self) -> tuple[tuple[tuple[float, float, float], ...], ...]:
        rng = random.Random(self.seed)
        eps: list[tuple[tuple[float, float, float], ...]] = []
        for _ in range(self.n_episodes):
            actions = tuple(tuple(round(rng.uniform(-1, 1), 4) for _ in range(3))
                            for _ in range(self.n_actions))
            eps.append(actions)
        return tuple(eps)

    def evaluate(self, policy: HeuristicPolicy) -> ScoreVector:
        w = policy.weights
        out: list[float] = []
        for ep, actions in enumerate(self.episodes()):
            chosen = max(actions, key=lambda x: w[0] * x[0] + w[1] * x[1] + w[2] * x[2])
            reward = (chosen[0] * self.true_w[0] + chosen[1] * self.true_w[1]
                      + chosen[2] * self.true_w[2])
            # 确定性实例噪声：与特征独立、由 episode 种子固定
            rng = random.Random(f"{self.seed}:{ep}")
            out.append(round(reward + rng.uniform(-0.05, 0.05), 6))
        return ScoreVector(tuple(out))


# 保留旧名以兼容 gates.py 的类型标注
BinPackTask = ContextualTask
