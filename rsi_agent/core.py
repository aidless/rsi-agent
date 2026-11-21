"""核心值类型。纯标准库，确定性，可哈希。

设计要点：
- ScoreVector 是逐 episode 的分数向量（越大越好），对策略本身不透明。
- PublicHistoryEntry 里的 weights/policy_hash/public_mean 全是"公开"信息：策略代码
  与它在 public 集上的表现，InnerAgent 可以看到；private 分数永远不在这里。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoreVector:
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class HeuristicPolicy:
    weights: tuple[float, float, float]
    name: str = "heuristic-v1"

    @property
    def hash(self) -> str:
        return canonical_hash({"weights": list(self.weights)})


@dataclass(frozen=True)
class PublicHistoryEntry:
    generation: int
    policy_hash: str
    weights: tuple[float, float, float]
    public_mean: float
    n_episodes: int
    promoted: bool
    gate_reason: str


@dataclass(frozen=True)
class PublicHistory:
    """InnerAgent 唯一能观察到的评测视图。构造上不含任何 private 字段。"""
    entries: tuple[PublicHistoryEntry, ...]

    def summary(self) -> str:
        if not self.entries:
            return "no history"
        best = max(self.entries, key=lambda e: e.public_mean)
        return (
            f"{len(self.entries)} generations; best public_mean={best.public_mean:.4f} "
            f"(policy={best.policy_hash[:12]}); promoted={sum(e.promoted for e in self.entries)}"
        )


class Policy(Protocol):
    weights: tuple[float, float, float]
