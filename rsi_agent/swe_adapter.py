r"""SWE-bench Verified 适配层：把每实例的 F2P/P2P 打分接到 rsi-agent 的 ScoreVector。

真实链路（SWE-bench 官方 swebench 包，container 内跑测试）：
    策略权重 → patch（code_gen，真实需 LLM 生成）→ swebench 跑 repo 测试 →
    resolved(bool) / F2P 比例 → 逐实例分数

本机 2026-11-19 实测前置缺环：swebench 包缺失、HF 超时、GITHUB_TOKEN unset
（docker 可用但不足以端到端）。故两环显式可插拔：
- code_gen: weights → patch 字符串元组（注入确定性/真实生成器）
- session: 真实 swebench 或注入 mock

缺 code_gen 时 evaluate() 抛 SWEBenchUnavailable，绝不静默假装跑通。
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Protocol

from .core import HeuristicPolicy, ScoreVector

DEFAULT_INSTANCE_IDS = ("django__django-11099", "sympy__sympy-13146")


class SWEBenchUnavailable(RuntimeError):
    """SWE-bench 端到端缺环（无 swebench / 无数据 / 无 patch 生成器）。"""


class _Session(Protocol):
    def evaluate_patch(self, instance_id: str, patch: str): ...


@dataclass
class SWEBenchAdapter:
    """把 SWE-bench Verified 的逐实例打分适配成 rsi-agent 可用的 task 接口。"""
    instance_ids: tuple[str, ...] = DEFAULT_INSTANCE_IDS
    code_gen: Callable[[tuple[float, float, float]], tuple[str, ...]] | None = None
    session: object | None = None

    def __post_init__(self):
        self._session = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("swebench") is not None

    def _get_session(self):
        if self._session is None:
            if self.session is not None:
                self._session = self.session
            else:
                raise SWEBenchUnavailable(
                    "未提供 swebench session，且本机 swebench 包缺失；"
                    "注入 session 才能端到端跑，见 scripts/check_swe_readiness.py")
        return self._session

    def score_patches(self, patches: tuple[str, ...]) -> tuple[float, ...]:
        """对每个 patch 在每个 instance 上跑测试，返回逐实例 F2P/resolved 分数。"""
        sess = self._get_session()
        out: list[float] = []
        for patch in patches:
            for iid in self.instance_ids:
                r = sess.evaluate_patch(iid, patch)
                out.append(float(getattr(r, "resolved", getattr(r, "f2p_ratio", 0.0))))
        return tuple(out)

    def episodes(self):
        return tuple(() for _ in self.instance_ids)

    def evaluate(self, policy: HeuristicPolicy) -> ScoreVector:
        if self.code_gen is None:
            raise SWEBenchUnavailable(
                "权重→patch 的生成环未提供（本机无 LLM key / swebench）；"
                "注入 code_gen 才能端到端跑，见 scripts/check_swe_readiness.py")
        patches = self.code_gen(policy.weights)
        return ScoreVector(self.score_patches(patches))


def mock_session(resolved: tuple[float, ...]):
    """造一个假 SWE-bench session，用于冒烟测试（不联网、不装包）。"""

    class _Mock:
        def evaluate_patch(self, instance_id: str, patch: str):
            idx = len(getattr(self, "_calls", []))
            setattr(self, "_calls", getattr(self, "_calls", []) + [(instance_id, patch)])
            return type("R", (), {"resolved": resolved[idx % len(resolved)]})()

    return _Mock()


def fixed_code_gen(seed: int = 0, n_patches: int = 2):
    """确定性 mock patch 生成器：权重不同 → patch 不同，便于冒烟。

    返回 n_patches 个候选 patch；score_patches 对每个 patch × 每个 instance 打分，
    故分数向量长度 = n_patches × len(instance_ids)。默认 2×2=4。
    """
    def gen(weights: tuple[float, float, float]) -> tuple[str, ...]:
        return tuple(
            "--- a/x.py\n+++ b/x.py\n"
            f"+# w={tuple(round(x, 3) for x in weights)} seed={seed} patch={i}"
            for i in range(n_patches)
        )
    return gen
