r"""ALE-Bench 适配层：把 ALE-Bench 的标量评分接到 rsi-agent 的 ScoreVector 上。

真实链路是：策略权重 → 代码（需 LLM 生成）→ ALE-Bench 容器内跑代码 →
overall_absolute_score → 逐 case 配对分数。本机 2026-11-19 实测：ale_bench 包缺失、
HF 不可达、无任何 LLM key → "权重→代码"与"容器跑代码"两环都不成立，端到端不可行。

因此适配器把可插拔的两环显式暴露：
- code_gen: Callable[[weights], tuple[str, ...]]，权重 → 待测代码；
- session: 真实用 ale_bench.start(...)，可注入 mock 做冒烟。

缺 code_gen 时 evaluate() 抛 ALEBenchUnavailable，绝不静默假装跑通。
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Protocol

from .core import HeuristicPolicy, ScoreVector

DEFAULT_PROBLEM = "ahc001"
DEFAULT_LANG = "python"


class ALEBenchUnavailable(RuntimeError):
    """ALE-Bench 端到端缺环（无 ale_bench / 无数据 / 无代码生成器）。"""


class _Session(Protocol):
    def public_eval(self, code: str, code_language: str = "python"): ...


@dataclass
class ALEBenchAdapter:
    """把 ALE-Bench 评分适配成 rsi-agent 可用的 task 接口。"""
    problem_id: str = DEFAULT_PROBLEM
    lite_version: bool = True
    n_cases: int = 8
    code_gen: Callable[[tuple[float, float, float]], tuple[str, ...]] | None = None
    session: object | None = None

    def __post_init__(self):
        self._session = None

    # ---- 可用性（只读探测，不联网）----
    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("ale_bench") is not None

    def _get_session(self):
        if self._session is None:
            if self.session is not None:
                self._session = self.session
            else:
                import ale_bench  # 依赖缺失时在这里明确抛 ImportError
                self._session = ale_bench.start(problem_id=self.problem_id,
                                                lite_version=self.lite_version)
        return self._session

    # ---- 真实评分 ----
    def score_cases(self, codes: tuple[str, ...]) -> tuple[float, ...]:
        """对每段代码跑 public_eval，返回逐 case 的 overall_absolute_score。"""
        sess = self._get_session()
        out: list[float] = []
        for code in codes:
            result = sess.public_eval(code, code_language=DEFAULT_LANG)
            out.append(float(result.overall_absolute_score))
        return tuple(out)

    # ---- rsi-agent task 接口（供 PromotionGate/runner 直接使用）----
    def episodes(self):
        # ALE-Bench 没有 episode 概念，用 case 数占位（runner 只取长度）
        return tuple(() for _ in range(self.n_cases))

    def evaluate(self, policy: HeuristicPolicy) -> ScoreVector:
        if self.code_gen is None:
            raise ALEBenchUnavailable(
                "权重→代码的生成环未提供（本机无 LLM key）；"
                "注入 code_gen 才能端到端跑，见 scripts/check_ale_readiness.py")
        codes = self.code_gen(policy.weights)
        scores = self.score_cases(codes)
        return ScoreVector(scores)


def mock_session(case_scores: tuple[float, ...]):
    """造一个假 ALE-Bench session，用于冒烟测试（不联网、不装包）。"""

    class _Mock:
        def public_eval(self, code: str, code_language: str = "python"):
            idx = len(getattr(self, "_calls", []))
            setattr(self, "_calls", getattr(self, "_calls", []) + [code])
            return type("R", (), {"overall_absolute_score": case_scores[idx % len(case_scores)]})()

    return _Mock()


def fixed_code_gen(seed: int = 0):
    """确定性 mock 代码生成器：权重不同 → 代码不同，便于冒烟。"""
    def gen(weights: tuple[float, float, float]) -> tuple[str, ...]:
        return tuple(f"# case {i} w={tuple(round(x, 3) for x in weights)}"
                     for i in range(8))
    return gen
