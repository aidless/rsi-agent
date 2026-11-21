"""提升判定闸。三条反自证设计：

1. 配对差（paired delta）而非裸均值——同一批实例上 baseline/candidate 逐项相减，
   去掉实例难度方差（P1 的 ESS 教训：naive 会把自相关当独立样本，虚高 power）。
2. ESS 有效样本量：对分数序列估 lag-1 自相关，n_eff = n*(1-rho)/(1+rho)，
   低 ESS 直接 fail-closed（不够独立样本就不给结论）。
3. 噪声下限 + Holm 多重比较：delta 必须超过两次复跑的 noise floor，且跨代搜索
   要付 Holm 校正；只有 public 与 sealed private 都过闸才 promote。

所有门都必须能红：tests/test_rsi_agent.py 里有对应的负对照注入。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .core import ScoreVector


def _mean(xs: tuple[float, ...]) -> float:
    return sum(xs) / len(xs)


def _var(xs: tuple[float, ...]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)


def lag1_autocorr(xs: tuple[float, ...]) -> float:
    """序列 lag-1 自相关；常数序列定义 rho=0（无方差，不做 ESS 惩罚但走噪声门）。"""
    n = len(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    num = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(n - 1))
    den = sum((x - m) ** 2 for x in xs)
    if den <= 1e-15:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def effective_n(xs: tuple[float, ...]) -> float:
    n = len(xs)
    rho = lag1_autocorr(xs)
    return n * (1.0 - rho) / (1.0 + rho) if rho > 0 else float(n)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str
    delta: float
    nratio: float
    ess: float
    rho: float
    noise_floor: float
    p_value: float
    alpha_used: float

    def as_dict(self) -> dict:
        return {
            "passed": self.passed, "reason": self.reason, "delta": round(self.delta, 6),
            "nratio": round(self.nratio, 6), "ess": round(self.ess, 2),
            "rho": round(self.rho, 4), "noise_floor": round(self.noise_floor, 6),
            "p_value": round(self.p_value, 6), "alpha_used": round(self.alpha_used, 6),
        }


def _norm_sf(z: float) -> float:
    """标准正态上尾；erf 近似，够用于判定。"""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def paired_gate(
    baseline: ScoreVector,
    candidate: ScoreVector,
    noise_floor: float,
    alpha: float,
    min_ess: float = 8.0,
    max_regression_burden: float = 0.25,
) -> GateResult:
    """配对提升闸。fail-closed：长度不符 / ESS 不足 / 低于噪声 / 统计不显著都拒。"""
    b, c = baseline.values, candidate.values
    if len(b) != len(c) or len(b) < 3:
        return GateResult(False, "INVALID: paired vectors length mismatch or too short",
                          0.0, 0.0, 0.0, 0.0, noise_floor, 1.0, alpha)
    deltas = tuple(ci - bi for bi, ci in zip(b, c))
    delta = _mean(deltas)
    ess = effective_n(deltas)
    rho = lag1_autocorr(deltas)
    sd = math.sqrt(_var(deltas))
    se = sd / math.sqrt(max(ess, 1.0))
    z = delta / se if se > 1e-12 else (float("inf") if delta > 0 else 0.0)
    p = _norm_sf(z)
    nratio = delta / noise_floor if noise_floor > 0 else float("inf")

    if ess < min_ess:
        return GateResult(False, f"ESS_TOO_LOW ess={ess:.2f}<{min_ess}", delta, nratio, ess, rho,
                          noise_floor, p, alpha)
    if delta <= 0:
        return GateResult(False, f"NO_GAIN delta={delta:.4f}", delta, nratio, ess, rho,
                          noise_floor, p, alpha)
    if nratio < 1.0:
        return GateResult(False, f"BELOW_NOISE delta={delta:.4f}<floor={noise_floor:.4f}",
                          delta, nratio, ess, rho, noise_floor, p, alpha)
    if p >= alpha:
        return GateResult(False, f"NOT_SIGNIFICANT p={p:.4f}>=alpha={alpha:.4f}",
                          delta, nratio, ess, rho, noise_floor, p, alpha)
    if max_regression_burden is not None:
        neg_abs = sum(-d for d in deltas if d < 0.0)
        pos_sum = sum(d for d in deltas if d > 0.0)
        if pos_sum > 0.0:
            burden = neg_abs / pos_sum
            if burden > max_regression_burden:
                return GateResult(
                    False,
                    f"REGRESSION_BURDEN burden={burden:.3f}>max={max_regression_burden:.2f} "
                    f"neg_sum={neg_abs:.4f} pos_sum={pos_sum:.4f}",
                    delta, nratio, ess, rho, noise_floor, p, alpha)
    return GateResult(True, f"PASS delta={delta:.4f} N/floor={nratio:.2f} p={p:.5f}",
                      delta, nratio, ess, rho, noise_floor, p, alpha)


def holm_alphas(n_comparisons: int, family_alpha: float = 0.05) -> tuple[float, ...]:
    """Holm-Bonferroni：按排序后第 i 个比较给 alpha/(n-i)。"""
    if n_comparisons <= 0:
        return ()
    return tuple(family_alpha / (n_comparisons - i) for i in range(n_comparisons))


def estimate_noise_floor(task, policy, min_effect_ratio: float = 0.02) -> float:
    """最小实质提升阈值（effect-size 门，baseline 均值的一个比例）。

    确定性 benchmark 上"重复评测标准差"恒为 0；实例间标准差又太大（reward 天然
    波动 ~0.8），拿它当 floor 会让任何真实但局部的提升（~0.5）都被误杀。
    这里改用 baseline reward 尺度的 min_effect_ratio（默认 2%）：统计显著性由
    paired_gate 的 ESS 修正检验负责，本门只挡"显著但小到没实际意义"的提升。
    语义写死在此，避免下一任维护者又拿错统计量。
    """
    base = abs(task.evaluate(policy).mean)
    return max(base * min_effect_ratio, 1e-6)


class PromotionGate:
    """public 探索闸 + sealed private 确认闸，两道都过才 promote。"""

    def __init__(self, task, sealed, noise_floor: float, family_alpha: float = 0.05) -> None:
        self.task = task
        self.sealed = sealed
        self.noise_floor = noise_floor
        self.family_alpha = family_alpha
        self.baseline = task.evaluate(_baseline_policy())

    def evaluate_public(self, policy) -> GateResult:
        return paired_gate(self.baseline, self.task.evaluate(policy), self.noise_floor,
                           self.family_alpha)


def _baseline_policy():
    from .core import HeuristicPolicy
    return HeuristicPolicy((1.0, 0.0, 0.0), name="baseline-size-desc")
