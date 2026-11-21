"""NC（non-circular）前置 model-free 审计闸。

问题背景（papers-b 的 NC 威胁）：outer 闸只看"下游分有没有提升"，但若被审的
scorer 本身被污染（乐观、饱和、选择反转），下游分再漂亮也是循环论证。本闸在
任何分数进入 outer 确认闸之前，先在受控 probe grid 上量出三件事：

- selection_error：真值有序处是否出现严格反转（代理同分 = 中性，不算反转，
  使"饱和"成为可被独立检出的轴）；
- signed_gap：代理分相对真值的裸均值差（正 = 乐观 padding；用裸差而非 min-max，
  因为 min-max 会抹掉均匀抬升这种最难查的攻击）；
- saturation：代理分的重复比例（粗粒度/常数代理会让选择看起来"零错误"）。

真值锚（probe.true_quality）由外部任务给出，与被审 scorer 无关 —— 这是"非循环"
的字面含义：审计用的真值不来自被审量本身。
"""
from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-12


@dataclass(frozen=True)
class Probe:
    weights: tuple[float, float, float]
    true_quality: float


@dataclass(frozen=True)
class NCAuditResult:
    passed: bool
    reason: str
    selection_error: float
    signed_gap: float
    saturation: float
    n_probes: int

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "selection_error": round(self.selection_error, 6),
            "signed_gap": round(self.signed_gap, 6),
            "saturation": round(self.saturation, 6),
            "n_probes": self.n_probes,
        }


def model_free_audit(
    probes: list[Probe],
    proxy_score,
    selection_error_max: float = 0.15,
    gap_tol: float = 0.0,
    saturation_max: float = 0.5,
    min_probes: int = 8,
) -> NCAuditResult:
    """在 probe grid 上审计 proxy_score(probes) -> list[float]。

    gap_tol 默认 0（零乐观）；调用方若用带噪真值可传 noise_floor 作容差。
    各判定严格大于阈值才拒，恰等于阈值放行（边界语义钉死，测试 t7）。
    """
    proxy = list(proxy_score(probes))
    n = len(probes)
    if len(proxy) != n:
        return NCAuditResult(False, f"PROXY_LENGTH_MISMATCH: {len(proxy)}!={n}", 0.0, 0.0, 0.0, n)
    if n < min_probes:
        return NCAuditResult(False, f"INSUFFICIENT_PROBES: n={n}<{min_probes}", 0.0, 0.0, 0.0, n)

    truths = [p.true_quality for p in probes]
    flips = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            tdiff = truths[i] - truths[j]
            if abs(tdiff) < _EPS:       # 真值无差异：不可比，跳过
                continue
            pdiff = proxy[i] - proxy[j]
            if abs(pdiff) < _EPS:       # 代理同分：中性，不计反转（饱和独立成轴）
                continue
            pairs += 1
            if (tdiff > 0) != (pdiff > 0):
                flips += 1
    selection_error = flips / pairs if pairs else 0.0

    signed_gap = (sum(proxy) - sum(truths)) / n
    saturation = 1.0 - len(set(proxy)) / n

    if selection_error > selection_error_max:
        reason = f"SELECTION_CORRUPTED: se={selection_error:.4f}>{selection_error_max}"
    elif signed_gap > gap_tol:
        reason = f"OPTIMISTIC_GAP: gap={signed_gap:.4f}>{gap_tol}"
    elif saturation > saturation_max:
        reason = f"SATURATION_STARVATION: sat={saturation:.4f}>{saturation_max}"
    else:
        reason = f"PASS: se={selection_error:.4f} gap={signed_gap:.4f} sat={saturation:.4f}"
    return NCAuditResult(reason.startswith("PASS"), reason,
                         selection_error, signed_gap, saturation, n)
