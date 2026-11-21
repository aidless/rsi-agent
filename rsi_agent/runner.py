"""主编排：把 InnerAgent 的自我改写接进"公开探索 → 密封确认 → 哈希链审计"的闭环。

一次 run 的诚实结论只能是四种之一：
- SELF_IMPROVEMENT_VERIFIED：public 闸与 sealed 确认闸都过，且审计链完整；
- NO_VERIFIED_GAIN：搜索过但没一道确认闸过（"没测出提升"是合法结论，不编造）；
- AUDIT_FAILURE：审计链自校验失败（fail-closed，绝不报成功）；
- NC_AUDIT_FAILED：scorer 未过 NC 前置审计（主循环一步未跑即拒，sealed 零求值）。

绝不出现"因为 AUC 高就说成功"这种 P2 式自证——分数只对外部 scorer 负责。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .audit import AuditLog
from .barriers import SealedPrivateSet
from .core import HeuristicPolicy, PublicHistory, PublicHistoryEntry
from .gates import GateResult, PromotionGate, estimate_noise_floor, holm_alphas
from .nc_guard import NCAuditResult, Probe, model_free_audit
from .outer import InnerAgent, baseline_policy
from .tasks import BinPackTask


@dataclass
class GenerationRecord:
    generation: int
    policy_hash: str
    weights: tuple[float, float, float]
    public: GateResult
    private: GateResult | None
    promoted: bool


@dataclass
class RunResult:
    verdict: str
    audit_ok: bool
    audit_detail: str
    generations: list[GenerationRecord] = field(default_factory=list)
    sealed_evaluations: int = 0
    best_public_mean: float = 0.0
    baseline_public_mean: float = 0.0
    final_weights: tuple[float, float, float] = (1.0, 0.0, 0.0)
    nc_audit: NCAuditResult | None = None

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "audit_ok": self.audit_ok,
            "audit_detail": self.audit_detail,
            "nc_audit": self.nc_audit.as_dict() if self.nc_audit else None,
            "sealed_evaluations": self.sealed_evaluations,
            "baseline_public_mean": round(self.baseline_public_mean, 6),
            "best_public_mean": round(self.best_public_mean, 6),
            "final_weights": list(self.final_weights),
            "generations": [
                {
                    "generation": g.generation,
                    "policy_hash": g.policy_hash[:12],
                    "weights": list(g.weights),
                    "public": g.public.as_dict(),
                    "private": g.private.as_dict() if g.private else None,
                    "promoted": g.promoted,
                }
                for g in self.generations
            ],
        }


class RSIRunner:
    def __init__(
        self,
        public_seed: int = 100,
        private_seed: int = 99991,
        generations: int = 12,
        inner_seed: int = 7,
        family_alpha: float = 0.05,
        nc_probes: int = 0,
        nc_scorer=None,
        isolated: bool = False,
    ) -> None:
        self.public_task = BinPackTask(seed=public_seed)
        if isolated:
            from .isolated_sealed import IsolatedSealedPrivateSet
            self.sealed = IsolatedSealedPrivateSet(seed=private_seed)
        else:
            self.sealed = SealedPrivateSet(BinPackTask(seed=private_seed))
        self.generations = generations
        self.inner = InnerAgent(seed=inner_seed)
        self.family_alpha = family_alpha
        self.nc_probes = nc_probes
        self.nc_scorer = nc_scorer
        self.audit = AuditLog()
        self._sealed_baseline = None

    def run(self) -> RunResult:
        gate = PromotionGate(self.public_task, self.sealed,
                             noise_floor=estimate_noise_floor(self.public_task, baseline_policy()),
                             family_alpha=self.family_alpha)
        alphas = holm_alphas(self.generations, self.family_alpha)
        base_score = gate.baseline.mean
        history_entries: list[PublicHistoryEntry] = []
        result = RunResult(verdict="", audit_ok=False, audit_detail="",
                           baseline_public_mean=base_score)

        self.audit.append("run_start", {"generations": self.generations,
                                        "alpha_family": self.family_alpha,
                                        "public_seed": self.public_task.seed,
                                        "private_seed": "SEALED"})

        # NC 前置审计：任何分数进 outer 闸之前，先在受控 probe grid 上验 scorer
        # 非循环。真值锚来自 public_task（外部任务），与被审 scorer 无关。
        if self.nc_probes > 0:
            probes = self._build_probes(self.nc_probes)
            scorer = self.nc_scorer or (lambda ps: [p.true_quality for p in ps])
            nc = model_free_audit(probes, scorer)
            result.nc_audit = nc
            self.audit.append("nc_audit", nc.as_dict())
            if not nc.passed:
                ok, detail = self.audit.verify()
                result.audit_ok, result.audit_detail = ok, detail
                result.verdict = "NC_AUDIT_FAILED" if ok else "AUDIT_FAILURE"
                return result

        promoted_policy = baseline_policy()
        promoted_best_score = gate.baseline.mean
        for g in range(self.generations):
            history = PublicHistory(tuple(history_entries))
            proposal = self.inner.propose(history)
            self.audit.append("propose", {"generation": g, "weights": list(proposal.weights),
                                          "policy_hash": proposal.hash})

            # 公开集探索闸（付 Holm 校正）
            pub = gate.evaluate_public(proposal)
            # gate 的 family_alpha 每代不同：用 holm_alphas 覆盖其 alpha
            pub_alpha = alphas[g]
            pub = _realpha(pub, pub_alpha)
            self.audit.append("public_gate", {"generation": g, **pub.as_dict()})

            priv: GateResult | None = None
            promoted = False
            if pub.passed:
                # 只有过公开闸才触碰密封集——这是" sealed confirmation"的语义
                private_score = self.sealed.evaluate(proposal)
                priv = paired_gate_from_scores(self.sealed_baseline(), private_score,
                                               gate.noise_floor, pub_alpha)
                self.audit.append("private_confirm", {"generation": g, **priv.as_dict()})
                if priv.passed:
                    promoted = True
                    # final_weights 取"被确认提升中公开分最高"的那个，而非最后一个
                    pm = self.public_task.evaluate(proposal).mean
                    if pm > promoted_best_score:
                        promoted_best_score = pm
                        promoted_policy = proposal

            entry = PublicHistoryEntry(
                generation=g, policy_hash=proposal.hash, weights=proposal.weights,
                public_mean=self.public_task.evaluate(proposal).mean,
                n_episodes=len(self.public_task.episodes()),
                promoted=promoted, gate_reason=pub.reason,
            )
            history_entries.append(entry)
            result.generations.append(GenerationRecord(g, proposal.hash, proposal.weights,
                                                       pub, priv, promoted))
            result.best_public_mean = max(result.best_public_mean, entry.public_mean)
            self.audit.append("history_update", {"generation": g, "public_mean": entry.public_mean,
                                                 "promoted": promoted})

        result.sealed_evaluations = self.sealed.evaluation_count
        result.final_weights = promoted_policy.weights
        promoted_any = any(g.promoted for g in result.generations)
        self.audit.append("final", {"promoted_any": promoted_any,
                                    "final_weights": list(promoted_policy.weights)})

        ok, detail = self.audit.verify()
        result.audit_ok, result.audit_detail = ok, detail
        if not ok:
            result.verdict = "AUDIT_FAILURE"
        elif promoted_any:
            result.verdict = "SELF_IMPROVEMENT_VERIFIED"
        else:
            result.verdict = "NO_VERIFIED_GAIN"
        return result

    def sealed_baseline(self):
        if self._sealed_baseline is None:
            self._sealed_baseline = self.sealed.evaluate(baseline_policy())
        return self._sealed_baseline

    def _build_probes(self, n: int) -> list[Probe]:
        """围绕 baseline 做确定性高斯扰动；真值 = 外部任务评分。"""
        rng = random.Random(f"nc-probe:{self.inner.seed}")
        probes: list[Probe] = []
        for _ in range(n):
            w = tuple(round(rng.uniform(-1, 1), 4) for _ in range(3))
            quality = self.public_task.evaluate(HeuristicPolicy(w, name="nc-probe")).mean
            probes.append(Probe(weights=w, true_quality=quality))
        return probes


def paired_gate_from_scores(baseline, candidate, noise_floor, alpha):
    from .gates import paired_gate
    return paired_gate(baseline, candidate, noise_floor, alpha)


def _realpha(result: GateResult, alpha: float) -> GateResult:
    """用 Holm 分位 alpha 重判 p 值；其余字段不变。"""
    from dataclasses import replace
    passed = result.passed and result.p_value < alpha
    reason = result.reason if passed else f"HOLM_REJECT p={result.p_value:.5f}>=alpha={alpha:.5f}"
    return replace(result, passed=passed, reason=reason, alpha_used=alpha)


def run_demo(**kwargs) -> RunResult:
    return RSIRunner(**kwargs).run()
