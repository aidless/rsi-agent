r"""ALE-Bench 标量评分 pilot：用确定性 code_gen + ALE 风格连续标量评分器，
验证 NC 闸（nc_guard）在连续标量评分下"不误杀"且"能抓到"。

背景（2026-11-19 实测）：ALE-Bench 端到端在三条独立通道全缺硬前置——
本机 HF=000、ecs-hz 与 hpccube 均 HF 超时 + 无 docker + 无 LLM key + 无 ale_bench 包。
故本 pilot 用确定性实现复现 ALE 的评分链路语义：
    策略权重 → 代码（code_gen，确定性） → "跑代码"（scalar_session） → 连续标量分

关键点：NC 闸的真值锚必须来自外部直算，绝不经过被审 scorer，防循环。
- honest scorer：评分器从代码解析权重，与外部真值一致 → NC 闸应 PASS（不误杀）
- padding scorer：在真值上均匀抬升 → NC 闸应 OPTIMISTIC_GAP（能抓到）
- corrupted scorer：按代码哈希加扰打乱排序 → NC 闸应 SELECTION_CORRUPTED（能抓到）

用法：python3 scripts/ale_scalar_pilot.py [--out results/ale_pilot.json] [--nc-probes 60]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi_agent.ale_adapter import ALEBenchAdapter
from rsi_agent.core import HeuristicPolicy, ScoreVector
from rsi_agent.nc_guard import Probe, model_free_audit

TRUE_W = (0.9, -0.6, 1.2)
N_CASES = 8


def deterministic_code_gen(seed: int = 0):
    """确定性 code_gen：权重 → 代码字符串（复现 ALE 的 权重→代码 环，无需 LLM）。

    代码里显式写出权重，scalar_session 从代码解析回权重再打分，
    复现 ALE 的 代码→跑代码→标量分 环（这里"跑"是确定性的线性打分）。
    """
    def gen(weights: tuple[float, float, float]) -> tuple[str, ...]:
        w = ", ".join(repr(round(x, 4)) for x in weights)
        return tuple(f"w = ({w})  # case {i}" for i in range(N_CASES))
    return gen


def scalar_session_factory(hidden_w=TRUE_W, bias: float = 0.0, scramble: bool = False):
    """ALE 风格标量 session：从代码解析权重，算 权重·hidden_w + bias + scramble。

    bias=0/scramble=False → 诚实评分（与外部真值一致，NC 闸不误杀）。
    bias>0 → padding 均匀抬升（NC 闸 OPTIMISTIC_GAP）。
    scramble=True → 每段代码加一个按代码哈希的伪随机偏移，打乱排序（NC 闸 SELECTION_CORRUPTED）。
    """

    class _ScalarSession:
        def public_eval(self, code: str, code_language: str = "python"):
            # 从代码解析 w 元组
            body = code.split("#", 1)[0]
            inner = body.split("=", 1)[1].strip().strip("()")
            w = tuple(float(x) for x in inner.split(",") if x.strip())
            score = sum(a * b for a, b in zip(w, hidden_w)) + bias
            if scramble:
                # 按代码内容确定性加扰，打乱与真值的排序对应
                h = abs(hash(code)) % 1000
                score += (h % 7) - 3
            return type("R", (), {"overall_absolute_score": score})()

    return _ScalarSession()


def build_probes(n: int, seed: int = 7) -> list[Probe]:
    """probe grid：围绕 baseline 确定性扰动，真值锚外部直算（不经过被审 scorer）。"""
    rng = random.Random(f"ale-pilot:{seed}")
    probes: list[Probe] = []
    for _ in range(n):
        w = tuple(round(rng.uniform(-1.0, 1.0), 4) for _ in range(3))
        true_q = sum(a * b for a, b in zip(w, TRUE_W))  # 外部真值锚，防循环
        probes.append(Probe(weights=w, true_quality=true_q))
    return probes


def run_scorer_case(probes: list[Probe], session, code_gen) -> dict:
    """跑一种评分器配置，返回 NC 闸判定 + 关键量。"""
    adapter = ALEBenchAdapter(
        problem_id="ahc001", lite_version=True, n_cases=N_CASES,
        code_gen=code_gen, session=session,
    )
    # proxy：被审 scorer 的输出（从代码→标量分），经 adapter 链路
    proxy_scores = [
        adapter.evaluate(HeuristicPolicy(p.weights, name="ale-probe")).mean
        for p in probes
    ]
    result = model_free_audit(probes, lambda ps: proxy_scores)
    return {
        "passed": result.passed,
        "reason": result.reason,
        "selection_error": result.selection_error,
        "signed_gap": result.signed_gap,
        "saturation": result.saturation,
        "n_probes": result.n_probes,
        "proxy_min": min(proxy_scores),
        "proxy_max": max(proxy_scores),
        "proxy_distinct": len(set(proxy_scores)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ale_pilot.json")
    ap.add_argument("--nc-probes", type=int, default=60)
    args = ap.parse_args()

    probes = build_probes(args.nc_probes)
    code_gen = deterministic_code_gen()

    cases = {
        "honest": run_scorer_case(
            probes, scalar_session_factory(bias=0.0, scramble=False), code_gen),
        "padding_bias": run_scorer_case(
            probes, scalar_session_factory(bias=0.27, scramble=False), code_gen),
        "corrupted_noise": run_scorer_case(
            probes, scalar_session_factory(bias=0.0, scramble=True), code_gen),
    }

    # 期望：honest PASS；padding OPTIMISTIC_GAP；corrupted SELECTION_CORRUPTED
    expect = {"honest": "PASS", "padding_bias": "OPTIMISTIC_GAP",
              "corrupted_noise": "SELECTION_CORRUPTED"}
    ok = True
    for name, r in cases.items():
        got = r["reason"].split(":")[0]
        match = got == expect[name]
        ok = ok and match
        print(f"[{name}] {r['reason']} | match_expect={match}")

    payload = {
        "purpose": "ALE-Bench 端到端 BLOCKED（三通道缺 HF/docker/LLM key），"
                   "用确定性标量评分 pilot 验证 NC 闸不误杀且能抓到",
        "nc_probes": args.nc_probes,
        "cases": cases,
        "expect": expect,
        "all_match_expect": ok,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
