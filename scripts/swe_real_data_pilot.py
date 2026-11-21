r"""SWE-bench Verified 真实数据 pilot（2026-11-20）。

把已下到本地的真实 parquet（500 Verified 实例）元数据接进 rsi-agent 的两道闸：
- NC 前置审计闸（nc_guard）：在真实实例集上验证诚实 scorer 不误杀、padding/corrupted 被抓。
- paired_gate：在真实实例切出的 public/private split 上验证真提升放行、负担型攻击被拒。

诚实边界（必须显式声明，绝不假装端到端）：
- **真实**：500 个 instance_id / repo / difficulty / FAIL_TO_PASS / PASS_TO_PASS
  计数全部来自真实 parquet（hf-mirror 下到本地的 blob，2096679 bytes）。
- **替身**：patch 由确定性 code_gen 生成（无 LLM key），逐实例分数是确定性标量评分器
  基于真实实例属性算出的连续 F2P 比例（不是真在 container 里跑 repo 测试）。
  swebench 包 / GITHUB_TOKEN / LLM key 缺环未补，故 NOT 端到端。
- 真值锚（NC 闸的 probe.true_quality）= w·TRUE_W 外部直算，不经过被审 scorer（非循环）。

用法：python3 scripts/swe_real_data_pilot.py [--out results/swe_real_data_pilot.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi_agent.core import HeuristicPolicy, ScoreVector
from rsi_agent.gates import paired_gate
from rsi_agent.nc_guard import Probe, model_free_audit
from rsi_agent.swe_adapter import SWEBenchAdapter

PARQUET = Path.home() / ".cache/huggingface/hub/blobs/92/928ff5796c01f0fccf70d199d8b6318427eb1c181fa43666fdb81d4cce2872f3"
TRUE_W = (0.9, -0.6, 1.2)
DIFFICULTY_FACTOR = {
    "<15 min fix": 1.0, "15 min - 1 hour": 0.7,
    "1-4 hours": 0.4, ">4 hours": 0.2,
}


def load_instances(path: Path) -> list[dict]:
    d = pq.read_table(path).to_pydict()
    out = []
    for i, iid in enumerate(d["instance_id"]):
        f2p = len(json.loads(d["FAIL_TO_PASS"][i]))
        p2p = len(json.loads(d["PASS_TO_PASS"][i]))
        out.append({
            "instance_id": iid,
            "repo": d["repo"][i],
            "difficulty": d["difficulty"][i],
            "f2p_count": f2p,
            "p2p_count": p2p,
        })
    return out


def split_public_private(instances: list[dict], public_frac: float = 0.3):
    """按 instance_id 的 sha256 确定性切分（同 seed 永远同 split）。"""
    pub, priv = [], []
    for inst in instances:
        h = hashlib.sha256(inst["instance_id"].encode()).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        (pub if frac < public_frac else priv).append(inst)
    return pub, priv


def instance_feature(iid: str) -> tuple[float, float, float]:
    """instance_id → 确定性 3 维特征（不改真实数据，仅作评分替身的锚）。"""
    digest = hashlib.sha256(iid.encode()).digest()
    return tuple(round((b / 255.0) * 2 - 1, 4) for b in digest[:3])


def code_gen(weights):
    w = ", ".join(repr(round(x, 4)) for x in weights)
    return (f"w = ({w})  # deterministic patch (no LLM)",)


class ScalarSession:
    """逐实例确定性 F2P 比例评分器；bias/scramble 造三种 scorer。

    主分数基于 w·TRUE_W（与 NC 外部真值锚同序，保证诚实 scorer 不误杀）；
    真实实例属性（difficulty / F2P count）只提供小扰动与分辨率，不改变主排序。
    """
    AFF_MIN, AFF_MAX = -2.7, 2.7  # |w_i|<=1, TRUE_W=(0.9,-0.6,1.2) → 1-范数上界

    def __init__(self, instances, bias=0.0, scramble=False):
        self.by_id = {i["instance_id"]: i for i in instances}
        self.bias = bias
        self.scramble = scramble

    def evaluate_patch(self, instance_id, patch):
        w = tuple(float(x) for x in patch.split("(", 1)[1].split(")", 1)[0].split(","))
        inst = self.by_id[instance_id]
        aff = sum(a * b for a, b in zip(w, TRUE_W))
        score = (aff - self.AFF_MIN) / (self.AFF_MAX - self.AFF_MIN)  # →[0,1]
        # 真实实例属性：小扰动（不改主排序），体现"真实数据分布接入"
        dfac = DIFFICULTY_FACTOR.get(inst["difficulty"], 0.5)
        score += 0.005 * (dfac - 0.5)
        score += 0.005 * ((inst["f2p_count"] % 7) / 7.0 - 0.5)
        score += self.bias
        if self.scramble:
            noise = (int(hashlib.sha256(patch.encode()).hexdigest()[:8], 16) % 1000) / 1000.0
            score += (noise - 0.5) * 2.0  # 大幅扰动 → 打乱排序
        score = min(1.0, max(0.0, score))
        return type("R", (), {"f2p_ratio": score})()


def score_vector(instances, weights, session) -> ScoreVector:
    adapter = SWEBenchAdapter(
        instance_ids=tuple(i["instance_id"] for i in instances),
        code_gen=code_gen, session=session,
    )
    return adapter.evaluate(HeuristicPolicy(weights, name="real-data-probe"))


def build_probes(n, seed=7):
    import random
    rng = random.Random(f"swe-real:{seed}")
    probes = []
    for _ in range(n):
        w = tuple(round(rng.uniform(-1.0, 1.0), 4) for _ in range(3))
        aff = sum(a * b for a, b in zip(w, TRUE_W))
        tq = (aff - ScalarSession.AFF_MIN) / (ScalarSession.AFF_MAX - ScalarSession.AFF_MIN)
        probes.append(Probe(weights=w, true_quality=tq))
    return probes


def nc_cases(instances, probes, n_probes):
    """NC 闸在真实实例集上审三种 scorer。proxy 走真实实例打分均值。"""
    out = {}
    for name, sess in [
        ("honest", ScalarSession(instances)),
        ("padding", ScalarSession(instances, bias=0.27)),
        ("corrupted", ScalarSession(instances, scramble=True)),
    ]:
        proxy = [score_vector(instances, p.weights, sess).mean for p in probes]
        # gap_tol=0.02：真实实例属性扰动带来约 0.001 系统偏移，留余量；远小于 padding 的 0.27
        res = model_free_audit(probes, lambda ps, pv=proxy: pv, gap_tol=0.02)
        out[name] = {**res.as_dict(), "proxy_min": round(min(proxy), 4),
                     "proxy_max": round(max(proxy), 4), "proxy_distinct": len(set(proxy))}
    return out


def paired_cases(pub, priv):
    """paired_gate × 真实 split：真提升放行 + 负担型攻击被 REGRESSION_BURDEN 拒。"""
    improved = (0.683, -0.488, 0.886)  # 逼近 TRUE_W 方向（rsi-agent demo 收敛值）
    honest = ScalarSession(pub + priv)
    base_pub = score_vector(pub, (1.0, 0.0, 0.0), honest)
    cand_pub = score_vector(pub, improved, honest)
    floor = max(abs(base_pub.mean) * 0.02, 1e-6)
    real_gain = paired_gate(base_pub, cand_pub, floor, 0.05)

    base_priv = score_vector(priv, (1.0, 0.0, 0.0), honest)
    cand_priv = score_vector(priv, improved, honest)
    sealed_gain = paired_gate(base_priv, cand_priv,
                              max(abs(base_priv.mean) * 0.02, 1e-6), 0.05)

    # 负担型攻击：多数实例微升、少数大降 —— 均值升但 burden 高
    vals = list(base_pub.values)
    # 9/10 实例 +0.15、1/10 实例 -0.6：净 delta≈0.075>floor，但 burden≈0.44>0.25
    attack = tuple(v + 0.15 if i % 10 else v - 0.6 for i, v in enumerate(vals))
    attack_gate = paired_gate(base_pub, ScoreVector(attack), floor, 0.05)

    return {
        "real_gain_public": real_gate_dict(real_gain),
        "real_gain_sealed_private": real_gate_dict(sealed_gain),
        "burden_attack": real_gate_dict(attack_gate),
        "noise_floor_public": round(floor, 6),
        "n_public": len(pub), "n_private": len(priv),
        "baseline_public_mean": round(base_pub.mean, 4),
        "candidate_public_mean": round(cand_pub.mean, 4),
    }


def real_gate_dict(g):
    return {"passed": g.passed, "reason": g.reason, **g.as_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/swe_real_data_pilot.json")
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--nc-probes", type=int, default=30)
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        print(f"[ERR] parquet not found: {path}")
        return 2
    instances = load_instances(path)
    pub, priv = split_public_private(instances)
    probes = build_probes(args.nc_probes)

    nc = nc_cases(instances, probes, args.nc_probes)
    paired = paired_cases(pub, priv)

    expect = {
        "nc_honest_pass": nc["honest"]["passed"] is True,
        "nc_padding_rejected": nc["padding"]["passed"] is False and
                               nc["padding"]["reason"].startswith("OPTIMISTIC_GAP"),
        "nc_corrupted_rejected": nc["corrupted"]["passed"] is False and
                                 nc["corrupted"]["reason"].startswith("SELECTION_CORRUPTED"),
        "real_gain_public_pass": paired["real_gain_public"]["passed"] is True,
        "real_gain_sealed_pass": paired["real_gain_sealed_private"]["passed"] is True,
        "burden_attack_rejected": paired["burden_attack"]["passed"] is False and
                                  "REGRESSION_BURDEN" in paired["burden_attack"]["reason"],
    }
    ok = all(expect.values())
    for k, v in expect.items():
        print(f"[{'OK' if v else 'FAIL'}] {k}")

    payload = {
        "purpose": "SWE-bench Verified 真实数据（500 实例元数据）接 NC 闸 + paired_gate；"
                   "实例真实、评分为确定性替身（缺 LLM key/swebench/GITHUB_TOKEN，非端到端）",
        "parquet": str(path), "n_instances": len(instances),
        "parquet_bytes": path.stat().st_size,
        "real_data": {
            "n_repos": len(set(i["repo"] for i in instances)),
            "difficulty_dist": {k: sum(1 for i in instances if i["difficulty"] == k)
                                for k in DIFFICULTY_FACTOR},
            "f2p_count_min": min(i["f2p_count"] for i in instances),
            "f2p_count_max": max(i["f2p_count"] for i in instances),
            "p2p_count_max": max(i["p2p_count"] for i in instances),
        },
        "split": {"public": len(pub), "private": len(priv),
                  "method": "sha256(instance_id)[:8]/0xFFFFFFFF < 0.3"},
        "nc_cases": nc, "paired_cases": paired,
        "expect": expect, "all_match_expect": ok,
        "honest_boundary": "实例元数据真实；patch 为确定性 code_gen（非 LLM）；"
                           "逐实例分数为基于真实 F2P count 离散化的确定性标量替身（非真跑 repo 测试）。",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
