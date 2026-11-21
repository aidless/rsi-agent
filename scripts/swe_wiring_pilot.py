r"""SWE-bench Verified 接线 pilot：验证适配器接线正确 + NC 闸在二值 F2P 分数下的行为。

背景（2026-11-19 实测）：SWE-bench 端到端前置缺环——swebench 包缺失、HF 超时、
GITHUB_TOKEN unset（docker 可用但不足以端到端）。本 pilot 用确定性 patch_gen +
mock resolved 结果验证接线语义：
    权重 → patch（code_gen）→ evaluate_patch（mock session）→ 逐实例分数 → ScoreVector

同时验证一个诚实边界：二值 resolved（全 0/全 1）会让 NC 闸因 saturation 高而拒，
这是**正确行为**（二值分数无法做 selection_error 诊断），不是 bug。
真实 SWE-bench 用 F2P 比例（连续）作为 NC 闸输入才合适。

用法：python3 scripts/swe_wiring_pilot.py [--out results/swe_pilot.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi_agent.core import HeuristicPolicy
from rsi_agent.nc_guard import Probe, model_free_audit
from rsi_agent.swe_adapter import (
    SWEBenchAdapter, fixed_code_gen, mock_session,
)


def main() -> int:
    # 1) 接线冒烟：两 instance × 分数序列
    sess = mock_session(resolved=(1.0, 0.0, 1.0, 0.5))
    adapter = SWEBenchAdapter(
        instance_ids=("i1", "i2"), code_gen=fixed_code_gen(), session=sess)
    sv = adapter.evaluate(HeuristicPolicy((0.9, -0.6, 1.2)))
    wiring_ok = (len(sv.values) == 4 and sv.values == (1.0, 0.0, 1.0, 0.5))
    print(f"[wiring] values={sv.values} mean={round(sv.mean, 4)} ok={wiring_ok}")

    # 2) NC 闸：用 F2P 比例（连续）作 proxy，真值锚外部直算
    n = 12
    probes: list[Probe] = []
    for i in range(n):
        tq = round(0.05 + 0.07 * i, 6)  # 严格递增的连续真值
        probes.append(Probe(weights=(float(i), 0.0, 0.0), true_quality=tq))
    honest = model_free_audit(probes, lambda ps: [p.true_quality for p in ps])
    print(f"[nc-continuous] {honest.reason}")
    # 二值化代理（resolved 0/1）：应被 saturation 轴抓住
    binary_proxy = [1.0 if p.true_quality >= 0.5 else 0.0 for p in probes]
    binary = model_free_audit(probes, lambda ps: binary_proxy)
    print(f"[nc-binary] {binary.reason}")

    expect = {
        "wiring_ok": True,
        "nc_continuous_pass": honest.passed is True,
        "nc_binary_rejected": binary.passed is False,
    }
    got = {
        "wiring_ok": wiring_ok,
        "nc_continuous_pass": honest.passed is True,
        "nc_binary_rejected": binary.passed is False,
    }
    ok = all(got[k] == v for k, v in expect.items())
    payload = {
        "purpose": "SWE-bench 端到端 BLOCKED（swebench/HF/GITHUB_TOKEN），"
                   "用确定性 patch_gen + mock session 验证接线 + NC 闸二值边界",
        "wiring": {"values": list(sv.values), "mean": sv.mean},
        "nc_continuous": honest.as_dict(),
        "nc_binary": binary.as_dict(),
        "expect": expect,
        "got": got,
        "all_match_expect": ok,
    }
    out = Path("results/swe_pilot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
