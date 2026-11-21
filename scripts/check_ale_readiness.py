"""ALE-Bench 接入前置检查（只读探测，不安装、不联网下载）。

在真正把 rsi-agent 接到 ALE-Bench 之前，先回答"这一格能不能跑开"。检查项与
判定（fail-closed）：

- 依赖包 ale_bench 可导入；
- 数据可用：HF 可达 或 ALE_BENCH_DATA 指向本地目录；
- Docker 可用（ALE-Bench 每题在容器里跑代码）；
- API key（ALE-Bench 需要 LLM 来生成/改写代码）。

燃料分级（2026-11-19 加）：ALE 端到端缺环时不再只报 BLOCKED，而是给出本地
held-out 替代燃料的可行性。fuel_grade 三态：
- ALE_REAL：上面五条 ALE 项全过，能用真 ALE-Bench 端到端；
- LOCAL_HELD_OUT：ALE 项不全过，但本地 ContextualTask 满足三条件（public/private
  seed 不重叠、特征与真值无关、逐实例难度有差异）⇒ 可用本地真 held-out 燃料；
- NONE：两者都不满足，无可用燃料。

退出码与 ready/blocked/verdict 的原语义保持不变（0 = 端到端就绪；1 = 有阻塞项，
fuel_grade 另报本地替代燃料可行性；2 = 用法错）。fuel_grade 是纯新增信息，
不改变任何既有消费者的退出码预期。
本机 2026-11-19 实测结论：ale_bench 缺失、HF 不可达、无 API key → 端到端不可行，
但本地 held-out 三条件成立 → fuel_grade=LOCAL_HELD_OUT。这里把那次实测固化成
可复跑的判据。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

HF_DATA_ID = "SakanaAI/ALE-Bench"


def _check_import(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"item": f"import:{name}", "ok": spec is not None,
            "detail": "found" if spec else "not importable"}


def _check_hf() -> dict:
    url = f"https://huggingface.co/{HF_DATA_ID}/resolve/main/README.md"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            ok = 200 <= resp.status < 400
            return {"item": "hf_reachable", "ok": ok,
                    "detail": f"HTTP {resp.status}" if ok else f"HTTP {resp.status}"}
    except Exception as e:  # noqa: BLE001 - 探测要如实记录任何失败
        return {"item": "hf_reachable", "ok": False,
                "detail": f"unreachable: {type(e).__name__}"}


def _check_data_dir() -> dict:
    p = os.environ.get("ALE_BENCH_DATA", "").strip()
    if p and Path(p).is_dir():
        return {"item": "local_data_dir", "ok": True, "detail": p}
    return {"item": "local_data_dir", "ok": False,
            "detail": "ALE_BENCH_DATA unset or not a directory"}


def _check_docker() -> dict:
    import shutil
    exe = shutil.which("docker")
    if not exe:
        return {"item": "docker", "ok": False, "detail": "docker not on PATH"}
    import subprocess
    try:
        out = subprocess.run([exe, "version", "--format", "{{.Server.Version}}"],
                             capture_output=True, text=True, timeout=8)
        if out.returncode == 0 and out.stdout.strip():
            return {"item": "docker", "ok": True, "detail": out.stdout.strip()}
        return {"item": "docker", "ok": False, "detail": f"server unreachable (rc={out.returncode})"}
    except Exception as e:  # noqa: BLE001
        return {"item": "docker", "ok": False, "detail": f"{type(e).__name__}"}


def _check_keys() -> dict:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
            "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    present = {k: bool(os.environ.get(k)) for k in keys}
    return {"item": "api_keys", "ok": any(present.values()),
            "detail": {k: v for k, v in present.items()}}


def classify_fuel(ale_ok: bool, local_ok: bool) -> str:
    """燃料分级纯函数：ALE 全过用真源，否则本地 held-out 兜底，都没有才 NONE。"""
    if ale_ok:
        return "ALE_REAL"
    if local_ok:
        return "LOCAL_HELD_OUT"
    return "NONE"


def _check_local_held_out() -> dict:
    """本地 held-out 燃料三条件（不联网、只读验证 ContextualTask 本身）。

    1) seeds_disjoint：public(100) 与 private(99991) 生成的样本序列不同；
    2) features_independent_of_truth：改 true_w 不改 episodes() ⇒ 特征生成不读真值；
    3) truth_determines_reward：改 true_w 会改 evaluate() ⇒ 真值只决定奖励；
    4) instance_difficulty_varies：同 policy 下各 episode 奖励有方差。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from rsi_agent.core import HeuristicPolicy
        from rsi_agent.tasks import ContextualTask, TRUE_W

        pub = ContextualTask(seed=100)
        priv = ContextualTask(seed=99991)
        seeds_ok = pub.episodes() != priv.episodes()

        flipped_w = tuple(-w for w in TRUE_W)
        flipped = ContextualTask(seed=100, true_w=flipped_w)
        feat_ok = pub.episodes() == flipped.episodes()

        policy = HeuristicPolicy((1.0, 0.0, 0.0))
        truth_ok = pub.evaluate(policy).values != flipped.evaluate(policy).values

        vals = pub.evaluate(policy).values
        difficulty_ok = len(set(round(v, 6) for v in vals)) > 1

        ok = seeds_ok and feat_ok and truth_ok and difficulty_ok
        return {"item": "local_held_out", "ok": ok,
                "detail": {"seeds_disjoint": seeds_ok,
                           "features_independent_of_truth": feat_ok,
                           "truth_determines_reward": truth_ok,
                           "instance_difficulty_varies": difficulty_ok}}
    except Exception as e:  # noqa: BLE001
        return {"item": "local_held_out", "ok": False,
                "detail": f"local held-out probe failed: {type(e).__name__}: {e}"}


def check() -> dict:
    items = [_check_import("ale_bench"), _check_hf(), _check_data_dir(),
             _check_docker(), _check_keys()]
    ale_ok = all(i["ok"] for i in items)
    local = _check_local_held_out()
    return {"ready": ale_ok, "items": items + [local],
            "blocked": [i["item"] for i in items if not i["ok"]],
            "verdict": "READY" if ale_ok else "BLOCKED",
            "fuel_grade": classify_fuel(ale_ok, local["ok"])}


def selftest() -> int:
    cases = [
        ((True, True), "ALE_REAL"),
        ((False, True), "LOCAL_HELD_OUT"),
        ((False, False), "NONE"),
        ((True, False), "ALE_REAL"),
    ]
    for args, want in cases:
        got = classify_fuel(*args)
        assert got == want, f"classify_fuel{args}={got} want {want}"
    local = _check_local_held_out()
    assert local["ok"] is True, f"本地 held-out 三条件应成立: {local}"
    result = check()
    assert result["fuel_grade"] in {"ALE_REAL", "LOCAL_HELD_OUT", "NONE"}
    assert any(i["item"] == "local_held_out" for i in result["items"])
    print(f"selftest OK: classify 4 组 + local ok={local['ok']} "
          f"+ fuel_grade={result['fuel_grade']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--selftest":
        return selftest()
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["blocked"]:
        print(f"\n[blocked] 端到端不可行，缺: {', '.join(result['blocked'])}",
              file=sys.stderr)
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
