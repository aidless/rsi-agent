"""SWE-bench Verified 接入前置检查（只读探测，不安装、不联网下载）。

在把 rsi-agent 接到 SWE-bench Verified 500 之前，先回答"这一格能不能跑开"。
检查项与判定（fail-closed）：

- 依赖包 swebench 可导入；
- 数据可用：HF 可达 或 SWE_BENCH_DATA 指向本地目录；
- Docker 可用（SWE-bench 每题在容器里跑 F2P/P2P 测试）；
- GITHUB_TOKEN（SWE-bench 拉 repo/建镜像要 GitHub 访问）。

fuel_grade 三态（与 check_ale_readiness.py 同构）：
- SWE_REAL：上面四项全过，能用真 SWE-bench 端到端；
- LOCAL_HELD_OUT：SWE 项不全过，但本地 ContextualTask 满足 held-out 三条件；
- NONE：两者都不满足。

本机 2026-11-19 实测：swebench 缺失、HF 超时、GITHUB_TOKEN unset、docker 可用 ⇒
端到端不可行，但本地 held-out 成立 ⇒ fuel_grade=LOCAL_HELD_OUT。
退出码：0=READY，1=BLOCKED（fuel_grade 另报本地替代燃料），2=用法错。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

HF_DATA_ID = "princeton-nlp/SWE-bench_Verified"
HF_PARQUET = "data/test-00000-of-00001.parquet"


def _check_import(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"item": f"import:{name}", "ok": spec is not None,
            "detail": "found" if spec else "not importable"}


def _check_hf() -> dict:
    url = f"https://huggingface.co/{HF_DATA_ID}/resolve/main/{HF_PARQUET}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return {"item": "hf_reachable", "ok": 200 <= resp.status < 400,
                    "detail": f"HTTP {resp.status}"}
    except Exception as e:  # noqa: BLE001
        return {"item": "hf_reachable", "ok": False,
                "detail": f"unreachable: {type(e).__name__}"}


def _check_data_dir() -> dict:
    p = os.environ.get("SWE_BENCH_DATA", "").strip()
    if p and Path(p).is_dir():
        return {"item": "local_data_dir", "ok": True, "detail": p}
    return {"item": "local_data_dir", "ok": False,
            "detail": "SWE_BENCH_DATA unset or not a directory"}


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
        return {"item": "docker", "ok": False,
                "detail": f"server unreachable (rc={out.returncode})"}
    except Exception as e:  # noqa: BLE001
        return {"item": "docker", "ok": False, "detail": type(e).__name__}


def _check_github_token() -> dict:
    present = bool(os.environ.get("GITHUB_TOKEN"))
    return {"item": "github_token", "ok": present,
            "detail": "SET" if present else "unset"}


def classify_fuel(swe_ok: bool, local_ok: bool) -> str:
    """纯函数：SWE 全过用真源，否则本地 held-out 兜底，都没有才 NONE。"""
    if swe_ok:
        return "SWE_REAL"
    if local_ok:
        return "LOCAL_HELD_OUT"
    return "NONE"


def _check_local_held_out() -> dict:
    """本地 held-out 燃料四条件（复用 ALE 的同一判据，不联网）：见 check_ale_readiness.py。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from rsi_agent.core import HeuristicPolicy
        from rsi_agent.tasks import ContextualTask, TRUE_W

        pub = ContextualTask(seed=100)
        priv = ContextualTask(seed=99991)
        seeds_ok = pub.episodes() != priv.episodes()
        flipped = ContextualTask(seed=100, true_w=tuple(-w for w in TRUE_W))
        feat_ok = pub.episodes() == flipped.episodes()
        policy = HeuristicPolicy((1.0, 0.0, 0.0))
        truth_ok = pub.evaluate(policy).values != flipped.evaluate(policy).values
        difficulty_ok = len(set(round(v, 6) for v in pub.evaluate(policy).values)) > 1
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
    items = [_check_import("swebench"), _check_hf(), _check_data_dir(),
             _check_docker(), _check_github_token()]
    swe_ok = all(i["ok"] for i in items)
    local = _check_local_held_out()
    return {"ready": swe_ok, "items": items + [local],
            "blocked": [i["item"] for i in items if not i["ok"]],
            "verdict": "READY" if swe_ok else "BLOCKED",
            "fuel_grade": classify_fuel(swe_ok, local["ok"])}


def selftest() -> int:
    for args, want in [((True, True), "SWE_REAL"), ((False, True), "LOCAL_HELD_OUT"),
                       ((False, False), "NONE"), ((True, False), "SWE_REAL")]:
        assert classify_fuel(*args) == want, f"classify_fuel{args} != {want}"
    local = _check_local_held_out()
    assert local["ok"] is True, f"本地 held-out 三条件应成立: {local}"
    print(f"selftest OK: classify 4 组 + local ok={local['ok']}")
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
