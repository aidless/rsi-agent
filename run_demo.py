"""CLI：跑一次 RSI 循环，打印诚实判决 + 写 JSON 结果。

用法：
    python3 run_demo.py                       # 默认 12 代
    python3 run_demo.py --generations 20 --out results/demo.json
    python3 run_demo.py --self-test           # 跑全套反证/负对照
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rsi_agent.core import HeuristicPolicy
from rsi_agent.gates import paired_gate
from rsi_agent.runner import run_demo
from rsi_agent.tasks import BinPackTask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--public-seed", type=int, default=100)
    ap.add_argument("--private-seed", type=int, default=99991)
    ap.add_argument("--inner-seed", type=int, default=7)
    ap.add_argument("--nc-probes", type=int, default=0,
                    help="NC 前置审计的 probe 数；0=关闭（默认）")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
        ok = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        return 0 if ok else 1

    result = run_demo(generations=args.generations, public_seed=args.public_seed,
                      private_seed=args.private_seed, inner_seed=args.inner_seed,
                      nc_probes=args.nc_probes)
    payload = result.as_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n[written] {args.out}", file=sys.stderr)
    return 0 if result.verdict == "SELF_IMPROVEMENT_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
