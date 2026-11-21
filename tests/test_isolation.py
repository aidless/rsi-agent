r"""进程隔离升级验收：V1–V4，全过才算隔离真正成立。

设计：私有集 task 只在 worker 子进程内构造，主进程经 Pipe 只传 policy、收
ScoreVector。主进程内存里没有 task 实例，gc 扫不到。

- V1 正确性：隔离版与进程内版在同一 seed/同一 policy 上 evaluate 逐位一致。
- V2 真隔离：隔离版构造后，主进程 gc.get_objects() 里没有 BinPackTask 实例。
- V4 门能红（反证）：进程内版 SealedPrivateSet 构造后，主进程 gc 能扫到——
  证明 V2 的判据确实能区分两种实现，不是恒假。
- V3 端到端无回归：run_demo(isolated=True) 仍 SELF_IMPROVEMENT_VERIFIED，且
  sealed 调用次数与进程内版一致。

TDD：本文件先于实现写入，实现未存在时 import 即红。
"""
from __future__ import annotations

import gc
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rsi_agent.barriers import SealedPrivateSet
from rsi_agent.core import HeuristicPolicy
from rsi_agent.isolated_sealed import IsolatedSealedPrivateSet
from rsi_agent.tasks import BinPackTask


def _scan_tasks():
    return [o for o in gc.get_objects() if isinstance(o, BinPackTask)]


class TestIsolation(unittest.TestCase):
    def test_v1_isolated_matches_process_local(self):
        """同 seed 同 policy，隔离版与进程内版逐位一致。"""
        seed = 424242
        iso = IsolatedSealedPrivateSet(seed=seed)
        try:
            loc = SealedPrivateSet(BinPackTask(seed=seed))
            p = HeuristicPolicy((0.7, -0.3, 1.1))
            a = iso.evaluate(p)
            b = loc.evaluate(p)
            self.assertEqual(list(a.values), list(b.values))
            self.assertEqual(len(a), len(b))
        finally:
            iso.close()

    def test_v2_main_process_has_no_task_instance(self):
        """隔离版构造前后，主进程 BinPackTask 实例集合不变（不新增）。

        用增量而非绝对计数：同一测试进程里其它用例的进程内版会把 task 留在
        模块级 _SEALED 字典里，那是它们的遗留、不是隔离版引入的。隔离成立的
        语义是"隔离版不往主进程添实例"。
        """
        gc.collect()
        before = {id(o) for o in _scan_tasks()}
        iso = IsolatedSealedPrivateSet(seed=777)
        try:
            gc.collect()
            after = {id(o) for o in _scan_tasks()}
            self.assertEqual(after - before, set(),
                             "隔离版构造后主进程新增了私有集实例，隔离未成立")
        finally:
            iso.close()

    def test_v4_process_local_breaks_isolation_detection(self):
        """进程内版能被 gc 扫到 —— 证明 V2 判据能区分，不是恒假。"""
        local = SealedPrivateSet(BinPackTask(seed=555))
        found = [o for o in gc.get_objects() if isinstance(o, BinPackTask)]
        self.assertTrue(found, "进程内版本应能被扫到；扫不到说明 V2 判据失能")
        del local

    def test_v3_demo_with_isolated_no_regression(self):
        """隔离版端到端仍 SELF_IMPROVEMENT_VERIFIED，调用次数与进程内版一致。"""
        from rsi_agent.runner import RSIRunner

        iso_runner = RSIRunner(isolated=True, public_seed=100, private_seed=99991,
                               generations=12, inner_seed=7)
        r_iso = iso_runner.run()
        iso_runner.sealed.close()

        loc_runner = RSIRunner(public_seed=100, private_seed=99991,
                               generations=12, inner_seed=7)
        r_loc = loc_runner.run()

        self.assertEqual(r_iso.verdict, "SELF_IMPROVEMENT_VERIFIED", r_iso.audit_detail)
        self.assertEqual(r_loc.verdict, "SELF_IMPROVEMENT_VERIFIED", r_loc.audit_detail)
        self.assertEqual(r_iso.sealed_evaluations, r_loc.sealed_evaluations,
                         "隔离版与进程内版 sealed 调用次数不一致")
        self.assertEqual(list(r_iso.final_weights), list(r_loc.final_weights))


if __name__ == "__main__":
    unittest.main(verbosity=2)
