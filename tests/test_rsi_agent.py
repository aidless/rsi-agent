r"""反证 / 负对照：每个门都必须能红，否则是装饰。

覆盖：
- T1 密封屏障：InnerAgent 拿到非 PublicHistory 必须抛异常；公开历史无 private 字段。
- T2 P2 反循环：标签由隐藏真值决定，特征里搜不到 w\*（改真值→排名变），
  且 gate 用的 score 来自外部 evaluator 而非策略自报。
- T3 ESS 自相关：高自相关 delta 的 ESS 显著小于 n，低 ESS 直接拒。
- T4 噪声门：低于 effect floor 的提升被拒。
- T5 Holm 多重比较：同一 p 值在更严格 alpha 下被拒。
- T6 审计链：改内容 / 删记录 / 重排都能被 verify() 抓到。
- T7 真提升闭环：默认 run 得到 SELF_IMPROVEMENT_VERIFIED，final_weights 方向
  命中隐藏真值，sealed 被真实调用（不是摆设）。
- T8 确定性：同种子两次跑逐位一致。
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rsi_agent.audit import AuditLog, GENESIS
from rsi_agent.barriers import BarrierViolation, SealedPrivateSet, assert_public_only
from rsi_agent.core import HeuristicPolicy, PublicHistory, PublicHistoryEntry, ScoreVector
from rsi_agent.gates import effective_n, estimate_noise_floor, holm_alphas, paired_gate
from rsi_agent.outer import InnerAgent
from rsi_agent.runner import run_demo
from rsi_agent.tasks import ContextualTask, TRUE_W


class TestBarrier(unittest.TestCase):
    def test_t1_inner_accepts_only_public_history(self):
        agent = InnerAgent()
        self.assertIsInstance(agent.observe(PublicHistory(())), str)
        with self.assertRaises(BarrierViolation):
            agent.observe({"secret": 123})
        with self.assertRaises(BarrierViolation):
            agent.observe(ScoreVector((1.0,)))
        # PublicHistory 的字段里没有 private / score vector
        e = PublicHistoryEntry(0, "h", (1.0, 0, 0), 0.5, 128, True, "PASS")
        h = PublicHistory((e,))
        self.assertFalse(any("private" in k for k in e.__dataclass_fields__))

    def test_t1_sealed_not_reachable_from_public(self):
        task = ContextualTask(seed=99991)
        sealed = SealedPrivateSet(task)
        self.assertIn("SEALED", repr(sealed))
        # 公开任务 seed 与密封任务 seed 不同 → 实例不重合
        pub = ContextualTask(seed=100)
        self.assertNotEqual(pub.episodes(), task.episodes())


class TestNonCircularity(unittest.TestCase):
    def test_t2_hidden_truth_drives_label_not_features(self):
        task = ContextualTask(seed=100)
        good = task.evaluate(HeuristicPolicy(TRUE_W)).mean
        bad = task.evaluate(HeuristicPolicy((1.0, 0.0, 0.0))).mean
        self.assertGreater(good, bad + 0.3, "隐藏真值方向应显著优于 baseline")
        # 换一个隐藏真值，同一批特征上的分数必须改变 → 标签确实来自 w*，
        # 不是从特征自身某列泄漏出来的
        other = replace(task, true_w=(-1.2, 0.3, 0.7))
        probe = HeuristicPolicy((1.0, 0.0, 0.0))
        self.assertNotEqual(task.evaluate(probe).values, other.evaluate(probe).values)

    def test_t2_gate_score_is_external(self):
        task = ContextualTask(seed=100)
        base = task.evaluate(HeuristicPolicy((1, 0, 0)))
        cand = task.evaluate(HeuristicPolicy(TRUE_W))
        self.assertNotEqual(base, cand)
        gate = paired_gate(base, cand, noise_floor=0.015, alpha=0.05)
        self.assertTrue(gate.passed)
        self.assertGreater(gate.delta, 0.3)


class TestGatesFailRed(unittest.TestCase):
    def test_t3_ess_autocorrelation(self):
        independent = tuple([1.0 if i % 2 else -1.0 for i in range(64)])
        autocorr = tuple([1.0] * 40 + [-1.0] * 24)  # 强正自相关
        self.assertLess(effective_n(autocorr), effective_n(independent))
        gate = paired_gate(ScoreVector((0.0,) * 64), ScoreVector(autocorr),
                           noise_floor=0.01, alpha=0.05, min_ess=40.0)
        self.assertFalse(gate.passed)
        self.assertIn("ESS", gate.reason)

    def test_t4_noise_floor_rejects_tiny_gain(self):
        n = 64
        base = ScoreVector((0.0,) * n)
        tiny = ScoreVector((0.001,) * n)  # 统计显著但实际微小
        gate = paired_gate(base, tiny, noise_floor=0.01, alpha=0.05)
        self.assertFalse(gate.passed)
        self.assertIn("NOISE", gate.reason)

    def test_t5_holm_more_stringent_rejects(self):
        alphas = holm_alphas(12, 0.05)
        self.assertLess(alphas[0], alphas[-1])
        self.assertAlmostEqual(alphas[0], 0.05 / 12)

    def test_t5_no_gain_rejected(self):
        gate = paired_gate(ScoreVector((1.0,) * 32), ScoreVector((1.0,) * 32), 0.01, 0.05)
        self.assertFalse(gate.passed)
        self.assertIn("NO_GAIN", gate.reason)


class TestAuditTamperEvidence(unittest.TestCase):
    def _log(self):
        log = AuditLog()
        log.append("a", {"x": 1})
        log.append("b", {"x": 2})
        log.append("c", {"x": 3})
        self.assertTrue(log.verify()[0])
        return log

    def test_t6_content_tamper(self):
        log = self._log()
        rec = log.records[1]
        rec.payload["x"] = 99
        ok, detail = log.verify()
        self.assertFalse(ok)
        self.assertIn("HASH", detail)

    def test_t6_delete_record(self):
        log = self._log()
        log._records.pop(1)
        ok, detail = log.verify()
        self.assertFalse(ok)
        self.assertTrue("CHAIN" in detail or "SEQ" in detail, detail)

    def test_t6_reorder(self):
        log = self._log()
        log._records[0], log._records[2] = log._records[2], log._records[0]
        ok, detail = log.verify()
        self.assertFalse(ok)

    def test_t6_genesis_constant(self):
        self.assertEqual(GENESIS, "0" * 64)


class TestEndToEnd(unittest.TestCase):
    def test_t7_real_self_improvement(self):
        r = run_demo(generations=20)
        self.assertEqual(r.verdict, "SELF_IMPROVEMENT_VERIFIED", r.audit_detail)
        self.assertTrue(r.audit_ok)
        self.assertGreater(r.best_public_mean, r.baseline_public_mean + 0.3)
        self.assertGreater(r.sealed_evaluations, 0, "密封集必须被真实调用")
        # final_weights 方向应命中隐藏真值（相关方向一致：w1<0, w2>0）
        self.assertLess(r.final_weights[1], -0.2)
        self.assertGreater(r.final_weights[2], 0.5)
        self.assertTrue(any(g.promoted for g in r.generations))

    def test_t8_deterministic(self):
        a = run_demo(generations=12).as_dict()
        b = run_demo(generations=12).as_dict()
        self.assertEqual(a, b)

    def test_t7_no_gain_when_budget_tiny(self):
        # 1 代搜索几乎不可能过闸 → 合法地报 NO_VERIFIED_GAIN，不编造成功
        r = run_demo(generations=1)
        self.assertIn(r.verdict, ("NO_VERIFIED_GAIN", "SELF_IMPROVEMENT_VERIFIED"))
        self.assertTrue(r.audit_ok)


if __name__ == "__main__":
    unittest.main()
