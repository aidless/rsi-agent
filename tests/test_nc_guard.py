"""NC 前置 model-free 审计闸的负对照/反证测试。

审计对象是被审 scorer 在 probe grid 上的表现：
- selection_error：真值有序时是否出现严格反转（代理同分 = 中性，不算反转，
  这样饱和轴才能独立被测出来）；
- signed_gap：代理分相对真值的裸均值差（正 = 乐观；用裸差而非 min-max，
  因为 min-max 会抹掉均匀抬升这种典型 padding 攻击）；
- saturation：代理分的重复比例（粗粒度/常数代理会让选择看起来"零错误"）。

verifier：每个断言都必须对应一条可复算的量（返回码 + 字段值），不靠感觉。
"""
import unittest

from rsi_agent.nc_guard import NCAuditResult, Probe, model_free_audit
from rsi_agent.runner import RSIRunner


def _grid(n=12, lo=0.30, hi=0.90):
    """真值受控的 probe grid：真值严格递增、等距、互不相同。"""
    step = (hi - lo) / (n - 1)
    return [Probe(weights=(1.0, 0.0, 0.0), true_quality=lo + i * step) for i in range(n)]


def _scores(probes, fn):
    return [fn(p.true_quality) for p in probes]


class TestNCGuard(unittest.TestCase):
    def test_t1_honest_scorer_passes(self):
        probes = _grid()
        r = model_free_audit(probes, lambda ps: _scores(ps, lambda q: q))
        self.assertTrue(r.passed, r.reason)
        self.assertIn("PASS", r.reason)
        self.assertEqual(r.selection_error, 0.0)
        self.assertEqual(r.signed_gap, 0.0)
        self.assertEqual(r.saturation, 0.0)

    def test_t2_optimistic_padding_rejected(self):
        # 常数抬升 +0.27 不改变序（se=0），但裸均值差 +0.27 > gap_tol=0 → 拒
        probes = _grid()
        r = model_free_audit(probes, lambda ps: _scores(ps, lambda q: q + 0.27))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason.split(":")[0], "OPTIMISTIC_GAP")
        self.assertAlmostEqual(r.signed_gap, 0.27, places=6)

    def test_t3_random_proxy_selection_corrupted(self):
        # 确定性伪随机（无外部依赖）：交替高低制造大量严格反转，se≈0.5 > 0.15 → 拒
        probes = _grid(n=12)
        seq = [0.9 if i % 2 == 0 else 0.1 for i in range(12)]
        r = model_free_audit(probes, lambda ps: seq)
        self.assertFalse(r.passed)
        self.assertEqual(r.reason.split(":")[0], "SELECTION_CORRUPTED")
        self.assertGreater(r.selection_error, 0.15)

    def test_t4_constant_proxy_saturation_starvation(self):
        # 全同代理：真值有序但代理无区分 → se=0（同分中性）、gap=0（取真值均值）、
        # saturation=1-1/n → 拒。若把同分算成反转，这一例会被 se 抢先判，测不出饱和轴。
        probes = _grid(n=12)
        c = sum(p.true_quality for p in probes) / len(probes)
        r = model_free_audit(probes, lambda ps: [c] * len(ps))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason.split(":")[0], "SATURATION_STARVATION")
        self.assertEqual(r.selection_error, 0.0)
        self.assertAlmostEqual(r.saturation, 1.0 - 1.0 / 12, places=6)

    def test_t5_coarse_rounding_saturation_independent_of_se(self):
        # 单调粗粒度 round(true, 1) 在密集真值上：不产生严格反转（se=0），
        # 但 distinct 代理值塌到 2 个 → saturation 高 → 拒。
        # gap_tol=0.01 隔离舍入带来的微小均值漂移，确保拒绝原因是饱和而非乐观。
        probes = _grid(n=12, lo=0.501, hi=0.510)
        r = model_free_audit(probes, lambda ps: _scores(ps, lambda q: round(q, 1)),
                             gap_tol=0.01)
        self.assertFalse(r.passed)
        self.assertEqual(r.reason.split(":")[0], "SATURATION_STARVATION")
        self.assertEqual(r.selection_error, 0.0)
        self.assertGreater(r.saturation, 0.5)

    def test_t6_insufficient_probes(self):
        probes = _grid(n=3)
        r = model_free_audit(probes, lambda ps: _scores(ps, lambda q: q))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason.split(":")[0], "INSUFFICIENT_PROBES")
        self.assertEqual(r.n_probes, 3)

    def test_t7_boundary_se_equal_max_passes(self):
        # n=8 → 28 个可比真值对；反转前 2 个（prefix 反转）→ 恰好 1 个严格反转。
        # se = 1/28；阈值设成同一浮点 → 严格 > 不成立 → PASS。反向钉死边界语义。
        probes = _grid(n=8)
        seq = list(reversed([p.true_quality for p in probes[:2]])) + [p.true_quality for p in probes[2:]]
        boundary = 1 / 28
        r = model_free_audit(probes, lambda ps: seq, selection_error_max=boundary)
        self.assertTrue(r.passed, r.reason)
        self.assertAlmostEqual(r.selection_error, boundary, places=12)

    def test_t8_result_as_dict_is_serializable(self):
        r = NCAuditResult(True, "PASS test", 0.0, 0.0, 0.0, 8)
        d = r.as_dict()
        self.assertEqual(d["reason"], "PASS test")
        self.assertEqual(d["n_probes"], 8)
        self.assertIsInstance(d["selection_error"], float)

    def test_t9_honest_default_scorer_does_not_block_demo(self):
        # 诚实默认 scorer（真值本身）在 60 个 probe 上过闸，demo 仍应验证成功。
        r = RSIRunner(generations=60, nc_probes=60).run()
        self.assertIsNotNone(r.nc_audit)
        self.assertTrue(r.nc_audit.passed, r.nc_audit.reason)
        self.assertEqual(r.verdict, "SELF_IMPROVEMENT_VERIFIED", r.verdict)
        self.assertTrue(r.audit_ok, r.audit_detail)

    def test_t10_poisoned_scorer_fails_closed_before_loop(self):
        # 被黑的 scorer 对真值做常数 padding → OPTIMISTIC_GAP → 提前失败，
        # 且 sealed 从未求值（not probed），审计链仍完整。
        r = RSIRunner(generations=20, nc_probes=60,
                      nc_scorer=lambda ps: [p.true_quality + 0.5 for p in ps]).run()
        self.assertEqual(r.verdict, "NC_AUDIT_FAILED", r.verdict)
        self.assertFalse(r.nc_audit.passed)
        self.assertIn("OPTIMISTIC_GAP", r.nc_audit.reason)
        self.assertEqual(r.sealed_evaluations, 0)
        self.assertEqual(len(r.generations), 0)
        self.assertTrue(r.audit_ok, r.audit_detail)


if __name__ == "__main__":
    unittest.main()
