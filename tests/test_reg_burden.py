import unittest
from rsi_agent.core import ScoreVector
from rsi_agent.gates import paired_gate


class TestRegBurden(unittest.TestCase):
    def test_t9_regression_burden_rejects_mean_positive_but_instance_crushing(self):
        # 大样本负担型攻击：190 实例 +0.05、10 实例 -0.5 均匀交错。
        # 均值 +0.0225、ess=200、p=0.004、nratio=2.25 全过旧闸，但 10 个实例
        # 各回归 -0.5、负担 0.526 > 0.25 → 必须拒（旧闸放行 = 漏洞本体）。
        d = tuple(-0.5 if i % 20 == 19 else 0.05 for i in range(200))
        b = ScoreVector((0.0,) * 200)
        c = ScoreVector(d)
        gate = paired_gate(b, c, noise_floor=0.01, alpha=0.05)
        self.assertFalse(gate.passed)
        self.assertIn("BURDEN", gate.reason)

    def test_t9_true_improvement_not_rejected(self):
        # 真提升形态：18 实例 +0.05、2 实例 -0.01 交错，负担 0.022 低 → 应过
        d = tuple(-0.01 if (i % 10 == 9 and i < 20) else 0.05 for i in range(20))
        b = ScoreVector((0.0,) * 20)
        c = ScoreVector(d)
        gate = paired_gate(b, c, noise_floor=0.01, alpha=0.05)
        self.assertTrue(gate.passed, gate.reason)


if __name__ == "__main__":
    unittest.main()
