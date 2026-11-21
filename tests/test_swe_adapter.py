"""SWE-bench 适配器反证测试：接线正确 + 缺环 fail-closed。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi_agent.core import HeuristicPolicy
from rsi_agent.swe_adapter import (
    SWEBenchAdapter, SWEBenchUnavailable, fixed_code_gen, mock_session,
)
from scripts.check_swe_readiness import classify_fuel


class TestSWEBenchAdapter(unittest.TestCase):
    def test_missing_code_gen_raises(self):
        a = SWEBenchAdapter(session=mock_session((1.0,)))
        with self.assertRaises(SWEBenchUnavailable):
            a.evaluate(HeuristicPolicy((1.0, 0.0, 0.0)))

    def test_missing_session_raises_on_score(self):
        a = SWEBenchAdapter(code_gen=fixed_code_gen())
        with self.assertRaises(SWEBenchUnavailable):
            a.evaluate(HeuristicPolicy((1.0, 0.0, 0.0)))

    def test_mock_wiring(self):
        a = SWEBenchAdapter(instance_ids=("i1", "i2"),
                            code_gen=fixed_code_gen(),
                            session=mock_session((1.0, 0.0)))
        sv = a.evaluate(HeuristicPolicy((0.9, -0.6, 1.2)))
        self.assertEqual(sv.values, (1.0, 0.0, 1.0, 0.0))

    def test_availability_is_bool(self):
        self.assertIsInstance(SWEBenchAdapter.available(), bool)

    def test_fuel_classify(self):
        self.assertEqual(classify_fuel(True, True), "SWE_REAL")
        self.assertEqual(classify_fuel(False, True), "LOCAL_HELD_OUT")
        self.assertEqual(classify_fuel(False, False), "NONE")


if __name__ == "__main__":
    unittest.main()
