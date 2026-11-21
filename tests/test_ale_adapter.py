"""ALE-Bench 适配器的 mock 冒烟测试（不联网、不装 ale_bench、无 API key）。

verifier：真实接口已核（ale_bench.start / session.public_eval(...).overall_absolute_score），
这里用注入的假 session + 假 code_gen 验证适配层的接线正确，并把"缺环必须失败"
钉成反证，杜绝"看起来跑通实际空转"。
"""
import unittest

from rsi_agent.ale_adapter import (
    ALEBenchAdapter,
    ALEBenchUnavailable,
    fixed_code_gen,
    mock_session,
)
from rsi_agent.core import HeuristicPolicy


class _FakeResult:
    def __init__(self, score):
        self.overall_absolute_score = score


class _RecordingSession:
    """按调用顺序返回预置分数，并记录收到的代码。"""
    def __init__(self, scores):
        self.scores = list(scores)
        self.codes = []

    def public_eval(self, code, code_language="python"):
        self.codes.append((code, code_language))
        return _FakeResult(self.scores[(len(self.codes) - 1) % len(self.scores)])


class TestALEAdapter(unittest.TestCase):
    def test_a1_missing_code_gen_fails_closed(self):
        adapter = ALEBenchAdapter(session=_RecordingSession([0.5] * 8))
        with self.assertRaises(ALEBenchUnavailable):
            adapter.evaluate(HeuristicPolicy((1.0, 0.0, 0.0)))

    def test_a2_mock_smoke_scores_wire_to_scorevector(self):
        scores = tuple(0.1 * i for i in range(8))
        adapter = ALEBenchAdapter(n_cases=8, code_gen=fixed_code_gen(),
                                  session=_RecordingSession(scores))
        sv = adapter.evaluate(HeuristicPolicy((1.0, 0.0, 0.0)))
        self.assertEqual(sv.values, scores)
        self.assertAlmostEqual(sv.mean, sum(scores) / 8)

    def test_a3_code_language_is_python(self):
        sess = _RecordingSession([0.5] * 8)
        adapter = ALEBenchAdapter(n_cases=8, code_gen=fixed_code_gen(), session=sess)
        adapter.evaluate(HeuristicPolicy((0.5, -0.5, 1.0)))
        self.assertTrue(sess.codes)
        self.assertTrue(all(lang == "python" for _, lang in sess.codes))

    def test_a4_weight_change_changes_generated_code(self):
        gen = fixed_code_gen()
        c1 = gen((1.0, 0.0, 0.0))[0]
        c2 = gen((0.0, 1.0, 0.0))[0]
        self.assertNotEqual(c1, c2)

    def test_a5_availability_check_is_readonly(self):
        # 只断言 availability 是可调用的 bool，不假设本机是否装了 ale_bench
        self.assertIsInstance(ALEBenchAdapter.available(), bool)


if __name__ == "__main__":
    unittest.main()
