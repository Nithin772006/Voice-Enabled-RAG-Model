import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guardrails.off_topic import OffTopicGuardrail
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.grounding import GroundingValidator

class TestGuardrails(unittest.TestCase):

    def test_off_topic_guardrail(self):
        guard = OffTopicGuardrail()
        is_safe, msg = guard.evaluate("UNSAFE")
        self.assertFalse(is_safe)
        self.assertIn("பாதுகாப்பற்றது", msg)

        is_safe_ood, msg_ood = guard.evaluate("OUT_OF_DOMAIN")
        self.assertFalse(is_safe_ood)

        is_ok, _ = guard.evaluate("IN_DOMAIN")
        self.assertTrue(is_ok)

    def test_confidence_guardrail(self):
        guard = ConfidenceGuardrail(threshold=0.60)
        top_cand = [{"text": "இந்தியாவின் தலைநகரம் புது தில்லி ஆகும்.", "vector_score": 0.85, "reranker_score": 0.90}]
        is_passed, score, signals = guard.evaluate(top_cand, "தலைநகரம்")
        self.assertTrue(is_passed)
        self.assertGreaterEqual(score, 0.60)

    def test_grounding_validator(self):
        validator = GroundingValidator()
        context = [{"text": "புது தில்லி இந்தியாவின் தலைநகரம் ஆகும்."}]
        answer = "புது தில்லி தலைநகரம்"
        is_grounded, ratio, g_case = validator.evaluate(answer, context)
        self.assertTrue(is_grounded)
        self.assertGreater(ratio, 0.30)
        self.assertEqual(g_case, "CASE_A_GROUNDED")

if __name__ == "__main__":
    unittest.main()
