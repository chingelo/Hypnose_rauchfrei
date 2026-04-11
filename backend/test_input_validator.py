from __future__ import annotations

import unittest

from input_validator import (
    ACTION_ACCEPT,
    ACTION_CLARIFY_ACOUSTIC,
    ACTION_CLARIFY_SEMANTIC,
    ACTION_REJECT,
    InputValidator,
)


def _matcher(text: str, options: list[str]) -> tuple[str | None, float]:
    normalized = str(text or "").lower()
    for option in options:
        if str(option).lower() in normalized:
            return option, 0.95
    return None, 0.10


class InputValidatorTests(unittest.TestCase):
    def test_rejects_low_confidence_noise(self) -> None:
        validator = InputValidator()
        decision = validator.classify_input(
            transcript="hm",
            stt_confidence=0.20,
            expected_options=["hell", "dunkel"],
            phase="4",
            semantic_matcher=_matcher,
        )
        self.assertEqual(decision.action, ACTION_REJECT)

    def test_accepts_option_match_with_sufficient_confidence(self) -> None:
        validator = InputValidator()
        decision = validator.classify_input(
            transcript="Es ist eher hell.",
            stt_confidence=0.78,
            expected_options=["hell", "dunkel"],
            phase="4",
            semantic_matcher=_matcher,
        )
        self.assertEqual(decision.action, ACTION_ACCEPT)
        self.assertEqual(decision.intent, "hell")

    def test_semantic_clarify_on_option_mismatch(self) -> None:
        validator = InputValidator()
        decision = validator.classify_input(
            transcript="Das ist kompliziert.",
            stt_confidence=0.81,
            expected_options=["hell", "dunkel"],
            phase="4",
            semantic_matcher=_matcher,
        )
        self.assertEqual(decision.action, ACTION_CLARIFY_SEMANTIC)

    def test_acoustic_clarify_for_uncertain_free_text(self) -> None:
        validator = InputValidator()
        decision = validator.classify_input(
            transcript="mhm",
            stt_confidence=0.45,
            expected_options=[],
            phase="3",
            semantic_matcher=_matcher,
        )
        self.assertEqual(decision.action, ACTION_CLARIFY_ACOUSTIC)

    def test_accept_threshold_can_be_overridden_per_phase(self) -> None:
        validator = InputValidator(min_accept_confidence=0.70)
        decision_default = validator.classify_input(
            transcript="ruhig",
            stt_confidence=0.66,
            expected_options=[],
            phase="2",
            semantic_matcher=_matcher,
        )
        decision_override = validator.classify_input(
            transcript="ruhig",
            stt_confidence=0.66,
            expected_options=[],
            phase="2",
            semantic_matcher=_matcher,
            accept_confidence_override=0.62,
        )
        self.assertNotEqual(decision_default.action, ACTION_ACCEPT)
        self.assertEqual(decision_override.action, ACTION_ACCEPT)


if __name__ == "__main__":
    unittest.main()
