from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


ACTION_ACCEPT = "accept"
ACTION_CLARIFY_ACOUSTIC = "clarify_acoustic"
ACTION_CLARIFY_SEMANTIC = "clarify_semantic"
ACTION_CLARIFY_CONTEXT = "clarify_context"
ACTION_REJECT = "reject"


@dataclass(frozen=True)
class ValidationDecision:
    action: str
    reason: str
    confidence: float
    semantic_score: float
    intent: str | None = None


class InputValidator:
    def __init__(
        self,
        *,
        min_reject_confidence: float = 0.40,
        min_accept_confidence: float = 0.70,
        min_option_confidence: float = 0.50,
        min_option_similarity: float = 0.62,
    ) -> None:
        self.min_reject_confidence = float(min_reject_confidence)
        self.min_accept_confidence = float(min_accept_confidence)
        self.min_option_confidence = float(min_option_confidence)
        self.min_option_similarity = float(min_option_similarity)

    @staticmethod
    def estimate_confidence_from_text(text: str) -> float:
        raw = str(text or "").strip()
        if not raw:
            return 0.0

        compact = re.sub(r"\s+", " ", raw).strip().lower()
        tokens = re.findall(r"[a-z0-9\u00e4\u00f6\u00fc\u00df]+", compact)
        if not tokens:
            return 0.0

        if len(tokens) == 1 and len(tokens[0]) <= 2:
            return 0.28

        filler = {"aeh", "eh", "hm", "hmm", "mhm", "uh", "um", "mmm"}
        if set(tokens).issubset(filler):
            return 0.22

        greetings = {"hallo", "hello", "hey", "hi", "moin", "servus"}
        if len(tokens) <= 2 and set(tokens).issubset(greetings):
            return 0.35

        base = min(0.95, 0.42 + (0.09 * min(len(tokens), 6)))
        if any(len(token) > 18 for token in tokens):
            base -= 0.08
        if re.search(r"[^0-9a-zA-Z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df\s\.,!\?-]", raw):
            base -= 0.06

        return max(0.0, min(1.0, base))

    def classify_input(
        self,
        *,
        transcript: str,
        phase: str,
        expected_options: list[str] | None,
        semantic_matcher: Callable[[str, list[str]], tuple[str | None, float]],
        stt_confidence: float | None = None,
        reject_confidence_override: float | None = None,
        accept_confidence_override: float | None = None,
        option_confidence_override: float | None = None,
        option_similarity_override: float | None = None,
    ) -> ValidationDecision:
        _ = phase
        text = str(transcript or "").strip()
        if not text:
            return ValidationDecision(
                action=ACTION_REJECT,
                reason="empty",
                confidence=0.0,
                semantic_score=0.0,
                intent=None,
            )

        confidence = (
            self.estimate_confidence_from_text(text)
            if stt_confidence is None
            else max(0.0, min(1.0, float(stt_confidence)))
        )
        min_reject_confidence = (
            self.min_reject_confidence
            if reject_confidence_override is None
            else max(0.0, min(1.0, float(reject_confidence_override)))
        )
        min_accept_confidence = (
            self.min_accept_confidence
            if accept_confidence_override is None
            else max(0.0, min(1.0, float(accept_confidence_override)))
        )
        min_option_confidence = (
            self.min_option_confidence
            if option_confidence_override is None
            else max(0.0, min(1.0, float(option_confidence_override)))
        )
        min_option_similarity = (
            self.min_option_similarity
            if option_similarity_override is None
            else max(0.0, min(1.0, float(option_similarity_override)))
        )

        if confidence < min_reject_confidence:
            return ValidationDecision(
                action=ACTION_REJECT,
                reason="low_confidence",
                confidence=confidence,
                semantic_score=0.0,
                intent=None,
            )

        normalized = text.lower()
        tokens = re.findall(r"[a-z0-9\u00e4\u00f6\u00fc\u00df]+", normalized)
        filler = {"aeh", "eh", "hm", "hmm", "mhm", "uh", "um", "mmm"}
        if tokens and set(tokens).issubset(filler):
            return ValidationDecision(
                action=ACTION_CLARIFY_ACOUSTIC,
                reason="filler_only",
                confidence=confidence,
                semantic_score=0.0,
                intent=None,
            )

        if len(text) < 2 and normalized not in {"ja", "nein"}:
            return ValidationDecision(
                action=ACTION_CLARIFY_ACOUSTIC,
                reason="too_short",
                confidence=confidence,
                semantic_score=0.0,
                intent=None,
            )

        options = [str(option).strip().lower() for option in (expected_options or []) if str(option).strip()]
        if options:
            matched, similarity = semantic_matcher(text, options)
            if (
                matched
                and confidence >= min_option_confidence
                and similarity >= min_option_similarity
            ):
                return ValidationDecision(
                    action=ACTION_ACCEPT,
                    reason="option_match",
                    confidence=confidence,
                    semantic_score=similarity,
                    intent=matched,
                )
            if confidence < min_accept_confidence:
                return ValidationDecision(
                    action=ACTION_CLARIFY_ACOUSTIC,
                    reason="option_low_confidence",
                    confidence=confidence,
                    semantic_score=similarity,
                    intent=None,
                )
            return ValidationDecision(
                action=ACTION_CLARIFY_SEMANTIC,
                reason="option_mismatch",
                confidence=confidence,
                semantic_score=similarity,
                intent=None,
            )

        if confidence >= min_accept_confidence:
            return ValidationDecision(
                action=ACTION_ACCEPT,
                reason="free_text_confident",
                confidence=confidence,
                semantic_score=0.0,
                intent=text,
            )

        return ValidationDecision(
            action=ACTION_CLARIFY_CONTEXT,
            reason="free_text_unclear",
            confidence=confidence,
            semantic_score=0.0,
            intent=None,
        )
