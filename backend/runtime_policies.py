from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeoutPolicy:
    open_response_default_s: int = 18
    yes_no_default_s: int = 14
    scale_default_s: int = 18
    phase4_first_question_s: int = 14
    phase4_followup_s: int = 16
    inactivity_auto_end_s: int = 600


@dataclass(frozen=True)
class RetryPolicy:
    yes_no_max: int = 2
    scale_max: int = 3
    open_max: int = 1
    phase4_core_max: int = 2


TIMEOUT_POLICY = TimeoutPolicy()
RETRY_POLICY = RetryPolicy()
