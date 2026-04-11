from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_api_guard import build_live_api_budget


def _write_approval(
    path: Path,
    *,
    allowed_scripts: list[str],
    max_calls: int,
    expires_at: datetime,
) -> None:
    payload = {
        "allowed_scripts": allowed_scripts,
        "approved_by": "test",
        "reason": "unit test",
        "max_calls": max_calls,
        "expires_at": expires_at.isoformat(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class LiveApiGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.approval_path = Path(self._tempdir.name) / "live_api_approval.json"
        self._previous_env = os.environ.get("OPENAI_LIVE_API_ALLOWED")
        os.environ.pop("OPENAI_LIVE_API_ALLOWED", None)

    def tearDown(self) -> None:
        if self._previous_env is None:
            os.environ.pop("OPENAI_LIVE_API_ALLOWED", None)
        else:
            os.environ["OPENAI_LIVE_API_ALLOWED"] = self._previous_env
        self._tempdir.cleanup()

    def test_blocks_when_live_env_flag_missing(self) -> None:
        _write_approval(
            self.approval_path,
            allowed_scripts=["run_session_validation_matrix.py"],
            max_calls=10,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with self.assertRaisesRegex(RuntimeError, "OPENAI_LIVE_API_ALLOWED=1"):
            build_live_api_budget(
                "run_session_validation_matrix.py",
                estimated_calls=5,
                requested_max_calls=5,
                approval_file=self.approval_path,
            )

    def test_blocks_when_approval_is_expired(self) -> None:
        os.environ["OPENAI_LIVE_API_ALLOWED"] = "1"
        _write_approval(
            self.approval_path,
            allowed_scripts=["run_session_validation_matrix.py"],
            max_calls=10,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        with self.assertRaisesRegex(RuntimeError, "abgelaufen"):
            build_live_api_budget(
                "run_session_validation_matrix.py",
                estimated_calls=5,
                requested_max_calls=5,
                approval_file=self.approval_path,
            )

    def test_blocks_when_script_is_not_approved(self) -> None:
        os.environ["OPENAI_LIVE_API_ALLOWED"] = "1"
        _write_approval(
            self.approval_path,
            allowed_scripts=["run_session_validation_matrix.py"],
            max_calls=10,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with self.assertRaisesRegex(RuntimeError, "nicht freigegeben"):
            build_live_api_budget(
                "run_session_sandbox.py",
                estimated_calls=5,
                requested_max_calls=5,
                approval_file=self.approval_path,
            )

    def test_blocks_when_requested_limit_exceeds_estimate_or_approval(self) -> None:
        os.environ["OPENAI_LIVE_API_ALLOWED"] = "1"
        _write_approval(
            self.approval_path,
            allowed_scripts=["run_session_validation_matrix.py"],
            max_calls=8,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with self.assertRaisesRegex(RuntimeError, "braucht mindestens 9 Calls"):
            build_live_api_budget(
                "run_session_validation_matrix.py",
                estimated_calls=9,
                requested_max_calls=8,
                approval_file=self.approval_path,
            )
        with self.assertRaisesRegex(RuntimeError, "ueberschreitet die Repo-Freigabe 8"):
            build_live_api_budget(
                "run_session_validation_matrix.py",
                estimated_calls=7,
                requested_max_calls=9,
                approval_file=self.approval_path,
            )

    def test_budget_stops_after_max_calls(self) -> None:
        os.environ["OPENAI_LIVE_API_ALLOWED"] = "1"
        _write_approval(
            self.approval_path,
            allowed_scripts=["run_session_validation_matrix.py"],
            max_calls=3,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        budget = build_live_api_budget(
            "run_session_validation_matrix.py",
            estimated_calls=3,
            requested_max_calls=3,
            approval_file=self.approval_path,
        )
        budget.consume("first")
        budget.consume("second")
        budget.consume("third")
        with self.assertRaisesRegex(RuntimeError, "Budget erschopft"):
            budget.consume("fourth")


if __name__ == "__main__":
    unittest.main()
