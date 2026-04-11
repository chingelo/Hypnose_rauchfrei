from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_APPROVAL_FILE = Path(__file__).resolve().parent / "live_api_approval.json"


@dataclass(frozen=True)
class LiveApiApproval:
    allowed_scripts: tuple[str, ...]
    approved_by: str
    reason: str
    max_calls: int
    expires_at: str

    @classmethod
    def load(cls, path: Path) -> "LiveApiApproval":
        if not path.exists():
            raise RuntimeError(
                "Live-OpenAI ist standardmaessig blockiert. "
                f"Freigabedatei fehlt: {path}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Freigabedatei ist kein gueltiges JSON: {path}") from exc

        allowed_scripts = raw.get("allowed_scripts") or []
        approved_by = str(raw.get("approved_by") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        expires_at = str(raw.get("expires_at") or "").strip()
        max_calls = int(raw.get("max_calls") or 0)

        if not allowed_scripts or not all(isinstance(item, str) and item.strip() for item in allowed_scripts):
            raise RuntimeError("Freigabedatei braucht 'allowed_scripts' als nicht-leere Liste.")
        if not approved_by:
            raise RuntimeError("Freigabedatei braucht 'approved_by'.")
        if not reason:
            raise RuntimeError("Freigabedatei braucht 'reason'.")
        if max_calls < 1:
            raise RuntimeError("Freigabedatei braucht 'max_calls' > 0.")
        if not expires_at:
            raise RuntimeError("Freigabedatei braucht 'expires_at'.")
        return cls(
            allowed_scripts=tuple(item.strip() for item in allowed_scripts),
            approved_by=approved_by,
            reason=reason,
            max_calls=max_calls,
            expires_at=expires_at,
        )

    def ensure_valid_for(self, script_name: str) -> None:
        if script_name not in self.allowed_scripts:
            allowed = ", ".join(self.allowed_scripts)
            raise RuntimeError(
                f"Live-OpenAI fuer {script_name} ist nicht freigegeben. "
                f"Erlaubt laut Freigabedatei: {allowed}"
            )
        expiry = datetime.fromisoformat(self.expires_at)
        now = datetime.now(expiry.tzinfo) if expiry.tzinfo is not None else datetime.now()
        if now > expiry:
            raise RuntimeError(
                f"Live-Freigabe fuer {script_name} ist abgelaufen: {self.expires_at}"
            )


@dataclass
class LiveApiCallBudget:
    script_name: str
    approval_file: Path
    estimated_calls: int
    allowed_calls: int
    consumed_calls: int = 0

    def consume(self, label: str) -> None:
        if self.consumed_calls >= self.allowed_calls:
            raise RuntimeError(
                f"Live-API-Budget erschopft fuer {self.script_name}. "
                f"Erlaubt: {self.allowed_calls}, bereits verbraucht: {self.consumed_calls}, letzter Schritt: {label}"
            )
        self.consumed_calls += 1

    def summary(self) -> str:
        return (
            f"Live-OpenAI freigegeben fuer {self.script_name}: "
            f"geschaetzt={self.estimated_calls}, erlaubt={self.allowed_calls}, "
            f"Freigabe={self.approval_file}"
        )


def build_live_api_budget(
    script_name: str,
    *,
    estimated_calls: int,
    requested_max_calls: int | None,
    approval_file: str | Path | None = None,
) -> LiveApiCallBudget:
    if (os.getenv("OPENAI_LIVE_API_ALLOWED") or "").strip() != "1":
        raise RuntimeError(
            "Live-OpenAI ist blockiert. Setze OPENAI_LIVE_API_ALLOWED=1 nur fuer einen bewusst freigegebenen Lauf."
        )
    if requested_max_calls is None or requested_max_calls < 1:
        raise RuntimeError("Fuer Live-OpenAI ist --max-api-calls <n> verpflichtend.")

    approval_path = Path(approval_file) if approval_file else DEFAULT_APPROVAL_FILE
    approval = LiveApiApproval.load(approval_path)
    approval.ensure_valid_for(script_name)

    if estimated_calls > requested_max_calls:
        raise RuntimeError(
            f"Geplanter Lauf fuer {script_name} braucht mindestens {estimated_calls} Calls, "
            f"aber --max-api-calls ist nur {requested_max_calls}."
        )
    if requested_max_calls > approval.max_calls:
        raise RuntimeError(
            f"--max-api-calls {requested_max_calls} ueberschreitet die Repo-Freigabe {approval.max_calls} "
            f"fuer {script_name}."
        )

    return LiveApiCallBudget(
        script_name=script_name,
        approval_file=approval_path,
        estimated_calls=estimated_calls,
        allowed_calls=requested_max_calls,
    )
