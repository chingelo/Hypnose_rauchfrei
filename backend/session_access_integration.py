from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STORE_LOCK = RLock()

integration_router = APIRouter(tags=["integration"])


class IntegrationCreateSessionAccessPayload(BaseModel):
    order_id: str = Field(..., min_length=1)
    customer_id: str = Field(..., min_length=1)
    product_key: str = Field(..., min_length=1)
    locale: str = "de-CH"
    session_id: str | None = None
    ttl_minutes: int = Field(default=180, ge=5, le=10080)
    return_url: str | None = None


class IntegrationSessionWebhookPayload(BaseModel):
    event_type: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    order_id: str | None = None
    customer_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def _runtime_state_dir() -> Path:
    configured = (os.getenv("TEST_APP_RUNTIME_STATE_DIR") or "").strip()
    if configured:
        return Path(configured).resolve()
    return (PROJECT_ROOT / "backend" / "runtime_state").resolve()


def _session_access_path() -> Path:
    return _runtime_state_dir() / "session_access_records.json"


def _session_status_path() -> Path:
    return _runtime_state_dir() / "session_status_records.json"


def _integration_events_path() -> Path:
    return _runtime_state_dir() / "integration_events.json"


def _safe_json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return copy.deepcopy(default)
    except Exception:
        logging.getLogger(__name__).exception("Failed to load JSON file: %s", path)
        return copy.deepcopy(default)


def _safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _integration_signing_secret() -> bytes:
    raw = (os.getenv("INTEGRATION_SIGNING_SECRET") or "").strip()
    if raw:
        return raw.encode("utf-8")
    return b"dev-integration-secret-change-me"


def _integration_webhook_secret() -> bytes:
    raw = (os.getenv("INTEGRATION_WEBHOOK_SECRET") or "").strip()
    if raw:
        return raw.encode("utf-8")
    return _integration_signing_secret()


def _sign_token_payload(payload_b64: str) -> str:
    digest = hmac.new(
        _integration_signing_secret(),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def _encode_session_token(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    signature = _sign_token_payload(payload_b64)
    return f"v1.{payload_b64}.{signature}"


def build_webhook_signature(body: bytes) -> str:
    digest = hmac.new(_integration_webhook_secret(), body, hashlib.sha256).digest()
    return _b64url_encode(digest)


def _verify_webhook_signature_or_401(body: bytes, signature_header: str | None) -> None:
    if not body:
        raise HTTPException(status_code=400, detail="Webhook body is empty.")
    provided = str(signature_header or "").strip()
    if not provided:
        raise HTTPException(status_code=401, detail="Missing webhook signature.")
    expected = build_webhook_signature(body)
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")


def decode_session_token(token: str) -> dict[str, Any]:
    raw = str(token or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Missing session token.")
    parts = raw.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise HTTPException(status_code=400, detail="Invalid session token format.")
    _, payload_b64, signature = parts
    expected = _sign_token_payload(payload_b64)
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid session token signature.")
    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Session token payload is invalid.") from exc
    expires_at = int(payload.get("exp", 0) or 0)
    if expires_at and _utc_ts() > expires_at:
        raise HTTPException(status_code=401, detail="Session token expired.")
    return payload


def _load_session_access_records() -> dict[str, dict[str, Any]]:
    payload = _safe_json_load(_session_access_path(), {})
    if isinstance(payload, dict):
        return payload
    return {}


def _save_session_access_records(records: dict[str, dict[str, Any]]) -> None:
    _safe_json_write(_session_access_path(), records)


def _load_session_status_records() -> dict[str, dict[str, Any]]:
    payload = _safe_json_load(_session_status_path(), {})
    if isinstance(payload, dict):
        return payload
    return {}


def _save_session_status_records(records: dict[str, dict[str, Any]]) -> None:
    _safe_json_write(_session_status_path(), records)


def _load_integration_events() -> list[dict[str, Any]]:
    payload = _safe_json_load(_integration_events_path(), [])
    if isinstance(payload, list):
        return payload
    return []


def _save_integration_events(events: list[dict[str, Any]]) -> None:
    _safe_json_write(_integration_events_path(), events)


def _merge_defined(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        target[key] = value
    return target


def _upsert_session_status_record(session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        raise ValueError("session_id fehlt.")

    now_iso = _now_iso()
    with _STORE_LOCK:
        records = _load_session_status_records()
        existing = dict(records.get(clean_session_id) or {})
        record = {
            "session_id": clean_session_id,
            "created_at": existing.get("created_at") or now_iso,
            "updated_at": now_iso,
            "program_type": existing.get("program_type") or "rauchfrei",
            "status": existing.get("status") or "ready",
            "source": existing.get("source") or "runtime",
            "last_step": existing.get("last_step") if existing.get("last_step") is not None else 1,
            "phase4_active": bool(existing.get("phase4_active", False)),
            "phase4_node": existing.get("phase4_node"),
        }
        _merge_defined(record, existing)
        _merge_defined(record, updates)
        record["session_id"] = clean_session_id
        record["updated_at"] = now_iso
        records[clean_session_id] = record
        _save_session_status_records(records)
        return dict(record)


def _find_access_record_id_for_session(
    records: dict[str, dict[str, Any]],
    session_id: str,
) -> str | None:
    best_access_id = None
    best_exp = -1
    clean_session_id = str(session_id or "").strip()
    for access_id, record in records.items():
        if str(record.get("session_id") or "").strip() != clean_session_id:
            continue
        expires_at = int(record.get("expires_at_ts") or 0)
        if expires_at >= best_exp:
            best_access_id = access_id
            best_exp = expires_at
    return best_access_id


def _set_access_status_for_session(session_id: str, access_status: str) -> None:
    if access_status not in {"active", "started", "completed", "aborted", "expired", "failed"}:
        return
    with _STORE_LOCK:
        records = _load_session_access_records()
        access_id = _find_access_record_id_for_session(records, session_id)
        if not access_id:
            return
        record = dict(records.get(access_id) or {})
        if not record:
            return
        record["status"] = access_status
        record["updated_at"] = _now_iso()
        records[access_id] = record
        _save_session_access_records(records)


def _append_integration_event(event: dict[str, Any]) -> None:
    with _STORE_LOCK:
        events = _load_integration_events()
        events.append(event)
        _save_integration_events(events)


def _build_start_url(return_url: str | None, session_token: str) -> str | None:
    raw = str(return_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    query_items = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "session_token"]
    query_items.append(("session_token", session_token))
    updated = parsed._replace(query=urlencode(query_items))
    return urlunparse(updated)


def create_session_access(payload: IntegrationCreateSessionAccessPayload) -> dict[str, Any]:
    session_id = str(payload.session_id or "").strip() or f"session_{secrets.token_urlsafe(10)}"
    access_id = secrets.token_urlsafe(16)
    issued_at = _utc_ts()
    expires_at = issued_at + int(payload.ttl_minutes) * 60
    token_payload = {
        "access_id": access_id,
        "session_id": session_id,
        "order_id": payload.order_id,
        "customer_id": payload.customer_id,
        "product_key": payload.product_key,
        "locale": payload.locale,
        "iat": issued_at,
        "exp": expires_at,
    }
    session_token = _encode_session_token(token_payload)
    now_iso = _now_iso()

    with _STORE_LOCK:
        session_records = _load_session_status_records()
        session_record = dict(session_records.get(session_id) or {})
        session_record = _merge_defined(
            {
                "session_id": session_id,
                "created_at": session_record.get("created_at") or now_iso,
                "updated_at": now_iso,
                "status": session_record.get("status") or "ready",
                "program_type": session_record.get("program_type") or "rauchfrei",
                "source": session_record.get("source") or "website_integration",
                "last_step": session_record.get("last_step") if session_record.get("last_step") is not None else 1,
                "phase4_active": bool(session_record.get("phase4_active", False)),
                "phase4_node": session_record.get("phase4_node"),
            },
            {
                "order_id": payload.order_id,
                "customer_id": payload.customer_id,
                "product_key": payload.product_key,
                "locale": payload.locale,
                "access_id": access_id,
                "return_url": payload.return_url,
            },
        )
        session_records[session_id] = session_record
        _save_session_status_records(session_records)

        access_records = _load_session_access_records()
        access_records[access_id] = {
            "access_id": access_id,
            "session_id": session_id,
            "order_id": payload.order_id,
            "customer_id": payload.customer_id,
            "product_key": payload.product_key,
            "locale": payload.locale,
            "status": "active",
            "created_at": now_iso,
            "updated_at": now_iso,
            "expires_at_ts": expires_at,
            "return_url": payload.return_url,
        }
        _save_session_access_records(access_records)

    return {
        "status": "ok",
        "access_id": access_id,
        "session_id": session_id,
        "session_token": session_token,
        "expires_at_ts": expires_at,
        "start_url": _build_start_url(payload.return_url, session_token),
    }


def resolve_session_token(session_token: str) -> dict[str, Any]:
    token_payload = decode_session_token(session_token)
    access_id = str(token_payload.get("access_id") or "").strip()
    with _STORE_LOCK:
        access_records = _load_session_access_records()
        access_record = dict(access_records.get(access_id) or {})
        if access_id and not access_record:
            raise HTTPException(status_code=401, detail="Session access is unknown.")
        if access_record:
            access_status = str(access_record.get("status") or "active").lower()
            if access_status not in {"active", "started"}:
                raise HTTPException(status_code=401, detail="Session access is not active.")
            access_record["last_resolved_at"] = _now_iso()
            access_record["updated_at"] = _now_iso()
            access_records[access_id] = access_record
            _save_session_access_records(access_records)
        else:
            access_status = "active"

    return {
        "status": "ok",
        "session_id": token_payload.get("session_id"),
        "order_id": token_payload.get("order_id"),
        "customer_id": token_payload.get("customer_id"),
        "product_key": token_payload.get("product_key"),
        "locale": token_payload.get("locale"),
        "expires_at_ts": token_payload.get("exp"),
        "access_status": access_status,
    }


def _apply_webhook_status_update(payload: IntegrationSessionWebhookPayload) -> None:
    event_type = str(payload.event_type or "").strip().lower()
    session_status_map = {
        "session_started": "active",
        "session_completed": "completed",
        "session_aborted": "aborted",
        "session_expired": "expired",
        "session_failed": "failed",
        "session_reset": "reset",
    }
    access_status_map = {
        "session_started": "started",
        "session_completed": "completed",
        "session_aborted": "aborted",
        "session_expired": "expired",
        "session_failed": "failed",
    }

    session_status = session_status_map.get(event_type)
    if session_status:
        update_payload = {
            "status": session_status,
            "order_id": payload.order_id,
            "customer_id": payload.customer_id,
            "phase4_active": session_status == "active",
            "last_channel": "integration_webhook",
            "last_event_type": event_type,
        }
        _upsert_session_status_record(payload.session_id, update_payload)

    access_status = access_status_map.get(event_type)
    if access_status:
        _set_access_status_for_session(payload.session_id, access_status)


def record_session_response(
    *,
    session_id: str,
    reply: str,
    model: str,
    phase4_active: bool,
    phase4_node: str | None,
    channel: str,
) -> None:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        return
    status = "completed" if str(phase4_node or "").strip().lower() == "completed" else "active"
    try:
        _upsert_session_status_record(
            clean_session_id,
            {
                "status": status,
                "last_channel": channel,
                "last_model": model,
                "last_message_preview": str(reply or "").strip()[:240],
                "last_activity_at": _now_iso(),
                "phase4_active": phase4_active,
                "phase4_node": phase4_node,
                "last_step": 4 if phase4_node else 1,
            },
        )
        _set_access_status_for_session(
            clean_session_id,
            "completed" if status == "completed" else "started",
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to record session response for session %s",
            clean_session_id,
        )


def record_session_reset(session_id: str) -> None:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        return
    try:
        _upsert_session_status_record(
            clean_session_id,
            {
                "status": "reset",
                "phase4_active": False,
                "phase4_node": None,
                "last_channel": "session_reset",
                "last_activity_at": _now_iso(),
            },
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to record session reset for session %s",
            clean_session_id,
        )


def seed_session_status_record(
    *,
    session_id: str,
    status: str = "ready",
    program_type: str = "rauchfrei",
    source: str = "intake",
    last_step: int = 1,
    phase4_active: bool = False,
    phase4_node: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        return
    updates: dict[str, Any] = {
        "status": status,
        "program_type": program_type,
        "source": source,
        "last_step": last_step,
        "phase4_active": phase4_active,
        "phase4_node": phase4_node,
        "last_channel": source,
        "last_activity_at": _now_iso(),
    }
    if extra_fields:
        updates.update(extra_fields)
    try:
        _upsert_session_status_record(clean_session_id, updates)
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to seed session record for session %s",
            clean_session_id,
        )


@integration_router.post("/integration/create-session-access")
@integration_router.post("/v1/integration/create-session-access")
def integration_create_session_access(payload: IntegrationCreateSessionAccessPayload) -> dict[str, Any]:
    return create_session_access(payload)


@integration_router.get("/integration/resolve-session-token")
@integration_router.get("/v1/integration/resolve-session-token")
def integration_resolve_session_token(session_token: str) -> dict[str, Any]:
    return resolve_session_token(session_token)


@integration_router.post("/integration/session-webhook")
@integration_router.post("/v1/integration/session-webhook")
async def integration_session_webhook(request: Request) -> dict[str, Any]:
    body = await request.body()
    _verify_webhook_signature_or_401(body, request.headers.get("X-Integration-Signature"))
    try:
        parsed = json.loads(body.decode("utf-8"))
        payload = IntegrationSessionWebhookPayload(**parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook payload.") from exc

    event = {
        "event_id": secrets.token_urlsafe(12),
        "event_type": payload.event_type.strip(),
        "session_id": payload.session_id.strip(),
        "order_id": payload.order_id,
        "customer_id": payload.customer_id,
        "payload": payload.payload,
        "received_at": _now_iso(),
    }
    _append_integration_event(event)
    _apply_webhook_status_update(payload)
    return {"status": "accepted", "event_id": event["event_id"]}


@integration_router.get("/integration/session-status/{session_id}")
@integration_router.get("/v1/integration/session-status/{session_id}")
def integration_session_status(session_id: str) -> dict[str, Any]:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    with _STORE_LOCK:
        records = _load_session_status_records()
        record = dict(records.get(clean_session_id) or {})
    if not record:
        raise HTTPException(status_code=404, detail="Session not found.")

    return {
        "status": "ok",
        "session_id": clean_session_id,
        "session_status": record.get("status"),
        "last_step": record.get("last_step"),
        "program_type": record.get("program_type"),
        "updated_at": record.get("updated_at"),
        "phase4_active": bool(record.get("phase4_active", False)),
        "phase4_node": record.get("phase4_node"),
        "last_channel": record.get("last_channel"),
        "last_model": record.get("last_model"),
        "order_id": record.get("order_id"),
        "customer_id": record.get("customer_id"),
        "product_key": record.get("product_key"),
        "locale": record.get("locale"),
    }
