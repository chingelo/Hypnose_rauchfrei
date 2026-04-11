from __future__ import annotations

import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from session_access_integration import (
    build_webhook_signature,
    integration_router,
    record_session_reset,
    record_session_response,
)


class SessionAccessIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.prev_runtime_state_dir = os.environ.get("TEST_APP_RUNTIME_STATE_DIR")
        self.prev_signing_secret = os.environ.get("INTEGRATION_SIGNING_SECRET")
        self.prev_webhook_secret = os.environ.get("INTEGRATION_WEBHOOK_SECRET")
        os.environ["TEST_APP_RUNTIME_STATE_DIR"] = self.tempdir.name
        os.environ["INTEGRATION_SIGNING_SECRET"] = "unit-test-signing-secret"
        os.environ["INTEGRATION_WEBHOOK_SECRET"] = "unit-test-webhook-secret"
        app = FastAPI()
        app.include_router(integration_router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tempdir.cleanup()
        if self.prev_runtime_state_dir is None:
            os.environ.pop("TEST_APP_RUNTIME_STATE_DIR", None)
        else:
            os.environ["TEST_APP_RUNTIME_STATE_DIR"] = self.prev_runtime_state_dir
        if self.prev_signing_secret is None:
            os.environ.pop("INTEGRATION_SIGNING_SECRET", None)
        else:
            os.environ["INTEGRATION_SIGNING_SECRET"] = self.prev_signing_secret
        if self.prev_webhook_secret is None:
            os.environ.pop("INTEGRATION_WEBHOOK_SECRET", None)
        else:
            os.environ["INTEGRATION_WEBHOOK_SECRET"] = self.prev_webhook_secret

    def _create_access(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "order_id": "order-123",
            "customer_id": "customer-456",
            "product_key": "rauchfrei-basic",
            "locale": "de-CH",
        }
        payload.update(overrides)
        response = self.client.post("/integration/create-session-access", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_create_access_and_resolve_token(self) -> None:
        created = self._create_access(return_url="https://example.com/hypnose/start")
        self.assertEqual(created["status"], "ok")
        self.assertTrue(created["session_id"])
        self.assertTrue(created["session_token"])
        self.assertIn("session_token=", created["start_url"])

        resolved = self.client.get(
            "/integration/resolve-session-token",
            params={"session_token": created["session_token"]},
        )
        self.assertEqual(resolved.status_code, 200, resolved.text)
        payload = resolved.json()
        self.assertEqual(payload["session_id"], created["session_id"])
        self.assertEqual(payload["product_key"], "rauchfrei-basic")
        self.assertEqual(payload["access_status"], "active")

        status = self.client.get(f"/integration/session-status/{created['session_id']}")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["session_status"], "ready")

    def test_runtime_response_marks_session_and_access_as_started(self) -> None:
        created = self._create_access()
        record_session_response(
            session_id=str(created["session_id"]),
            reply="Wir starten jetzt in Phase 4.",
            model="phase4_orchestrator_v1",
            phase4_active=True,
            phase4_node="activation_ready",
            channel="phase4",
        )

        status = self.client.get(f"/integration/session-status/{created['session_id']}")
        self.assertEqual(status.status_code, 200, status.text)
        payload = status.json()
        self.assertEqual(payload["session_status"], "active")
        self.assertTrue(payload["phase4_active"])
        self.assertEqual(payload["phase4_node"], "activation_ready")
        self.assertEqual(payload["last_channel"], "phase4")

        resolved = self.client.get(
            "/integration/resolve-session-token",
            params={"session_token": created["session_token"]},
        )
        self.assertEqual(resolved.status_code, 200, resolved.text)
        self.assertEqual(resolved.json()["access_status"], "started")

    def test_webhook_updates_session_status_when_signature_matches(self) -> None:
        created = self._create_access()
        body = json.dumps(
            {
                "event_type": "session_completed",
                "session_id": created["session_id"],
                "order_id": "order-123",
                "customer_id": "customer-456",
                "payload": {"source": "website"},
            }
        ).encode("utf-8")
        signature = build_webhook_signature(body)

        response = self.client.post(
            "/integration/session-webhook",
            content=body,
            headers={"X-Integration-Signature": signature, "Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "accepted")

        status = self.client.get(f"/integration/session-status/{created['session_id']}")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["session_status"], "completed")

    def test_reset_marks_session_as_reset(self) -> None:
        created = self._create_access()
        record_session_reset(str(created["session_id"]))

        status = self.client.get(f"/integration/session-status/{created['session_id']}")
        self.assertEqual(status.status_code, 200, status.text)
        payload = status.json()
        self.assertEqual(payload["session_status"], "reset")
        self.assertFalse(payload["phase4_active"])


if __name__ == "__main__":
    unittest.main()
