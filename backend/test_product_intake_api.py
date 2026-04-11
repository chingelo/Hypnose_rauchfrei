from __future__ import annotations

import json
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from product_intake_api import product_router
from session_access_integration import integration_router


class ProductIntakeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.prev_runtime_state_dir = os.environ.get("TEST_APP_RUNTIME_STATE_DIR")
        self.prev_signing_secret = os.environ.get("INTEGRATION_SIGNING_SECRET")
        self.prev_webhook_secret = os.environ.get("INTEGRATION_WEBHOOK_SECRET")
        os.environ["TEST_APP_RUNTIME_STATE_DIR"] = self.tempdir.name
        os.environ["INTEGRATION_SIGNING_SECRET"] = "product-intake-signing-secret"
        os.environ["INTEGRATION_WEBHOOK_SECRET"] = "product-intake-webhook-secret"

        app = FastAPI()
        app.include_router(product_router)
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

    def test_package_endpoint_returns_structured_steps(self) -> None:
        response = self.client.get("/api/rauchfrei-package")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["key"], "rauchfrei")
        self.assertGreaterEqual(len(payload["steps"]), 6)
        self.assertFalse(payload["assets_available"])

    def test_guided_forms_fall_back_to_defaults_when_catalog_is_empty(self) -> None:
        response = self.client.get("/api/rauchfrei-guided-forms")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("questionnaire", payload)
        self.assertIn("journal", payload)
        self.assertGreater(len(payload["questionnaire"]["questions"]), 3)

    def test_guided_summary_is_stored_locally(self) -> None:
        response = self.client.post(
            "/api/rauchfrei-guided-summary",
            json={
                "form_key": "questionnaire",
                "answers": ["Seit zehn Jahren", "Zehn", "Im Stress"],
                "session_id": "session_demo",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["form_key"], "questionnaire")
        self.assertIn("Gefuehrter Rauchstopp-Fragebogen", payload["summary_text"])
        storage_path = payload["storage"]["path"]
        with open(storage_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored["session_id"], "session_demo")

    def test_guided_followup_uses_contextual_prompt(self) -> None:
        response = self.client.post(
            "/api/rauchfrei-guided-followup",
            json={
                "form_key": "journal",
                "question_index": 1,
                "answer": "Vor allem Stress im Buero",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Koerper", response.json()["followup"])

    def test_intake_creates_local_session_seed_and_status(self) -> None:
        response = self.client.post(
            "/api/intake",
            json={
                "vorname": "Mara",
                "email": "mara@example.com",
                "anliegen": "Ich will endlich rauchfrei werden.",
                "datenschutz_zustimmung": True,
                "program_type": "rauchfrei",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "created")
        self.assertEqual(payload["problem_type"], "rauchfrei")
        self.assertEqual(payload["technik_key"], "rauchfrei_guided_session")

        status = self.client.get(f"/integration/session-status/{payload['session_id']}")
        self.assertEqual(status.status_code, 200, status.text)
        status_payload = status.json()
        self.assertEqual(status_payload["session_status"], "ready")
        self.assertEqual(status_payload["program_type"], "rauchfrei")
        self.assertEqual(status_payload["last_channel"], "intake")

    def test_intake_requires_consent(self) -> None:
        response = self.client.post(
            "/api/intake",
            json={
                "vorname": "Mara",
                "anliegen": "Rauchfrei werden",
                "datenschutz_zustimmung": False,
            },
        )
        self.assertEqual(response.status_code, 400, response.text)


if __name__ == "__main__":
    unittest.main()
