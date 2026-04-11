from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openai_semantic_backend import _load_env_with_bom_fallback, sanitize_model_json, sanitize_semantic_payload


class OpenAISemanticBackendTests(unittest.TestCase):
    def test_sanitize_model_json_replaces_non_finite_floats(self) -> None:
        raw = {
            "intent": "pleasant",
            "confidence": float("inf"),
            "nested": {"score": float("-inf")},
            "items": [1.0, float("nan")],
        }

        sanitized = sanitize_model_json(raw)

        self.assertNotIn("confidence", sanitized)
        self.assertEqual(sanitized["nested"], {})
        self.assertEqual(sanitized["items"][0], 1.0)
        self.assertIsNone(sanitized["items"][1])

    def test_sanitize_semantic_payload_drops_out_of_range_confidence(self) -> None:
        payload = {
            "intent": "pleasant",
            "action": "transition",
            "next_node": "hell_hypnose_loch_intro",
            "confidence": 90000,
        }

        sanitized = sanitize_semantic_payload(payload)

        self.assertNotIn("confidence", sanitized)

    def test_load_env_with_bom_fallback_reads_openai_key_from_utf8_sig_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\nHYPNOSE_ROUTER_MODEL_ID=gpt-5-mini\n", encoding="utf-8-sig")
            with patch.dict(os.environ, {}, clear=True):
                _load_env_with_bom_fallback(env_path)
                self.assertEqual(os.getenv("OPENAI_API_KEY"), "test-key")
                self.assertEqual(os.getenv("HYPNOSE_ROUTER_MODEL_ID"), "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
