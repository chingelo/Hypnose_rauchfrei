from __future__ import annotations

import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
FORBIDDEN_MARKERS = (
    r"C:\Projekte\hypnose_systemV2",
    "C:/Projekte/hypnose_systemV2",
)
EXCLUDED_FILES = {
    "test_no_external_v2_paths.py",
}


class NoExternalV2PathsTests(unittest.TestCase):
    def test_active_backend_python_files_do_not_reference_external_v2_paths(self) -> None:
        offending: list[str] = []
        for path in BACKEND_ROOT.glob("*.py"):
            if path.name in EXCLUDED_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in FORBIDDEN_MARKERS):
                offending.append(path.name)
        self.assertEqual(offending, [], f"External V2 paths still referenced in: {offending}")


if __name__ == "__main__":
    unittest.main()
