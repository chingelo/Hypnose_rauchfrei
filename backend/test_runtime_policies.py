from __future__ import annotations

import unittest

from runtime_policies import RETRY_POLICY, TIMEOUT_POLICY, RetryPolicy, TimeoutPolicy


class RuntimePoliciesTests(unittest.TestCase):
    def test_timeout_policy_defaults_match_expected_product_limits(self) -> None:
        self.assertEqual(TIMEOUT_POLICY, TimeoutPolicy())
        self.assertEqual(TIMEOUT_POLICY.inactivity_auto_end_s, 600)
        self.assertEqual(TIMEOUT_POLICY.phase4_followup_s, 16)

    def test_retry_policy_defaults_are_available_globally(self) -> None:
        self.assertEqual(RETRY_POLICY, RetryPolicy())
        self.assertEqual(RETRY_POLICY.yes_no_max, 2)
        self.assertEqual(RETRY_POLICY.phase4_core_max, 2)


if __name__ == "__main__":
    unittest.main()
