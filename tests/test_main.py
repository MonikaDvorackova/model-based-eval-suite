import unittest

from main import build_aigov_event, evaluate, evaluate_suite


class ModelBasedEvaluationTests(unittest.TestCase):
    def test_seeded_rollout_is_reproducible(self):
        self.assertEqual(evaluate("aggressive", 4, 12), evaluate("aggressive", 4, 12))
        self.assertNotEqual(evaluate("aggressive", 4, 12), evaluate("aggressive", 5, 12))

    def test_suite_reports_per_seed_and_policy_aggregates(self):
        report = evaluate_suite([0, 1, 2], horizon=12)
        self.assertEqual(len(report["episodes"]), 6)
        self.assertEqual(set(report["policy_summaries"]), {"aggressive", "conservative"})
        self.assertEqual(report["evaluation_seeds"], [0, 1, 2])
        self.assertTrue(all(row["mean_transition_error"] >= 0 for row in report["episodes"]))

    def test_aigov_event_matches_evaluation_reported_contract(self):
        report = evaluate_suite([0, 1], horizon=10)
        event = build_aigov_event(report, "test-run", threshold=0.2,
                                  timestamp="2026-01-01T00:00:00Z")
        payload = event["payload"]
        self.assertEqual(event["event_type"], "evaluation_reported")
        self.assertEqual(event["run_id"], "test-run")
        self.assertIsInstance(payload["metric"], str)
        self.assertIsInstance(payload["value"], float)
        self.assertIsInstance(payload["threshold"], float)
        self.assertIsInstance(payload["passed"], bool)
        self.assertTrue(payload["model_version_id"])
        self.assertTrue(payload["dataset_id"])
        self.assertTrue(payload["ai_system_id"])
        self.assertEqual(event["ts_utc"], "2026-01-01T00:00:00Z")

    def test_unknown_policy_and_empty_seed_set_fail(self):
        with self.assertRaises(ValueError):
            evaluate("missing", 0)
        with self.assertRaises(ValueError):
            evaluate_suite([])


if __name__ == "__main__":
    unittest.main()
