import unittest

from src.evaluation import evaluate_query, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "id": "q1",
            "split": "test",
            "gold_source_id": "alice11",
            "gold_unit_id": "unit-1",
            "evidence_quote": "verified evidence",
        }

    def test_hit_at_3_requires_gold_source_unit_and_full_evidence(self) -> None:
        hits = [
            {
                "rank": 1,
                "document_id": "alice11",
                "page": "unit-2",
                "metadata": {"unit_id": "unit-2"},
                "text": "verified evidence",
            },
            {
                "rank": 2,
                "document_id": "alice11",
                "page": "unit-1",
                "metadata": {"unit_id": "unit-1"},
                "text": "verified evidence is present here",
            },
        ]

        result = evaluate_query(self.question, hits)

        self.assertFalse(result["hit_at_1"])
        self.assertTrue(result["evidence_hit_at_3"])
        self.assertEqual(result["evidence_hit_ranks"], [2])

    def test_summary_reports_counts_by_split_and_strategy(self) -> None:
        traces = [
            {"strategy": "A", "split": "dev", "hit_at_1": True, "evidence_hit_at_3": True},
            {"strategy": "A", "split": "test", "hit_at_1": False, "evidence_hit_at_3": True},
        ]

        summary = summarize(traces)

        self.assertEqual(summary["A"]["dev"]["questions"], 1)
        self.assertEqual(summary["A"]["dev"]["hit_at_1"], 1)
        self.assertEqual(summary["A"]["test"]["evidence_hit_at_3"], 1)


if __name__ == "__main__":
    unittest.main()
