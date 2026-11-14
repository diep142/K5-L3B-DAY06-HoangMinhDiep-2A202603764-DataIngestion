import unittest

from src.ingestion import load_text_units
from src.run import BENCHMARK_PATH, DATA_DIR, _read_jsonl, _validate_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_questions_have_valid_splits_filters_and_source_evidence(self) -> None:
        pages = load_text_units(DATA_DIR / "alice_units.jsonl")
        questions = _read_jsonl(BENCHMARK_PATH)

        _validate_benchmark(questions, pages)

        self.assertEqual(len(questions), 20)
        self.assertEqual(sum(question["split"] == "dev" for question in questions), 15)
        self.assertEqual(sum(question["split"] == "test" for question in questions), 5)


if __name__ == "__main__":
    unittest.main()
