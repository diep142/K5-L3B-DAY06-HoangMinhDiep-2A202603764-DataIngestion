import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.embeddings import DEFAULT_LLM_MODEL, LLMClient, OFFICIAL_GEMINI_BASE_URL
from src.models import Chunk
from src.run import _chunk_stats


class ChunkMetricTests(unittest.TestCase):
    def test_chunk_stats_report_character_and_token_distributions(self) -> None:
        chunks = [
            Chunk("c1", "doc", "source", 1, "Alice went down"),
            Chunk("c2", "doc", "source", 1, "the rabbit hole"),
        ]

        stats = _chunk_stats(chunks, [4, 5])

        self.assertEqual(stats["chunk_count"], 2)
        self.assertEqual(stats["total_characters"], 30)
        self.assertEqual(stats["total_tokens"], 9)
        self.assertEqual(stats["min_tokens"], 4)
        self.assertEqual(stats["median_tokens"], 4.5)
        self.assertEqual(stats["mean_tokens"], 4.5)
        self.assertEqual(stats["max_tokens"], 5)

    def test_chunk_stats_reject_misaligned_token_counts(self) -> None:
        chunk = Chunk("c1", "doc", "source", 1, "Alice")

        with self.assertRaises(ValueError):
            _chunk_stats([chunk], [])


class LLMClientTests(unittest.TestCase):
    @patch("src.embeddings.load_dotenv")
    @patch("src.embeddings.urllib.request.urlopen")
    def test_from_env_uses_proxy_and_sends_chat_completion(
        self,
        urlopen: MagicMock,
        load_dotenv: MagicMock,
    ) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "A test answer"}}],
        }).encode("utf-8")
        environment = {
            "GEMINI_API_KEY": "test-key",
            "API_BASE_URL": "https://proxy.example/v1/",
            "LLM_MODEL": "proxy-model",
        }

        with patch.dict(os.environ, environment, clear=True):
            client = LLMClient.from_env()
            answer = client.complete([{"role": "user", "content": "Question"}])

        load_dotenv.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://proxy.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data)["model"], "proxy-model")
        self.assertEqual(answer, "A test answer")

    @patch("src.embeddings.load_dotenv")
    def test_from_env_defaults_to_official_gemini_endpoint(
        self,
        load_dotenv: MagicMock,
    ) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            client = LLMClient.from_env()

        self.assertEqual(client.base_url, OFFICIAL_GEMINI_BASE_URL.rstrip("/"))
        self.assertEqual(client.model, DEFAULT_LLM_MODEL)

    @patch("src.embeddings.load_dotenv")
    def test_from_env_requires_a_provider_key(self, load_dotenv: MagicMock) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                LLMClient.from_env()


if __name__ == "__main__":
    unittest.main()
