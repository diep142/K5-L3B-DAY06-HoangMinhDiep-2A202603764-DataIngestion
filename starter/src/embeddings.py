"""Local FastEmbed adapter using the same real model for documents and queries."""
import json
import os
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

OFFICIAL_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Return the FastEmbed tokenizer's token count for each input text."""
        ...


class FastEmbedder:
    """Generate embeddings locally; no API credential or mock fallback is used."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from fastembed import TextEmbedding

        supported_models = TextEmbedding.list_supported_models()
        model_info = next(
            (item for item in supported_models if item["model"] == model_name),
            None,
        )
        if model_info is None:
            names = ", ".join(item["model"] for item in supported_models)
            raise ValueError(f"Unsupported FastEmbed model {model_name!r}; supported: {names}")
        self.model_name = model_name
        self.dimensions = int(model_info["dim"])
        self.package_version = version("fastembed")
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = [vector.tolist() for vector in self._model.embed(texts, batch_size=8)]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} document vectors; received {len(vectors)}")
        self._validate_vectors(vectors)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = [vector.tolist() for vector in self._model.query_embed(question)]
        if len(vectors) != 1:
            raise RuntimeError(f"Expected one query vector; received {len(vectors)}")
        self._validate_vectors(vectors)
        return vectors[0]

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [self._model.token_count(text) for text in texts]

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Vector {index} has {len(vector)} dimensions; expected {self.dimensions}"
                )


class LLMClient:
    """Call Gemini's OpenAI-compatible API or a configured compatible proxy."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if not model.strip():
            raise ValueError("model is required")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @classmethod
    def from_env(cls) -> "LLMClient":
        starter_dir = Path(__file__).resolve().parents[1]
        load_dotenv(starter_dir / ".env")
        api_key = (os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or LLM_API_KEY) in starter/.env or the environment."
            )
        base_url = (
            os.getenv("API_BASE_URL", "").strip()
            or os.getenv("LLM_BASE_URL", "").strip()
            or OFFICIAL_GEMINI_BASE_URL
        )
        model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL
        return cls(api_key, base_url, model)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            raise ValueError("At least one chat message is required")
        endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(
                {"model": self.model, "messages": messages},
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"LLM endpoint returned HTTP {error.code}.") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach the configured LLM endpoint: {error.reason}") from error

        if not isinstance(payload, dict):
            raise RuntimeError("LLM endpoint returned an invalid response object.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("LLM endpoint returned no completion choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM endpoint returned a completion without a message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM endpoint returned an empty text completion.")
        return content
