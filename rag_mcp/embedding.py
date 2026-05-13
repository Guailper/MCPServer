"""Embedding providers for standalone RAG."""

from __future__ import annotations

import hashlib
import math
import re

import httpx

from rag_mcp.config import Settings


class EmbeddingClient:
    """Generate embeddings with a zero-dependency hash provider or an API provider."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._local_client = None

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        provider = self.settings.embedding_provider
        if provider == "hash":
            return [self._hash_embed(text) for text in texts]
        if provider == "sentence_transformers":
            return self._embed_with_sentence_transformers(texts)
        if provider in {"openai", "openai_compatible", "api"}:
            return self._embed_with_openai_compatible_api(texts)

        raise ValueError(f"未知 embedding provider：{provider}")

    def _hash_embed(self, text: str) -> list[float]:
        dimensions = self.settings.embedding_dimensions
        vector = [0.0] * dimensions
        tokens = self._tokenize(text)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        return self._normalize(vector)

    def _tokenize(self, text: str) -> list[str]:
        lowered_text = text.lower()
        words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered_text)
        if words:
            return words

        return [character for character in lowered_text if not character.isspace()]

    def _embed_with_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._local_client is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 sentence-transformers，请安装："
                    "python -m pip install -e .[local-embedding]"
                ) from exc

            self._local_client = SentenceTransformer(self.settings.embedding_model)

        vectors = self._local_client.encode(texts, normalize_embeddings=True)
        return [self._to_float_list(vector) for vector in vectors]

    def _embed_with_openai_compatible_api(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.embedding_base_url or not self.settings.embedding_api_key:
            raise RuntimeError(
                "OpenAI-compatible embedding 未配置，请设置 "
                "RAG_EMBEDDING_BASE_URL 和 RAG_EMBEDDING_API_KEY。"
            )

        url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {self.settings.embedding_api_key}"}
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }

        with httpx.Client(timeout=self.settings.embedding_timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        vectors_by_index: dict[int, list[float]] = {}
        for item in data.get("data", []):
            index = int(item.get("index", len(vectors_by_index)))
            vectors_by_index[index] = self._normalize(self._to_float_list(item["embedding"]))

        return [vectors_by_index[index] for index in range(len(texts))]

    def _to_float_list(self, vector) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        return [float(value) for value in vector]

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]
