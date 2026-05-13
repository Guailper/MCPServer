"""Runtime settings for the standalone RAG MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_csv_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    values = [value.strip() for value in raw_value.split(",")]
    return [value for value in values if value]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Configuration shared by indexing and retrieval."""

    store_path: Path
    default_knowledge_base_id: str
    chunk_target_size: int
    chunk_overlap_size: int
    max_file_size_bytes: int
    top_k: int
    min_score: float
    supported_extensions: set[str]
    embedding_provider: str
    embedding_dimensions: int
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_timeout_seconds: int


@lru_cache
def get_settings() -> Settings:
    supported_extensions = {
        extension.lower()
        for extension in _get_csv_env(
            "RAG_SUPPORTED_EXTENSIONS",
            [".txt", ".md", ".markdown"],
        )
    }

    return Settings(
        store_path=_resolve_path(os.getenv("RAG_STORE_PATH", ".rag_store/chunks.jsonl")),
        default_knowledge_base_id=os.getenv("RAG_DEFAULT_KNOWLEDGE_BASE_ID", "default").strip()
        or "default",
        chunk_target_size=max(100, _get_int_env("RAG_CHUNK_TARGET_SIZE", 900)),
        chunk_overlap_size=max(0, _get_int_env("RAG_CHUNK_OVERLAP_SIZE", 120)),
        max_file_size_bytes=max(1024, _get_int_env("RAG_MAX_FILE_SIZE_BYTES", 2_000_000)),
        top_k=max(1, _get_int_env("RAG_TOP_K", 5)),
        min_score=_get_float_env("RAG_MIN_SCORE", 0.05),
        supported_extensions=supported_extensions,
        embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "hash").strip().lower(),
        embedding_dimensions=max(32, _get_int_env("RAG_EMBEDDING_DIMENSIONS", 384)),
        embedding_base_url=os.getenv("RAG_EMBEDDING_BASE_URL", "").strip(),
        embedding_api_key=os.getenv("RAG_EMBEDDING_API_KEY", "").strip(),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embedding_timeout_seconds=max(1, _get_int_env("RAG_EMBEDDING_TIMEOUT_SECONDS", 30)),
    )
