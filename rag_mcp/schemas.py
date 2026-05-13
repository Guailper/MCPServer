"""Shared data structures for RAG indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A text document loaded from disk."""

    document_id: str
    title: str
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """A chunk generated from a document."""

    chunk_id: str
    knowledge_base_id: str
    document_id: str
    document_title: str
    source_path: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredChunk(TextChunk):
    """A persisted chunk with an embedding vector."""

    embedding: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its similarity score."""

    chunk: StoredChunk
    score: float
