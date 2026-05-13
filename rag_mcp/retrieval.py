"""Retrieve relevant chunks from the vector store."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rag_mcp.config import Settings
from rag_mcp.embedding import EmbeddingClient
from rag_mcp.vector_store import JsonlVectorStore


class RagRetriever:
    """Coordinate query embedding and vector search."""

    def __init__(self, settings: Settings, store: JsonlVectorStore) -> None:
        self.settings = settings
        self.store = store
        self.embedding_client = EmbeddingClient(settings)

    def search(
        self,
        query: str,
        knowledge_base_ids: list[str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        normalized_query = " ".join((query or "").split())
        if not normalized_query:
            return {
                "ok": False,
                "error": {"code": "EMPTY_QUERY", "message": "query 不能为空。"},
                "chunks": [],
            }

        selected_knowledge_base_ids = self._normalize_knowledge_base_ids(knowledge_base_ids)
        resolved_top_k = max(1, top_k or self.settings.top_k)
        resolved_min_score = self.settings.min_score if min_score is None else float(min_score)

        try:
            query_embedding = self.embedding_client.embed_query(normalized_query)
            results = self.store.search(
                query_embedding=query_embedding,
                knowledge_base_ids=selected_knowledge_base_ids,
                top_k=resolved_top_k,
                min_score=resolved_min_score,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                },
                "chunks": [],
            }

        return {
            "ok": True,
            "query": normalized_query,
            "knowledge_base_ids": selected_knowledge_base_ids,
            "top_k": resolved_top_k,
            "min_score": resolved_min_score,
            "count": len(results),
            "chunks": [self._result_to_dict(result) for result in results],
        }

    def _result_to_dict(self, result) -> dict[str, Any]:
        chunk_data = asdict(result.chunk)
        # embedding 只用于内部相似度计算，返回给 MCP 客户端会造成结果过大且不利于阅读。
        chunk_data.pop("embedding", None)
        chunk_data["score"] = result.score
        return chunk_data

    def _normalize_knowledge_base_ids(self, knowledge_base_ids: list[str] | None) -> list[str]:
        if not knowledge_base_ids:
            return [self.settings.default_knowledge_base_id]

        normalized_ids: list[str] = []
        for knowledge_base_id in knowledge_base_ids:
            normalized_id = str(knowledge_base_id or "").strip()
            if normalized_id and normalized_id not in normalized_ids:
                normalized_ids.append(normalized_id)

        return normalized_ids or [self.settings.default_knowledge_base_id]
