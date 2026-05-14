"""RAG 检索流程。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rag_mcp.config import Settings
from rag_mcp.embedding import EmbeddingClient
from rag_mcp.vector_store import MilvusVectorStore


class RagRetriever:
    """协调查询向量化和 Milvus 相似度检索。"""

    def __init__(self, settings: Settings, store: MilvusVectorStore) -> None:
        """初始化检索器。

        Args:
            settings: RAG MCP 的运行配置。
            store: Milvus 存储适配器。
        """

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
        """检索与问题最相关的知识库片段。

        Args:
            query: 用户问题或检索语句。
            knowledge_base_ids: 指定知识库列表；为空时使用默认知识库。
            top_k: 最终返回数量；为空时使用配置默认值。
            min_score: 最低相似度分数；为空时使用配置默认值。

        Returns:
            MCP 工具可直接返回的结构化结果。
        """

        normalized_query = " ".join((query or "").split())
        if not normalized_query:
            raise ValueError("query 不能为空。")

        selected_knowledge_base_ids = self._normalize_knowledge_base_ids(knowledge_base_ids)
        resolved_final_top_k = max(1, top_k or self.settings.final_top_k)
        resolved_vector_top_k = max(resolved_final_top_k, self.settings.vector_top_k)
        resolved_min_score = self.settings.min_score if min_score is None else float(min_score)

        query_embedding = self.embedding_client.embed_query(normalized_query)
        results = self.store.search(
            query_embedding=query_embedding,
            knowledge_base_ids=selected_knowledge_base_ids,
            top_k=resolved_vector_top_k,
            min_score=resolved_min_score,
        )
        selected_results = results[:resolved_final_top_k]

        return {
            "ok": True,
            "query": normalized_query,
            "knowledge_base_ids": selected_knowledge_base_ids,
            "top_k": resolved_final_top_k,
            "min_score": resolved_min_score,
            "count": len(selected_results),
            "chunks": [self._result_to_dict(result) for result in selected_results],
        }

    def _result_to_dict(self, result) -> dict[str, Any]:
        chunk_data = asdict(result.chunk)
        chunk_data["score"] = result.score
        return chunk_data

    def _normalize_knowledge_base_ids(self, knowledge_base_ids: list[str] | None) -> list[str]:
        if not knowledge_base_ids:
            return self.settings.default_knowledge_base_ids

        normalized_ids: list[str] = []
        for knowledge_base_id in knowledge_base_ids:
            normalized_id = str(knowledge_base_id or "").strip()
            if normalized_id and normalized_id not in normalized_ids:
                normalized_ids.append(normalized_id)

        return normalized_ids or self.settings.default_knowledge_base_ids
