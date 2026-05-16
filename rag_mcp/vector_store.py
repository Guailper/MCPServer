"""Milvus 向量存储适配层。"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from rag_mcp.config import Settings
from rag_mcp.schemas import SearchResult, TextChunk


class MilvusVectorStore:
    """封装 RAG chunk 在 Milvus 中的写入、检索和删除。

    该类刻意保持很薄：Embedding 由上层完成，Milvus 只接收已经生成好的向量。
    这样索引流程更容易测试，也方便以后替换 LangChain Embeddings 提供方。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 Milvus 存储。

        Args:
            settings: Milvus 连接、collection 名称和检索参数。
        """

        self.settings = settings
        self._client = None

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        overwrite_knowledge_base: bool = False,
    ) -> dict[str, Any]:
        """写入一批 chunk。

        Args:
            chunks: 待写入的文本块。
            embeddings: 与 chunks 顺序一一对应的向量。
            overwrite_knowledge_base: 是否先删除目标知识库下的旧 chunk。

        Returns:
            包含写入数量和 Milvus collection 的执行结果。
        """

        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致。")

        if not chunks:
            return {"stored_chunks": 0, "collection": self.settings.milvus_collection}

        vector_dim = len(embeddings[0])
        if vector_dim <= 0:
            raise ValueError("embedding 向量不能为空。")

        self.ensure_collection(vector_dim)

        if overwrite_knowledge_base:
            self.delete_knowledge_base(chunks[0].knowledge_base_id)

        rows = [self._to_milvus_row(chunk, embedding) for chunk, embedding in zip(chunks, embeddings, strict=True)]
        result = self._get_client().insert(
            collection_name=self.settings.milvus_collection,
            data=rows,
        )
        return {
            "stored_chunks": len(rows),
            "collection": self.settings.milvus_collection,
            "milvus_result": result,
        }

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        knowledge_base_ids: list[str],
        top_k: int,
        min_score: float,
        search_mode: str = "hybrid",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[SearchResult]:
        """在指定知识库中执行向量相似度检索。

        Args:
            query_embedding: 查询文本的向量。
            knowledge_base_ids: 允许检索的知识库 ID 列表。
            top_k: Milvus 召回数量。
            min_score: 最低相似度阈值。

        Returns:
            按相似度从高到低排序的检索结果。
        """

        if not query_embedding or not knowledge_base_ids:
            return []

        client = self._get_client()
        collection_name = self.settings.milvus_collection
        if not client.has_collection(collection_name):
            return []
        self._validate_collection_schema(collection_name)

        if search_mode == "hybrid":
            results = self._hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                knowledge_base_ids=knowledge_base_ids,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        elif search_mode == "sparse":
            results = self._sparse_search(
                query_text=query_text,
                knowledge_base_ids=knowledge_base_ids,
                top_k=top_k,
            )
        elif search_mode == "dense":
            results = self._dense_search(
                query_embedding=query_embedding,
                knowledge_base_ids=knowledge_base_ids,
                top_k=top_k,
            )
        else:
            raise ValueError(f"不支持的检索模式：{search_mode}")

        search_results: list[SearchResult] = []
        for item in results[0] if results else []:
            score = self._extract_score(item)
            if score is not None and score < min_score:
                continue

            entity = item.get("entity", item)
            search_results.append(SearchResult(chunk=self._to_text_chunk(entity), score=score))

        return search_results

    def list_knowledge_bases(self) -> dict[str, Any]:
        """统计当前 collection 中每个知识库的 chunk 数量。

        Returns:
            知识库 ID 和 chunk 数量列表。
        """

        client = self._get_client()
        collection_name = self.settings.milvus_collection
        if not client.has_collection(collection_name):
            return {"collection": collection_name, "knowledge_bases": []}

        rows = client.query(
            collection_name=collection_name,
            filter="chunk_public_id != ''",
            output_fields=["knowledge_base_public_id"],
            limit=self.settings.list_limit,
        )
        counts = Counter(str(row.get("knowledge_base_public_id", "")) for row in rows)
        knowledge_bases = [
            {"knowledge_base_id": knowledge_base_id, "chunk_count": count}
            for knowledge_base_id, count in sorted(counts.items())
            if knowledge_base_id
        ]
        return {"collection": collection_name, "knowledge_bases": knowledge_bases}

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any]:
        """删除某个知识库下的所有 chunk。

        Args:
            knowledge_base_id: 要删除的知识库 ID。

        Returns:
            Milvus 删除结果。
        """

        normalized_id = (knowledge_base_id or "").strip()
        if not normalized_id:
            raise ValueError("knowledge_base_id 不能为空。")

        client = self._get_client()
        collection_name = self.settings.milvus_collection
        if not client.has_collection(collection_name):
            return {"collection": collection_name, "deleted": False}

        result = client.delete(
            collection_name=collection_name,
            filter=f"knowledge_base_public_id == {json.dumps(normalized_id, ensure_ascii=False)}",
        )
        return {"collection": collection_name, "deleted": True, "milvus_result": result}

    def ensure_collection(self, vector_dim: int) -> None:
        """确保 RAG collection 已存在。

        Args:
            vector_dim: 当前 embedding 模型输出的向量维度。
        """

        client = self._get_client()
        collection_name = self.settings.milvus_collection
        if client.has_collection(collection_name):
            if not self._missing_schema_fields(collection_name):
                return
            client.drop_collection(collection_name)

        from pymilvus import DataType

        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("chunk_public_id", DataType.VARCHAR, max_length=128)
        schema.add_field("knowledge_base_public_id", DataType.VARCHAR, max_length=128)
        schema.add_field("document_public_id", DataType.VARCHAR, max_length=128)
        schema.add_field("document_title", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("page_no", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        self._add_bm25_function(schema)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    def health_check(self) -> dict[str, Any]:
        """返回 Milvus 连接和 collection 状态。"""

        client = self._get_client()
        collection_name = self.settings.milvus_collection
        exists = client.has_collection(collection_name)
        missing_schema_fields = self._missing_schema_fields(collection_name) if exists else []
        return {
            "milvus_uri": self.settings.milvus_uri,
            "milvus_deployment_mode": self.settings.milvus_deployment_mode,
            "milvus_database": self.settings.milvus_database,
            "collection": collection_name,
            "collection_exists": exists,
            "collection_schema_compatible": exists and not missing_schema_fields,
            "missing_schema_fields": missing_schema_fields,
            "hybrid_search_supported": exists and self._supports_sparse_search(collection_name),
        }

    def _get_client(self):
        if self._client is not None:
            return self._client

        self._validate_connection_config()
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError("缺少 pymilvus 依赖，无法连接 Milvus。") from exc

        self._client = MilvusClient(
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token or None,
            db_name=self.settings.milvus_database or "default",
            timeout=self.settings.request_timeout_seconds,
        )
        return self._client

    def _validate_connection_config(self) -> None:
        mode = self.settings.milvus_deployment_mode
        if mode != "standalone":
            raise ValueError(f"当前 RAG MCP 仅按 Milvus Standalone 配置运行，不支持的 MILVUS_DEPLOYMENT_MODE：{mode}")

        uri = self.settings.milvus_uri.strip()
        parsed_uri = urlparse(uri)
        normalized_uri = uri.replace("\\", "/").lower()
        looks_like_local_file = (
            parsed_uri.scheme == "file"
            or normalized_uri.endswith((".db", ".sqlite", ".sqlite3"))
            or (not parsed_uri.scheme and "/" in normalized_uri)
        )
        if looks_like_local_file:
            raise ValueError(
                "当前配置为 Milvus Standalone，请将 MILVUS_URI 设置为服务地址，"
                "例如 http://127.0.0.1:19530，不要使用本地 .db 文件路径。"
            )

    def _to_milvus_row(self, chunk: TextChunk, embedding: list[float]) -> dict[str, Any]:
        metadata = {
            **chunk.metadata,
            "source_path": chunk.source_path,
        }
        return {
            "chunk_public_id": chunk.chunk_id,
            "knowledge_base_public_id": chunk.knowledge_base_id,
            "document_public_id": chunk.document_id,
            "document_title": chunk.document_title,
            "chunk_index": chunk.chunk_index,
            "page_no": chunk.page_no or 0,
            "content": chunk.content,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "dense_embedding": embedding,
        }

    def _dense_search(
        self,
        query_embedding: list[float],
        knowledge_base_ids: list[str],
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        return self._get_client().search(
            collection_name=self.settings.milvus_collection,
            data=[query_embedding],
            anns_field="dense_embedding",
            limit=max(1, top_k),
            filter=self._build_kb_filter(knowledge_base_ids),
            output_fields=self._output_fields(),
            search_params={"metric_type": "COSINE"},
        )

    def _sparse_search(
        self,
        query_text: str,
        knowledge_base_ids: list[str],
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        return self._get_client().search(
            collection_name=self.settings.milvus_collection,
            data=[query_text],
            anns_field="sparse_embedding",
            limit=max(1, top_k),
            filter=self._build_kb_filter(knowledge_base_ids),
            output_fields=self._output_fields(),
            search_params={"metric_type": "BM25"},
        )

    def _hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        knowledge_base_ids: list[str],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[list[dict[str, Any]]]:
        from pymilvus import AnnSearchRequest, WeightedRanker

        dense_request = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "COSINE"},
            limit=max(1, top_k),
            expr=self._build_kb_filter(knowledge_base_ids),
        )
        sparse_request = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},
            limit=max(1, top_k),
            expr=self._build_kb_filter(knowledge_base_ids),
        )
        ranker = WeightedRanker(float(dense_weight), float(sparse_weight))
        return self._get_client().hybrid_search(
            collection_name=self.settings.milvus_collection,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=max(1, top_k),
            output_fields=self._output_fields(),
        )

    def _output_fields(self) -> list[str]:
        return [
            "chunk_public_id",
            "knowledge_base_public_id",
            "document_public_id",
            "document_title",
            "chunk_index",
            "page_no",
            "content",
            "metadata_json",
        ]

    def _supports_sparse_search(self, collection_name: str) -> bool:
        return self._field_exists(collection_name, "sparse_embedding")

    def _validate_collection_schema(self, collection_name: str) -> None:
        missing_fields = self._missing_schema_fields(collection_name)
        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise RuntimeError(
                f"Milvus collection {collection_name!r} 使用的是旧 schema，缺少字段：{missing_text}。"
                "请先执行入库操作，服务会自动按新 schema 重建该 collection。"
            )

    def _missing_schema_fields(self, collection_name: str) -> list[str]:
        required_fields = {"dense_embedding", "sparse_embedding", "content"}
        return sorted(field for field in required_fields if not self._field_exists(collection_name, field))

    def _field_exists(self, collection_name: str, field_name: str) -> bool:
        try:
            description = self._get_client().describe_collection(collection_name)
        except Exception:
            return False

        fields = description.get("fields", []) if isinstance(description, dict) else []
        return any(
            field.get("name") == field_name or field.get("field_name") == field_name
            for field in fields
            if isinstance(field, dict)
        )

    def _add_bm25_function(self, schema: Any) -> None:
        from pymilvus import Function, FunctionType

        schema.add_function(
            Function(
                name="content_bm25",
                input_field_names=["content"],
                output_field_names=["sparse_embedding"],
                function_type=FunctionType.BM25,
            )
        )

    def _to_text_chunk(self, row: dict[str, Any]) -> TextChunk:
        metadata_json = row.get("metadata_json") or "{}"
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}

        page_no = int(row.get("page_no") or 0)
        source_path = metadata.get("source_path", "") if isinstance(metadata, dict) else ""
        return TextChunk(
            chunk_id=str(row.get("chunk_public_id", "")),
            knowledge_base_id=str(row.get("knowledge_base_public_id", "")),
            document_id=str(row.get("document_public_id", "")),
            document_title=str(row.get("document_title") or ""),
            source_path=str(source_path or ""),
            chunk_index=int(row.get("chunk_index") or 0),
            page_no=page_no or None,
            content=str(row.get("content") or ""),
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def _build_kb_filter(self, knowledge_base_ids: list[str]) -> str:
        values = json.dumps(knowledge_base_ids, ensure_ascii=False)
        return f"knowledge_base_public_id in {values}"

    def _extract_score(self, item: dict[str, Any]) -> float | None:
        raw_score = item.get("distance", item.get("score"))
        if raw_score is None:
            return None

        return float(raw_score)
