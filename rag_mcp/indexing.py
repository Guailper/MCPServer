"""RAG 索引流程。"""

from __future__ import annotations

import hashlib
from typing import Any

from rag_mcp.chunker import TextChunker
from rag_mcp.config import Settings
from rag_mcp.document_loader import DocumentLoader
from rag_mcp.embedding import EmbeddingClient
from rag_mcp.schemas import Document
from rag_mcp.vector_store import MilvusVectorStore


class RagIndexer:
    """协调文档读取、分块、向量化和 Milvus 写入。"""

    def __init__(self, settings: Settings, store: MilvusVectorStore) -> None:
        """初始化索引器。

        Args:
            settings: RAG MCP 的运行配置。
            store: Milvus 存储适配器。
        """

        self.settings = settings
        self.store = store
        self.loader = DocumentLoader(settings)
        self.chunker = TextChunker(settings)
        self.embedding_client = EmbeddingClient(settings)

    def index_path(
        self,
        path: str,
        knowledge_base_id: str | None = None,
        recursive: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """索引本地文件或目录。

        Args:
            path: 文件或目录路径。
            knowledge_base_id: 目标知识库 ID；为空时使用默认知识库。
            recursive: path 为目录时是否递归读取。
            overwrite: 是否先删除该知识库下的旧 chunk。

        Returns:
            索引结果统计。
        """

        normalized_kb_id = self._normalize_knowledge_base_id(knowledge_base_id)
        documents = self.loader.load_path(path, recursive=recursive)
        return self._index_documents(documents, normalized_kb_id, overwrite)

    def index_text(
        self,
        text: str,
        knowledge_base_id: str | None = None,
        document_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """索引调用方直接传入的文本。

        Args:
            text: 待索引的正文。
            knowledge_base_id: 目标知识库 ID；为空时使用默认知识库。
            document_id: 文档 ID；为空时根据文本内容自动生成。
            title: 文档标题。
            metadata: 附加元数据。
            overwrite: 是否先删除该知识库下的旧 chunk。

        Returns:
            索引结果统计。
        """

        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("text 不能为空。")

        normalized_kb_id = self._normalize_knowledge_base_id(knowledge_base_id)
        resolved_document_id = (document_id or "").strip() or self._build_text_document_id(normalized_kb_id, normalized_text)
        document = Document(
            document_id=resolved_document_id,
            title=(title or resolved_document_id).strip(),
            source_path="",
            content=normalized_text,
            metadata=metadata or {},
        )
        return self._index_documents([document], normalized_kb_id, overwrite)

    def _index_documents(
        self,
        documents: list[Document],
        knowledge_base_id: str,
        overwrite: bool,
    ) -> dict[str, Any]:
        chunks = [
            chunk
            for document in documents
            for chunk in self.chunker.split(document, knowledge_base_id)
        ]
        embeddings = self.embedding_client.embed_documents([chunk.content for chunk in chunks])
        store_result = self.store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            overwrite_knowledge_base=overwrite,
        )
        return {
            "ok": True,
            "knowledge_base_id": knowledge_base_id,
            "documents": len(documents),
            "chunks": len(chunks),
            **store_result,
        }

    def _normalize_knowledge_base_id(self, knowledge_base_id: str | None) -> str:
        normalized = (knowledge_base_id or "").strip()
        return normalized or self.settings.default_knowledge_base_ids[0]

    def _build_text_document_id(self, knowledge_base_id: str, text: str) -> str:
        payload = f"{knowledge_base_id}\0{text}"
        return f"doc_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"
