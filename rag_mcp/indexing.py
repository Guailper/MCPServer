"""Index local documents into the vector store."""

from __future__ import annotations

from typing import Any

from rag_mcp.chunker import TextChunker
from rag_mcp.config import Settings
from rag_mcp.document_loader import DocumentLoader
from rag_mcp.embedding import EmbeddingClient
from rag_mcp.vector_store import JsonlVectorStore


class RagIndexer:
    """Coordinate document loading, chunking, embedding, and persistence."""

    def __init__(self, settings: Settings, store: JsonlVectorStore) -> None:
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
        normalized_kb_id = self._normalize_knowledge_base_id(knowledge_base_id)

        try:
            documents = self.loader.load_path(path, recursive=recursive)
            chunks = [
                chunk
                for document in documents
                for chunk in self.chunker.split(document, normalized_kb_id)
            ]
            embeddings = self.embedding_client.embed_documents([chunk.content for chunk in chunks])
            store_result = self.store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                overwrite_knowledge_base=overwrite,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                },
            }

        return {
            "ok": True,
            "knowledge_base_id": normalized_kb_id,
            "documents": len(documents),
            "chunks": len(chunks),
            **store_result,
        }

    def _normalize_knowledge_base_id(self, knowledge_base_id: str | None) -> str:
        normalized = (knowledge_base_id or "").strip()
        return normalized or self.settings.default_knowledge_base_id
