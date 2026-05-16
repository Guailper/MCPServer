"""Long-lived RAG service container."""

from __future__ import annotations

from functools import lru_cache

from rag_mcp.chunker import TextChunker
from rag_mcp.config import Settings, get_settings
from rag_mcp.document_loader import DocumentLoader
from rag_mcp.embedding import EmbeddingClient
from rag_mcp.indexing import RagIndexer
from rag_mcp.reranking import RerankClient
from rag_mcp.retrieval import RagRetriever
from rag_mcp.vector_store import MilvusVectorStore


class RagApplication:
    """Owns reusable clients and pipeline objects for the MCP process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = MilvusVectorStore(settings)
        self.embedding_client = EmbeddingClient(settings)
        self.rerank_client = RerankClient(settings)
        self.loader = DocumentLoader(settings)
        self.chunker = TextChunker(settings)
        self.indexer = RagIndexer(
            settings=settings,
            store=self.store,
            loader=self.loader,
            chunker=self.chunker,
            embedding_client=self.embedding_client,
        )
        self.retriever = RagRetriever(
            settings=settings,
            store=self.store,
            embedding_client=self.embedding_client,
            rerank_client=self.rerank_client,
        )


@lru_cache
def get_application() -> RagApplication:
    return RagApplication(get_settings())
