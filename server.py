"""Standalone RAG MCP server."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from rag_mcp.config import get_settings
from rag_mcp.indexing import RagIndexer
from rag_mcp.retrieval import RagRetriever
from rag_mcp.vector_store import JsonlVectorStore


mcp = FastMCP("standalone-rag")


def _build_store() -> JsonlVectorStore:
    settings = get_settings()
    return JsonlVectorStore(settings.store_path)


@mcp.tool()
def rag_index_path(
    path: str,
    knowledge_base_id: str | None = None,
    recursive: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Index a file or directory into a local RAG knowledge base."""

    settings = get_settings()
    indexer = RagIndexer(settings=settings, store=_build_store())
    return indexer.index_path(
        path=path,
        knowledge_base_id=knowledge_base_id,
        recursive=recursive,
        overwrite=overwrite,
    )


@mcp.tool()
def rag_search(
    query: str,
    knowledge_base_ids: list[str] | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Search indexed RAG chunks and return relevant context snippets."""

    settings = get_settings()
    retriever = RagRetriever(settings=settings, store=_build_store())
    return retriever.search(
        query=query,
        knowledge_base_ids=knowledge_base_ids,
        top_k=top_k,
        min_score=min_score,
    )


@mcp.tool()
def rag_list_knowledge_bases() -> dict[str, Any]:
    """List indexed knowledge bases and chunk counts."""

    return _build_store().list_knowledge_bases()


@mcp.tool()
def rag_delete_knowledge_base(knowledge_base_id: str) -> dict[str, Any]:
    """Delete all chunks under a knowledge base."""

    return _build_store().delete_knowledge_base(knowledge_base_id)


@mcp.tool()
def rag_health_check() -> dict[str, Any]:
    """Return runtime configuration and storage status."""

    settings = get_settings()
    store = _build_store()
    return {
        "ok": True,
        "embedding_provider": settings.embedding_provider,
        "store_path": str(settings.store_path),
        "store_exists": settings.store_path.exists(),
        "default_knowledge_base_id": settings.default_knowledge_base_id,
        "supported_extensions": sorted(settings.supported_extensions),
        "knowledge_bases": store.list_knowledge_bases()["knowledge_bases"],
    }


if __name__ == "__main__":
    mcp.run()
