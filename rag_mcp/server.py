"""RAG MCP tool entrypoints."""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP

from rag_mcp.application import get_application
from rag_mcp.config import get_settings


mcp = FastMCP("rag-mcp")


@mcp.tool()
def rag_index_text(
    text: str,
    knowledge_base_id: str | None = None,
    document_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Index plain text into a knowledge base."""

    return get_application().indexer.index_text(
        text=text,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        title=title,
        metadata=metadata,
        overwrite=overwrite,
    )


@mcp.tool()
def rag_index_path(
    path: str,
    knowledge_base_id: str | None = None,
    recursive: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Index a local file or directory."""

    return get_application().indexer.index_path(
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
    search_mode: str | None = None,
    rerank: bool | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> dict[str, Any]:
    """Search indexed chunks with dense, sparse, or hybrid retrieval."""

    return get_application().retriever.search(
        query=query,
        knowledge_base_ids=knowledge_base_ids,
        top_k=top_k,
        min_score=min_score,
        search_mode=search_mode,
        rerank=rerank,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )


@mcp.tool()
def rag_list_knowledge_bases() -> dict[str, Any]:
    """List knowledge bases in the current Milvus collection."""

    return {"ok": True, **get_application().store.list_knowledge_bases()}


@mcp.tool()
def rag_delete_knowledge_base(knowledge_base_id: str) -> dict[str, Any]:
    """Delete all chunks under one knowledge base."""

    return {"ok": True, **get_application().store.delete_knowledge_base(knowledge_base_id)}


@mcp.tool()
def rag_health_check() -> dict[str, Any]:
    """Return runtime configuration and Milvus collection status."""

    settings = get_settings()
    app = get_application()
    return {
        "ok": True,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_model_path": settings.embedding_model_path,
        "milvus_deployment_mode": settings.milvus_deployment_mode,
        "search_mode": settings.search_mode,
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "rerank_model_path": settings.rerank_model_path,
        "model_device": settings.model_device,
        "default_knowledge_base_ids": settings.default_knowledge_base_ids,
        "supported_extensions": sorted(settings.supported_extensions),
        **app.store.health_check(),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse MCP transport arguments."""

    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--sse", action="store_true", help="Use SSE transport")
    parser.add_argument("--streamable-http", action="store_true", help="Use streamable-http transport")
    parser.add_argument("--stateless", action="store_true", help="Enable stateless mode for streamable-http")
    parser.add_argument("--host", type=str, default="localhost", help="HTTP transport host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP transport port")
    return parser.parse_args()


def main() -> None:
    """Start the RAG MCP server."""

    args = parse_arguments()
    if args.sse:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
        return

    if args.streamable_http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        if args.stateless:
            mcp.settings.stateless_http = True
            mcp.settings.json_response = True
        mcp.run(transport="streamable-http")
        return

    mcp.run()


if __name__ == "__main__":
    main()
