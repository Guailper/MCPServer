"""RAG MCP 工具入口。"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP

from rag_mcp.config import get_settings
from rag_mcp.indexing import RagIndexer
from rag_mcp.retrieval import RagRetriever
from rag_mcp.vector_store import MilvusVectorStore


mcp = FastMCP("rag-mcp")


def _build_store() -> MilvusVectorStore:
    """创建 Milvus 存储实例。"""

    return MilvusVectorStore(get_settings())


@mcp.tool()
def rag_index_text(
    text: str,
    knowledge_base_id: str | None = None,
    document_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """把一段文本写入 RAG 知识库。

    Args:
        text: 要索引的正文。
        knowledge_base_id: 目标知识库 ID；为空时使用默认知识库。
        document_id: 文档 ID；为空时根据文本内容自动生成。
        title: 文档标题。
        metadata: 附加元数据，会写入 Milvus 的 metadata_json。
        overwrite: 是否先清空该知识库下已有 chunk。

    Returns:
        索引结果统计，包括文档数、chunk 数和 Milvus 写入结果。
    """

    settings = get_settings()
    indexer = RagIndexer(settings=settings, store=_build_store())
    return indexer.index_text(
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
    """索引本地文件或目录。

    Args:
        path: 本地文件或目录路径。
        knowledge_base_id: 目标知识库 ID；为空时使用默认知识库。
        recursive: path 为目录时是否递归读取子目录。
        overwrite: 是否先清空该知识库下已有 chunk。

    Returns:
        索引结果统计，包括文档数、chunk 数和 Milvus 写入结果。
    """

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
    """检索知识库中与问题相关的上下文片段。

    Args:
        query: 用户问题或检索语句。
        knowledge_base_ids: 限定检索的知识库 ID 列表；为空时使用默认知识库。
        top_k: 最终返回的 chunk 数量。
        min_score: 最低相似度分数。

    Returns:
        命中的 chunk 列表，每条包含内容、来源文档、chunk 序号、分数和元数据。
    """

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
    """列出当前 Milvus collection 中的知识库。

    Returns:
        知识库 ID 和对应 chunk 数量。
    """

    return {"ok": True, **_build_store().list_knowledge_bases()}


@mcp.tool()
def rag_delete_knowledge_base(knowledge_base_id: str) -> dict[str, Any]:
    """删除某个知识库下的所有 chunk。

    Args:
        knowledge_base_id: 要删除的知识库 ID。

    Returns:
        Milvus 删除结果。
    """

    return {"ok": True, **_build_store().delete_knowledge_base(knowledge_base_id)}


@mcp.tool()
def rag_health_check() -> dict[str, Any]:
    """检查 RAG MCP 的核心运行状态。

    Returns:
        Milvus 连接配置、collection 状态、embedding 配置和默认知识库。
    """

    settings = get_settings()
    store = _build_store()
    return {
        "ok": True,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_model_path": settings.embedding_model_path,
        "default_knowledge_base_ids": settings.default_knowledge_base_ids,
        "supported_extensions": sorted(settings.supported_extensions),
        **store.health_check(),
    }


def parse_arguments() -> argparse.Namespace:
    """解析 MCP 传输层启动参数。"""

    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--sse", action="store_true", help="使用 SSE transport")
    parser.add_argument("--streamable-http", action="store_true", help="使用 streamable-http transport")
    parser.add_argument("--stateless", action="store_true", help="streamable-http 下启用无状态模式")
    parser.add_argument("--host", type=str, default="localhost", help="HTTP transport 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="HTTP transport 监听端口")
    return parser.parse_args()


def main() -> None:
    """启动 RAG MCP 服务。"""

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
