"""RAG 索引和检索流程中共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """待索引的纯文本文档。

    Attributes:
        document_id: 文档唯一 ID。文件索引时由路径和内容生成；文本索引时可由调用方传入。
        title: 文档标题，用于检索结果展示。
        source_path: 来源路径。直接索引文本时可为空字符串。
        content: 文档正文。
        metadata: 附加元数据，会随 chunk 一起写入 Milvus。
    """

    document_id: str
    title: str
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """由文档切分得到的文本块。

    Attributes:
        chunk_id: chunk 唯一 ID，写入 Milvus 的 chunk_public_id。
        knowledge_base_id: 知识库 ID，用于多知识库过滤。
        document_id: 所属文档 ID。
        document_title: 所属文档标题。
        source_path: 所属文档路径。
        chunk_index: chunk 在文档内的顺序。
        content: chunk 文本内容。
        page_no: 页码；普通文本没有页码时为 None。
        metadata: 附加元数据。
    """

    chunk_id: str
    knowledge_base_id: str
    document_id: str
    document_title: str
    source_path: str
    chunk_index: int
    content: str
    page_no: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Milvus 检索返回的一条结果。

    Attributes:
        chunk: 命中的文本块。
        score: Milvus 返回的相似度分数。
    """

    chunk: TextChunk
    score: float | None
