"""中文优先的文本分块工具。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from rag_mcp.config import Settings
from rag_mcp.schemas import Document, TextChunk


class TextChunker:
    """把长文档切分成适合向量检索的重叠 chunk。

    分块策略来自 CookingAgent 原有 RAG 工具：优先保留段落、标题和列表结构；
    当段落过长时再按中文句末标点切分，最后才使用硬切分兜底。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化分块器。

        Args:
            settings: 包含 chunk 大小、最大长度和重叠长度的运行配置。
        """

        self.settings = settings
        self.target_size = settings.chunk_target_size
        self.max_size = max(settings.chunk_target_size, settings.chunk_max_size)
        self.overlap_size = min(settings.chunk_overlap_size, self.target_size // 2)

    def split(self, document: Document, knowledge_base_id: str) -> list[TextChunk]:
        """切分单个文档。

        Args:
            document: 待切分的文本文档。
            knowledge_base_id: 写入 Milvus 时使用的知识库 ID。

        Returns:
            按原文顺序排列的 chunk 列表。
        """

        cleaned_text = self._clean_text(document.content)
        if not cleaned_text:
            return []

        paragraphs = self._split_paragraphs(cleaned_text)
        raw_chunks = self._merge_paragraphs(paragraphs)

        chunks: list[TextChunk] = []
        for index, content in enumerate(raw_chunks):
            chunk_id = self._build_chunk_id(document.document_id, knowledge_base_id, index, content)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    knowledge_base_id=knowledge_base_id,
                    document_id=document.document_id,
                    document_title=document.title,
                    source_path=document.source_path,
                    chunk_index=index,
                    content=content,
                    page_no=self._coerce_page_no(document.metadata.get("page_no")),
                    metadata=dict(document.metadata),
                )
            )

        return chunks

    def _clean_text(self, text: str) -> str:
        """统一换行和空白，但保留段落边界。"""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _split_paragraphs(self, text: str) -> list[str]:
        """优先按空行切段；超长段落再按句子切开。"""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        refined: list[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.max_size:
                refined.append(paragraph)
            else:
                refined.extend(self._split_long_paragraph(paragraph))

        return refined

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """把超长段落按中文/英文句末标点拆开。"""

        sentences = [part for part in re.split(r"(?<=[。！？!?；;])", paragraph) if part.strip()]
        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self.max_size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._hard_split(sentence))
                continue

            candidate = f"{current}{sentence}" if current else sentence
            if len(candidate) <= self.max_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = sentence

        if current:
            pieces.append(current)

        return pieces or self._hard_split(paragraph)

    def _hard_split(self, text: str) -> list[str]:
        """保证极端长文本也不会超过 Milvus 字段和配置限制。"""

        return [
            text[start : start + self.max_size].strip()
            for start in range(0, len(text), self.max_size)
            if text[start : start + self.max_size].strip()
        ]

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """把段落合并到目标长度，并把上一块末尾带入下一块。"""

        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            separator_length = 2 if current_parts else 0
            candidate_length = current_length + separator_length + len(paragraph)

            if current_parts and candidate_length > self.target_size:
                current_chunk = "\n\n".join(current_parts).strip()
                chunks.append(current_chunk)
                current_parts = self._build_overlap_parts(current_chunk)
                current_length = sum(len(part) for part in current_parts) + max(0, len(current_parts) - 1) * 2

            current_parts.append(paragraph)
            current_length += (2 if current_length else 0) + len(paragraph)

            if current_length >= self.max_size:
                current_chunk = "\n\n".join(current_parts).strip()
                chunks.append(current_chunk)
                current_parts = self._build_overlap_parts(current_chunk)
                current_length = sum(len(part) for part in current_parts) + max(0, len(current_parts) - 1) * 2

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())

        return [chunk for chunk in chunks if chunk]

    def _build_overlap_parts(self, previous_chunk: str) -> list[str]:
        if self.overlap_size <= 0:
            return []

        overlap = previous_chunk[-self.overlap_size :].strip()
        return [overlap] if overlap else []

    def _coerce_page_no(self, value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_chunk_id(
        self,
        document_id: str,
        knowledge_base_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        payload = f"{knowledge_base_id}\0{document_id}\0{chunk_index}\0{content}"
        return f"rag_chunk_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"
