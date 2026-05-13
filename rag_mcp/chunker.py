"""Text chunking for local RAG indexing."""

from __future__ import annotations

import hashlib
import re

from rag_mcp.config import Settings
from rag_mcp.schemas import Document, TextChunk


class TextChunker:
    """Split documents into overlapping chunks while preserving paragraph boundaries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def split(self, document: Document, knowledge_base_id: str) -> list[TextChunk]:
        paragraphs = self._split_paragraphs(document.content)
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
                    metadata=document.metadata,
                )
            )

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [
            " ".join(part.strip().split())
            for part in re.split(r"\n\s*\n|\n", normalized_text)
        ]
        return [paragraph for paragraph in paragraphs if paragraph]

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        chunks: list[str] = []
        current_parts: list[str] = []
        current_size = 0

        for paragraph in paragraphs:
            if len(paragraph) > self.settings.chunk_target_size:
                self._flush_current(chunks, current_parts)
                current_parts = []
                current_size = 0
                chunks.extend(self._split_long_text(paragraph))
                continue

            next_size = current_size + len(paragraph) + (1 if current_parts else 0)
            if current_parts and next_size > self.settings.chunk_target_size:
                self._flush_current(chunks, current_parts)
                overlap_text = self._build_overlap(chunks[-1])
                current_parts = [overlap_text] if overlap_text else []
                current_size = len(overlap_text)

            current_parts.append(paragraph)
            current_size += len(paragraph) + (1 if current_size else 0)

        self._flush_current(chunks, current_parts)
        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        step = max(1, self.settings.chunk_target_size - self.settings.chunk_overlap_size)

        while start < len(text):
            end = start + self.settings.chunk_target_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        return chunks

    def _build_overlap(self, text: str) -> str:
        if self.settings.chunk_overlap_size <= 0:
            return ""

        return text[-self.settings.chunk_overlap_size :].strip()

    def _flush_current(self, chunks: list[str], current_parts: list[str]) -> None:
        content = "\n".join(part for part in current_parts if part).strip()
        if content:
            chunks.append(content)

    def _build_chunk_id(
        self,
        document_id: str,
        knowledge_base_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        payload = f"{knowledge_base_id}\0{document_id}\0{chunk_index}\0{content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
