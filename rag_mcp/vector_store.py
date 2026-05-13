"""JSONL-backed vector store for standalone RAG."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from rag_mcp.schemas import SearchResult, StoredChunk, TextChunk


class JsonlVectorStore:
    """Persist chunks as JSON Lines and run cosine search in-process."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        overwrite_knowledge_base: bool = False,
    ) -> dict[str, Any]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致。")

        existing_chunks = self.load_chunks()
        if overwrite_knowledge_base and chunks:
            target_kb = chunks[0].knowledge_base_id
            existing_chunks = [
                chunk for chunk in existing_chunks if chunk.knowledge_base_id != target_kb
            ]

        by_chunk_id = {chunk.chunk_id: chunk for chunk in existing_chunks}
        for chunk, embedding in zip(chunks, embeddings):
            by_chunk_id[chunk.chunk_id] = StoredChunk(
                **asdict(chunk),
                embedding=embedding,
            )

        self._write_chunks(list(by_chunk_id.values()))
        return {
            "ok": True,
            "stored_chunks": len(chunks),
            "total_chunks": len(by_chunk_id),
            "store_path": str(self.store_path),
        }

    def search(
        self,
        query_embedding: list[float],
        knowledge_base_ids: list[str],
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        selected_ids = set(knowledge_base_ids)
        results: list[SearchResult] = []

        for chunk in self.load_chunks():
            if selected_ids and chunk.knowledge_base_id not in selected_ids:
                continue

            score = self._cosine_similarity(query_embedding, chunk.embedding)
            if score < min_score:
                continue

            results.append(SearchResult(chunk=chunk, score=score))

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def load_chunks(self) -> list[StoredChunk]:
        if not self.store_path.exists():
            return []

        chunks: list[StoredChunk] = []
        with self.store_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                chunks.append(StoredChunk(**data))

        return chunks

    def list_knowledge_bases(self) -> dict[str, Any]:
        counts = Counter(chunk.knowledge_base_id for chunk in self.load_chunks())
        knowledge_bases = [
            {"knowledge_base_id": knowledge_base_id, "chunk_count": count}
            for knowledge_base_id, count in sorted(counts.items())
        ]
        return {
            "ok": True,
            "store_path": str(self.store_path),
            "knowledge_bases": knowledge_bases,
        }

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any]:
        normalized_id = (knowledge_base_id or "").strip()
        if not normalized_id:
            return {
                "ok": False,
                "error": {"code": "EMPTY_KNOWLEDGE_BASE_ID", "message": "知识库 ID 不能为空。"},
            }

        chunks = self.load_chunks()
        kept_chunks = [chunk for chunk in chunks if chunk.knowledge_base_id != normalized_id]
        deleted_count = len(chunks) - len(kept_chunks)
        self._write_chunks(kept_chunks)

        return {
            "ok": True,
            "knowledge_base_id": normalized_id,
            "deleted_chunks": deleted_count,
            "remaining_chunks": len(kept_chunks),
        }

    def _write_chunks(self, chunks: list[StoredChunk]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self.store_path.parent),
            newline="\n",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            for chunk in sorted(chunks, key=lambda item: item.chunk_id):
                temp_file.write(json.dumps(asdict(chunk), ensure_ascii=False))
                temp_file.write("\n")

        temp_path.replace(self.store_path)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)
