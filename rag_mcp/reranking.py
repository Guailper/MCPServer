"""Rerank model wrapper used by the retrieval pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from rag_mcp.config import Settings
from rag_mcp.schemas import SearchResult


class _PairScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query-document pairs."""


class _SentenceTransformerCrossEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("缺少 sentence-transformers 依赖，无法加载本地 rerank 模型。") from exc

        self._model = CrossEncoder(model_name, device=device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw_scores = self._model.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        return [float(score) for score in raw_scores]


class RerankClient:
    """Lazy-load and reuse a local rerank model.

    LangChain is preferred when available because it keeps model wiring aligned
    with the embedding side. The sentence-transformers fallback uses the same
    model files and keeps the service usable in lean environments.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._scorer: _PairScorer | None = None

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not self.settings.rerank_enabled or not results:
            return results[:top_k]

        candidates = results[: max(top_k, self.settings.rerank_top_n)]
        pairs = [(query, result.chunk.content) for result in candidates]
        scores = self._get_scorer().score(pairs)

        reranked: list[SearchResult] = []
        for result, rerank_score in zip(candidates, scores, strict=True):
            metadata = dict(result.chunk.metadata)
            metadata["retrieval_score"] = result.score
            metadata["rerank_score"] = rerank_score
            reranked.append(
                SearchResult(
                    chunk=replace(result.chunk, metadata=metadata),
                    score=rerank_score,
                )
            )

        reranked.sort(key=lambda item: item.score if item.score is not None else float("-inf"), reverse=True)
        return reranked[:top_k]

    def _get_scorer(self) -> _PairScorer:
        if self._scorer is not None:
            return self._scorer

        provider = self.settings.rerank_provider
        if provider in {"none", "off", "disabled"}:
            raise RuntimeError("Rerank 已关闭，不应加载 rerank 模型。")

        if provider in {"local", "local_cross_encoder", "huggingface", "langchain"}:
            self._scorer = self._build_local_cross_encoder()
            return self._scorer

        raise ValueError(f"不支持的 RAG_RERANK_PROVIDER：{provider}")

    def _build_local_cross_encoder(self) -> _PairScorer:
        model_name = self.settings.rerank_model_path or self.settings.rerank_model

        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        except ImportError:
            return _SentenceTransformerCrossEncoder(model_name, self.settings.model_device)

        class _LangChainCrossEncoder:
            def __init__(self, wrapped_model) -> None:
                self._wrapped_model = wrapped_model

            def score(self, pairs: list[tuple[str, str]]) -> list[float]:
                raw_scores = self._wrapped_model.score(pairs)
                return [float(score) for score in raw_scores]

        return _LangChainCrossEncoder(
            HuggingFaceCrossEncoder(
                model_name=model_name,
                model_kwargs={"device": self.settings.model_device},
            )
        )
