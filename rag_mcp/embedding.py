"""基于 LangChain 的 Embedding 封装。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings

from rag_mcp.config import Settings


class EmbeddingClient:
    """统一管理查询和文档的向量化。

    这里显式使用 LangChain 的 Embeddings 接口，后续无论切换到本地
    HuggingFace 模型、OpenAI 兼容接口，还是其他 LangChain 支持的模型，
    索引和检索流程都不需要改动。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 Embedding 客户端。

        Args:
            settings: RAG MCP 的运行配置。
        """

        self.settings = settings
        self._embedding_model: Embeddings | None = None

    def embed_query(self, query: str) -> list[float]:
        """生成单条查询文本的向量。

        Args:
            query: 用户问题或检索语句。

        Returns:
            查询文本对应的浮点向量。
        """

        return self._to_float_list(self._get_model().embed_query(query))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成文档 chunk 的向量。

        Args:
            texts: 待向量化的 chunk 文本序列。

        Returns:
            与输入顺序一致的向量列表。
        """

        if not texts:
            return []

        vectors = self._get_model().embed_documents(list(texts))
        return [self._to_float_list(vector) for vector in vectors]

    def _get_model(self) -> Embeddings:
        if self._embedding_model is not None:
            return self._embedding_model

        provider = self.settings.embedding_provider
        if provider in {"local", "local_huggingface", "huggingface"}:
            self._embedding_model = self._build_huggingface_embedding()
            return self._embedding_model

        if provider in {"openai", "openai_compatible", "api"}:
            self._embedding_model = self._build_openai_embedding()
            return self._embedding_model

        raise ValueError(f"不支持的 RAG_EMBEDDING_PROVIDER：{provider}")

    def _build_huggingface_embedding(self) -> Embeddings:
        """创建本地 HuggingFace Embedding 模型。

        Returns:
            HuggingFaceEmbeddings 实例。
        """

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("缺少 langchain-huggingface 依赖，无法加载本地 embedding 模型。") from exc

        # 使用本地路径优先，避免启动 MCP 时重复从网络拉取模型。
        return HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_path,
            model_kwargs={"device": self.settings.model_device},
            encode_kwargs={"normalize_embeddings": self.settings.embedding_normalize},
        )

    def _build_openai_embedding(self) -> Embeddings:
        """创建 OpenAI 兼容接口的 Embedding 模型。

        Returns:
            LangChain OpenAIEmbeddings 实例。
        """

        if not self.settings.embedding_base_url or not self.settings.embedding_api_key:
            raise RuntimeError("使用 OpenAI 兼容 embedding 时必须配置 RAG_EMBEDDING_BASE_URL 和 RAG_EMBEDDING_API_KEY。")

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("缺少 langchain-openai 依赖，无法调用 OpenAI 兼容 embedding。") from exc

        return OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.embedding_api_key,
            base_url=self.settings.embedding_base_url.rstrip("/"),
            timeout=self.settings.request_timeout_seconds,
        )

    def _to_float_list(self, vector: Any) -> list[float]:
        """把 numpy、torch 或普通列表统一转换为 Python float 列表。"""

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        return [float(value) for value in vector]
