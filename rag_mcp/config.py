"""RAG MCP 服务的运行配置。

本模块只负责把环境变量转换成清晰的 Python 配置对象，不在这里做业务判断。
这样索引、检索和 MCP 工具入口都可以共享同一份配置，后续排查问题也更直接。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_int_env(name: str, default: int) -> int:
    """读取整数环境变量。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在或格式错误时使用的默认值。

    Returns:
        解析后的整数值。
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    """读取浮点数环境变量。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在或格式错误时使用的默认值。

    Returns:
        解析后的浮点数值。
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在时使用的默认值。

    Returns:
        支持 true/false、1/0、yes/no 等常见写法的布尔值。
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_csv_env(name: str, default: list[str]) -> list[str]:
    """读取逗号分隔的字符串列表。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在时使用的默认列表。

    Returns:
        去掉空白和空项后的字符串列表。
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    values = [value.strip() for value in raw_value.split(",")]
    return [value for value in values if value]


def _resolve_path(raw_path: str) -> Path:
    """把环境变量里的路径解析为绝对路径。

    Args:
        raw_path: 可能是绝对路径、相对路径或带 ~ 的路径。

    Returns:
        规范化后的绝对路径。
    """

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """索引和检索流程共享的配置。"""

    milvus_uri: str
    milvus_token: str
    milvus_database: str
    milvus_collection: str
    milvus_deployment_mode: str

    default_knowledge_base_ids: list[str]
    supported_extensions: set[str]
    max_file_size_bytes: int

    chunk_target_size: int
    chunk_max_size: int
    chunk_overlap_size: int

    vector_top_k: int
    final_top_k: int
    min_score: float
    search_mode: str
    dense_weight: float
    sparse_weight: float
    rerank_enabled: bool
    rerank_provider: str
    rerank_model: str
    rerank_model_path: str
    rerank_top_n: int
    model_device: str
    request_timeout_seconds: int
    list_limit: int

    embedding_provider: str
    embedding_model: str
    embedding_model_path: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_normalize: bool


@lru_cache
def get_settings() -> Settings:
    """加载并缓存 RAG MCP 配置。

    Returns:
        Settings: 已经完成默认值填充的不可变配置对象。
    """

    load_dotenv(PROJECT_ROOT / ".env")

    supported_extensions = {
        extension.lower()
        for extension in _get_csv_env(
            "RAG_SUPPORTED_EXTENSIONS",
            [".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm", ".csv", ".xlsx", ".json", ".jsonl"],
        )
    }

    default_kb_ids = _get_csv_env("RAG_DEFAULT_KNOWLEDGE_BASE_IDS", ["default"])

    return Settings(
        milvus_uri=os.getenv("MILVUS_URI", os.getenv("MILVUS_URL", "http://127.0.0.1:19530")).strip(),
        milvus_token=os.getenv("MILVUS_TOKEN", "").strip(),
        milvus_database=os.getenv("MILVUS_DATABASE", os.getenv("MILVUS_DB", "default")).strip(),
        milvus_collection=os.getenv("MILVUS_COLLECTION", "rag_chunks").strip(),
        milvus_deployment_mode=os.getenv("MILVUS_DEPLOYMENT_MODE", os.getenv("MILVUS_MODE", "standalone")).strip().lower(),
        default_knowledge_base_ids=default_kb_ids or ["default"],
        supported_extensions=supported_extensions,
        max_file_size_bytes=max(1024, _get_int_env("RAG_MAX_FILE_SIZE_BYTES", 2_000_000)),
        chunk_target_size=max(100, _get_int_env("RAG_CHUNK_TARGET_SIZE", 700)),
        chunk_max_size=max(100, _get_int_env("RAG_CHUNK_MAX_SIZE", 1000)),
        chunk_overlap_size=max(0, _get_int_env("RAG_CHUNK_OVERLAP_SIZE", 100)),
        vector_top_k=max(1, _get_int_env("RAG_VECTOR_TOP_K", 20)),
        final_top_k=max(1, _get_int_env("RAG_FINAL_TOP_K", 5)),
        min_score=_get_float_env("RAG_MIN_SCORE", 0.25),
        search_mode=os.getenv("RAG_SEARCH_MODE", "hybrid").strip().lower(),
        dense_weight=max(0.0, _get_float_env("RAG_DENSE_WEIGHT", 0.7)),
        sparse_weight=max(0.0, _get_float_env("RAG_SPARSE_WEIGHT", 0.3)),
        rerank_enabled=_get_bool_env("RAG_RERANK_ENABLED", True),
        rerank_provider=os.getenv("RAG_RERANK_PROVIDER", "local_cross_encoder").strip().lower(),
        rerank_model=os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base").strip(),
        rerank_model_path=str(
            _resolve_path(
                os.getenv("LOCAL_RERANK_MODEL_PATH", "../models/bge-reranker-base").strip()
            )
        ),
        rerank_top_n=max(1, _get_int_env("RAG_RERANK_TOP_N", 50)),
        model_device=os.getenv("RAG_MODEL_DEVICE", os.getenv("MODEL_DEVICE", "cuda")).strip().lower(),
        request_timeout_seconds=max(1, _get_int_env("RAG_REQUEST_TIMEOUT_SECONDS", 30)),
        list_limit=max(100, _get_int_env("RAG_LIST_LIMIT", 10_000)),
        embedding_provider=os.getenv(
            "RAG_EMBEDDING_PROVIDER",
            os.getenv("EMBEDDING_PROVIDER", "local_huggingface"),
        ).strip().lower(),

        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")).strip(),
        embedding_model_path=str(
            _resolve_path(os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "../models/bge-small-zh-v1.5").strip())
        ),
        embedding_base_url=os.getenv("RAG_EMBEDDING_BASE_URL", "").strip(),
        embedding_api_key=os.getenv("RAG_EMBEDDING_API_KEY", "").strip(),
        embedding_normalize=_get_bool_env("RAG_EMBEDDING_NORMALIZE", True),
    )
