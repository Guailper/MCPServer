"""Command line helper for indexing local data into the RAG Milvus collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_mcp.application import get_application


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local files into the RAG knowledge base.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(PROJECT_ROOT / "data"),
        help="File or directory to index. Defaults to the project's data directory.",
    )
    parser.add_argument(
        "--knowledge-base-id",
        "--kb",
        default=None,
        help="Target knowledge base id. Defaults to RAG_DEFAULT_KNOWLEDGE_BASE_IDS[0].",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only index files directly under the target directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing chunks in the target knowledge base before inserting new chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    target_path = Path(args.path).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"入库路径不存在：{target_path}")

    app = get_application()
    result = app.indexer.index_path(
        path=str(target_path),
        knowledge_base_id=args.knowledge_base_id,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
    )
    print(_to_pretty_json(result))


def _to_pretty_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
