"""Evaluate rag_mcp retrieval with simple local metrics and optional Ragas metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from rag_mcp.application import get_application


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rag_mcp retrieval with Ragas.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "eval" / "ragas_cases.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval" / "results"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument("--sparse-weight", type=float, default=None)
    parser.add_argument("--run-ragas", action="store_true", help="Run Ragas LLM-based context metrics.")
    parser.add_argument("--evaluator-model", default=os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini"))
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_arguments()

    app = get_application()
    _check_milvus(app)

    samples = _load_jsonl(Path(args.dataset))
    records = [_evaluate_one(app, sample, args) for sample in samples]
    summary = _summarize(records)

    ragas_result = None
    if args.run_ragas:
        ragas_result = _run_ragas(records, args.evaluator_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_json(output_dir / f"rag_eval_{timestamp}.json", records, summary, ragas_result)
    _write_csv(output_dir / f"rag_eval_{timestamp}.csv", records)

    print(json.dumps({"summary": summary, "ragas": ragas_result}, ensure_ascii=False, indent=2, default=str))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("question"):
                raise ValueError(f"{path}:{line_no} 缺少 question 字段。")
            rows.append(row)
    return rows


def _check_milvus(app) -> None:
    _check_milvus_port(app.settings.milvus_uri)
    try:
        health = app.store.health_check()
    except Exception as exc:
        settings = app.settings
        raise SystemExit(
            "无法连接 Milvus Standalone，请先确认服务已启动且端口可访问。\n"
            f"当前 MILVUS_URI={settings.milvus_uri}, MILVUS_DATABASE={settings.milvus_database}, "
            f"MILVUS_COLLECTION={settings.milvus_collection}\n"
            "Windows 可用命令检查：Test-NetConnection -ComputerName 127.0.0.1 -Port 19530"
        ) from exc

    if health["collection_exists"] and not health["collection_schema_compatible"]:
        missing_fields = ", ".join(health["missing_schema_fields"])
        raise SystemExit(
            f"Milvus collection {health['collection']!r} 不是新 schema，缺少字段：{missing_fields}。\n"
            "请先重新入库；入库时会自动按新 schema 重建 collection。"
        )


def _check_milvus_port(uri: str) -> None:
    parsed_uri = urlparse(uri)
    host = parsed_uri.hostname or uri.split(":")[0]
    port = parsed_uri.port or 19530
    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise SystemExit(
            f"Milvus Standalone 端口不可达：{host}:{port}\n"
            "请先启动 Milvus，或检查 .env 中的 MILVUS_URI 是否正确。"
        ) from exc


def _evaluate_one(app, sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    result = app.retriever.search(
        query=sample["question"],
        top_k=args.top_k,
        search_mode=args.search_mode,
        rerank=args.rerank,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    chunks = result["chunks"]
    retrieved_contexts = [chunk["content"] for chunk in chunks]
    source_paths = [chunk.get("source_path", "") for chunk in chunks]
    expected_paths = sample.get("expected_source_paths") or []

    return {
        "question": sample["question"],
        "reference": sample.get("reference", ""),
        "reference_contexts": sample.get("reference_contexts", []),
        "retrieved_contexts": retrieved_contexts,
        "source_paths": source_paths,
        "scores": [chunk.get("score") for chunk in chunks],
        "latency_ms": latency_ms,
        "hit": _hit(expected_paths, source_paths),
        "mrr": _mrr(expected_paths, source_paths),
    }


def _hit(expected_paths: list[str], source_paths: list[str]) -> int | None:
    if not expected_paths:
        return None
    return int(any(_contains_expected(path, expected_paths) for path in source_paths))


def _mrr(expected_paths: list[str], source_paths: list[str]) -> float | None:
    if not expected_paths:
        return None
    for index, path in enumerate(source_paths, start=1):
        if _contains_expected(path, expected_paths):
            return 1.0 / index
    return 0.0


def _contains_expected(source_path: str, expected_paths: list[str]) -> bool:
    normalized_source = source_path.replace("\\", "/").lower()
    return any(str(expected).replace("\\", "/").lower() in normalized_source for expected in expected_paths)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [record["hit"] for record in records if record["hit"] is not None]
    mrrs = [record["mrr"] for record in records if record["mrr"] is not None]
    latencies = [record["latency_ms"] for record in records]
    return {
        "count": len(records),
        "hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


def _run_ragas(records: list[dict[str, Any]], evaluator_model: str) -> dict[str, Any]:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import llm_factory
    try:
        from ragas.metrics.collections import LLMContextPrecisionWithReference, LLMContextRecall
    except ImportError:
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=record["question"],
                retrieved_contexts=record["retrieved_contexts"],
                reference=record["reference"],
                reference_contexts=record["reference_contexts"],
            )
            for record in records
        ]
    )
    llm = llm_factory(
        evaluator_model,
        base_url=os.getenv("RAGAS_EVAL_BASE_URL") or None,
        api_key=os.getenv("RAGAS_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
    )
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm,
    )
    return dict(result)


def _write_json(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    ragas_result: dict[str, Any] | None,
) -> None:
    payload = {"summary": summary, "ragas": ragas_result, "records": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "hit", "mrr", "latency_ms", "source_paths"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "question": record["question"],
                    "hit": record["hit"],
                    "mrr": record["mrr"],
                    "latency_ms": record["latency_ms"],
                    "source_paths": " | ".join(record["source_paths"]),
                }
            )


if __name__ == "__main__":
    main()
