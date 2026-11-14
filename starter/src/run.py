"""Run the starter's A/B/C ingestion and retrieval benchmark."""
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import FastEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_text_units
from .models import Chunk, DocumentPage
from .store import PgVectorStore

STARTER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = STARTER_DIR / "data"
BENCHMARK_PATH = STARTER_DIR / "benchmark" / "questions.jsonl"
RESULTS_DIR = STARTER_DIR / "results"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STRATEGIES = ("A", "B", "C")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_benchmark(
    questions: list[dict[str, Any]],
    pages: list[DocumentPage],
) -> None:
    if len(questions) < 20:
        raise ValueError(f"Expected at least 20 questions; found {len(questions)}")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Benchmark question IDs must be unique")

    split_counts = {
        split: sum(question["split"] == split for question in questions)
        for split in ("dev", "test")
    }
    if split_counts["dev"] < 15 or split_counts["test"] < 5:
        raise ValueError(f"Expected at least 15 dev / 5 test questions; found {split_counts}")

    by_unit = {str(page.metadata["unit_id"]): page for page in pages}
    filtered = [question for question in questions if question.get("filter")]
    if len(filtered) < 3 or not any(question["split"] == "test" for question in filtered):
        raise ValueError("At least 3 questions must use filters, including one test question")

    source_ids = {page.document_id for page in pages}
    for question in questions:
        if question["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid split {question['split']!r} in {question['id']}")
        if question["gold_source_id"] not in source_ids:
            raise ValueError(f"Unknown gold source in {question['id']}")
        unit = by_unit.get(question["gold_unit_id"])
        if unit is None:
            raise ValueError(f"Unknown gold unit in {question['id']}: {question['gold_unit_id']}")
        if question["evidence_quote"] not in unit.text:
            raise ValueError(
                f"Evidence for {question['id']} was not found verbatim in "
                f"{question['gold_unit_id']}"
            )
        filters = question.get("filter") or {}
        if filters and not question.get("filter_context"):
            raise ValueError(f"Filtered question {question['id']} lacks filter context")
        for field, value in filters.items():
            if unit.metadata.get(field) != value:
                raise ValueError(f"Filter metadata mismatch for {question['id']}: {field}")


def _chunk_stats(chunks: list[Chunk], token_counts: list[int]) -> dict[str, Any]:
    if len(chunks) != len(token_counts):
        raise ValueError(
            f"Expected one token count per chunk; got {len(token_counts)} counts "
            f"for {len(chunks)} chunks"
        )
    lengths = [len(chunk.text) for chunk in chunks]
    token_lengths = token_counts
    return {
        "chunk_count": len(chunks),
        "total_characters": sum(lengths),
        "min_characters": min(lengths) if lengths else 0,
        "median_characters": statistics.median(lengths) if lengths else 0,
        "mean_characters": round(statistics.mean(lengths), 2) if lengths else 0,
        "max_characters": max(lengths) if lengths else 0,
        "total_tokens": sum(token_lengths),
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "median_tokens": statistics.median(token_lengths) if token_lengths else 0,
        "mean_tokens": round(statistics.mean(token_lengths), 2) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
    }


def _matches_metadata(actual: dict[str, Any], requested: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in requested.items())


def _write_audit_templates(
    pages: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
    traces: list[dict[str, Any]],
) -> None:
    by_unit = {str(page.metadata["unit_id"]): page for page in pages}
    unit_ids = list(by_unit)
    sample_indexes = [round(index * (len(unit_ids) - 1) / 9) for index in range(10)]
    coherence_rows: list[dict[str, Any]] = []
    for sample_number, unit_index in enumerate(sample_indexes, start=1):
        unit_id = unit_ids[unit_index]
        source_sample = by_unit[unit_id].text[:100]
        for strategy in STRATEGIES:
            chunk = next(
                (
                    candidate for candidate in chunks_by_strategy[strategy]
                    if candidate.metadata.get("unit_id") == unit_id
                    and source_sample[:60] in candidate.text
                ),
                None,
            )
            if chunk is None:
                raise RuntimeError(f"No {strategy} chunk contains preselected sample {unit_id}")
            coherence_rows.append({
                "sample_id": f"coherence-{sample_number:02d}",
                "strategy": strategy,
                "unit_id": unit_id,
                "source_sample": source_sample,
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.text,
                "score_0_to_2": None,
                "rationale": None,
            })

    test_traces = [trace for trace in traces if trace["split"] == "test"]
    grounding_rows = [
        {
            "question_id": trace["question_id"],
            "strategy": trace["strategy"],
            "question": trace["question"],
            "expected_answer": trace["expected_answer"],
            "evidence_quote": trace["evidence_quote"],
            "hits": trace["hits"],
            "score_0_to_2": None,
            "rationale": None,
        }
        for trace in test_traces
    ]
    metadata_rows = [
        {
            "question_id": trace["question_id"],
            "strategy": trace["strategy"],
            "chunk_id": hit["chunk_id"],
            "actual_source": hit["source"],
            "actual_unit_id": (hit.get("metadata") or {}).get("unit_id", hit["page"]),
            "actual_chapter": (hit.get("metadata") or {}).get("chapter"),
            "requested_filter": trace["filter"],
            "metadata_correct_and_complete": None,
            "notes": None,
        }
        for trace in test_traces
        for hit in trace["hits"]
    ]

    _write_jsonl(RESULTS_DIR / "coherence_audit_template.jsonl", coherence_rows)
    _write_jsonl(RESULTS_DIR / "grounding_audit_template.jsonl", grounding_rows)
    _write_jsonl(RESULTS_DIR / "metadata_audit_template.jsonl", metadata_rows)
    (RESULTS_DIR / "manual_audit_instructions.txt").write_text(
        "Complete these labels by manually reviewing source and retrieved chunks.\n"
        "Coherence: 0 = cut/unclear; 1 = partly understandable or missing context; "
        "2 = self-contained and preserves the idea.\n"
        "Grounding: 0 = no evidence; 1 = incomplete evidence/context; "
        "2 = enough evidence to answer and trace to source.\n"
        "Metadata: compare source, unit/page, chapter and requested filter to the source; "
        "record errors in notes. Blank labels are intentional; retrieval matches are not "
        "human assessment labels.\n"
        "Coherence sample: 10 evenly spaced units in source order, same for all strategies. "
        "Grounding/metadata sample: all 5 held-out test questions, top-3 per strategy.\n",
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv(STARTER_DIR / ".env")
    corpus_path = DATA_DIR / "alice_units.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    pages = load_text_units(corpus_path)
    questions = _read_jsonl(BENCHMARK_PATH)
    _validate_benchmark(questions, pages)
    if len(pages) < 30:
        raise ValueError(f"Corpus has {len(pages)} units; at least 30 are required")
    short_units = [
        str(page.metadata["unit_id"])
        for page in pages
        if len(page.text) < 1500
    ]
    if short_units:
        raise ValueError(
            "Normalized corpus units must each contain at least 1,500 characters; "
            f"short units: {', '.join(short_units)}"
        )

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing; configure it in starter/.env")
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    model_started = time.perf_counter()
    embedder = FastEmbedder(model_name)
    model_initialization_seconds = time.perf_counter() - model_started
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()
    backend_info = store.backend_info()

    chunking_started = time.perf_counter()
    fixed_chunks = [
        chunk for page in pages
        for chunk in fixed_size(page, size=500, overlap=50)
    ]
    fixed_chunking_seconds = time.perf_counter() - chunking_started
    chunking_started = time.perf_counter()
    paragraph_chunks = [
        chunk for page in pages
        for chunk in by_paragraph(page, max_size=1000)
    ]
    paragraph_chunking_seconds = time.perf_counter() - chunking_started
    chunks_by_strategy = {
        "A": fixed_chunks,
        "B": paragraph_chunks,
        "C": paragraph_chunks,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    strategy_stats: dict[str, Any] = {}
    vectors_by_strategy: dict[str, list[list[float]]] = {}
    token_counts_by_strategy: dict[str, list[int]] = {}
    for strategy in ("A", "B"):
        chunks = chunks_by_strategy[strategy]
        tokenization_started = time.perf_counter()
        token_counts = embedder.count_tokens([chunk.text for chunk in chunks])
        tokenization_seconds = time.perf_counter() - tokenization_started
        started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        embedding_seconds = time.perf_counter() - started
        storage_started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        storage_seconds = time.perf_counter() - storage_started
        vectors_by_strategy[strategy] = vectors
        token_counts_by_strategy[strategy] = token_counts
        chunking_seconds = (
            fixed_chunking_seconds if strategy == "A" else paragraph_chunking_seconds
        )
        strategy_stats[strategy] = {
            **_chunk_stats(chunks, token_counts),
            "chunking_seconds": round(chunking_seconds, 4),
            "tokenization_seconds": round(tokenization_seconds, 4),
            "embedding_seconds": round(embedding_seconds, 4),
            "storage_seconds": round(storage_seconds, 4),
            "ingest_seconds": round(
                chunking_seconds + tokenization_seconds + embedding_seconds + storage_seconds,
                4,
            ),
            "cost_usd": 0,
        }

    chunks_c = chunks_by_strategy["C"]
    storage_started = time.perf_counter()
    store.upsert("C", chunks_c, vectors_by_strategy["B"])
    storage_seconds = time.perf_counter() - storage_started
    strategy_stats["C"] = {
        **_chunk_stats(chunks_c, token_counts_by_strategy["B"]),
        "chunking_seconds": 0,
        "tokenization_seconds": 0,
        "embedding_seconds": 0,
        "storage_seconds": round(storage_seconds, 4),
        "ingest_seconds": round(storage_seconds, 4),
        "cost_usd": 0,
        "chunking_reused_from": "B (identical chunks)",
        "tokenization_reused_from": "B (identical chunks)",
        "embedding_reused_from": "B (identical chunks)",
    }

    query_embedding_started = time.perf_counter()
    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in questions
    }
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    traces: list[dict[str, Any]] = []
    retrieval_seconds_by_strategy = {strategy: 0.0 for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for question in questions:
            requested_filter = question.get("filter") if strategy == "C" else None
            started = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=requested_filter,
            )
            retrieval_ms = (time.perf_counter() - started) * 1000
            retrieval_seconds_by_strategy[strategy] += retrieval_ms / 1000
            ranked_hits = [
                {**hit, "rank": rank}
                for rank, hit in enumerate(hits, start=1)
            ]
            evaluation = evaluate_query(question, ranked_hits)
            filter_compliant = None
            if strategy == "C" and question.get("filter"):
                filter_compliant = bool(ranked_hits) and all(
                    _matches_metadata(hit.get("metadata") or {}, question["filter"])
                    for hit in ranked_hits
                )
            traces.append({
                "question_id": question["id"],
                "split": question["split"],
                "type": question["type"],
                "strategy": strategy,
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "gold_source_id": question["gold_source_id"],
                "gold_unit_id": question["gold_unit_id"],
                "evidence_quote": question["evidence_quote"],
                "filter": question.get("filter"),
                "filter_applied": requested_filter,
                "filter_compliant": filter_compliant,
                "retrieval_ms": round(retrieval_ms, 3),
                "hit_at_1": evaluation["hit_at_1"],
                "evidence_hit_at_3": evaluation["evidence_hit_at_3"],
                "evidence_hit_ranks": evaluation["evidence_hit_ranks"],
                "hits": evaluation["hits"],
            })

    retrieval_summary = summarize(traces)
    for strategy in STRATEGIES:
        strategy_retrievals = [
            trace["retrieval_ms"] for trace in traces if trace["strategy"] == strategy
        ]
        strategy_stats[strategy]["retrieval_seconds"] = round(
            retrieval_seconds_by_strategy[strategy], 4
        )
        strategy_stats[strategy]["mean_retrieval_ms"] = round(
            statistics.mean(strategy_retrievals), 3
        )
        test_filtered = [
            trace for trace in traces
            if trace["strategy"] == strategy
            and trace["split"] == "test"
            and trace["filter"]
        ]
        retrieval_summary[strategy]["test"]["filtered_questions"] = len(test_filtered)
        retrieval_summary[strategy]["test"]["filter_compliant"] = (
            sum(bool(trace["filter_compliant"]) for trace in test_filtered)
            if strategy == "C"
            else None
        )

    _write_jsonl(RESULTS_DIR / "retrieval_traces.jsonl", traces)
    _write_json(RESULTS_DIR / "strategy_metrics.json", {
        "retrieval": retrieval_summary,
        "strategies": strategy_stats,
        "manual_assessment": "Pending; complete the generated audit templates.",
    })
    _write_json(RESULTS_DIR / "run_manifest.json", {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus_path.relative_to(STARTER_DIR)),
        "benchmark": str(BENCHMARK_PATH.relative_to(STARTER_DIR)),
        "question_count": len(questions),
        "benchmark_language": "English",
        "test_split_note": (
            "The same five test question/evidence pairs were evaluated previously "
            "with Vietnamese question text. Treat this English run as exploratory, "
            "not as an untouched independent holdout."
        ),
        "corpus_units": len(pages),
        "total_normalized_characters": sum(len(page.text) for page in pages),
        "minimum_normalized_unit_characters": min(len(page.text) for page in pages),
        "normalized_unit_characters": [
            {"unit_id": page.metadata["unit_id"], "char_count": len(page.text)}
            for page in pages
        ],
        "split_counts": {
            split: sum(question["split"] == split for question in questions)
            for split in ("dev", "test")
        },
        "embedding_provider": "FastEmbed (local)",
        "embedding_model": embedder.model_name,
        "embedding_package_version": embedder.package_version,
        "embedding_dimensions": embedder.dimensions,
        "document_query_handling": "Same model; embed() for documents and query_embed() for queries.",
        "model_initialization_seconds": round(model_initialization_seconds, 4),
        "query_embedding_seconds": round(query_embedding_seconds, 4),
        "ingest_timing_scope": (
            "Chunking, tokenization, document embedding and vector-store writes; "
            "model initialization is reported separately. Strategy C reuses B's "
            "chunks, token counts and vectors, so its ingest time is incremental storage."
        ),
        "token_count_method": "FastEmbed TextEmbedding.token_count",
        "vector_store": "PostgreSQL + pgvector",
        **backend_info,
        "distance_metric": "cosine",
        "search_mode": "Exact; no approximate vector index created",
        "metadata_filters": ["chapter", "section"],
        "strategies": {
            "A": "500-character fixed chunks, overlap 50, no filter",
            "B": "Paragraph-aware chunks up to 1,000 characters, no filter",
            "C": "Same chunks as B, apply question-provided metadata filter",
        },
        "cost_usd": 0,
    })
    _write_audit_templates(pages, chunks_by_strategy, traces)

    for strategy in STRATEGIES:
        print(f"{strategy}: {strategy_stats[strategy]['chunk_count']} chunks")
        for split in ("dev", "test"):
            metrics = retrieval_summary[strategy][split]
            print(
                f"  {split}: Hit@1 {metrics['hit_at_1']}/{metrics['questions']}; "
                f"Evidence Hit@3 {metrics['evidence_hit_at_3']}/{metrics['questions']}"
            )
    print(f"Results saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
