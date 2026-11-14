"""Exact-evidence retrieval evaluation; human labels remain a separate audit."""
from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Mark a hit only when the gold source/unit and complete quote both match."""
    gold_units = set(question.get("gold_unit_ids", [question.get("gold_unit_id")]))
    gold_units.discard(None)
    evidence_quotes = [
        question["evidence_quote"],
        *question.get("alternative_evidence_quotes", []),
    ]
    reviewed_hits = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        unit_id = str(metadata.get("unit_id", hit.get("page", "")))
        source_matches = hit.get("document_id") == question["gold_source_id"]
        unit_matches = unit_id in gold_units
        quote_matches = any(quote in hit.get("text", "") for quote in evidence_quotes)
        reviewed_hits.append({
            **hit,
            "gold_source_match": source_matches,
            "gold_unit_match": unit_matches,
            "evidence_quote_match": quote_matches,
            "evidence_hit": source_matches and unit_matches and quote_matches,
        })
    return {
        "hit_at_1": bool(reviewed_hits and reviewed_hits[0]["evidence_hit"]),
        "evidence_hit_at_3": any(hit["evidence_hit"] for hit in reviewed_hits[:3]),
        "evidence_hit_ranks": [
            hit["rank"] for hit in reviewed_hits[:3] if hit["evidence_hit"]
        ],
        "hits": reviewed_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval counts by strategy and benchmark split."""
    summary: dict[str, Any] = {}
    keys = sorted({(trace["strategy"], trace["split"]) for trace in traces})
    for strategy, split in keys:
        selected = [
            trace for trace in traces
            if trace["strategy"] == strategy and trace["split"] == split
        ]
        denominator = len(selected)
        summary.setdefault(strategy, {})[split] = {
            "questions": denominator,
            "hit_at_1": sum(bool(trace["hit_at_1"]) for trace in selected),
            "evidence_hit_at_3": sum(bool(trace["evidence_hit_at_3"]) for trace in selected),
        }
    return summary
