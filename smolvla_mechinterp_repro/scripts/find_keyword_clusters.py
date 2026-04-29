#!/usr/bin/env python3
"""
Phase 2 keyword-based concept candidate extraction for SmolVLA value vectors.

This script scans the value-vector catalog and scores each vector against a
small set of concept families such as fast/slow/high/low/up/safe/risk.

The goal is not to do final clustering yet. Instead, it gives us:
- counts of concept hits by layer,
- top candidate vectors per concept,
- an inspectable seed list for later steering and clustering work.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catalog_analysis_utils import (
    alpha_only,
    iter_catalog_records,
    normalize_token_text,
    simple_stem,
    weighted_rank_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "results" / "value_vector_catalog_top30.jsonl"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "keyword_concept_candidates.json"
DEFAULT_OUTPUT_MD = ROOT / "results" / "keyword_concept_candidates.md"

DEFAULT_CONCEPTS: dict[str, list[str]] = {
    "fast": ["fast", "faster", "fastest", "quick", "quicker", "quickly", "swift"],
    "slow": ["slow", "slower", "slowest", "slowly", "calm", "gentle"],
    "high": ["high", "higher", "highest"],
    "low": ["low", "lower", "lowest"],
    "up": ["up", "upper", "upward", "upwards", "above"],
    "safe": ["safe", "safer", "safely", "safety"],
    "risk": ["risk", "risks", "risky"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find keyword-aligned value-vector candidates from the SmolVLA catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog-jsonl",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Input value-vector catalog.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the structured concept summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Where to write the human-readable report.",
    )
    parser.add_argument(
        "--top-token-limit",
        type=int,
        default=30,
        help="Only inspect the first N decoded tokens for each vector.",
    )
    parser.add_argument(
        "--top-candidates-per-concept",
        type=int,
        default=100,
        help="How many top-scoring vectors to keep per concept.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N vectors.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=",".join(DEFAULT_CONCEPTS.keys()),
        help="Comma-separated concept names to analyze.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def token_matches_alias(token: str, alias: str) -> bool:
    norm = normalize_token_text(token)
    alpha = alpha_only(token)
    stem = simple_stem(token)
    alias_norm = alias.lower()
    return norm == alias_norm or alpha == alias_norm or stem == alias_norm


def find_concept_hits(tokens: list[str], aliases: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rank, token in enumerate(tokens):
        matched_aliases = [alias for alias in aliases if token_matches_alias(token, alias)]
        if matched_aliases:
            hits.append(
                {
                    "rank": rank,
                    "token": token,
                    "aliases": matched_aliases,
                    "normalized": normalize_token_text(token),
                    "alpha_only": alpha_only(token),
                    "stem": simple_stem(token),
                }
            )
    return hits


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        candidate["match_count"],
        candidate["weighted_rank_score"],
        -candidate["best_rank"],
        candidate["value_vector_norm"],
    )


def write_markdown_report(
    output_md: Path,
    *,
    catalog_path: Path,
    top_token_limit: int,
    concept_summaries: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Keyword Concept Candidates")
    lines.append("")
    lines.append(f"- Catalog: `{catalog_path}`")
    lines.append(f"- Top-token limit: `{top_token_limit}`")
    lines.append("")

    for concept, summary in concept_summaries.items():
        lines.append(f"## {concept}")
        lines.append("")
        lines.append(f"- Candidate vectors: `{summary['candidate_count']}`")
        lines.append(f"- Layers with at least one hit: `{summary['num_layers_with_hits']}`")
        lines.append(f"- Peak layer hit count: `{summary['max_layer_hit_count']}`")
        lines.append("")
        lines.append("| rank | layer | vector | matches | best rank | score | matched tokens |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for rank, candidate in enumerate(summary["top_candidates"][:15], start=1):
            matched_tokens = ", ".join(hit["token"] for hit in candidate["hits"])
            lines.append(
                f"| {rank} | {candidate['layer_idx']} | {candidate['vector_index']} | "
                f"{candidate['match_count']} | {candidate['best_rank']} | "
                f"{candidate['weighted_rank_score']:.3f} | {matched_tokens} |"
            )
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    catalog_path = args.catalog_jsonl.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    ensure_parent(output_json)
    ensure_parent(output_md)

    requested_concepts = [item.strip() for item in args.concepts.split(",") if item.strip()]
    unknown = [item for item in requested_concepts if item not in DEFAULT_CONCEPTS]
    if unknown:
        raise SystemExit(f"Unknown concepts requested: {unknown}")

    concept_aliases = {name: DEFAULT_CONCEPTS[name] for name in requested_concepts}
    concept_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in concept_aliases}
    layer_hit_counts: dict[str, Counter[int]] = {name: Counter() for name in concept_aliases}

    total_records = 0
    for record in iter_catalog_records(catalog_path):
        total_records += 1
        tokens = record["top_tokens"][: args.top_token_limit]
        for concept, aliases in concept_aliases.items():
            hits = find_concept_hits(tokens, aliases)
            if not hits:
                continue
            layer_hit_counts[concept][record["layer_idx"]] += 1
            concept_candidates[concept].append(
                {
                    "layer_idx": record["layer_idx"],
                    "vector_index": record["vector_index"],
                    "global_vector_index": record["global_vector_index"],
                    "layer_path": record["layer_path"],
                    "match_count": len(hits),
                    "best_rank": min(hit["rank"] for hit in hits),
                    "weighted_rank_score": weighted_rank_score([hit["rank"] for hit in hits]),
                    "hits": hits,
                    "top_tokens_preview": tokens[:10],
                    "value_vector_norm": record["value_vector_norm"],
                }
            )
        if args.progress_every and total_records % args.progress_every == 0:
            print(f"processed {total_records} vectors", flush=True)

    concept_summaries: dict[str, Any] = {}
    for concept, candidates in concept_candidates.items():
        sorted_candidates = sorted(candidates, key=candidate_sort_key, reverse=True)
        top_candidates = sorted_candidates[: args.top_candidates_per_concept]
        per_layer = [
            {"layer_idx": int(layer_idx), "hit_count": int(hit_count)}
            for layer_idx, hit_count in sorted(layer_hit_counts[concept].items())
        ]
        concept_summaries[concept] = {
            "aliases": concept_aliases[concept],
            "candidate_count": len(candidates),
            "num_layers_with_hits": len(layer_hit_counts[concept]),
            "max_layer_hit_count": max(layer_hit_counts[concept].values()) if layer_hit_counts[concept] else 0,
            "layer_hit_counts": per_layer,
            "top_candidates": top_candidates,
        }

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_jsonl": str(catalog_path),
        "top_token_limit": args.top_token_limit,
        "top_candidates_per_concept": args.top_candidates_per_concept,
        "total_records_scanned": total_records,
        "concept_summaries": concept_summaries,
    }

    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown_report(
        output_md,
        catalog_path=catalog_path,
        top_token_limit=args.top_token_limit,
        concept_summaries=concept_summaries,
    )

    print(f"Wrote concept summary JSON to {output_json}")
    print(f"Wrote concept summary Markdown to {output_md}")
    for concept, summary in concept_summaries.items():
        print(
            f"{concept:>5}: candidates={summary['candidate_count']:5d} "
            f"layers={summary['num_layers_with_hits']:2d} "
            f"peak_layer_hits={summary['max_layer_hit_count']:4d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
