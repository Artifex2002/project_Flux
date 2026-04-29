#!/usr/bin/env python3
"""
Phase 2 lightweight pattern analysis for SmolVLA value vectors.

This is a heuristic approximation of the paper's "meaningful if 4 of the top 30
tokens share a pattern" workflow. Because we only have token strings and not a
human annotation pass, the output is best treated as a triage tool:

- meaningful_pattern: at least one strong repeated token pattern appears,
- semantic_guess: the pattern looks word-level and human-readable,
- non_semantic_guess: the pattern is mostly prefix-fragment, special-token, or
  otherwise surface-form driven.
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
    is_action_token,
    is_angle_token,
    is_wordish_token,
    iter_catalog_records,
    normalize_token_text,
    simple_stem,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "results" / "value_vector_catalog_top30.jsonl"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "value_vector_pattern_summary.json"
DEFAULT_OUTPUT_MD = ROOT / "results" / "value_vector_pattern_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze lightweight token-pattern structure in SmolVLA value vectors.",
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
        help="Where to write the structured summary.",
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
        help="Only inspect the first N tokens from each record.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=4,
        help="Minimum repeated support required to call a pattern meaningful.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=4,
        help="Prefix length used for surface-form pattern checks.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=20,
        help="How many example vectors to retain for each pattern category.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N records.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def classify_record(tokens: list[str], *, min_support: int, prefix_len: int) -> dict[str, Any]:
    norm_tokens = [normalize_token_text(token) for token in tokens]
    alpha_tokens = [alpha_only(token) for token in tokens]
    wordish_tokens = [token for token in tokens if is_wordish_token(token)]
    stem_tokens = [simple_stem(token) for token in tokens if len(alpha_only(token)) >= 3]

    exact_counts = Counter(token for token in norm_tokens if token)
    prefix_counts = Counter(token[:prefix_len] for token in alpha_tokens if len(token) >= prefix_len)
    stem_counts = Counter(stem for stem in stem_tokens if len(stem) >= 3)

    action_count = sum(1 for token in tokens if is_action_token(token))
    angle_count = sum(1 for token in tokens if is_angle_token(token))

    candidates: list[dict[str, Any]] = []

    if action_count >= min_support:
        candidates.append(
            {
                "pattern_type": "action_token_family",
                "support": action_count,
                "semantic_guess": False,
                "non_semantic_guess": True,
                "pattern_value": "<Ac####>",
            }
        )
    if angle_count >= min_support:
        candidates.append(
            {
                "pattern_type": "special_token_family",
                "support": angle_count,
                "semantic_guess": False,
                "non_semantic_guess": True,
                "pattern_value": "<...>",
            }
        )

    if exact_counts:
        exact_token, exact_support = exact_counts.most_common(1)[0]
        if exact_support >= min_support:
            semantic_guess = exact_token.isalpha() and len(exact_token) >= 3
            candidates.append(
                {
                    "pattern_type": "exact_repeat_family",
                    "support": exact_support,
                    "semantic_guess": semantic_guess,
                    "non_semantic_guess": not semantic_guess,
                    "pattern_value": exact_token,
                }
            )

    if stem_counts:
        stem_value, stem_support = stem_counts.most_common(1)[0]
        if stem_support >= min_support:
            semantic_guess = len(stem_value) >= 3 and all(ch.isalpha() for ch in stem_value)
            candidates.append(
                {
                    "pattern_type": "stem_family",
                    "support": stem_support,
                    "semantic_guess": semantic_guess,
                    "non_semantic_guess": not semantic_guess,
                    "pattern_value": stem_value,
                }
            )

    if prefix_counts:
        prefix_value, prefix_support = prefix_counts.most_common(1)[0]
        if prefix_support >= min_support:
            candidates.append(
                {
                    "pattern_type": "prefix_family",
                    "support": prefix_support,
                    "semantic_guess": False,
                    "non_semantic_guess": True,
                    "pattern_value": prefix_value,
                }
            )

    if not candidates:
        return {
            "meaningful_pattern": False,
            "semantic_guess": False,
            "non_semantic_guess": False,
            "best_pattern": None,
            "action_token_present": action_count > 0,
            "special_token_present": angle_count > 0,
            "wordish_token_count": len(wordish_tokens),
        }

    priority = {
        "stem_family": 4,
        "exact_repeat_family": 3,
        "action_token_family": 2,
        "special_token_family": 1,
        "prefix_family": 0,
    }
    best_pattern = max(candidates, key=lambda item: (item["support"], priority[item["pattern_type"]]))

    return {
        "meaningful_pattern": True,
        "semantic_guess": bool(best_pattern["semantic_guess"]),
        "non_semantic_guess": bool(best_pattern["non_semantic_guess"]),
        "best_pattern": best_pattern,
        "action_token_present": action_count > 0,
        "special_token_present": angle_count > 0,
        "wordish_token_count": len(wordish_tokens),
    }


def write_markdown_report(
    output_md: Path,
    *,
    catalog_path: Path,
    summary: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Value Vector Pattern Summary")
    lines.append("")
    lines.append(f"- Catalog: `{catalog_path}`")
    lines.append(f"- Top-token limit: `{summary['top_token_limit']}`")
    lines.append(f"- Min support: `{summary['min_support']}`")
    lines.append("")
    overall = summary["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Total vectors: `{overall['total_vectors']}`")
    lines.append(f"- Meaningful-pattern vectors: `{overall['meaningful_pattern_count']}`")
    lines.append(f"- Semantic-guess vectors: `{overall['semantic_guess_count']}`")
    lines.append(f"- Non-semantic-guess vectors: `{overall['non_semantic_guess_count']}`")
    lines.append(f"- Action-token-present vectors: `{overall['action_token_present_count']}`")
    lines.append(f"- Special-token-present vectors: `{overall['special_token_present_count']}`")
    lines.append("")

    lines.append("## Layer Summary")
    lines.append("")
    lines.append("| layer | total | meaningful | semantic-ish | non-semantic-ish | action-like | special-token |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for layer_summary in summary["per_layer"]:
        lines.append(
            f"| {layer_summary['layer_idx']} | {layer_summary['total_vectors']} | "
            f"{layer_summary['meaningful_pattern_count']} | {layer_summary['semantic_guess_count']} | "
            f"{layer_summary['non_semantic_guess_count']} | {layer_summary['action_token_present_count']} | "
            f"{layer_summary['special_token_present_count']} |"
        )
    lines.append("")

    for section_name in ("semantic_examples", "non_semantic_examples", "action_like_examples"):
        pretty_name = section_name.replace("_", " ").title()
        lines.append(f"## {pretty_name}")
        lines.append("")
        lines.append("| layer | vector | pattern | support | top tokens |")
        lines.append("| --- | --- | --- | --- | --- |")
        for example in summary[section_name]:
            best_pattern = example.get("best_pattern") or {}
            lines.append(
                f"| {example['layer_idx']} | {example['vector_index']} | "
                f"{best_pattern.get('pattern_type', 'n/a')}:{best_pattern.get('pattern_value', 'n/a')} | "
                f"{best_pattern.get('support', 0)} | {', '.join(example['top_tokens_preview'])} |"
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

    overall = {
        "total_vectors": 0,
        "meaningful_pattern_count": 0,
        "semantic_guess_count": 0,
        "non_semantic_guess_count": 0,
        "action_token_present_count": 0,
        "special_token_present_count": 0,
    }
    per_layer: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "total_vectors": 0,
            "meaningful_pattern_count": 0,
            "semantic_guess_count": 0,
            "non_semantic_guess_count": 0,
            "action_token_present_count": 0,
            "special_token_present_count": 0,
        }
    )
    examples = {
        "semantic_examples": [],
        "non_semantic_examples": [],
        "action_like_examples": [],
    }
    pattern_type_counts: Counter[str] = Counter()

    for record in iter_catalog_records(catalog_path):
        overall["total_vectors"] += 1
        layer_stats = per_layer[record["layer_idx"]]
        layer_stats["total_vectors"] += 1
        tokens = record["top_tokens"][: args.top_token_limit]
        result = classify_record(tokens, min_support=args.min_support, prefix_len=args.prefix_len)

        if result["meaningful_pattern"]:
            overall["meaningful_pattern_count"] += 1
            layer_stats["meaningful_pattern_count"] += 1
            pattern_type_counts[result["best_pattern"]["pattern_type"]] += 1
        if result["semantic_guess"]:
            overall["semantic_guess_count"] += 1
            layer_stats["semantic_guess_count"] += 1
            if len(examples["semantic_examples"]) < args.examples_per_category:
                examples["semantic_examples"].append(
                    {
                        "layer_idx": record["layer_idx"],
                        "vector_index": record["vector_index"],
                        "best_pattern": result["best_pattern"],
                        "top_tokens_preview": tokens[:10],
                    }
                )
        if result["non_semantic_guess"]:
            overall["non_semantic_guess_count"] += 1
            layer_stats["non_semantic_guess_count"] += 1
            if len(examples["non_semantic_examples"]) < args.examples_per_category:
                examples["non_semantic_examples"].append(
                    {
                        "layer_idx": record["layer_idx"],
                        "vector_index": record["vector_index"],
                        "best_pattern": result["best_pattern"],
                        "top_tokens_preview": tokens[:10],
                    }
                )
        if result["action_token_present"]:
            overall["action_token_present_count"] += 1
            layer_stats["action_token_present_count"] += 1
            if len(examples["action_like_examples"]) < args.examples_per_category:
                examples["action_like_examples"].append(
                    {
                        "layer_idx": record["layer_idx"],
                        "vector_index": record["vector_index"],
                        "best_pattern": result["best_pattern"],
                        "top_tokens_preview": tokens[:10],
                    }
                )
        if result["special_token_present"]:
            overall["special_token_present_count"] += 1
            layer_stats["special_token_present_count"] += 1

        if args.progress_every and overall["total_vectors"] % args.progress_every == 0:
            print(f"processed {overall['total_vectors']} vectors", flush=True)

    per_layer_summary = []
    for layer_idx in sorted(per_layer):
        stats = per_layer[layer_idx]
        per_layer_summary.append({"layer_idx": layer_idx, **stats})

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_jsonl": str(catalog_path),
        "top_token_limit": args.top_token_limit,
        "min_support": args.min_support,
        "prefix_len": args.prefix_len,
        "overall": overall,
        "pattern_type_counts": dict(pattern_type_counts),
        "per_layer": per_layer_summary,
        **examples,
    }

    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown_report(output_md, catalog_path=catalog_path, summary=output)

    print(f"Wrote pattern summary JSON to {output_json}")
    print(f"Wrote pattern summary Markdown to {output_md}")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
