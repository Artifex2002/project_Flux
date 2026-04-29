#!/usr/bin/env python3
"""
Analyze Phase 6 Tier 1 init-state transfer experiments.

This script reads a logged eval-grid run directory and produces paired
cluster-vs-random comparisons for each concept / alpha setting over explicit
init-state contexts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired init-state transfer effects from an eval-grid run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Eval-grid run directory containing summary.json and per-run outputs.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_displacement",
        choices=["avg_displacement", "max_height"],
        help="Per-rollout metric used for init-state transfer comparisons.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Optional comma-separated concept filter.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Optional comma-separated alpha filter.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults inside run-dir.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output Markdown path. Defaults inside run-dir.",
    )
    return parser.parse_args()


def parse_csv_str(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_float(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def classify_effect(effect: float, anchor_effect: float) -> str:
    anchor_mag = abs(anchor_effect)
    if anchor_mag < 1e-9:
        return "collapse" if abs(effect) < 1e-9 else "unstable"

    ratio = abs(effect) / anchor_mag
    if effect * anchor_effect < 0 and ratio >= 0.1:
        return "flip"
    if ratio >= 0.5:
        return "stable"
    if ratio >= 0.1:
        return "weaken"
    return "collapse"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rollout_map(run_json: dict[str, Any], metric: str) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for rollout in run_json.get("rollouts", []):
        init_state_idx = rollout.get("init_state_idx")
        if init_state_idx is None:
            continue
        mapping[int(init_state_idx)] = float(rollout[metric])
    return mapping


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary.json under {run_dir}")

    summary = load_json(summary_path)
    rows = summary.get("runs", [])
    concept_filter = set(parse_csv_str(args.concepts) or [])
    alpha_filter = set(parse_csv_float(args.alphas) or [])

    none_row = next((row for row in rows if row["condition"] == "none"), None)
    none_rollouts = {}
    if none_row is not None:
        none_rollouts = build_rollout_map(load_json(Path(none_row["output_json"])), args.metric)

    comparisons = []
    concepts = sorted({row["concept"] for row in rows if row["concept"] is not None})
    for concept in concepts:
        if concept_filter and concept not in concept_filter:
            continue
        alphas = sorted({row["alpha"] for row in rows if row["concept"] == concept and row["alpha"] is not None})
        for alpha in alphas:
            if alpha_filter and alpha not in alpha_filter:
                continue

            random_row = next(
                (
                    row
                    for row in rows
                    if row["condition"] == "random_matched" and row["concept"] == concept and row["alpha"] == alpha
                ),
                None,
            )
            cluster_row = next(
                (
                    row
                    for row in rows
                    if row["condition"] == "cluster" and row["concept"] == concept and row["alpha"] == alpha
                ),
                None,
            )
            if random_row is None or cluster_row is None:
                continue

            random_rollouts = build_rollout_map(load_json(Path(random_row["output_json"])), args.metric)
            cluster_rollouts = build_rollout_map(load_json(Path(cluster_row["output_json"])), args.metric)
            paired_init_states = sorted(set(random_rollouts) & set(cluster_rollouts))
            per_state = []
            effects = []

            for init_state_idx in paired_init_states:
                random_value = random_rollouts[init_state_idx]
                cluster_value = cluster_rollouts[init_state_idx]
                effect = cluster_value - random_value
                effects.append(effect)
                per_state.append(
                    {
                        "init_state_idx": init_state_idx,
                        "none_value": none_rollouts.get(init_state_idx),
                        "random_value": random_value,
                        "cluster_value": cluster_value,
                        "cluster_minus_random": effect,
                    }
                )

            effect_mean = mean(effects)
            effect_std = std(effects)
            for state in per_state:
                state["classification"] = classify_effect(state["cluster_minus_random"], effect_mean)

            classification_counts = Counter(state["classification"] for state in per_state)
            comparisons.append(
                {
                    "concept": concept,
                    "alpha": alpha,
                    "metric": args.metric,
                    "num_paired_init_states": len(paired_init_states),
                    "none_mean": mean(
                        [value for key, value in none_rollouts.items() if key in paired_init_states]
                    )
                    if paired_init_states
                    else None,
                    "random_mean": mean([random_rollouts[idx] for idx in paired_init_states]),
                    "cluster_mean": mean([cluster_rollouts[idx] for idx in paired_init_states]),
                    "cluster_minus_random_mean": effect_mean,
                    "cluster_minus_random_std": effect_std,
                    "classification_counts": dict(classification_counts),
                    "paired_init_states": per_state,
                }
            )

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run_dir / f"init_state_transfer_{args.metric}.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else run_dir / f"init_state_transfer_{args.metric}.md"
    )

    payload = {
        "run_dir": str(run_dir),
        "metric": args.metric,
        "comparisons": comparisons,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Init-State Transfer Summary",
        "",
        f"- Metric: `{args.metric}`",
        f"- Comparisons: `{len(comparisons)}`",
        "",
        "| concept | alpha | paired init states | none mean | random mean | cluster mean | cluster-random | stable | weaken | collapse | flip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for comparison in comparisons:
        counts = comparison["classification_counts"]
        lines.append(
            f"| {comparison['concept']} | {comparison['alpha']} | {comparison['num_paired_init_states']} | "
            f"{comparison['none_mean'] if comparison['none_mean'] is not None else ''} | "
            f"{comparison['random_mean']:.6f} | {comparison['cluster_mean']:.6f} | "
            f"{comparison['cluster_minus_random_mean']:.6f} | "
            f"{counts.get('stable', 0)} | {counts.get('weaken', 0)} | "
            f"{counts.get('collapse', 0)} | {counts.get('flip', 0)} |"
        )

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved JSON summary to: {output_json}")
    print(f"Saved Markdown summary to: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
