#!/usr/bin/env python3
"""
Analyze Phase 7 Tier 2 nuisance-perturbation transfer experiments.

This compares a clean anchor eval-grid run against a perturbed eval-grid run and
measures how the cluster-vs-random effect changes across matched init-state
contexts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_init_state_transfer import (
    build_rollout_map,
    classify_effect,
    load_json,
    mean,
    parse_csv_float,
    parse_csv_str,
    std,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean vs perturbed cluster effects across matched init states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--anchor-run-dir",
        type=Path,
        required=True,
        help="Eval-grid run directory for the clean anchor condition.",
    )
    parser.add_argument(
        "--perturbed-run-dir",
        type=Path,
        required=True,
        help="Eval-grid run directory for the perturbed condition.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_displacement",
        choices=["avg_displacement", "max_height"],
        help="Per-rollout metric used for transfer comparisons.",
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
        help="Optional output JSON path. Defaults inside the perturbed run dir.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output Markdown path. Defaults inside the perturbed run dir.",
    )
    return parser.parse_args()


def load_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary.json under {run_dir}")
    return load_json(summary_path).get("runs", [])


def pick_row(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    concept: str | None = None,
    alpha: float | None = None,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["condition"] == condition
            and row.get("concept") == concept
            and row.get("alpha") == alpha
        ),
        None,
    )


def summarize_vision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    perturbations = sorted({row.get("vision_perturbation") or "none" for row in rows})
    targets = sorted({row.get("vision_target") or "both" for row in rows})
    strengths = sorted({row.get("vision_strength") for row in rows if row.get("vision_strength") is not None})
    return {
        "perturbations": perturbations,
        "targets": targets,
        "strengths": strengths,
    }


def build_effect_maps(
    rows: list[dict[str, Any]],
    *,
    concept: str,
    alpha: float,
    metric: str,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]] | None:
    none_row = pick_row(rows, condition="none")
    random_row = pick_row(rows, condition="random_matched", concept=concept, alpha=alpha)
    cluster_row = pick_row(rows, condition="cluster", concept=concept, alpha=alpha)
    if none_row is None or random_row is None or cluster_row is None:
        return None

    none_map = build_rollout_map(load_json(Path(none_row["output_json"])), metric)
    random_map = build_rollout_map(load_json(Path(random_row["output_json"])), metric)
    cluster_map = build_rollout_map(load_json(Path(cluster_row["output_json"])), metric)
    return none_map, random_map, cluster_map


def main() -> int:
    args = parse_args()
    anchor_run_dir = args.anchor_run_dir.resolve()
    perturbed_run_dir = args.perturbed_run_dir.resolve()

    anchor_rows = load_summary_rows(anchor_run_dir)
    perturbed_rows = load_summary_rows(perturbed_run_dir)
    concept_filter = set(parse_csv_str(args.concepts) or [])
    alpha_filter = set(parse_csv_float(args.alphas) or [])

    concepts = sorted(
        {row["concept"] for row in anchor_rows + perturbed_rows if row.get("concept") is not None}
    )
    comparisons = []

    for concept in concepts:
        if concept_filter and concept not in concept_filter:
            continue
        alphas = sorted(
            {
                row["alpha"]
                for row in anchor_rows + perturbed_rows
                if row.get("concept") == concept and row.get("alpha") is not None
            }
        )
        for alpha in alphas:
            if alpha_filter and alpha not in alpha_filter:
                continue

            anchor_maps = build_effect_maps(anchor_rows, concept=concept, alpha=alpha, metric=args.metric)
            perturbed_maps = build_effect_maps(perturbed_rows, concept=concept, alpha=alpha, metric=args.metric)
            if anchor_maps is None or perturbed_maps is None:
                continue

            anchor_none, anchor_random, anchor_cluster = anchor_maps
            pert_none, pert_random, pert_cluster = perturbed_maps
            paired_init_states = sorted(
                set(anchor_random)
                & set(anchor_cluster)
                & set(pert_random)
                & set(pert_cluster)
            )
            if not paired_init_states:
                continue

            anchor_effects = []
            perturbed_effects = []
            per_state = []

            for init_state_idx in paired_init_states:
                anchor_effect = anchor_cluster[init_state_idx] - anchor_random[init_state_idx]
                pert_effect = pert_cluster[init_state_idx] - pert_random[init_state_idx]
                anchor_effects.append(anchor_effect)
                perturbed_effects.append(pert_effect)
                per_state.append(
                    {
                        "init_state_idx": init_state_idx,
                        "anchor_none_value": anchor_none.get(init_state_idx),
                        "anchor_random_value": anchor_random[init_state_idx],
                        "anchor_cluster_value": anchor_cluster[init_state_idx],
                        "anchor_cluster_minus_random": anchor_effect,
                        "perturbed_none_value": pert_none.get(init_state_idx),
                        "perturbed_random_value": pert_random[init_state_idx],
                        "perturbed_cluster_value": pert_cluster[init_state_idx],
                        "perturbed_cluster_minus_random": pert_effect,
                        "effect_shift": pert_effect - anchor_effect,
                    }
                )

            anchor_mean = mean(anchor_effects)
            perturbed_mean = mean(perturbed_effects)
            for state in per_state:
                state["classification"] = classify_effect(
                    state["perturbed_cluster_minus_random"],
                    state["anchor_cluster_minus_random"],
                )

            comparisons.append(
                {
                    "concept": concept,
                    "alpha": alpha,
                    "metric": args.metric,
                    "num_paired_init_states": len(paired_init_states),
                    "anchor_effect_mean": anchor_mean,
                    "anchor_effect_std": std(anchor_effects),
                    "perturbed_effect_mean": perturbed_mean,
                    "perturbed_effect_std": std(perturbed_effects),
                    "effect_shift_mean": mean(
                        [state["effect_shift"] for state in per_state]
                    ),
                    "overall_classification": classify_effect(perturbed_mean, anchor_mean),
                    "classification_counts": dict(
                        Counter(state["classification"] for state in per_state)
                    ),
                    "paired_init_states": per_state,
                }
            )

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else perturbed_run_dir / f"visual_perturbation_transfer_{args.metric}.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else perturbed_run_dir / f"visual_perturbation_transfer_{args.metric}.md"
    )

    payload = {
        "anchor_run_dir": str(anchor_run_dir),
        "perturbed_run_dir": str(perturbed_run_dir),
        "metric": args.metric,
        "anchor_vision": summarize_vision(anchor_rows),
        "perturbed_vision": summarize_vision(perturbed_rows),
        "comparisons": comparisons,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Visual Perturbation Transfer Summary",
        "",
        f"- Metric: `{args.metric}`",
        f"- Anchor run: `{anchor_run_dir}`",
        f"- Perturbed run: `{perturbed_run_dir}`",
        f"- Anchor vision: `{payload['anchor_vision']}`",
        f"- Perturbed vision: `{payload['perturbed_vision']}`",
        "",
        "| concept | alpha | paired init states | anchor effect | perturbed effect | shift | overall | stable | weaken | collapse | flip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for comparison in comparisons:
        counts = comparison["classification_counts"]
        lines.append(
            f"| {comparison['concept']} | {comparison['alpha']} | {comparison['num_paired_init_states']} | "
            f"{comparison['anchor_effect_mean']:.6f} | {comparison['perturbed_effect_mean']:.6f} | "
            f"{comparison['effect_shift_mean']:.6f} | {comparison['overall_classification']} | "
            f"{counts.get('stable', 0)} | {counts.get('weaken', 0)} | "
            f"{counts.get('collapse', 0)} | {counts.get('flip', 0)} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved JSON to: {output_json}")
    print(f"Saved Markdown to: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
