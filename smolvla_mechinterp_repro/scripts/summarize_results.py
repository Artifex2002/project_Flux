#!/usr/bin/env python3
"""
Summarize the main SmolVLA steering results across Phases 5, 6, and 7.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPRO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPRO_ROOT / "results" / "eval_grids"

DEFAULT_PHASE5 = EVAL_ROOT / "phase5_alpha_sweep_late_core_initial" / "summary.json"
DEFAULT_PHASE6_SUMMARY = EVAL_ROOT / "phase6_tier1_fast_risk_initial" / "summary.json"
DEFAULT_PHASE6_TRANSFER = EVAL_ROOT / "phase6_tier1_fast_risk_initial" / "init_state_transfer_avg_displacement.json"
DEFAULT_BRIGHT_SUMMARY = EVAL_ROOT / "phase7_tier2_primary_brightness_initial_v2" / "summary.json"
DEFAULT_BRIGHT_TRANSFER = EVAL_ROOT / "phase7_tier2_primary_brightness_initial_v2" / "visual_perturbation_transfer_avg_displacement.json"
DEFAULT_OCCL_SUMMARY = EVAL_ROOT / "phase7_tier2_primary_occlusion_initial" / "summary.json"
DEFAULT_OCCL_TRANSFER = EVAL_ROOT / "phase7_tier2_primary_occlusion_initial" / "visual_perturbation_transfer_avg_displacement.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_run(summary: dict[str, Any], condition: str, concept: str | None = None, alpha: float | None = None) -> dict[str, Any]:
    for row in summary["runs"]:
        if row["condition"] != condition:
            continue
        if row.get("concept") != concept:
            continue
        if alpha is not None and row.get("alpha") != alpha:
            continue
        return row
    raise KeyError((condition, concept, alpha))


def get_transfer(transfer: dict[str, Any], concept: str, alpha: float) -> dict[str, Any]:
    for row in transfer["comparisons"]:
        if row["concept"] == concept and row["alpha"] == alpha:
            return row
    raise KeyError((concept, alpha))


def build_phase5_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    concepts = sorted({row["concept"] for row in summary["runs"] if row["concept"] in {"fast", "risk"}})
    for concept in concepts:
        alphas = sorted(row["alpha"] for row in summary["runs"] if row["concept"] == concept and row["condition"] == "cluster")
        for alpha in alphas:
            random_row = get_run(summary, "random_matched", concept, alpha)
            cluster_row = get_run(summary, "cluster", concept, alpha)
            rows.append(
                {
                    "phase": "phase5_alpha_sweep",
                    "concept": concept,
                    "setting": f"alpha={alpha}",
                    "baseline_or_random_mean": random_row["avg_speed_mean"],
                    "cluster_mean": cluster_row["avg_speed_mean"],
                    "cluster_minus_random": cluster_row["avg_speed_mean"] - random_row["avg_speed_mean"],
                    "classification": "",
                    "details": "",
                }
            )
    return rows


def build_phase6_rows(summary: dict[str, Any], transfer: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    none_mean = get_run(summary, "none")["avg_speed_mean"]
    for concept in ("fast", "risk"):
        random_row = get_run(summary, "random_matched", concept, 10.0)
        cluster_row = get_run(summary, "cluster", concept, 10.0)
        transfer_row = get_transfer(transfer, concept, 10.0)
        rows.append(
            {
                "phase": "phase6_init_state_transfer",
                "concept": concept,
                "setting": "clean",
                "baseline_or_random_mean": random_row["avg_speed_mean"],
                "cluster_mean": cluster_row["avg_speed_mean"],
                "cluster_minus_random": transfer_row["cluster_minus_random_mean"],
                "classification": "; ".join(f"{k}={v}" for k, v in sorted(transfer_row["classification_counts"].items())),
                "details": f"none_mean={none_mean:.6f}",
            }
        )
    return rows


def build_phase7_rows(summary: dict[str, Any], transfer: dict[str, Any], setting: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in ("fast", "risk"):
        random_row = get_run(summary, "random_matched", concept, 10.0)
        cluster_row = get_run(summary, "cluster", concept, 10.0)
        transfer_row = get_transfer(transfer, concept, 10.0)
        rows.append(
            {
                "phase": f"phase7_{setting}",
                "concept": concept,
                "setting": setting,
                "baseline_or_random_mean": random_row["avg_speed_mean"],
                "cluster_mean": cluster_row["avg_speed_mean"],
                "cluster_minus_random": transfer_row["perturbed_effect_mean"],
                "classification": transfer_row["overall_classification"],
                "details": "; ".join(f"{k}={v}" for k, v in sorted(transfer_row["classification_counts"].items())),
            }
        )
    return rows


def build_headlines(phase5: dict[str, Any], phase6: dict[str, Any], bright: dict[str, Any], occl: dict[str, Any]) -> list[str]:
    risk_alpha10 = get_run(phase5, "cluster", "risk", 10.0)["avg_speed_mean"]
    risk_random10 = get_run(phase5, "random_matched", "risk", 10.0)["avg_speed_mean"]
    phase6_risk = get_transfer(phase6, "risk", 10.0)
    bright_risk = get_transfer(bright, "risk", 10.0)
    occl_risk = get_transfer(occl, "risk", 10.0)
    phase6_fast = get_transfer(phase6, "fast", 10.0)
    occl_fast = get_transfer(occl, "fast", 10.0)
    return [
        (
            "At alpha=10, the risk cluster reduced mean displacement to "
            f"{risk_alpha10:.6f} versus {risk_random10:.6f} for its matched random control."
        ),
        (
            "Risk transferred robustly across fixed init states with classification counts "
            f"{phase6_risk['classification_counts']}."
        ),
        (
            "Risk also remained stable under brightness and occlusion, with perturbed cluster-minus-random effects "
            f"{bright_risk['perturbed_effect_mean']:.6f} and {occl_risk['perturbed_effect_mean']:.6f}."
        ),
        (
            "Fast transferred more weakly: init-state counts "
            f"{phase6_fast['classification_counts']} and occlusion effect "
            f"{occl_fast['perturbed_effect_mean']:.6f}."
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "concept",
        "setting",
        "baseline_or_random_mean",
        "cluster_mean",
        "cluster_minus_random",
        "classification",
        "details",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], headlines: list[str]) -> None:
    lines = [
        "# SmolVLA Steering Summary",
        "",
        "## Headline Findings",
        "",
    ]
    for line in headlines:
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            "## Compact Results Table",
            "",
            "| phase | concept | setting | random/baseline | cluster | cluster-random | classification | details |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['concept']} | {row['setting']} | "
            f"{row['baseline_or_random_mean']:.6f} | {row['cluster_mean']:.6f} | "
            f"{row['cluster_minus_random']:.6f} | {row['classification']} | {row['details']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the main results across Phases 5, 6, and 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=REPRO_ROOT / "results" / "final_summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase5_summary = load_json(DEFAULT_PHASE5)
    phase6_summary = load_json(DEFAULT_PHASE6_SUMMARY)
    phase6_transfer = load_json(DEFAULT_PHASE6_TRANSFER)
    bright_summary = load_json(DEFAULT_BRIGHT_SUMMARY)
    bright_transfer = load_json(DEFAULT_BRIGHT_TRANSFER)
    occl_summary = load_json(DEFAULT_OCCL_SUMMARY)
    occl_transfer = load_json(DEFAULT_OCCL_TRANSFER)

    rows = []
    rows.extend(build_phase5_rows(phase5_summary))
    rows.extend(build_phase6_rows(phase6_summary, phase6_transfer))
    rows.extend(build_phase7_rows(bright_summary, bright_transfer, "brightness"))
    rows.extend(build_phase7_rows(occl_summary, occl_transfer, "occlusion"))

    headlines = build_headlines(phase5_summary, phase6_transfer, bright_transfer, occl_transfer)

    payload = {
        "headline_findings": headlines,
        "rows": rows,
        "source_files": {
            "phase5_summary": str(DEFAULT_PHASE5),
            "phase6_summary": str(DEFAULT_PHASE6_SUMMARY),
            "phase6_transfer": str(DEFAULT_PHASE6_TRANSFER),
            "brightness_summary": str(DEFAULT_BRIGHT_SUMMARY),
            "brightness_transfer": str(DEFAULT_BRIGHT_TRANSFER),
            "occlusion_summary": str(DEFAULT_OCCL_SUMMARY),
            "occlusion_transfer": str(DEFAULT_OCCL_TRANSFER),
        },
    }

    json_path = args.output_dir / "smolvla_steering_summary.json"
    csv_path = args.output_dir / "smolvla_steering_summary.csv"
    md_path = args.output_dir / "smolvla_steering_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, headlines)

    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
