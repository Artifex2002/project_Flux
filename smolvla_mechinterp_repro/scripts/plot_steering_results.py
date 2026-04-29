#!/usr/bin/env python3
"""
Generate canonical Phase 8 report figures for SmolVLA steering results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPRO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPRO_ROOT / "results" / "eval_grids"
FINAL_SUMMARY_ROOT = REPRO_ROOT / "results" / "final_summary"

PHASE5_SUMMARY = EVAL_ROOT / "phase5_alpha_sweep_late_core_initial" / "summary.json"
PHASE6_TRANSFER = EVAL_ROOT / "phase6_tier1_fast_risk_initial" / "init_state_transfer_avg_displacement.json"
BRIGHT_TRANSFER = EVAL_ROOT / "phase7_tier2_primary_brightness_initial_v2" / "visual_perturbation_transfer_avg_displacement.json"
OCCL_TRANSFER = EVAL_ROOT / "phase7_tier2_primary_occlusion_initial" / "visual_perturbation_transfer_avg_displacement.json"
FINAL_SUMMARY_JSON = FINAL_SUMMARY_ROOT / "smolvla_steering_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_alpha_sweep(output_dir: Path) -> str:
    data = load_json(PHASE5_SUMMARY)
    rows = data["runs"]
    concepts = ["fast", "risk"]
    colors = {"random_matched": "#d95f02", "cluster": "#1b9e77"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, concept in zip(axes, concepts):
        for condition in ("random_matched", "cluster"):
            selected = [
                row for row in rows
                if row["condition"] == condition and row["concept"] == concept
            ]
            selected.sort(key=lambda row: row["alpha"])
            ax.plot(
                [row["alpha"] for row in selected],
                [row["avg_speed_mean"] for row in selected],
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=colors[condition],
                label="Matched random" if condition == "random_matched" else "Concept cluster",
            )
        ax.set_title(concept.capitalize())
        ax.set_xlabel("Activation strength (alpha)")
        ax.set_xticks([2.5, 5.0, 10.0])
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Mean end-effector displacement")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Phase 5 Alpha Sweep", fontsize=13, y=1.03)
    filename = "report_figure1_alpha_sweep.png"
    save_fig(output_dir / filename)
    return filename


def plot_init_state_transfer(output_dir: Path) -> str:
    data = load_json(PHASE6_TRANSFER)
    comps = {comp["concept"]: comp for comp in data["comparisons"]}
    concepts = ["fast", "risk"]
    colors = {"fast": "#7570b3", "risk": "#e7298a"}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    for ax, concept in zip(axes, concepts):
        comp = comps[concept]
        states = [row["init_state_idx"] for row in comp["paired_init_states"]]
        values = [row["cluster_minus_random"] for row in comp["paired_init_states"]]
        labels = [row["classification"] for row in comp["paired_init_states"]]
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        bars = ax.bar(states, values, color=colors[concept], alpha=0.9)
        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8,
                rotation=90,
            )
        ax.set_title(
            f"{concept.capitalize()} | mean={comp['cluster_minus_random_mean']:.4f}\n"
            f"{comp['classification_counts']}"
        )
        ax.set_xlabel("Init state")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Cluster - matched random")
    fig.suptitle("Phase 6 Init-State Transfer", fontsize=13, y=1.05)
    filename = "report_figure2_init_state_transfer.png"
    save_fig(output_dir / filename)
    return filename


def plot_transfer_stability(output_dir: Path) -> str:
    bright = load_json(BRIGHT_TRANSFER)
    occl = load_json(OCCL_TRANSFER)
    clean = load_json(PHASE6_TRANSFER)

    effect_map = {"clean": {}, "brightness": {}, "occlusion": {}}
    for comp in clean["comparisons"]:
        effect_map["clean"][comp["concept"]] = comp["cluster_minus_random_mean"]
    for comp in bright["comparisons"]:
        effect_map["brightness"][comp["concept"]] = comp["perturbed_effect_mean"]
    for comp in occl["comparisons"]:
        effect_map["occlusion"][comp["concept"]] = comp["perturbed_effect_mean"]

    labels = ["Clean", "Brightness", "Occlusion"]
    x = [0, 1, 2]
    width = 0.34
    colors = {"fast": "#1f78b4", "risk": "#33a02c"}

    fig, ax = plt.subplots(figsize=(8.7, 4.6))
    for idx, concept in enumerate(("fast", "risk")):
        offsets = [item + (-width / 2 if idx == 0 else width / 2) for item in x]
        vals = [effect_map["clean"][concept], effect_map["brightness"][concept], effect_map["occlusion"][concept]]
        ax.bar(offsets, vals, width=width, color=colors[concept], label=concept.capitalize())

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cluster - matched random")
    ax.set_title("Transfer Stability Across Visual Settings")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    filename = "report_figure3_transfer_stability.png"
    save_fig(output_dir / filename)
    return filename


def plot_effect_shift(output_dir: Path) -> str:
    bright = load_json(BRIGHT_TRANSFER)
    occl = load_json(OCCL_TRANSFER)
    shift_map = {
        "brightness": {comp["concept"]: comp["effect_shift_mean"] for comp in bright["comparisons"]},
        "occlusion": {comp["concept"]: comp["effect_shift_mean"] for comp in occl["comparisons"]},
    }
    x = [0, 1]
    labels = ["Brightness", "Occlusion"]
    width = 0.34
    colors = {"fast": "#6a3d9a", "risk": "#b15928"}

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for idx, concept in enumerate(("fast", "risk")):
        offsets = [item + (-width / 2 if idx == 0 else width / 2) for item in x]
        vals = [shift_map["brightness"][concept], shift_map["occlusion"][concept]]
        ax.bar(offsets, vals, width=width, color=colors[concept], label=concept.capitalize())

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Perturbed effect - clean effect")
    ax.set_title("Effect Shift Relative to Clean Anchor")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    filename = "report_figure4_effect_shift.png"
    save_fig(output_dir / filename)
    return filename


def write_manifest(path: Path, files: list[str]) -> None:
    lines = [
        "# Plot Manifest",
        "",
        "Generated canonical Phase 8 report figures:",
        "",
    ]
    descriptions = {
        "report_figure1_alpha_sweep.png": "Phase 5 alpha sweep for fast and risk.",
        "report_figure2_init_state_transfer.png": "Phase 6 per-init-state transfer effects.",
        "report_figure3_transfer_stability.png": "Clean vs brightness vs occlusion transfer effects.",
        "report_figure4_effect_shift.png": "Shift in cluster-minus-random effect relative to clean.",
    }
    for file in files:
        lines.append(f"- `{file}`: {descriptions[file]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical report figures for SmolVLA steering results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=FINAL_SUMMARY_ROOT / "plots")
    parser.add_argument("--summary-json", type=Path, default=FINAL_SUMMARY_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_json.exists():
        raise FileNotFoundError(
            f"Expected summary JSON at {args.summary_json}. Run summarize_results.py first."
        )

    files = [
        plot_alpha_sweep(args.output_dir),
        plot_init_state_transfer(args.output_dir),
        plot_transfer_stability(args.output_dir),
        plot_effect_shift(args.output_dir),
    ]
    write_manifest(args.output_dir / "PLOT_MANIFEST.md", files)

    print(f"Saved plots to: {args.output_dir}")
    for file in files:
        print(f"- {args.output_dir / file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
