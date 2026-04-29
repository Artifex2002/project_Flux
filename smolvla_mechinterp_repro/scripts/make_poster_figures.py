#!/usr/bin/env python3
"""
Generate poster figures for the SmolVLA mechanistic steering project.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPRO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPRO_ROOT / "results" / "eval_grids"
POSTER_ROOT = REPRO_ROOT / "poster_materials"
FIGURES_ROOT = POSTER_ROOT / "figures"

PHASE5_SUMMARY = RESULTS_ROOT / "phase5_alpha_sweep_late_core_initial" / "summary.json"
PHASE6_TRANSFER = RESULTS_ROOT / "phase6_tier1_fast_risk_initial" / "init_state_transfer_avg_displacement.json"
BRIGHT_TRANSFER = RESULTS_ROOT / "phase7_tier2_primary_brightness_initial_v2" / "visual_perturbation_transfer_avg_displacement.json"
OCCL_TRANSFER = RESULTS_ROOT / "phase7_tier2_primary_occlusion_initial" / "visual_perturbation_transfer_avg_displacement.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_phase5_alpha_sweep() -> None:
    data = load_json(PHASE5_SUMMARY)
    rows = data["runs"]
    concepts = ["fast", "risk"]
    colors = {"cluster": "#1b9e77", "random_matched": "#d95f02"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, concept in zip(axes, concepts):
        for condition in ["random_matched", "cluster"]:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["concept"] == concept
            ]
            selected.sort(key=lambda row: row["alpha"])
            alphas = [row["alpha"] for row in selected]
            values = [row["avg_speed_mean"] for row in selected]
            label = "Matched random" if condition == "random_matched" else "Cluster"
            ax.plot(alphas, values, marker="o", linewidth=2.4, color=colors[condition], label=label)

        ax.set_title(f"{concept.capitalize()} concept")
        ax.set_xlabel("Alpha")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xticks([2.5, 5.0, 10.0])
    axes[0].set_ylabel("Mean end-effector displacement")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Phase 5: Steering effect vs activation strength", fontsize=13, y=1.03)
    save_figure(FIGURES_ROOT / "figure1_phase5_alpha_sweep.png")


def plot_phase6_init_state_transfer() -> None:
    data = load_json(PHASE6_TRANSFER)
    comps = {comp["concept"]: comp for comp in data["comparisons"]}
    concepts = ["fast", "risk"]
    colors = {"fast": "#7570b3", "risk": "#e7298a"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, concept in zip(axes, concepts):
        comp = comps[concept]
        states = [row["init_state_idx"] for row in comp["paired_init_states"]]
        effects = [row["cluster_minus_random"] for row in comp["paired_init_states"]]
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.bar(states, effects, color=colors[concept], alpha=0.9)
        ax.set_title(
            f"{concept.capitalize()} | mean effect = {comp['cluster_minus_random_mean']:.4f}\n"
            f"{comp['classification_counts']}"
        )
        ax.set_xlabel("Init state index")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Cluster - matched random\n(mean displacement)")
    fig.suptitle("Phase 6: Transfer across fixed visual contexts", fontsize=13, y=1.05)
    save_figure(FIGURES_ROOT / "figure2_phase6_init_state_transfer.png")


def plot_phase7_perturbation_transfer() -> None:
    bright = load_json(BRIGHT_TRANSFER)
    occl = load_json(OCCL_TRANSFER)
    concepts = ["fast", "risk"]

    effect_map = {
        "clean": {},
        "brightness": {},
        "occlusion": {},
    }
    for comp in load_json(PHASE6_TRANSFER)["comparisons"]:
        effect_map["clean"][comp["concept"]] = comp["cluster_minus_random_mean"]
    for comp in bright["comparisons"]:
        effect_map["brightness"][comp["concept"]] = comp["perturbed_effect_mean"]
    for comp in occl["comparisons"]:
        effect_map["occlusion"][comp["concept"]] = comp["perturbed_effect_mean"]

    labels = ["Clean", "Brightness", "Occlusion"]
    colors = {"fast": "#1f78b4", "risk": "#33a02c"}
    x = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for idx, concept in enumerate(concepts):
        offsets = [item + (-width / 2 if idx == 0 else width / 2) for item in x]
        values = [
            effect_map["clean"][concept],
            effect_map["brightness"][concept],
            effect_map["occlusion"][concept],
        ]
        ax.bar(offsets, values, width=width, label=concept.capitalize(), color=colors[concept])

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cluster - matched random\n(mean displacement)")
    ax.set_title("Phase 7: Steering effect under visual perturbations")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    save_figure(FIGURES_ROOT / "figure3_phase7_perturbation_transfer.png")


def plot_condition_means_across_settings() -> None:
    clean = load_json(RESULTS_ROOT / "phase6_tier1_fast_risk_initial" / "summary.json")
    bright = load_json(RESULTS_ROOT / "phase7_tier2_primary_brightness_initial_v2" / "summary.json")
    occl = load_json(RESULTS_ROOT / "phase7_tier2_primary_occlusion_initial" / "summary.json")

    def lookup(summary: dict, condition: str, concept: str | None = None) -> float:
        for row in summary["runs"]:
            if row["condition"] == condition and row["concept"] == concept:
                return row["avg_speed_mean"]
        raise KeyError((condition, concept))

    settings = ["Clean", "Brightness", "Occlusion"]
    keys = [
        ("none", None, "None"),
        ("random_matched", "fast", "Random fast"),
        ("cluster", "fast", "Cluster fast"),
        ("random_matched", "risk", "Random risk"),
        ("cluster", "risk", "Cluster risk"),
    ]
    datasets = [clean, bright, occl]
    palette = ["#666666", "#d95f02", "#1b9e77", "#e6ab02", "#e7298a"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.14
    x = list(range(len(settings)))
    for idx, (condition, concept, label) in enumerate(keys):
        offsets = [item + (idx - 2) * width for item in x]
        values = [lookup(dataset, condition, concept) for dataset in datasets]
        ax.bar(offsets, values, width=width, label=label, color=palette[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("Mean end-effector displacement")
    ax.set_title("Condition means across clean and perturbed settings")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    save_figure(FIGURES_ROOT / "figure4_condition_means_across_settings.png")


def main() -> int:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    plot_phase5_alpha_sweep()
    plot_phase6_init_state_transfer()
    plot_phase7_perturbation_transfer()
    plot_condition_means_across_settings()
    print(f"Saved poster figures to: {FIGURES_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
