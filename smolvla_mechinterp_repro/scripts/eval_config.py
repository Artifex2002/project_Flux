#!/usr/bin/env python3
"""
Evaluation presets for SmolVLA mechanistic steering experiments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "phase5_smoke_fast": {
        "description": "Minimal end-to-end smoke grid for fast steering on task 0.",
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast"],
        "prompt_concepts": ["fast"],
        "conditions": ["none", "prompt", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 1,
        "max_steps": 1,
        "log_every": 1,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase5_pilot_late_core": {
        "description": "Small logged pilot grid covering stable late-layer concepts.",
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "safe", "risk"],
        "prompt_concepts": ["fast"],
        "conditions": ["none", "prompt", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 1,
        "max_steps": 1,
        "log_every": 1,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase5_alpha_sweep_late_core": {
        "description": (
            "More informative Phase 5 sweep on task 0 with late-layer core concepts, "
            "multiple rollouts, longer horizons, and an alpha sweep."
        ),
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "safe", "risk"],
        "prompt_concepts": ["fast", "safe", "risk"],
        "conditions": ["none", "prompt", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [2.5, 5.0, 10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 3,
        "max_steps": 5,
        "log_every": 1,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase6_tier1_init_state_smoke": {
        "description": (
            "Phase 6 Tier 1 smoke test: paired init-state transfer on task 0 for "
            "fast/safe/risk using recommended late candidates."
        ),
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "safe", "risk"],
        "prompt_concepts": [],
        "conditions": ["none", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 4,
        "init_state_indices": [0, 1, 2, 3],
        "max_steps": 10,
        "log_every": 2,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase6_tier1_init_state_core": {
        "description": (
            "Phase 6 Tier 1 core experiment: paired init-state transfer on task 0 "
            "for fast/safe/risk with a modest alpha sweep."
        ),
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "safe", "risk"],
        "prompt_concepts": [],
        "conditions": ["none", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [5.0, 10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 6,
        "init_state_indices": [0, 1, 2, 3, 4, 5],
        "max_steps": 10,
        "log_every": 2,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase7_tier2_primary_brightness_smoke": {
        "description": (
            "Phase 7 Tier 2 smoke test: paired init-state transfer under a primary-camera "
            "brightness shift for fast and risk."
        ),
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "risk"],
        "prompt_concepts": [],
        "conditions": ["none", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 4,
        "init_state_indices": [0, 1, 2, 3],
        "vision_perturbation": "brightness",
        "vision_target": "primary",
        "vision_strength": 0.15,
        "max_steps": 10,
        "log_every": 2,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase7_tier2_clean_smoke": {
        "description": (
            "Phase 7 Tier 2 clean anchor run matched to the nuisance-perturbation smoke "
            "presets for fast and risk."
        ),
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "risk"],
        "prompt_concepts": [],
        "conditions": ["none", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 4,
        "init_state_indices": [0, 1, 2, 3],
        "vision_perturbation": "none",
        "vision_target": "both",
        "vision_strength": None,
        "max_steps": 10,
        "log_every": 2,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
    "phase7_tier2_primary_occlusion_smoke": {
        "description": (
            "Phase 7 Tier 2 smoke test: paired init-state transfer under a primary-camera "
            "occlusion patch for fast and risk."
        ),
        "suite": "libero_10",
        "task_indices": [0],
        "concepts": ["fast", "risk"],
        "prompt_concepts": [],
        "conditions": ["none", "random_matched", "cluster"],
        "cluster_partition": "recommended",
        "alphas": [10.0],
        "max_neurons": [None],
        "layer_scopes": ["candidate"],
        "num_rollouts": 4,
        "init_state_indices": [0, 1, 2, 3],
        "vision_perturbation": "occlusion",
        "vision_target": "primary",
        "vision_strength": 0.2,
        "max_steps": 10,
        "log_every": 2,
        "seed": 42,
        "device": "cpu",
        "random_match_mode": "per_layer",
    },
}


def get_preset_config(name: str) -> dict[str, Any]:
    if name not in PRESET_CONFIGS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESET_CONFIGS)}")
    return deepcopy(PRESET_CONFIGS[name])
