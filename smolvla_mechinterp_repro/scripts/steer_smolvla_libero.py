#!/usr/bin/env python3
"""
Phase 4 steering runner for SmolVLA on LIBERO.

This script replaces the old random-output-hook baseline with a general steering
engine that:

- reads steering candidates from Phase 3.5 selection artifacts,
- injects selected FFN activations at the correct pre-down-proj hook point,
- supports concept clusters, matched random controls, prompt-only controls,
- supports layer-restricted steering for single-layer / early / late / full use,
- saves rollout metrics plus hook-debug statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as exc:
    print(f"ERROR: Could not import SmolVLAPolicy from lerobot: {exc}")
    print("Make sure lerobot is installed in the active environment.")
    raise SystemExit(1)

try:
    from lerobot.policies.factory import make_pre_post_processors
except ImportError:
    make_pre_post_processors = None


REPRO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_BUNDLE = REPRO_ROOT / "results" / "selected_cluster_candidates.pt"
DEFAULT_OUTPUT_DIR = REPRO_ROOT / "results" / "steering_runs"
_TO_TENSOR = T.Compose([T.Resize((256, 256)), T.ToTensor()])


@dataclass
class SteeringSpec:
    label: str
    source_mode: str
    concept_name: str | None
    candidate_id: str | None
    template_candidate_id: str | None
    alpha: float
    layer_scope: str
    partition_name: str | None
    cluster_id: int | None
    num_neurons: int
    num_layers_touched: int
    top_layers: list[dict[str, Any]]
    layer_to_vectors: dict[int, list[int]]
    extra: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steer SmolVLA on LIBERO using Phase 3.5 candidate clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceVLA/smolvla_libero",
        help="HuggingFace checkpoint to load.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_10",
        help="LIBERO benchmark suite. 'libero_long' aliases to 'libero_10'.",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=0,
        help="LIBERO task index.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts to execute.",
    )
    parser.add_argument(
        "--init-state-indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated init-state indices to use for each rollout. "
            "When provided, length must match --num-rollouts."
        ),
    )
    parser.add_argument(
        "--vision-perturbation",
        type=str,
        default="none",
        choices=["none", "brightness", "contrast", "gaussian_noise", "gaussian_blur", "occlusion"],
        help="Optional visual perturbation applied to camera observations.",
    )
    parser.add_argument(
        "--vision-target",
        type=str,
        default="both",
        choices=["primary", "wrist", "both"],
        help="Which camera stream should receive the visual perturbation.",
    )
    parser.add_argument(
        "--vision-strength",
        type=float,
        default=None,
        help="Perturbation strength. Interpretation depends on the perturbation type.",
    )
    parser.add_argument(
        "--vision-seed",
        type=int,
        default=None,
        help="Optional seed for deterministic visual perturbations. Defaults to the rollout seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per rollout.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Progress print frequency inside rollouts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for the policy.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="cluster",
        choices=["none", "prompt", "cluster", "random_matched"],
        help="Steering / baseline condition to run.",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default=None,
        help="Prefix text used when --condition prompt.",
    )
    parser.add_argument(
        "--candidate-bundle-pt",
        type=Path,
        default=DEFAULT_CANDIDATE_BUNDLE,
        help="Selected cluster candidate bundle from Phase 3.5.",
    )
    parser.add_argument(
        "--candidate-id",
        type=str,
        default=None,
        help="Explicit candidate id, e.g. late__fast__cluster_6.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Concept name used to select a recommended / ranked candidate.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="recommended",
        help="Partition to use when selecting by concept: recommended/full/early/late.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Overwrite value used for selected FFN activations.",
    )
    parser.add_argument(
        "--layer-scope",
        type=str,
        default="candidate",
        choices=["candidate", "single_layer", "early_only", "late_only"],
        help="How to filter the chosen candidate before hooking.",
    )
    parser.add_argument(
        "--single-layer-idx",
        type=int,
        default=None,
        help="Used with --layer-scope single_layer. Defaults to the candidate's dominant layer.",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="Optional cap on the number of steered neurons after filtering.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=None,
        help="Seed used when subsampling candidate neurons.",
    )
    parser.add_argument(
        "--random-match-mode",
        type=str,
        default="per_layer",
        choices=["per_layer", "total_only"],
        help="How matched random controls are sampled.",
    )
    parser.add_argument(
        "--debug-steering",
        action="store_true",
        help="Print detailed steering setup and collect hook statistics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve steering and register hooks, but do not run the environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for result JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit output JSON path. Overrides --output-dir naming.",
    )
    return parser.parse_args()


def parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def resolve_vision_strength(perturbation: str, strength: float | None) -> float | None:
    if perturbation == "none":
        return None
    if strength is not None:
        return strength
    defaults = {
        "brightness": 0.15,
        "contrast": 0.6,
        "gaussian_noise": 0.08,
        "gaussian_blur": 5.0,
        "occlusion": 0.2,
    }
    return defaults[perturbation]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args: argparse.Namespace):
    print(f"\nLoading model: {args.model_id}")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.eval()

    device = select_device(args.device)
    policy = policy.to(device)

    preprocess_fn = None
    postprocess_fn = None
    if make_pre_post_processors is not None:
        try:
            preprocess_fn, postprocess_fn = make_pre_post_processors(
                policy.config,
                args.model_id,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            print("Pre/post processors built successfully.")
        except Exception as exc:
            print(f"WARNING: Could not build pre/post processors: {exc}")
            print("  Will attempt raw inference.")

    elapsed = time.time() - t0
    print(f"Model loaded on {device} in {elapsed:.1f}s")
    return policy, device, preprocess_fn, postprocess_fn


def get_text_layers(policy) -> Any:
    return policy.model.vlm_with_expert.vlm.model.text_model.layers


def setup_env(args: argparse.Namespace):
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "Could not import LIBERO. Install it in the active environment to run rollouts."
        ) from exc

    print(f"\nSetting up LIBERO environment...")
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_aliases = {"libero_long": "libero_10"}
    suite_name = suite_aliases.get(getattr(args, "suite", "libero_10"), getattr(args, "suite", "libero_10"))

    if suite_name not in benchmark_dict:
        raise KeyError(f"Suite '{suite_name}' not found. Available: {sorted(benchmark_dict.keys())}")

    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(args.task_idx)
    task_description = task.language
    task_bddl_path = task_suite.get_task_bddl_file_path(args.task_idx)
    print(f"Task {args.task_idx}: {task_description}")
    print(f"  BDDL file: {task_bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_path,
        camera_heights=256,
        camera_widths=256,
        render_camera="agentview",
    )
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        task.problem_folder,
        task.init_states_file,
    )
    try:
        init_states = torch.load(init_states_path, weights_only=False)
    except TypeError:
        init_states = torch.load(init_states_path)
    print(f"  Init states loaded: {init_states.shape}")
    return env, task_suite, task_description, init_states


def _extract_image_tensor(obs, candidate_keys, label):
    for key in candidate_keys:
        if key in obs:
            arr = obs[key]
            if isinstance(arr, np.ndarray):
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
                return _TO_TENSOR(pil_img)
    print(f"  WARNING: No key found for {label}. Tried: {candidate_keys}")
    return None


def build_vision_context(args, *, seed_offset: int) -> dict[str, Any] | None:
    if args.vision_perturbation == "none":
        return None

    strength = resolve_vision_strength(args.vision_perturbation, args.vision_strength)
    seed = args.vision_seed if args.vision_seed is not None else args.seed + 10_000 + seed_offset
    context: dict[str, Any] = {
        "perturbation": args.vision_perturbation,
        "target": args.vision_target,
        "strength": strength,
        "seed": seed,
    }

    if args.vision_perturbation == "occlusion":
        rng = random.Random(seed)
        frac = float(np.clip(strength if strength is not None else 0.2, 0.02, 0.95))
        patch_size = max(1, int(round(256 * frac)))
        boxes = {}
        for camera_name in ("primary", "wrist"):
            max_top = max(0, 256 - patch_size)
            max_left = max(0, 256 - patch_size)
            boxes[camera_name] = {
                "top": rng.randint(0, max_top),
                "left": rng.randint(0, max_left),
                "size": patch_size,
            }
        context["occlusion_boxes"] = boxes

    return context


def apply_vision_perturbation(
    image: torch.Tensor | None,
    *,
    camera_name: str,
    vision_context: dict[str, Any] | None,
) -> torch.Tensor | None:
    if image is None or vision_context is None:
        return image

    target = vision_context["target"]
    if target != "both" and target != camera_name:
        return image

    out = image.clone()
    perturbation = vision_context["perturbation"]
    strength = vision_context["strength"]
    if perturbation == "brightness":
        return torch.clamp(out + float(strength), 0.0, 1.0)

    if perturbation == "contrast":
        factor = float(max(0.0, strength))
        return torch.clamp((out - 0.5) * factor + 0.5, 0.0, 1.0)

    if perturbation == "gaussian_noise":
        generator = torch.Generator(device="cpu")
        offset = 0 if camera_name == "primary" else 1_337
        generator.manual_seed(int(vision_context["seed"]) + offset)
        noise = torch.randn(out.shape, generator=generator, dtype=out.dtype)
        return torch.clamp(out + float(strength) * noise, 0.0, 1.0)

    if perturbation == "gaussian_blur":
        kernel_size = max(1, int(round(float(strength))))
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(0.1, float(strength) / 3.0)
        return TF.gaussian_blur(out, [kernel_size, kernel_size], [sigma, sigma])

    if perturbation == "occlusion":
        box = vision_context["occlusion_boxes"][camera_name]
        top = box["top"]
        left = box["left"]
        size = box["size"]
        out[:, top : top + size, left : left + size] = 0.0
        return out

    return out


def format_obs(obs, task_description, device, preprocess_fn=None, vision_context=None, _printed_keys=[False]):
    if not _printed_keys[0]:
        _printed_keys[0] = True
        print("\n  [DEBUG] Raw observation keys and shapes:")
        for key, value in sorted(obs.items()):
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: type={type(value).__name__}")

    image_primary = _extract_image_tensor(obs, ["agentview_image", "agentview_rgb"], "primary camera")
    image_wrist = _extract_image_tensor(obs, ["robot0_eye_in_hand_image", "wrist_image"], "wrist camera")
    image_primary = apply_vision_perturbation(image_primary, camera_name="primary", vision_context=vision_context)
    image_wrist = apply_vision_perturbation(image_wrist, camera_name="wrist", vision_context=vision_context)

    state_parts = []
    if "robot0_eef_pos" in obs:
        state_parts.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32).flatten())
    if "robot0_eef_quat" in obs:
        state_parts.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).flatten())
    if "robot0_gripper_qpos" in obs:
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).flatten()
        state_parts.append(np.array([gripper.mean()], dtype=np.float32))
    state = np.concatenate(state_parts) if state_parts else np.zeros(8, dtype=np.float32)

    frame = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": task_description}
    if image_primary is not None:
        frame["observation.images.image"] = image_primary
    if image_wrist is not None:
        frame["observation.images.image2"] = image_wrist

    if preprocess_fn is not None:
        try:
            return preprocess_fn(frame)
        except Exception as exc:
            print(f"  WARNING: preprocess_fn failed ({exc}), attempting manual formatting")

    obs_dict = {"observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device), "task": [task_description]}
    if image_primary is not None:
        obs_dict["observation.images.image"] = image_primary.unsqueeze(0).to(device)
    if image_wrist is not None:
        obs_dict["observation.images.image2"] = image_wrist.unsqueeze(0).to(device)
    return obs_dict


def _action_to_numpy(raw_action):
    if isinstance(raw_action, dict):
        if "action" in raw_action:
            raw_action = raw_action["action"]
        else:
            raw_action = next(iter(raw_action.values()))

    if isinstance(raw_action, torch.Tensor):
        action_np = raw_action.detach().cpu().numpy()
    else:
        action_np = np.asarray(raw_action)

    while action_np.ndim > 1:
        action_np = action_np[0]
    return action_np


def _extract_ee_pos(obs):
    for key in ["robot0_eef_pos", "eef_pos", "ee_pos"]:
        if key in obs:
            return np.asarray(obs[key], dtype=np.float64).flatten()[:3]
    return np.zeros(3, dtype=np.float64)


def run_rollout(
    env,
    policy,
    task_description,
    args,
    device,
    *,
    preprocess_fn=None,
    postprocess_fn=None,
    seed_offset=0,
    init_states=None,
    init_state_idx: int | None = None,
    vision_context: dict[str, Any] | None = None,
):
    env.seed(args.seed + seed_offset)
    obs = env.reset()
    resolved_init_idx = None
    if init_states is not None:
        resolved_init_idx = init_state_idx if init_state_idx is not None else seed_offset % len(init_states)
        if resolved_init_idx < 0 or resolved_init_idx >= len(init_states):
            raise IndexError(
                f"Requested init_state_idx={resolved_init_idx}, but only {len(init_states)} init states are available."
            )
        obs = env.set_init_state(init_states[resolved_init_idx])
        print(f"  Using init state {resolved_init_idx}")

    ee_positions = []
    success = False
    total_steps = 0
    step_times = []
    infer_times = []
    rollout_start = time.time()
    up_axis = 2
    up_label = "z"

    for step_idx in range(args.max_steps):
        step_start = time.time()
        formatted_obs = format_obs(
            obs,
            task_description,
            device,
            preprocess_fn=preprocess_fn,
            vision_context=vision_context,
        )

        infer_start = time.time()
        with torch.no_grad():
            raw_action = policy.select_action(formatted_obs)
        infer_times.append(time.time() - infer_start)

        if postprocess_fn is not None:
            try:
                raw_action = postprocess_fn(raw_action)
            except Exception:
                pass

        action_np = _action_to_numpy(raw_action)
        ee_pos = _extract_ee_pos(obs)
        ee_positions.append(ee_pos.tolist())

        obs, reward, done, info = env.step(action_np)
        total_steps = step_idx + 1
        step_times.append(time.time() - step_start)

        if args.log_every and total_steps % args.log_every == 0:
            avg_step = sum(step_times) / len(step_times)
            avg_infer = sum(infer_times) / len(infer_times)
            elapsed = time.time() - rollout_start
            print(
                f"  step {total_steps}/{args.max_steps} | "
                f"avg_step={avg_step:.2f}s avg_infer={avg_infer:.2f}s "
                f"last_z={ee_pos[2]:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if env.check_success():
            success = True
            ee_positions.append(_extract_ee_pos(obs).tolist())
            break
        if done:
            ee_positions.append(_extract_ee_pos(obs).tolist())
            break

    ee_arr = np.array(ee_positions)
    if len(ee_arr) > 1:
        diffs = np.diff(ee_arr, axis=0)
        displacements = np.linalg.norm(diffs, axis=1).tolist()
        avg_displacement = float(np.mean(displacements))
    else:
        displacements = []
        avg_displacement = 0.0

    max_height = float(ee_arr[:, up_axis].max()) if len(ee_arr) > 0 else 0.0
    return {
        "init_state_idx": resolved_init_idx,
        "vision_context": vision_context,
        "ee_positions": ee_positions,
        "displacements": displacements,
        "avg_displacement": avg_displacement,
        "max_height": max_height,
        "max_height_axis": up_label,
        "total_steps": total_steps,
        "success": success,
    }


def resolve_candidate_from_bundle(
    bundle: dict[str, Any],
    *,
    candidate_id: str | None,
    concept: str | None,
    partition: str,
) -> tuple[str, dict[str, Any]]:
    candidates = bundle["candidates"]
    recommended = bundle["recommended_candidates"]

    if candidate_id is not None:
        if candidate_id not in candidates:
            raise KeyError(f"Candidate id '{candidate_id}' not found in bundle.")
        return candidate_id, candidates[candidate_id]

    if concept is None:
        raise ValueError("Provide either --candidate-id or --concept for cluster-based conditions.")

    if partition == "recommended":
        if concept not in recommended:
            raise KeyError(f"No recommended candidate found for concept '{concept}'.")
        resolved_id = recommended[concept]["candidate_id"]
        return resolved_id, candidates[resolved_id]

    matches = [
        (cand_id, cand)
        for cand_id, cand in candidates.items()
        if cand["concept_name"] == concept and cand["partition_name"] == partition
    ]
    if not matches:
        raise KeyError(f"No candidate found for concept='{concept}' in partition='{partition}'.")
    matches.sort(key=lambda item: (item[1]["cosine_similarity"], item[1]["cluster_size"]), reverse=True)
    return matches[0]


def filter_candidate_members(
    candidate_id: str,
    candidate: dict[str, Any],
    *,
    num_layers: int,
    layer_scope: str,
    single_layer_idx: int | None,
) -> dict[str, Any]:
    layer_idx = candidate["layer_idx"].to(dtype=torch.int64)
    vector_index = candidate["vector_index"].to(dtype=torch.int64)
    global_vector_indices = candidate["global_vector_indices"].to(dtype=torch.int64)

    if layer_scope == "candidate":
        mask = torch.ones_like(layer_idx, dtype=torch.bool)
    elif layer_scope == "single_layer":
        target_layer = single_layer_idx
        if target_layer is None:
            target_layer = int(candidate["top_layers"][0]["layer_idx"])
        mask = layer_idx == target_layer
    elif layer_scope == "early_only":
        mask = layer_idx < (num_layers // 2)
    elif layer_scope == "late_only":
        mask = layer_idx >= (num_layers // 2)
    else:
        raise ValueError(f"Unsupported layer scope: {layer_scope}")

    filtered = {
        "candidate_id": candidate_id,
        "partition_name": candidate["partition_name"],
        "concept_name": candidate["concept_name"],
        "cluster_id": int(candidate["cluster_id"]),
        "cosine_similarity": float(candidate["cosine_similarity"]),
        "layer_scope": layer_scope,
        "layer_idx": layer_idx[mask],
        "vector_index": vector_index[mask],
        "global_vector_indices": global_vector_indices[mask],
    }
    if filtered["global_vector_indices"].numel() == 0:
        raise ValueError(
            f"Layer scope '{layer_scope}' removed all members for candidate '{candidate_id}'."
        )
    return filtered


def maybe_subsample_members(
    filtered: dict[str, Any],
    *,
    max_neurons: int | None,
    subsample_seed: int,
) -> dict[str, Any]:
    if max_neurons is None or filtered["global_vector_indices"].numel() <= max_neurons:
        return filtered

    generator = torch.Generator(device="cpu")
    generator.manual_seed(subsample_seed)
    keep = torch.randperm(filtered["global_vector_indices"].numel(), generator=generator)[:max_neurons]
    keep = torch.sort(keep).values
    return {
        **filtered,
        "layer_idx": filtered["layer_idx"][keep],
        "vector_index": filtered["vector_index"][keep],
        "global_vector_indices": filtered["global_vector_indices"][keep],
    }


def sample_random_matched_members(
    template: dict[str, Any],
    embeddings_bundle: dict[str, Any],
    *,
    num_layers: int,
    match_mode: str,
    seed: int,
) -> dict[str, Any]:
    emb_global = embeddings_bundle["global_vector_index"].to(dtype=torch.int64)
    emb_layer = embeddings_bundle["layer_idx"].to(dtype=torch.int64)
    emb_vector = embeddings_bundle["vector_index"].to(dtype=torch.int64)
    template_globals = set(int(item) for item in template["global_vector_indices"].tolist())

    if template["partition_name"] == "early":
        partition_mask = emb_layer < (num_layers // 2)
    elif template["partition_name"] == "late":
        partition_mask = emb_layer >= (num_layers // 2)
    else:
        partition_mask = torch.ones_like(emb_layer, dtype=torch.bool)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if match_mode == "per_layer":
        sampled_indices: list[torch.Tensor] = []
        layer_counts = Counter(int(layer) for layer in template["layer_idx"].tolist())
        for layer, count in sorted(layer_counts.items()):
            layer_mask = partition_mask & (emb_layer == layer)
            pool_rows = torch.nonzero(layer_mask, as_tuple=False).squeeze(-1)
            pool_rows = torch.tensor(
                [row for row in pool_rows.tolist() if int(emb_global[row].item()) not in template_globals],
                dtype=torch.long,
            )
            if pool_rows.numel() < count:
                raise ValueError(
                    f"Not enough candidates to sample {count} random neurons from layer {layer}."
                )
            chosen = pool_rows[torch.randperm(pool_rows.numel(), generator=generator)[:count]]
            sampled_indices.append(chosen)
        rows = torch.cat(sampled_indices, dim=0)
    else:
        pool_rows = torch.nonzero(partition_mask, as_tuple=False).squeeze(-1)
        pool_rows = torch.tensor(
            [row for row in pool_rows.tolist() if int(emb_global[row].item()) not in template_globals],
            dtype=torch.long,
        )
        count = int(template["global_vector_indices"].numel())
        if pool_rows.numel() < count:
            raise ValueError(f"Not enough candidates to sample {count} matched random neurons.")
        rows = pool_rows[torch.randperm(pool_rows.numel(), generator=generator)[:count]]

    rows = torch.sort(rows).values
    sampled_layers = emb_layer[rows]
    sampled_vectors = emb_vector[rows]
    sampled_globals = emb_global[rows]

    return {
        "candidate_id": f"random_matched__{template['candidate_id']}",
        "partition_name": template["partition_name"],
        "concept_name": template["concept_name"],
        "cluster_id": None,
        "cosine_similarity": None,
        "layer_scope": template["layer_scope"],
        "layer_idx": sampled_layers,
        "vector_index": sampled_vectors,
        "global_vector_indices": sampled_globals,
        "template_candidate_id": template["candidate_id"],
    }


def summarize_member_layers(layer_idx: torch.Tensor) -> list[dict[str, int]]:
    counts = Counter(int(layer) for layer in layer_idx.tolist())
    return [
        {"layer_idx": layer, "count": count}
        for layer, count in counts.most_common(10)
    ]


def build_layer_to_vectors(layer_idx: torch.Tensor, vector_index: torch.Tensor) -> dict[int, list[int]]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for layer, vector in zip(layer_idx.tolist(), vector_index.tolist(), strict=True):
        grouped[int(layer)].add(int(vector))
    return {layer: sorted(vectors) for layer, vectors in sorted(grouped.items())}


def build_steering_spec(
    resolved: dict[str, Any],
    *,
    condition: str,
    alpha: float,
    concept_name: str | None,
) -> SteeringSpec:
    layer_to_vectors = build_layer_to_vectors(resolved["layer_idx"], resolved["vector_index"])
    return SteeringSpec(
        label=resolved["candidate_id"],
        source_mode=condition,
        concept_name=concept_name,
        candidate_id=resolved["candidate_id"],
        template_candidate_id=resolved.get("template_candidate_id"),
        alpha=alpha,
        layer_scope=resolved["layer_scope"],
        partition_name=resolved["partition_name"],
        cluster_id=resolved.get("cluster_id"),
        num_neurons=int(resolved["global_vector_indices"].numel()),
        num_layers_touched=len(layer_to_vectors),
        top_layers=summarize_member_layers(resolved["layer_idx"]),
        layer_to_vectors=layer_to_vectors,
        extra={},
    )


def register_pre_down_proj_hooks(
    policy,
    layer_to_vectors: dict[int, list[int]],
    *,
    alpha: float,
    debug: bool,
) -> tuple[list[Any], dict[str, Any]]:
    text_layers = get_text_layers(policy)
    stats = {
        "alpha": alpha,
        "layers": {},
    }
    handles = []

    for layer_idx, neuron_indices in layer_to_vectors.items():
        if layer_idx < 0 or layer_idx >= len(text_layers):
            raise IndexError(f"Requested layer {layer_idx}, but model only has {len(text_layers)} layers.")
        target = text_layers[layer_idx].mlp.down_proj
        if max(neuron_indices) >= int(target.in_features):
            raise IndexError(
                f"Layer {layer_idx} neuron index out of range for down_proj.in_features={target.in_features}."
            )

        stats["layers"][str(layer_idx)] = {
            "num_neurons": len(neuron_indices),
            "call_count": 0,
            "last_input_shape": None,
            "total_assignments": 0,
        }
        cpu_index = torch.tensor(neuron_indices, dtype=torch.long)
        index_cache: dict[str, torch.Tensor] = {}

        def hook_fn(inputs, *, layer_idx=layer_idx, cpu_index=cpu_index, index_cache=index_cache):
            if not inputs:
                return inputs
            hidden = inputs[0]
            device_key = str(hidden.device)
            if device_key not in index_cache:
                index_cache[device_key] = cpu_index.to(hidden.device)
            index_tensor = index_cache[device_key]

            steered = hidden.clone()
            steered.index_fill_(-1, index_tensor, alpha)

            layer_stats = stats["layers"][str(layer_idx)]
            layer_stats["call_count"] += 1
            layer_stats["last_input_shape"] = list(hidden.shape)
            positions = int(np.prod(hidden.shape[:-1])) if hidden.ndim > 1 else 1
            layer_stats["total_assignments"] += positions * int(index_tensor.numel())
            return (steered, *inputs[1:])

        handle = target.register_forward_pre_hook(lambda module, inputs, hook_fn=hook_fn: hook_fn(inputs))
        handles.append(handle)
        if debug:
            print(
                f"Registered pre-hook on layer {layer_idx} down_proj "
                f"for {len(neuron_indices)} neurons (alpha={alpha})"
            )

    return handles, stats


def candidate_summary_for_json(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "partition_name": candidate["partition_name"],
        "concept_name": candidate["concept_name"],
        "cluster_id": candidate.get("cluster_id"),
        "cosine_similarity": candidate.get("cosine_similarity"),
        "layer_scope": candidate["layer_scope"],
        "num_neurons": int(candidate["global_vector_indices"].numel()),
        "top_layers": summarize_member_layers(candidate["layer_idx"]),
    }


def make_output_filename(args: argparse.Namespace, steering_label: str) -> str:
    safe_label = steering_label.replace("/", "_")
    return f"task{args.task_idx}_{args.condition}_{safe_label}_seed{args.seed}.json"


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    explicit_init_state_indices = parse_int_csv(args.init_state_indices)
    if explicit_init_state_indices is not None and len(explicit_init_state_indices) != args.num_rollouts:
        raise ValueError(
            f"--init-state-indices length ({len(explicit_init_state_indices)}) must match --num-rollouts ({args.num_rollouts})."
        )

    if args.output_json is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SmolVLA Steering Runner")
    print("=" * 72)
    print(f"Condition   : {args.condition}")
    print(f"Task index  : {args.task_idx}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Max steps   : {args.max_steps}")
    print(f"Seed        : {args.seed}")
    if args.condition in {"cluster", "random_matched"}:
        print(f"Alpha       : {args.alpha}")
        print(f"Layer scope : {args.layer_scope}")
    if args.vision_perturbation != "none":
        print(f"Vision pert.: {args.vision_perturbation}")
        print(f"Vision targ.: {args.vision_target}")
        print(f"Vision str. : {resolve_vision_strength(args.vision_perturbation, args.vision_strength)}")

    policy, device, preprocess_fn, postprocess_fn = load_model(args)
    num_layers = len(get_text_layers(policy))

    candidate_bundle = None
    embeddings_bundle = None
    selected_candidate = None
    steering_spec = None
    hook_handles = []
    hook_stats = None

    if args.condition in {"cluster", "random_matched"}:
        candidate_bundle = torch.load(args.candidate_bundle_pt.resolve(), map_location="cpu")
        resolved_candidate_id, raw_candidate = resolve_candidate_from_bundle(
            candidate_bundle,
            candidate_id=args.candidate_id,
            concept=args.concept,
            partition=args.partition,
        )
        filtered_candidate = filter_candidate_members(
            resolved_candidate_id,
            raw_candidate,
            num_layers=num_layers,
            layer_scope=args.layer_scope,
            single_layer_idx=args.single_layer_idx,
        )
        filtered_candidate = maybe_subsample_members(
            filtered_candidate,
            max_neurons=args.max_neurons,
            subsample_seed=args.subsample_seed if args.subsample_seed is not None else args.seed,
        )

        if args.condition == "random_matched":
            embeddings_path = Path(candidate_bundle["embeddings_pt"]).resolve()
            embeddings_bundle = torch.load(embeddings_path, map_location="cpu")
            selected_candidate = sample_random_matched_members(
                filtered_candidate,
                embeddings_bundle,
                num_layers=num_layers,
                match_mode=args.random_match_mode,
                seed=args.seed,
            )
        else:
            selected_candidate = filtered_candidate

        steering_spec = build_steering_spec(
            selected_candidate,
            condition=args.condition,
            alpha=args.alpha,
            concept_name=args.concept,
        )
        print(f"Resolved steering candidate: {steering_spec.label}")
        print(f"  Partition      : {steering_spec.partition_name}")
        print(f"  Num neurons    : {steering_spec.num_neurons}")
        print(f"  Layers touched : {steering_spec.num_layers_touched}")
        print(f"  Top layers     : {steering_spec.top_layers[:5]}")

        hook_handles, hook_stats = register_pre_down_proj_hooks(
            policy,
            steering_spec.layer_to_vectors,
            alpha=args.alpha,
            debug=args.debug_steering,
        )

    elif args.condition == "prompt":
        if not args.prompt_prefix:
            raise ValueError("Provide --prompt-prefix when --condition prompt.")

    if args.dry_run:
        print("Dry run complete.")
        for handle in hook_handles:
            handle.remove()
        return 0

    env_args = SimpleNamespace(task_idx=args.task_idx, suite=args.suite)
    env, task_suite, task_description, init_states = setup_env(env_args)

    if args.condition == "prompt":
        task_description = f"{args.prompt_prefix} {task_description}"

    results = []
    rollout_start = time.time()
    try:
        for rollout_idx in range(args.num_rollouts):
            print(f"\nRollout {rollout_idx + 1}/{args.num_rollouts}...")
            rollout_args = SimpleNamespace(
                seed=args.seed,
                max_steps=args.max_steps,
                log_every=args.log_every,
            )
            rollout_init_state_idx = (
                explicit_init_state_indices[rollout_idx] if explicit_init_state_indices is not None else None
            )
            rollout_vision_context = build_vision_context(args, seed_offset=rollout_idx)
            result = run_rollout(
                env,
                policy,
                task_description,
                rollout_args,
                device,
                preprocess_fn=preprocess_fn,
                postprocess_fn=postprocess_fn,
                seed_offset=rollout_idx,
                init_states=init_states,
                init_state_idx=rollout_init_state_idx,
                vision_context=rollout_vision_context,
            )
            results.append(result)
            print(
                f"  avg_displacement={result['avg_displacement']:.4f}, "
                f"max_height={result['max_height']:.4f}, "
                f"success={result['success']}, "
                f"steps={result['total_steps']}"
            )
    finally:
        for handle in hook_handles:
            handle.remove()
        env.close()

    elapsed = time.time() - rollout_start
    avg_speeds = [r["avg_displacement"] for r in results]
    max_heights = [r["max_height"] for r in results]
    successes = [r["success"] for r in results]

    serializable_results = []
    for rollout_index, result in enumerate(results):
        serializable_results.append(
            {
                "rollout_index": rollout_index,
                **result,
                "vision_context": result["vision_context"],
                "ee_positions": [list(pos) for pos in result["ee_positions"]],
                "displacements": list(result["displacements"]),
            }
        )

    steering_json = None
    if steering_spec is not None and selected_candidate is not None:
        steering_json = {
            "label": steering_spec.label,
            "source_mode": steering_spec.source_mode,
            "concept_name": steering_spec.concept_name,
            "candidate_id": steering_spec.candidate_id,
            "template_candidate_id": steering_spec.template_candidate_id,
            "alpha": steering_spec.alpha,
            "layer_scope": steering_spec.layer_scope,
            "partition_name": steering_spec.partition_name,
            "cluster_id": steering_spec.cluster_id,
            "num_neurons": steering_spec.num_neurons,
            "num_layers_touched": steering_spec.num_layers_touched,
            "top_layers": steering_spec.top_layers,
            "candidate_summary": candidate_summary_for_json(selected_candidate),
            "hook_stats": hook_stats,
        }

    output = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": args.model_id,
        "suite": args.suite,
        "task_idx": args.task_idx,
        "task_description": task_description,
        "condition": args.condition,
        "prompt_prefix": args.prompt_prefix,
        "seed": args.seed,
        "device": str(device),
        "num_rollouts": args.num_rollouts,
        "init_state_indices": explicit_init_state_indices,
        "vision": {
            "perturbation": args.vision_perturbation,
            "target": args.vision_target,
            "strength": resolve_vision_strength(args.vision_perturbation, args.vision_strength),
            "seed": args.vision_seed,
        },
        "max_steps": args.max_steps,
        "elapsed_seconds": round(elapsed, 3),
        "summary": {
            "avg_speed_mean": float(np.mean(avg_speeds)) if avg_speeds else 0.0,
            "avg_speed_std": float(np.std(avg_speeds)) if avg_speeds else 0.0,
            "max_height_mean": float(np.mean(max_heights)) if max_heights else 0.0,
            "max_height_std": float(np.std(max_heights)) if max_heights else 0.0,
            "success_rate": (sum(successes) / len(successes)) if successes else 0.0,
            "success_count": int(sum(successes)),
        },
        "steering": steering_json,
        "rollouts": serializable_results,
    }

    steering_label = steering_spec.label if steering_spec is not None else args.condition
    output_path = (
        args.output_json.resolve()
        if args.output_json is not None
        else (args.output_dir / make_output_filename(args, steering_label))
    )
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nResults saved to {output_path}")
    print(f"Success rate: {sum(successes)}/{len(successes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
