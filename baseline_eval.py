#!/usr/bin/env python3
"""
baseline_eval.py — Baseline evaluation for Project Lumina
==========================================================

Reproduces baseline conditions from:
  "Mechanistic Interpretability for Steering VLAs" (arXiv:2509.00328)

Runs SmolVLA (smolvla_libero, 7-DOF) in LIBERO-Long simulation tasks
with optional steering conditions: prompt variants or random neuron injection.

Usage:
  python baseline_eval.py --condition none --task_idx 0 --num_rollouts 10
  python baseline_eval.py --condition prompt_fast --task_idx 0
  python baseline_eval.py --condition random_injection --task_idx 0 --alpha 10.0
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS AND SETUP
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import os
import json
import time
import argparse

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not found. Activate the project_Lumina conda env:")
    print("  conda activate project_Lumina")
    sys.exit(1)

try:
    import torchvision.transforms as T
except ImportError:
    print("ERROR: torchvision not found. Install it or activate project_Lumina env.")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not found. pip install Pillow")
    sys.exit(1)

# ── LeRobot / SmolVLA ────────────────────────────────────────────────────────
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as e:
    print(f"ERROR: Could not import SmolVLAPolicy from lerobot: {e}")
    print("Make sure lerobot is installed: pip install lerobot")
    sys.exit(1)

try:
    from lerobot.policies.factory import make_pre_post_processors
except ImportError:
    # Older lerobot versions may not have this; we'll handle it at runtime
    make_pre_post_processors = None
    print("WARNING: Could not import make_pre_post_processors from lerobot.policies.factory")
    print("  Will attempt raw inference without pre/post processing.")

# ── LIBERO ────────────────────────────────────────────────────────────────────
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    print(f"ERROR: Could not import LIBERO: {e}")
    print("Make sure libero is installed. Try:")
    print("  pip install libero")
    print("  # or clone from https://github.com/Lifelong-Robot-Learning/LIBERO")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation of SmolVLA on LIBERO-Long tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id", type=str, default="HuggingFaceVLA/smolvla_libero",
        help="HuggingFace model ID for the SmolVLA checkpoint",
    )
    parser.add_argument(
        "--task_idx", type=int, default=0,
        help="Which LIBERO-Long task to run (0-9)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=10,
        help="Number of rollouts per condition",
    )
    parser.add_argument(
        "--max_steps", type=int, default=600,
        help="Maximum environment steps per rollout",
    )
    parser.add_argument(
        "--log_every", type=int, default=25,
        help="Print rollout progress every N steps (0 disables)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--condition", type=str, default="none",
        choices=["none", "prompt_fast", "prompt_slow",
                 "prompt_high", "prompt_low", "random_injection"],
        help="Steering condition to apply",
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0,
        help="Steering coefficient for neuron injection",
    )
    parser.add_argument(
        "--output_dir", type=str, default="baseline_results",
        help="Directory to save result JSON files",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(args):
    """Load SmolVLA policy and optional pre/post processors."""
    print(f"\nLoading model: {args.model_id}")
    t0 = time.time()

    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.eval()

    # Use MPS if available (macOS), else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = policy.to(device)

    elapsed = time.time() - t0
    print(f"Model loaded on {device} in {elapsed:.1f}s")

    # Try to build pre/post processors (mirrors smolvla_minimal_inference_check.py)
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
        except Exception as e:
            print(f"WARNING: Could not build pre/post processors: {e}")
            print("  Will attempt raw inference.")

    return policy, device, preprocess_fn, postprocess_fn


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_env(args):
    """Load LIBERO task suite and create the OffScreenRenderEnv."""
    print(f"\nSetting up LIBERO environment...")
    benchmark_dict = benchmark.get_benchmark_dict()

    # The paper refers to "LIBERO-Long" which maps to "libero_10" in the codebase
    # (10 long-horizon multi-step manipulation tasks).
    # Available suites: libero_spatial, libero_object, libero_goal, libero_90, libero_10
    SUITE_ALIASES = {
        "libero_long": "libero_10",
    }
    suite_name = getattr(args, "suite", "libero_10")
    suite_name = SUITE_ALIASES.get(suite_name, suite_name)

    print(f"  Benchmark suite: {suite_name}")
    print(f"  Available suites: {list(benchmark_dict.keys())}")

    if suite_name not in benchmark_dict:
        print(f"ERROR: Suite '{suite_name}' not found. Available: {list(benchmark_dict.keys())}")
        sys.exit(1)

    task_suite = benchmark_dict[suite_name]()

    task = task_suite.get_task(args.task_idx)
    task_description = task.language  # LIBERO Task objects use .language attribute
    task_bddl_path = task_suite.get_task_bddl_file_path(args.task_idx)
    print(f"Task {args.task_idx}: {task_description}")
    print(f"  BDDL file: {task_bddl_path}")

    env_args = {
        "bddl_file_name": task_bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
        "render_camera": "agentview",
    }
    env = OffScreenRenderEnv(**env_args)
    print("Environment created.")

    # Load initial states for reproducible evaluation
    init_states = task_suite.get_task_init_states(args.task_idx)
    print(f"  Init states loaded: {init_states.shape}")

    return env, task_suite, task_description, init_states


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONDITION SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def apply_condition(args, task_description, policy, device):
    """
    Modify task_description or register a forward hook depending on --condition.
    Returns (modified_task_description, hook_handle_or_None).
    """
    hook_handle = None

    if args.condition == "none":
        pass  # no modification
    elif args.condition == "prompt_fast":
        task_description = "fast " + task_description
    elif args.condition == "prompt_slow":
        task_description = "slow " + task_description
    elif args.condition == "prompt_high":
        task_description = "high " + task_description
    elif args.condition == "prompt_low":
        task_description = "low " + task_description
    elif args.condition == "random_injection":
        # Pick 10 random neuron indices from the FFN hidden dim
        # We need to discover the hidden dim size first
        hidden_dim = _get_hidden_dim(policy)
        rng = np.random.RandomState(args.seed)
        neuron_indices = rng.choice(hidden_dim, size=10, replace=False).tolist()
        print(f"Random injection: {len(neuron_indices)} neurons from dim={hidden_dim}")
        print(f"  Neuron indices: {neuron_indices}")
        hook_handle = register_steering_hook(
            policy, neuron_indices, args.alpha, layer_idx=25
        )

    print(f"Condition '{args.condition}' applied. Task description: \"{task_description}\"")
    return task_description, hook_handle


def _get_hidden_dim(policy):
    """Try to discover the FFN hidden dimension from the model."""
    # Try common paths to a down_proj layer
    search_paths = [
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj",
        "model.model.layers.0.mlp.down_proj",
    ]
    for name, module in policy.named_modules():
        if "down_proj" in name and hasattr(module, "in_features"):
            print(f"  Found down_proj at: {name}  (in={module.in_features}, out={module.out_features})")
            return module.in_features

    # Fallback: scan for any Linear layer in an mlp block
    for name, module in policy.named_modules():
        if "mlp" in name and hasattr(module, "in_features"):
            print(f"  Fallback — found MLP linear at: {name}  (in={module.in_features})")
            return module.in_features

    print("  WARNING: Could not discover hidden dim. Defaulting to 2048.")
    return 2048


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OBSERVATION FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

# Image transform: HWC uint8 numpy → CHW float32 tensor [0,1] at 256x256
# (The LeRobot preprocessor handles normalization, so we just need to_tensor)
_to_tensor = T.Compose([T.Resize((256, 256)), T.ToTensor()])


def format_obs(obs, task_description, device, preprocess_fn=None, _printed_keys=[False]):
    """
    Convert a raw LIBERO observation dict into the format expected by SmolVLA.

    The smolvla_libero checkpoint expects:
      - observation.images.image:  (3, 256, 256) float32 — agentview camera
      - observation.images.image2: (3, 256, 256) float32 — wrist camera
      - observation.state:         (8,) float32 — eef_pos(3) + eef_quat(4) + gripper(1)
      - task:                      str — language instruction

    The preprocess pipeline then handles batching, tokenization, normalization,
    and device placement.
    """
    # ── Debug: print obs keys once ────────────────────────────────────────────
    if not _printed_keys[0]:
        _printed_keys[0] = True
        print("\n  [DEBUG] Raw observation keys and shapes:")
        for k, v in sorted(obs.items()):
            if isinstance(v, np.ndarray):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"    {k}: type={type(v).__name__}")

    # ── Extract camera images as CHW float32 tensors ──────────────────────────
    image_primary = _extract_image_tensor(obs, ["agentview_image", "agentview_rgb"], "primary camera")
    image_wrist = _extract_image_tensor(obs, ["robot0_eye_in_hand_image", "wrist_image"], "wrist camera")

    # ── Extract robot state (8-dim: pos3 + quat4 + gripper1) ─────────────────
    state_parts = []
    if "robot0_eef_pos" in obs:
        state_parts.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32).flatten())
    if "robot0_eef_quat" in obs:
        state_parts.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).flatten())
    if "robot0_gripper_qpos" in obs:
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).flatten()
        # smolvla_libero expects 1 gripper value, LIBERO gives 2 (left/right finger)
        # Use the mean of both fingers as a single gripper state
        state_parts.append(np.array([gripper.mean()], dtype=np.float32))

    state = np.concatenate(state_parts) if state_parts else np.zeros(8, dtype=np.float32)

    # ── Build frame dict for the LeRobot preprocess pipeline ──────────────────
    # Keys must match what smolvla_libero was trained with:
    #   observation.images.image  (agentview)
    #   observation.images.image2 (wrist)
    frame = {}
    if image_primary is not None:
        frame["observation.images.image"] = image_primary
    if image_wrist is not None:
        frame["observation.images.image2"] = image_wrist
    frame["observation.state"] = torch.tensor(state, dtype=torch.float32)
    frame["task"] = task_description

    # ── Run through preprocess pipeline ───────────────────────────────────────
    if preprocess_fn is not None:
        try:
            batch = preprocess_fn(frame)
            return batch
        except Exception as e:
            print(f"  WARNING: preprocess_fn failed ({e}), attempting manual formatting")

    # ── Fallback: manual batch construction (tokenization still needed) ───────
    # This is a best-effort fallback — it will likely fail because the policy
    # requires tokenized language. But we try anyway for debugging.
    obs_dict = {}
    if image_primary is not None:
        obs_dict["observation.images.image"] = image_primary.unsqueeze(0).to(device)
    if image_wrist is not None:
        obs_dict["observation.images.image2"] = image_wrist.unsqueeze(0).to(device)
    obs_dict["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    obs_dict["task"] = [task_description]
    return obs_dict


def _extract_image_tensor(obs, candidate_keys, label):
    """Try candidate keys in order, return a CHW float32 tensor or None."""
    for k in candidate_keys:
        if k in obs:
            arr = obs[k]
            if isinstance(arr, np.ndarray):
                # Robosuite images: uint8 HWC or float [0,1] HWC
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
                return _to_tensor(pil_img)  # (C, H, W) float32 [0,1]
    print(f"  WARNING: No key found for {label}. Tried: {candidate_keys}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FORWARD HOOK FOR INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def register_steering_hook(policy, neuron_indices, alpha, layer_idx=25):
    """
    Register a forward hook on the down_proj of FFN layer `layer_idx`.
    Overrides activations at neuron_indices with scalar alpha.
    Returns the hook handle (call handle.remove() to clean up).
    """

    def hook_fn(module, input, output):
        output[:, :, neuron_indices] = alpha
        return output

    # Try known paths to the down_proj in SmolVLA's wrapped LLM
    candidate_paths = [
        # smolvla wraps a VLM with an expert head
        lambda: policy.model.vlm_with_expert.vlm.model.text_model.layers[layer_idx].mlp.down_proj,
        # simpler path (some checkpoints)
        lambda: policy.model.model.layers[layer_idx].mlp.down_proj,
    ]

    for path_fn in candidate_paths:
        try:
            target = path_fn()
            handle = target.register_forward_hook(hook_fn)
            print(f"Hook registered on layer {layer_idx} down_proj, "
                  f"{len(neuron_indices)} neurons, alpha={alpha}")
            return handle
        except (AttributeError, IndexError):
            continue

    # If none of the candidate paths worked, print diagnostic info
    print("WARNING: Could not find down_proj at expected paths.")
    print("  Scanning model for down_proj modules:")
    found_any = False
    for name, module in policy.named_modules():
        if "down_proj" in name:
            print(f"    {name}")
            found_any = True
    if not found_any:
        print("    (none found — model may use a different FFN architecture)")
    print("  Please update the paths in register_steering_hook().")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SINGLE ROLLOUT
# ═══════════════════════════════════════════════════════════════════════════════

def run_rollout(env, policy, task_description, args, device,
                preprocess_fn=None, postprocess_fn=None,
                seed_offset=0, init_states=None):
    """
    Execute a single rollout.

    Returns a dict with ee_positions, displacements, avg_displacement,
    max_height, total_steps, and success.
    """
    # Seed and reset
    env.seed(args.seed + seed_offset)
    obs = env.reset()

    # Set init state for reproducibility (LIBERO provides 50 per task)
    if init_states is not None:
        init_idx = seed_offset % len(init_states)
        obs = env.set_init_state(init_states[init_idx])
        print(f"  Using init state {init_idx}")

    ee_positions = []
    success = False
    total_steps = 0
    step_times = []
    infer_times = []
    rollout_start = time.time()

    # Determine which axis is "up" — in LIBERO/Robosuite, Z is up
    UP_AXIS = 2  # index into [x, y, z]
    UP_LABEL = "z"

    try:
        for step_idx in range(args.max_steps):
            step_start = time.time()
            # ── Format observation ────────────────────────────────────────
            formatted_obs = format_obs(
                obs, task_description, device, preprocess_fn=preprocess_fn
            )

            # ── Policy inference ──────────────────────────────────────────
            infer_start = time.time()
            with torch.no_grad():
                raw_action = policy.select_action(formatted_obs)
            infer_times.append(time.time() - infer_start)

            # ── Post-process if available ─────────────────────────────────
            if postprocess_fn is not None:
                try:
                    raw_action = postprocess_fn(raw_action)
                except Exception:
                    pass  # use raw action if postprocess fails

            # ── Convert action to numpy ───────────────────────────────────
            action_np = _action_to_numpy(raw_action)

            # ── Record end-effector position ──────────────────────────────
            ee_pos = _extract_ee_pos(obs)
            ee_positions.append(ee_pos.tolist())

            # ── Step environment ──────────────────────────────────────────
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

            # Check task success via LIBERO's dedicated method
            task_success = env.check_success()
            if task_success:
                success = True
                # Record final position too
                ee_pos_final = _extract_ee_pos(obs)
                ee_positions.append(ee_pos_final.tolist())
                break

            if done:
                # Episode ended (horizon reached) without success
                ee_pos_final = _extract_ee_pos(obs)
                ee_positions.append(ee_pos_final.tolist())
                break

    except Exception as e:
        print(f"  ERROR during rollout at step {total_steps}: {e}")
        import traceback
        traceback.print_exc()

    # ── Compute metrics ───────────────────────────────────────────────────────
    ee_arr = np.array(ee_positions)  # (T, 3)
    if len(ee_arr) > 1:
        diffs = np.diff(ee_arr, axis=0)
        displacements = np.linalg.norm(diffs, axis=1).tolist()
        avg_displacement = float(np.mean(displacements))
    else:
        displacements = []
        avg_displacement = 0.0

    max_height = float(ee_arr[:, UP_AXIS].max()) if len(ee_arr) > 0 else 0.0

    return {
        "ee_positions": ee_positions,
        "displacements": displacements,
        "avg_displacement": avg_displacement,
        "max_height": max_height,
        "max_height_axis": UP_LABEL,
        "total_steps": total_steps,
        "success": success,
    }


def _action_to_numpy(raw_action):
    """Convert policy output (tensor or dict) to a flat numpy action array."""
    if isinstance(raw_action, dict):
        # Some policies return a dict with an "action" key
        if "action" in raw_action:
            raw_action = raw_action["action"]
        else:
            # Grab the first tensor value
            raw_action = next(iter(raw_action.values()))

    if isinstance(raw_action, torch.Tensor):
        action_np = raw_action.detach().cpu().numpy()
    else:
        action_np = np.asarray(raw_action)

    # Flatten extra dimensions: (1, chunk, action_dim) → (action_dim,)
    # We only need the first step of a chunked prediction
    while action_np.ndim > 1:
        action_np = action_np[0]

    return action_np


def _extract_ee_pos(obs):
    """Extract 3D end-effector position from obs dict."""
    ee_keys = ["robot0_eef_pos", "eef_pos", "ee_pos"]
    for k in ee_keys:
        if k in obs:
            return np.asarray(obs[k], dtype=np.float64).flatten()[:3]
    # Fallback: return zeros
    return np.zeros(3, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 9 & 10. MAIN LOOP + SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 70)
    print("Project Lumina — Baseline Evaluation")
    print("=" * 70)
    print(f"  Model       : {args.model_id}")
    print(f"  Task index  : {args.task_idx}")
    print(f"  Condition   : {args.condition}")
    print(f"  Num rollouts: {args.num_rollouts}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Log every   : {args.log_every}")
    print(f"  Seed        : {args.seed}")
    if args.condition == "random_injection":
        print(f"  Alpha       : {args.alpha}")

    # ── Load model ────────────────────────────────────────────────────────────
    policy, device, preprocess_fn, postprocess_fn = load_model(args)

    # ── Setup environment ─────────────────────────────────────────────────────
    env, task_suite, task_description, init_states = setup_env(args)

    # ── Apply condition ───────────────────────────────────────────────────────
    task_description, hook_handle = apply_condition(
        args, task_description, policy, device
    )

    # ── Run rollouts ──────────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    for i in range(args.num_rollouts):
        print(f"\nRollout {i + 1}/{args.num_rollouts}...")
        result = run_rollout(
            env, policy, task_description, args, device,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            seed_offset=i,
            init_states=init_states,
        )
        results.append(result)
        print(f"  avg_displacement={result['avg_displacement']:.4f}, "
              f"max_height={result['max_height']:.4f}, "
              f"success={result['success']}, "
              f"steps={result['total_steps']}")

    elapsed = time.time() - t_start
    print(f"\nAll rollouts completed in {elapsed:.1f}s")

    # ── Clean up hook ─────────────────────────────────────────────────────────
    if hook_handle is not None:
        hook_handle.remove()
        print("Steering hook removed.")

    # ── Summary stats ─────────────────────────────────────────────────────────
    avg_speeds = [r["avg_displacement"] for r in results]
    max_heights = [r["max_height"] for r in results]
    successes = [r["success"] for r in results]

    print(f"\n{'=' * 50}")
    print(f"  SUMMARY  (condition={args.condition})")
    print(f"{'=' * 50}")
    print(f"  Avg speed (mean) : {np.mean(avg_speeds):.4f} +/- {np.std(avg_speeds):.4f}")
    print(f"  Max height (mean): {np.mean(max_heights):.4f} +/- {np.std(max_heights):.4f}")
    print(f"  Success rate     : {sum(successes)}/{args.num_rollouts}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"task{args.task_idx}_{args.condition}_seed{args.seed}.json"
    filepath = os.path.join(args.output_dir, filename)

    # Convert ee_positions to serializable format (lists, not numpy)
    serializable_results = []
    for r in results:
        sr = dict(r)
        sr["ee_positions"] = [list(p) for p in sr["ee_positions"]]
        sr["displacements"] = list(sr["displacements"])
        serializable_results.append(sr)

    output = {
        "model_id": args.model_id,
        "task_idx": args.task_idx,
        "task_description": task_description,
        "condition": args.condition,
        "alpha": args.alpha if args.condition == "random_injection" else None,
        "num_rollouts": args.num_rollouts,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "summary": {
            "avg_speed_mean": float(np.mean(avg_speeds)),
            "avg_speed_std": float(np.std(avg_speeds)),
            "max_height_mean": float(np.mean(max_heights)),
            "max_height_std": float(np.std(max_heights)),
            "success_rate": sum(successes) / args.num_rollouts,
            "success_count": sum(successes),
        },
        "rollouts": serializable_results,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filepath}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    env.close()
    print("Environment closed. Done.")


if __name__ == "__main__":
    main()
