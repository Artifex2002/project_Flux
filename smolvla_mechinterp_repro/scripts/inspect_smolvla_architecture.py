#!/usr/bin/env python3
"""
Phase 0 architecture inspection for SmolVLA mechanistic steering reproduction.

This script does two things:
1. Statically inspects the SmolVLA VLM text stack and records candidate FFN hook points.
2. Optionally runs a dummy forward pass to confirm the actual tensor shapes seen by
   `gate_proj`, `up_proj`, and `down_proj`.

The main goal is to answer a practical steering question:
for paper-style FFN activation steering, should we hook the `down_proj` input,
the `down_proj` output, or a different tensor?
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = ROOT / "results" / "phase0_architecture_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SmolVLA architecture and candidate steering hook points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceVLA/smolvla_libero",
        help="SmolVLA policy checkpoint to inspect.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the machine-readable inspection summary.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face downloads instead of forcing local cached files only.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda", "auto"],
        help="Device to use for the optional dummy forward probe.",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default="pick up the block",
        help="Dummy language instruction for the forward probe.",
    )
    parser.add_argument(
        "--skip-forward-probe",
        action="store_true",
        help="Only collect static module metadata and skip the dummy inference probe.",
    )
    return parser.parse_args()


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


def shape_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    if isinstance(value, (tuple, list)):
        return [int(dim) for dim in value]
    return None


def feature_to_dict(feature: Any) -> dict[str, Any]:
    feature_type = getattr(feature, "type", None)
    if feature_type is not None and hasattr(feature_type, "value"):
        feature_type = feature_type.value
    return {
        "type": str(feature_type),
        "shape": shape_list(getattr(feature, "shape", None)),
    }


def tensor_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": shape_list(value),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, list):
        if value and isinstance(value[0], torch.Tensor):
            return {
                "kind": "tensor_list",
                "length": len(value),
                "item_shape": shape_list(value[0]),
                "item_dtype": str(value[0].dtype),
            }
        return {
            "kind": "list",
            "length": len(value),
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "keys": sorted(value.keys()),
        }
    return {
        "kind": type(value).__name__,
    }


def build_dummy_frame(policy: SmolVLAPolicy, task_text: str) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for feature_name, feature in policy.config.input_features.items():
        feature_info = feature_to_dict(feature)
        if feature_info["type"] == "VISUAL":
            frame[feature_name] = torch.zeros(feature.shape, dtype=torch.float32)
        elif feature_info["type"] == "STATE":
            frame[feature_name] = torch.zeros(feature.shape, dtype=torch.float32)
    frame["task"] = task_text
    return frame


def load_policy(args: argparse.Namespace, device: torch.device) -> SmolVLAPolicy:
    local_files_only = not args.allow_network
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    policy = SmolVLAPolicy.from_pretrained(
        args.model_id,
        local_files_only=local_files_only,
    )
    policy.eval()
    policy.to(device)
    return policy


def collect_static_summary(policy: SmolVLAPolicy, device: torch.device) -> dict[str, Any]:
    text_layers = policy.model.vlm_with_expert.vlm.model.text_model.layers
    lm_head = policy.model.vlm_with_expert.vlm.lm_head
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    total_value_vectors = 0
    layer_summaries: list[dict[str, Any]] = []

    for layer_idx, layer in enumerate(text_layers):
        mlp = layer.mlp
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        total_value_vectors += int(down_proj.in_features)
        layer_summaries.append(
            {
                "layer_idx": layer_idx,
                "layer_path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}",
                "mlp_type": type(mlp).__name__,
                "act_fn": type(getattr(mlp, "act_fn", None)).__name__,
                "gate_proj": {
                    "path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.gate_proj",
                    "weight_shape": shape_list(gate_proj.weight),
                    "in_features": int(gate_proj.in_features),
                    "out_features": int(gate_proj.out_features),
                },
                "up_proj": {
                    "path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.up_proj",
                    "weight_shape": shape_list(up_proj.weight),
                    "in_features": int(up_proj.in_features),
                    "out_features": int(up_proj.out_features),
                },
                "down_proj": {
                    "path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.down_proj",
                    "weight_shape": shape_list(down_proj.weight),
                    "in_features": int(down_proj.in_features),
                    "out_features": int(down_proj.out_features),
                    "value_vector_source": {
                        "matrix": "down_proj.weight",
                        "matrix_shape": shape_list(down_proj.weight),
                        "value_vector_axis": "columns",
                        "num_value_vectors": int(down_proj.in_features),
                        "value_vector_dim": int(down_proj.out_features),
                        "reason": (
                            "PyTorch Linear stores weights as [out_features, in_features]. "
                            "Each FFN value vector is therefore a column of down_proj.weight."
                        ),
                    },
                },
                "recommended_hook": {
                    "module_path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.down_proj",
                    "hook_type": "forward_pre_hook",
                    "target_tensor": "down_proj input",
                    "target_last_dim": int(down_proj.in_features),
                    "paper_alignment": "Matches the FFN activation f_theta(x) in the paper.",
                },
                "alternative_hook": {
                    "module_path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.gate_proj",
                    "hook_type": "forward_hook",
                    "target_tensor": "gate_proj output",
                    "target_last_dim": int(gate_proj.out_features),
                    "paper_alignment": "Closest to the appendix gate-proj variant, not Equation (5) exactly.",
                },
                "not_recommended_hook": {
                    "module_path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.down_proj",
                    "hook_type": "forward_hook",
                    "target_tensor": "down_proj output",
                    "target_last_dim": int(down_proj.out_features),
                    "reason": (
                        "This operates after the value vectors have already been combined and projected back "
                        "to residual space, so it does not overwrite individual FFN activations."
                    ),
                },
            }
        )

    return {
        "inspect_device": str(device),
        "policy_class": type(policy).__name__,
        "policy_config_class": type(policy.config).__name__,
        "policy_input_features": {
            name: feature_to_dict(feature) for name, feature in policy.config.input_features.items()
        },
        "policy_output_features": {
            name: feature_to_dict(feature) for name, feature in policy.config.output_features.items()
        },
        "vlm_backbone": {
            "base_model_id": policy.config.vlm_model_name,
            "num_text_layers": len(text_layers),
            "hidden_size": int(lm_head.weight.shape[1]),
            "vocab_size": int(lm_head.weight.shape[0]),
            "lm_head_weight_shape": shape_list(lm_head.weight),
            "tokenizer_vocab_size": len(tokenizer),
            "total_value_vectors": total_value_vectors,
        },
        "action_expert": {
            "attention_mode": policy.config.attention_mode,
            "num_vlm_layers_config": int(policy.config.num_vlm_layers),
            "num_expert_layers_config": int(policy.config.num_expert_layers),
            "num_vlm_layers_runtime": int(policy.model.vlm_with_expert.num_vlm_layers),
            "num_expert_layers_runtime": int(policy.model.vlm_with_expert.num_expert_layers),
            "expert_hidden_size": int(policy.model.vlm_with_expert.expert_hidden_size),
            "self_attn_every_n_layers": int(policy.config.self_attn_every_n_layers),
            "handoff_modules": {
                "state_proj": {
                    "path": "model.state_proj",
                    "weight_shape": shape_list(policy.model.state_proj.weight),
                },
                "action_in_proj": {
                    "path": "model.action_in_proj",
                    "weight_shape": shape_list(policy.model.action_in_proj.weight),
                },
                "action_out_proj": {
                    "path": "model.action_out_proj",
                    "weight_shape": shape_list(policy.model.action_out_proj.weight),
                },
            },
        },
        "text_layers": layer_summaries,
    }


def collect_forward_probe(
    policy: SmolVLAPolicy,
    model_id: str,
    task_text: str,
    device: torch.device,
) -> dict[str, Any]:
    preprocess, _ = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    dummy_frame = build_dummy_frame(policy, task_text)
    batch = preprocess(dummy_frame)
    batch_summary = {key: tensor_summary(value) for key, value in batch.items()}

    text_layers = policy.model.vlm_with_expert.vlm.model.text_model.layers
    layer_indices = sorted({0, len(text_layers) - 1})
    captures: dict[int, dict[str, Any]] = {idx: {} for idx in layer_indices}
    hooks = []

    for layer_idx in layer_indices:
        mlp = text_layers[layer_idx].mlp

        def gate_hook(_module, _args, output, *, idx: int = layer_idx) -> None:
            captures[idx]["gate_proj_output_shape"] = shape_list(output)
            captures[idx]["gate_proj_output_dtype"] = str(output.dtype)

        def up_hook(_module, _args, output, *, idx: int = layer_idx) -> None:
            captures[idx]["up_proj_output_shape"] = shape_list(output)
            captures[idx]["up_proj_output_dtype"] = str(output.dtype)

        def down_pre_hook(_module, args, *, idx: int = layer_idx) -> None:
            input_tensor = args[0]
            captures[idx]["down_proj_input_shape"] = shape_list(input_tensor)
            captures[idx]["down_proj_input_dtype"] = str(input_tensor.dtype)

        def down_hook(_module, _args, output, *, idx: int = layer_idx) -> None:
            captures[idx]["down_proj_output_shape"] = shape_list(output)
            captures[idx]["down_proj_output_dtype"] = str(output.dtype)

        hooks.append(mlp.gate_proj.register_forward_hook(gate_hook))
        hooks.append(mlp.up_proj.register_forward_hook(up_hook))
        hooks.append(mlp.down_proj.register_forward_pre_hook(down_pre_hook))
        hooks.append(mlp.down_proj.register_forward_hook(down_hook))

    with torch.no_grad():
        action = policy.select_action(batch)

    for hook in hooks:
        hook.remove()

    return {
        "dummy_task_text": task_text,
        "dummy_frame_keys": sorted(dummy_frame.keys()),
        "dummy_frame_summary": {key: tensor_summary(value) for key, value in dummy_frame.items()},
        "preprocessed_batch_summary": batch_summary,
        "action_shape": shape_list(action),
        "action_dtype": str(action.dtype),
        "layer_probes": [
            {
                "layer_idx": layer_idx,
                "layer_path": f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp",
                **captures[layer_idx],
            }
            for layer_idx in layer_indices
        ],
    }


def build_conclusions(summary: dict[str, Any]) -> dict[str, Any]:
    hidden_size = summary["vlm_backbone"]["hidden_size"]
    first_layer = summary["text_layers"][0]
    intermediate_size = first_layer["down_proj"]["in_features"]
    conclusions = {
        "value_vector_source_matrix": "down_proj.weight",
        "value_vector_orientation": "columns",
        "value_vector_dim": first_layer["down_proj"]["out_features"],
        "num_value_vectors_per_layer": intermediate_size,
        "correct_paper_style_hook": {
            "module_path": first_layer["recommended_hook"]["module_path"],
            "hook_type": "forward_pre_hook",
            "target": "down_proj input",
            "target_last_dim": intermediate_size,
        },
        "why_not_down_proj_forward_hook": (
            "A forward hook on down_proj changes the residual-space output of size "
            f"{hidden_size}, not the FFN activation vector of size {intermediate_size}. "
            "That means it does not match the paper's intervention operator."
        ),
        "smolvla_specific_note": (
            "Steering happens inside the SmolVLA VLM text backbone, while the final robot actions are produced "
            "by a downstream action expert. This is an adaptation of the paper rather than a literal one-to-one reproduction."
        ),
    }

    runtime_probe = summary.get("runtime_forward_probe")
    if runtime_probe and "layer_probes" in runtime_probe:
        conclusions["runtime_confirmation"] = runtime_probe["layer_probes"]
    return conclusions


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        policy = load_policy(args, device)
    except Exception as exc:
        print(f"Failed to load policy '{args.model_id}': {exc}", file=sys.stderr)
        return 1

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "local_files_only": not args.allow_network,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "source_files": {
            "smolvla_policy": inspect.getfile(SmolVLAPolicy),
            "policy_instance_model": inspect.getfile(policy.model.__class__),
            "vlm_with_expert_model": inspect.getfile(policy.model.vlm_with_expert.__class__),
            "vlm_model_class": inspect.getfile(policy.model.vlm_with_expert.vlm.__class__),
        },
    }
    summary.update(collect_static_summary(policy, device))

    if args.skip_forward_probe:
        summary["runtime_forward_probe"] = {
            "skipped": True,
            "reason": "--skip-forward-probe was provided.",
        }
    else:
        try:
            summary["runtime_forward_probe"] = collect_forward_probe(
                policy=policy,
                model_id=args.model_id,
                task_text=args.task_text,
                device=device,
            )
        except Exception as exc:
            summary["runtime_forward_probe"] = {
                "skipped": False,
                "error": str(exc),
            }

    summary["conclusions"] = build_conclusions(summary)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    first_layer = summary["text_layers"][0]
    print("=" * 72)
    print("SmolVLA Phase 0 Architecture Inspection")
    print("=" * 72)
    print(f"Model ID:              {summary['model_id']}")
    print(f"Inspect device:        {summary['inspect_device']}")
    print(f"Base VLM:              {summary['vlm_backbone']['base_model_id']}")
    print(f"Text layers:           {summary['vlm_backbone']['num_text_layers']}")
    print(f"Residual hidden size:  {summary['vlm_backbone']['hidden_size']}")
    print(f"FFN intermediate size: {first_layer['down_proj']['in_features']}")
    print(f"Total value vectors:   {summary['vlm_backbone']['total_value_vectors']}")
    print(
        "Recommended hook:      "
        f"{first_layer['recommended_hook']['module_path']} "
        f"via {first_layer['recommended_hook']['hook_type']}"
    )
    print(
        "Why:                   "
        "the down_proj input is the FFN activation vector that the paper overwrites."
    )

    runtime_probe = summary["runtime_forward_probe"]
    if "layer_probes" in runtime_probe:
        first_probe = runtime_probe["layer_probes"][0]
        print(
            "Runtime probe:         "
            f"down_proj input {first_probe.get('down_proj_input_shape')} -> "
            f"output {first_probe.get('down_proj_output_shape')}"
        )
    elif "error" in runtime_probe:
        print(f"Runtime probe:         failed ({runtime_probe['error']})")
    else:
        print("Runtime probe:         skipped")

    print(f"Summary written to:    {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
