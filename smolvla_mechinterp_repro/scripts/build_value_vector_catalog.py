#!/usr/bin/env python3
"""
Phase 1 value-vector catalog builder for SmolVLA.

This script extracts FFN value vectors from every SmolVLA VLM text layer,
projects them into token space using the VLM lm_head, and writes a structured
JSONL catalog with top-k decoded tokens per value vector.

The implementation is intentionally shard-friendly:
- select a layer range,
- cap vectors per layer for smoke tests,
- tune the vector batch size,
- run fully offline from local Hugging Face cache when desired.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSONL = ROOT / "results" / "value_vector_catalog_top30.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a structured SmolVLA value-vector catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceVLA/smolvla_libero",
        help="SmolVLA checkpoint to inspect.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Path to the JSONL catalog output.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for a summary JSON. Defaults to <output-jsonl>.summary.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top projected tokens to store per value vector.",
    )
    parser.add_argument(
        "--vector-batch-size",
        type=int,
        default=128,
        help="Number of value vectors to decode at once.",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=0,
        help="Inclusive start index for text layers to process.",
    )
    parser.add_argument(
        "--layer-stop",
        type=int,
        default=None,
        help="Exclusive stop index for text layers to process. Defaults to all remaining layers.",
    )
    parser.add_argument(
        "--max-vectors-per-layer",
        type=int,
        default=None,
        help="Optional cap for vectors per layer, useful for smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device used for projection computation.",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Computation dtype for the value-vector projection step.",
    )
    parser.add_argument(
        "--include-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store full-vocab softmax probabilities for the selected top-k tokens.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model downloads instead of forcing local cache only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file. Defaults to <output-jsonl>.log.",
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


def resolve_compute_dtype(dtype_arg: str, device: torch.device, default_dtype: torch.dtype) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cpu":
        return torch.float32
    return default_dtype


def ensure_writable(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")


def load_policy(model_id: str, allow_network: bool, device: torch.device) -> SmolVLAPolicy:
    local_files_only = not allow_network
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    policy = SmolVLAPolicy.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    policy.eval()
    policy.to(device)
    return policy


def render_token_factory(tokenizer):
    cache: dict[int, str] = {}

    def render(token_id: int) -> str:
        if token_id not in cache:
            cache[token_id] = tokenizer.decode([token_id], skip_special_tokens=False)
        return cache[token_id]

    return render, cache


def batched_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("smolvla_value_vector_catalog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_layer_record(
    *,
    model_id: str,
    layer_idx: int,
    layer_path: str,
    vector_index: int,
    global_vector_index: int,
    vector_tensor: torch.Tensor,
    token_ids: list[int],
    token_texts: list[str],
    top_logits: list[float],
    top_probs: list[float] | None,
    down_proj_weight_shape: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": model_id,
        "layer_idx": layer_idx,
        "layer_path": layer_path,
        "vector_index": vector_index,
        "global_vector_index": global_vector_index,
        "value_vector_source": {
            "matrix": "down_proj.weight",
            "matrix_column_index": vector_index,
            "matrix_shape": down_proj_weight_shape,
            "vector_dim": hidden_size,
            "num_vectors_in_layer": intermediate_size,
        },
        "value_vector_norm": float(torch.linalg.vector_norm(vector_tensor).item()),
        "top_k": top_k,
        "top_token_ids": token_ids,
        "top_tokens": token_texts,
        "top_token_logits": top_logits,
    }
    if top_probs is not None:
        record["top_token_probs"] = top_probs
    return record


def main() -> int:
    args = parse_args()
    output_jsonl = args.output_jsonl.resolve()
    summary_json = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    )
    log_file = (
        args.log_file.resolve()
        if args.log_file is not None
        else output_jsonl.with_suffix(output_jsonl.suffix + ".log")
    )

    try:
        ensure_writable(output_jsonl, args.overwrite)
        ensure_writable(summary_json, args.overwrite)
        ensure_writable(log_file, args.overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = setup_logger(log_file)
    device = select_device(args.device)
    wall_start = time.time()

    try:
        policy = load_policy(args.model_id, args.allow_network, device)
    except Exception as exc:
        logger.error("Failed to load policy '%s': %s", args.model_id, exc)
        return 1

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    render_token, token_cache = render_token_factory(tokenizer)
    text_layers = policy.model.vlm_with_expert.vlm.model.text_model.layers
    lm_head = policy.model.vlm_with_expert.vlm.lm_head

    if args.layer_start < 0 or args.layer_start >= len(text_layers):
        print(f"--layer-start must be in [0, {len(text_layers) - 1}]", file=sys.stderr)
        return 1

    layer_stop = len(text_layers) if args.layer_stop is None else min(args.layer_stop, len(text_layers))
    if layer_stop <= args.layer_start:
        print("--layer-stop must be greater than --layer-start", file=sys.stderr)
        return 1

    compute_dtype = resolve_compute_dtype(args.compute_dtype, device, lm_head.weight.dtype)
    lm_head_weight = lm_head.weight.detach().to(device=device, dtype=compute_dtype)
    lm_head_bias = None
    if getattr(lm_head, "bias", None) is not None:
        lm_head_bias = lm_head.bias.detach().to(device=device, dtype=compute_dtype)

    hidden_size = int(lm_head_weight.shape[1])
    vocab_size = int(lm_head_weight.shape[0])
    selected_layers = list(range(args.layer_start, layer_stop))

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "output_jsonl": str(output_jsonl),
        "log_file": str(log_file),
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "lm_head_dtype_original": str(lm_head.weight.dtype),
        "local_files_only": not args.allow_network,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "top_k": int(args.top_k),
        "include_probs": bool(args.include_probs),
        "vector_batch_size": int(args.vector_batch_size),
        "layer_start": int(args.layer_start),
        "layer_stop": int(layer_stop),
        "max_vectors_per_layer": None if args.max_vectors_per_layer is None else int(args.max_vectors_per_layer),
        "num_text_layers_total": len(text_layers),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "layers_processed": [],
        "records_written": 0,
        "token_cache_size": 0,
    }

    logger.info("=" * 72)
    logger.info("SmolVLA Phase 1 Value-Vector Catalog Builder")
    logger.info("=" * 72)
    logger.info("Model ID:              %s", args.model_id)
    logger.info("Device:                %s", device)
    logger.info("Compute dtype:         %s", compute_dtype)
    logger.info("Layer range:           [%s, %s)", args.layer_start, layer_stop)
    logger.info("Top-k:                 %s", args.top_k)
    logger.info("Vector batch size:     %s", args.vector_batch_size)
    if args.max_vectors_per_layer is not None:
        logger.info("Max vectors/layer:     %s", args.max_vectors_per_layer)
    logger.info("Output JSONL:          %s", output_jsonl)
    logger.info("Summary JSON:          %s", summary_json)
    logger.info("Log file:              %s", log_file)

    records_written = 0
    layer_offsets: list[int] = []
    offset = 0
    for layer in text_layers:
        layer_offsets.append(offset)
        offset += int(layer.mlp.down_proj.in_features)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for layer_idx in selected_layers:
            layer = text_layers[layer_idx]
            down_proj = layer.mlp.down_proj
            layer_path = f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}.mlp.down_proj"
            weight = down_proj.weight.detach()
            vectors = weight.T.contiguous()
            intermediate_size = int(vectors.shape[0])
            if args.max_vectors_per_layer is not None:
                intermediate_size = min(intermediate_size, args.max_vectors_per_layer)
                vectors = vectors[:intermediate_size]

            layer_start_time = time.time()
            batch_ranges = batched_ranges(intermediate_size, args.vector_batch_size)
            logger.info(
                "Layer %02d | vectors=%d | batches=%d | path=%s",
                layer_idx,
                intermediate_size,
                len(batch_ranges),
                layer_path,
            )

            for batch_start, batch_stop in batch_ranges:
                batch_vectors = vectors[batch_start:batch_stop].to(device=device, dtype=compute_dtype)
                logits = F.linear(batch_vectors, lm_head_weight, lm_head_bias)
                top_logits_tensor, top_token_ids_tensor = torch.topk(logits, k=args.top_k, dim=-1)

                top_probs_tensor = None
                if args.include_probs:
                    probs = torch.softmax(logits, dim=-1)
                    top_probs_tensor = torch.gather(probs, dim=-1, index=top_token_ids_tensor)

                top_token_ids_cpu = top_token_ids_tensor.detach().cpu().tolist()
                top_logits_cpu = top_logits_tensor.detach().to(torch.float32).cpu().tolist()
                top_probs_cpu = (
                    top_probs_tensor.detach().to(torch.float32).cpu().tolist()
                    if top_probs_tensor is not None
                    else None
                )
                batch_vectors_cpu = batch_vectors.detach().to(torch.float32).cpu()

                for row_idx, token_ids in enumerate(top_token_ids_cpu):
                    vector_index = batch_start + row_idx
                    global_vector_index = layer_offsets[layer_idx] + vector_index
                    token_texts = [render_token(int(token_id)) for token_id in token_ids]
                    record = build_layer_record(
                        model_id=args.model_id,
                        layer_idx=layer_idx,
                        layer_path=layer_path,
                        vector_index=vector_index,
                        global_vector_index=global_vector_index,
                        vector_tensor=batch_vectors_cpu[row_idx],
                        token_ids=[int(token_id) for token_id in token_ids],
                        token_texts=token_texts,
                        top_logits=[float(value) for value in top_logits_cpu[row_idx]],
                        top_probs=(
                            [float(value) for value in top_probs_cpu[row_idx]]
                            if top_probs_cpu is not None
                            else None
                        ),
                        down_proj_weight_shape=[int(dim) for dim in down_proj.weight.shape],
                        hidden_size=hidden_size,
                        intermediate_size=int(down_proj.in_features),
                        top_k=args.top_k,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1
                handle.flush()

                elapsed = time.time() - layer_start_time
                processed = batch_stop
                logger.info(
                    "  processed %4d/%4d vectors in %5.1fs (records=%d)",
                    processed,
                    intermediate_size,
                    elapsed,
                    records_written,
                )

                del logits
                del top_logits_tensor
                del top_token_ids_tensor
                if top_probs_tensor is not None:
                    del top_probs_tensor

            summary["layers_processed"].append(
                {
                    "layer_idx": layer_idx,
                    "layer_path": layer_path,
                    "vectors_written": intermediate_size,
                    "down_proj_weight_shape": [int(dim) for dim in down_proj.weight.shape],
                    "elapsed_seconds": round(time.time() - layer_start_time, 3),
                }
            )

    elapsed_total = time.time() - wall_start
    summary["records_written"] = records_written
    summary["token_cache_size"] = len(token_cache)
    summary["elapsed_seconds"] = round(elapsed_total, 3)

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("-" * 72)
    logger.info("Records written:       %d", records_written)
    logger.info("Unique decoded tokens: %d", len(token_cache))
    logger.info("Elapsed seconds:       %.1f", elapsed_total)
    logger.info("Summary JSON:          %s", summary_json)
    logger.info("Log file:              %s", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
