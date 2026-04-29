#!/usr/bin/env python3
"""
Phase 3 semantic embedding builder for SmolVLA value vectors.

Implements the paper's top-token semantic embedding approximation:

For each value vector, take the top-N projected tokens and logits, compute a
softmax over those logits, and form a weighted average of the corresponding
output token embeddings from the language modeling head.

By default this uses the top-5 tokens, matching the paper's appendix
description for semantic embedding construction.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from catalog_analysis_utils import iter_catalog_records
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "results" / "value_vector_catalog_top30.jsonl"
DEFAULT_OUTPUT_PT = ROOT / "results" / "value_vector_semantic_embeddings_top5.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-style semantic embeddings for SmolVLA value vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog-jsonl",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Input value-vector catalog JSONL.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceVLA/smolvla_libero",
        help="SmolVLA checkpoint used to fetch lm_head token embeddings.",
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
        help="Path for the tensor bundle output.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <output-pt>.summary.json.",
    )
    parser.add_argument(
        "--top-n-tokens",
        type=int,
        default=5,
        help="How many top tokens to use for the semantic embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="How many catalog records to process at once.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for the embedding computation.",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype for weighting and embedding aggregation.",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Stored dtype for the semantic embedding matrix.",
    )
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply L2 normalization to the final semantic embeddings before saving.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N processed records.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face downloads instead of forcing local cache only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def ensure_writable(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")


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


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


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


def count_records(path: Path, max_records: int | None) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            total += 1
            if max_records is not None and total >= max_records:
                break
    return total


def flush_batch(
    *,
    batch_records: list[dict[str, Any]],
    start_index: int,
    top_n_tokens: int,
    token_embedding_matrix: torch.Tensor,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    l2_normalize: bool,
    semantic_embeddings: torch.Tensor,
    layer_indices: torch.Tensor,
    vector_indices: torch.Tensor,
    global_vector_indices: torch.Tensor,
    token_ids_store: torch.Tensor,
    token_logits_store: torch.Tensor,
    token_weight_store: torch.Tensor,
) -> int:
    batch_size = len(batch_records)
    token_ids = torch.tensor(
        [record["top_token_ids"][:top_n_tokens] for record in batch_records],
        dtype=torch.long,
        device=token_embedding_matrix.device,
    )
    token_logits = torch.tensor(
        [record["top_token_logits"][:top_n_tokens] for record in batch_records],
        dtype=compute_dtype,
        device=token_embedding_matrix.device,
    )
    token_weights = torch.softmax(token_logits, dim=-1)
    token_embeddings = token_embedding_matrix[token_ids]
    weighted_embeddings = (token_weights.unsqueeze(-1) * token_embeddings).sum(dim=1)
    if l2_normalize:
        weighted_embeddings = F.normalize(weighted_embeddings, p=2, dim=-1)

    end_index = start_index + batch_size
    semantic_embeddings[start_index:end_index] = weighted_embeddings.to(dtype=output_dtype, device="cpu")
    layer_indices[start_index:end_index] = torch.tensor(
        [record["layer_idx"] for record in batch_records],
        dtype=torch.int16,
    )
    vector_indices[start_index:end_index] = torch.tensor(
        [record["vector_index"] for record in batch_records],
        dtype=torch.int32,
    )
    global_vector_indices[start_index:end_index] = torch.tensor(
        [record["global_vector_index"] for record in batch_records],
        dtype=torch.int32,
    )
    token_ids_store[start_index:end_index] = token_ids.to(dtype=torch.int32, device="cpu")
    token_logits_store[start_index:end_index] = token_logits.to(dtype=torch.float32, device="cpu")
    token_weight_store[start_index:end_index] = token_weights.to(dtype=torch.float32, device="cpu")
    return end_index


def main() -> int:
    args = parse_args()
    catalog_path = args.catalog_jsonl.resolve()
    output_pt = args.output_pt.resolve()
    summary_json = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else output_pt.with_suffix(output_pt.suffix + ".summary.json")
    )

    try:
        ensure_writable(output_pt, args.overwrite)
        ensure_writable(summary_json, args.overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    wall_start = time.time()
    device = select_device(args.device)
    compute_dtype = dtype_from_name(args.compute_dtype)
    output_dtype = dtype_from_name(args.output_dtype)

    total_records = count_records(catalog_path, args.max_records)
    print("=" * 72)
    print("SmolVLA Phase 3 Semantic Embedding Builder")
    print("=" * 72)
    print(f"Catalog:                {catalog_path}")
    print(f"Model ID:               {args.model_id}")
    print(f"Records to process:     {total_records}")
    print(f"Top-N tokens:           {args.top_n_tokens}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Device:                 {device}")
    print(f"Compute dtype:          {compute_dtype}")
    print(f"Output dtype:           {output_dtype}")
    print(f"L2 normalize:           {args.l2_normalize}")

    try:
        policy = load_policy(args.model_id, args.allow_network, device)
    except Exception as exc:
        print(f"Failed to load policy '{args.model_id}': {exc}", file=sys.stderr)
        return 1

    text_layers = policy.model.vlm_with_expert.vlm.model.text_model.layers
    num_layers = len(text_layers)
    lm_head_weight = policy.model.vlm_with_expert.vlm.lm_head.weight.detach().to(device=device, dtype=compute_dtype)
    embedding_dim = int(lm_head_weight.shape[1])

    semantic_embeddings = torch.empty((total_records, embedding_dim), dtype=output_dtype, device="cpu")
    layer_indices = torch.empty((total_records,), dtype=torch.int16, device="cpu")
    vector_indices = torch.empty((total_records,), dtype=torch.int32, device="cpu")
    global_vector_indices = torch.empty((total_records,), dtype=torch.int32, device="cpu")
    token_ids_store = torch.empty((total_records, args.top_n_tokens), dtype=torch.int32, device="cpu")
    token_logits_store = torch.empty((total_records, args.top_n_tokens), dtype=torch.float32, device="cpu")
    token_weight_store = torch.empty((total_records, args.top_n_tokens), dtype=torch.float32, device="cpu")

    batch_records: list[dict[str, Any]] = []
    processed = 0
    with torch.inference_mode():
        for record in iter_catalog_records(catalog_path):
            if args.max_records is not None and processed + len(batch_records) >= args.max_records:
                break
            if len(record["top_token_ids"]) < args.top_n_tokens or len(record["top_token_logits"]) < args.top_n_tokens:
                raise ValueError(
                    f"Record layer={record['layer_idx']} vector={record['vector_index']} "
                    f"does not have at least {args.top_n_tokens} tokens/logits."
                )
            batch_records.append(record)
            if len(batch_records) >= args.batch_size:
                processed = flush_batch(
                    batch_records=batch_records,
                    start_index=processed,
                    top_n_tokens=args.top_n_tokens,
                    token_embedding_matrix=lm_head_weight,
                    compute_dtype=compute_dtype,
                    output_dtype=output_dtype,
                    l2_normalize=args.l2_normalize,
                    semantic_embeddings=semantic_embeddings,
                    layer_indices=layer_indices,
                    vector_indices=vector_indices,
                    global_vector_indices=global_vector_indices,
                    token_ids_store=token_ids_store,
                    token_logits_store=token_logits_store,
                    token_weight_store=token_weight_store,
                )
                batch_records = []
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"processed {processed} / {total_records}", flush=True)

        if batch_records:
            processed = flush_batch(
                batch_records=batch_records,
                start_index=processed,
                top_n_tokens=args.top_n_tokens,
                token_embedding_matrix=lm_head_weight,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
                l2_normalize=args.l2_normalize,
                semantic_embeddings=semantic_embeddings,
                layer_indices=layer_indices,
                vector_indices=vector_indices,
                global_vector_indices=global_vector_indices,
                token_ids_store=token_ids_store,
                token_logits_store=token_logits_store,
                token_weight_store=token_weight_store,
            )

    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "catalog_jsonl": str(catalog_path),
        "top_n_tokens": args.top_n_tokens,
        "l2_normalize": args.l2_normalize,
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "output_dtype": str(output_dtype),
        "embedding_dim": embedding_dim,
        "num_records": processed,
        "num_layers": num_layers,
        "semantic_embeddings": semantic_embeddings[:processed],
        "layer_idx": layer_indices[:processed],
        "vector_index": vector_indices[:processed],
        "global_vector_index": global_vector_indices[:processed],
        "top_token_ids": token_ids_store[:processed],
        "top_token_logits": token_logits_store[:processed],
        "top_token_weights": token_weight_store[:processed],
    }
    torch.save(bundle, output_pt)

    elapsed_total = time.time() - wall_start
    summary = {
        "generated_at_utc": bundle["generated_at_utc"],
        "model_id": args.model_id,
        "catalog_jsonl": str(catalog_path),
        "output_pt": str(output_pt),
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "output_dtype": str(output_dtype),
        "top_n_tokens": args.top_n_tokens,
        "l2_normalize": args.l2_normalize,
        "max_records": args.max_records,
        "num_records": processed,
        "num_layers": num_layers,
        "embedding_dim": embedding_dim,
        "elapsed_seconds": round(elapsed_total, 3),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Saved semantic embeddings to: {output_pt}")
    print(f"Saved summary JSON to:        {summary_json}")
    print(f"Records written:              {processed}")
    print(f"Embedding dim:               {embedding_dim}")
    print(f"Elapsed seconds:             {elapsed_total:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
