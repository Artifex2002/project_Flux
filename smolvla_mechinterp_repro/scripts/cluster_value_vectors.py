#!/usr/bin/env python3
"""
Phase 3 clustering for SmolVLA value-vector semantic embeddings.

This script clusters the semantic embeddings built in Phase 3 and ranks the
resulting clusters against target concepts using cosine similarity.

The design goal is practical local analysis rather than an exact paper clone:
- cluster embeddings over the full model and early/late layer partitions,
- save cluster assignments in a reusable machine-readable bundle,
- rank clusters by similarity to concept anchors such as `up` or `fast`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if sys.platform == "darwin":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
except ImportError:
    KMeans = None
    MiniBatchKMeans = None

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS = ROOT / "results" / "value_vector_semantic_embeddings_top5.pt"
DEFAULT_OUTPUT_PT = ROOT / "results" / "value_vector_clusters_top5.pt"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "value_vector_clusters_top5.summary.json"
DEFAULT_OUTPUT_MD = ROOT / "results" / "value_vector_clusters_top5.summary.md"

DEFAULT_CONCEPTS: dict[str, list[str]] = {
    "fast": ["fast", "faster", "fastest", "quick", "quicker", "quickly", "swift"],
    "slow": ["slow", "slower", "slowest", "slowly", "calm", "gentle"],
    "high": ["high", "higher", "highest"],
    "low": ["low", "lower", "lowest"],
    "up": ["up", "upper", "upward", "upwards", "above"],
    "safe": ["safe", "safer", "safely", "safety"],
    "risk": ["risk", "risks", "risky"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Phase 3 semantic embeddings for SmolVLA value vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-pt",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Input tensor bundle from build_semantic_embeddings.py.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceVLA/smolvla_libero",
        help="SmolVLA checkpoint used to fetch tokenizer and lm_head embeddings.",
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
        help="Where to write the reusable clustering bundle.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the structured clustering summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Where to write the human-readable clustering report.",
    )
    parser.add_argument(
        "--partition-mode",
        type=str,
        default="both",
        choices=["full", "halves", "both"],
        help="Which layer partitions to cluster.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=128,
        help="Number of clusters for each partition.",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="minibatch-kmeans",
        choices=["kmeans", "minibatch-kmeans"],
        help="Clustering algorithm.",
    )
    parser.add_argument(
        "--normalize-before-clustering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize semantic embeddings before clustering.",
    )
    parser.add_argument(
        "--top-clusters-per-concept",
        type=int,
        default=10,
        help="How many top-ranked clusters to keep per concept.",
    )
    parser.add_argument(
        "--top-layers-per-cluster",
        type=int,
        default=5,
        help="How many dominant layers to show for each cluster summary.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=",".join(DEFAULT_CONCEPTS.keys()),
        help="Comma-separated concept names to score.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for clustering.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for model loading and concept-vector construction.",
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
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    return torch.device("cpu")


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


def resolve_concepts(concepts_arg: str) -> dict[str, list[str]]:
    requested = [item.strip() for item in concepts_arg.split(",") if item.strip()]
    unknown = [item for item in requested if item not in DEFAULT_CONCEPTS]
    if unknown:
        raise ValueError(f"Unknown concepts requested: {unknown}")
    return {name: DEFAULT_CONCEPTS[name] for name in requested}


def build_concept_vector(aliases: list[str], tokenizer, lm_head_weight: torch.Tensor) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for alias in aliases:
        text_variants = [alias, f" {alias}"]
        for text in text_variants:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            token_id_tensor = torch.tensor(token_ids, dtype=torch.long, device=lm_head_weight.device)
            pieces.append(lm_head_weight[token_id_tensor].mean(dim=0))
    if not pieces:
        raise ValueError(f"Could not build concept vector from aliases: {aliases}")
    concept_vector = torch.stack(pieces, dim=0).mean(dim=0)
    return F.normalize(concept_vector, p=2, dim=0)


def partition_indices(
    layer_idx: torch.Tensor,
    partition_mode: str,
    *,
    total_num_layers: int | None = None,
) -> dict[str, torch.Tensor]:
    num_layers = total_num_layers if total_num_layers is not None else int(layer_idx.max().item()) + 1
    midpoint = num_layers // 2
    full_mask = torch.ones_like(layer_idx, dtype=torch.bool)
    early_mask = layer_idx < midpoint
    late_mask = layer_idx >= midpoint

    partitions: dict[str, torch.Tensor] = {}
    if partition_mode in {"full", "both"}:
        partitions["full"] = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
    if partition_mode in {"halves", "both"}:
        partitions["early"] = torch.nonzero(early_mask, as_tuple=False).squeeze(-1)
        partitions["late"] = torch.nonzero(late_mask, as_tuple=False).squeeze(-1)
    return partitions


def cluster_embeddings(
    embeddings: torch.Tensor,
    *,
    method: str,
    num_clusters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = embeddings.to(dtype=torch.float32, device="cpu")
    effective_clusters = min(num_clusters, data.shape[0])
    if effective_clusters <= 0:
        raise ValueError("No embeddings available for clustering.")

    if KMeans is not None and MiniBatchKMeans is not None:
        numpy_data = data.numpy()
        if method == "kmeans":
            estimator = KMeans(
                n_clusters=effective_clusters,
                random_state=seed,
                n_init=10,
            )
        else:
            estimator = MiniBatchKMeans(
                n_clusters=effective_clusters,
                random_state=seed,
                batch_size=min(4096, max(256, effective_clusters * 4)),
                n_init=10,
            )
        if threadpool_limits is not None:
            with threadpool_limits(limits=1):
                fitted_labels = estimator.fit_predict(numpy_data)
        else:
            fitted_labels = estimator.fit_predict(numpy_data)
        labels = torch.from_numpy(fitted_labels).to(dtype=torch.int32)
        centers = torch.from_numpy(estimator.cluster_centers_).to(dtype=torch.float32)
        return labels, centers

    print("scikit-learn not available; falling back to torch k-means", flush=True)
    return torch_kmeans(data, num_clusters=effective_clusters, seed=seed)


def torch_kmeans(
    data: torch.Tensor,
    *,
    num_clusters: int,
    seed: int,
    max_iters: int = 40,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial_indices = torch.randperm(data.shape[0], generator=generator)[:num_clusters]
    centers = data[initial_indices].clone()

    for _ in range(max_iters):
        distances = torch.cdist(data, centers, p=2)
        labels = torch.argmin(distances, dim=1)

        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(labels, minlength=num_clusters)
        new_centers.index_add_(0, labels, data)

        non_empty = counts > 0
        if non_empty.any():
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty].unsqueeze(-1)
        if (~non_empty).any():
            refill_indices = torch.randperm(data.shape[0], generator=generator)[: int((~non_empty).sum().item())]
            new_centers[~non_empty] = data[refill_indices]

        shift = torch.linalg.norm(new_centers - centers, dim=1).max().item()
        centers = new_centers
        if shift < tol:
            break

    final_distances = torch.cdist(data, centers, p=2)
    final_labels = torch.argmin(final_distances, dim=1).to(dtype=torch.int32)
    return final_labels, centers.to(dtype=torch.float32)


def summarize_cluster_layers(
    layer_values: torch.Tensor,
    *,
    top_layers_per_cluster: int,
) -> list[dict[str, int]]:
    counts = Counter(int(layer) for layer in layer_values.tolist())
    top_layers = counts.most_common(top_layers_per_cluster)
    return [{"layer_idx": layer_idx, "count": count} for layer_idx, count in top_layers]


def partition_summary(
    *,
    partition_name: str,
    selected_indices: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    embeddings: torch.Tensor,
    layer_idx: torch.Tensor,
    vector_index: torch.Tensor,
    global_vector_index: torch.Tensor,
    concept_vectors: dict[str, torch.Tensor],
    top_clusters_per_concept: int,
    top_layers_per_cluster: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    partition_layers = layer_idx[selected_indices]
    partition_vectors = vector_index[selected_indices]
    partition_globals = global_vector_index[selected_indices]

    centers = F.normalize(centers, p=2, dim=-1)

    cluster_member_counts = Counter(int(label) for label in labels.tolist())
    cluster_summaries: list[dict[str, Any]] = []
    cluster_details: dict[str, torch.Tensor] = {}

    for cluster_id in sorted(cluster_member_counts):
        member_mask = labels == cluster_id
        member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
        member_global_indices = partition_globals[member_indices]
        member_layers = partition_layers[member_indices]
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "size": int(member_indices.numel()),
                "top_layers": summarize_cluster_layers(
                    member_layers,
                    top_layers_per_cluster=top_layers_per_cluster,
                ),
                "sample_members": [
                    {
                        "layer_idx": int(partition_layers[idx].item()),
                        "vector_index": int(partition_vectors[idx].item()),
                        "global_vector_index": int(partition_globals[idx].item()),
                    }
                    for idx in member_indices[:5].tolist()
                ],
            }
        )
        cluster_details[f"cluster_{cluster_id}_global_indices"] = member_global_indices.clone()

    concept_rankings: dict[str, list[dict[str, Any]]] = {}
    cluster_concept_scores: dict[str, torch.Tensor] = {}
    for concept, concept_vector in concept_vectors.items():
        scores = torch.matmul(centers, concept_vector.cpu())
        cluster_concept_scores[f"{concept}_scores"] = scores
        ranked = torch.argsort(scores, descending=True)
        concept_rankings[concept] = []
        for cluster_id in ranked[:top_clusters_per_concept].tolist():
            member_mask = labels == cluster_id
            member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
            concept_rankings[concept].append(
                {
                    "cluster_id": int(cluster_id),
                    "cosine_similarity": float(scores[cluster_id].item()),
                    "cluster_size": int(member_indices.numel()),
                    "top_layers": summarize_cluster_layers(
                        partition_layers[member_indices],
                        top_layers_per_cluster=top_layers_per_cluster,
                    ),
                    "sample_members": [
                        {
                            "layer_idx": int(partition_layers[idx].item()),
                            "vector_index": int(partition_vectors[idx].item()),
                            "global_vector_index": int(partition_globals[idx].item()),
                        }
                        for idx in member_indices[:5].tolist()
                    ],
                }
            )

    summary = {
        "partition_name": partition_name,
        "num_vectors": int(selected_indices.numel()),
        "num_clusters": len(cluster_member_counts),
        "cluster_summaries": cluster_summaries,
        "concept_rankings": concept_rankings,
    }
    details = {
        "selected_indices": selected_indices.clone().to(dtype=torch.int32),
        "labels": labels.clone(),
        "centers": centers.clone(),
        **cluster_details,
        **cluster_concept_scores,
    }
    return summary, details


def write_markdown_report(output_md: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Value Vector Cluster Summary")
    lines.append("")
    lines.append(f"- Embeddings bundle: `{summary['embeddings_pt']}`")
    lines.append(f"- Model ID: `{summary['model_id']}`")
    lines.append(f"- Cluster method: `{summary['cluster_method']}`")
    lines.append(f"- Normalize before clustering: `{summary['normalize_before_clustering']}`")
    lines.append("")

    for partition in summary["partitions"]:
        lines.append(f"## {partition['partition_name']}")
        lines.append("")
        lines.append(f"- Vectors: `{partition['num_vectors']}`")
        lines.append(f"- Clusters: `{partition['num_clusters']}`")
        lines.append("")
        for concept, rankings in partition["concept_rankings"].items():
            lines.append(f"### {concept}")
            lines.append("")
            lines.append("| rank | cluster | cosine | size | top layers |")
            lines.append("| --- | --- | --- | --- | --- |")
            for rank, item in enumerate(rankings[:5], start=1):
                top_layers = ", ".join(
                    f"L{entry['layer_idx']} ({entry['count']})" for entry in item["top_layers"]
                )
                lines.append(
                    f"| {rank} | {item['cluster_id']} | {item['cosine_similarity']:.4f} | "
                    f"{item['cluster_size']} | {top_layers} |"
                )
            lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    embeddings_pt = args.embeddings_pt.resolve()
    output_pt = args.output_pt.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()

    try:
        ensure_writable(output_pt, args.overwrite)
        ensure_writable(output_json, args.overwrite)
        ensure_writable(output_md, args.overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    wall_start = time.time()
    device = select_device(args.device)
    concept_aliases = resolve_concepts(args.concepts)

    bundle = torch.load(embeddings_pt, map_location="cpu")
    embeddings = bundle["semantic_embeddings"].to(dtype=torch.float32)
    layer_idx = bundle["layer_idx"].to(dtype=torch.int64)
    vector_index = bundle["vector_index"].to(dtype=torch.int64)
    global_vector_index = bundle["global_vector_index"].to(dtype=torch.int64)
    total_num_layers = int(bundle["num_layers"]) if "num_layers" in bundle else None

    if args.normalize_before_clustering:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    print("=" * 72)
    print("SmolVLA Phase 3 Clustering")
    print("=" * 72)
    print(f"Embeddings bundle:       {embeddings_pt}")
    print(f"Records:                 {embeddings.shape[0]}")
    print(f"Embedding dim:           {embeddings.shape[1]}")
    print(f"Partition mode:          {args.partition_mode}")
    print(f"Cluster method:          {args.cluster_method}")
    print(f"Clusters / partition:    {args.num_clusters}")
    print(f"Device for concept vecs: {device}")

    try:
        policy = load_policy(args.model_id, args.allow_network, device)
    except Exception as exc:
        print(f"Failed to load policy '{args.model_id}': {exc}", file=sys.stderr)
        return 1

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    lm_head_weight = policy.model.vlm_with_expert.vlm.lm_head.weight.detach().to(device=device, dtype=torch.float32)
    concept_vectors = {
        concept: build_concept_vector(aliases, tokenizer, lm_head_weight)
        for concept, aliases in concept_aliases.items()
    }

    partitions = partition_indices(
        layer_idx,
        args.partition_mode,
        total_num_layers=total_num_layers,
    )
    partition_summaries: list[dict[str, Any]] = []
    partition_details: dict[str, dict[str, torch.Tensor]] = {}

    for partition_name, selected_indices in partitions.items():
        if selected_indices.numel() == 0:
            print(f"skipping empty partition '{partition_name}'", flush=True)
            continue
        print(f"clustering partition '{partition_name}' with {selected_indices.numel()} vectors", flush=True)
        labels, centers = cluster_embeddings(
            embeddings[selected_indices],
            method=args.cluster_method,
            num_clusters=args.num_clusters,
            seed=args.seed,
        )
        summary, details = partition_summary(
            partition_name=partition_name,
            selected_indices=selected_indices,
            labels=labels,
            centers=centers,
            embeddings=embeddings,
            layer_idx=layer_idx,
            vector_index=vector_index,
            global_vector_index=global_vector_index,
            concept_vectors=concept_vectors,
            top_clusters_per_concept=args.top_clusters_per_concept,
            top_layers_per_cluster=args.top_layers_per_cluster,
        )
        partition_summaries.append(summary)
        partition_details[partition_name] = details

    if not partition_summaries:
        print("No non-empty partitions were available for clustering.", file=sys.stderr)
        return 1

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embeddings_pt": str(embeddings_pt),
        "model_id": args.model_id,
        "cluster_method": args.cluster_method,
        "normalize_before_clustering": args.normalize_before_clustering,
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "concept_aliases": concept_aliases,
        "partitions": partition_details,
    }
    torch.save(artifact, output_pt)

    elapsed_seconds = time.time() - wall_start
    summary = {
        "generated_at_utc": artifact["generated_at_utc"],
        "embeddings_pt": str(embeddings_pt),
        "output_pt": str(output_pt),
        "model_id": args.model_id,
        "cluster_method": args.cluster_method,
        "normalize_before_clustering": args.normalize_before_clustering,
        "num_clusters": args.num_clusters,
        "partition_mode": args.partition_mode,
        "top_clusters_per_concept": args.top_clusters_per_concept,
        "top_layers_per_cluster": args.top_layers_per_cluster,
        "seed": args.seed,
        "device": str(device),
        "num_records": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "concept_aliases": concept_aliases,
        "partitions": partition_summaries,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(output_md, summary)

    print("-" * 72)
    print(f"Saved clustering bundle to: {output_pt}")
    print(f"Saved summary JSON to:      {output_json}")
    print(f"Saved summary Markdown to:  {output_md}")
    print(f"Elapsed seconds:            {elapsed_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
