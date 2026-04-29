#!/usr/bin/env python3
"""
Phase 3.5 steering-ready cluster candidate selection for SmolVLA.

This script sits between clustering and steering:

- reads the full cluster summary and cluster bundle,
- filters concept-ranked clusters by practical criteria such as size,
  cosine score, partition, and optional layer concentration,
- resolves each surviving cluster to explicit (layer_idx, vector_index) members,
- writes a compact JSON/Markdown report plus a tensor bundle that Phase 4 can
  consume directly for steering.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_SUMMARY_JSON = ROOT / "results" / "value_vector_clusters_top5.summary.json"
DEFAULT_CLUSTER_BUNDLE_PT = ROOT / "results" / "value_vector_clusters_top5.pt"
DEFAULT_EMBEDDINGS_PT = ROOT / "results" / "value_vector_semantic_embeddings_top5.pt"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "selected_cluster_candidates.json"
DEFAULT_OUTPUT_MD = ROOT / "results" / "selected_cluster_candidates.md"
DEFAULT_OUTPUT_PT = ROOT / "results" / "selected_cluster_candidates.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and export steering-ready concept clusters for SmolVLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cluster-summary-json",
        type=Path,
        default=DEFAULT_CLUSTER_SUMMARY_JSON,
        help="Structured cluster summary JSON from cluster_value_vectors.py.",
    )
    parser.add_argument(
        "--cluster-bundle-pt",
        type=Path,
        default=DEFAULT_CLUSTER_BUNDLE_PT,
        help="Tensor bundle from cluster_value_vectors.py.",
    )
    parser.add_argument(
        "--embeddings-pt",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PT,
        help="Semantic embedding bundle used to recover layer/vector metadata.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the structured selection summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Where to write the human-readable selection report.",
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
        help="Where to write the steering-ready candidate bundle.",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default="full,early,late",
        help="Comma-separated partitions to consider.",
    )
    parser.add_argument(
        "--preferred-partitions",
        type=str,
        default="late,full,early",
        help="Priority order used for one recommended candidate per concept.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="fast,slow,high,low,up,safe,risk",
        help="Comma-separated concepts to keep.",
    )
    parser.add_argument(
        "--top-k-per-concept",
        type=int,
        default=5,
        help="Maximum filtered candidates to keep for each (partition, concept).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=64,
        help="Minimum cluster size required to keep a candidate.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="Optional maximum cluster size.",
    )
    parser.add_argument(
        "--min-cosine-sim",
        type=float,
        default=0.0,
        help="Minimum concept-cluster cosine similarity.",
    )
    parser.add_argument(
        "--min-top-layer-fraction",
        type=float,
        default=0.0,
        help="Minimum fraction of members that must come from the dominant layer.",
    )
    parser.add_argument(
        "--top-layers-per-candidate",
        type=int,
        default=10,
        help="How many top layers to keep in each candidate summary.",
    )
    parser.add_argument(
        "--sample-members-per-candidate",
        type=int,
        default=10,
        help="How many sample members to keep in the JSON/Markdown reports.",
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


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_requested(requested: list[str], available: set[str], label: str) -> None:
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Available: {sorted(available)}")


def build_global_index_lookup(global_vector_index: torch.Tensor) -> dict[int, int]:
    return {
        int(global_idx): row_idx
        for row_idx, global_idx in enumerate(global_vector_index.tolist())
    }


def summarize_layers(layer_values: torch.Tensor, *, top_k: int) -> list[dict[str, int]]:
    counts = Counter(int(layer_idx) for layer_idx in layer_values.tolist())
    return [
        {"layer_idx": layer_idx, "count": count}
        for layer_idx, count in counts.most_common(top_k)
    ]


def make_candidate_id(partition: str, concept: str, cluster_id: int) -> str:
    return f"{partition}__{concept}__cluster_{cluster_id}"


def extract_candidate(
    *,
    partition_name: str,
    concept_name: str,
    ranking_item: dict[str, Any],
    cluster_partition_bundle: dict[str, Any],
    global_index_lookup: dict[int, int],
    embedding_layer_idx: torch.Tensor,
    embedding_vector_index: torch.Tensor,
    top_layers_per_candidate: int,
    sample_members_per_candidate: int,
) -> dict[str, Any]:
    cluster_id = int(ranking_item["cluster_id"])
    candidate_id = make_candidate_id(partition_name, concept_name, cluster_id)
    global_indices = cluster_partition_bundle[f"cluster_{cluster_id}_global_indices"].to(dtype=torch.int64)
    row_indices = torch.tensor(
        [global_index_lookup[int(global_idx)] for global_idx in global_indices.tolist()],
        dtype=torch.long,
    )
    member_layers = embedding_layer_idx[row_indices]
    member_vectors = embedding_vector_index[row_indices]
    top_layers = summarize_layers(member_layers, top_k=top_layers_per_candidate)
    cluster_size = int(global_indices.numel())
    top_layer_fraction = (top_layers[0]["count"] / cluster_size) if top_layers and cluster_size else 0.0
    sample_members = [
        {
            "layer_idx": int(member_layers[idx].item()),
            "vector_index": int(member_vectors[idx].item()),
            "global_vector_index": int(global_indices[idx].item()),
        }
        for idx in range(min(sample_members_per_candidate, cluster_size))
    ]
    return {
        "candidate_id": candidate_id,
        "partition_name": partition_name,
        "concept_name": concept_name,
        "cluster_id": cluster_id,
        "cosine_similarity": float(ranking_item["cosine_similarity"]),
        "cluster_size": cluster_size,
        "num_active_layers": int(member_layers.unique().numel()),
        "top_layer_fraction": float(top_layer_fraction),
        "top_layers": top_layers,
        "sample_members": sample_members,
        "global_vector_indices": global_indices,
        "layer_idx": member_layers.to(dtype=torch.int16),
        "vector_index": member_vectors.to(dtype=torch.int32),
    }


def passes_filters(
    candidate: dict[str, Any],
    *,
    min_cluster_size: int,
    max_cluster_size: int | None,
    min_cosine_sim: float,
    min_top_layer_fraction: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate["cluster_size"] < min_cluster_size:
        reasons.append("cluster_too_small")
    if max_cluster_size is not None and candidate["cluster_size"] > max_cluster_size:
        reasons.append("cluster_too_large")
    if candidate["cosine_similarity"] < min_cosine_sim:
        reasons.append("cosine_too_low")
    if candidate["top_layer_fraction"] < min_top_layer_fraction:
        reasons.append("layer_concentration_too_low")
    return (len(reasons) == 0), reasons


def build_recommended_candidates(
    selected: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    concepts: list[str],
    preferred_partitions: list[str],
) -> dict[str, dict[str, Any]]:
    recommended: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        chosen: dict[str, Any] | None = None
        for partition_name in preferred_partitions:
            partition_candidates = selected.get(partition_name, {}).get(concept, [])
            if partition_candidates:
                chosen = partition_candidates[0]
                break
        if chosen is not None:
            recommended[concept] = {
                key: value
                for key, value in chosen.items()
                if key not in {"global_vector_indices", "layer_idx", "vector_index"}
            }
    return recommended


def build_recommended_reuse(recommended: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    reuse: dict[str, list[str]] = defaultdict(list)
    for concept, candidate in recommended.items():
        key = f"{candidate['partition_name']}::cluster_{candidate['cluster_id']}"
        reuse[key].append(concept)
    return {key: concepts for key, concepts in reuse.items() if len(concepts) > 1}


def write_markdown_report(
    output_md: Path,
    *,
    summary: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Selected Cluster Candidates")
    lines.append("")
    lines.append(f"- Cluster summary: `{summary['cluster_summary_json']}`")
    lines.append(f"- Cluster bundle: `{summary['cluster_bundle_pt']}`")
    lines.append(f"- Embeddings bundle: `{summary['embeddings_pt']}`")
    lines.append(f"- Partitions: `{', '.join(summary['partitions'])}`")
    lines.append(f"- Concepts: `{', '.join(summary['concepts'])}`")
    lines.append(
        "- Filters: "
        f"`min_cluster_size={summary['selection_criteria']['min_cluster_size']}`, "
        f"`max_cluster_size={summary['selection_criteria']['max_cluster_size']}`, "
        f"`min_cosine_sim={summary['selection_criteria']['min_cosine_sim']}`, "
        f"`min_top_layer_fraction={summary['selection_criteria']['min_top_layer_fraction']}`"
    )
    lines.append("")

    lines.append("## Recommended")
    lines.append("")
    lines.append("| concept | partition | cluster | cosine | size | top-layer frac | top layers |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for concept in summary["concepts"]:
        candidate = summary["recommended_candidates"].get(concept)
        if candidate is None:
            lines.append(f"| {concept} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        top_layers = ", ".join(
            f"L{entry['layer_idx']} ({entry['count']})" for entry in candidate["top_layers"][:5]
        )
        lines.append(
            f"| {concept} | {candidate['partition_name']} | {candidate['cluster_id']} | "
            f"{candidate['cosine_similarity']:.4f} | {candidate['cluster_size']} | "
            f"{candidate['top_layer_fraction']:.3f} | {top_layers} |"
        )
    lines.append("")

    for partition_name in summary["partitions"]:
        lines.append(f"## {partition_name}")
        lines.append("")
        for concept in summary["concepts"]:
            concept_result = summary["selected_candidates"][partition_name][concept]
            lines.append(f"### {concept}")
            lines.append("")
            lines.append(
                f"- Selected: `{concept_result['num_selected']}` "
                f"(skipped: `{concept_result['num_skipped']}`)"
            )
            lines.append("")
            lines.append("| rank | cluster | cosine | size | top-layer frac | active layers | top layers |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for rank, candidate in enumerate(concept_result["candidates"], start=1):
                top_layers = ", ".join(
                    f"L{entry['layer_idx']} ({entry['count']})" for entry in candidate["top_layers"][:5]
                )
                lines.append(
                    f"| {rank} | {candidate['cluster_id']} | {candidate['cosine_similarity']:.4f} | "
                    f"{candidate['cluster_size']} | {candidate['top_layer_fraction']:.3f} | "
                    f"{candidate['num_active_layers']} | {top_layers} |"
                )
            lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cluster_summary_json = args.cluster_summary_json.resolve()
    cluster_bundle_pt = args.cluster_bundle_pt.resolve()
    embeddings_pt = args.embeddings_pt.resolve()
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_pt = args.output_pt.resolve()

    try:
        ensure_writable(output_json, args.overwrite)
        ensure_writable(output_md, args.overwrite)
        ensure_writable(output_pt, args.overwrite)
    except FileExistsError as exc:
        print(str(exc))
        return 1

    summary = json.loads(cluster_summary_json.read_text(encoding="utf-8"))
    cluster_bundle = torch.load(cluster_bundle_pt, map_location="cpu")
    embeddings_bundle = torch.load(embeddings_pt, map_location="cpu")

    available_partitions = {part["partition_name"] for part in summary["partitions"]}
    requested_partitions = parse_csv_arg(args.partitions)
    preferred_partitions = parse_csv_arg(args.preferred_partitions)
    validate_requested(requested_partitions, available_partitions, "partitions")
    validate_requested(preferred_partitions, available_partitions, "preferred partitions")

    available_concepts = set(summary["concept_aliases"].keys())
    requested_concepts = parse_csv_arg(args.concepts)
    validate_requested(requested_concepts, available_concepts, "concepts")

    partition_summary_map = {
        part["partition_name"]: part
        for part in summary["partitions"]
    }
    partition_bundle_map = cluster_bundle["partitions"]

    embedding_global_index = embeddings_bundle["global_vector_index"].to(dtype=torch.int64)
    embedding_layer_idx = embeddings_bundle["layer_idx"].to(dtype=torch.int64)
    embedding_vector_index = embeddings_bundle["vector_index"].to(dtype=torch.int64)
    global_index_lookup = build_global_index_lookup(embedding_global_index)

    selected_candidates: dict[str, dict[str, Any]] = {}
    steering_bundle_candidates: dict[str, Any] = {}

    for partition_name in requested_partitions:
        selected_candidates[partition_name] = {}
        partition_summary = partition_summary_map[partition_name]
        partition_bundle = partition_bundle_map[partition_name]
        concept_rankings = partition_summary["concept_rankings"]

        for concept_name in requested_concepts:
            kept: list[dict[str, Any]] = []
            skipped = Counter()
            for ranking_item in concept_rankings[concept_name]:
                candidate = extract_candidate(
                    partition_name=partition_name,
                    concept_name=concept_name,
                    ranking_item=ranking_item,
                    cluster_partition_bundle=partition_bundle,
                    global_index_lookup=global_index_lookup,
                    embedding_layer_idx=embedding_layer_idx,
                    embedding_vector_index=embedding_vector_index,
                    top_layers_per_candidate=args.top_layers_per_candidate,
                    sample_members_per_candidate=args.sample_members_per_candidate,
                )
                keep, reasons = passes_filters(
                    candidate,
                    min_cluster_size=args.min_cluster_size,
                    max_cluster_size=args.max_cluster_size,
                    min_cosine_sim=args.min_cosine_sim,
                    min_top_layer_fraction=args.min_top_layer_fraction,
                )
                if not keep:
                    for reason in reasons:
                        skipped[reason] += 1
                    continue

                steering_bundle_candidates[candidate["candidate_id"]] = {
                    "partition_name": candidate["partition_name"],
                    "concept_name": candidate["concept_name"],
                    "cluster_id": candidate["cluster_id"],
                    "cosine_similarity": candidate["cosine_similarity"],
                    "cluster_size": candidate["cluster_size"],
                    "num_active_layers": candidate["num_active_layers"],
                    "top_layer_fraction": candidate["top_layer_fraction"],
                    "top_layers": candidate["top_layers"],
                    "global_vector_indices": candidate["global_vector_indices"].clone(),
                    "layer_idx": candidate["layer_idx"].clone(),
                    "vector_index": candidate["vector_index"].clone(),
                }

                kept.append(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in {"global_vector_indices", "layer_idx", "vector_index"}
                    }
                )
                if len(kept) >= args.top_k_per_concept:
                    break

            selected_candidates[partition_name][concept_name] = {
                "num_selected": len(kept),
                "num_skipped": int(sum(skipped.values())),
                "skipped_reasons": dict(skipped),
                "candidates": kept,
            }

    recommended_candidates = build_recommended_candidates(
        {
            partition_name: {
                concept_name: concept_result["candidates"]
                for concept_name, concept_result in concept_results.items()
            }
            for partition_name, concept_results in selected_candidates.items()
        },
        concepts=requested_concepts,
        preferred_partitions=preferred_partitions,
    )
    recommended_reuse = build_recommended_reuse(recommended_candidates)

    output_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster_summary_json": str(cluster_summary_json),
        "cluster_bundle_pt": str(cluster_bundle_pt),
        "embeddings_pt": str(embeddings_pt),
        "output_pt": str(output_pt),
        "partitions": requested_partitions,
        "preferred_partitions": preferred_partitions,
        "concepts": requested_concepts,
        "selection_criteria": {
            "top_k_per_concept": args.top_k_per_concept,
            "min_cluster_size": args.min_cluster_size,
            "max_cluster_size": args.max_cluster_size,
            "min_cosine_sim": args.min_cosine_sim,
            "min_top_layer_fraction": args.min_top_layer_fraction,
            "top_layers_per_candidate": args.top_layers_per_candidate,
            "sample_members_per_candidate": args.sample_members_per_candidate,
        },
        "selected_candidates": selected_candidates,
        "recommended_candidates": recommended_candidates,
        "recommended_cluster_reuse": recommended_reuse,
    }
    output_json.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")
    write_markdown_report(output_md, summary=output_summary)

    output_bundle = {
        "generated_at_utc": output_summary["generated_at_utc"],
        "cluster_summary_json": str(cluster_summary_json),
        "cluster_bundle_pt": str(cluster_bundle_pt),
        "embeddings_pt": str(embeddings_pt),
        "partitions": requested_partitions,
        "preferred_partitions": preferred_partitions,
        "concepts": requested_concepts,
        "selection_criteria": output_summary["selection_criteria"],
        "recommended_candidates": recommended_candidates,
        "recommended_cluster_reuse": recommended_reuse,
        "candidates": steering_bundle_candidates,
    }
    torch.save(output_bundle, output_pt)

    print(f"Wrote selection summary JSON to {output_json}")
    print(f"Wrote selection summary Markdown to {output_md}")
    print(f"Wrote steering candidate bundle to {output_pt}")
    for concept_name, candidate in recommended_candidates.items():
        print(
            f"{concept_name:>5}: partition={candidate['partition_name']:<5} "
            f"cluster={candidate['cluster_id']:<4d} size={candidate['cluster_size']:<5d} "
            f"cos={candidate['cosine_similarity']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
