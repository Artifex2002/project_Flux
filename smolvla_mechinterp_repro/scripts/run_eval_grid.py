#!/usr/bin/env python3
"""
Phase 5 logged evaluation-grid runner for SmolVLA steering.

Runs steer_smolvla_libero.py repeatedly under a structured grid, while writing:

- a frozen grid config,
- a run plan,
- per-run stdout logs,
- per-run result JSONs,
- a manifest JSONL with status and metadata,
- aggregate summary JSON / CSV / Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_config import PRESET_CONFIGS, get_preset_config


ROOT = Path(__file__).resolve().parents[1]
STEERING_SCRIPT = ROOT / "scripts" / "steer_smolvla_libero.py"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "eval_grids"


@dataclass
class RunSpec:
    run_id: str
    task_idx: int
    condition: str
    concept: str | None
    prompt_prefix: str | None
    alpha: float | None
    max_neurons: int | None
    layer_scope: str | None
    cluster_partition: str | None
    random_match_mode: str | None
    init_state_indices: list[int] | None
    vision_perturbation: str | None
    vision_target: str | None
    vision_strength: float | None
    vision_seed: int | None
    num_rollouts: int
    max_steps: int
    log_every: int
    seed: int
    output_json: str
    log_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a logged Phase 5 evaluation grid for SmolVLA steering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="phase5_smoke_fast",
        choices=sorted(PRESET_CONFIGS),
        help="Named preset from eval_config.py.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run directory name. Defaults to preset + timestamp.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing all eval-grid runs.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Override benchmark suite.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task indices override.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Comma-separated steering concepts override.",
    )
    parser.add_argument(
        "--prompt-concepts",
        type=str,
        default=None,
        help="Comma-separated prompt baseline concepts override.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated conditions override. Any of: none,prompt,random_matched,cluster",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values override.",
    )
    parser.add_argument(
        "--max-neurons",
        type=str,
        default=None,
        help="Comma-separated max-neuron caps override. Use 'none' for full cluster.",
    )
    parser.add_argument(
        "--layer-scopes",
        type=str,
        default=None,
        help="Comma-separated layer scopes override.",
    )
    parser.add_argument(
        "--cluster-partition",
        type=str,
        default=None,
        help="Override cluster partition argument passed to steer_smolvla_libero.py.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=None,
        help="Override rollouts per run.",
    )
    parser.add_argument(
        "--init-state-indices",
        type=str,
        default=None,
        help="Comma-separated init-state indices override. Length must match num-rollouts.",
    )
    parser.add_argument(
        "--vision-perturbation",
        type=str,
        default=None,
        help="Optional visual perturbation override passed to steer_smolvla_libero.py.",
    )
    parser.add_argument(
        "--vision-target",
        type=str,
        default=None,
        help="Optional camera target override for visual perturbations.",
    )
    parser.add_argument(
        "--vision-strength",
        type=float,
        default=None,
        help="Optional perturbation strength override.",
    )
    parser.add_argument(
        "--vision-seed",
        type=int,
        default=None,
        help="Optional perturbation seed override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps per rollout.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Override progress log frequency.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override base seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device argument passed to steer_smolvla_libero.py.",
    )
    parser.add_argument(
        "--candidate-bundle-pt",
        type=Path,
        default=None,
        help="Optional override for selected cluster candidate bundle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the plan but do not execute subprocesses.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the grid immediately if any run fails.",
    )
    return parser.parse_args()


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str | None) -> list[int] | None:
    items = parse_csv_list(value)
    if items is None:
        return None
    return [int(item) for item in items]


def parse_float_csv(value: str | None) -> list[float] | None:
    items = parse_csv_list(value)
    if items is None:
        return None
    return [float(item) for item in items]


def parse_optional_int_csv(value: str | None) -> list[int | None] | None:
    items = parse_csv_list(value)
    if items is None:
        return None
    result: list[int | None] = []
    for item in items:
        if item.lower() in {"none", "full", "all"}:
            result.append(None)
        else:
            result.append(int(item))
    return result


def slugify_optional(value: str | None) -> str:
    if value is None:
        return "none"
    return value.replace("/", "_").replace(" ", "_")


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = get_preset_config(args.preset)
    if args.suite is not None:
        config["suite"] = args.suite
    if args.tasks is not None:
        config["task_indices"] = parse_int_csv(args.tasks)
    if args.concepts is not None:
        config["concepts"] = parse_csv_list(args.concepts)
    if args.prompt_concepts is not None:
        config["prompt_concepts"] = parse_csv_list(args.prompt_concepts)
    if args.conditions is not None:
        config["conditions"] = parse_csv_list(args.conditions)
    if args.alphas is not None:
        config["alphas"] = parse_float_csv(args.alphas)
    if args.max_neurons is not None:
        config["max_neurons"] = parse_optional_int_csv(args.max_neurons)
    if args.layer_scopes is not None:
        config["layer_scopes"] = parse_csv_list(args.layer_scopes)
    if args.cluster_partition is not None:
        config["cluster_partition"] = args.cluster_partition
    if args.num_rollouts is not None:
        config["num_rollouts"] = args.num_rollouts
    if args.init_state_indices is not None:
        config["init_state_indices"] = parse_int_csv(args.init_state_indices)
    if args.vision_perturbation is not None:
        config["vision_perturbation"] = args.vision_perturbation
    if args.vision_target is not None:
        config["vision_target"] = args.vision_target
    if args.vision_strength is not None:
        config["vision_strength"] = args.vision_strength
    if args.vision_seed is not None:
        config["vision_seed"] = args.vision_seed
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.log_every is not None:
        config["log_every"] = args.log_every
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.candidate_bundle_pt is not None:
        config["candidate_bundle_pt"] = str(args.candidate_bundle_pt.resolve())
    return config


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{args.preset}_{timestamp}"


def build_run_specs(config: dict[str, Any], run_dir: Path) -> list[RunSpec]:
    runs_dir = run_dir / "runs"
    logs_dir = run_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    specs: list[RunSpec] = []
    counter = 0

    def next_run_id() -> str:
        nonlocal counter
        value = f"{counter:03d}"
        counter += 1
        return value

    def build_paths(run_id: str, label: str) -> tuple[str, str]:
        slug = label.replace("/", "_")
        output_json = runs_dir / f"{run_id}_{slug}.json"
        log_file = logs_dir / f"{run_id}_{slug}.log"
        return str(output_json), str(log_file)

    for task_idx in config["task_indices"]:
        if "none" in config["conditions"]:
            run_id = next_run_id()
            output_json, log_file = build_paths(run_id, f"task{task_idx}_none")
            specs.append(
                RunSpec(
                    run_id=run_id,
                    task_idx=task_idx,
                    condition="none",
                    concept=None,
                    prompt_prefix=None,
                    alpha=None,
                    max_neurons=None,
                    layer_scope=None,
                    cluster_partition=None,
                    random_match_mode=None,
                    init_state_indices=config.get("init_state_indices"),
                    vision_perturbation=config.get("vision_perturbation"),
                    vision_target=config.get("vision_target"),
                    vision_strength=config.get("vision_strength"),
                    vision_seed=config.get("vision_seed"),
                    num_rollouts=config["num_rollouts"],
                    max_steps=config["max_steps"],
                    log_every=config["log_every"],
                    seed=config["seed"],
                    output_json=output_json,
                    log_file=log_file,
                )
            )

        if "prompt" in config["conditions"]:
            for concept in config["prompt_concepts"]:
                run_id = next_run_id()
                output_json, log_file = build_paths(run_id, f"task{task_idx}_prompt_{concept}")
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        task_idx=task_idx,
                        condition="prompt",
                        concept=concept,
                        prompt_prefix=concept,
                        alpha=None,
                        max_neurons=None,
                        layer_scope=None,
                        cluster_partition=None,
                        random_match_mode=None,
                        init_state_indices=config.get("init_state_indices"),
                        vision_perturbation=config.get("vision_perturbation"),
                        vision_target=config.get("vision_target"),
                        vision_strength=config.get("vision_strength"),
                        vision_seed=config.get("vision_seed"),
                        num_rollouts=config["num_rollouts"],
                        max_steps=config["max_steps"],
                        log_every=config["log_every"],
                        seed=config["seed"],
                        output_json=output_json,
                        log_file=log_file,
                    )
                )

        for concept in config["concepts"]:
            for alpha in config["alphas"]:
                for max_neurons in config["max_neurons"]:
                    for layer_scope in config["layer_scopes"]:
                        if "random_matched" in config["conditions"]:
                            run_id = next_run_id()
                            size_slug = "full" if max_neurons is None else f"n{max_neurons}"
                            output_json, log_file = build_paths(
                                run_id,
                                f"task{task_idx}_random_{concept}_a{alpha:g}_{layer_scope}_{size_slug}",
                            )
                            specs.append(
                                RunSpec(
                                    run_id=run_id,
                                    task_idx=task_idx,
                                    condition="random_matched",
                                    concept=concept,
                                    prompt_prefix=None,
                                    alpha=alpha,
                                    max_neurons=max_neurons,
                                    layer_scope=layer_scope,
                                    cluster_partition=config["cluster_partition"],
                                    random_match_mode=config["random_match_mode"],
                                    init_state_indices=config.get("init_state_indices"),
                                    vision_perturbation=config.get("vision_perturbation"),
                                    vision_target=config.get("vision_target"),
                                    vision_strength=config.get("vision_strength"),
                                    vision_seed=config.get("vision_seed"),
                                    num_rollouts=config["num_rollouts"],
                                    max_steps=config["max_steps"],
                                    log_every=config["log_every"],
                                    seed=config["seed"],
                                    output_json=output_json,
                                    log_file=log_file,
                                )
                            )
                        if "cluster" in config["conditions"]:
                            run_id = next_run_id()
                            size_slug = "full" if max_neurons is None else f"n{max_neurons}"
                            output_json, log_file = build_paths(
                                run_id,
                                f"task{task_idx}_cluster_{concept}_a{alpha:g}_{layer_scope}_{size_slug}",
                            )
                            specs.append(
                                RunSpec(
                                    run_id=run_id,
                                    task_idx=task_idx,
                                    condition="cluster",
                                    concept=concept,
                                    prompt_prefix=None,
                                    alpha=alpha,
                                    max_neurons=max_neurons,
                                    layer_scope=layer_scope,
                                    cluster_partition=config["cluster_partition"],
                                    random_match_mode=config["random_match_mode"],
                                    init_state_indices=config.get("init_state_indices"),
                                    vision_perturbation=config.get("vision_perturbation"),
                                    vision_target=config.get("vision_target"),
                                    vision_strength=config.get("vision_strength"),
                                    vision_seed=config.get("vision_seed"),
                                    num_rollouts=config["num_rollouts"],
                                    max_steps=config["max_steps"],
                                    log_every=config["log_every"],
                                    seed=config["seed"],
                                    output_json=output_json,
                                    log_file=log_file,
                                )
                            )
    return specs


def build_command(spec: RunSpec, config: dict[str, Any]) -> list[str]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "project_Lumina",
        "python",
        str(STEERING_SCRIPT),
        "--suite",
        config["suite"],
        "--task-idx",
        str(spec.task_idx),
        "--num-rollouts",
        str(spec.num_rollouts),
        "--max-steps",
        str(spec.max_steps),
        "--log-every",
        str(spec.log_every),
        "--seed",
        str(spec.seed),
        "--device",
        config["device"],
        "--condition",
        spec.condition,
        "--output-json",
        spec.output_json,
    ]
    if "candidate_bundle_pt" in config:
        cmd.extend(["--candidate-bundle-pt", config["candidate_bundle_pt"]])
    if spec.prompt_prefix is not None:
        cmd.extend(["--prompt-prefix", spec.prompt_prefix])
    if spec.concept is not None:
        cmd.extend(["--concept", spec.concept])
    if spec.alpha is not None:
        cmd.extend(["--alpha", str(spec.alpha)])
    if spec.layer_scope is not None:
        cmd.extend(["--layer-scope", spec.layer_scope])
    if spec.cluster_partition is not None:
        cmd.extend(["--partition", spec.cluster_partition])
    if spec.random_match_mode is not None:
        cmd.extend(["--random-match-mode", spec.random_match_mode])
    if spec.init_state_indices is not None:
        cmd.extend(["--init-state-indices", ",".join(str(idx) for idx in spec.init_state_indices)])
    if spec.vision_perturbation is not None:
        cmd.extend(["--vision-perturbation", spec.vision_perturbation])
    if spec.vision_target is not None:
        cmd.extend(["--vision-target", spec.vision_target])
    if spec.vision_strength is not None:
        cmd.extend(["--vision-strength", str(spec.vision_strength)])
    if spec.vision_seed is not None:
        cmd.extend(["--vision-seed", str(spec.vision_seed)])
    if spec.max_neurons is not None:
        cmd.extend(["--max-neurons", str(spec.max_neurons)])
    return cmd


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def stream_subprocess(cmd: list[str], *, env: dict[str, str], log_file: Path) -> int:
    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def flatten_result(spec: RunSpec, result_json: dict[str, Any]) -> dict[str, Any]:
    summary = result_json.get("summary", {})
    steering = result_json.get("steering") or {}
    return {
        "run_id": spec.run_id,
        "task_idx": spec.task_idx,
        "condition": spec.condition,
        "concept": spec.concept,
        "prompt_prefix": spec.prompt_prefix,
        "alpha": spec.alpha,
        "max_neurons": spec.max_neurons,
        "layer_scope": spec.layer_scope,
        "cluster_partition": spec.cluster_partition,
        "init_state_indices": spec.init_state_indices,
        "vision_perturbation": spec.vision_perturbation,
        "vision_target": spec.vision_target,
        "vision_strength": spec.vision_strength,
        "vision_seed": spec.vision_seed,
        "output_json": spec.output_json,
        "avg_speed_mean": summary.get("avg_speed_mean"),
        "avg_speed_std": summary.get("avg_speed_std"),
        "max_height_mean": summary.get("max_height_mean"),
        "max_height_std": summary.get("max_height_std"),
        "success_rate": summary.get("success_rate"),
        "success_count": summary.get("success_count"),
        "steering_label": steering.get("label"),
        "steering_candidate_id": steering.get("candidate_id"),
        "num_neurons": steering.get("num_neurons"),
        "num_layers_touched": steering.get("num_layers_touched"),
        "elapsed_seconds": result_json.get("elapsed_seconds"),
    }


def write_summary_files(run_dir: Path, aggregate_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> None:
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    summary_md = run_dir / "summary.md"

    write_json(summary_json, {"runs": aggregate_rows, "manifest": manifest_rows})

    fieldnames = [
        "run_id",
        "task_idx",
        "condition",
        "concept",
        "prompt_prefix",
        "alpha",
        "max_neurons",
        "layer_scope",
        "cluster_partition",
        "init_state_indices",
        "vision_perturbation",
        "vision_target",
        "vision_strength",
        "vision_seed",
        "avg_speed_mean",
        "max_height_mean",
        "success_rate",
        "success_count",
        "steering_label",
        "num_neurons",
        "num_layers_touched",
        "elapsed_seconds",
        "output_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    lines = [
        "# Eval Grid Summary",
        "",
        f"- Successful runs: `{len(aggregate_rows)}`",
        "",
        "| run | task | condition | concept | alpha | perturbation | target | max neurons | success rate | avg speed | max height | steering |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['run_id']} | {row['task_idx']} | {row['condition']} | {row['concept'] or ''} | "
            f"{row['alpha'] if row['alpha'] is not None else ''} | "
            f"{row['vision_perturbation'] or ''} | "
            f"{row['vision_target'] or ''} | "
            f"{row['max_neurons'] if row['max_neurons'] is not None else 'full'} | "
            f"{row['success_rate']:.3f} | {row['avg_speed_mean']:.4f} | {row['max_height_mean']:.4f} | "
            f"{row['steering_label'] or ''} |"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = resolve_config(args)
    run_name = make_run_name(args)
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.jsonl"
    config_path = run_dir / "config.json"
    plan_path = run_dir / "run_plan.json"

    specs = build_run_specs(config, run_dir)

    frozen_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "run_name": run_name,
        "steering_script": str(STEERING_SCRIPT),
        "config": config,
    }
    write_json(config_path, frozen_config)
    write_json(plan_path, {"runs": [asdict(spec) for spec in specs]})

    print("=" * 72)
    print("SmolVLA Eval Grid")
    print("=" * 72)
    print(f"Run dir : {run_dir}")
    print(f"Preset  : {args.preset}")
    print(f"Runs    : {len(specs)}")
    print(f"Dry run : {args.dry_run}")

    aggregate_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    child_env = os.environ.copy()
    child_env.setdefault("MPLCONFIGDIR", "/tmp")
    child_env.setdefault("HF_HUB_OFFLINE", "1")
    child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    if args.dry_run:
        print("Dry run only; wrote config and plan files.")
        return 0

    for spec in specs:
        cmd = build_command(spec, config)
        print("\n" + "-" * 72)
        print(f"Run {spec.run_id} | task={spec.task_idx} condition={spec.condition} concept={spec.concept}")
        print("Command:")
        print(" ".join(cmd))

        started_at = datetime.now(timezone.utc).isoformat()
        wall_start = time.time()
        exit_code = stream_subprocess(cmd, env=child_env, log_file=Path(spec.log_file))
        elapsed = time.time() - wall_start
        finished_at = datetime.now(timezone.utc).isoformat()

        manifest_entry = {
            "run_id": spec.run_id,
            "task_idx": spec.task_idx,
            "condition": spec.condition,
            "concept": spec.concept,
            "alpha": spec.alpha,
            "max_neurons": spec.max_neurons,
            "layer_scope": spec.layer_scope,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": round(elapsed, 3),
            "exit_code": exit_code,
            "output_json": spec.output_json,
            "log_file": spec.log_file,
            "status": "success" if exit_code == 0 else "failed",
        }

        if exit_code == 0 and Path(spec.output_json).exists():
            result_json = json.loads(Path(spec.output_json).read_text(encoding="utf-8"))
            flat = flatten_result(spec, result_json)
            aggregate_rows.append(flat)
            manifest_entry["summary"] = {
                "success_rate": flat["success_rate"],
                "avg_speed_mean": flat["avg_speed_mean"],
                "max_height_mean": flat["max_height_mean"],
            }
        else:
            manifest_entry["summary"] = None

        append_jsonl(manifest_path, manifest_entry)
        manifest_rows.append(manifest_entry)
        write_summary_files(run_dir, aggregate_rows, manifest_rows)

        if exit_code != 0 and args.stop_on_error:
            print("Stopping on first error.")
            return exit_code

    print("\nCompleted eval grid.")
    print(f"Results written under: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
