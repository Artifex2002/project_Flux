#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_baselines.sh — Run all 3 baseline conditions for LIBERO-Long task 0
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   chmod +x run_baselines.sh
#   ./run_baselines.sh
#
# Each condition writes results to baseline_results/
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # exit on first error

echo "========================================"
echo " Project Lumina — Baseline Evaluation"
echo "========================================"
echo ""

# ── Condition 1: No steering (baseline) ──────────────────────────────────────
echo ">>> [1/3] Running: condition=none"
conda run -n project_Lumina python baseline_eval.py \
    --condition none \
    --task_idx 0 \
    --num_rollouts 10
echo ""

# ── Condition 2: Prompt-based fast steering ──────────────────────────────────
echo ">>> [2/3] Running: condition=prompt_fast"
conda run -n project_Lumina python baseline_eval.py \
    --condition prompt_fast \
    --task_idx 0 \
    --num_rollouts 10
echo ""

# ── Condition 3: Random neuron injection ─────────────────────────────────────
echo ">>> [3/3] Running: condition=random_injection"
conda run -n project_Lumina python baseline_eval.py \
    --condition random_injection \
    --task_idx 0 \
    --num_rollouts 10
echo ""

echo "========================================"
echo " All baselines complete!"
echo " Results in: baseline_results/"
echo "========================================"
