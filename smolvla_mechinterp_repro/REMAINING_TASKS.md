# Remaining Tasks Checklist

This checklist reflects the **current actual project state**, not just the older planning boxes in `README.md`.

## Must Do

These are the items that matter most for finishing the project well.

### Reporting and final synthesis

- [ ] Update [`README.md`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/README.md) so completed Phase 5, 6, and 7 items match the experiments we actually ran.
- [ ] Create `scripts/summarize_results.py` to aggregate key results across Phase 5, 6, and 7.
- [ ] Create `scripts/plot_steering_results.py` for final report figures beyond the poster figures.
- [ ] Produce one final summary table covering:
  - clean baseline
  - matched-random steering
  - cluster steering
  - init-state transfer
  - brightness transfer
  - occlusion transfer
- [ ] Write a compact “SmolVLA vs paper” divergence summary:
  - continuous-action policy instead of action-token generation
  - intervention applied in the text backbone of a VLM+expert policy
  - motion-statistic effects clearer than task-success effects

### Final experiment interpretation

- [ ] Write down the final takeaways for each concept:
  - `risk`
  - `fast`
  - `safe` or replacement control
- [ ] Decide which result is the primary headline claim for the final report.
- [ ] Decide which plots/tables will be the canonical ones in the report.

## Should Do

These would materially strengthen the project if we have time.

### Phase 7 boundary-finding

- [ ] Run a stronger occlusion severity sweep:
  - `0.2`
  - `0.35`
  - `0.5`
- [ ] Run a both-camera occlusion experiment using the same paired init states.
- [ ] Run one additional nuisance perturbation family:
  - blur, or
  - Gaussian noise
- [ ] Identify the first perturbation setting where:
  - `fast` weakens substantially, collapses, or flips
  - `risk` weakens substantially, collapses, or flips

### Phase 5 evaluation breadth

- [ ] Add a cluster-size sweep for at least the strongest concept(s), especially `risk`.
- [ ] Run a longer-horizon comparison for the strongest clean condition(s).
- [ ] Test at least 1 to 2 additional LIBERO tasks beyond task 0, if runtime allows.

### Control concepts

- [ ] Run a paired transfer experiment for `safe`, or explicitly replace it with a better negative/control concept.
- [ ] Document clearly whether the control concept is:
  - semantically weak,
  - cluster-entangled, or
  - behaviorally uninformative

## Nice To Have

These are valuable, but not necessary for a strong finish.

### Additional analysis

- [ ] Add depth-localization comparisons:
  - early only
  - late only
  - full candidate
- [ ] Compare whether steering strength correlates with:
  - cluster size
  - layer concentration
  - concept-anchor similarity
- [ ] Add per-init-state visualizations for the best and worst transfer cases.

### Broader experiments

- [ ] Try another concept beyond `fast` and `risk` if a clean candidate emerges.
- [ ] Test camera ablations:
  - primary only
  - wrist only
  - both
- [ ] Explore whether stronger perturbations change task success, not just motion statistics.

## Suggested Execution Order

### Track A: finishable core

1. [ ] Update the README/checklist to match reality.
2. [ ] Build `summarize_results.py`.
3. [ ] Build `plot_steering_results.py`.
4. [ ] Produce final summary tables and report figures.
5. [ ] Write the “SmolVLA vs paper” divergence section.

### Track B: strongest next science result

1. [ ] Run stronger occlusion severity sweep.
2. [ ] Run both-camera occlusion.
3. [ ] Run blur or noise perturbation.
4. [ ] Identify where transfer first weakens or breaks.

### Track C: optional breadth

1. [ ] Run cluster-size sweep.
2. [ ] Add extra LIBERO task(s).
3. [ ] Add control-concept transfer experiment.

## Current Status Summary

- [x] Phase 0 architecture sanity check
- [x] Phase 1 full value-vector extraction
- [x] Phase 2 semantic keyword/pattern analysis
- [x] Phase 3 semantic embeddings and clustering
- [x] Phase 3.5 steering-ready cluster selection
- [x] Phase 4 paper-aligned steering implementation
- [x] Phase 5 alpha sweep and baseline comparison
- [x] Phase 6 paired init-state transfer for `fast` and `risk`
- [x] Phase 7 brightness transfer experiment
- [x] Phase 7 occlusion transfer experiment
- [ ] Phase 8 final synthesis and report packaging

## Recommended Immediate Next Step

- [ ] Start Phase 8 by implementing `scripts/summarize_results.py` and updating the README to match the completed work.
