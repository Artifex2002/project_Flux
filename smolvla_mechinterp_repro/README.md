# SmolVLA Mechanistic Steering Reproduction

This folder is the isolated workspace for adapting the paper
_Mechanistic Interpretability for Steering Vision-Language-Action Models_
to `SmolVLA`.

We are keeping this separate from the original exploratory files:

- `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/baseline_eval.py`
- `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/value_vector_analysis/reading_weights.py`
- `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/value_vector_analysis/generate_random_weights.py`

## Goal

Build a clean, reproducible SmolVLA-specific pipeline for:

1. extracting FFN value vectors from the SmolVLA VLM backbone,
2. assigning semantics to those vectors via token projections,
3. selecting concept-aligned neuron sets or clusters,
4. steering SmolVLA at inference time with those neuron sets, and
5. evaluating whether steering changes robot behavior on LIBERO tasks.

## Important adaptation note

The paper's clearest simulation reproduction is on OPENVLA, which predicts action tokens directly.
SmolVLA is architecturally different: it uses a VLM backbone plus an action expert that outputs
continuous action chunks.

So our reproduction target is:

- preserve the paper's interpretability and steering logic as closely as possible,
- apply it to the SmolVLA VLM backbone,
- then measure whether those interventions causally affect the downstream action expert outputs.

## Concrete implementation plan

### Phase 0: Environment and architecture sanity checks

- [x] Verify `SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")` works in the intended env.
- [x] Enumerate the exact VLM text layers, FFN modules, hidden sizes, and candidate hook points.
- [x] Confirm whether the correct intervention point is pre-`down_proj`, post-`down_proj`, or gate projection.
- [x] Confirm whether SmolVLA's `lm_head` and tokenizer are compatible with value-vector decoding for all target layers.

Planned script:
- `scripts/inspect_smolvla_architecture.py`  [implemented]

Deliverable:
- a machine-readable architecture summary with layer names, tensor shapes, and safe hook targets.
- current artifact: `results/phase0_architecture_summary.json`

### Phase 1: Full value-vector extraction

- [x] Iterate over every FFN value vector in every target text layer, not just a random sample.
- [x] Decode each value vector into token space using the language modeling head.
- [x] Save top-k tokens, scores, layer id, neuron id, and source module path.
- [x] Decide whether to save logits, probabilities, or both.
- [x] Save outputs in structured form such as JSONL, Parquet, or CSV.

Planned script:
- `scripts/build_value_vector_catalog.py`  [implemented]

Deliverable:
- a full catalog of SmolVLA value vectors and their token projections.
- current full-run artifacts:
  `results/value_vector_catalog_top30.jsonl`
  `results/value_vector_catalog_top30.jsonl.summary.json`
  `results/value_vector_catalog_top30.jsonl.log`
- current smoke-test artifacts:
  `results/value_vector_catalog_smoketest.jsonl`
  `results/value_vector_catalog_smoketest.jsonl.summary.json`

### Phase 2: Paper-style semantic labeling and exploratory analysis

- [x] Reproduce the paper's "meaningful if 4 of top 30 tokens share a pattern" workflow in a lightweight way.
- [x] Add simple keyword search over decoded top tokens for concepts like `fast`, `slow`, `high`, `low`, `up`, `safe`, and `risk`.
- [x] Quantify how often semantically meaningful vectors appear by layer.
- [x] Check whether action-like tokens or structured control tokens appear in the decoded outputs.

Planned scripts:
- `scripts/analyze_value_vector_patterns.py`  [implemented]
- `scripts/find_keyword_clusters.py`  [implemented]

Deliverable:
- a short report describing which concepts are easiest to recover in SmolVLA and at which layers.
- current artifacts:
  `results/keyword_concept_candidates.json`
  `results/keyword_concept_candidates.md`
  `results/value_vector_pattern_summary.json`
  `results/value_vector_pattern_summary.md`

### Phase 3: Semantic embeddings and clustering

- [x] Implement the paper's semantic embedding approximation from top token projections.
- [x] Build one semantic embedding per value vector.
- [x] Cluster embeddings over the full model and over early/late layer partitions.
- [x] Support at least one practical clustering method that runs locally, even if we do not use cuML.
- [x] Rank clusters by cosine similarity to target concepts such as `up`.

Planned scripts:
- `scripts/build_semantic_embeddings.py`  [implemented]
- `scripts/cluster_value_vectors.py`  [implemented]
- `scripts/select_cluster_candidates.py`  [implemented]

Deliverable:
- reusable cluster files mapping concept labels to `(layer, neuron)` sets.
- current full-run artifacts:
  `results/value_vector_semantic_embeddings_top5.pt`
  `results/value_vector_semantic_embeddings_top5.pt.summary.json`
  `results/value_vector_clusters_top5.pt`
  `results/value_vector_clusters_top5.summary.json`
  `results/value_vector_clusters_top5.summary.md`
  `results/selected_cluster_candidates.json`
  `results/selected_cluster_candidates.md`
  `results/selected_cluster_candidates.pt`
- current smoke-test artifacts:
  `results/value_vector_semantic_embeddings_top5_smoketest.pt`
  `results/value_vector_semantic_embeddings_top5_smoketest.pt.summary.json`
  `results/value_vector_clusters_top5_smoketest.pt`
  `results/value_vector_clusters_top5_smoketest.summary.json`
  `results/value_vector_clusters_top5_smoketest.summary.md`

### Phase 4: Steering implementation

- [x] Replace the current random-injection baseline with a general steering engine.
- [x] Inject selected neuron sets at the correct FFN activation point.
- [x] Support layer-restricted steering: single-layer, early-only, late-only, and full-depth.
- [x] Support both concept-based clusters and random matched-size control clusters.
- [x] Add a debug mode that logs how often the intervention fires and on which tensors.

Planned script:
- `scripts/steer_smolvla_libero.py`  [implemented]

Deliverable:
- a reusable steering runner that can apply paper-style interventions to SmolVLA rollouts.

### Phase 5: Baselines and evaluation grid

- [x] Baseline 1: no intervention.
- [x] Baseline 2: prompt modification only.
- [x] Baseline 3: random matched-size neuron cluster.
- [x] Baseline 4: concept-aligned steering cluster.
- [x] Sweep activation coefficient `alpha`.
- [ ] Sweep cluster size.
- [ ] Run all 10 LIBERO-Long tasks with 10 rollouts each where feasible.

Planned scripts:
- `scripts/run_eval_grid.py`  [implemented]
- `scripts/eval_config.py`  [implemented]

Deliverable:
- a structured result table for every task, condition, cluster size, and activation strength.
- current core artifact:
  `results/eval_grids/phase5_alpha_sweep_late_core_initial/summary.json`

### Phase 6: Visual-context transfer experiments

- [x] Add explicit paired init-state selection for rollouts.
- [x] Record init-state indices in per-rollout result JSON files.
- [x] Add Tier 1 presets for fixed-task, varying-init-state experiments.
- [x] Add a transfer-analysis helper that compares cluster vs matched-random effects per init state.
- [x] Run paired init-state transfer experiments for `fast` and `risk`.
- [ ] Extend paired init-state transfer experiments to `safe` or a replacement control concept.
- [x] Classify per-context outcomes as stable, weaken, flip, or collapse.

Planned scripts:
- `scripts/eval_config.py`  [phase-6 presets added]
- `scripts/run_eval_grid.py`  [paired init-state support added]
- `scripts/analyze_init_state_transfer.py`  [implemented]

Deliverable:
- paired transfer tables showing whether cluster effects persist across fixed visual contexts.
- current core artifacts:
  `results/eval_grids/phase6_tier1_fast_risk_initial/summary.json`
  `results/eval_grids/phase6_tier1_fast_risk_initial/init_state_transfer_avg_displacement.json`
  `results/eval_grids/phase6_tier1_fast_risk_initial/init_state_transfer_avg_displacement.md`

### Phase 7: Nuisance-visual-perturbation transfer experiments

- [x] Add visual perturbation controls to the steering runner.
- [x] Thread visual perturbation settings through the eval grid and logged summaries.
- [x] Add smoke presets for clean-anchor, brightness-shift, and occlusion tests.
- [x] Add a helper to compare perturbed runs against a clean anchor.
- [x] Run paired clean-vs-perturbed transfer experiments for `risk` and `fast`.
- [x] Classify nuisance-perturbation outcomes as stable, weaken, flip, or collapse.

Planned scripts:
- `scripts/eval_config.py`  [phase-7 presets added]
- `scripts/run_eval_grid.py`  [vision perturbation support added]
- `scripts/analyze_visual_perturbation_transfer.py`  [implemented]

Deliverable:
- clean-vs-perturbed transfer tables showing which steering effects survive nuisance visual shifts.
- current core artifacts:
  `results/eval_grids/phase7_tier2_primary_brightness_initial_v2/summary.json`
  `results/eval_grids/phase7_tier2_primary_brightness_initial_v2/visual_perturbation_transfer_avg_displacement.json`
  `results/eval_grids/phase7_tier2_primary_brightness_initial_v2/visual_perturbation_transfer_avg_displacement.md`
  `results/eval_grids/phase7_tier2_primary_occlusion_initial/summary.json`
  `results/eval_grids/phase7_tier2_primary_occlusion_initial/visual_perturbation_transfer_avg_displacement.json`
  `results/eval_grids/phase7_tier2_primary_occlusion_initial/visual_perturbation_transfer_avg_displacement.md`

### Phase 8: Metrics, comparisons, and plots

- [ ] Recompute the current metrics: average end-effector displacement, max height, success rate.
- [ ] Add depth-localization comparisons: early vs late vs full.
- [ ] Add paired comparisons between fast and slow interventions where applicable.
- [ ] Add summary tables and plots for concept efficacy and baseline comparisons.
- [ ] Record any places where SmolVLA diverges from the paper due to architecture differences.

Planned scripts:
- `scripts/summarize_results.py`
- `scripts/plot_steering_results.py`

Deliverable:
- paper-style summary figures and a compact reproduction report.

## Recommended build order

- [x] 1. `inspect_smolvla_architecture.py`
- [x] 2. `build_value_vector_catalog.py`
- [x] 3. `find_keyword_clusters.py`
- [x] 4. `build_semantic_embeddings.py`
- [x] 5. `cluster_value_vectors.py`
- [x] 6. `steer_smolvla_libero.py`
- [x] 7. `run_eval_grid.py`
- [x] 8. `analyze_init_state_transfer.py`
- [x] 9. `analyze_visual_perturbation_transfer.py`
- [ ] 10. `summarize_results.py`

## Folder layout

```text
smolvla_mechinterp_repro/
├── README.md
├── configs/
├── results/
└── scripts/
```

## First milestone

The first milestone is modest and important:

- produce a full catalog of SmolVLA value vectors,
- identify a few plausible `fast` / `slow` / `up` / `high` candidates,
- verify the correct steering hook point with a small controlled rollout.

Once that is stable, the rest of the paper reproduction becomes much more mechanical.
