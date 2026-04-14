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

- [ ] Iterate over every FFN value vector in every target text layer, not just a random sample.
- [ ] Decode each value vector into token space using the language modeling head.
- [ ] Save top-k tokens, scores, layer id, neuron id, and source module path.
- [ ] Decide whether to save logits, probabilities, or both.
- [ ] Save outputs in structured form such as JSONL, Parquet, or CSV.

Planned script:
- `scripts/build_value_vector_catalog.py`  [implemented]

Deliverable:
- a full catalog of SmolVLA value vectors and their token projections.
- current smoke-test artifacts:
  `results/value_vector_catalog_smoketest.jsonl`
  `results/value_vector_catalog_smoketest.jsonl.summary.json`

### Phase 2: Paper-style semantic labeling and exploratory analysis

- [ ] Reproduce the paper's "meaningful if 4 of top 30 tokens share a pattern" workflow in a lightweight way.
- [ ] Add simple keyword search over decoded top tokens for concepts like `fast`, `slow`, `high`, `low`, `up`, `safe`, and `risk`.
- [ ] Quantify how often semantically meaningful vectors appear by layer.
- [ ] Check whether action-like tokens or structured control tokens appear in the decoded outputs.

Planned scripts:
- `scripts/analyze_value_vector_patterns.py`
- `scripts/find_keyword_clusters.py`

Deliverable:
- a short report describing which concepts are easiest to recover in SmolVLA and at which layers.

### Phase 3: Semantic embeddings and clustering

- [ ] Implement the paper's semantic embedding approximation from top token projections.
- [ ] Build one semantic embedding per value vector.
- [ ] Cluster embeddings over the full model and over early/late layer partitions.
- [ ] Support at least one practical clustering method that runs locally, even if we do not use cuML.
- [ ] Rank clusters by cosine similarity to target concepts such as `up`.

Planned scripts:
- `scripts/build_semantic_embeddings.py`
- `scripts/cluster_value_vectors.py`

Deliverable:
- reusable cluster files mapping concept labels to `(layer, neuron)` sets.

### Phase 4: Steering implementation

- [ ] Replace the current random-injection baseline with a general steering engine.
- [ ] Inject selected neuron sets at the correct FFN activation point.
- [ ] Support layer-restricted steering: single-layer, early-only, late-only, and full-depth.
- [ ] Support both concept-based clusters and random matched-size control clusters.
- [ ] Add a debug mode that logs how often the intervention fires and on which tensors.

Planned script:
- `scripts/steer_smolvla_libero.py`

Deliverable:
- a reusable steering runner that can apply paper-style interventions to SmolVLA rollouts.

### Phase 5: Baselines and evaluation grid

- [ ] Baseline 1: no intervention.
- [ ] Baseline 2: prompt modification only.
- [ ] Baseline 3: random matched-size neuron cluster.
- [ ] Baseline 4: concept-aligned steering cluster.
- [ ] Sweep activation coefficient `alpha`.
- [ ] Sweep cluster size.
- [ ] Run all 10 LIBERO-Long tasks with 10 rollouts each where feasible.

Planned scripts:
- `scripts/run_eval_grid.py`
- `scripts/eval_config.py`

Deliverable:
- a structured result table for every task, condition, cluster size, and activation strength.

### Phase 6: Metrics, comparisons, and plots

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
- [ ] 2. `build_value_vector_catalog.py`
- [ ] 3. `find_keyword_clusters.py`
- [ ] 4. `build_semantic_embeddings.py`
- [ ] 5. `cluster_value_vectors.py`
- [ ] 6. `steer_smolvla_libero.py`
- [ ] 7. `run_eval_grid.py`
- [ ] 8. `summarize_results.py`

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
