# Poster Draft

## Suggested Title
Mechanistic Steering of SmolVLA: Do Cluster Interventions Transfer Across Visual Contexts?

## One-Sentence Thesis
We adapt a mechanistic interpretability pipeline from language-model value vectors to the SmolVLA vision-language-action policy and show that some steering clusters, especially `risk`, transfer robustly across changing visual contexts and nuisance perturbations.

## Suggested 3-Column Layout

### Column 1: Problem and Motivation

#### Why this problem matters
- Vision-language-action models are powerful, but their internal control mechanisms are hard to interpret.
- If we can identify meaningful internal steering directions, we may be able to control robot behavior without retraining the full policy.
- A key open question is whether these steering directions are truly semantic, or whether they only work in one narrow visual setting.

#### Core research question
- Under what kinds of visual variation does the effect of amplifying a concept cluster remain stable, weaken, flip sign, or collapse?

#### Main hypothesis
- If a cluster is a genuine semantic control direction, its behavioral effect should transfer across different initial states and mild visual perturbations.
- If it is brittle or entangled with specific image features, the effect should weaken or disappear when the visual input changes.

#### Setting
- Policy: `HuggingFaceVLA/smolvla_libero`
- Benchmark: LIBERO task 0
- Model family: SmolVLA with a SmolVLM backbone and continuous-action policy head
- Behavioral metric used in this poster: mean end-effector displacement

#### Figure placement
- Put **Figure 1** here.
- File: [figure1_phase5_alpha_sweep.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure1_phase5_alpha_sweep.png)

#### Figure 1 caption
Phase 5 alpha sweep comparing concept-aligned clusters against matched random controls. `Risk` shows a strong and monotonic steering effect; `fast` is weaker but still separates from matched random.

---

### Column 2: Method

#### Method overview
1. Extract all FFN value vectors from SmolVLA’s text transformer.
2. Decode value vectors into token space using the language-model head.
3. Build semantic embeddings and cluster value vectors.
4. Select candidate clusters for concepts such as `fast` and `risk`.
5. Intervene at the correct FFN location with a `forward_pre_hook` on `mlp.down_proj`.
6. Evaluate the effect of steering on LIBERO rollouts.

#### Important technical detail
- The intervention is applied **before** `down_proj`, not on the `down_proj` output.
- This matters because the paper’s method overwrites FFN neuron activations, not the already-mixed residual output.

#### Experimental design

##### Phase 5: causal sanity check
- Compare cluster steering to matched random controls.
- Sweep activation strength `alpha`.
- Measure whether concept clusters change behavior more than random neuron sets.

##### Phase 6: visual-context transfer
- Fix the task and instruction.
- Vary only the init-state visual context.
- Evaluate whether the cluster-vs-random effect persists across paired visual contexts.

##### Phase 7: nuisance perturbation transfer
- Apply perturbations to the primary camera image.
- Tested perturbations so far:
  - brightness shift
  - occlusion patch
- Compare clean anchor runs against perturbed runs using the same init states and steering settings.

#### Metrics
- Mean end-effector displacement
- Max height
- Success rate

#### Figure placement
- Put **Figure 2** here.
- File: [figure2_phase6_init_state_transfer.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure2_phase6_init_state_transfer.png)

#### Figure 2 caption
Transfer across paired init-state contexts. `Risk` remains stable on all four fixed visual contexts, while `fast` transfers but is weaker and more context-sensitive.

---

### Column 3: Results and Takeaways

#### Key result 1: phase 5 found real mechanistic signal
- `Risk` cluster steering was much stronger than matched random controls.
- At `alpha = 10`:
  - matched-random `risk`: `0.00735`
  - cluster `risk`: `0.00343`
- `Fast` also separated from matched random, but the effect was much smaller.

#### Key result 2: transfer is not all-or-nothing
- In Phase 6, `risk` was stable on all 4 fixed init-state contexts.
- `Fast` was stable on 3 out of 4 init states and near-collapse on 1.
- This suggests some clusters are robust control directions, while others are more context-sensitive.

#### Key result 3: visual perturbations did not break `risk`
- Under brightness shift:
  - `risk` remained stable on all 4 paired init states.
  - `fast` remained stable overall, though one near-zero case crossed sign.
- Under occlusion:
  - `risk` remained stable on all 4 paired init states.
  - `fast` also remained stable, but its effect weakened.

#### Compact summary table

| Experiment | Fast | Risk |
| --- | --- | --- |
| Phase 5 alpha sweep | weak but real | strong and monotonic |
| Phase 6 init-state transfer | partial transfer | robust transfer |
| Phase 7 brightness | stable overall | stable |
| Phase 7 occlusion | stable but weaker | stable, slightly stronger |

#### What we think this means
- `Risk` looks like a robust internal control direction in SmolVLA.
- `Fast` appears real but more fragile.
- At least some mechanistic interventions transfer across visual changes instead of only working in one exact scene.

#### Limitations
- We measure motion statistics more clearly than task success.
- All current experiments are on one LIBERO task.
- The perturbations used so far are still moderate, so we have not yet found the failure boundary.
- SmolVLA outputs continuous actions, so this is an adaptation of the paper rather than a literal one-to-one reproduction.

#### Next steps
- Sweep stronger occlusion severities and multi-camera perturbations.
- Test blur/noise to separate information loss from corruption.
- Expand to more tasks and longer rollouts.
- Find the boundary where cluster effects weaken, collapse, or flip.

#### Figure placement
- Put **Figure 3** and **Figure 4** here.
- Files:
  - [figure3_phase7_perturbation_transfer.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure3_phase7_perturbation_transfer.png)
  - [figure4_condition_means_across_settings.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure4_condition_means_across_settings.png)

#### Figure 3 caption
Cluster-minus-random effect under clean, brightness, and occlusion settings. `Risk` stays strong across all conditions; `fast` remains present but is smaller and more fragile.

#### Figure 4 caption
Raw condition means across clean and perturbed settings. Perturbations shift overall behavior only slightly, while the strongest `risk` steering effect remains clearly separated from baseline and matched random.

## Suggested 60-Second Spoken Pitch
- We adapted a mechanistic steering method to SmolVLA and asked whether internal steering directions transfer across visual changes.
- First, we found concept clusters by decoding and clustering FFN value vectors, then intervened at the correct FFN hook point during LIBERO rollouts.
- We found that `risk` is a strong and robust steering direction: it beats matched random controls and remains stable across changing init states, brightness shifts, and occlusion.
- `Fast` also transfers, but it is weaker and more context-sensitive.
- Our current conclusion is that some mechanistic interventions in a VLA are genuinely portable across visual contexts, but not all clusters are equally robust.

## Short Takeaway Box
Some internal steering directions in SmolVLA transfer across visual changes. `Risk` is robust; `fast` is weaker and more fragile.
