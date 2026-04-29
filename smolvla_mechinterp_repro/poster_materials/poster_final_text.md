# Final Poster Text

## Title
Mechanistic Steering in SmolVLA Transfers Across Visual Contexts

## Subtitle
Testing whether internal steering clusters remain effective across scene changes and nuisance visual perturbations

## Column 1: Problem

### Why This Matters
- Vision-language-action models can solve robot tasks, but their internal control mechanisms are hard to interpret.
- If we can identify meaningful internal directions, we may be able to steer behavior without retraining the full policy.
- The key question is whether these interventions are truly semantic or only work in one narrow visual setting.

### Research Question
Under what kinds of visual variation does the effect of amplifying a steering cluster remain stable, weaken, flip, or collapse?

### Hypothesis
- Robust concept clusters should transfer across visual changes.
- Brittle or entangled clusters should weaken or disappear when the input changes.

### Setting
- Model: `HuggingFaceVLA/smolvla_libero`
- Domain: LIBERO task 0
- Policy type: vision-language model with continuous-action control
- Main metric: mean end-effector displacement

### Main Figure For This Column
- Use [`figure1_phase5_alpha_sweep.png`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure1_phase5_alpha_sweep.png)

### Figure Caption
Concept-aligned clusters change behavior more than matched random neuron sets. The strongest effect appears for `risk`.

---

## Column 2: Method

### Pipeline
1. Extract FFN value vectors from SmolVLA’s text transformer.
2. Decode vectors into token space with the LM head.
3. Build semantic embeddings and cluster the vectors.
4. Select candidate clusters such as `fast` and `risk`.
5. Intervene during rollout by overwriting FFN activations before `down_proj`.
6. Compare cluster steering to matched random controls in LIBERO.

### Important Mechanistic Detail
- We intervene with a `forward_pre_hook` on `mlp.down_proj`.
- This matches the paper’s FFN-activation intervention.
- We do **not** hook the `down_proj` output, because that would modify the already-mixed residual stream instead of individual FFN activations.

### Experiments

#### Phase 5: Causal Sanity Check
- Compare cluster steering vs matched random controls
- Sweep activation strength `alpha`
- Ask: do concept clusters change behavior more than random neuron sets?

#### Phase 6: Init-State Transfer
- Same task and instruction
- Different paired init states
- Ask: does the steering effect persist across fixed visual contexts?

#### Phase 7: Nuisance Perturbation Transfer
- Perturb only the primary camera image
- Tested:
  - brightness shift
  - occlusion patch
- Ask: does the steering effect survive visual corruption?

### Main Figure For This Column
- Use [`figure2_phase6_init_state_transfer.png`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure2_phase6_init_state_transfer.png)

### Figure Caption
`Risk` transfers across all four paired init-state contexts. `Fast` transfers too, but its effect is smaller and more context-sensitive.

---

## Column 3: Results

### Main Result
Some mechanistic steering directions in SmolVLA transfer across visual context changes.

### Phase 5: Mechanistic Signal Exists
- `Risk` cluster steering strongly outperforms matched random controls.
- At `alpha = 10`:
  - matched-random `risk`: `0.00735`
  - cluster `risk`: `0.00343`
- `Fast` also separates from matched random, but the effect is much smaller.

### Phase 6: Transfer Is Not All-Or-Nothing
- `Risk`: stable on all 4 paired init states
- `Fast`: stable on 3 of 4 init states, near-collapse on 1
- Interpretation:
  - some clusters are robust control directions
  - others are real but more fragile

### Phase 7: Visual Perturbations

#### Brightness Shift
- `Risk`: stable on all 4 paired init states
- `Fast`: stable overall, with one near-zero case crossing sign

#### Occlusion
- `Risk`: stable on all 4 paired init states
- `Fast`: stable, but weaker than in the clean setting

### Key Interpretation
- `Risk` looks like a robust internal control direction.
- `Fast` appears real, but less robust.
- Visual perturbations changed overall behavior only slightly, while the strongest steering effects remained.

### Important Note
- Our concept labels are semantic shorthand from the clustering stage.
- In this task, the `fast` cluster does **not** literally make the robot move faster; it reduces motion relative to its matched random control.
- So the key causal claim is about consistent behavioral influence, not about label semantics alone.

### Compact Summary

| Experiment | Fast | Risk |
| --- | --- | --- |
| Alpha sweep | weak but real | strong and monotonic |
| Init-state transfer | partial transfer | robust transfer |
| Brightness | stable overall | stable |
| Occlusion | stable but weaker | stable |

### Limitations
- Effects are clearest in motion statistics, not task success.
- Current experiments use one LIBERO task.
- We have not yet found the perturbation severity where transfer breaks.
- SmolVLA uses continuous actions, so this is an adaptation of the original paper’s setting.

### Next Steps
- Increase perturbation severity
- Test blur and noise
- Add multi-camera perturbations
- Expand to more tasks and longer rollouts
- Find the boundary where effects weaken, collapse, or flip

### Main Figures For This Column
- Use [`figure3_phase7_perturbation_transfer.png`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure3_phase7_perturbation_transfer.png)
- Use [`figure4_condition_means_across_settings.png`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/poster_materials/figures/figure4_condition_means_across_settings.png)

### Figure Captions
`figure3`: Cluster-minus-random effect under clean, brightness, and occlusion settings. `Risk` stays strong across all conditions. `Fast` remains present, but weaker.

`figure4`: Raw condition means across clean and perturbed settings. Perturbations shift overall behavior only slightly, while strong `risk` steering remains clearly separated from baseline and matched random.

---

## Takeaway Box
Mechanistic steering in SmolVLA is not purely scene-specific.  
`Risk` transfers robustly across init states, brightness, and occlusion.  
`Fast` transfers too, but is smaller and more fragile.

## 45-Second Pitch
We adapted a mechanistic steering pipeline to SmolVLA and tested whether internal steering clusters transfer across visual changes. We found that concept-aligned clusters, especially `risk`, change behavior more than matched random neuron sets. That effect remains stable across paired init states and nuisance perturbations like brightness shifts and occlusion. The main takeaway is that some internal steering directions in a vision-language-action model are genuinely portable across visual contexts, while others are weaker and more fragile.
