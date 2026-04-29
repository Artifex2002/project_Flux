# Canonical Report Assets

This note defines the **canonical** figures, tables, and text artifacts to use in the final report.

## Recommended Core Figures

### Figure 1: Alpha sweep
- File: [report_figure1_alpha_sweep.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/final_summary/plots/report_figure1_alpha_sweep.png)
- Why include it:
  - establishes that concept-aligned steering beats matched random controls
  - shows `risk` is stronger than `fast`
  - gives the first clean causal result

### Figure 2: Init-state transfer
- File: [report_figure2_init_state_transfer.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/final_summary/plots/report_figure2_init_state_transfer.png)
- Why include it:
  - directly answers the visual-context transfer question for fixed init states
  - highlights the `risk` vs `fast` contrast clearly

### Figure 3: Transfer stability across visual perturbations
- File: [report_figure3_transfer_stability.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/final_summary/plots/report_figure3_transfer_stability.png)
- Why include it:
  - summarizes clean vs brightness vs occlusion in one place
  - is the best single figure for the “does it transfer across visual changes?” claim

### Figure 4: Effect shift relative to clean
- File: [report_figure4_effect_shift.png](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/final_summary/plots/report_figure4_effect_shift.png)
- Why include it:
  - quantifies whether perturbations strengthen or weaken the steering effect
  - useful if the report has room for a fourth figure

## Recommended Core Tables

### Table 1: Compact main results table
- File: [smolvla_steering_summary.md](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/final_summary/smolvla_steering_summary.md)
- Section: `Compact Results Table`
- Why include it:
  - compactly captures Phase 5, 6, and 7
  - easy to cite exact numeric results

### Table 2: Transfer classification summary
- Source files:
  - [phase6 init-state transfer](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase6_tier1_fast_risk_initial/init_state_transfer_avg_displacement.json)
  - [brightness transfer](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase7_tier2_primary_brightness_initial_v2/visual_perturbation_transfer_avg_displacement.json)
  - [occlusion transfer](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase7_tier2_primary_occlusion_initial/visual_perturbation_transfer_avg_displacement.json)
- Why include it:
  - separates “effect size” from “stability classification”
  - supports claims like “risk was stable across all four contexts”

## Recommended Narrative Order

Use this order in the report:

1. Mechanistic setup and faithful adaptation
2. Phase 5 causal evidence
3. Phase 6 init-state transfer
4. Phase 7 visual perturbation transfer
5. SmolVLA-vs-paper divergence discussion
6. Limitations and next steps

## Minimum Report Set

If the report needs to stay compact, the minimum set I recommend is:

- Figure 1
- Figure 2
- Figure 3
- Table 1
- Divergence summary

## Optional Extras

- Figure 4 if there is room
- poster figures for presentation use
- deeper per-init-state tables if an appendix is allowed
