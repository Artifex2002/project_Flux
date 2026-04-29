# Visual Perturbation Transfer Summary

- Metric: `avg_displacement`
- Anchor run: `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase6_tier1_fast_risk_initial`
- Perturbed run: `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase7_tier2_primary_occlusion_initial`
- Anchor vision: `{'perturbations': ['none'], 'targets': ['both'], 'strengths': []}`
- Perturbed vision: `{'perturbations': ['occlusion'], 'targets': ['primary'], 'strengths': [0.2]}`

| concept | alpha | paired init states | anchor effect | perturbed effect | shift | overall | stable | weaken | collapse | flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast | 10.0 | 4 | -0.000473 | -0.000393 | 0.000079 | stable | 4 | 0 | 0 | 0 |
| risk | 10.0 | 4 | -0.004789 | -0.005000 | -0.000211 | stable | 4 | 0 | 0 | 0 |
