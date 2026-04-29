# Visual Perturbation Transfer Summary

- Metric: `avg_displacement`
- Anchor run: `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase6_tier1_fast_risk_initial`
- Perturbed run: `/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/smolvla_mechinterp_repro/results/eval_grids/phase7_tier2_primary_brightness_initial_v2`
- Anchor vision: `{'perturbations': ['none'], 'targets': ['both'], 'strengths': []}`
- Perturbed vision: `{'perturbations': ['brightness'], 'targets': ['primary'], 'strengths': [0.15]}`

| concept | alpha | paired init states | anchor effect | perturbed effect | shift | overall | stable | weaken | collapse | flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast | 10.0 | 4 | -0.000473 | -0.000569 | -0.000096 | stable | 3 | 0 | 0 | 1 |
| risk | 10.0 | 4 | -0.004789 | -0.004901 | -0.000112 | stable | 4 | 0 | 0 | 0 |
