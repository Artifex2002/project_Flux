# SmolVLA Steering Summary

## Headline Findings

- At alpha=10, the risk cluster reduced mean displacement to 0.003435 versus 0.007345 for its matched random control.
- Risk transferred robustly across fixed init states with classification counts {'stable': 4}.
- Risk also remained stable under brightness and occlusion, with perturbed cluster-minus-random effects -0.004901 and -0.005000.
- Fast transferred more weakly: init-state counts {'collapse': 1, 'stable': 3} and occlusion effect -0.000393.

## Compact Results Table

| phase | concept | setting | random/baseline | cluster | cluster-random | classification | details |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase5_alpha_sweep | fast | alpha=2.5 | 0.007004 | 0.005964 | -0.001040 |  |  |
| phase5_alpha_sweep | fast | alpha=5.0 | 0.006721 | 0.005816 | -0.000905 |  |  |
| phase5_alpha_sweep | fast | alpha=10.0 | 0.006380 | 0.006081 | -0.000299 |  |  |
| phase5_alpha_sweep | risk | alpha=2.5 | 0.008028 | 0.005101 | -0.002927 |  |  |
| phase5_alpha_sweep | risk | alpha=5.0 | 0.007737 | 0.004148 | -0.003588 |  |  |
| phase5_alpha_sweep | risk | alpha=10.0 | 0.007345 | 0.003435 | -0.003911 |  |  |
| phase6_init_state_transfer | fast | clean | 0.006506 | 0.006033 | -0.000473 | collapse=1; stable=3 | none_mean=0.008390 |
| phase6_init_state_transfer | risk | clean | 0.007542 | 0.002753 | -0.004789 | stable=4 | none_mean=0.008390 |
| phase7_brightness | fast | brightness | 0.006588 | 0.006019 | -0.000569 | stable | flip=1; stable=3 |
| phase7_brightness | risk | brightness | 0.007656 | 0.002755 | -0.004901 | stable | stable=4 |
| phase7_occlusion | fast | occlusion | 0.006650 | 0.006257 | -0.000393 | stable | stable=4 |
| phase7_occlusion | risk | occlusion | 0.007844 | 0.002844 | -0.005000 | stable | stable=4 |
