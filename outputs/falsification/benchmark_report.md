# Falsification-First Hypothesis Discrimination Benchmark Report

**Benchmark Horizon**: 6 adaptive steps  
**Seeds**: [42, 101, 2024]  
**Worlds Evaluated**: World 1 ($H_1$), World 2 ($H_2$), World 3 ($H_3$)  

## Summary Results by World and Policy

| world | true_hypothesis | policy | final_true_weight | final_entropy | best_k0 | mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| Synthetic_World_1_H1_True | H1 | discovery_only | 0.1491 | 0.4212 | 0.0107 | 42.0000 |
| Synthetic_World_1_H1_True | H1 | hybrid | 0.0000 | 0.0000 | 0.0107 | 38.0000 |
| Synthetic_World_1_H1_True | H1 | pure_falsification | 0.0000 | 0.0000 | 0.0102 | 18.0000 |
| Synthetic_World_1_H1_True | H1 | random_action | 0.9983 | 0.0125 | 0.0107 | 38.0000 |
| Synthetic_World_1_H1_True | H1 | uncertainty_only | 0.0001 | 0.0007 | 0.0102 | 42.0000 |
| Synthetic_World_2_H2_True | H2 | discovery_only | 0.0141 | 0.0741 | 0.0220 | 42.0000 |
| Synthetic_World_2_H2_True | H2 | hybrid | 0.0000 | 0.0000 | 0.0220 | 38.0000 |
| Synthetic_World_2_H2_True | H2 | pure_falsification | 0.0000 | 0.0000 | 0.0188 | 18.0000 |
| Synthetic_World_2_H2_True | H2 | random_action | 0.0000 | 0.3694 | 0.0188 | 38.0000 |
| Synthetic_World_2_H2_True | H2 | uncertainty_only | 0.0000 | 0.0069 | 0.0188 | 42.0000 |
| Synthetic_World_3_H3_True | H3 | discovery_only | 0.8104 | 0.4856 | 0.0108 | 42.0000 |
| Synthetic_World_3_H3_True | H3 | hybrid | 1.0000 | 0.0000 | 0.0139 | 38.0000 |
| Synthetic_World_3_H3_True | H3 | pure_falsification | 1.0000 | 0.0000 | 0.0105 | 18.0000 |
| Synthetic_World_3_H3_True | H3 | random_action | 0.0355 | 0.1535 | 0.0134 | 38.0000 |
| Synthetic_World_3_H3_True | H3 | uncertainty_only | 1.0000 | 0.0000 | 0.0139 | 42.0000 |

## Scientific Interpretation
- **Pure Falsification (HIG)** preferentially selects experiments maximizing hypothesis entropy reduction.
- **Hybrid Policy** balances hypothesis discrimination with property discovery.
- **Discovery Only (BoTorch BO)** finds high property values rapidly but allocates zero budget to structural characterization.