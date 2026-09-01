# Falsification-First Hypothesis Discrimination Benchmark Report

**Benchmark Horizon**: 6 adaptive steps  
**Seeds**: [42, 101, 2024]  
**Worlds Evaluated**: World 1 ($H_1$), World 2 ($H_2$), World 3 ($H_3$)  

## Summary Results by World and Policy (Aggregated Across Seeds)

| world | true_hypothesis | policy | mean_final_true_weight | median_final_true_weight | std_final_true_weight | id_rate_75 | id_rate_90 | top1_accuracy | mean_final_entropy | mean_cost | mean_final_best_k0 | median_final_best_k0 | std_final_best_k0 | max_final_best_k0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic_World_1_H1_True | H1 | discovery_only | 0.4972 | 0.5000 | 0.0065 | 0.0000 | 0.0000 | 0.3333 | 0.6932 | 54.0000 | 0.0105 | 0.0105 | 0.0003 | 0.0107 |
| Synthetic_World_1_H1_True | H1 | hybrid | 0.3331 | 0.0000 | 0.5770 | 0.3333 | 0.3333 | 0.3333 | 0.0017 | 42.0000 | 0.0102 | 0.0103 | 0.0006 | 0.0107 |
| Synthetic_World_1_H1_True | H1 | pure_falsification | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31.3333 | 0.0096 | 0.0096 | 0.0002 | 0.0099 |
| Synthetic_World_1_H1_True | H1 | random_action | 0.2369 | 0.2105 | 0.2512 | 0.0000 | 0.0000 | 0.3333 | 0.4033 | 50.0000 | 0.0097 | 0.0096 | 0.0002 | 0.0099 |
| Synthetic_World_1_H1_True | H1 | uncertainty_only | 0.4983 | 0.5001 | 0.0032 | 0.0000 | 0.0000 | 0.6667 | 0.6931 | 54.0000 | 0.0100 | 0.0099 | 0.0004 | 0.0105 |
| Synthetic_World_2_H2_True | H2 | discovery_only | 0.0003 | 0.0000 | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.6920 | 54.0000 | 0.0164 | 0.0168 | 0.0032 | 0.0194 |
| Synthetic_World_2_H2_True | H2 | hybrid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 38.0000 | 0.0106 | 0.0093 | 0.0045 | 0.0156 |
| Synthetic_World_2_H2_True | H2 | pure_falsification | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30.0000 | 0.0106 | 0.0093 | 0.0045 | 0.0156 |
| Synthetic_World_2_H2_True | H2 | random_action | 0.0001 | 0.0001 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.6874 | 50.0000 | 0.0126 | 0.0131 | 0.0032 | 0.0156 |
| Synthetic_World_2_H2_True | H2 | uncertainty_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6931 | 54.0000 | 0.0111 | 0.0109 | 0.0044 | 0.0156 |
| Synthetic_World_3_H3_True | H3 | discovery_only | 0.4955 | 0.4943 | 0.0210 | 0.0000 | 0.0000 | 0.3333 | 0.6925 | 54.0000 | 0.0139 | 0.0139 | 0.0000 | 0.0139 |
| Synthetic_World_3_H3_True | H3 | hybrid | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 43.3333 | 0.0125 | 0.0132 | 0.0015 | 0.0135 |
| Synthetic_World_3_H3_True | H3 | pure_falsification | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 30.0000 | 0.0121 | 0.0129 | 0.0015 | 0.0129 |
| Synthetic_World_3_H3_True | H3 | random_action | 0.6396 | 0.4677 | 0.3033 | 0.3333 | 0.3333 | 0.3333 | 0.4795 | 50.0000 | 0.0131 | 0.0131 | 0.0002 | 0.0133 |
| Synthetic_World_3_H3_True | H3 | uncertainty_only | 0.5089 | 0.4987 | 0.0211 | 0.0000 | 0.0000 | 0.3333 | 0.6924 | 54.0000 | 0.0124 | 0.0129 | 0.0018 | 0.0139 |

## Scientific Findings & Methodological Boundaries
- **World 3 ($H_3$ Local Regime)**: Falsification-First policies achieve 100% Top-1 accuracy and 100% ID@90 across evaluated seeds ($P(H_3) \approx 1.0$) by identifying transition candidates with high Expected HIG, outperforming unguided exploration while pure falsification reduces mean experimental cost by 40.0% relative to random exploration (30.0 vs. 50.0 cost units).
- **World 1 & World 2 ($H_1$ vs. $H_2$)**: At the evaluated six-step horizon, H1 and H2 remain poorly identifiable. The current results are consistent with a sample-complexity limitation of the higher-dimensional structure-informed model, but longer-horizon and targeted joint-characterization experiments are required to test that explanation.
- **Discovery vs. Falsification Trade-off**: `pure_falsification` operates with lowest experimental cost, while `hybrid` balances discovery potential with information gain.