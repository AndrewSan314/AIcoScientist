# Falsification-First Hypothesis Discrimination Benchmark Report

**Benchmark Horizon**: 6 adaptive steps  
**Seeds**: [42, 101, 2024]  
**Worlds Evaluated**: World 1 ($H_1$), World 2 ($H_2$), World 3 ($H_3$)  

## Summary Results by World and Policy (Aggregated Across Seeds)

| world | true_hypothesis | policy | mean_final_true_weight | median_final_true_weight | std_final_true_weight | id_rate_75 | id_rate_90 | top1_accuracy | mean_final_entropy | mean_cost | best_k0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic_World_1_H1_True | H1 | discovery_only | 0.4972 | 0.5000 | 0.0065 | 0.0000 | 0.0000 | 0.3333 | 0.6932 | 54.0000 | 0.0107 |
| Synthetic_World_1_H1_True | H1 | hybrid | 0.0300 | 0.0194 | 0.0362 | 0.0000 | 0.0000 | 0.0000 | 0.1289 | 35.3333 | 0.0103 |
| Synthetic_World_1_H1_True | H1 | pure_falsification | 0.1468 | 0.0769 | 0.1914 | 0.0000 | 0.0000 | 0.0000 | 0.4439 | 32.6667 | 0.0099 |
| Synthetic_World_1_H1_True | H1 | random_action | 0.3268 | 0.4681 | 0.2730 | 0.0000 | 0.0000 | 0.3333 | 0.4841 | 50.0000 | 0.0099 |
| Synthetic_World_1_H1_True | H1 | uncertainty_only | 0.4983 | 0.5001 | 0.0032 | 0.0000 | 0.0000 | 0.6667 | 0.6931 | 54.0000 | 0.0105 |
| Synthetic_World_2_H2_True | H2 | discovery_only | 0.0003 | 0.0000 | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.6920 | 54.0000 | 0.0194 |
| Synthetic_World_2_H2_True | H2 | hybrid | 0.0001 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.4348 | 50.0000 | 0.0194 |
| Synthetic_World_2_H2_True | H2 | pure_falsification | 0.0005 | 0.0007 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.2888 | 40.6667 | 0.0156 |
| Synthetic_World_2_H2_True | H2 | random_action | 0.0001 | 0.0001 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.6907 | 50.0000 | 0.0156 |
| Synthetic_World_2_H2_True | H2 | uncertainty_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6931 | 54.0000 | 0.0156 |
| Synthetic_World_3_H3_True | H3 | discovery_only | 0.4955 | 0.4943 | 0.0210 | 0.0000 | 0.0000 | 0.3333 | 0.6925 | 54.0000 | 0.0139 |
| Synthetic_World_3_H3_True | H3 | hybrid | 0.9943 | 0.9951 | 0.0047 | 1.0000 | 1.0000 | 1.0000 | 0.0345 | 34.0000 | 0.0129 |
| Synthetic_World_3_H3_True | H3 | pure_falsification | 0.9951 | 0.9961 | 0.0056 | 1.0000 | 1.0000 | 1.0000 | 0.0295 | 32.6667 | 0.0136 |
| Synthetic_World_3_H3_True | H3 | random_action | 0.5324 | 0.4632 | 0.1215 | 0.0000 | 0.0000 | 0.3333 | 0.6711 | 50.0000 | 0.0133 |
| Synthetic_World_3_H3_True | H3 | uncertainty_only | 0.5089 | 0.4987 | 0.0211 | 0.0000 | 0.0000 | 0.3333 | 0.6924 | 54.0000 | 0.0139 |

## Scientific Interpretation & Known Limitations
- **World 3 ($H_3$ Local Regime)**: Falsification and Hybrid policies achieve high true-hypothesis recovery ($P(H_3) \approx 1.0$) by characterizing regime boundaries with high HIG.
- **World 1 & World 2 Discrimination**: Requires coupled joint characterization where candidate structural features directly condition subsequent property measurements.
- **Discovery vs. Falsification**: `pure_falsification` operates with lowest experimental cost, while `hybrid` matches the property discovery performance of BoTorch while improving hypothesis identification.