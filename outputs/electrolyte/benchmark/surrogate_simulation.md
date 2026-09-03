# In-Silico Surrogate Simulation Benchmark Report

**Status:** `SIMULATED SURROGATE ORACLE - In-Silico Computational Approximation Only`  
**Oracle Kind:** `SIMULATED_SURROGATE`  
**Physical Synthesis:** `False`  
**Search Space Slice:** 50,000 LiFSI Virtual Candidates  
**Working Set Size:** 200  
**Screening Runtime:** 0.0618 seconds  
**Surrogate Model Family:** ExtraTreesRegressor (100 trees, max_depth=8)  
**Oracle Working-Set Max Capacity:** 0.4376  

## Policy Closed-Loop Performance (Mean ± Std over Seeds 42, 101, 2024)
| Policy | Best Simulated Cap | Mean Simulated Cap | Cum. HIG (nats) | HIG / action (nats) | Entropy reduction | Mean Regret | Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RANDOM** | 0.3142 ± 0.0993 | 0.1118 ± 0.0070 | 0.6595 ± 0.2599 | 0.0439 ± 0.0173 | 0.5053 | 0.1285 | 15 |
| **BOTORCH_EI_DIRECT** | 0.2239 ± 0.0306 | 0.1281 ± 0.0068 | 0.5750 ± 0.1329 | 0.0384 ± 0.0089 | 0.3459 | 0.2136 | 15 |
| **BOTORCH_GPUCB_DIRECT** | 0.2247 ± 0.0364 | 0.1302 ± 0.0022 | 0.5542 ± 0.2128 | 0.0370 ± 0.0142 | 0.3343 | 0.2129 | 15 |
| **PURE_FALSIFICATION** | 0.3512 ± 0.1290 | 0.1423 ± 0.0317 | 0.6959 ± 0.1297 | 0.0464 ± 0.0087 | 0.8470 | 0.0916 | 15 |
| **HYBRID_DEFAULT** | 0.3082 ± 0.0781 | 0.1318 ± 0.0180 | 0.9807 ± 0.1603 | 0.0654 ± 0.0107 | 0.6643 | 0.1294 | 15 |
| **DISCOVERY_ONLY** | 0.2239 ± 0.0306 | 0.1281 ± 0.0068 | 0.5578 ± 0.1171 | 0.0372 ± 0.0078 | 0.3459 | 0.2136 | 15 |

> [!IMPORTANT]
> This simulation evaluates algorithmic screening throughput and working-set information capture. It does not represent wet-lab physical experimental synthesis.  
> **Disclaimer:** Computational simulation under frozen surrogate. Not physical experimental validation.
