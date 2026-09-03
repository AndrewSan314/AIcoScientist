# In-Silico Surrogate Simulation Benchmark Report

**Status:** `SIMULATED SURROGATE ORACLE - In-Silico Computational Approximation Only`  
**Oracle Kind:** `SIMULATED_SURROGATE`  
**Physical Synthesis:** `False`  
**Search Space Slice:** 50,000 LiFSI Virtual Candidates  
**Working Set Size:** 200  
**Screening Runtime:** 0.0244 seconds  
**Surrogate Model Family:** ExtraTreesRegressor (100 trees, max_depth=8)  

## Policy Closed-Loop Performance
| Policy | Best Simulated Capacity | Mean Simulated Capacity | Cumulative HIG (nats) | Queried Steps |
| :--- | :---: | :---: | :---: | :---: |
| **HYBRID** | 0.2748 | 0.1435 | 0.6361 | 15 |
| **DISCOVERY_ONLY** | 0.1620 | 0.1267 | 0.6853 | 15 |
| **RANDOM** | 0.2368 | 0.0980 | 0.0000 | 15 |

> [!IMPORTANT]
> This simulation evaluates algorithmic screening throughput and working-set information capture. It does not represent wet-lab physical experimental synthesis.  
> **Disclaimer:** Computational simulation under frozen surrogate. Not physical experimental validation.
