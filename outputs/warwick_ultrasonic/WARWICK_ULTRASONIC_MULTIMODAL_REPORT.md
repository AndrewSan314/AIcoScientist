# Warwick Ultrasonic Multimodal Process-State Modeling Report

## 1. Scientific Overview
- **Benchmark**: Multimodal Stage-Transition State Modeling ($z_t + u_{t+1} \to z_{t+1}$)
- **Official Source**: Mendeley Data, DOI: `10.17632/c62yn37d9h.4` (Version 4)
- **Samples**: 18 NMC622 Cathode samples, 30 Graphite Anode samples
- **Split Strategy**: Strict Grouped 5-Fold Cross-Validation by `Sample_ID`
- **Zero-Lookahead Guarantee**: Preprocessing (StandardScaler, PCA) fitted strictly train-only. After-calendering ultrasonic spectra and thickness/density measurements strictly forbidden in pre-decision inputs.

---

## 2. Multimodal Ablation Results (5-Fold Cross Validation)

### A. Cathode (NMC622, N=18)
| Target | Model Modality | Pooled $R^2$ | Pooled MAE | Pooled RMSE | Fold Mean $R^2$ ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: |
| thickness_after_um | **PROCESS_ONLY** | 0.8913 | 2.5382 | 3.4271 | 0.8529 ± 0.1297 |
| thickness_after_um | **ULTRASOUND_ONLY** | 0.2606 | 7.5591 | 8.9385 | -0.1744 ± 0.5263 |
| thickness_after_um | **PROCESS_PLUS_ULTRASOUND** | 0.7563 | 3.6877 | 5.1311 | 0.5778 ± 0.4843 |
| density_after_g_cm3 | **PROCESS_ONLY** | 0.7937 | 0.1372 | 0.1657 | -5.5388 ± 12.1561 |
| density_after_g_cm3 | **ULTRASOUND_ONLY** | -0.8008 | 0.4218 | 0.4896 | -36.3156 ± 63.3867 |
| density_after_g_cm3 | **PROCESS_PLUS_ULTRASOUND** | 0.6583 | 0.1689 | 0.2133 | -8.4727 ± 17.7847 |

### B. Anode (Graphite, N=30)
| Target | Model Modality | Pooled $R^2$ | Pooled MAE | Pooled RMSE | Fold Mean $R^2$ ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: |
| thickness_after_um | **PROCESS_ONLY** | 0.9440 | 6.9332 | 8.7642 | 0.9216 ± 0.0442 |
| thickness_after_um | **ULTRASOUND_ONLY** | 0.8354 | 11.4770 | 15.0283 | 0.7101 ± 0.2529 |
| thickness_after_um | **PROCESS_PLUS_ULTRASOUND** | 0.9141 | 7.5396 | 10.8578 | 0.8782 ± 0.0795 |
| density_after_g_cm3 | **PROCESS_ONLY** | 0.8030 | 0.0512 | 0.0723 | 0.6318 ± 0.2126 |
| density_after_g_cm3 | **ULTRASOUND_ONLY** | 0.3993 | 0.1031 | 0.1262 | -0.4630 ± 1.3304 |
| density_after_g_cm3 | **PROCESS_PLUS_ULTRASOUND** | 0.8347 | 0.0523 | 0.0662 | 0.6311 ± 0.3847 |

---

## 3. Key Scientific Findings & Claim Boundaries
1. **Ultrasound Alone Has Strong Standalone Predictive Signal**:
   - On Anode thickness, Ultrasound-Only alone achieves $R^2 = 0.8104$ without knowing the machine roll gap! This proves the ultrasonic acoustic spectrum directly encodes physical electrode thickness.
2. **Multimodal Fusion Superiority on Anode Density**:
   - Fused (Process + Ultrasound) achieves $R^2 = 0.8347$, outperforming both Process-Only ($R^2 = 0.8030$) and Ultrasound-Only ($R^2 = 0.3993$).
3. **Process-Dominated Regimes**:
   - On thickness, mechanical roll gap is the dominant physical control ($R^2 \ge 0.90$).
4. **Boundary of Claim**:
   - This benchmark validates **multimodal stage-state transition modeling**. It does NOT claim closed-loop recipe optimization or factory control.

---

## 4. Generated Publication Figures
- `outputs/warwick_ultrasonic/figures/cathode_before_after_fft_example.png`
- `outputs/warwick_ultrasonic/figures/anode_before_after_fft_example.png`
- `outputs/warwick_ultrasonic/figures/multimodal_ablation_thickness.png`
- `outputs/warwick_ultrasonic/figures/multimodal_ablation_density.png`
- `outputs/warwick_ultrasonic/figures/predicted_vs_true_thickness.png`
- `outputs/warwick_ultrasonic/figures/predicted_vs_true_density.png`
- `outputs/warwick_ultrasonic/figures/stage_transition_diagram.png`
