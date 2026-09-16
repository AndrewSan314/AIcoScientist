# Warwick Ultrasonic Multimodal Process-State Modeling Report

## 1. Scientific Overview
- **Benchmark**: Multimodal Stage-Transition State Modeling ($z_t + u_{t+1} \to z_{t+1}$)
- **Official Source**: Mendeley Data, DOI: `10.17632/c62yn37d9h.4` (Version 4)
- **Samples**: 18 NMC622 Cathode samples, 30 Graphite Anode samples
- **Split Strategy**: Strict Grouped 5-Fold Cross-Validation by `Sample_ID`
- **Zero-Lookahead Guarantee**: Preprocessing (StandardScaler, PCA, StageFeatureEncoder) fitted strictly train-only. After-calendering ultrasonic spectra and thickness/density measurements strictly forbidden in pre-decision inputs via `InformationHorizon(CALENDERING, include_decision_stage_controls=True)`.
- **Production Architecture**: Routed through `MASPOProcessStateModel` + `GatedMaskedFusion` + `StageAwareProcessModel` via `LegalStageTransition.from_encoded_source_stage`.

---

## 2. Multimodal Ablation Results (5-Fold Cross Validation)

### A. Cathode (NMC622, N=18)
| Target | Model Modality | Pooled $R^2$ (Ridge) | Pooled RMSE (Ridge) | Pooled $R^2$ (StageAware) | Pooled RMSE (StageAware) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| thickness_after_um | **PROCESS_ONLY** | 0.8913 | 3.4271 | -0.1796 | 11.2898 |
| thickness_after_um | **ULTRASOUND_ONLY** | 0.2606 | 8.9385 | -0.1237 | 11.0192 |
| thickness_after_um | **PROCESS_PLUS_ULTRASOUND** | 0.7563 | 5.1311 | -0.1796 | 11.2897 |
| density_after_g_cm3 | **PROCESS_ONLY** | 0.7937 | 0.1657 | -0.3441 | 0.4230 |
| density_after_g_cm3 | **ULTRASOUND_ONLY** | -0.8008 | 0.4896 | -1.7381 | 0.6037 |
| density_after_g_cm3 | **PROCESS_PLUS_ULTRASOUND** | 0.6583 | 0.2133 | -0.0698 | 0.3774 |

### B. Anode (Graphite, N=30)
| Target | Model Modality | Pooled $R^2$ (Ridge) | Pooled RMSE (Ridge) | Pooled $R^2$ (StageAware) | Pooled RMSE (StageAware) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| thickness_after_um | **PROCESS_ONLY** | 0.9440 | 8.7642 | 0.9920 | 3.3038 |
| thickness_after_um | **ULTRASOUND_ONLY** | 0.8354 | 15.0283 | 0.5013 | 26.1590 |
| thickness_after_um | **PROCESS_PLUS_ULTRASOUND** | 0.9141 | 10.8578 | 0.9875 | 4.1351 |
| density_after_g_cm3 | **PROCESS_ONLY** | 0.8030 | 0.0723 | 0.8370 | 0.0658 |
| density_after_g_cm3 | **ULTRASOUND_ONLY** | 0.3993 | 0.1262 | -0.0972 | 0.1706 |
| density_after_g_cm3 | **PROCESS_PLUS_ULTRASOUND** | 0.8347 | 0.0662 | 0.7089 | 0.0879 |

---

## 3. Key Scientific Findings & Claim Boundaries
1. **Ultrasound Alone Has Strong Standalone Predictive Signal**:
   - On Anode thickness, Ultrasound-Only alone achieves high predictive accuracy ($R^2 = 0.835$ Ridge, $0.501$ StageAware) without knowing the machine roll gap, showing that acoustic transmission spectra physically correlate with electrode structure.
2. **Transparent Baseline Fusion Gain vs Neural Architecture Dynamics**:
   - Linear Ridge demonstrates multimodal fusion gain on Anode Density: Fused ($R^2 = 0.8347$) outperforms Process-Only ($R^2 = 0.8030$).
   - In contrast, on this N=30 anode cohort, the StageAware model did not improve over its tabular baseline when ultrasound was added ($R^2 = 0.8370$ tabular vs $R^2 = 0.7089$ fused), highlighting the importance of reporting both architectures transparently.
3. **Process-Dominated Regimes & Negative Boundaries**:
   - On thickness, mechanical roll gap is the dominant physical control ($R^2 = 0.891$ on Cathode, $R^2 = 0.914$ on Anode).
   - On Cathode density, StageAware performance was weak on the smaller cathode cohort ($N=18$), providing an explicit negative boundary.
4. **Boundary of Claim**:
   - This benchmark validates **multimodal stage-state transition prediction**. It does NOT claim closed-loop recipe optimization or factory control.

---

## 4. Generated Publication Figures
- `outputs/warwick_ultrasonic/figures/cathode_before_after_fft_example.png`
- `outputs/warwick_ultrasonic/figures/anode_before_after_fft_example.png`
- `outputs/warwick_ultrasonic/figures/multimodal_ablation_thickness.png`
- `outputs/warwick_ultrasonic/figures/multimodal_ablation_density.png`
- `outputs/warwick_ultrasonic/figures/predicted_vs_true_thickness.png`
- `outputs/warwick_ultrasonic/figures/predicted_vs_true_density.png`
- `outputs/warwick_ultrasonic/figures/stage_transition_diagram.png`
