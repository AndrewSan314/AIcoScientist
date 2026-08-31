# Au-Ir-Rh Autonomous SECCM Experimental Benchmark: Dataset Audit Report

## 1. Raw Dataset Provenance & Cryptographic Verification

All source archives have been audited and cryptographically verified:

| Archive Name | SHA-256 Digest | Size (Bytes) | Description |
| :--- | :--- | :--- | :--- |
| **EDX_dataset.zip** | 95340043ccecec0e9c92c75900c1992d3f92e98fc72cc80c895b53289f5b8791 | 11,963 | Energy-dispersive X-ray spectroscopy composition measurements |
| **SECCM_dataset.zip** | 6e30cdb3a5ecd257daa091bfc2b6cfaa7889d27938e27ae614977d8845f0ffc0 | 6,608,379 | Scanning electrochemical cell microscopy voltammograms and fitted kinetic parameters |
| **XRD_dataset.zip** | c87af16d875a97f32127753cec0b6ac179efb3735ec9386eaebf5f130592cfde | 25,794,320 | High-throughput X-ray diffraction spectra across 1,026 areas |

---

## 2. Experimental Composition Space (EDX)

The dataset contains three physical combinatorial thin-film gradient libraries:
- **Au-rich**: 342 physical areas
- **Ir-rich**: 342 physical areas
- **Rh-rich**: 342 physical areas
- **Total physical spots**: 1,026 areas

Composition variables satisfy the ternary constraint:
\text{Au} + \text{Ir} + \text{Rh} = 100 \text{ at}\%
The optimization space is parameterized via two independent coordinates $ with $\text{Rh} = 100 - \text{Au} - \text{Ir}$.

---

## 3. Physical Target Properties (SECCM)

| Target Property | Source Column | Unit | Optimization Goal | Min Value | Median Value | Max Value | Best Candidate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **^0$ (Primary)** | k^0 [cm/s] | $\text{cm/s}$ | **Maximize** | .000100$ | .005112$ | **.014201$** | AUIRH_Au-rich_170 |
| **{\text{lim}}$ (Secondary)** | i_lim [A/cm^2] | $\text{A/cm}^2$ | Maximize | .000002$ | .000038$ | .000139$ | AUIRH_Ir-rich_095 |
| **$\alpha$ (Secondary)** | lpha [a.u.] | dimensionless | Maximize | .300000$ | .512400$ | .880000$ | AUIRH_Rh-rich_204 |

### Measurement Noise Characterization:
- Au-Ir-Rh target values are derived from real experimental electrochemical measurements.
- Raw LSV curves contain source-provided standard-deviation columns across potential scans, but this curve-level uncertainty is not propagated into candidate-specific heteroscedastic ^0$ uncertainties in Stage 1.
- Stage 1 closed-loop optimization uses a Gaussian Process surrogate with a learned observation-noise term (WhiteKernel).
- True Monte Carlo NEI explicitly integrates over the posterior distribution of the latent function, insulating optimization from measurement artifacts under the fitted GP noise model.

---

## 4. Programmatic XRD Spectroscopic Audit

All 1,026 .xy diffractograms across the three libraries were programmatically audited:

- **Total Spectra Audited**: Exactly **1,026** files (342 Au-rich + 342 Ir-rich + 342 Rh-rich)
- **Data Points per Spectrum**: Exactly **4,500** points per file
- **Angular Range (\theta$)**: **.00000^\circ \rightarrow 99.98000^\circ$**
- **Angular Step Size**: Uniform step of **.02000^\circ$**
- **Grid Consistency**: **100% Identical** across all 1,026 files
- **Stage 1 Firewall Policy**: XRD spectra are strictly quarantined and excluded from Stage 1 optimizer features.

---

## 5. Clean Candidate Pool & Perimeter Handling

Of the 342 physical areas per library, exactly **20 perimeter areas** per wafer were not evaluated in SECCM:
\text{Excluded Perimeter IDs} = [12, 24, 25, 26, 38, 39, 40, 41, 55, 56, 280, 281, 295, 296, 297, 298, 310, 311, 312, 324]

- **Final Clean Candidate Pool**: Exactly **966** fully-joined measured candidates ( \times 3$).
- **Candidate Identifier Schema**: AUIRH_<LIBRARY>_<AREA> (e.g. AUIRH_Au-rich_170).
