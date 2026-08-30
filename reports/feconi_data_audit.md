# Data Audit Report: NIST Fe-Co-Ni Combinatorial Benchmark Dataset

## 1. Provenance and Metadata
- **Source**: NIST REMI Combinatorial Library (`usnistgov/remi`)
- **File**: `remi/data/Combinatorial Libraries/Fe-Co-Ni/FeCoNi_benchmark_dataset_220501a.mat`
- **File SHA256**: `aaee4ddb6cf711e789e1edd6358145746611b273cecf2257699ea057d5feb1dc`
- **Primary Publications**:
  - Wang, A., Liang, H., McDannald, A., Takeuchi, I., & Kusne, A. G. (2022). *Benchmarking Active Learning Strategies for Materials Optimization and Discovery*. Oxford Open Materials Science, 2(1), itac006. (arXiv:2204.05838)
  - Yoo, Y. K., Xue, Q., Chu, Y. S., et al. (2006). *Identification of amorphous phases in the Fe–Ni–Co ternary alloy system using continuous phase diagram material chips*. Intermetallics, 14(3), 241–247.

---

## 2. Structure, Shapes, and Types

| Key | Array Shape | Data Type | Units / Meaning | Missing / Inf | Value Range (Min - Max) | Mean ± Std |
|---|---|---|---|---|---|---|
| `C` | `(921, 3)` | `float64` | Atomic % composition | 0 / 0 | 3.6 - 91.7 % | 33.33 ± 21.13 % |
| `Coer` | `(921, 1)` | `float64` | Magnetic coercivity [mT] | 0 / 0 | 1.279 - 10.934 mT | 3.631 ± 2.177 mT |
| `Kerr` | `(921, 1)` | `float64` | Kerr rotation [mrad] | 0 / 0 | 0.0132 - 0.8250 mrad | 0.3683 ± 0.1725 mrad |
| `TTH` | `(1, 89)` | `float64` | Two-theta angle [°] | 0 / 0 | 42.60 - 47.00 ° | 44.80 ± 1.28 ° |
| `XRD` | `(921, 89)` | `float64` | X-ray diffraction intensity | 0 / 0 | 121.0 - 36186.0 | 3732.6 ± 5978.2 |

---

## 3. Data Integrity & Verification

1. **Cleaned 921-Sample Benchmark**:
   - Total sample count: **921**.
   - Verified that the dataset is the complete, cleaned benchmark from NIST REMI. No arbitrary filtering or cleaning was applied.
2. **Composition Row Sums**:
   - Minimum row sum: **99.90%**
   - Maximum row sum: **100.10%**
   - Mean row sum: **100.0034% ± 0.0493%**
   - (Variations are purely due to standard single-decimal rounding of raw measured compositions).
3. **Uniqueness**:
   - Duplicate compositions: **0** (921 unique compositions).
   - Duplicate XRD spectra: **0** (921 unique diffraction curves).
   - Duplicate target vectors: **0** (921 unique property pairs).
4. **Composition Spacing**:
   - Regular triangular/ternary grid sampling.
   - Nearest-neighbor distance in composition space: **2.687 at.%** (min) to **2.828 at.%** (max), mean **2.699 at.%**.

---

## 4. Composition Column Mapping

```yaml
composition_column_mapping:
  column_0: Co
  column_1: Fe
  column_2: Ni

evidence:
  - Kerr rotation max (0.82504 mrad) at C=[39.8, 55.8, 4.4] aligns with the Fe-Co binary (column 2 / Ni = 4.4%)
  - Coercivity global max (10.9340 mT) at C=[53.9, 5.9, 40.2] aligns with the Co-Ni binary (column 1 / Fe = 5.9%) exactly as documented in Wang et al. (2022) Figure 4(c)
  - Pure Ni corner (C[:, 2] > 85%) exhibits near-zero Kerr rotation (~0.0228 mrad), characteristic of elemental Ni
  - Pure Fe-rich corner (C[:, 1] > 85%) has mean Kerr rotation 0.578 mrad; pure Co-rich corner (C[:, 0] > 85%) has mean Kerr rotation 0.473 mrad

confidence: HIGH
```

### Global Maxima Locations
- **Kerr Rotation Global Optimum**:
  - Sample Index: `760` (`FECONI_760`)
  - Value: **0.82504 mrad**
  - Composition: **Co: 39.8 at.%, Fe: 55.8 at.%, Ni: 4.4 at.%** (near Fe-Co binary)
- **Magnetic Coercivity Global Optimum**:
  - Sample Index: `64` (`FECONI_064`)
  - Value: **10.9340 mT**
  - Composition: **Co: 53.9 at.%, Fe: 5.9 at.%, Ni: 40.2 at.%** (near Co-Ni binary / amorphous-crystalline interface)

---

## 5. Architectural Readiness
- **2D Independent Feature Representation**: `X = [Co, Fe]` (with `Ni = 100 - Co - Fe`).
- **Offline Oracle**: Encapsulates 921 measured physical samples, strictly forbids continuous interpolation, and guards unrevealed target properties and XRD spectra from optimizer visibility.
- **Stage Isolation**: XRD spectra are preserved for downstream structural characterization modeling (Stage A/B) but strictly hidden during the Optimizer Benchmark (v1).
