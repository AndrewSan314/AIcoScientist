# Battery AI Co-Scientist

Research-grade Bayesian optimization, probabilistic modeling, and closed-loop experimental design platform for battery materials synthesis and fast-charging protocol discovery.

```
MATERIAL / PROCESS INPUTS (Pre-Experiment Controllable Variables)
        ↓
FABRICATION & SYNTHESIS
        ↓
POST-FABRICATION CHARACTERIZATION (Structure, Morphology, Spectroscopy)
        ↓
BATTERY PERFORMANCE / PROPERTIES (Cycle Life, Retention, Energy Density)
        ↓
PROBABILISTIC SURROGATE (RF, XGBoost, GP + Uncertainty Calibration)
        ↓
BAYESIAN ACQUISITION ENGINE (True Joint-Posterior NEI, TuRBO, GP-UCB)
        ↓
EXPLAINABLE EXPERIMENT PROPOSAL (Reason Codes, Uncertainty Bounds)
        ↓
EXPERIMENTAL OBSERVATION / SIMULATOR ORACLE (Firewalled Evaluation)
        ↓
MODEL POSTERIOR UPDATE & RECURSIVE ACTIVE LEARNING
        ↺
```

---

## Capabilities & Key Features

1. **Generic Scientific Workflow**: Domain-agnostic architecture separating controllable pre-experiment variables from post-experiment characterizations and target metrics.
2. **True Joint-Posterior Monte Carlo NEI**: Canonical Noisy Expected Improvement drawing joint Gaussian fantasy realizations $\mathbf{f}_{\text{obs}} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{obs}}, \mathbf{\Sigma}_{\text{obs}})$, handling observation noise and correlated posterior incumbents via Rao-Blackwellized Monte Carlo.
3. **TuRBO Engine in Normalized Coordinates**: Trust Region Bayesian Optimization operating in normalized $[0, 1]^d$ hypercube space with dynamic expansion/contraction state machines, noise-tolerant improvement thresholds, and deterministic global escape exploration.
4. **Adaptive Bayesian Optimization Controller**: Epistemic uncertainty-aware controller dynamically shifting between UCB exploration, True NEI exploitation, and uncertainty reduction based on convergence rate and budget horizon.
5. **Supervised Regression Suite**: Random Forest baseline, XGBoost challenger, and Gaussian Process surrogate with empirical uncertainty calibration (50%, 80%, 90%, 95% predictive interval coverage, Gaussian NLL, and standardized residuals).
6. **Firewalled Closed-Loop Architecture**: Strict separation between optimization policies and evaluation oracles; zero leakage of latent simulator truths or regret into search algorithms.
7. **Structured Explainability & Serialization**: Every proposed candidate includes structured reason codes (`reason_code`), human-interpretable rationales (`recommendation_reason`), feature distances to previous experiments, and full JSON state checkpointing (`save_state` / `load_state`).

---

## Supported Datasets & Benchmarks

| Dataset | Type | Design Space | Target Property |
| :--- | :--- | :--- | :--- |
| **`si_mxene`** | Experimental | Nano-Si / Ti3C2Tx MXene composite anode synthesis | Capacity Retention at 100 cycles (`retention_100`) |
| **`attia`** | Simulator / Grid | 4-step fast charging (224 discrete policies) | Battery Cycle Life (`simulated_lifetime`, cycles) |
| **`attia_continuous`**| Simulator / Continuous | Continuous 3D/4D charging space ($C_1, C_2, C_3 \to C_4$) | Latent thermal-Arrhenius lifetime (`reference_true_lifetime`) |
| **`severson`** | Experimental | 124 Commercial LFP/graphite fast-charging cells | Cycle Life to 80% nominal capacity (`cycle_life`) |
| **`dynamic_cycling`** | Experimental | 22 NMC/graphite cells under dynamic driving profiles | Lifetime under dynamic cycling (`lifetime_cycles`) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AndrewSan314/AIcoScientist.git
cd AIcoScientist

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## CLI Execution Guide

The unified CLI runner `run_pipeline.py` supports all datasets and pipeline modes:

```bash
# 1. Train supervised models on Si-MXene dataset (RF, XGBoost, GP + Uncertainty Calibration)
python run_pipeline.py --dataset si_mxene --mode train

# 2. Generate explainable experiment recommendations with GP-UCB / True NEI
python run_pipeline.py --dataset si_mxene --mode recommend

# 3. Run closed-loop discrete benchmark on Attia 2020 dataset
python run_pipeline.py --dataset attia --mode benchmark

# 4. Run full continuous Bayesian Optimization benchmark across 30 seeds
python run_pipeline.py --dataset attia --mode continuous_benchmark

# 5. Launch interactive Streamlit exploration dashboard
streamlit run app/streamlit_app.py
```

---

## Test Suite & Continuous Integration

Run the comprehensive pytest suite (181 tests):

```bash
# Run fast unit tests
pytest -q -m "not slow" -p no:cacheprovider

# Run full test suite
pytest -q -p no:cacheprovider
```

CI workflows automatically run on GitHub Actions across Linux (`ubuntu-latest`) and Windows (`windows-latest`) on Python 3.11 and 3.12 (`.github/workflows/test.yml`).

---

## Key Output Artifacts

- `outputs/overnight_upgrade_report.md`: Comprehensive 27-section scientific and benchmarking report.
- `outputs/model_metrics.json`: Supervised model holdout metrics and GP uncertainty calibration.
- `outputs/attia_continuous/benchmark_summary.json`: Continuous benchmark metrics, paired Wilcoxon tests, and sample efficiency to thresholds.
- `outputs/attia_continuous/turbo_state_history.csv`: Complete 18-column TuRBO trust region state trace.
- `outputs/attia_continuous/run_manifest.json`: Continuous benchmark run manifest.
- `outputs/attia_continuous/continuous_reference_manifest.json`: Derivative-free continuous reference verification.
