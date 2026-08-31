# AIcoScientist Discovery Console: Demo & Recording Guide

**Product Identity**: AIcoScientist — Autonomous Scientific Decision System
**Research Thesis**: *"From optimizing candidate materials to deciding what science to do next."*
**Validation Domain**: Au-Ir-Rh Multimodal Materials Dataset (966 physical SECCM library samples)
**Target Application (Future Scope)**: Battery Materials R&D

---

## 1. Quick Launch

Launch the interactive discovery console with one command (runs locally, 100% offline, zero external LLM/API dependencies):

```powershell
.\.venv\Scripts\streamlit.exe run app/demo_ai_scientist.py
```
*(or `.venv\Scripts\python.exe -m streamlit run app/demo_ai_scientist.py`)*

Or run the benchmark execution:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.epistemic_action_benchmark
```

---

## 2. Exact Deterministic Demo Trajectory (`seed=42`)

Under default initialization (`seed=42`, 6 initial property tests, 4 initial XRD measurements):

| Step                                               | Candidate ID          | Action Type        | Target Hypothesis                 | Scientific Info | BoTorch Discovery | Net Total Value  |
| :------------------------------------------------- | :-------------------- | :----------------- | :-------------------------------- | :-------------- | :---------------- | :--------------- |
| **Recommendation 1**                         | `AUIRH_Au-rich_127` | **PROPERTY** | **H2 (Structure-Mediated)** | 0.9328          | 0.9998            | **0.9326** |
| **Recommendation 2** (after executing Rec 1) | `AUIRH_Ir-rich_177` | **PROPERTY** | **H1 (Direct Composition)** | 1.0000          | 0.9872            | **0.9898** |

*Note: The candidate selection and action ranking dynamically adapt as empirical measurements are revealed through the offline oracle.*

---

## 3. 90-Second Recording Script & Video Flow

| Timestamp              | Video Section                           | Screen Focus / Action                                               | Spoken / Visual Narrative                                                                                                                                                                                                            |
| :--------------------- | :-------------------------------------- | :------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0:00 – 0:10** | **Campaign Setup**                | Wide view of Discovery Console showing 966 candidates               | *"Most materials AI systems act as black-box optimizers searching for high scores. AIcoScientist is different: it autonomously decides what scientific experiment to run next and why."*                                           |
| **0:10 – 0:25** | **Scientific Hypotheses**         | Left column: H1, H2, H3 cards with evidence weight bars             | *"It maintains explicit, competing scientific hypotheses: Is electrocatalytic activity explained purely by nominal composition (H1), mediated by XRD crystal structure (H2), or localized in structural transition regimes (H3)?"* |
| **0:25 – 0:40** | **Autonomous Recommendation**     | Click**💡 Ask AI Scientist**; show Next Best Experiment Card  | *"Clicking 'Ask AI Scientist' evaluates all valid actions. Guided by BoTorch Bayesian optimization and epistemic information gain, it recommends candidate AUIRH_Au-rich_127 to test hypothesis H2."*                              |
| **0:40 – 0:55** | **Experiment Execution & Reveal** | Click**⚡ Run Experiment**; show real revealed measurement    | *"Executing the experiment reveals the real measured property from the physical sample. The hypothesis evidence shifts in real time from empirical observation events without data leakage."*                                      |
| **0:55 – 1:10** | **Sequential Adaptation**         | Click**💡 Ask AI Scientist** again; system proposes next step | *"With observation history updated, the policy adapts: it evaluates the updated candidate landscape and selects candidate AUIRH_Ir-rich_177 under updated hypothesis weights."*                                                    |
| **1:10 – 1:22** | **Counterfactual Reasoning**      | Open**⚖️ Why Not Another Candidate?** expander              | *"AIcoScientist provides auditable counterfactuals, explaining numerically why candidate AUIRH_Au-rich_127 was chosen over alternative options under current policy value weights."*                                               |
| **1:22 – 1:30** | **Timeline & Battery Roadmap**    | Bottom section: Action Timeline & Battery Roadmap Panel             | *"From Au-Ir-Rh real-world validation to future battery materials R&D—AIcoScientist: Run fewer experiments, learn more from each one."*                                                                                           |

---

## 4. Key Invariants & Claim Boundaries

1. **Strict Offline Firewall**: Decision policies never access unrevealed XRD spectra or target $k^0$ kinetics before the corresponding action is executed.
2. **Exact Physical Sample Mapping**: All 966 candidates map 1:1 to real experimental files (`.xy` diffractograms with 4500 numeric rows and SECCM fit tables).
3. **Evidence-Weighted Beliefs**: Hypothesis probabilities are normalized softmax evidence weights, not ungrounded LLM scores or uncalibrated physical mechanism assertions. Initial seed data produces zero fake evidence events.
4. **BoTorch Candidate Scoring**: Property candidate discovery potential is scored directly via `BoTorchBackend` Expected Improvement.
5. **Roadmap Separation**: Au-Ir-Rh is an electrocatalytic library; battery materials R&D is explicitly labeled as the future target application roadmap.
