# Literature & Novelty Audit: Falsification-First Scientific Experiment Design

**Domain**: Autonomous Materials Discovery & Model Discrimination  
**Date**: September 2026  
**Target Identity**: "Falsification-first autonomous materials science through competing-hypothesis experimental design."  
**Core Product**: AIcoScientist — Autonomous Scientific Decision System  

---

## 1. Executive Summary & Research Motivation

Standard autonomous materials science workflows almost exclusively operate as **black-box optimizers**: given a target property $y$ (e.g., ionic conductivity, catalytic activity $k^0$, battery cycle life), they use Bayesian Optimization (BO) or active learning to find $\arg\max_x y(x)$.

While effective at finding high-performing materials, pure property optimization is **scientifically blind**:
1. It does not test or refine scientific theories or hypotheses about *why* materials behave as they do.
2. It cannot arbitrate between competing physical mechanisms (e.g., is catalytic activity determined purely by surface alloy composition, or mediated by XRD-observable crystal lattice strain/phase regimes?).
3. It cannot prioritize characterization experiments (such as X-ray diffraction, spectroscopy) whose primary value is epistemic (reducing uncertainty over scientific models) rather than direct property discovery.

**AIcoScientist** addresses this foundational gap by treating experiment selection as **Bayesian Optimal Experimental Design (OED) for Model Discrimination**: maintaining explicit, competing predictive scientific hypotheses, forecasting their divergent predictions across multimodal candidate actions (characterization vs. performance testing), and selecting experiments that maximize the **Expected Hypothesis Information Gain (HIG)**.

---

## 2. Structured Literature Review (2023–2026 Focus)

| # | Paper & Authors | Year | Domain | Hypothesis Representation | Experiment Selection Objective | Modality Handling | Ground Truth Status | Evaluation Protocol | Relevance & Gap Remaining |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Atkinson, Fedorov, & Drovandi** (*Stat. Sci.*) | 2023 | Optimal Experimental Design (Statistics) | Parametric competing likelihood families $\{M_1, \dots, M_K\}$. | Expected Shannon information gain on model index $I(M; Y \mid d)$ / T-optimality. | Single scalar/vector output. | Known synthetic models. | Analytical & Monte Carlo comparison of T-optimal vs. D-optimal designs. | **Relevance**: Classical foundation for model discrimination. **Gap**: Does not address multimodal characterization-vs-performance decision trade-offs in materials science. |
| **2** | **Kleinegesse & Gutmann** (*AISTATS / NeurIPS*) | 2023 | Bayesian Experimental Design (ML) | Neural / Implicit probabilistic models with parameter priors. | Mutual Information Neural Estimation (MINE) for OED. | Single continuous domain. | Synthetic benchmark simulators. | Simulation-based Bayesian optimal design over parameter spaces. | **Relevance**: Scalable mutual information estimators. **Gap**: Focuses on parameter inference within a single model rather than discrete hypothesis discrimination across multimodal materials spaces. |
| **3** | **Rainforth, Foster, et al.** (*JMLR*) | 2024 | Active Inference & OED | Probabilistic graphical models and hierarchical priors. | Nested Monte Carlo (NMC) and variational lower bounds on EIG. | Synthetic vector measurements. | Controlled synthetic priors. | Convergence bounds for mutual information estimators. | **Relevance**: Rigorous bounds for information gain calculations. **Gap**: Pure theoretical ML without physical action cost trade-offs or experimental oracle firewalls. |
| **4** | **Kusne, Noack, Stach, et al.** (*Nat. Commun. / Materials Today*) | 2023–2025 | Autonomous Materials Laboratories (SLAM / CAMEO) | Phase mapping & crystal structure phase diagram models. | Structural uncertainty reduction & active phase boundary identification. | XRD characterization on combinatorial libraries. | Empirical materials libraries (thin films). | Phase diagram mapping efficiency vs. uniform grid. | **Relevance**: Gold standard for autonomous XRD phase mapping. **Gap**: Optimizes structure mapping itself; does not formalize competing hypotheses linking structure to functional electrocatalytic performance. |
| **5** | **MacLeod, Parlane, et al.** (*Science / Nat. Synth.*) | 2023–2024 | Self-Driving Labs (Ada / Robo-chemistry) | Single surrogate GP predicting physical performance metrics. | Standard Expected Improvement (EI), Upper Confidence Bound (UCB). | Single performance modality (film conductivity, yield). | Physical robot measurements. | Speed of finding high-performing candidate. | **Relevance**: Benchmarking autonomous physical execution. **Gap**: Optimization-only; zero hypothesis discrimination or multimodal epistemic action selection. |
| **6** | **Boiko, MacKnight, Gomes, et al. (Coscientist)** (*Nature*) | 2023 | LLM Agents for Chemistry | Unstructured natural language text prompts in LLM memory. | Prompt-driven LLM planning and chemical synthesis routing. | Organic synthesis actions via robotic APIs. | Qualitative chemistry benchmarks (Suzuki/Sonogashira). | Task completion success rate. | **Relevance**: Multi-agent LLM reasoning in lab settings. **Gap**: Unstructured reasoning; lacks formal quantitative predictive distributions, calibrated entropy calculations, or mathematical OED. |
| **7** | **Sakana AI / Lu et al. (The AI Scientist)** (*arXiv*) | 2024 | Autonomous AI Research | Natural language scientific papers and Python ML code. | LLM-driven idea generation, code execution, and manuscript review. | ML code scripts and textual writeups. | Automated peer review scoring. | Number of generated papers and simulated review scores. | **Relevance**: Conceptual vision of end-to-end scientific AI. **Gap**: Software/ML domain only; does not execute physical materials experiments or solve multimodal model discrimination. |
| **8** | **Bran, Cox, White, et al. (ChemCrow)** (*Nat. Mach. Intell.*) | 2024 | Chemical LLM Agent Tool Use | LLM tool integration with chemical calculators and search. | Heuristic LLM tool-calling. | Molecular property query tools. | Literature retrieval & reaction planning. | Expert chemical evaluation of generated protocols. | **Relevance**: Clean separation of agent reasoning and deterministic tool backends. **Gap**: Does not maintain formal probabilistic hypotheses or calculate Expected Information Gain. |
| **9** | **Alverson, Lookman, et al.** (*npj Comput. Mater.*) | 2025 | Active Learning for Materials Science | Dual Gaussian process models (composition vs. structure). | Multi-information source active learning (MISAL). | Multi-fidelity simulations (DFT vs. experiments). | Synthetic & historical materials data. | Reduction in RMSE across composition-property landscape. | **Relevance**: Multi-source modeling in materials science. **Gap**: Focuses on regression error reduction (MISAL) rather than hypothesis falsification and model discrimination. |

---

## 3. Foundational Conceptual Taxonomy

To maintain scientific integrity and prevent conflation of terms, AIcoScientist adheres to the following rigorous definitions:

```
                            Decision Space in Materials AI
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
Property Optimization             Uncertainty Reduction              Model Discrimination
 (Bayesian Optimization)             (Active Learning)                 (Hypothesis Testing)
         │                                 │                                 │
 Goal: Find x* = argmax y(x)       Goal: Minimize Var[f(x)]           Goal: Maximize I(H; Y_a | D)
 Criteria: EI, UCB, NEI, TS       Criteria: Max-Entropy, ALC         Criteria: HIG, EHD, T-optimal
 Focus: Best performance           Focus: Global accuracy             Focus: Refuting wrong models
```

1. **Property Optimization**: Maximizes an experimental outcome $y(x)$ using an acquisition function (e.g., BoTorch Expected Improvement). It does not care which underlying model is physically correct, as long as it finds a high value.
2. **Uncertainty Reduction (Active Learning)**: Selects points to minimize overall regression predictive variance across the candidate space ($\max_x \sigma^2(x)$). It treats the model architecture as fixed.
3. **Model Discrimination (Bayesian OED)**: Selects points where two or more distinct hypothesis families $H_1, H_2, \dots, H_K$ make **conflicting predictions**, maximizing the expected drop in hypothesis entropy:
   $$\text{HIG}(a) = I(H; Y_a \mid \mathcal{D}) = \mathcal{H}[P(H \mid \mathcal{D})] - \mathbb{E}_{y \sim p(y \mid a, \mathcal{D})}[\mathcal{H}[P(H \mid \mathcal{D}, y)]]$$
4. **Causal Intervention (Pearl's Framework)**: Requires active physical control over latent causal variables (e.g., forcing crystal strain independently of composition $P(y \mid \text{do}(z), x)$). Because experimental libraries sample fixed synthesized points, our framework performs **observational model discrimination under experimental selection**, not unconstrained do-calculus.
5. **Falsification (Popperian Experiment Design)**: Specifically designs tests targeting high-risk predictions of a leading hypothesis where empirical observations could refute or drastically penalize its likelihood.

---

## 4. Defensible Novelty Statement

> **Defensible Novelty Statement**:
> AIcoScientist is the first closed-loop materials science system that integrates:
> 1. Formal probabilistic modeling of competing scientific hypotheses ($H_1$: Composition-Sufficient, $H_2$: Structure-Informed, $H_3$: Local Structural-Regime);
> 2. Multimodal Expected Hypothesis Information Gain (HIG) estimation over both structural characterization (XRD) and functional electrochemical performance (SECCM $k^0$);
> 3. Production Bayesian Optimization infrastructure (BoTorch) strictly isolated for candidate property discovery;
> 4. Pre-registered prediction tracking through an append-only cryptographic ledger and a strict offline oracle firewall;
> 5. Quantitative true-hypothesis recovery evaluation on controlled synthetic worlds alongside real experimental Au-Ir-Rh validation.

---

## 5. Claims We MUST NOT Make (Forbidden Overclaims)

1. **DO NOT claim discovery of ultimate physical causal truth**: On the real Au-Ir-Rh dataset, we evaluate sequential predictive support among specified surrogate model families. We do NOT claim to have "proven the fundamental quantum-mechanical mechanism" or "discovered new physics."
2. **DO NOT claim first-ever Bayesian experimental design**: Bayesian OED for model discrimination has a long mathematical history (Box & Hill, Atkinson & Fedorov, Rainforth). Our contribution is its specialized multimodal realization in materials science with candidate discovery trade-offs.
3. **DO NOT claim causal do-calculus intervention**: Physical library synthesis couples composition and structure; we cannot independently set lattice constants to arbitrary non-physical values via do-interventions.
4. **DO NOT claim exact Bayesian parameter marginalization**: Our hypothesis belief update uses sequential predictive log-score evidence accumulation, which is an established, computationally tractable sequential predictive likelihood approximation, not an analytic infinite-dimensional marginal likelihood.
5. **DO NOT claim battery materials discovery in current demo**: Au-Ir-Rh is an electrocatalytic library for hydrogen/oxygen reaction kinetics; battery materials R&D is explicitly documented as future roadmap application scope.
