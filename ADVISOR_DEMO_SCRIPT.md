# AIcoScientist — Advisor Demonstration Script (6–8 Minute Talk Track)

**Target Audience:** Academic Advisor / Research Committee  
**Goal:** Establish that AIcoScientist is a mathematically rigorous, multi-hypothesis scientific decision engine, not just a black-box property optimizer.  
**Hotkeys:** Press `P` to enter Presenter Mode. Use `Right Arrow` to advance scenes, and `N` to toggle speaker notes.

---

## Timeline & Talk Track

```text
0:00 — 1:15 | Scene 1: The Research Question & Domain-Agnostic Contribution (Workspace 1)
1:15 — 2:30 | Scene 2: Discovery Lab — Competing Hypotheses & Predictive Distributions (Workspace 1)
2:30 — 3:45 | Scene 3: Next Experiment — Candidate × Modality Trade-off & Exact Score (Workspace 1)
3:45 — 5:15 | Scene 4: The Scientific Wow Moment (Preregister → Reveal → Update Loop) (Workspace 1)
5:15 — 6:30 | Scene 5: Benchmark Validation — 180 Trajectories & Recovery Dynamics (Workspace 2)
6:30 — 8:00 | Scene 6: Real A-Lab Evidence, Electrolyte Scale & System Governance (Workspace 2 & 3)
```

---

### Scene 1: The Research Question & Contribution (0:00 – 1:15)
**On Screen:** *Workspace 1: Discovery Lab (`/discovery`)*  
**Visual Action:** Point to the top banner and core manifesto: *"Autonomous Multimodal Decision & Epistemic Inference Loop."*

> "Good morning, Professor.
> 
> Most machine learning in materials science asks a single question: *'Which material is predicted to have the highest property value?'*
> 
> But in a real laboratory, physical experiments are costly, irreversible, and multi-modal. A scientist does not merely measure final device performance—they choose characterization tools like XRD or Rietveld refinement to determine *why* a reaction succeeded or failed, confirming or falsifying competing mechanistic hypotheses.
> 
> AIcoScientist is an autonomous decision framework that asks: **'Which experiment should we perform next, and why?'**
> 
> As you can see on this Mission Control dashboard, our system is structured into three unified workspaces:
> 1. **Discovery Lab:** The active decision engine and epistemic inference loop.
> 2. **Evidence & Benchmarks:** 5 core research questions backed by 180 benchmark trajectories, 1,035 real physical experiments, and 333,000 electrolyte candidates.
> 3. **Research System:** Reusable domain-agnostic abstractions, an immutable 5,333-record audit ledger, and 50 software verification gates.
> 
> Today, I will walk you through how our system makes these decisions."

---

### Scene 2: Discovery Lab: Competing Hypotheses & Predictions (1:15 – 2:30)
**On Screen:** *Workspace 1: Discovery Lab (`/discovery`) — Analytical Grid*  
**Visual Action:** Show the **Hypothesis Belief Trajectory** chart, then switch to the **Predictive Distributions** view. Point to the right-hand **Hypothesis Observatory**.

> "Let us look at our primary workspace, the **Discovery Lab**.
> 
> In the top-left analytical view, notice the **Hypothesis Belief Trajectory**. Instead of a single black-box property model, AIcoScientist maintains three competing mechanistic hypotheses:
> 1. $H_1$: Phase Purity Limited (Emerald)
> 2. $H_2$: Composition Homogeneity Limited (Amber)
> 3. $H_3$: Morphology Kinetics Limited (Violet)
> 
> Note our explicit epistemic notice: these belief weights represent *relative explanatory model likelihoods* among simplified competing models, not a physical claim that one theory is absolute truth.
> 
> If we toggle the tab to **Predictive Distributions**, you see the Gaussian probability densities $p(y \mid a, H_k)$ that each hypothesis preregisters before any physical evidence is observed.
> 
> Notice the amber banner: **Observation Blinding Firewalled**. Prior to data acquisition, ground truth is strictly locked away."

---

### Scene 3: Next Experiment: Candidate × Modality Trade-off (2:30 – 3:45)
**On Screen:** *Workspace 1: Discovery Lab — Lower Explorer & Score Decomposition*  
**Visual Action:** Toggle the Candidate Space Explorer between **[Action Matrix Heatmap]**, **[3D Stark Hologram]**, and **[Counterfactual Table]**. Point to the **Exact Score Decomposition Waterfall** on the right.

> "Now, how does AIcoScientist decide what to test next?
> 
> In the lower explorer, we toggle between the **Candidate × Modality Matrix**, our **3D Stark Hologram Sphere**, and the **24-Action Counterfactual Table**.
> 
> Look at the **Exact Score Decomposition Waterfall** on the right:
> $$S(a) = w_H \cdot \widetilde{HIG}(a) + w_D \cdot \widetilde{D}(a) - w_C \cdot \widetilde{C}(a)$$
> 
> Notice that $S(a)$ is a **signed dimensionless composite scalar**. Only the raw Expected Hypothesis Information Gain carries the unit of *nats*.
> 
> At Step 1, the engine selected `controlled-2` with **XRD characterization**. Why? Because XRD yields 0.506 nats of information gain at half the cost of a full outcome test. It maximizes scientific discovery per dollar of laboratory budget.
> 
> If you inspect the 3D Hologram, you see the 12 authentic candidate materials mapped across the discovery sphere, surrounded by an emerald orbital ring highlighting our selected action, while decorative lattice points are honestly marked without fake Pareto flags."

---

### Scene 4: The Scientific Wow Moment (Preregister → Reveal → Update) (3:45 – 5:15)
**On Screen:** *Workspace 1: Discovery Lab — The 4-State Preregistration & Replay Banner*  
**Visual Action:** Click through the four sequence states:
1. State A: Scored
2. State B: Preregistered (Lock Preregistration)
3. State C: Evidence Reveal (Reveal Observation)
4. State D: Belief Shift (Execute Belief Update)

> "Now, here is the central scientific interaction: **The 4-State Preregistration & Bayesian Loop**.
> 
> In traditional machine learning for materials, retrospective evaluations suffer from severe hindsight bias. AIcoScientist enforces a cryptographic protocol:
> 
> **State A (Scored):** Hypotheses are scored, predictive distributions are generated, but observations are strictly firewalled.
> 
> *(Click State B: Preregistered)*
> **State B:** The candidate, modality, and predictive distributions are locked into an immutable evidence ledger event with an exact timestamp. The prediction is sealed.
> 
> *(Click State C: Evidence Reveal)*
> **State C:** The physical measurement is revealed. Notice the vertical blue line on our Predictive Distribution chart: the real observed value is plotted against the firewalled curves.
> 
> *(Click State D: Belief Shift)*
> **State D:** The Bayesian posterior update executes. Watch the belief bars shift: $H_1$ Phase Purity jumps from 33.3% to 68.2%, log Bayes factors are recorded, entropy decreases by 0.42 nats, and the next step is queued.
> 
> Every transition in this sequence is recorded in our 5,333-event audit ledger."

---

### Scene 5: Benchmark Validation: Clean vs Stress Worlds (5:15 – 6:30)
**On Screen:** *Workspace 2: Evidence & Benchmarks (`/benchmarks`)*  
**Visual Action:** Click **Q1 (Controlled Clean Worlds)**, then **Q3 (180 Trajectory Benchmark)**. Toggle metrics between MAP Recovery Rate, Final Posterior, and Steps to Posterior > 0.8.

> "A critical advisor question is: *'Does this active decision framework actually outperform standard policies?'*
> 
> In our **Evidence & Benchmarks** workspace, we organize results around five fundamental research questions:
> 
> Under **Q1 (Clean Worlds)**: When observation models match true system physics, AIcoScientist achieves **100% MAP hypothesis recovery** across all 30 clean trajectories, reaching high posterior confidence ($P > 0.8$) on the very first characterization step.
> 
> Under **Q3 (180-Trajectory Multi-Policy Benchmark)**: We compared 6 distinct decision policies across 6 simulated worlds and 5 seeds:
> - `Pure HIG` recovers the true hypothesis fastest (mean 1.2 steps), but completely ignores material utility.
> - `Discovery Only` behaves like standard Bayesian optimization—finding high-utility materials but failing to disambiguate underlying mechanisms.
> - `HYBRID` achieves an optimal trade-off: 100% MAP hypothesis recovery while bounding experimental cost.
> 
> Under **Q2 (Stress Worlds)**: We subjected the engine to severe noise, prior misspecification, and unmodeled distortion, verifying graceful degradation rather than catastrophic failure."

---

### Scene 6: Real A-Lab Evidence, Electrolyte Scale & System Governance (6:30 – 8:00)
**On Screen:** *Workspace 2 (Q4 & Q5) and Workspace 3: Research System (`/system`)*  
**Visual Action:** Select **Q4 (1,035 A-Lab Physical Samples)**, click sample `PG_0309`, show calibration coverage. Switch to Workspace 3 (`Research System`), show **48/50 Validation Gates** and **SHA-256 Manifest**.

> "Finally, let us examine real physical data, massive scalability, and system governance:
> 
> Under **Q4**: We evaluated the framework on the **A-Lab Precursor Genome**—1,035 authentic inorganic synthesis experiments. You can search any sample: selecting `PG_0309` reveals its genuine formula $Co_3B_3H_9O_{13}$, precursor chemistry, 200°C heating schedule, and Rietveld $R_{wp}$.
> 
> Notice our scientific honesty:
> 1. SEM and EDS modalities lack candidate sample ID linkage in the published dataset. Rather than hallucinating synthetic linkages, we honestly mark them **NOT AVAILABLE**.
> 2. Our calibration chart displays **A_LAB_CALIBRATION_PARTIAL**: the 50% credible interval covers 95.2% of data points, showing that our model is conservatively over-dispersed rather than falsely overconfident.
> 
> Under **Q5**: At scale, our Stage-1 screening filtered **333,333 virtual electrolyte candidates** down to 200 in **2.535 seconds** with zero latent regret. We report an honest negative result: while standard BoTorch EI achieves lower single-property regret on the ExtraTrees surrogate oracle, our Hybrid policy achieves nearly double the mechanistic entropy reduction.
> 
> In **Workspace 3 (Research System)**, our governance dashboard reports **48 out of 50 Boolean validation gates passed**, backed by complete SHA-256 cryptographic manifest verification. The 2 unpassed gates reflect our documented scientific boundaries: conservative refinement variance and cross-family chemistry transfer.
> 
> AIcoScientist demonstrates a complete, principled shift from heuristic trial-and-error to autonomous, evidence-driven scientific discovery.
> 
> Thank you, and I look forward to your questions."

---

## Advisor FAQ Preparation

### Q1: "Why not just use standard Bayesian Optimization (e.g. BoTorch EI)?"
**Answer:**  
Standard BO is univariate and blind to physical mechanism. It assumes every measurement action is an identical property test. In real experimental science, you have cheap diagnostic measurements (XRD, Raman) and expensive destructive tests (cycling, synthesis). AIcoScientist jointly selects *which material* and *which measurement modality*, optimizing information gain per unit of experimental budget.

### Q2: "What exactly are the 5,333 ledger events?"
**Answer:**  
They are immutable audit records in `outputs/alab/multimodal/evidence_ledger.jsonl`. Each record logs an action scoring, a preregistration, an observation reveal, or a Bayesian belief update. They verify that predictions were recorded strictly prior to data reveal across all 180 benchmark trajectories. They are audit events, not 5,333 wet-lab physical syntheses.

### Q3: "Why did you mark SEM/EDS as NOT AVAILABLE in A-Lab?"
**Answer:**  
Because in the published Precursor Genome dataset, SEM and EDS raw archives are grouped by precursor formulation, but do not contain sample-level ID metadata linking them directly to the 1,035 individual synthesis pellets. Rather than guessing or hallucinating synthetic links, our domain adapter enforces a strict contract: unlinked modalities are marked NOT AVAILABLE.
