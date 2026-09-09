# AIcoScientist — Advisor Demonstration Script (6–8 Minute Talk Track)

**Target Audience:** Academic Advisor / Research Committee  
**Goal:** Establish that AIcoScientist is a mathematically rigorous, multi-hypothesis scientific decision engine, not just a black-box property optimizer.  
**Hotkeys:** Press `P` to enter Presenter Mode. Use `Right Arrow` to advance scenes, and `N` to toggle speaker notes.

---

## Timeline & Talk Track

```text
0:00 — 1:00 | Scene 1: The Research Question
1:00 — 2:00 | Scene 2: Universal Architecture & Decision Loop
2:00 — 3:30 | Scene 3: Decision Cockpit & Competing Hypotheses
3:30 — 5:00 | Scene 4: The Scientific Wow Moment (Preregister → Reveal → Update)
5:00 — 6:00 | Scene 5: Evaluation Breadth (180 Trajectories)
6:00 — 7:00 | Scene 6: Real A-Lab Evidence & Scientific Boundaries
7:00 — 7:45 | Scene 7: Electrolyte Screening Scale & Negative Result
7:45 — 8:30 | Scene 8: Research Contributions & Prospective Next Steps
```

---

### Scene 1: The Research Question (0:00 – 1:00)
**On Screen:** *Tab 1: Research Overview (`/overview`)*  
**Visual Action:** Point to the main headline: *"From property optimization to evidence-driven scientific decisions."*

> "Good morning, Professor.
> 
> Most machine learning in materials science asks a single question: *'Which material is predicted to have the highest property value?'*
> 
> But in a real laboratory, physical experiments are costly, irreversible, and multi-modal. A scientist does not merely measure final device performance—they choose characterization tools like XRD or Rietveld refinement to determine *why* a reaction succeeded or failed, confirming or falsifying competing mechanistic hypotheses.
> 
> AIcoScientist is an autonomous decision framework that asks: **'Which experiment should we perform next, and why?'**
> 
> As you can see on this overview screen, our methodology is built around a 9-stage closed loop, validated across 1,035 real inorganic synthesis samples, 180 controlled policy trajectories, and 5,333 auditable evidence ledger records. Today, I'll walk you through how our system makes these decisions."

---

### Scene 2: Universal Architecture (1:00 – 2:00)
**On Screen:** *Tab 6: Architecture & Ledger (`/architecture`)*  
**Visual Action:** Hover over `MaterialDomainAdapter` and `MultimodalDecisionEngine`.

> "The first key research contribution is that this framework is **domain-agnostic**.
> 
> Look at the core abstractions: `MaterialDomainAdapter` cleanly separates the candidate chemistry representation from the active inference engine. 
> 
> Whether we are synthesizing solid-state ceramics, screening 333,000 battery electrolyte formulations, or exploring thin-film electrocatalysts, the exact same mathematical core enumerates joint actions, computes Expected Hypothesis Information Gain in nats, locks preregistrations, and executes Bayesian updates.
> 
> This is a reusable scientific operating system, not a one-off chemistry script."

---

### Scene 3: Decision Cockpit & Competing Hypotheses (2:00 – 3:30)
**On Screen:** *Tab 2: Decision Cockpit (`/cockpit`)*  
**Visual Action:** Click on the three zones: Left (Hypotheses), Center (Candidate Space), Right (Hero Recommendation).

> "Now let's step into the **Decision Cockpit**.
> 
> On the left, notice the **Hypothesis Observatory**. Instead of fitting a single black-box model, AIcoScientist maintains three competing mechanistic hypotheses:
> 1. $H_1$: Phase Purity Limited
> 2. $H_2$: Composition Homogeneity Limited
> 3. $H_3$: Morphology Kinetics Limited
> 
> Note our explicit disclaimer: these weights represent *relative explanatory model likelihoods* among simplified competing models, not a physical claim that one simplified theory is absolute truth.
> 
> In the center is the candidate space. Notice that measurements have physical costs and dependencies: Rietveld refinement requires a prior XRD scan, and synthesis outcome testing consumes the physical specimen.
> 
> On the right is our **Hero Recommendation**. The engine chose not only candidate `Syn-C`, but specifically selected **XRD characterization**. 
> 
> Look at the waterfall score breakdown:
> $$\text{Score} = w_{\text{hig}} \cdot \text{HIG} + w_{\text{disc}} \cdot \text{Discovery} - w_{\text{cost}} \cdot \text{Cost}$$
> It chose XRD because it yields 0.506 nats of expected information gain at half the cost of an outcome test, maximizing scientific insight per dollar spent."

---

### Scene 4: The Scientific Wow Moment (3:30 – 5:00)
**On Screen:** *Tab 2: Decision Cockpit Banner (Preregister → Reveal → Update)*  
**Visual Action:** Click through the sequence buttons:
1. State A (Scored) → Click **"Lock Preregistration (State B)"**
2. State B (Locked) → Click **"Reveal Observation (State C)"**
3. State C (Reveal) → Click **"Execute Belief Update (State D)"**

> "Now, here is the central interaction that separates AIcoScientist from standard AI demos: **The Preregister → Reveal → Update Loop**.
> 
> In conventional research, models suffer from hindsight bias. In AIcoScientist:
> 
> **State A:** The action is selected and predictions are generated, but the observation is strictly firewalled.
> 
> *(Click Lock Preregistration)*
> **State B:** The prediction distributions and falsification criteria are committed to an immutable ledger event with a cryptographic timestamp. The prediction is locked.
> 
> *(Click Reveal Observation)*
> **State C:** The physical or canonical observation is revealed. Notice we display real canonical descriptors and refinement phase fractions—zero synthesized traces.
> 
> *(Click Execute Belief Update)*
> **State D:** Watch the belief bars update smoothly. Because the target phase fraction exceeded 0.94, $H_1$ Phase Purity posterior jumps from 33% to 68%. The log Bayes factor confirms strong diagnostic evidence, entropy is reduced by 0.42 nats, and the next action is immediately queued.
> 
> Every step of this chain is permanently auditable."

---

### Scene 5: Evaluation Breadth (5:00 – 6:00)
**On Screen:** *Tab 4: Policy Benchmark Lab (`/benchmarks`)*  
**Visual Action:** Switch world to `STRESS_WORLD_H1_PHASE_PURITY`, toggle between `HYBRID` and `PURE_HIG`.

> "A common advisor question is: *'Does this actually outperform standard policies?'*
> 
> Here in the Benchmark Laboratory, we evaluated **180 full closed-loop trajectories** across 6 policies, 6 worlds, and 5 random seeds.
> 
> The scientific story is that different policies optimize distinct objectives:
> - `Pure HIG` recovers the true hypothesis the fastest (mean 1.2 steps), but completely ignores material utility.
> - `Discovery Only` acts like standard Bayesian optimization—it finds high-utility materials but struggles to distinguish mechanisms.
> - `HYBRID` achieves a proven compromise: 100% MAP hypothesis recovery with bounded experimental expenditure.
> 
> Below, our Monte Carlo sensitivity analysis comparing 12 vs 32 samples explains why we chose MC=32: MC=12 had an 80% rank correlation but introduced action-ranking jitter. We eliminated that jitter with MC=32."

---

### Scene 6: Real A-Lab Evidence & Scientific Boundaries (6:00 – 7:00)
**On Screen:** *Tab 3: A-Lab Evidence Atlas (`/alab`)*  
**Visual Action:** Select `PG_0309` (LiNi0.5Mn1.5O4), point to the Modality Table, then scroll to Calibration.

> "To test real-world fidelity, we evaluated AIcoScientist on the **A-Lab Precursor Genome**—1,035 real solid-state synthesis experiments from Berkeley and Zenodo.
> 
> Here is where our scientific rigor is on full display:
> - For 1,030 samples, canonical XRD scans and Rietveld refinements are linked and replayable.
> - But notice rows 4 and 5: SEM and EDS archives exist at the precursor level, but lack candidate sample ID linkage. Rather than faking linkage, our system explicitly flags them as **NOT AVAILABLE** and excludes them from replay.
> 
> Furthermore, look at our calibration table: Refinement phase fraction coverage is marked **A_LAB_CALIBRATION_PARTIAL** because the 50% interval covers 95.2% of points. The model is conservative and over-dispersed, rather than overconfident.
> 
> And our holdout evaluation honestly states that out-of-family elemental generalization is **NOT ESTABLISHED**. We present these boundaries as evidence of genuine scientific integrity."

---

### Scene 7: Electrolyte Screening Scale & Negative Result (7:00 – 7:45)
**On Screen:** *Tab 5: Electrolyte Discovery (`/electrolyte`)*  
**Visual Action:** Trace the 5-stage screening funnel from 333,333 to WS=200.

> "At massive scale, we applied this to high-entropy battery electrolytes.
> 
> Our Stage-1 screening filters a combinatorial space of **333,333 virtual LiFSI candidates** down to a 200-candidate working set in **2.535 seconds**, recovering 100% of the latent maximum with zero screening gap.
> 
> In closed-loop surrogate simulation, we report an honest trade-off: BoTorch Expected Improvement achieves lower regret for single-property exploitation (0.0257 vs 0.0788), but Hybrid achieves nearly double the scientific entropy reduction (0.995 vs 0.564 nats). We prioritize learning the system over single-point curve fitting."

---

### Scene 8: Contributions & Research Readiness (7:45 – 8:30)
**On Screen:** *Tab 7: Research Readiness (`/readiness`)*  
**Visual Action:** Point to the **48/50 Boolean Validation Gates Passed** header, and show the 2 failing gates.

> "In conclusion:
> 
> Our research readiness screen reports **48 out of 50 Boolean validation gates passed**. 
> 
> I want to emphasize that 48/50 is *not* a model accuracy metric. It represents passing 48 strict software and statistical gates, including zero HIG bound violations and 100% pre-reveal preregistration guarantees.
> 
> The 2 failing gates are not software bugs—they are our documented scientific frontiers: conservative refinement variance and cross-family chemistry transfer.
> 
> AIcoScientist proves that we can formalize the scientific method into an autonomous, multi-modal decision loop with complete provenance.
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
