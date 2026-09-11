# UI Comprehension Audit

## Current problem

The previous interface was scientifically careful but led with implementation language: source score provenance, posterior telemetry, and verification machinery. A first-time researcher had to infer the product story from several panels. The decision matrix—the distinctive candidate × measurement selection problem—appeared after a secondary trajectory panel, while an optimization result was not visually separated from the next-experiment recommendation.

## User confusion addressed

| Question | Before | After |
| --- | --- | --- |
| What does it select? | Implied by score cards and controls | `NEXT EXPERIMENT` explicitly names the candidate × measurement action. |
| Why that action? | Internal score fields appeared first | The matrix is framed as all feasible experiments; its selected cell is the recorded choice. |
| Is it live? | Replay limits were present but diffuse | Evidence mode remains visible in the hero copy and result labels. |
| What happens after selection? | Four implementation states | Choose → Lock prediction → Reveal evidence → Update model support. |
| Where is the best material? | Could be confused with the selected action | The surrogate view has a separate `BEST CANDIDATE FOUND SO FAR` block. |

## Story and hierarchy

1. **Next experiment** — the recorded candidate × measurement action, its score, information value, and cost.
2. **Why this experiment** — the candidate × modality matrix; each cell is a feasible experiment.
3. **What was expected** — predictive distributions explain where models disagree before observation.
4. **What was learned** — a locked prediction, revealed evidence, and posterior model-weight update.
5. **Does it work?** — controlled, historical replay, and surrogate benchmark evidence.

Primary information is the next-experiment card, the decision matrix, and the four-state evidence loop. Predictive, trajectory, trade-off, and score views support that narrative. Calibration, sensitivity, provenance, event ledger, alternative actions, and the presentation-only candidate layout remain advanced evidence.

## Graph hierarchy

| Graph | Role | Plain-language job |
| --- | --- | --- |
| Candidate × Modality Decision Matrix | Core story | Shows every feasible experiment and the recorded selection. |
| Predictive Distribution | Supporting | Shows what each model expected before the observation. |
| Hypothesis Belief Trajectory | Supporting | Shows how evidence changes model support over time. |
| Recorded Decision Score | Supporting | Shows persisted information, discovery, cost, and total-score fields. |
| Trade-off Scatter | Advanced | Lets researchers inspect information-versus-cost alternatives. |
| Policy Trajectory | Supporting evidence | Compares strategies under the same controlled benchmark. |
| Electrolyte Optimization Trajectory | Core for optimization | Shows best-found value across recorded surrogate queries. |
| Calibration / MC sensitivity | Advanced methodology | Tests uncertainty calibration and estimator stability. |
| Hologram candidate-space view | Optional exploration | Presentation-only layout; not scientific geometry. |
| Attia / FeCoNi figures | Supporting benchmark evidence | Need a dedicated source-backed evidence route before being promoted in the main UI. |

## Terminology fixes

- **Information gain (HIG):** expected reduction in uncertainty about which model best explains the system.
- **Preregister:** record the expected outcome before revealing the measurement.
- **Posterior model weight:** how much the current evidence favors a model; not proof of a physical mechanism.
- **Surrogate:** a frozen computational model used in place of a new physical experiment.
- **Latent value / regret:** source-model value and its gap to the surrogate reference optimum.

## Before → after rationale

The visual direction stays Scientific Editorial / Analytical Instrumentation: restrained crimson, off-white field, precise typography, and charts over decoration. The change is an information-architecture correction, not a theme redesign. The memorable anchor is now the explicit **candidate × measurement** decision matrix instead of a decorative candidate-space visualization.

The applied direction scores 13/15 on the design-feasibility index: high context fit and feasibility, with low consistency risk because it reuses existing cards, typography, and chart components.
