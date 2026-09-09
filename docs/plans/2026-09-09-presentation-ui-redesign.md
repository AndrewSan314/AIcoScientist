# AIcoScientist Presentation UI Redesign Plan

**Status:** Ready for user approval  
**Target:** Desktop advisor demo  
**Implementation stack:** Existing React 19 + Vite 8 + Tailwind CSS 4 + Recharts  
**Implementation agent:** Luna, `xhigh`

## 1. Outcome

Turn the current static-looking dashboard into a believable product demonstration:

1. The user chooses a real, already-supported dataset.
2. The user reviews a small set of meaningful engine settings.
3. The user starts an analysis/replay and sees clear progress.
4. Results appear only after the run completes.
5. Multiple graphs explain both the selected action and the engine's validated advantages.

The redesign must remain scientifically honest. Official results come from the deterministic snapshot and must be described as validated replays or benchmark evidence, not as newly computed live experiments.

## 2. Scope and non-goals

### In scope

- Desktop UI and information hierarchy.
- Discovery Lab workflow and interaction state.
- Navigation clarity.
- Presenter Mode reliability and story flow.
- Visual redesign using the existing stack.
- Reuse and restyling of the existing chart components.
- Removal of visible version, commit, source-path, and raw-developer metadata.
- Loading, empty, success, and recoverable error states for the new workflow.

### Out of scope

- Responsive/mobile redesign.
- Framework migration.
- New charting, animation, icon, state-management, or font dependencies.
- Uploading arbitrary user datasets.
- Re-running expensive scientific benchmarks in the browser.
- Changing scientific algorithms, benchmark outputs, or validated claims.
- Pretending that precomputed snapshot results are live computation.

## 3. Current problems to fix

- The first screen immediately exposes final results, so it looks like a static report instead of a usable research product.
- Three top-level workspaces contain additional tab systems: Discovery has chart and explorer switchers, Evidence has five large question cards, and Research System has four more tabs.
- Most sections use the same large rounded white-card treatment, so primary and secondary information compete equally.
- Heavy monospace, uppercase labels, pills, pale emerald surfaces, Lucide icons, and `rounded-3xl` containers create a generic AI-dashboard appearance.
- The top navigation does not explain what each workspace helps the user accomplish.
- Presenter Mode is only discoverable through the `P` shortcut because the current Header ignores its presenter-related props.
- Pressing `P` can show Scene 1 while leaving a different workspace underneath it.
- Presenter scene changes can fail to update the active benchmark question because `initialQuestionId` is only used as the initial React state.
- Current documentation describes eight scenes while the code contains six.
- Scene 6 combines A-Lab, electrolyte scaling, and governance but only routes to benchmark question 4.
- SHA, branch/version-like metadata, raw JSON access, source file paths, hard-coded footer counters, and the provenance manifest distract from the teaching narrative.

## 4. Chosen design direction: Scientific Editorial Console

The product should feel like a contemporary scientific publication that can be operated, rather than a futuristic AI cockpit.

### Visual language

- Canvas: warm off-white `#F4F3EE`.
- Primary surface: soft white `#FCFCFA`.
- Primary ink: charcoal-green `#17201F`.
- Secondary ink: `#66706C`.
- UI accent: mineral teal `#0F766E`.
- Strong accent: `#0A5C56`.
- Hairline border: `#D9DFDB`.
- Warning: muted ochre `#A46B1F`.
- Error: muted brick `#A8483D`.
- Chart series may use a restrained, color-blind-conscious palette; the one-accent rule applies to navigation and UI chrome, not to scientific series that require categorical distinction.

### Typography

- Use a native offline-safe stack: `"Segoe UI Variable", "Aptos", "Helvetica Neue", Arial, sans-serif`.
- Use monospace only for measurements, formulas, identifiers, and table numbers.
- Use sentence case for headings and labels.
- Increase body readability; eliminate `text-2xs`/`text-3xs` from important explanatory content.
- Use tabular numerals for metrics.

### Shape and depth

- Page sections use whitespace and thin rules before containers.
- Reserve surfaces for charts, selectors, and interactive inspection.
- Main containers: 14–16px radius; controls: 8–10px radius.
- Remove `rounded-3xl` repetition and almost all generic shadows.
- No neon, outer glow, glassmorphism, gradient text, 3D decoration, or perpetual animation.
- Keep motion functional: 180–240ms transitions for selection, reveal, progress, and chart changes.
- Honor `prefers-reduced-motion`.

## 5. Information architecture

Rename the three primary destinations so their purpose is visible immediately:

| Current | New label | Supporting text |
|---|---|---|
| Discovery Lab | **Run discovery** | Choose data, run the engine, inspect its decision |
| Evidence & Benchmarks | **Evidence** | Compare policies, robustness, real-data replay, and scale |
| Research System | **How it works** | Architecture, audit trail, and verification |

Header changes:

- Keep the AIcoScientist wordmark.
- Remove numeric hotkey badges from the main navigation.
- Add a visible `Start guided demo` action.
- Keep keyboard shortcuts in Presenter Mode or speaker notes, not in persistent chrome.
- Do not display branch, SHA, version, generated timestamp, or status jargon in the header/footer.

## 6. Discovery Lab product workflow

Use a three-state screen controlled by explicit React state: `setup`, `running`, and `results`.

### State A — Set up a run

No final chart or result is visible on initial entry.

Show a compact four-step progress label: `Select data → Configure → Run → Review evidence`.

Dataset choices must map to data already present in `snapshot.json`:

1. **Controlled synthesis world** — flagship multimodal hypothesis-recovery campaign.
2. **A-Lab historical replay** — 1,035 physical synthesis records and related calibration/sample evidence.
3. **Electrolyte virtual search** — large-pool screening and surrogate closed-loop evidence.

Each choice shows only: dataset name, evidence type, candidate count, available modalities, and what question it answers. Do not show paths, versions, hashes, or raw JSON.

Configuration:

- Policy selector with `HYBRID` as the recommended default.
- Seed selector only when the selected evidence actually supports it.
- Put HIG/discovery/cost weights inside a collapsed `Advanced settings` disclosure.
- Show a one-sentence explanation of the selected policy.
- Primary CTA: `Run analysis`.

### State B — Running

Use a restrained inline progress sequence rather than a spinner-only page:

1. Preparing selected evidence.
2. Building candidate × measurement action space.
3. Scoring information gain, discovery value, and cost.
4. Preparing recommendation and evidence views.

The interface may impose a short minimum display time so the state change is legible, but copy must say `validated replay` for snapshot-backed results. Do not claim that the expensive benchmark was recomputed.

Provide `Cancel` or `Back to setup`. Disable duplicate run clicks.

### State C — Results

Lead with one clear answer:

- Recommended candidate and measurement.
- One-sentence explanation.
- Total score with HIG, discovery, and cost components.
- `Why this action?` and `Why not the alternative?` text when supplied by data.
- Actions: `Run another analysis` and `Continue guided story`.

Then reveal the existing scientific content through progressive disclosure, not one long wall of cards.

## 7. Graph strategy

The UI should contain many useful graphs without showing all of them at once.

### Discovery result story

Reuse and restyle these existing components:

1. `HypothesisBeliefTrajectoryChart` — demonstrates how evidence shifts posterior beliefs.
2. `PredictiveDistributionChart` — shows how competing hypotheses predict different observations.
3. `TradeoffScatterChart` — explains the information/discovery/cost trade-off among actions.
4. `ScoreWaterfallChart` — explains why the selected action wins.
5. `CandidateModalityHeatmap` — proves that the engine searches candidate × measurement actions rather than ranking materials only.

Default presentation order:

- Hero result and score explanation.
- Belief trajectory as the primary graph.
- Two-column evidence row: predictive distributions + action trade-off.
- Full-width candidate × modality heatmap.
- Detailed counterfactual table behind `Inspect all actions`.

Remove `3D Stark Hologram` from the primary navigation. It may remain as a secondary experimental view only if it does not compete with the scientific graphs.

### Evidence workspace

Replace the five large question cards with a compact Q1–Q5 question rail. Every question begins with a plain-language `What this proves` statement and then its strongest visualization.

- Q1: inference recovery — policy trajectory/recovery comparison.
- Q2: robustness — clean vs stress worlds and MC-sensitivity evidence.
- Q3: policy advantage — policy comparison curves/table with HYBRID highlighted only where the data supports it.
- Q4: real-data evidence — calibration coverage, A-Lab replay, and sample/XRD inspector.
- Q5: scale — screening funnel, candidate-pool scale/runtime, and electrolyte optimization trajectory.

Graph rules:

- Use consistent axes, gridlines, tooltip layout, legend order, and numeric formatting.
- Prefer direct series labels when possible.
- Put a one-sentence interpretation below every graph.
- Clearly distinguish `controlled`, `historical replay`, `surrogate`, and `physical measurement` evidence.
- Preserve negative results and limitations. Demonstrate the engine's advantages through comparisons; do not rewrite qualified results into universal superiority claims.

## 8. How it works workspace

Reduce four navigation items to three:

1. **Architecture** — decision loop and reusable scientific abstractions.
2. **Audit trail** — preregistration and evidence ledger.
3. **Verification** — the 50-gate matrix, with failing gates clearly visible.

Remove the cryptographic provenance manifest from primary navigation. If reproducibility detail must remain accessible, place it in a small `Method details` disclosure at the end of Audit Trail.

Remove source file paths and `Protocol: Standard Interface Ready` repetition from cards. Replace the six equal architecture cards with one readable pipeline diagram plus a concise domain registry.

## 9. Presenter Mode

Add a visible `Start guided demo` button and use seven synchronized scenes:

1. Research question and dataset selection.
2. Configure and run the engine.
3. Recommended experiment and score decomposition.
4. Preregister → reveal → belief update.
5. Policy and robustness evidence.
6. A-Lab physical-data replay and calibration.
7. Electrolyte scale evidence and system verification.

Required behavior fixes:

- Starting Presenter Mode by button or `P` must call the same handler and synchronize workspace, dataset, run state, question, chart view, and campaign step.
- Scene metadata must carry the full target state, not only `workspace` and an occasionally used `questionId`.
- Evidence and Discovery components must react when presenter-controlled props change.
- Update speaker notes and documentation to exactly seven scenes.
- Keep overlay copy short enough not to truncate at the target desktop viewport.

## 10. Remove visual and developer noise

Remove from the presentation UI:

- Snapshot version.
- Generated timestamp.
- Branch and commit SHA.
- Footer SHA and hard-coded counters.
- Raw benchmark JSON button/drawer.
- Source file paths.
- Repeated version/status badges.
- Numeric hotkey badges in the main nav.
- The provenance manifest as a top-level subtab.
- Unnecessary uppercase labels and decorative pills.

Keep evidence-type labels only where they prevent scientific misinterpretation, such as `Controlled benchmark`, `Historical replay`, `Surrogate`, and `Physical measurement`.

## 11. Implementation sequence

### P0 — Functional demo flow

1. Add the `setup → running → results` state machine to Discovery Lab.
2. Add supported dataset selection and compact configuration.
3. Gate all current result views behind completion of `Run analysis`.
4. Map each dataset choice to the appropriate existing snapshot sections.
5. Add loading, cancellation/back, no-data, and recoverable error states.
6. Preserve truthful evidence-mode labels.

### P0 — Navigation and Presenter Mode

1. Rename and clarify top-level navigation.
2. Restore a visible Presenter Mode entry point.
3. Fix keyboard launch and scene synchronization.
4. Fix controlled benchmark-question synchronization.
5. Replace the current scene configuration with the seven-scene story.

### P1 — Visual system

1. Introduce the Scientific Editorial Console tokens in `index.css`.
2. Reduce card, pill, shadow, all-caps, and monospace overuse.
3. Restyle page headers, controls, tables, empty states, and feedback.
4. Apply the same graph typography, palette, grids, tooltips, and captions across all chart components.

### P1 — Workspace simplification

1. Convert Evidence Q1–Q5 cards into a compact question rail.
2. Reorder graphs around proof statements.
3. Reduce How it works to Architecture, Audit trail, and Verification.
4. Remove/hide developer and version metadata listed above.

### P1 — Verification

1. Run build and lint.
2. Run the existing presentation backend/UI verification.
3. Exercise the complete desktop workflow for all three datasets.
4. Exercise all seven Presenter Mode scenes with button and keyboard navigation.
5. Capture desktop screenshots for setup, running, results, Evidence, How it works, and Presenter Mode.
6. Have a fresh reviewer perform a rendered usability and visual audit; fix blockers and major findings before completion.

## 12. Expected files

Prefer the smallest coherent diff and reuse existing components. Expected touch points:

- `presentation/frontend/src/App.tsx`
- `presentation/frontend/src/index.css`
- `presentation/frontend/src/types/mission_control.ts`
- `presentation/frontend/src/components/Header.tsx`
- `presentation/frontend/src/components/PresenterMode.tsx`
- `presentation/frontend/src/components/SpeakerNotesModal.tsx`
- `presentation/frontend/src/views/DiscoveryLabWorkspace.tsx`
- `presentation/frontend/src/views/EvidenceBenchmarksWorkspace.tsx`
- `presentation/frontend/src/views/ResearchSystemWorkspace.tsx`
- Existing files under `presentation/frontend/src/components/charts/`
- `presentation/frontend/capture_screenshots.js`
- `ADVISOR_DEMO_SCRIPT.md`
- `PRESENTATION_DEMO.md`

Do not add new components merely to create abstractions. Split only if the Discovery workflow becomes harder to review in its current file.

## 13. Acceptance criteria

- Opening Run discovery shows dataset selection, not completed results.
- The user can select each of the three supported evidence sources and start a run.
- The running state clearly communicates progress and prevents duplicate execution.
- Results appear only after completion and change according to the selected dataset/evidence source.
- At least five existing scientific graph types are accessible from the Discovery/Evidence story.
- Every major graph includes a clear interpretation and evidence-type context.
- The UI does not claim live computation when displaying snapshot-backed results.
- Top-level navigation is understandable without documentation.
- No visible version, SHA, branch, generated timestamp, raw JSON, or source path remains on the main presentation surfaces.
- Presenter Mode is visible, uses seven scenes, and always opens the correct state.
- Scientific limitations, negative results, and failing gates remain visible.
- Existing data flows and scientific outputs are not modified.
- `npm run build`, `npm run lint`, and `python presentation/scripts/verify_ui.py` pass.
- The redesigned desktop screenshots contain no blocker or major usability findings after fresh review.

## 14. Agent implementation prompt

```text
You are implementing the approved AIcoScientist presentation UI redesign in:

F:\AI\GTIP

Read and execute this plan completely:

F:\AI\GTIP\docs\plans\2026-09-09-presentation-ui-redesign.md

Goal:
Transform the existing React presentation from a static, crowded dashboard into a desktop-first interactive product demo. Run discovery must start with selecting one of the supported data/evidence sources, configuring the run, clicking Run analysis, seeing progress, and only then revealing validated results and scientific graphs. Apply the Scientific Editorial Console style defined in the plan. Remove visible version/developer noise. Preserve scientific honesty and all existing verified data.

Mandatory workflow:

1. Read F:\AI\GTIP\AGENTS.md and follow it.
2. Read the using-superpowers skill, then the redesign-existing-projects, design-taste-frontend, and design-ux skills before acting.
3. Inspect git status and preserve all user changes. You are not alone in the repository; do not revert unrelated work.
4. Use Semble as the first search for every unfamiliar implementation area. Use GitNexus for call/context relationships.
5. Before editing every function, class, React component, or handler, run GitNexus upstream impact analysis for that symbol. Report HIGH or CRITICAL risk before changing it.
6. Reuse the existing React, Tailwind, Recharts, snapshot data, components, and backend. Add no dependencies unless absolutely required; the approved plan expects none.
7. Do not work on responsive/mobile behavior. Desktop presentation is the target.
8. Do not change scientific algorithms, benchmark files, result values, or qualified scientific claims.
9. Never present snapshot-backed benchmark output as newly computed live results. Use truthful labels such as Controlled benchmark, Historical replay, Surrogate, Physical measurement, or Validated replay.
10. Implement every P0 and P1 item in the plan. Do not stop after restyling; the Discovery setup → running → results flow and Presenter Mode fixes are required.

Implementation constraints:

- Use the existing dataset sections in snapshot.json:
  - flagship_campaign / benchmarks for controlled synthesis evidence;
  - alab_replay_campaign / alab_audit / samples for A-Lab replay;
  - electrolyte_screening / electrolyte_simulation for electrolyte evidence.
- Keep HYBRID as the recommended default while preserving other valid policies.
- Put detailed weights and seed controls behind progressive disclosure.
- Results must remain hidden until Run analysis completes.
- Reuse and visually unify the existing graph components before creating new graphs.
- Keep multiple graphs, but reveal them in a narrative order so the page remains readable.
- Remove the 3D hologram from primary navigation unless it clearly supports a secondary scientific inspection use case.
- Remove visible version, SHA, branch, generated timestamp, raw JSON, source paths, redundant status badges, and footer telemetry.
- Reduce Research System navigation to Architecture, Audit trail, and Verification; move essential provenance detail into a secondary Method details disclosure.
- Add a visible Start guided demo button.
- Presenter Mode must use the seven scenes in the plan and synchronize all required UI state regardless of whether it starts by button or keyboard.
- Preserve keyboard access, focus visibility, honest failure states, and prefers-reduced-motion.

Verification:

1. Run npm run build in presentation/frontend.
2. Run npm run lint in presentation/frontend.
3. Run python presentation/scripts/verify_ui.py from the repository root.
4. Start the real presentation server and verify the desktop UI in the browser.
5. Test all three dataset flows, including setup, running, results, rerun, and error/no-data handling.
6. Test all seven Presenter Mode scenes through both controls and keyboard shortcuts.
7. Update capture_screenshots.js and capture the six critical desktop states listed in the plan.
8. Run GitNexus detect_changes on all uncommitted changes and verify only expected symbols and flows are affected.
9. Perform a fresh rendered usability/visual review after implementation. Fix every blocker and major finding.

Do not commit unless the user explicitly asks. Finish with a concise report listing changed files, verification commands/results, scientific-behavior guarantees, and any genuinely unresolved issue.
```

## 15. Decision log

- Desktop-only scope was chosen because tomorrow's advisor demo is the immediate goal.
- Scientific Editorial Console was chosen over dark cockpit and generic SaaS styles because it is distinctive, readable, and appropriate for academic evidence.
- Discovery becomes an operable workflow rather than an immediately populated report.
- Deterministic replay is the default because official evidence is snapshot-backed; the existing live endpoint is only a toy diagnostic and must not be presented as the official run.
- Existing graphs are reused to minimize risk and preserve verified data bindings.
- Version/developer metadata is removed from presentation surfaces because it does not support the demo narrative.
- Many graphs remain, but progressive disclosure prevents them from recreating the current information overload.
