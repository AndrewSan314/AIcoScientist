# AIcoScientist exhibition — implementation and verification report

Reviewed 2026-09-21. Status: implementation complete and ready for project review. This report replaces earlier unmeasured performance and scientific claims.

## Delivered experience

- A persistent Three.js canvas carries the visitor through the cleanroom, equipment, authored electrode microstructure, source-backed sequential replay and evidence scenes without page reloads.
- Original Blender-authored calender and coater GLBs replace the flagship primitive assemblies at runtime. PMREM studio reflections, shadow lighting and oblique operating-side camera positions expose the mechanisms.
- The microstructure distinguishes layered graphite flakes from NMC secondary granules. Its 0–100% morph is deterministic, reversible and covered by a conservative nonintersection check. Displayed progress is illustrative geometry, not measured porosity or a physical simulation.
- Scene changes use one cancellable GSAP orchestration path, cross-fade visible worlds, expand the macro surface into the micro scene and respect `prefers-reduced-motion`.
- Optimization candidates and seed-11 replay values are exported from checked-in research artifacts. Unknown target values do not influence visible height, color, size, order or trajectory. The fabricated response surface was removed.

## Scientific provenance and boundaries

| Scenario | Dataset | Recorded replay | Result shown |
|---|---|---|---|
| Warwick NMC622 | Mendeley v1, DOI `10.17632/wwhm2frfmy.1`, CC0 1.0, 2023 | Seed 11: EXP_07, EXP_05, EXP_08, EXP_09, EXP_03 | Historical best EXP_03 is revealed at step 5. |
| Drakopoulos graphite | Mendeley v1, DOI `10.17632/4dh2h3tsf4.1`, CC BY 4.0, 2022; paper DOI `10.1016/j.xcrp.2021.100683` | Seed 11: d3602183e567, 7acc4561b3f7, f3ae9bcb49de, c1c280b7366f, fd150c39c22f | Recipe c1c280 is revealed at step 4. |

`presentation/frontend/src/data/export_exhibition.py` regenerates the frontend source payload from the original experiment tables, trajectories, policy summaries and manifests. `docs/exhibition/ASTRA_SCIENTIFIC_REVIEW.md` maps every corrected claim and source.

These are retrospective finite-pool demonstrations. They do not establish live PLC control, autonomous prospective experimentation, lifetime, safety, full-cell performance, commercial scale-up benefit or cross-chemistry transfer. Equipment and microstructure visuals are interpretive.

## Quality gates and evidence

| Gate | Evidence | Result |
|---|---|---|
| Baseline audit | `docs/exhibition/ASTRA_VISUAL_AUDIT.md`; all 12 original frames inspected and live navigation exercised | Passed |
| Authored assets | Blender script, two `.blend` sources, two local GLBs and `docs/exhibition/04_3D_ASSET_MANIFEST.md` | Passed |
| Morph integrity | 101 progress states, reverse cycles, fixed collector and scenario rebuild tests | Passed |
| Transition integrity | Cancellable world handoff, reduced-motion path and three intermediate transition captures | Passed after shortening outgoing-world exposure |
| Scientific integrity | Exact artifact comparison plus hidden-target mutation invariance for both scenarios | Passed |
| Browser QA | Required-control assertions, active-scene checks, model/render checks, 12 endpoint frames and 3 transition frames at 1920×1080 | Passed; 0 console/page errors |

Automated verification:

```text
npm test:      5 files, 30 tests passed
npm run build: passed (Vite reports the existing large-chunk warning)
browser QA:    15 screenshots captured, required states asserted
```

The final screenshot set is in `presentation/screenshots/exhibition-final/`. Tests fail if replay values diverge from the original artifacts, hidden outcomes alter pre-reveal rendering, morph envelopes overlap, controls disappear or the canvas renders empty geometry.

## Measured rendering sample

`docs/exhibition/PERFORMANCE_QA.json` records 300 frame samples on an NVIDIA GeForce RTX 4050 Laptop GPU through ANGLE/D3D11, headless Edge, 1920×1080 and DPR 1. Median frame time was 7.0 ms (142.9 fps equivalent); p95 was 12.2 ms. The sampled final Scene 4 frame reported 35 draw calls and 18,368 triangles.

This is a reproducible measurement under one headless desktop condition, not a universal 60 fps guarantee. Mobile, integrated-GPU, thermal-throttled and high-DPR performance remain unmeasured. The production JavaScript chunk is about 1.39 MB before gzip and remains the main delivery cost.

## Reproduce

```powershell
cd presentation/frontend
npm test
npm run build
npm run dev -- --host 127.0.0.1
node capture_exhibition_qa.js
```

Run the QA script against the local Vite server. It writes endpoint and transition screenshots plus `docs/exhibition/PERFORMANCE_QA.json`.
