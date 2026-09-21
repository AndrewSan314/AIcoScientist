# Scientific integrity review — exhibition

Reviewed 2026-09-21. Scope: frontend presentation and checked-in scientific artifacts; research engine unchanged.

## Findings and corrections

| Finding | Correction / evidence |
|---|---|
| Hidden targets placed every point vertically before reveal | Unknown candidates now use a common baseline. Only initial observations and completed replay selections receive outcome heights. Stems, color, size and trajectory obey the same reveal state. |
| Fabricated exponential response surface labeled as GP landscape | Removed. A control-space candidate field replaces it. No prediction grid for a specific step was established. Selected-candidate GP mean, posterior SD and expected improvement are genuinely present in the recorded step artifacts and are displayed separately. |
| Graphite list order and Recipe-01… labels encoded historical rank | Candidates now ordered by canonical ID with neutral ID-derived labels. |
| Hard-coded replay order, uncertainty and acquisition scores did not match seed 11 | Replaced from original trajectory JSON. Graphite selected IDs: d3602183e567, 7acc4561b3f7, f3ae9bcb49de, c1c280b7366f, fd150c39c22f. Warwick: EXP_07, EXP_05, EXP_08, EXP_09, EXP_03. |
| Initial best-so-far read the first post-selection result | Calculated from the three visible initial candidates. |
| Warwick roll gap/pass count were presented as legal optimizer controls | Removed from replay controls: the recorded horizon exposes roll temperature, target density and target coating weight. Loading regime labels describe the latter. Measured density is not a target control. |
| Warwick metadata had a fabricated DOI, wrong license, year and repository | Corrected to Mendeley version 1, DOI `10.17632/wwhm2frfmy.1`, CC0 1.0, 2023. |
| Graphite provenance mixed paper with dataset | Dataset DOI `10.17632/4dh2h3tsf4.1`, Mendeley version 1, CC BY 4.0, 2022; research paper remains cited by the benchmark manifest as `10.1016/j.xcrp.2021.100683`. |
| Warwick random Hit@1/3 used 1/18 and 3/18 despite 3 initial observations | Corrected to 1/15 and 3/15, consistent with EXACT_ANALYTICAL_RANDOM in policy summary. |
| Artificial continuous thickness/porosity and a hard-coded warning threshold | Removed from displayed metrics. Morph shows explicitly illustrative geometric progress; no measured porosity interpolation or failure threshold. |
| Unsupported weeks-to-discovery, physical pore simulation and universal +66.7% claim | Removed. Hero gives historical replay scope and evaluated seed count. |
| Trajectory buffer could not grow on repeated next-step actions | Rebuilds and disposes trajectory geometry at every step; back navigation restores the same initial visible state. |

## Source mapping

`presentation/frontend/src/data/export_exhibition.py` regenerates `exhibitionSource.json` directly from:

- `outputs/drakopoulos_rediscovery_v4/eligible_recipe_table.csv`: 12 complete recipes, 36 replicates, D30 in mAh/g; candidate means are rounded to two decimals in this CSV.
- `outputs/drakopoulos_rediscovery_v4/trajectories/AICOSCIENTIST_PROCESS_SURROGATE_trajectories.json`: seed 11, including full-precision selected outcomes, saved pre-reveal predictions, uncertainty and expected improvement.
- `outputs/drakopoulos_rediscovery_v4/policy_summary.csv`: unconstrained D30 policy, 10 seeds, Hit@1/3/5 = 30/50/100%.
- `outputs/warwick_nmc622_calendering/experiment_table.csv`: 18 conditions, 54 cells, dimensionless 5C/0.2C target, legal controls and replicate IDs.
- `outputs/warwick_nmc622_calendering/trajectories/aicoscientist_full_process_engine_seed_11.json`: recorded selection sequence, predictions and observations. Best condition appears at step 5.
- `outputs/warwick_nmc622_calendering/policy_summary.csv`: 10 seeds, Hit@1/3/5 = 0/50/100%; analytical random = 6.6667/20/33.3333%.
- Warwick `dataset_manifest.json`, `source_audit.json`, `decision_variable_audit.json`; graphite `benchmark_manifest.json` confirm source and benchmark boundaries.

Live publisher pages checked: [Warwick dataset](https://data.mendeley.com/datasets/wwhm2frfmy/1), [Graphite dataset](https://data.mendeley.com/datasets/4dh2h3tsf4/1). These confirm DOI/version/year/license. No new scientific results were computed.

## Boundaries

The scenarios are separate datasets and research contexts. Graphite demonstrates complete-recipe/coating selection, not independent validated optimization of every surrounding machine. Warwick maximizes a rate ratio in the finite historical pool; neither maximum density nor this ratio establishes overall battery performance, lifetime, safety, full-cell performance or commercial scale-up benefit.

The candidate plot is a projection: X/Z use selected legal controls, so settings that differ in an omitted control may overlap. Gray baseline height means unknown, not a zero outcome. Blue represents initial observations, teal acquired observations and orange the recorded best condition after reveal. Lines connect acquisitions in recorded order, not a fitted surface or physical process path. Initial observations are not connected as an invented chronology.

No reconstructed pore-scale structure or validated compression model was found. The microstructure specification retains authored dimensions for rendering compatibility, but physical units and fabricated porosity are not displayed as measurements. Equipment stage ranges without verified mapping are labeled illustrative context. The scene machinery is an educational representation, not an exact reconstruction of either source laboratory.

## Verification

- `npm test --prefix presentation/frontend -- --run src/data/exhibitionIntegrity.test.ts`: 3 passing tests. Mutating all still-hidden outcomes to extreme values leaves visible geometry/material/ordering unchanged at every replay step for both datasets; reverse replay restores initial state. The test also compares every displayed step prediction, uncertainty, acquisition, selection and outcome with original seed-11 artifacts.
- `npm run build --prefix presentation/frontend`: passed. Existing bundle-size warning remains.
- GitNexus impact checks for edited rendering methods, validation and scene components were LOW, with App/PersistentWorldCanvas integration as the relevant direct flow.

This review establishes provenance and presentation boundaries, not independent replication of the original experimental measurements. Visual QA and measured rendering performance are documented separately.
