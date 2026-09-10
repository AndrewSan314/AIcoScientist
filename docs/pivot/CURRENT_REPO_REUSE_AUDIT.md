# Current repository reuse audit

Status: 2026-09-11. This audit was made at `bdaf235772ba9f1fd1bcdbfbba0e102456557a04` on `integration/multimodal-scientific-engine`.

| Component | Files | Current role | Decision | New role | Migration risk | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| Optimizer backend | `src/optimization/backend.py`, `botorch_backend.py`, `finite_pool.py`, `proposal.py` | Official BoTorch finite-pool BO, strict identity, resume-safe proposals | KEEP / MODIFY | Process-recipe acquisition backend | Medium: preserve scalar callers while adding process contracts | existing backend tests; `tests/process/test_constraints.py` |
| Generic search/constraints | `src/optimization/search_space.py`, `constraints.py` | Variables, feasibility, deterministic sampling | KEEP / COMPOSE | Process search-space validation | Low | existing search-space tests; `tests/process/test_process_space.py` |
| Dataset contracts/cache | `src/datasets/base.py`, `cache.py`, `registry.py` | Tabular dataset specs, raw/processed manifests | KEEP / COMPOSE | Battery-process adapters and provenance manifests | Low | adapter/cache tests; `tests/process/test_dataset_adapters.py` |
| Replay/oracle | `src/evaluation/replay.py`, `oracle.py`, `metrics.py` | Historical reveal and regret metrics | KEEP / COMPOSE | No-lookahead recipe replay | Medium: source outcome must remain hidden until reveal | `tests/process/test_offline_replay_no_leakage.py` |
| Ledger/coordinator | `src/science/ledger.py`, `coordinator.py`, `records.py`, `provenance.py` | Deterministic experiment ledger and snapshots | KEEP / COMPOSE | Persist process-run/proposal provenance | Medium: no HIG coupling | `tests/process/test_resume.py`, provenance tests |
| Battery process legacy | `src/build_dataset.py`, `src/train_model.py`, `src/recommend.py`, `run_pipeline.py`, `data/raw/process_data.csv` | Legacy flat process/SEM/EDX pipeline | LEGACY / REUSE FEATURES ONLY | Baseline reference and image features; not the default path | High if treated as source-backed BPSS data | existing legacy tests; new adapter tests |
| SEM/EDX | `src/sem_features.py`, `src/sem_analysis.py`, `src/edx_features.py` | Image ingest, segmentation, morphology features | KEEP / ADAPT | IMAGE modality feature adapter | Low | `tests/test_sem_*`; `tests/process/test_modality_encoders.py` |
| Two-stage science models | `src/science/two_stage.py`, `direct_baseline.py`, `scientific_models.py` | Process → characterization → performance | KEEP / REFERENCE | Stage-aware scalar baseline | Medium: their domain contracts are not process-stage contracts | two-stage tests; stage-model tests |
| Multimodal science | `src/science/multimodal/*` | Hypothesis/evidence-driven modalities | LEGACY_RESEARCH_TRACK | Not imported by default process path | High if coupled to HIG | existing multimodal tests remain legacy coverage |
| Falsification/HIG | `src/science/falsification/*`, `hypothesis_*`, `decision_engine.py` | Falsification-first experimental policy | LEGACY_RESEARCH_TRACK | Auditable historical research only | High: prohibited dependency in process coordinator | boundary test |
| Legacy native optimization | `src/legacy/native_optimizer/*` | Historical custom BO/multiobjective references | LEGACY_RESEARCH_TRACK | Never a production process optimizer | Medium | legacy-only tests |
| UI/presentation | `app/*`, `presentation/*` | Current demonstration UI | MODIFY LATER | Out of scope for pivot implementation | Low | existing UI tests only |

`DEFAULT_PRODUCT_MODE = BATTERY_PROCESS_OPTIMIZATION`. New modules under `src/process/` must not import H1/H2/H3/HIG or `src.science.multimodal`.
