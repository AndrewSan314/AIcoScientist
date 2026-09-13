# Battery Process Stress Suite

Status: **PARTIAL**.

## Reproducibility

- Command: `scripts/run_battery_process_benchmark.py --output outputs/maspo_benchmark_status_20260911 --allow-unavailable --seed 42 --replay-seeds 3`
- Commit: `b3f82cb6de2a7d50961432d36ff56f0e4456e01d`
- Python: `3.14.2`
- Process tests (run separately): `python -m pytest -q tests/process -p no:cacheprovider`
- Artifact manifest: `manifest.json`

## Limitations

- Small, source-specific datasets; no chemistry-generalization claim.
- Missing modalities are omitted rather than imputed.
- Unavailable raw sources: artistic.
- No live production control or causal-effect claim.
