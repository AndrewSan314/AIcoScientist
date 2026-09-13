# Battery Process Stress Suite

Status: **PARTIAL**.

## Reproducibility

- Command: `scripts/run_battery_process_benchmark.py --output outputs/maspo_benchmark_final_20260911 --allow-unavailable --seed 42 --replay-seeds 3`
- Commit: `a0d6e87ce488c2e4e5f5ab143d30541b8b6dc671`
- Python: `3.14.2`
- Process tests (run separately): `python -m pytest -q tests/process -p no:cacheprovider`
- Artifact manifest: `manifest.json`

## Limitations

- Small, source-specific datasets; no chemistry-generalization claim.
- Missing modalities are omitted rather than imputed.
- Unavailable raw sources: artistic.
- No live production control or causal-effect claim.
