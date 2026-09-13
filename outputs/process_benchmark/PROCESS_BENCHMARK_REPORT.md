# Battery Process Stress Suite

Status: **PARTIAL**.

## Reproducibility

- Command: `scripts/run_battery_process_benchmark.py --allow-unavailable --output outputs/process_benchmark`
- Commit: `5289428a3a180e8bfad367bc65de48e33c657a2a`
- Python: `3.14.2`
- Process tests (run separately): `python -m pytest -q tests/process -p no:cacheprovider`
- Artifact manifest: `manifest.json`

## Limitations

- Small, source-specific datasets; no chemistry-generalization claim.
- Missing modalities are omitted rather than imputed.
- Unavailable raw sources: artistic.
- No live production control or causal-effect claim.
