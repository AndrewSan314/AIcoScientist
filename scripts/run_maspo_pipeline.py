from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.process.stages import ProcessStage
from src.process.surrogates.core import ArtisticRunDirectoryAdapter, GenericTabularAdapter
from src.process.surrogates.pipeline import PipelineConfig, run_pipeline


def synthetic_frame(seed: int = 42) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recipe in range(16):
        for replicate in range(2):
            cbd_fraction = 0.03 + recipe * 0.002
            pressure = 40.0 + (recipe % 8) * 5.0
            drying_porosity = 0.34 + 0.003 * recipe + 0.001 * replicate
            rows.append({
                "sample_id": f"synthetic-{recipe}-{replicate}", "run_id": f"run-{recipe}-{replicate}", "recipe_id": f"recipe-{recipe}", "group_id": f"recipe-{recipe}",
                "stage": "CALENDERING", "fidelity": "SYNTHETIC", "requested_horizon": 100_000,
                "cbd_fraction": cbd_fraction, "calender_pressure": pressure, "drying_porosity": drying_porosity,
                "slurry_viscosity": 1.2 + 0.02 * recipe,
                "capacity_retention": 0.76 + 1.1 * cbd_fraction - 0.0004 * (pressure - 55.0) ** 2 - 0.35 * (drying_porosity - 0.37) ** 2 + 0.001 * replicate,
            })
    return pd.DataFrame(rows)


def _adapter(dataset: dict, seed: int):
    kind = dataset.get("kind", "synthetic")
    if kind == "synthetic":
        return GenericTabularAdapter(synthetic_frame(seed), dataset["mapping"], source="synthetic-fixture")
    if kind == "csv":
        return GenericTabularAdapter(pd.read_csv(dataset["path"]), dataset["mapping"], source=str(dataset["path"]))
    if kind == "artistic_normalized":
        return ArtisticRunDirectoryAdapter(dataset["root"], stage=ProcessStage(dataset.get("stage", "CALENDERING")), targets=dataset["targets"])
    raise ValueError("dataset.kind must be synthetic, csv, or artistic_normalized")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset-pluggable, leakage-safe MASPO surrogate pipeline; it never runs ARTISTIC.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pipeline config must be a mapping")
    training, dataset = dict(raw.get("training", {})), dict(raw.get("dataset", {}))
    config = PipelineConfig(
        targets=tuple(training["targets"]), model_type=str(training.get("model_type", "gp")), seed=int(training.get("seed", 42)),
        validation_fraction=float(training.get("validation_fraction", 0.2)), test_fraction=float(training.get("test_fraction", 0.2)),
        objective_target=training.get("objective_target"), objective_sense=str(training.get("objective_sense", "maximize")),
        proposal_count=int(training.get("proposal_count", 3)), exploration_beta=float(training.get("exploration_beta", 1.0)),
    )
    report = run_pipeline(_adapter(dataset, config.seed), args.output or Path(raw.get("output", "outputs/maspo_pipeline")), config)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
