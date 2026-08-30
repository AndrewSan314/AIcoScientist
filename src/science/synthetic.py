from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.datasets.base import DatasetAdapter, DatasetSpec, TwoStageModelSpec
from src.optimization.search_space import ContinuousVariable, SearchSpace


class SyntheticExperimentOracle:
    """Simulates realistic stepwise laboratory measurements with non-linear physics and measurement noise."""

    def __init__(self, char_noise: float = 0.05, perf_noise: float = 5.0) -> None:
        self.char_noise = char_noise
        self.perf_noise = perf_noise

    def _get_seeded_rng(self, candidate_dict: Mapping[str, Any], seed_salt: int = 0) -> np.random.Generator:
        canon_str = ",".join(f"{k}:{candidate_dict[k]:.4f}" for k in sorted(candidate_dict.keys()) if isinstance(candidate_dict[k], (int, float)))
        h_val = int(hashlib.sha256((canon_str + f":salt_{seed_salt}").encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(h_val)

    def evaluate_characterization(
        self,
        candidate: Mapping[str, Any],
        seed: int | None = None,
    ) -> dict[str, float]:
        """Step 1: Generates post-experiment physical characterization (Structure/Morphology)."""
        rng = np.random.default_rng(seed) if seed is not None else self._get_seeded_rng(candidate, 101)
        x1 = float(candidate["x1"])
        x2 = float(candidate["x2"])
        x3 = float(candidate["x3"])

        # Non-linear characterization channels: z1 = sin(x1) + 0.05*x2, z2 = cos(x2/10) + 0.3*x3
        z1 = float(np.sin(x1) + 0.05 * x2 + rng.normal(0.0, self.char_noise))
        z2 = float(np.cos(x2 / 10.0) + 0.3 * x3 + rng.normal(0.0, self.char_noise))

        return {"z1": round(z1, 5), "z2": round(z2, 5)}

    def evaluate_performance(
        self,
        candidate: Mapping[str, Any],
        characterization: Mapping[str, Any],
        seed: int | None = None,
    ) -> dict[str, float]:
        """Step 2: Generates downstream performance outcome from Process + Characterization."""
        rng = np.random.default_rng(seed) if seed is not None else self._get_seeded_rng(candidate, 202)
        x1 = float(candidate["x1"])
        x3 = float(candidate["x3"])
        z1 = float(characterization["z1"])
        z2 = float(characterization["z2"])

        # Non-linear performance target: y = 500 + 40*x1 - 10*x3 + 80*z1 - 50*z2 + noise
        y = float(500.0 + 40.0 * x1 - 10.0 * x3 + 80.0 * z1 - 50.0 * z2 + rng.normal(0.0, self.perf_noise))

        return {"y": round(y, 3)}

    def evaluate_full(
        self,
        candidate: Mapping[str, Any],
        seed: int | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        chars = self.evaluate_characterization(candidate, seed=seed)
        perf = self.evaluate_performance(candidate, chars, seed=(seed + 1 if seed is not None else None))
        return chars, perf


class SyntheticScienceAdapter(DatasetAdapter):
    """Domain-generic synthetic science dataset adapter providing 3 process variables, 2 characterization channels, and 1 target."""

    def __init__(self, oracle: SyntheticExperimentOracle | None = None) -> None:
        self.oracle = oracle or SyntheticExperimentOracle()
        self._spec = DatasetSpec(
            name="synthetic_science",
            id_column="experiment_id",
            candidate_id_column="candidate_id",
            feature_columns=["x1", "x2", "x3", "z1", "z2"],
            target_column="y",
            objective="maximize",
            candidate_columns=["x1", "x2", "x3"],
            pre_experiment_features=["x1", "x2", "x3"],
            post_experiment_characterization=["z1", "z2"],
            targets=["y"],
            candidate_variables=["x1", "x2", "x3"],
            supports_prediction=True,
            supports_optimization=True,
        )
        self._two_stage_spec = TwoStageModelSpec(
            dataset_name="synthetic_science",
            process_features=["x1", "x2", "x3"],
            characterization_targets=["z1", "z2"],
            performance_targets=["y"],
        )

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    @property
    def two_stage_spec(self) -> TwoStageModelSpec:
        return self._two_stage_spec

    @property
    def search_space(self) -> SearchSpace:
        """Explicit generic SearchSpace declaring variable bounds and types."""
        return SearchSpace(
            name="synthetic_science_space",
            variables=[
                ContinuousVariable(name="x1", lower=1.0, upper=5.0),
                ContinuousVariable(name="x2", lower=10.0, upper=50.0),
                ContinuousVariable(name="x3", lower=0.1, upper=2.0),
            ],
        )

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        return self.load_initial_dataset(n_samples=15, seed=42)

    def load_initial_dataset(self, n_samples: int = 15, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1_vals = rng.uniform(1.0, 5.0, size=n_samples)
        x2_vals = rng.uniform(10.0, 50.0, size=n_samples)
        x3_vals = rng.uniform(0.1, 2.0, size=n_samples)

        rows: list[dict[str, Any]] = []
        for i in range(n_samples):
            cand = {
                "x1": float(x1_vals[i]),
                "x2": float(x2_vals[i]),
                "x3": float(x3_vals[i]),
            }
            chars, perf = self.oracle.evaluate_full(cand, seed=seed * 100 + i)
            cid = f"CAND_SYNTH_{i:03d}"
            eid = f"EXP_INIT_{i:03d}"
            rows.append({
                "experiment_id": eid,
                "candidate_id": cid,
                **cand,
                **chars,
                **perf,
            })

        return pd.DataFrame(rows)

    def candidate_space(
        self,
        observed: pd.DataFrame | None = None,
        n_candidates: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generates candidate pool of controllable pre-experiment features."""
        rng = np.random.default_rng(seed)
        x1_vals = rng.uniform(1.0, 5.0, size=n_candidates)
        x2_vals = rng.uniform(10.0, 50.0, size=n_candidates)
        x3_vals = rng.uniform(0.1, 2.0, size=n_candidates)

        cands: list[dict[str, Any]] = []
        for i in range(n_candidates):
            c_dict = {
                "candidate_id": f"CAND_S{seed}_{i:03d}",
                "x1": round(float(x1_vals[i]), 4),
                "x2": round(float(x2_vals[i]), 4),
                "x3": round(float(x3_vals[i]), 4),
            }
            cands.append(c_dict)

        cand_df = pd.DataFrame(cands)

        # Filter duplicates against observed
        if observed is not None and not observed.empty:
            obs_x = observed[["x1", "x2", "x3"]].to_numpy(dtype=float)
            cand_x = cand_df[["x1", "x2", "x3"]].to_numpy(dtype=float)
            ranges = np.array([4.0, 40.0, 1.9])
            # Keep candidates with normalized distance >= 0.01
            keep_idx = []
            for j in range(len(cand_x)):
                dist = np.min(np.sqrt(np.sum(((obs_x - cand_x[j]) / ranges) ** 2, axis=1)))
                if dist >= 0.01:
                    keep_idx.append(j)
            cand_df = cand_df.iloc[keep_idx].reset_index(drop=True)

        return cand_df
