from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
from src.science.actions import (
    ActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)

logger = logging.getLogger(__name__)


class UnmeasuredElectrolyteCandidateError(KeyError):
    """Raised when an unmeasured candidate is queried against the historical experimental oracle.

    Enforces fail-closed semantics: never imputes 0.0, nearest-neighbor, or simulated values.
    """
    pass


class HistoricalElectrolyteOracle:
    """Historical experimental reveal oracle for retrospective finite-pool replay.

    Strictly reveals ONLY genuine experimentally evaluated outcomes from the de-expanded campaign dataset.
    """

    def __init__(self, df_historical_outcomes: pd.DataFrame) -> None:
        self._lookup: dict[str, dict[str, Any]] = {}
        for _, row in df_historical_outcomes.iterrows():
            cid = str(row["candidate_id"])
            if cid not in self._lookup:
                self._lookup[cid] = row.to_dict()

    @property
    def candidate_count(self) -> int:
        return len(self._lookup)

    def is_measured(self, candidate_id: str) -> bool:
        return candidate_id in self._lookup

    def reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Reveals the historical experimental measurement for an action.

        Raises
        ------
        UnmeasuredElectrolyteCandidateError
            If the candidate was never physically synthesized and measured.
        ValueError
            If the action type is not CAPACITY_TEST.
        """
        act_norm = normalize_action_type(action.action_type)
        if act_norm != "CAPACITY_TEST":
            raise ValueError(f"HistoricalElectrolyteOracle only supports 'CAPACITY_TEST', got '{act_norm}'.")

        cid = action.candidate_id
        if cid not in self._lookup:
            raise UnmeasuredElectrolyteCandidateError(
                f"Candidate '{cid}' is unmeasured in the historical dataset. "
                f"Cannot reveal wet-lab outcome without physical experiment."
            )

        rec = self._lookup[cid]
        c_norm = float(rec["C_norm_20"])

        return ExperimentOutcome(
            action_id=action.action_id,
            candidate_id=cid,
            action_type="CAPACITY_TEST",
            revealed_data={"C_norm_20": c_norm},
            canonical_observation=c_norm,
            provenance={
                "oracle_kind": "historical_experimental_reveal",
                "experimental": True,
                "source_dataset": "AmanchukwuLab/AL-anode-free",
                "source_batch": int(rec.get("batch", -1)),
                "historical_outcome_id": str(rec.get("historical_outcome_id", "")),
                "de_expansion_status": str(rec.get("de_expansion_status", "UNKNOWN")),
                "solvent_smiles": str(rec.get("solv_comb_sm", "")),
                "salt_smiles": str(rec.get("canonical_salt", rec.get("salt_comb_sm", ""))),
            },
        )


class SurrogateElectrolyteOracle:
    """In-silico simulation surrogate oracle for large-pool (e.g. 333k LiFSI) closed-loop screening.

    Explicitly labeled as SIMULATION ONLY. Must never be conflated with real wet-lab measurements.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: tuple[str, ...] = ELECTROLYTE_SOLVENT_FEATURES,
        random_state: int = 42,
        noise_std: float = 0.02,
    ) -> None:
        from sklearn.ensemble import ExtraTreesRegressor

        self._feature_cols = list(feature_cols)
        self._noise_std = noise_std
        self._random_state = random_state
        self._simulation_seed = random_state

        X = df_train[self._feature_cols].values
        y = df_train["C_norm_20"].values

        # Use a frozen ExtraTrees ensemble (distinct from Gaussian Process / BayesianRidge policy models)
        self._model = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=random_state)
        self._model.fit(X, y)
        logger.info("Fitted SurrogateElectrolyteOracle on %d training rows.", len(X))

    def set_simulation_seed(self, seed: int) -> None:
        """Sets the benchmark simulation seed for reproducible deterministic noise."""
        self._simulation_seed = int(seed)

    def predict_latent(self, candidate_features: Union[np.ndarray, dict[str, Any]]) -> float:
        """Predict noiseless latent ground truth capacity f(x) for candidate features (offline omniscient evaluation only)."""
        if isinstance(candidate_features, dict):
            X = np.array([[float(candidate_features[f]) for f in self._feature_cols]], dtype=np.float64)
        else:
            X = np.atleast_2d(candidate_features)
        return float(self._model.predict(X)[0])

    def predict_latent_batch(self, candidate_features_matrix: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of noiseless latent ground truth capacity across an entire search space."""
        return self._model.predict(candidate_features_matrix)

    def predict(self, candidate_features: np.ndarray) -> float:
        """Predict noiseless latent ground truth capacity f(x) (canonical evaluation method)."""
        return self.predict_latent(candidate_features)

    def predict_capacity_loss(self, candidate_features: np.ndarray) -> float:
        """Alias for compatibility with offline surrogate evaluation."""
        return self.predict_latent(candidate_features)

    def get_simulated_noise(self, candidate_id: str, simulation_seed: int | None = None) -> float:
        """Generates deterministic simulated measurement noise for a (seed, candidate_id) pair.

        Ensures that querying the same candidate under the same benchmark seed yields the identical
        noisy observation regardless of policy query order or execution timeline.
        """
        import hashlib

        s_seed = self._simulation_seed if simulation_seed is None else int(simulation_seed)
        noise_key = f"{s_seed}_{candidate_id}".encode("utf-8")
        seed_hash = int(hashlib.sha256(noise_key).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_hash)
        return float(rng.normal(0.0, self._noise_std))

    def reveal(
        self,
        action: ScientificAction,
        candidate_features: np.ndarray,
        simulation_seed: int | None = None,
    ) -> ExperimentOutcome:
        """Simulates an experimental measurement y(x) = clip(f(x) + epsilon, 0, 1)."""
        act_norm = normalize_action_type(action.action_type)
        if act_norm != "CAPACITY_TEST":
            raise ValueError(f"SurrogateElectrolyteOracle only supports 'CAPACITY_TEST', got '{act_norm}'.")

        latent_truth = self.predict_latent(candidate_features)
        noise = self.get_simulated_noise(action.candidate_id, simulation_seed=simulation_seed)
        sim_val = float(np.clip(latent_truth + noise, 0.0, 1.0))

        return ExperimentOutcome(
            action_id=action.action_id,
            candidate_id=action.candidate_id,
            action_type="CAPACITY_TEST",
            revealed_data={"C_norm_20": sim_val},
            canonical_observation=sim_val,
            provenance={
                "oracle_kind": "surrogate_simulation",
                "experimental": False,
                "model_family": "ExtraTreesRegressor",
                "latent_oracle_capacity": round(latent_truth, 6),
                "noisy_simulated_observation": round(sim_val, 6),
                "simulated_noise": round(noise, 6),
                "simulated_noise_std": self._noise_std,
                "simulation_seed": self._simulation_seed if simulation_seed is None else simulation_seed,
                "label": "SIMULATED SURROGATE ORACLE - In-Silico Computational Approximation",
            },
        )

