from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.science.actions import (
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
)


def generate_simplex_candidates(n_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic candidate compositions on the Au-Ir-Rh ternary simplex."""
    rng = np.random.default_rng(seed)
    # Generate Dirichlet distributed compositions
    raw = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=n_samples) * 100.0
    rows = []
    for i, comp in enumerate(raw):
        cid = f"SYN_CAND_{i+1:03d}"
        rows.append(
            {
                "candidate_id": cid,
                "Library": "SYNTHETIC_BENCHMARK",
                "Area": 1,
                "Au": float(comp[0]),
                "Ir": float(comp[1]),
                "Rh": float(comp[2]),
            }
        )
    return pd.DataFrame(rows)


class SyntheticTruthWorld:
    """Base class for controlled synthetic benchmark environments with known ground-truth hypotheses."""

    def __init__(
        self,
        name: str,
        true_hypothesis_id: str,
        candidate_pool: pd.DataFrame | None = None,
        noise_level: float = 0.05,
        difficulty: str = "medium",
        seed: int = 42,
    ) -> None:
        self.name = name
        self.true_hypothesis_id = true_hypothesis_id
        self.difficulty = difficulty
        self.seed = seed
        self.noise_level = noise_level
        self._candidate_pool = (
            candidate_pool.copy() if candidate_pool is not None else generate_simplex_candidates(n_samples=150, seed=seed)
        )

        self._ground_truth: dict[str, dict[str, Any]] = {}
        self._revealed_xrd: dict[str, ExperimentOutcome] = {}
        self._revealed_property: dict[str, ExperimentOutcome] = {}

        self._generate_ground_truth()

    def _generate_ground_truth(self) -> None:
        """Subclasses implement the exact ground-truth generative process."""
        raise NotImplementedError

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns candidate pool without targets."""
        return self._candidate_pool[["candidate_id", "Library", "Area", "Au", "Ir", "Rh"]].copy()

    def get_revealed_xrd_ids(self) -> list[str]:
        return list(self._revealed_xrd.keys())

    def get_revealed_property_ids(self) -> list[str]:
        return list(self._revealed_property.keys())

    def get_revealed_xrd(self) -> dict[str, ExperimentOutcome]:
        return copy.deepcopy(self._revealed_xrd)

    def get_revealed_properties(self) -> dict[str, ExperimentOutcome]:
        return copy.deepcopy(self._revealed_property)

    def is_xrd_observed(self, candidate_id: str) -> bool:
        return candidate_id in self._revealed_xrd

    def is_property_observed(self, candidate_id: str) -> bool:
        return candidate_id in self._revealed_property

    def execute_xrd(self, candidate_id: str, step: int = 0) -> ExperimentOutcome:
        if candidate_id not in self._ground_truth:
            raise KeyError(f"Candidate '{candidate_id}' not found in synthetic world.")
        if candidate_id in self._revealed_xrd:
            raise ValueError(f"XRD already executed on '{candidate_id}'.")

        gt = self._ground_truth[candidate_id]
        emb = np.array(gt["xrd_embedding"], dtype=np.float64)

        outcome = ExperimentOutcome(
            action_id=f"syn_xrd_{candidate_id}",
            candidate_id=candidate_id,
            action_type=ExperimentActionType.XRD,
            revealed_data={
                "normalized_intensity": gt.get("xrd_spectrum", np.zeros(450)),
                "xrd_embedding": emb,
                "peak_two_theta": 42.0 + float(emb[0]) * 3.0,
            },
            provenance={
                "world_name": self.name,
                "true_hypothesis": self.true_hypothesis_id,
                "step": step,
            },
        )
        self._revealed_xrd[candidate_id] = outcome
        return outcome

    def execute_property(self, candidate_id: str, step: int = 0) -> ExperimentOutcome:
        if candidate_id not in self._ground_truth:
            raise KeyError(f"Candidate '{candidate_id}' not found in synthetic world.")
        if candidate_id in self._revealed_property:
            raise ValueError(f"Property already measured on '{candidate_id}'.")

        gt = self._ground_truth[candidate_id]
        k0 = float(gt["k0"])

        outcome = ExperimentOutcome(
            action_id=f"syn_prop_{candidate_id}",
            candidate_id=candidate_id,
            action_type=ExperimentActionType.PROPERTY,
            revealed_data={
                "k0": k0,
                "i_lim": 0.05,
                "alpha": 0.50,
            },
            provenance={
                "world_name": self.name,
                "true_hypothesis": self.true_hypothesis_id,
                "step": step,
            },
        )
        self._revealed_property[candidate_id] = outcome
        return outcome

    def execute(self, action: ScientificAction) -> ExperimentOutcome:
        if action.action_type == ExperimentActionType.XRD:
            return self.execute_xrd(action.candidate_id, step=action.requested_at_step)
        elif action.action_type == ExperimentActionType.PROPERTY:
            return self.execute_property(action.candidate_id, step=action.requested_at_step)
        raise ValueError(f"Unknown action type: {action.action_type}")

    def reset(self) -> None:
        self._revealed_xrd.clear()
        self._revealed_property.clear()


# ---------------------------------------------------------------------------
# World 1: H1 True (Composition-Sufficient)
# ---------------------------------------------------------------------------
class World1_CompositionSufficient(SyntheticTruthWorld):
    """World 1: Ground truth generated according to H1 (Composition-Sufficient).

    k0 = f(Au, Ir, Rh) + noise.
    Structure embedding z is generated independently and does NOT improve k0 prediction.
    """

    def __init__(self, candidate_pool: pd.DataFrame | None = None, noise_level: float = 0.05, seed: int = 42) -> None:
        super().__init__(
            name="Synthetic_World_1_H1_True",
            true_hypothesis_id="H1",
            candidate_pool=candidate_pool,
            noise_level=noise_level,
            seed=seed,
        )

    def _generate_ground_truth(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._ground_truth.clear()

        for _, row in self._candidate_pool.iterrows():
            cid = row["candidate_id"]
            au = row["Au"] / 100.0
            ir = row["Ir"] / 100.0
            rh = row["Rh"] / 100.0

            # Smooth nonlinear composition surface
            true_k0 = 0.002 + 0.008 * ir + 0.004 * np.sin(np.pi * au) + 0.003 * (rh**2)
            noise_k0 = rng.normal(loc=0.0, scale=self.noise_level * 0.001)
            obs_k0 = max(1e-5, true_k0 + noise_k0)

            # Structure embedding is smooth baseline with independent uncoupled variation
            emb = np.zeros(8, dtype=np.float64)
            emb[0] = float(np.cos(np.pi * au))
            emb[1] = float(np.sin(np.pi * rh))
            emb[2:] = rng.normal(loc=0.0, scale=0.1, size=6)

            self._ground_truth[cid] = {
                "k0": obs_k0,
                "xrd_embedding": emb,
                "xrd_spectrum": np.linspace(0, 1, 450) * emb[0],
            }


# ---------------------------------------------------------------------------
# World 2: H2 True (Structure-Informed)
# ---------------------------------------------------------------------------
class World2_StructureInformed(SyntheticTruthWorld):
    """World 2: Ground truth generated according to H2 (Structure-Informed).

    Latent structural parameter z1 is imperfectly predicted by composition.
    k0 strongly depends on z1: k0 = g(z1) + h(composition) + noise.
    Measuring XRD directly reveals z1, providing significant predictive advantage.
    """

    def __init__(self, candidate_pool: pd.DataFrame | None = None, noise_level: float = 0.05, seed: int = 42) -> None:
        super().__init__(
            name="Synthetic_World_2_H2_True",
            true_hypothesis_id="H2",
            candidate_pool=candidate_pool,
            noise_level=noise_level,
            seed=seed,
        )

    def _generate_ground_truth(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._ground_truth.clear()

        for _, row in self._candidate_pool.iterrows():
            cid = row["candidate_id"]
            au = row["Au"] / 100.0
            ir = row["Ir"] / 100.0
            rh = row["Rh"] / 100.0

            # Latent physical structure has stochastic variance not predictable from nominal composition
            latent_z1 = float(np.sin(3.0 * np.pi * au) * np.cos(2.0 * np.pi * rh) + rng.normal(loc=0.0, scale=0.35))
            latent_z2 = float(np.cos(np.pi * ir) + rng.normal(loc=0.0, scale=0.20))

            # Property directly mediated by structure
            true_k0 = 0.002 + 0.007 * (latent_z1**2) + 0.003 * ir
            noise_k0 = rng.normal(loc=0.0, scale=self.noise_level * 0.0005)
            obs_k0 = max(1e-5, true_k0 + noise_k0)

            emb = np.zeros(8, dtype=np.float64)
            emb[0] = latent_z1
            emb[1] = latent_z2
            emb[2:] = rng.normal(loc=0.0, scale=0.05, size=6)

            self._ground_truth[cid] = {
                "k0": obs_k0,
                "xrd_embedding": emb,
                "xrd_spectrum": np.linspace(0, 1, 450) * (latent_z1 + 2.0),
            }


# ---------------------------------------------------------------------------
# World 3: H3 True (Local Structural-Regime)
# ---------------------------------------------------------------------------
class World3_LocalStructuralRegime(SyntheticTruthWorld):
    """World 3: Ground truth generated according to H3 (Local Structural-Regime).

    A localized region in composition space (e.g. Rh > 40%, Au < 30%) exhibits a sharp
    structural regime shift where properties diverge from smooth global gradients.
    """

    def __init__(self, candidate_pool: pd.DataFrame | None = None, noise_level: float = 0.05, seed: int = 42) -> None:
        super().__init__(
            name="Synthetic_World_3_H3_True",
            true_hypothesis_id="H3",
            candidate_pool=candidate_pool,
            noise_level=noise_level,
            seed=seed,
        )

    def _generate_ground_truth(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._ground_truth.clear()

        for _, row in self._candidate_pool.iterrows():
            cid = row["candidate_id"]
            au = row["Au"]
            ir = row["Ir"]
            rh = row["Rh"]

            # Define localized regime boundary
            in_regime_a = rh > 45.0 and au < 35.0
            in_regime_b = ir > 50.0

            if in_regime_a:
                # Discontinuous regime A
                true_k0 = 0.012 + 0.002 * (rh / 100.0)
                emb = np.array([2.5, -1.8, 0.8, 0.4, 0.0, 0.1, -0.2, 0.3], dtype=np.float64)
            elif in_regime_b:
                # Discontinuous regime B
                true_k0 = 0.008 + 0.003 * (ir / 100.0)
                emb = np.array([-1.5, 2.0, -0.5, 0.2, 0.3, -0.1, 0.1, 0.0], dtype=np.float64)
            else:
                # Baseline background regime
                true_k0 = 0.003 + 0.002 * (au / 100.0)
                emb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

            noise_k0 = rng.normal(loc=0.0, scale=self.noise_level * 0.0005)
            obs_k0 = max(1e-5, true_k0 + noise_k0)
            emb = emb + rng.normal(loc=0.0, scale=0.08, size=8)

            self._ground_truth[cid] = {
                "k0": obs_k0,
                "xrd_embedding": emb,
                "xrd_spectrum": np.linspace(0, 1, 450) * (emb[0] + 3.0),
            }
