from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Objective:
    """Specification of a single experimental target objective."""
    name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    threshold: float | None = None
    unit: str = ""

    def is_better(self, val_a: float, val_b: float) -> bool:
        if self.direction == "maximize":
            return val_a > val_b
        return val_a < val_b

    def normalize_for_maximization(self, val: float | np.ndarray) -> float | np.ndarray:
        """Converts minimization objective to standard maximization coordinate for hypervolume calculations."""
        if self.direction == "minimize":
            return -val
        return val


@dataclass(frozen=True)
class MultiObjectiveSpec:
    """Contract for multi-objective optimization across multiple battery performance metrics.

    Examples:
    - maximize capacity
    - maximize lifetime
    - maximize retention
    - minimize degradation
    - minimize charging time
    """
    objectives: list[Objective]
    reference_point: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("MultiObjectiveSpec must contain at least one Objective.")
        names = [obj.name for obj in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate objective names detected: {names}")

    @property
    def objective_names(self) -> list[str]:
        return [obj.name for obj in self.objectives]


def is_pareto_dominated(candidate_vals: np.ndarray, all_vals: np.ndarray, directions: Sequence[str]) -> bool:
    """Determines if candidate_vals is strictly dominated by any row in all_vals.

    A point x is dominated by y if y is at least as good in all objectives and strictly better in at least one.
    """
    # Transform all to maximization orientation
    signs = np.array([1.0 if d == "maximize" else -1.0 for d in directions])
    cand_norm = candidate_vals * signs
    all_norm = all_vals * signs

    # Compare
    greater_or_equal = all_norm >= cand_norm
    strictly_greater = all_norm > cand_norm

    # Dominated if any other point is >= in all objectives and > in at least one
    dominated = np.any(np.all(greater_or_equal, axis=1) & np.any(strictly_greater, axis=1))
    return bool(dominated)


def compute_pareto_front(
    df: pd.DataFrame,
    objectives: Sequence[Objective],
) -> pd.DataFrame:
    """Computes the non-dominated Pareto frontier from candidate/observed points."""
    obj_names = [obj.name for obj in objectives]
    for name in obj_names:
        if name not in df.columns:
            raise ValueError(f"Objective column {name!r} not found in DataFrame.")

    vals = df[obj_names].to_numpy(dtype=float)
    directions = [obj.direction for obj in objectives]

    non_dominated_indices: list[int] = []
    for i in range(len(vals)):
        other_indices = [j for j in range(len(vals)) if j != i]
        if not other_indices:
            non_dominated_indices.append(i)
            continue
        if not is_pareto_dominated(vals[i], vals[other_indices], directions):
            non_dominated_indices.append(i)

    return df.iloc[non_dominated_indices].copy().reset_index(drop=True)


class MultiObjectiveAcquisition(ABC):
    """Abstract interface for multi-objective acquisition functions (e.g. qNEHVI).

    Avoids combining objectives via arbitrary weighted sums in the core architecture.
    """

    @abstractmethod
    def evaluate(
        self,
        candidate_means: np.ndarray,  # shape: (n_candidates, n_objectives)
        candidate_stds: np.ndarray,   # shape: (n_candidates, n_objectives)
        observed_means: np.ndarray,   # shape: (n_observed, n_objectives)
        spec: MultiObjectiveSpec,
    ) -> np.ndarray:  # shape: (n_candidates,)
        """Scores candidate points based on multi-objective expected improvement."""
        raise NotImplementedError


class PlaceholderqNEHVI(MultiObjectiveAcquisition):
    """Interface skeleton for future q-Noisy Expected Hypervolume Improvement (qNEHVI)."""

    def __init__(self, n_mc_samples: int = 512, seed: int = 42) -> None:
        self.n_mc_samples = n_mc_samples
        self.seed = seed

    def evaluate(
        self,
        candidate_means: np.ndarray,
        candidate_stds: np.ndarray,
        observed_means: np.ndarray,
        spec: MultiObjectiveSpec,
    ) -> np.ndarray:
        """Evaluates hypervolume improvement across multiple objectives."""
        # Clean multi-objective scoring interface preserving Pareto dominance semantics
        n_cands = len(candidate_means)
        scores = np.zeros(n_cands, dtype=float)

        # Baseline Pareto front from observed means
        obs_df = pd.DataFrame(observed_means, columns=spec.objective_names)
        pareto_obs = compute_pareto_front(obs_df, spec.objectives)

        # For each candidate, evaluate non-dominated contribution
        for i in range(n_cands):
            c_mean = candidate_means[i]
            c_std = candidate_stds[i]
            # Probabilistic optimism across objectives
            u = np.array([
                c_mean[j] + (1.0 if spec.objectives[j].direction == "maximize" else -1.0) * c_std[j]
                for j in range(len(spec.objectives))
            ])
            # Check if optimistic point dominates observed Pareto front
            is_dom = is_pareto_dominated(u, pareto_obs.to_numpy(dtype=float), [obj.direction for obj in spec.objectives])
            scores[i] = 1.0 if not is_dom else 0.0

        return scores
