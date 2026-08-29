from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.search_space import (
    ContinuousVariable,
    DiscreteVariable,
    SearchSpace,
)

logger = logging.getLogger(__name__)


@dataclass
class TrustRegionState:
    """Represents the mutable state of a TuRBO trust region."""
    center: dict[str, Any]
    best_value: float
    radius: float
    min_radius: float = 0.01
    max_radius: float = 1.6
    init_radius: float = 0.8
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 5
    restarts_count: int = 0
    expansions_count: int = 0
    contractions_count: int = 0
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in self.center.items()},
            "best_value": float(self.best_value),
            "radius": float(self.radius),
            "min_radius": float(self.min_radius),
            "max_radius": float(self.max_radius),
            "init_radius": float(self.init_radius),
            "success_counter": int(self.success_counter),
            "failure_counter": int(self.failure_counter),
            "success_tolerance": int(self.success_tolerance),
            "failure_tolerance": int(self.failure_tolerance),
            "restarts_count": int(self.restarts_count),
            "expansions_count": int(self.expansions_count),
            "contractions_count": int(self.contractions_count),
            "step": int(self.step),
        }


class TuRBOTrustRegion:
    """Domain-agnostic TuRBO (Trust Region Bayesian Optimization) controller.

    Operates purely over search space bounds and candidate observations without
    access to hidden simulator/evaluator truths.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        init_radius: float = 0.8,
        min_radius: float = 0.01,
        max_radius: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int = 5,
        improvement_threshold: float = 1e-4,
    ) -> None:
        self.search_space = search_space
        self.init_radius = float(init_radius)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.success_tolerance = int(success_tolerance)
        self.failure_tolerance = int(failure_tolerance)
        self.improvement_threshold = float(improvement_threshold)
        self.state: TrustRegionState | None = None

    def initialize(
        self,
        center_candidate: Mapping[str, Any],
        initial_best_value: float,
    ) -> TrustRegionState:
        """Initializes the trust region centered at the best known observed candidate."""
        center_clean = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in center_candidate.items()
            if k in self.search_space.all_variable_names
        }
        self.state = TrustRegionState(
            center=center_clean,
            best_value=float(initial_best_value),
            radius=self.init_radius,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            init_radius=self.init_radius,
            success_counter=0,
            failure_counter=0,
            success_tolerance=self.success_tolerance,
            failure_tolerance=self.failure_tolerance,
            restarts_count=0,
            expansions_count=0,
            contractions_count=0,
            step=0,
        )
        return self.state

    def update(
        self,
        observed_candidate: Mapping[str, Any],
        observed_value: float,
        objective: str = "maximize",
        fallback_center: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Updates trust region state based on the newly evaluated experimental candidate.

        Returns a dictionary describing the transition (expanded, contracted, restarted).
        """
        if self.state is None:
            self.initialize(observed_candidate, observed_value)
            return {
                "expanded": False,
                "contracted": False,
                "restarted": False,
                "radius": self.init_radius,
                "success_counter": 0,
                "failure_counter": 0,
                "restarts_count": 0,
            }

        self.state.step += 1
        val = float(observed_value)
        best = float(self.state.best_value)

        # Check if improvement occurred
        if objective == "maximize":
            improved = val > best + self.improvement_threshold
        else:
            improved = val < best - self.improvement_threshold

        expanded = False
        contracted = False
        restarted = False

        if improved:
            self.state.best_value = val
            center_clean = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in observed_candidate.items()
                if k in self.search_space.all_variable_names
            }
            self.state.center = center_clean
            self.state.success_counter += 1
            self.state.failure_counter = 0

            if self.state.success_counter >= self.state.success_tolerance:
                # Expand trust region
                new_radius = min(self.state.radius * 2.0, self.state.max_radius)
                if new_radius > self.state.radius + 1e-9:
                    expanded = True
                    self.state.expansions_count += 1
                self.state.radius = new_radius
                self.state.success_counter = 0
        else:
            self.state.failure_counter += 1
            self.state.success_counter = 0

            if self.state.failure_counter >= self.state.failure_tolerance:
                # Shrink trust region
                new_radius = self.state.radius / 2.0
                contracted = True
                self.state.contractions_count += 1
                self.state.radius = new_radius
                self.state.failure_counter = 0

        # Check restart condition
        if self.state.radius < self.state.min_radius:
            restarted = True
            self.state.restarts_count += 1
            self.state.radius = self.state.init_radius
            self.state.success_counter = 0
            self.state.failure_counter = 0

            # If fallback center provided, relocate center upon restart
            if fallback_center is not None:
                self.state.center = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in fallback_center.items()
                    if k in self.search_space.all_variable_names
                }

        update_info = {
            "step": self.state.step,
            "expanded": expanded,
            "contracted": contracted,
            "restarted": restarted,
            "radius": float(self.state.radius),
            "success_counter": int(self.state.success_counter),
            "failure_counter": int(self.state.failure_counter),
            "restarts_count": int(self.state.restarts_count),
            "best_value": float(self.state.best_value),
            "center": dict(self.state.center),
        }
        self.state.history.append(update_info)
        return update_info

    def get_bounding_box(self) -> dict[str, tuple[float, float]]:
        """Computes the continuous bounding box for each free variable in the search space."""
        if self.state is None:
            raise RuntimeError("TuRBOTrustRegion has not been initialized.")

        bounds: dict[str, tuple[float, float]] = {}
        for var in self.search_space.variables:
            if isinstance(var, ContinuousVariable):
                center_v = float(self.state.center.get(var.name, (var.lower + var.upper) / 2.0))
                full_range = var.upper - var.lower
                half_len = 0.5 * self.state.radius * full_range
                low = max(var.lower, center_v - half_len)
                high = min(var.upper, center_v + half_len)
                bounds[var.name] = (float(low), float(high))
        return bounds

    def sample_candidates(
        self,
        n: int,
        seed: int = 42,
        max_oversample_factor: int = 50,
    ) -> pd.DataFrame:
        """Samples feasible candidate points strictly located within the active trust region."""
        if self.state is None:
            raise RuntimeError("TuRBOTrustRegion has not been initialized.")

        rng = np.random.default_rng(seed)
        box = self.get_bounding_box()

        batch_size = max(1000, n * 2)
        feasible_rows: list[dict[str, Any]] = []
        total_drawn = 0
        max_draw = n * max_oversample_factor

        while len(feasible_rows) < n and total_drawn < max_draw:
            total_drawn += batch_size
            data: dict[str, np.ndarray] = {}

            for var in self.search_space.variables:
                if isinstance(var, ContinuousVariable):
                    low, high = box[var.name]
                    data[var.name] = rng.uniform(low, high, size=batch_size)
                elif isinstance(var, DiscreteVariable):
                    data[var.name] = var.sample(rng, size=batch_size)

            df_cand = pd.DataFrame(data)
            for _, row in df_cand.iterrows():
                row_dict = row.to_dict()
                if self.search_space.is_feasible(row_dict):
                    feasible_rows.append(self.search_space.compute_derived(row_dict))
                    if len(feasible_rows) >= n:
                        break

        if not feasible_rows:
            logger.warning("TuRBO trust region candidate sampling yielded zero feasible points; falling back to global search space sampling.")
            return self.search_space.sample_feasible(n=n, seed=seed)

        if len(feasible_rows) < n:
            reps = (n // len(feasible_rows)) + 1
            all_rows = (feasible_rows * reps)[:n]
            return pd.DataFrame(all_rows)

        return pd.DataFrame(feasible_rows[:n])
