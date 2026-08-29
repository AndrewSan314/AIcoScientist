from __future__ import annotations

from dataclasses import dataclass, field
import json
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
    """Represents the mutable state of a TuRBO trust region in normalized coordinates."""
    center: dict[str, Any]
    best_value: float
    length: float  # Current side length in [0, 1] normalized space
    min_length: float = 0.05
    max_length: float = 1.6
    init_length: float = 0.8
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 5
    restarts_count: int = 0
    expansions_count: int = 0
    contractions_count: int = 0
    global_escapes_count: int = 0
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    # Backward compatibility alias
    @property
    def radius(self) -> float:
        return self.length

    @radius.setter
    def radius(self, val: float) -> None:
        self.length = float(val)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in self.center.items()},
            "best_value": float(self.best_value),
            "length": float(self.length),
            "radius": float(self.length),
            "min_length": float(self.min_length),
            "max_length": float(self.max_length),
            "init_length": float(self.init_length),
            "success_counter": int(self.success_counter),
            "failure_counter": int(self.failure_counter),
            "success_tolerance": int(self.success_tolerance),
            "failure_tolerance": int(self.failure_tolerance),
            "restarts_count": int(self.restarts_count),
            "expansions_count": int(self.expansions_count),
            "contractions_count": int(self.contractions_count),
            "global_escapes_count": int(self.global_escapes_count),
            "step": int(self.step),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrustRegionState:
        return cls(
            center=dict(data.get("center", {})),
            best_value=float(data.get("best_value", 0.0)),
            length=float(data.get("length", data.get("radius", 0.8))),
            min_length=float(data.get("min_length", data.get("min_radius", 0.05))),
            max_length=float(data.get("max_length", data.get("max_radius", 1.6))),
            init_length=float(data.get("init_length", data.get("init_radius", 0.8))),
            success_counter=int(data.get("success_counter", 0)),
            failure_counter=int(data.get("failure_counter", 0)),
            success_tolerance=int(data.get("success_tolerance", 3)),
            failure_tolerance=int(data.get("failure_tolerance", 5)),
            restarts_count=int(data.get("restarts_count", 0)),
            expansions_count=int(data.get("expansions_count", 0)),
            contractions_count=int(data.get("contractions_count", 0)),
            global_escapes_count=int(data.get("global_escapes_count", 0)),
            step=int(data.get("step", 0)),
        )


class TuRBOTrustRegion:
    """Domain-agnostic TuRBO (Trust Region Bayesian Optimization) controller.

    Operates in normalized [0, 1]^d free-variable coordinates without
    access to hidden simulator or evaluator truths.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        init_length: float = 0.8,
        min_length: float = 0.05,
        max_length: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int = 5,
        improvement_threshold: float = 1e-3,
        global_escape_frequency: int = 6,
        init_radius: float | None = None,  # Backward compatibility
        min_radius: float | None = None,
        max_radius: float | None = None,
    ) -> None:
        self.search_space = search_space
        self.init_length = float(init_radius if init_radius is not None else init_length)
        self.min_length = float(min_radius if min_radius is not None else min_length)
        self.max_length = float(max_radius if max_radius is not None else max_length)
        self.success_tolerance = int(success_tolerance)
        self.failure_tolerance = int(failure_tolerance)
        self.improvement_threshold = float(improvement_threshold)
        self.global_escape_frequency = int(global_escape_frequency)
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
            length=self.init_length,
            min_length=self.min_length,
            max_length=self.max_length,
            init_length=self.init_length,
            success_counter=0,
            failure_counter=0,
            success_tolerance=self.success_tolerance,
            failure_tolerance=self.failure_tolerance,
            restarts_count=0,
            expansions_count=0,
            contractions_count=0,
            global_escapes_count=0,
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
        """Updates trust region state based on the newly evaluated experimental candidate."""
        if self.state is None:
            self.initialize(observed_candidate, observed_value)
            return {
                "expanded": False,
                "contracted": False,
                "restarted": False,
                "global_escape": False,
                "length": self.init_length,
                "radius": self.init_length,
                "success_counter": 0,
                "failure_counter": 0,
                "restarts_count": 0,
            }

        self.state.step += 1
        val = float(observed_value)
        best = float(self.state.best_value)

        # Check if meaningful improvement occurred
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
                new_len = min(self.state.length * 2.0, self.state.max_length)
                if new_len > self.state.length + 1e-9:
                    expanded = True
                    self.state.expansions_count += 1
                self.state.length = new_len
                self.state.success_counter = 0
        else:
            self.state.failure_counter += 1
            self.state.success_counter = 0

            if self.state.failure_counter >= self.state.failure_tolerance:
                # Shrink trust region
                new_len = self.state.length / 2.0
                contracted = True
                self.state.contractions_count += 1
                self.state.length = new_len
                self.state.failure_counter = 0

        # Check restart condition
        if self.state.length < self.state.min_length:
            restarted = True
            self.state.restarts_count += 1
            self.state.length = self.state.init_length
            self.state.success_counter = 0
            self.state.failure_counter = 0

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
            "length": float(self.state.length),
            "radius": float(self.state.length),
            "success_counter": int(self.state.success_counter),
            "failure_counter": int(self.state.failure_counter),
            "restarts_count": int(self.state.restarts_count),
            "best_value": float(self.state.best_value),
            "center": dict(self.state.center),
        }
        self.state.history.append(update_info)
        return update_info

    def should_global_escape(self, step: int) -> bool:
        """Determines if the optimizer should execute a global exploration query."""
        if self.global_escape_frequency <= 0:
            return False
        # Deterministic periodic escape schedule
        return bool(step > 0 and (step % self.global_escape_frequency == 0))

    def get_bounding_box(self) -> dict[str, tuple[float, float]]:
        """Computes the continuous bounding box for each free variable in normalized-then-unnormalized space."""
        if self.state is None:
            raise RuntimeError("TuRBOTrustRegion has not been initialized.")

        bounds: dict[str, tuple[float, float]] = {}
        for var in self.search_space.variables:
            if isinstance(var, ContinuousVariable):
                center_v = float(self.state.center.get(var.name, (var.lower + var.upper) / 2.0))
                full_range = var.upper - var.lower
                # Normalized center in [0, 1]
                z_center = (center_v - var.lower) / full_range if full_range > 1e-12 else 0.5
                # Half length in [0, 1]
                half_len = 0.5 * self.state.length
                z_low = max(0.0, z_center - half_len)
                z_high = min(1.0, z_center + half_len)
                # Unnormalize back to original domain
                low = var.lower + z_low * full_range
                high = var.lower + z_high * full_range
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
            logger.warning("TuRBO candidate sampling yielded zero feasible points; falling back to global sampling.")
            return self.search_space.sample_feasible(n=n, seed=seed)

        if len(feasible_rows) < n:
            reps = (n // len(feasible_rows)) + 1
            all_rows = (feasible_rows * reps)[:n]
            return pd.DataFrame(all_rows)

        return pd.DataFrame(feasible_rows[:n])
