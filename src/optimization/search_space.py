from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Variable:
    """Base class for search space variables."""
    name: str


@dataclass(frozen=True)
class ContinuousVariable(Variable):
    """Bounded continuous floating-point variable."""
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(f"ContinuousVariable {self.name!r} lower bound ({self.lower}) must be < upper bound ({self.upper})")

    def is_valid(self, val: float) -> bool:
        return bool(np.isfinite(val) and self.lower <= val <= self.upper)

    def sample_uniform(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=size)


@dataclass(frozen=True)
class DiscreteVariable(Variable):
    """Discrete variable with an explicit set of allowable values."""
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"DiscreteVariable {self.name!r} must contain at least one value")

    def is_valid(self, val: Any) -> bool:
        return val in self.values

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(self.values, size=size)


@dataclass(frozen=True)
class DerivedVariable(Variable):
    """Variable deterministically computed from other variables."""
    compute_fn: Callable[[Mapping[str, Any]], Any]
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def compute(self, candidate: Mapping[str, Any]) -> Any:
        return self.compute_fn(candidate)


@dataclass(frozen=True)
class Constraint:
    """Feasibility constraint applied to candidate design points."""
    name: str
    predicate: Callable[[Mapping[str, Any]], bool]
    description: str = ""

    def evaluate(self, candidate: Mapping[str, Any]) -> bool:
        try:
            return bool(self.predicate(candidate))
        except Exception:
            return False


class SearchSpace:
    """Domain-agnostic constrained search-space engine.

    Supports continuous variables, discrete variables, derived variables, non-linear
    constraints, rejection-based feasible sampling, and novelty/distance checking.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        derived_variables: Sequence[DerivedVariable] | None = None,
        constraints: Sequence[Constraint] | None = None,
        name: str = "custom_search_space",
    ) -> None:
        self.name = name
        self.variables = list(variables)
        self.derived_variables = list(derived_variables or [])
        self.constraints = list(constraints or [])

        self._var_dict = {v.name: v for v in self.variables}
        self._derived_dict = {v.name: v for v in self.derived_variables}

        # Check name uniqueness
        all_names = [v.name for v in self.variables] + [v.name for v in self.derived_variables]
        if len(all_names) != len(set(all_names)):
            raise ValueError(f"Duplicate variable names detected in SearchSpace: {all_names}")

    @property
    def free_variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def all_variable_names(self) -> list[str]:
        return [v.name for v in self.variables] + [v.name for v in self.derived_variables]

    def compute_derived(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Computes all derived variables for a candidate mapping."""
        res = dict(candidate)
        for d_var in self.derived_variables:
            res[d_var.name] = d_var.compute(res)
        return res

    def is_feasible(self, candidate: Mapping[str, Any] | pd.Series) -> bool:
        """Checks if candidate satisfies all variable bounds and non-linear constraints."""
        cand_dict = candidate.to_dict() if isinstance(candidate, pd.Series) else dict(candidate)

        # 1. Check free variable bounds / validity
        for var in self.variables:
            if var.name not in cand_dict:
                return False
            val = cand_dict[var.name]
            if isinstance(var, ContinuousVariable):
                if not var.is_valid(float(val)):
                    return False
            elif isinstance(var, DiscreteVariable):
                if not var.is_valid(val):
                    return False

        # 2. Compute derived variables
        try:
            full_cand = self.compute_derived(cand_dict)
        except Exception:
            return False

        # Check finite derived values
        for d_var in self.derived_variables:
            d_val = full_cand.get(d_var.name)
            if d_val is None or not np.isfinite(d_val):
                return False

        # 3. Check all constraints
        for constraint in self.constraints:
            if not constraint.evaluate(full_cand):
                return False

        return True

    def sample_unconstrained(self, n: int, seed: int = 42) -> pd.DataFrame:
        """Draws uniform unconstrained samples for all free design variables."""
        rng = np.random.default_rng(seed)
        data: dict[str, np.ndarray] = {}
        for var in self.variables:
            if isinstance(var, ContinuousVariable):
                data[var.name] = var.sample_uniform(rng, size=n)
            elif isinstance(var, DiscreteVariable):
                data[var.name] = var.sample(rng, size=n)
            else:
                raise TypeError(f"Unsupported variable type: {type(var)}")

        df = pd.DataFrame(data)
        # Compute derived variables across batch
        for d_var in self.derived_variables:
            derived_vals = [d_var.compute(row) for _, row in df.iterrows()]
            df[d_var.name] = derived_vals
        return df

    def sample_feasible(
        self,
        n: int,
        seed: int = 42,
        max_oversample_factor: int = 50,
    ) -> pd.DataFrame:
        """Draws deterministic feasible candidate samples using rejection sampling.

        Guarantees that all returned candidates satisfy 100% of declared bounds,
        derived formulas, and feasibility constraints.
        """
        rng = np.random.default_rng(seed)
        batch_size = max(1000, n * 2)
        feasible_rows: list[dict[str, Any]] = []

        total_drawn = 0
        max_draw = n * max_oversample_factor

        while len(feasible_rows) < n and total_drawn < max_draw:
            unconstrained = self.sample_unconstrained(n=batch_size, seed=int(rng.integers(0, 2**31 - 1)))
            total_drawn += batch_size

            for _, row in unconstrained.iterrows():
                row_dict = row.to_dict()
                if self.is_feasible(row_dict):
                    feasible_rows.append(self.compute_derived(row_dict))
                    if len(feasible_rows) >= n:
                        break

        if len(feasible_rows) < n:
            raise RuntimeError(
                f"Failed to sample {n} feasible candidates after {total_drawn} draws (obtained {len(feasible_rows)}). "
                "The search space constraints may be excessively restrictive."
            )

        return pd.DataFrame(feasible_rows[:n])

    def check_novelty(
        self,
        candidates: pd.DataFrame,
        reference_points: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        tol: float = 1e-3,
    ) -> pd.DataFrame:
        """Evaluates whether candidates are novel relative to a reference grid/set of observed points.

        Returns DataFrame containing:
        - min_distance: Euclidean distance to closest reference point in normalized/standard coordinates.
        - nearest_ref_id: Identifier or index of closest reference point.
        - is_novel: Boolean (True if min_distance > tol).
        """
        if feature_cols is None:
            feature_cols = self.all_variable_names

        cols = [c for c in feature_cols if c in candidates.columns and c in reference_points.columns]
        if not cols:
            raise ValueError(f"No matching feature columns found between candidates and reference points: {feature_cols}")

        cand_mat = candidates[cols].to_numpy(dtype=float)
        ref_mat = reference_points[cols].to_numpy(dtype=float)

        # Scale by coordinate ranges to make distance metric dimensionless
        ranges = np.ptp(ref_mat, axis=0)
        ranges = np.where(ranges <= 0, 1.0, ranges)

        scaled_cand = cand_mat / ranges
        scaled_ref = ref_mat / ranges

        min_distances: list[float] = []
        nearest_indices: list[int] = []

        for i in range(len(cand_mat)):
            dists = np.linalg.norm(scaled_ref - scaled_cand[i], axis=1)
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])
            min_distances.append(min_dist)
            nearest_indices.append(min_idx)

        res = pd.DataFrame(
            {
                "min_distance": min_distances,
                "nearest_ref_idx": nearest_indices,
                "is_novel": [d > tol for d in min_distances],
            },
            index=candidates.index,
        )
        return res

    def to_dict(self) -> dict[str, Any]:
        """Serializes search space specification into a dictionary for JSON reporting."""
        var_list = []
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                var_list.append({"name": v.name, "type": "continuous", "lower": v.lower, "upper": v.upper})
            elif isinstance(v, DiscreteVariable):
                var_list.append({"name": v.name, "type": "discrete", "values": list(v.values)})

        derived_list = [{"name": v.name, "depends_on": list(v.depends_on)} for v in self.derived_variables]
        constraint_list = [{"name": c.name, "description": c.description} for c in self.constraints]

        return {
            "search_space_name": self.name,
            "free_variables": var_list,
            "derived_variables": derived_list,
            "constraints": constraint_list,
        }
