"""Offline closed-loop rediscovery benchmark engine for battery process optimization.

Evaluates whether sequential Bayesian optimization policies can rediscover the
retrospectively observed best experimental recipe from sub-optimal starting points
without seeing target outcomes in advance.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.backend import resolve_strategy
from src.optimization.botorch_backend import BoTorchBackend
from src.optimization.objective import OptimizationObjective

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayStepRecord:
    """Record of a single sequential experiment step in rediscovery replay."""

    step: int
    selected_id: str
    revealed_target: float
    best_so_far: float
    simple_regret: float
    cumulative_regret: float
    hidden_best_rank: int
    is_hidden_best: bool
    predicted_mean: float | None = None
    predicted_std: float | None = None
    acquisition_value: float | None = None


@dataclass
class RediscoveryTrajectory:
    """Complete execution trajectory for a single policy and seed."""

    policy: str
    seed: int
    initial_candidate_ids: list[str]
    steps: list[ReplayStepRecord] = field(default_factory=list)
    hidden_best_id: str = ""
    hidden_best_value: float = 0.0
    initial_best_value: float = 0.0
    rediscovered: bool = False
    experiments_to_best: int | None = None
    final_simple_regret: float = 0.0
    final_cumulative_regret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "initial_candidate_ids": list(self.initial_candidate_ids),
            "hidden_best_id": self.hidden_best_id,
            "hidden_best_value": self.hidden_best_value,
            "initial_best_value": self.initial_best_value,
            "rediscovered": self.rediscovered,
            "experiments_to_best": self.experiments_to_best,
            "final_simple_regret": self.final_simple_regret,
            "final_cumulative_regret": self.final_cumulative_regret,
            "steps": [asdict(s) for s in self.steps],
        }


@dataclass
class PolicySummary:
    """Statistical summary of benchmark performance for a single optimization policy."""

    policy: str
    num_seeds: int
    success_rate: float
    mean_experiments_to_best: float | None
    median_experiments_to_best: float | None
    mean_simple_regret: float
    std_simple_regret: float
    mean_cumulative_regret: float
    std_cumulative_regret: float
    hit_rate_at_step: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BlindExperimentalOracle:
    """Firewall-protected oracle maintaining experimental history and hidden outcomes.

    Enforces strict physical constraints:
    - Target values are strictly hidden until explicitly revealed via `reveal()`.
    - `visible_candidates()` provides candidate IDs and control parameters only.
    - Double reveal of the same candidate raises ValueError.
    - Reveal of an unknown candidate raises KeyError.
    """

    def __init__(
        self,
        candidate_pool: pd.DataFrame,
        *,
        candidate_id_column: str = "candidate_id",
        target_column: str = "cell_capacity_mah",
        control_columns: Sequence[str] | None = None,
        minimize: bool = False,
        anonymize: bool = False,
        seed: int | None = None,
    ) -> None:
        if candidate_id_column not in candidate_pool.columns:
            raise KeyError(f"Candidate ID column '{candidate_id_column}' not found in candidate pool.")
        if target_column not in candidate_pool.columns:
            raise KeyError(f"Target column '{target_column}' not found in candidate pool.")
        if candidate_pool[candidate_id_column].isna().any():
            raise ValueError("Candidate ID column contains NaN values.")
        if candidate_pool[target_column].isna().any():
            raise ValueError("Target column contains NaN values in candidate pool.")

        pool_df = candidate_pool.copy().reset_index(drop=True)
        pool_df[candidate_id_column] = pool_df[candidate_id_column].astype(str)

        if control_columns is not None:
            ctrl_cols = list(control_columns)
            missing = [c for c in ctrl_cols if c not in pool_df.columns]
            if missing:
                raise KeyError(f"Specified control columns missing from pool: {missing}")
        else:
            exclude = {candidate_id_column, target_column, "run_id", "batch_id", "cell_id"}
            ctrl_cols = [
                c for c in pool_df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(pool_df[c])
            ]

        if not ctrl_cols:
            raise ValueError("No numeric control columns identified in candidate pool.")

        # Check candidate ID uniqueness
        if pool_df[candidate_id_column].duplicated().any():
            dups = pool_df[candidate_id_column][pool_df[candidate_id_column].duplicated()].unique().tolist()
            raise ValueError(f"Candidate IDs must be unique; duplicates found: {dups}")

        self._candidate_id_column = candidate_id_column
        self._target_column = target_column
        self._control_columns = ctrl_cols
        self._minimize = minimize

        # Optional anonymization
        self._anonymize = anonymize
        if anonymize:
            rng = np.random.default_rng(seed)
            shuffled_ids = pool_df[candidate_id_column].tolist()
            rng.shuffle(shuffled_ids)
            self._real_to_anon = {real_id: f"cand_{idx:03d}" for idx, real_id in enumerate(shuffled_ids)}
            self._anon_to_real = {v: k for k, v in self._real_to_anon.items()}
            pool_df["_original_id"] = pool_df[candidate_id_column]
            pool_df[candidate_id_column] = pool_df[candidate_id_column].map(self._real_to_anon)
        else:
            self._real_to_anon = {}
            self._anon_to_real = {}

        self._full_pool = pool_df[[self._candidate_id_column, self._target_column] + self._control_columns].copy()

        # Identify hidden best
        target_series = self._full_pool[self._target_column]
        best_idx = int(target_series.argmin() if minimize else target_series.argmax())
        best_row = self._full_pool.iloc[best_idx]
        self._hidden_best_id = str(best_row[self._candidate_id_column])
        self._hidden_best_value = float(best_row[self._target_column])

        # State tracking
        self._revealed_ids: set[str] = set()
        self._revealed_records: list[dict[str, Any]] = []

    @property
    def candidate_id_column(self) -> str:
        return self._candidate_id_column

    @property
    def target_column(self) -> str:
        return self._target_column

    @property
    def control_columns(self) -> list[str]:
        return list(self._control_columns)

    @property
    def minimize(self) -> bool:
        return self._minimize

    @property
    def num_total(self) -> int:
        return len(self._full_pool)

    @property
    def num_revealed(self) -> int:
        return len(self._revealed_ids)

    @property
    def num_unrevealed(self) -> int:
        return len(self._full_pool) - len(self._revealed_ids)

    @property
    def hidden_best_id(self) -> str:
        """ID of the retrospective best recipe (for post-run evaluation metrics only)."""
        return self._hidden_best_id

    @property
    def hidden_best_value(self) -> float:
        """Target value of the retrospective best recipe (for post-run evaluation metrics only)."""
        return self._hidden_best_value

    def is_revealed(self, candidate_id: str) -> bool:
        return str(candidate_id) in self._revealed_ids

    def visible_candidates(self) -> pd.DataFrame:
        """Returns unrevealed candidates.
        
        Strict firewall guarantee: Contains ONLY candidate ID and control features.
        Target values and outcome records are strictly omitted.
        """
        unrevealed_mask = ~self._full_pool[self._candidate_id_column].isin(self._revealed_ids)
        visible = self._full_pool.loc[unrevealed_mask, [self._candidate_id_column] + self._control_columns].copy()
        return visible.reset_index(drop=True)

    def revealed_history(self) -> pd.DataFrame:
        """Returns DataFrame of all revealed candidate experiments with outcomes."""
        if not self._revealed_records:
            cols = [self._candidate_id_column, self._target_column] + self._control_columns
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self._revealed_records)

    def reveal(self, candidate_id: str) -> dict[str, Any]:
        """Reveals true target value for an unrevealed candidate experiment.
        
        Raises:
            KeyError: if candidate_id is not in candidate pool.
            ValueError: if candidate_id has already been revealed.
        """
        cid = str(candidate_id)
        if cid in self._revealed_ids:
            raise ValueError(f"Candidate '{cid}' has already been revealed.")

        match = self._full_pool.loc[self._full_pool[self._candidate_id_column] == cid]
        if match.empty:
            raise KeyError(f"Candidate '{cid}' not found in experimental candidate pool.")

        row = match.iloc[0]
        record = {
            self._candidate_id_column: cid,
            self._target_column: float(row[self._target_column]),
        }
        for col in self._control_columns:
            record[col] = float(row[col])

        self._revealed_ids.add(cid)
        self._revealed_records.append(record)
        return record

    def secret_target(self, candidate_id: str, *, allow_evaluation: bool = False) -> float:
        """Read target value without reveal. Protected against cheating."""
        if not allow_evaluation:
            raise PermissionError("Accessing ground-truth target without calling reveal() is forbidden by oracle firewall.")
        cid = str(candidate_id)
        match = self._full_pool.loc[self._full_pool[self._candidate_id_column] == cid]
        if match.empty:
            raise KeyError(f"Candidate '{cid}' not found in candidate pool.")
        return float(match.iloc[0][self._target_column])


class RecipeAggregation:
    """Consolidates experimental replicates sharing identical process controls."""

    @staticmethod
    def aggregate_recipes(
        df: pd.DataFrame,
        control_columns: Sequence[str],
        target_column: str,
        *,
        id_column: str = "candidate_id",
        agg: str = "mean",
    ) -> pd.DataFrame:
        """Aggregates candidate records with identical controls into distinct recipes."""
        ctrls = list(control_columns)
        if df.empty:
            return pd.DataFrame(columns=[id_column, target_column] + ctrls)

        grouped = df.groupby(ctrls, as_index=False).agg({
            target_column: agg,
            id_column: "first",
        })
        return grouped[[id_column, target_column] + ctrls].reset_index(drop=True)


class RediscoveryReplay:
    """Sequential Bayesian optimization closed-loop replay coordinator."""

    def __init__(
        self,
        candidate_pool: pd.DataFrame,
        *,
        candidate_id_column: str = "candidate_id",
        target_column: str = "cell_capacity_mah",
        control_columns: Sequence[str] | None = None,
        minimize: bool = False,
        backend: Any | None = None,
    ) -> None:
        self._candidate_pool = candidate_pool.copy()
        self._candidate_id_column = candidate_id_column
        self._target_column = target_column
        self._control_columns = control_columns
        self._minimize = minimize
        self._backend = backend or BoTorchBackend()

    def _create_oracle(self) -> BlindExperimentalOracle:
        return BlindExperimentalOracle(
            self._candidate_pool,
            candidate_id_column=self._candidate_id_column,
            target_column=self._target_column,
            control_columns=self._control_columns,
            minimize=self._minimize,
        )

    def run(
        self,
        strategy: str,
        seed: int,
        *,
        initial_size: int = 3,
        max_steps: int | None = None,
        beta: float = 2.0,
    ) -> RediscoveryTrajectory:
        """Executes offline closed-loop rediscovery replay for a given policy and seed.
        
        Guarantees:
        - Fresh oracle initialized per replay run.
        - Initial design strictly excludes hidden best candidate.
        - Policy queries strictly see visible candidates (controls only) and revealed history.
        - Step-level metrics (regret, best so far, surrogate rank of hidden best) recorded.
        """
        oracle = self._create_oracle()
        hidden_best = oracle.hidden_best_id
        all_candidate_ids = oracle.visible_candidates()[oracle.candidate_id_column].tolist()

        # Filter out hidden best from pool eligible for initial design
        initial_eligible_pool = [cid for cid in all_candidate_ids if cid != hidden_best]
        if len(initial_eligible_pool) < initial_size:
            raise ValueError(
                f"Candidate pool size ({len(all_candidate_ids)}) insufficient for "
                f"initial size {initial_size} excluding best candidate."
            )

        rng = np.random.default_rng(seed)
        sampled_initial = rng.choice(initial_eligible_pool, size=initial_size, replace=False).tolist()

        # Reveal initial observations
        initial_values: list[float] = []
        for cid in sampled_initial:
            rec = oracle.reveal(cid)
            initial_values.append(rec[oracle.target_column])

        # Baseline best from initial design
        best_so_far = min(initial_values) if self._minimize else max(initial_values)
        initial_best_value = best_so_far

        trajectory = RediscoveryTrajectory(
            policy=strategy,
            seed=seed,
            initial_candidate_ids=list(sampled_initial),
            hidden_best_id=hidden_best,
            hidden_best_value=oracle.hidden_best_value,
            initial_best_value=initial_best_value,
        )

        budget = max_steps if max_steps is not None else oracle.num_unrevealed
        cumulative_regret = 0.0

        canonical_strat = resolve_strategy(strategy)

        for step in range(1, budget + 1):
            if oracle.num_unrevealed == 0:
                break

            visible = oracle.visible_candidates()
            revealed = oracle.revealed_history()

            selected_id: str
            pred_mean: float | None = None
            pred_std: float | None = None
            acq_val: float | None = None
            hidden_rank: int = 1

            if canonical_strat == "random":
                step_rng = np.random.default_rng(seed * 1000 + step)
                visible_ids = visible[oracle.candidate_id_column].tolist()
                selected_id = str(step_rng.choice(visible_ids))
                if hidden_best in visible_ids:
                    # Deterministic pseudo-rank among remaining candidates under random
                    perm = list(visible_ids)
                    step_rng.shuffle(perm)
                    hidden_rank = perm.index(hidden_best) + 1
                else:
                    hidden_rank = 1
            else:
                obj = OptimizationObjective(target_name=oracle.target_column, minimize=self._minimize)
                proposals = self._backend.propose(
                    observations=revealed,
                    candidate_pool=visible,
                    objective=obj,
                    feature_columns=oracle.control_columns,
                    candidate_id_column=oracle.candidate_id_column,
                    n=len(visible),
                    seed=seed * 1000 + step,
                    strategy=strategy,
                    beta=beta,
                )

                if not proposals:
                    raise RuntimeError("Surrogate proposal returned empty candidate list.")

                top_prop = proposals[0]
                selected_id = str(top_prop.candidate_id)
                pred_mean = top_prop.predicted_mean
                pred_std = top_prop.predicted_std
                acq_val = top_prop.acquisition_value

                # Find surrogate rank of hidden best among unrevealed candidates
                prop_ids = [str(p.candidate_id) for p in proposals]
                if hidden_best in prop_ids:
                    hidden_rank = prop_ids.index(hidden_best) + 1
                else:
                    hidden_rank = 1

            # Reveal selected candidate
            record = oracle.reveal(selected_id)
            revealed_val = float(record[oracle.target_column])

            is_hidden = (selected_id == hidden_best)
            if is_hidden and not trajectory.rediscovered:
                trajectory.rediscovered = True
                trajectory.experiments_to_best = step

            if self._minimize:
                best_so_far = min(best_so_far, revealed_val)
                simple_regret = max(0.0, best_so_far - oracle.hidden_best_value)
            else:
                best_so_far = max(best_so_far, revealed_val)
                simple_regret = max(0.0, oracle.hidden_best_value - best_so_far)

            cumulative_regret += simple_regret

            step_record = ReplayStepRecord(
                step=step,
                selected_id=selected_id,
                revealed_target=revealed_val,
                best_so_far=best_so_far,
                simple_regret=simple_regret,
                cumulative_regret=cumulative_regret,
                hidden_best_rank=hidden_rank,
                is_hidden_best=is_hidden,
                predicted_mean=pred_mean,
                predicted_std=pred_std,
                acquisition_value=acq_val,
            )
            trajectory.steps.append(step_record)

        trajectory.final_simple_regret = trajectory.steps[-1].simple_regret if trajectory.steps else 0.0
        trajectory.final_cumulative_regret = trajectory.steps[-1].cumulative_regret if trajectory.steps else 0.0

        return trajectory


def summarize_trajectories(trajectories: Sequence[RediscoveryTrajectory]) -> PolicySummary:
    """Aggregates trajectories across seeds for a given policy into a statistical summary."""
    if not trajectories:
        raise ValueError("Cannot summarize empty trajectory list.")

    policy = trajectories[0].policy
    num_seeds = len(trajectories)
    rediscovered_count = sum(1 for t in trajectories if t.rediscovered)
    success_rate = rediscovered_count / num_seeds

    successful_steps = [t.experiments_to_best for t in trajectories if t.experiments_to_best is not None]
    mean_exp_to_best = float(np.mean(successful_steps)) if successful_steps else None
    median_exp_to_best = float(np.median(successful_steps)) if successful_steps else None

    final_regrets = [t.final_simple_regret for t in trajectories]
    cum_regrets = [t.final_cumulative_regret for t in trajectories]

    mean_simple_regret = float(np.mean(final_regrets))
    std_simple_regret = float(np.std(final_regrets))
    mean_cum_regret = float(np.mean(cum_regrets))
    std_cum_regret = float(np.std(cum_regrets))

    # Compute top-1 hit rate at each step index
    max_steps = max(len(t.steps) for t in trajectories)
    hit_rate_at_step: dict[int, float] = {}
    for step_idx in range(1, max_steps + 1):
        hits = sum(
            1 for t in trajectories
            if t.experiments_to_best is not None and t.experiments_to_best <= step_idx
        )
        hit_rate_at_step[step_idx] = hits / num_seeds

    return PolicySummary(
        policy=policy,
        num_seeds=num_seeds,
        success_rate=success_rate,
        mean_experiments_to_best=mean_exp_to_best,
        median_experiments_to_best=median_exp_to_best,
        mean_simple_regret=mean_simple_regret,
        std_simple_regret=std_simple_regret,
        mean_cumulative_regret=mean_cum_regret,
        std_cumulative_regret=std_cum_regret,
        hit_rate_at_step=hit_rate_at_step,
    )


def run_rediscovery_benchmark(
    candidate_pool: pd.DataFrame,
    *,
    candidate_id_column: str = "candidate_id",
    target_column: str = "cell_capacity_mah",
    control_columns: Sequence[str] | None = None,
    minimize: bool = False,
    policies: Sequence[str] = ("random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement"),
    seeds: Sequence[int] = (11, 23, 42, 67, 101, 137, 179, 223, 281, 353),
    initial_size: int = 3,
    max_steps: int | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    """Runs full multi-policy, multi-seed offline closed-loop rediscovery benchmark."""
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column=candidate_id_column,
        target_column=target_column,
        control_columns=control_columns,
        minimize=minimize,
        backend=backend,
    )

    all_trajectories: dict[str, list[RediscoveryTrajectory]] = {}
    policy_summaries: list[PolicySummary] = []

    for pol in policies:
        pol_trajectories: list[RediscoveryTrajectory] = []
        for seed in seeds:
            traj = replay.run(
                strategy=pol,
                seed=seed,
                initial_size=initial_size,
                max_steps=max_steps,
            )
            pol_trajectories.append(traj)
        all_trajectories[pol] = pol_trajectories
        summary = summarize_trajectories(pol_trajectories)
        policy_summaries.append(summary)

    return {
        "candidate_id_column": candidate_id_column,
        "target_column": target_column,
        "control_columns": replay._control_columns,
        "minimize": minimize,
        "candidate_pool_size": len(candidate_pool),
        "initial_size": initial_size,
        "seeds": list(seeds),
        "policies": list(policies),
        "trajectories": {
            pol: [t.to_dict() for t in trajs]
            for pol, trajs in all_trajectories.items()
        },
        "summaries": [s.to_dict() for s in policy_summaries],
    }
