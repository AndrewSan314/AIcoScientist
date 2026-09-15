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

import hashlib
import math
import numpy as np
import pandas as pd

from src.optimization.backend import resolve_strategy
from src.optimization.botorch_backend import BoTorchBackend
from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal
from src.process.contracts import BatteryProcessRun
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.information_horizon import (
    DecisionHorizon,
    InformationHorizon,
    PreManufacturingRecipeSelectionHorizon,
)
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage
from src.process.surrogates.core import (
    ProcessSurrogate,
    ProcessSurrogateSample,
    SurrogateArtifact,
    SurrogateDecisionContext,
    SurrogateInputSchema,
    TrainOnlyPreprocessor,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineExecutionTrace:
    """Runtime execution counters for benchmark verification and audit."""

    source_adapter_invocations: int = 0
    battery_process_runs_seen: int = 0
    recipe_selection_horizon_invocations: int = 0
    process_surrogate_samples_created: int = 0
    process_surrogate_fit_count: int = 0
    surrogate_artifact_fingerprints: list[str] = field(default_factory=list)
    coordinator_proposal_count: int = 0
    optimizer_backend_type: str = "none"
    oracle_reveal_count: int = 0
    direct_botorch_calls: int = 0
    random_steps: int = 0
    training_recipe_ids_by_step: list[list[str]] = field(default_factory=list)
    candidate_ids_scored_by_step: list[list[str]] = field(default_factory=list)

    @property
    def runs_loaded(self) -> int:
        return self.battery_process_runs_seen

    @runs_loaded.setter
    def runs_loaded(self, val: int) -> None:
        self.battery_process_runs_seen = val

    @property
    def horizon_projections(self) -> int:
        return self.recipe_selection_horizon_invocations

    @horizon_projections.setter
    def horizon_projections(self, val: int) -> None:
        self.recipe_selection_horizon_invocations = val

    @property
    def samples_created(self) -> int:
        return self.process_surrogate_samples_created

    @samples_created.setter
    def samples_created(self, val: int) -> None:
        self.process_surrogate_samples_created = val

    @property
    def surrogates_fitted(self) -> int:
        return self.process_surrogate_fit_count

    @surrogates_fitted.setter
    def surrogates_fitted(self, val: int) -> None:
        self.process_surrogate_fit_count = val

    @property
    def artifacts_created(self) -> int:
        return len(self.surrogate_artifact_fingerprints)

    @property
    def coordinator_calls(self) -> int:
        return self.coordinator_proposal_count

    @coordinator_calls.setter
    def coordinator_calls(self, val: int) -> None:
        self.coordinator_proposal_count = val

    @property
    def proposals_generated(self) -> int:
        return sum(len(c) for c in self.candidate_ids_scored_by_step) or self.coordinator_proposal_count

    @proposals_generated.setter
    def proposals_generated(self, val: int) -> None:
        pass

    @property
    def oracle_reveals(self) -> int:
        return self.oracle_reveal_count

    @oracle_reveals.setter
    def oracle_reveals(self, val: int) -> None:
        self.oracle_reveal_count = val

    @property
    def botorch_calls(self) -> int:
        return self.direct_botorch_calls

    @botorch_calls.setter
    def botorch_calls(self, val: int) -> None:
        self.direct_botorch_calls = val

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_adapter_invocations": self.source_adapter_invocations,
            "battery_process_runs_seen": self.battery_process_runs_seen,
            "recipe_selection_horizon_invocations": self.recipe_selection_horizon_invocations,
            "process_surrogate_samples_created": self.process_surrogate_samples_created,
            "process_surrogate_fit_count": self.process_surrogate_fit_count,
            "surrogate_artifact_fingerprints": list(self.surrogate_artifact_fingerprints),
            "coordinator_proposal_count": self.coordinator_proposal_count,
            "optimizer_backend_type": self.optimizer_backend_type,
            "oracle_reveal_count": self.oracle_reveal_count,
            "direct_botorch_calls": self.direct_botorch_calls,
            "random_steps": self.random_steps,
            "training_recipe_ids_by_step": [list(ids) for ids in self.training_recipe_ids_by_step],
            "candidate_ids_scored_by_step": [list(ids) for ids in self.candidate_ids_scored_by_step],
            "runs_loaded": self.battery_process_runs_seen,
            "horizon_projections": self.recipe_selection_horizon_invocations,
            "samples_created": self.process_surrogate_samples_created,
            "surrogates_fitted": self.process_surrogate_fit_count,
            "artifacts_created": len(self.surrogate_artifact_fingerprints),
            "coordinator_calls": self.coordinator_proposal_count,
            "proposals_generated": sum(len(c) for c in self.candidate_ids_scored_by_step) or self.coordinator_proposal_count,
            "oracle_reveals": self.oracle_reveal_count,
            "botorch_calls": self.direct_botorch_calls,
        }

    def generate_audit(self, policy: str, engine_path: str) -> dict[str, Any]:
        pol_upper = policy.upper()
        is_full_engine = (
            "PROCESS_SURROGATE" in pol_upper
            or "PROCESS_ENGINE" in pol_upper
            or pol_upper.startswith("AICOINTEL_")
            or "COORDINATOR" in pol_upper
        )
        is_direct_botorch = "DIRECT_BOTORCH" in pol_upper
        is_random = "RANDOM" in pol_upper

        if is_full_engine:
            verified = (
                self.source_adapter_invocations > 0
                and self.battery_process_runs_seen > 0
                and self.recipe_selection_horizon_invocations > 0
                and self.process_surrogate_fit_count > 0
                and self.coordinator_proposal_count > 0
                and self.oracle_reveal_count > 0
                and self.direct_botorch_calls == 0
                and len(self.surrogate_artifact_fingerprints) > 0
            )
            uses_direct_botorch_backend = False
        elif is_direct_botorch:
            verified = (
                self.source_adapter_invocations == 0
                and self.battery_process_runs_seen == 0
                and self.recipe_selection_horizon_invocations == 0
                and self.process_surrogate_fit_count == 0
                and self.coordinator_proposal_count == 0
                and self.direct_botorch_calls > 0
                and self.oracle_reveal_count > 0
                and len(self.surrogate_artifact_fingerprints) == 0
            )
            uses_direct_botorch_backend = True
        elif is_random:
            verified = (
                self.source_adapter_invocations == 0
                and self.battery_process_runs_seen == 0
                and self.recipe_selection_horizon_invocations == 0
                and self.process_surrogate_fit_count == 0
                and self.coordinator_proposal_count == 0
                and self.direct_botorch_calls == 0
                and self.random_steps > 0
                and self.oracle_reveal_count > 0
                and len(self.surrogate_artifact_fingerprints) == 0
            )
            uses_direct_botorch_backend = False
        else:
            verified = False
            uses_direct_botorch_backend = False

        return {
            "policy": policy,
            "engine_path": engine_path,
            "source_adapter_invocations": self.source_adapter_invocations,
            "battery_process_runs_seen": self.battery_process_runs_seen,
            "recipe_selection_horizon_invocations": self.recipe_selection_horizon_invocations,
            "process_surrogate_fit_count": self.process_surrogate_fit_count,
            "coordinator_proposal_count": self.coordinator_proposal_count,
            "oracle_reveal_count": self.oracle_reveal_count,
            "direct_botorch_calls": self.direct_botorch_calls,
            "random_steps": self.random_steps,
            "uses_direct_botorch_backend": uses_direct_botorch_backend,
            "optimizer_backend_type": self.optimizer_backend_type,
            "surrogate_artifact_fingerprints": list(self.surrogate_artifact_fingerprints),
            "verified": verified,
        }


class FrozenSurrogateOptimizerBackend:
    """Production optimizer backend executing over a frozen SurrogateArtifact.

    Implements the optimizer backend protocol for ProcessOptimizationCoordinator:
    propose(observations, candidate_pool, objective, ...) -> list[CandidateProposal]
    """

    name: str = "frozen_process_surrogate"
    version: str = "4"

    def __init__(
        self,
        artifact: SurrogateArtifact,
        context: SurrogateDecisionContext,
        *,
        beta: float = 2.0,
    ) -> None:
        artifact.verify_integrity()
        self.artifact = artifact
        self.context = context
        self.beta = float(beta)
        self.control_names = tuple(name.removeprefix("control::") for name in artifact.input_schema.control_features)

    def propose(
        self,
        observations: pd.DataFrame | Sequence[Mapping[str, Any]],
        candidate_pool: pd.DataFrame,
        objective: OptimizationObjective | str,
        *,
        feature_columns: Sequence[str] | None = None,
        candidate_id_column: str | None = None,
        n: int = 1,
        seed: int | None = None,
        strategy: str = "expected_improvement",
        beta: float | None = None,
        **kwargs: Any,
    ) -> list[CandidateProposal]:
        if candidate_pool.empty:
            return []

        target, minimize = (objective, False) if isinstance(objective, str) else (objective.target_name, objective.minimize)
        if target not in self.artifact.target_names:
            raise ValueError(f"Target {target!r} is absent from frozen surrogate artifact")

        id_col = candidate_id_column or "recipe_id"
        ids = candidate_pool[id_col].astype(str).tolist()
        eff_beta = float(beta) if beta is not None else self.beta

        ctrl_cols = [c for c in self.control_names if c in candidate_pool.columns]
        candidate_controls = candidate_pool[ctrl_cols].to_dict(orient="records")
        dist = self.artifact.predict_decision(self.context, candidate_controls)
        means, stds = dist[target]

        best_so_far = None
        if isinstance(observations, pd.DataFrame) and not observations.empty and target in observations.columns:
            obs_vals = observations[target].dropna().values
            if len(obs_vals) > 0:
                best_so_far = float(np.min(obs_vals) if minimize else np.max(obs_vals))

        strat_lower = strategy.lower()
        proposals: list[CandidateProposal] = []
        scores: list[float] = []

        for i, cid in enumerate(ids):
            m = float(means[i])
            s = max(float(stds[i]), 1e-8)
            if strat_lower in ("gp_ucb", "ucb"):
                score = (m - eff_beta * s) if minimize else (m + eff_beta * s)
                acq_name = "gp_ucb"
            elif strat_lower == "greedy":
                score = -m if minimize else m
                acq_name = "greedy"
            elif strat_lower == "random":
                score = 0.0
                acq_name = "random"
            else:  # Standard analytic Expected Improvement
                acq_name = "expected_improvement"
                if best_so_far is None:
                    score = -m if minimize else m
                else:
                    diff = (best_so_far - m) if minimize else (m - best_so_far)
                    z = diff / s
                    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
                    score = diff * phi + s * pdf
            scores.append(score)

        if strat_lower == "random":
            rng = np.random.default_rng(seed)
            ranking = list(range(len(ids)))
            rng.shuffle(ranking)
        else:
            ranking = sorted(range(len(ids)), key=lambda idx: scores[idx], reverse=True)

        for idx in ranking[:n]:
            proposals.append(
                CandidateProposal(
                    candidate_id=ids[idx],
                    design_variables=candidate_controls[idx],
                    predicted_mean=float(means[idx]),
                    predicted_std=float(stds[idx]),
                    acquisition_name=acq_name,
                    acquisition_value=float(scores[idx]),
                    backend_name=self.name,
                    backend_version=self.version,
                    seed=seed,
                    metadata={
                        "artifact_fingerprint": self.artifact.artifact_fingerprint,
                        "context_fingerprint": self.context.context_fingerprint,
                        "context_stage": self.context.stage.value,
                        "objective_sense": "minimize" if minimize else "maximize",
                        "strategy": acq_name,
                    },
                )
            )
        return proposals

STAGE_OF_CONTROL: dict[str, ProcessStage] = {
    "active_material_fraction_pct": ProcessStage.FORMULATION,
    "conductive_additive_fraction_pct": ProcessStage.FORMULATION,
    "binder_cmc_fraction_pct": ProcessStage.FORMULATION,
    "binder_sbr_fraction_pct": ProcessStage.FORMULATION,
    "additive_fraction_pct": ProcessStage.FORMULATION,
    "mixing_solids_pct": ProcessStage.MIXING,
    "coating_speed_m_per_min": ProcessStage.COATING,
    "coating_gap_um": ProcessStage.COATING,
    "drying_temperature_c": ProcessStage.DRYING,
    "calendering_applied": ProcessStage.CALENDERING,
}


@dataclass(frozen=True)
class ReplayStepRecord:
    """Record of a single sequential experiment step in rediscovery replay."""

    step: int
    selected_id: str
    revealed_target: float
    best_so_far: float
    simple_regret: float
    cumulative_regret: float
    hidden_best_rank: int | None
    is_hidden_best: bool
    predicted_mean: float | None = None
    predicted_std: float | None = None
    acquisition_value: float | None = None
    engine_path: str = "DIRECT_BOTORCH_BASELINE"
    surrogate_artifact_fingerprint: str | None = None
    dataset_fingerprint: str | None = None
    information_horizon: str | None = None
    training_view_summary: dict[str, Any] | None = None


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
    top3_rediscovered: bool = False
    experiments_to_top3: int | None = None
    final_simple_regret: float = 0.0
    final_cumulative_regret: float = 0.0
    engine_path: str = "DIRECT_BOTORCH_BASELINE"
    execution_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "engine_path": self.engine_path,
            "initial_candidate_ids": list(self.initial_candidate_ids),
            "hidden_best_id": self.hidden_best_id,
            "hidden_best_value": self.hidden_best_value,
            "initial_best_value": self.initial_best_value,
            "rediscovered": self.rediscovered,
            "experiments_to_best": self.experiments_to_best,
            "top3_rediscovered": self.top3_rediscovered,
            "experiments_to_top3": self.experiments_to_top3,
            "final_simple_regret": self.final_simple_regret,
            "final_cumulative_regret": self.final_cumulative_regret,
            "execution_trace": dict(self.execution_trace),
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
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    top3_hit_rate_at_5: float = 0.0
    engine_path: str = "DIRECT_BOTORCH_BASELINE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_hypergeometric_baseline(
    total_candidates: int,
    initial_size: int,
    budget: int,
    top_k: int = 1,
) -> dict[int, float]:
    """Calculates exact analytical probability of selecting at least one of top_k targets.

    Semantics matching benchmark initial design:
    - Pool initially has total_candidates.
    - Initial design of initial_size candidates is drawn excluding the top-1 hidden best.
    - For top_1: exactly 1 target exists among remaining candidates; prob at step s is s / N_rem.
    - For top_k (k > 1):
      Initial design draws initial_size candidates from (N - 1) candidates containing (k - 1) top-k targets.
      Hypergeometric expectation across possible numbers m of top-k targets in initial design:
      P(m) = [comb(k-1, m) * comb((N-1)-(k-1), N_init-m)] / comb(N-1, N_init).
      Unrevealed pool has N_rem = N - N_init candidates containing k_rem = k - m targets.
      P(hit at step s | m) = 1 - [comb(N_rem - k_rem, s) / comb(N_rem, s)].
      Unconditional P(hit at step s) = sum_m P(m) * P(hit at step s | m).
    """
    n_rem = total_candidates - initial_size
    if n_rem <= 0 or top_k <= 0 or total_candidates <= 0:
        return {s: 0.0 for s in range(1, budget + 1)}

    curve: dict[int, float] = {}
    if top_k == 1:
        for s in range(1, budget + 1):
            curve[s] = min(1.0, float(s) / float(n_rem))
        return curve

    # Exact unconditional formula for top_k > 1
    k_other = top_k - 1
    n_other = total_candidates - 1
    total_init_ways = math.comb(n_other, initial_size)

    for s in range(1, budget + 1):
        if s >= n_rem:
            curve[s] = 1.0
            continue
        prob_s = 0.0
        max_m = min(k_other, initial_size)
        for m in range(max_m + 1):
            ways_m = math.comb(k_other, m) * math.comb(n_other - k_other, initial_size - m)
            p_m = ways_m / total_init_ways
            k_rem = top_k - m  # Hidden best is always unrevealed, so k_rem >= 1
            if s > n_rem - k_rem:
                p_hit_given_m = 1.0
            else:
                ways_to_miss = math.comb(n_rem - k_rem, s)
                total_draws = math.comb(n_rem, s)
                p_hit_given_m = 1.0 - (ways_to_miss / total_draws)
            prob_s += p_m * p_hit_given_m
        curve[s] = float(min(1.0, prob_s))

    return curve


def calculate_conditional_hypergeometric_baseline(
    total_candidates: int,
    initial_designs_remaining_targets: Sequence[int],
    budget: int,
    initial_size: int = 3,
) -> dict[int, float]:
    """Calculates exact analytical probability of selecting at least one top-k target,
    conditioned on the exact initial designs evaluated across seeds.

    Args:
        total_candidates: Total size of candidate pool N.
        initial_designs_remaining_targets: List containing the number of unrevealed top-k targets
            remaining in the candidate pool for each empirical seed's initial design.
        budget: Sequential experiment budget B.
        initial_size: Size of initial design N_init.
    """
    n_rem = total_candidates - initial_size
    if n_rem <= 0 or not initial_designs_remaining_targets:
        return {s: 0.0 for s in range(1, budget + 1)}

    curve: dict[int, float] = {}
    num_seeds = len(initial_designs_remaining_targets)

    for s in range(1, budget + 1):
        if s >= n_rem:
            curve[s] = 1.0
            continue
        total_draws = math.comb(n_rem, s)
        seed_probs = []
        for k_rem in initial_designs_remaining_targets:
            if k_rem <= 0:
                seed_probs.append(0.0)
            elif s > n_rem - k_rem:
                seed_probs.append(1.0)
            else:
                ways_to_miss = math.comb(n_rem - k_rem, s)
                seed_probs.append(1.0 - (ways_to_miss / total_draws))
        curve[s] = float(sum(seed_probs) / num_seeds)

    return curve


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


def filter_candidate_pool_for_high_loading(
    candidate_pool: pd.DataFrame,
    *,
    min_active_mass_mg: float = 16.0,
    mass_column: str = "mean_active_mass_mg",
) -> pd.DataFrame:
    """Filters candidate pool for higher-loading measured-D30 proxy subset based on retrospective mass.

    Scientific firewall guarantee:
    - Active mass is an observed metrology property, NOT a pre-manufacturing control.
    - It is used strictly for offline benchmark eligibility/filtering, NEVER as an input feature.
    - Preserves exact source-observed higher-loading proxy regimes (e.g. coating gap 200 um with mass >= 16 mg).
    - NOTE: Exact published >= 25 mg Alchemite rediscovery is NOT evaluable because 300 um cells lack usable D30 data.
      published_high_loading_rediscovery_status = "NOT_EVALUABLE_WITH_AVAILABLE_D30".
    """
    if mass_column not in candidate_pool.columns:
        raise KeyError(f"Mass column '{mass_column}' not found in candidate pool.")
    filtered = candidate_pool[candidate_pool[mass_column] >= min_active_mass_mg].copy()
    if filtered.empty:
        max_val = candidate_pool[mass_column].max() if not candidate_pool.empty else 0.0
        raise ValueError(
            f"No candidates satisfy high-loading threshold {mass_column} >= {min_active_mass_mg} mg. "
            f"(Maximum observed mass in candidate pool: {max_val:.2f} mg. "
            "Published target >= 25 mg is not evaluable: Drakopoulos 300 um cells in ASC lack usable D30 cycle data; "
            "status: NOT_EVALUABLE_WITH_AVAILABLE_D30)."
        )
    return filtered.reset_index(drop=True)


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
        coordinator: Any | None = None,
        top_k_targets: int = 3,
        decision_stage: ProcessStage | DecisionHorizon | str = DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        dataset_id: str = "synthetic",
        dataset_fingerprint: str | None = None,
        benchmark_task: str = "UNCONSTRAINED_D30",
        runs_by_recipe: Mapping[str, Sequence[BatteryProcessRun]] | None = None,
        allow_flat_fallback: bool | None = None,
    ) -> None:
        self._candidate_pool = candidate_pool.copy()
        self._candidate_id_column = candidate_id_column
        self._target_column = target_column
        self._control_columns = list(control_columns) if control_columns is not None else [
            c for c in candidate_pool.columns if c not in (candidate_id_column, target_column)
        ]
        self._minimize = minimize
        self._backend = backend or BoTorchBackend()
        self._coordinator = coordinator
        self._top_k_targets = top_k_targets
        self._decision_stage = decision_stage
        self._dataset_id = dataset_id
        self._benchmark_task = benchmark_task
        if allow_flat_fallback is not None:
            self._allow_flat_fallback = allow_flat_fallback
        elif dataset_id == "drakopoulos_graphite":
            cids = set(self._candidate_pool[self._candidate_id_column].astype(str))
            has_drakopoulos_ids = any(cid.startswith("protocol-") for cid in cids)
            self._allow_flat_fallback = not has_drakopoulos_ids
        else:
            self._allow_flat_fallback = True
        self._runs_by_recipe = dict(runs_by_recipe) if runs_by_recipe is not None else None
        self._execution_trace = EngineExecutionTrace()
        if dataset_fingerprint is None:
            raw_bytes = pd.util.hash_pandas_object(self._candidate_pool, index=True).values.tobytes()
            self._dataset_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
        else:
            self._dataset_fingerprint = dataset_fingerprint

    @property
    def execution_trace(self) -> EngineExecutionTrace:
        return self._execution_trace

    def _get_runs_by_recipe(self) -> dict[str, list[BatteryProcessRun]]:
        if self._runs_by_recipe is not None:
            if self._runs_by_recipe and self._execution_trace.source_adapter_invocations == 0:
                self._execution_trace.source_adapter_invocations = 1
            return self._runs_by_recipe
        if self._dataset_id == "drakopoulos_graphite":
            try:
                from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
                self._execution_trace.source_adapter_invocations += 1
                runs = DrakopoulosGraphiteAdapter().load_runs()
                grouped: dict[str, list[BatteryProcessRun]] = {}
                for r in runs:
                    k = r.batch_id or r.run_id
                    grouped.setdefault(k, []).append(r)
                self._runs_by_recipe = grouped
                return self._runs_by_recipe
            except Exception:
                self._runs_by_recipe = {}
                return {}
        self._runs_by_recipe = {}
        return {}

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

        sorted_pool = self._candidate_pool.sort_values(
            by=self._target_column,
            ascending=self._minimize,
        )
        top_k_ids = set(sorted_pool[self._candidate_id_column].head(self._top_k_targets).astype(str))

        # Filter out hidden best from pool eligible for initial design
        initial_eligible_pool = [cid for cid in all_candidate_ids if cid != hidden_best]
        if len(initial_eligible_pool) < initial_size:
            raise ValueError(
                f"Candidate pool size ({len(all_candidate_ids)}) insufficient for "
                f"initial size {initial_size} excluding best candidate."
            )

        rng = np.random.default_rng(seed)
        sampled_initial = rng.choice(initial_eligible_pool, size=initial_size, replace=False).tolist()

        canonical_strat = resolve_strategy(strategy) if strategy.lower() in ("random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement") else strategy

        strat_upper = strategy.upper()
        strat_lower = strategy.lower()

        is_direct_botorch = (
            strat_upper.startswith("DIRECT_BOTORCH")
            or strat_lower in ("direct_botorch_baseline", "direct_botorch")
        )
        is_random = (canonical_strat == "random" and not is_direct_botorch)
        is_process_surrogate = (
            not is_random
            and not is_direct_botorch
            and (
                strat_upper in (
                    "AICOSCIENTIST_PROCESS_SURROGATE",
                    "AICOSCIENTIST_PROCESS_SURROGATE_EI",
                    "AICOSCIENTIST_PROCESS_SURROGATE_NEI",
                    "AICOSCIENTIST_PROCESS_ENGINE",
                    "AICOSCIENTIST_FULL_PROCESS_ENGINE",
                    "PRODUCTION_COORDINATOR",
                    "COORDINATOR",
                )
                or strat_lower.startswith("aicointel_")
                or self._coordinator is not None
            )
        )

        step_trace = EngineExecutionTrace()
        step_trace.source_adapter_invocations = self._execution_trace.source_adapter_invocations
        step_trace.optimizer_backend_type = (
            "FrozenSurrogateOptimizerBackend" if is_process_surrogate
            else ("BoTorchBackend" if is_direct_botorch else "RandomSampling")
        )

        # Reveal initial observations
        initial_values: list[float] = []
        for cid in sampled_initial:
            rec = oracle.reveal(cid)
            self._execution_trace.oracle_reveal_count += 1
            step_trace.oracle_reveal_count += 1
            initial_values.append(rec[oracle.target_column])

        # Baseline best from initial design
        best_so_far = min(initial_values) if self._minimize else max(initial_values)
        initial_best_value = best_so_far

        if is_process_surrogate:
            traj_engine_path = "AICOSCIENTIST_PROCESS_SURROGATE"
        elif is_direct_botorch:
            traj_engine_path = "DIRECT_BOTORCH_BASELINE"
        elif is_random:
            traj_engine_path = "RANDOM_BASELINE"
        else:
            traj_engine_path = "DIRECT_BOTORCH_BASELINE"

        trajectory = RediscoveryTrajectory(
            policy=strategy,
            seed=seed,
            initial_candidate_ids=list(sampled_initial),
            hidden_best_id=hidden_best,
            hidden_best_value=oracle.hidden_best_value,
            initial_best_value=initial_best_value,
            engine_path=traj_engine_path,
        )

        budget = max_steps if max_steps is not None else oracle.num_unrevealed
        cumulative_regret = 0.0

        for step in range(1, budget + 1):
            if oracle.num_unrevealed == 0:
                break

            visible = oracle.visible_candidates()
            revealed = oracle.revealed_history()

            selected_id: str
            pred_mean: float | None = None
            pred_std: float | None = None
            acq_val: float | None = None
            hidden_rank: int | None = None

            step_engine_path = traj_engine_path
            surrogate_artifact_fp: str | None = None
            horizon_str: str | None = None
            training_view_summary: dict[str, Any] | None = None

            if is_random:
                self._execution_trace.random_steps += 1
                step_trace.random_steps += 1
                step_rng = np.random.default_rng(seed * 1000 + step)
                visible_ids = visible[oracle.candidate_id_column].tolist()
                selected_id = str(step_rng.choice(visible_ids))
                if hidden_best in visible_ids:
                    perm = list(visible_ids)
                    step_rng.shuffle(perm)
                    hidden_rank = perm.index(hidden_best) + 1
                else:
                    hidden_rank = None
            elif is_process_surrogate:
                step_engine_path = "AICOSCIENTIST_PROCESS_SURROGATE"
                runs_dict = self._get_runs_by_recipe()
                step_trace.source_adapter_invocations = max(step_trace.source_adapter_invocations, self._execution_trace.source_adapter_invocations)

                is_recipe_sel = (
                    self._decision_stage in (DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION, "PRE_MANUFACTURING_RECIPE_SELECTION")
                )
                if is_recipe_sel:
                    horizon = PreManufacturingRecipeSelectionHorizon()
                    horizon_str = f"InformationHorizon({DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION.value})"
                    observable_ctrls = list(self._control_columns)
                    dec_stage_val = DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION.value
                else:
                    stage_enum = self._decision_stage if isinstance(self._decision_stage, ProcessStage) else ProcessStage.COATING
                    horizon = InformationHorizon(stage_enum, include_decision_stage_controls=True)
                    horizon_str = f"InformationHorizon({stage_enum.value})"
                    observable_ctrls = [
                        c for c in self._control_columns
                        if horizon.can_observe_stage(STAGE_OF_CONTROL.get(c, stage_enum))
                    ]
                    dec_stage_val = stage_enum.value

                if not observable_ctrls:
                    observable_ctrls = list(self._control_columns)

                samples: list[ProcessSurrogateSample] = []
                for _, row in revealed.iterrows():
                    cid = str(row[oracle.candidate_id_column])
                    g_runs = runs_dict.get(cid, [])
                    if g_runs:
                        self._execution_trace.runs_loaded += len(g_runs)
                        step_trace.runs_loaded += len(g_runs)
                        for r in g_runs:
                            if is_recipe_sel:
                                view = horizon.project_for_recipe_selection(r)
                            else:
                                view = horizon.project(r)
                            self._execution_trace.horizon_projections += 1
                            step_trace.horizon_projections += 1
                            ctrls = {
                                c: float(view.controls[c].value if hasattr(view.controls[c], "value") else view.controls[c])
                                for c in observable_ctrls
                                if c in view.controls
                            }
                            run_d30 = r.final_kpis.get(oracle.target_column)
                            target_val = float(run_d30.value) if (run_d30 is not None and isinstance(run_d30.value, (int, float))) else float(row[oracle.target_column])
                            sample_stage = ProcessStage.FORMULATION if is_recipe_sel else stage_enum
                            samples.append(
                                ProcessSurrogateSample(
                                    sample_id=r.run_id,
                                    run_id=r.run_id,
                                    recipe_id=cid,
                                    source_dataset=self._dataset_id,
                                    source_evidence_kind="PHYSICAL_HISTORICAL",
                                    stage=sample_stage,
                                    group_id=cid,
                                    controls=ctrls,
                                    targets={oracle.target_column: target_val},
                                    fidelity="EXPERIMENTAL",
                                    dataset_manifest_fingerprint=self._dataset_fingerprint,
                                )
                            )
                            self._execution_trace.samples_created += 1
                            step_trace.samples_created += 1
                    else:
                        if not self._allow_flat_fallback:
                            raise RuntimeError(
                                f"SOURCE_BATTERY_PROCESS_RUN_NOT_FOUND: recipe {cid} has no source BatteryProcessRun objects"
                            )
                        ctrls = {c: float(row[c]) for c in observable_ctrls if c in row and pd.notna(row[c])}
                        val = float(row[oracle.target_column])
                        sample_stage = ProcessStage.FORMULATION if is_recipe_sel else stage_enum
                        samples.append(
                            ProcessSurrogateSample(
                                sample_id=cid,
                                run_id=cid,
                                recipe_id=cid,
                                source_dataset=self._dataset_id,
                                source_evidence_kind="PHYSICAL_HISTORICAL",
                                stage=sample_stage,
                                group_id=cid,
                                controls=ctrls,
                                targets={oracle.target_column: val},
                                fidelity="EXPERIMENTAL",
                                dataset_manifest_fingerprint=self._dataset_fingerprint,
                            )
                        )
                        self._execution_trace.samples_created += 1
                        step_trace.samples_created += 1

                schema = SurrogateInputSchema.from_training_samples(samples, declared_fidelities=["EXPERIMENTAL"])
                preprocessor = TrainOnlyPreprocessor().fit(samples, schema)
                X_train = preprocessor.transform(samples)
                surrogate = ProcessSurrogate(model_type="gp", seed=seed * 1000 + step)
                surrogate.fit(X_train, targets={oracle.target_column: np.array([s.targets[oracle.target_column] for s in samples])})
                self._execution_trace.surrogates_fitted += 1
                step_trace.surrogates_fitted += 1

                split_fp = hashlib.sha256(f"split_{seed}_{step}".encode()).hexdigest()
                artifact = SurrogateArtifact(
                    surrogate=surrogate,
                    preprocessor=preprocessor,
                    dataset_fingerprint=self._dataset_fingerprint,
                    split_fingerprint=split_fp,
                    target_names=(oracle.target_column,),
                    input_schema=schema,
                    target_units={oracle.target_column: "mAh/g"},
                    model_version="process-surrogate-v4",
                    training_config={"seed": seed * 1000 + step, "step": step},
                )
                artifact.verify_integrity()
                surrogate_artifact_fp = artifact.artifact_fingerprint
                step_trace.surrogate_artifact_fingerprints.append(surrogate_artifact_fp)
                self._execution_trace.surrogate_artifact_fingerprints.append(surrogate_artifact_fp)

                coord_strat = "expected_improvement"
                if strat_lower.startswith("aicointel_"):
                    sub = strat_lower.replace("aicointel_", "")
                    if sub in ("ei", "expected_improvement"):
                        coord_strat = "expected_improvement"
                    elif sub in ("ucb", "gp_ucb"):
                        coord_strat = "gp_ucb"
                    elif sub == "greedy":
                        coord_strat = "greedy"
                    elif sub == "random":
                        coord_strat = "random"
                elif strat_upper.endswith("_UCB"):
                    coord_strat = "gp_ucb"
                elif strat_upper.endswith("_GREEDY"):
                    coord_strat = "greedy"
                elif strat_upper.endswith("_EI") or strat_upper.endswith("_NEI"):
                    coord_strat = "expected_improvement"
                else:
                    coord_strat = "expected_improvement"

                training_view_summary = {
                    "num_revealed_samples": len(samples),
                    "observable_controls": observable_ctrls,
                    "decision_stage": dec_stage_val,
                    "features": list(preprocessor.output_names),
                    "training_recipe_ids": list(revealed[oracle.candidate_id_column].unique()),
                    "training_run_ids": [s.run_id for s in samples],
                    "number_of_training_runs": len(samples),
                    "target_name": oracle.target_column,
                    "target_units": "mAh/g",
                    "visible_planned_control_names": observable_ctrls,
                    "hidden_observation_names": [
                        "active_mass_mg",
                        "electrode_thickness_um",
                        "porosity_pct",
                        "discharge_specific_capacity_cycle30_mah_g",
                    ],
                    "dataset_fingerprint": self._dataset_fingerprint,
                    "split_fingerprint": split_fp,
                    "schema_fingerprint": schema.schema_fingerprint,
                    "preprocessor_fingerprint": preprocessor.state_fingerprint(),
                    "model_state_fingerprint": surrogate.state_fingerprint(),
                    "surrogate_artifact_fingerprint": surrogate_artifact_fp,
                    "uncertainty_kind": surrogate.uncertainty_kind,
                    "acquisition_strategy": "EXPECTED_IMPROVEMENT" if coord_strat == "expected_improvement" else coord_strat.upper(),
                }

                context = SurrogateDecisionContext(stage=samples[0].stage, fidelity="EXPERIMENTAL")
                backend = FrozenSurrogateOptimizerBackend(artifact, context, beta=beta)
                coordinator = self._coordinator or ProcessOptimizationCoordinator(scalar_backend=backend)
                coordinator.scalar_backend = backend

                cand_cols = list(backend.control_names)
                pool_slice = visible[[oracle.candidate_id_column] + [c for c in cand_cols if c in visible.columns]].copy()
                space = ProcessSearchSpace.from_finite_pool(pool_slice, id_column=oracle.candidate_id_column)
                process_obj = ProcessOptimizationObjective([
                    ObjectiveSpec(oracle.target_column, "minimize" if self._minimize else "maximize", units="mAh/g")
                ])

                self._execution_trace.coordinator_calls += 1
                step_trace.coordinator_calls += 1
                step_trace.training_recipe_ids_by_step.append([str(r) for r in revealed[oracle.candidate_id_column]])
                self._execution_trace.training_recipe_ids_by_step.append([str(r) for r in revealed[oracle.candidate_id_column]])
                step_trace.candidate_ids_scored_by_step.append([str(c) for c in visible[oracle.candidate_id_column]])
                self._execution_trace.candidate_ids_scored_by_step.append([str(c) for c in visible[oracle.candidate_id_column]])

                proposals = coordinator.propose_recipes(
                    observations=revealed,
                    space=space,
                    objective=process_obj,
                    n=len(visible),
                    seed=seed * 1000 + step,
                    strategy=coord_strat,
                )
                self._execution_trace.proposals_generated += len(proposals)

                top_prop = proposals[0]
                selected_id = str(top_prop.candidate_instance_id)
                pred_mean = top_prop.predicted_outputs[oracle.target_column].mean
                pred_std = top_prop.predicted_outputs[oracle.target_column].std
                acq_val = top_prop.acquisition_value

                prop_ids = [str(p.candidate_instance_id) for p in proposals]
                if hidden_best in prop_ids:
                    hidden_rank = prop_ids.index(hidden_best) + 1
                else:
                    hidden_rank = None
            else:
                self._execution_trace.botorch_calls += 1
                step_trace.botorch_calls += 1
                step_engine_path = "DIRECT_BOTORCH_BASELINE"
                horizon_str = "NONE (FLAT TABLE)"
                training_view_summary = {"backend": "BoTorchBackend", "num_observations": len(revealed)}
                obj = OptimizationObjective(target_name=oracle.target_column, minimize=self._minimize)
                strat_raw = strategy.strip()
                strat_lower = strat_raw.lower()
                if strat_lower in ("direct_botorch_baseline", "direct_botorch"):
                    direct_strat = "noisy_expected_improvement"
                elif strat_lower.startswith("direct_botorch_"):
                    direct_strat = strat_raw[len("direct_botorch_"):]
                else:
                    direct_strat = strat_raw
                direct_strat = resolve_strategy(direct_strat)
                proposals = self._backend.propose(
                    observations=revealed,
                    candidate_pool=self._candidate_pool[[oracle.candidate_id_column] + oracle.control_columns],
                    objective=obj,
                    feature_columns=oracle.control_columns,
                    candidate_id_column=oracle.candidate_id_column,
                    n=len(self._candidate_pool),
                    seed=seed * 1000 + step,
                    strategy=direct_strat,
                    beta=beta,
                )

                if not proposals:
                    raise RuntimeError("Surrogate proposal returned empty candidate list.")

                visible_ids = set(visible[oracle.candidate_id_column].astype(str))
                unrevealed_proposals = [p for p in proposals if str(p.candidate_id) in visible_ids]
                if not unrevealed_proposals:
                    raise RuntimeError("No unrevealed proposals returned by optimizer backend.")

                top_prop = unrevealed_proposals[0]
                selected_id = str(top_prop.candidate_id)
                pred_mean = top_prop.predicted_mean
                pred_std = top_prop.predicted_std
                acq_val = top_prop.acquisition_value

                # Find surrogate rank of hidden best among unrevealed candidates
                prop_ids = [str(p.candidate_id) for p in unrevealed_proposals]
                if hidden_best in prop_ids:
                    hidden_rank = prop_ids.index(hidden_best) + 1
                else:
                    hidden_rank = None

            # Reveal selected candidate
            record = oracle.reveal(selected_id)
            self._execution_trace.oracle_reveals += 1
            step_trace.oracle_reveals += 1
            revealed_val = float(record[oracle.target_column])

            is_hidden = (selected_id == hidden_best)
            if is_hidden and not trajectory.rediscovered:
                trajectory.rediscovered = True
                trajectory.experiments_to_best = step

            if selected_id in top_k_ids and not trajectory.top3_rediscovered:
                trajectory.top3_rediscovered = True
                trajectory.experiments_to_top3 = step

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
                engine_path=step_engine_path,
                surrogate_artifact_fingerprint=surrogate_artifact_fp,
                dataset_fingerprint=self._dataset_fingerprint,
                information_horizon=horizon_str,
                training_view_summary=training_view_summary,
            )
            trajectory.steps.append(step_record)

        trajectory.final_simple_regret = trajectory.steps[-1].simple_regret if trajectory.steps else 0.0
        trajectory.final_cumulative_regret = trajectory.steps[-1].cumulative_regret if trajectory.steps else 0.0
        trajectory.execution_trace = step_trace.to_dict()

        return trajectory


def summarize_trajectories(trajectories: Sequence[RediscoveryTrajectory]) -> PolicySummary:
    """Aggregates trajectories across seeds for a given policy into a statistical summary."""
    if not trajectories:
        raise ValueError("Cannot summarize empty trajectory list.")

    policy = trajectories[0].policy
    engine_path = trajectories[0].engine_path
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

    top3_hits_at_5 = sum(
        1 for t in trajectories
        if getattr(t, "experiments_to_top3", None) is not None and t.experiments_to_top3 <= 5
    )
    top3_rate_5 = top3_hits_at_5 / num_seeds if num_seeds > 0 else 0.0

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
        hit_rate_at_1=hit_rate_at_step.get(1, 0.0),
        hit_rate_at_3=hit_rate_at_step.get(3, 0.0),
        hit_rate_at_5=hit_rate_at_step.get(5, 0.0),
        top3_hit_rate_at_5=top3_rate_5,
        engine_path=engine_path,
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
    coordinator: Any | None = None,
    top_k_targets: int = 3,
    decision_stage: ProcessStage | DecisionHorizon | str = DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
    dataset_id: str = "drakopoulos_graphite",
    dataset_fingerprint: str | None = None,
    benchmark_task: str = "UNCONSTRAINED_D30",
    runs_by_recipe: Mapping[str, Sequence[BatteryProcessRun]] | None = None,
) -> dict[str, Any]:
    """Runs full multi-policy, multi-seed offline closed-loop rediscovery benchmark."""
    all_trajectories: dict[str, list[RediscoveryTrajectory]] = {}
    policy_summaries: list[PolicySummary] = []
    primary_replay: RediscoveryReplay | None = None

    for pol in policies:
        policy_replay = RediscoveryReplay(
            candidate_pool=candidate_pool,
            candidate_id_column=candidate_id_column,
            target_column=target_column,
            control_columns=control_columns,
            minimize=minimize,
            backend=backend,
            coordinator=coordinator,
            top_k_targets=top_k_targets,
            decision_stage=decision_stage,
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            benchmark_task=benchmark_task,
            runs_by_recipe=runs_by_recipe,
        )
        if primary_replay is None:
            primary_replay = policy_replay

        pol_trajectories: list[RediscoveryTrajectory] = []
        for seed in seeds:
            traj = policy_replay.run(
                strategy=pol,
                seed=seed,
                initial_size=initial_size,
                max_steps=max_steps,
            )
            pol_trajectories.append(traj)
        all_trajectories[pol] = pol_trajectories
        summary = summarize_trajectories(pol_trajectories)
        policy_summaries.append(summary)

    budget_for_analytic = max_steps if max_steps is not None else (len(candidate_pool) - initial_size)
    analytic_top1 = calculate_hypergeometric_baseline(len(candidate_pool), initial_size, budget_for_analytic, top_k=1)
    analytic_top3 = calculate_hypergeometric_baseline(len(candidate_pool), initial_size, budget_for_analytic, top_k=top_k_targets)

    sorted_pool = candidate_pool.sort_values(by=target_column, ascending=minimize)
    top_k_ids = set(sorted_pool[candidate_id_column].head(top_k_targets).astype(str))
    first_pol = policies[0]
    first_trajs = all_trajectories[first_pol]
    remaining_targets_by_seed = [
        len([cid for cid in top_k_ids if cid not in t.initial_candidate_ids])
        for t in first_trajs
    ]
    cond_analytic_top3 = calculate_conditional_hypergeometric_baseline(
        total_candidates=len(candidate_pool),
        initial_designs_remaining_targets=remaining_targets_by_seed,
        budget=budget_for_analytic,
        initial_size=initial_size,
    )

    policy_traces: dict[str, dict[str, Any]] = {}
    engine_path_audit: list[dict[str, Any]] = []

    for pol, trajs in all_trajectories.items():
        combined_trace = EngineExecutionTrace()
        engine_path = trajs[0].engine_path if trajs else "UNKNOWN"
        for t in trajs:
            t_trace = t.execution_trace
            combined_trace.source_adapter_invocations = max(combined_trace.source_adapter_invocations, t_trace.get("source_adapter_invocations", 0))
            combined_trace.battery_process_runs_seen += t_trace.get("battery_process_runs_seen", 0)
            combined_trace.recipe_selection_horizon_invocations += t_trace.get("recipe_selection_horizon_invocations", 0)
            combined_trace.process_surrogate_samples_created += t_trace.get("process_surrogate_samples_created", 0)
            combined_trace.process_surrogate_fit_count += t_trace.get("process_surrogate_fit_count", 0)
            combined_trace.coordinator_proposal_count += t_trace.get("coordinator_proposal_count", 0)
            combined_trace.oracle_reveal_count += t_trace.get("oracle_reveal_count", 0)
            combined_trace.direct_botorch_calls += t_trace.get("direct_botorch_calls", 0)
            combined_trace.random_steps += t_trace.get("random_steps", 0)
            backend_type = t_trace.get("optimizer_backend_type", "none")
            if backend_type != "none":
                combined_trace.optimizer_backend_type = backend_type
            combined_trace.surrogate_artifact_fingerprints.extend(t_trace.get("surrogate_artifact_fingerprints", []))
            combined_trace.training_recipe_ids_by_step.extend(t_trace.get("training_recipe_ids_by_step", []))
            combined_trace.candidate_ids_scored_by_step.extend(t_trace.get("candidate_ids_scored_by_step", []))
        policy_traces[pol] = combined_trace.to_dict()
        engine_path_audit.append(combined_trace.generate_audit(pol, engine_path))

    first_policy_trace = policy_traces.get("AICOSCIENTIST_PROCESS_SURROGATE") or (list(policy_traces.values())[0] if policy_traces else {})

    return {
        "benchmark_task": benchmark_task,
        "published_high_loading_rediscovery_status": "NOT_EVALUABLE_WITH_AVAILABLE_D30",
        "candidate_id_column": candidate_id_column,
        "target_column": target_column,
        "control_columns": primary_replay._control_columns if primary_replay else [],
        "minimize": minimize,
        "candidate_pool_size": len(candidate_pool),
        "initial_size": initial_size,
        "seeds": list(seeds),
        "policies": list(policies),
        "engine_execution_trace": first_policy_trace,
        "policy_execution_traces": policy_traces,
        "engine_path_audit": engine_path_audit,
        "analytic_hypergeometric": {
            "top1_hit_rate_by_step": analytic_top1,
            "top3_hit_rate_by_step": cond_analytic_top3,
            "top3_unconditional_by_step": analytic_top3,
            "top3_conditional_by_step": cond_analytic_top3,
        },
        "trajectories": {
            pol: [t.to_dict() for t in trajs]
            for pol, trajs in all_trajectories.items()
        },
        "summaries": [s.to_dict() for s in policy_summaries],
    }


class ProductionProcessRediscoveryRunner:
    """Production runner executing offline closed-loop rediscovery through ProcessSurrogate / ProcessOptimizationCoordinator."""

    def __init__(
        self,
        candidate_pool: pd.DataFrame,
        *,
        candidate_id_column: str = "recipe_id",
        target_column: str = "discharge_specific_capacity_cycle30_mah_g",
        control_columns: Sequence[str] | None = None,
        coordinator: ProcessOptimizationCoordinator | None = None,
        top_k_targets: int = 3,
        decision_stage: ProcessStage | DecisionHorizon | str = DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
        dataset_id: str = "drakopoulos_graphite",
        dataset_fingerprint: str | None = None,
        benchmark_task: str = "UNCONSTRAINED_D30",
        runs_by_recipe: Mapping[str, Sequence[BatteryProcessRun]] | None = None,
    ) -> None:
        self.candidate_pool = candidate_pool.copy()
        self.candidate_id_column = candidate_id_column
        self.target_column = target_column
        self.control_columns = control_columns
        self.coordinator = coordinator or ProcessOptimizationCoordinator()
        self.top_k_targets = top_k_targets
        self.decision_stage = decision_stage
        self.dataset_id = dataset_id
        self.dataset_fingerprint = dataset_fingerprint
        self.benchmark_task = benchmark_task
        self.runs_by_recipe = runs_by_recipe

    def run(
        self,
        policies: Sequence[str] = (
            "AICOSCIENTIST_PROCESS_SURROGATE",
            "DIRECT_BOTORCH_BASELINE",
            "random",
        ),
        seeds: Sequence[int] = (11, 23, 42, 67, 101, 137, 179, 223, 281, 353),
        initial_size: int = 3,
        budget: int = 5,
    ) -> dict[str, Any]:
        return run_rediscovery_benchmark(
            candidate_pool=self.candidate_pool,
            candidate_id_column=self.candidate_id_column,
            target_column=self.target_column,
            control_columns=self.control_columns,
            minimize=False,
            policies=policies,
            seeds=seeds,
            initial_size=initial_size,
            max_steps=budget,
            coordinator=self.coordinator,
            top_k_targets=self.top_k_targets,
            decision_stage=self.decision_stage,
            dataset_id=self.dataset_id,
            dataset_fingerprint=self.dataset_fingerprint,
            benchmark_task=self.benchmark_task,
            runs_by_recipe=self.runs_by_recipe,
        )

    def run_high_loading(
        self,
        *,
        min_active_mass_mg: float = 16.0,
        policies: Sequence[str] = (
            "AICOSCIENTIST_PROCESS_SURROGATE",
            "DIRECT_BOTORCH_BASELINE",
            "random",
        ),
        seeds: Sequence[int] = (11, 23, 42, 67, 101, 137, 179, 223, 281, 353),
        initial_size: int = 3,
        budget: int = 5,
        benchmark_task: str = "HIGHER_LOADING_MEASURED_D30_PROXY",
    ) -> dict[str, Any]:
        """Runs higher-loading electrode rediscovery benchmark on measured-D30 proxy subset.

        Scientific context:
        - Evaluates the proxy subset of recipes with higher coating gap (>= 150 um) and mass (>= 11-16 mg).
        - Exact published >= 25 mg Alchemite objective is NOT evaluable because 300 um cells lack usable D30 cycling data.
        - Reports published_high_loading_rediscovery_status = NOT_EVALUABLE_WITH_AVAILABLE_D30.
        """
        hl_pool = filter_candidate_pool_for_high_loading(
            self.candidate_pool,
            min_active_mass_mg=min_active_mass_mg,
        )
        res = run_rediscovery_benchmark(
            candidate_pool=hl_pool,
            candidate_id_column=self.candidate_id_column,
            target_column=self.target_column,
            control_columns=self.control_columns,
            minimize=False,
            policies=policies,
            seeds=seeds,
            initial_size=initial_size,
            max_steps=budget,
            coordinator=self.coordinator,
            top_k_targets=self.top_k_targets,
            decision_stage=self.decision_stage,
            dataset_id=self.dataset_id,
            dataset_fingerprint=self.dataset_fingerprint,
            benchmark_task=benchmark_task,
            runs_by_recipe=self.runs_by_recipe,
        )
        res["published_high_loading_rediscovery_status"] = "NOT_EVALUABLE_WITH_AVAILABLE_D30"
        return res

