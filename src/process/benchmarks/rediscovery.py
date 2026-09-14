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
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.information_horizon import InformationHorizon
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage
from src.process.surrogates.core import (
    ProcessSurrogate,
    ProcessSurrogateSample,
    SurrogateArtifact,
    SurrogateInputSchema,
    TrainOnlyPreprocessor,
)

logger = logging.getLogger(__name__)

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

    Pool initially has total_candidates.
    initial_size candidates are drawn excluding the top_k targets.
    Remaining pool has N_rem = total_candidates - initial_size candidates, containing top_k targets.
    At step s in [1, budget], s candidates have been drawn sequentially without replacement.

    P(at least 1 of top_k in s draws) = 1 - [comb(N_rem - top_k, s) / comb(N_rem, s)]
    """
    n_rem = total_candidates - initial_size
    if n_rem <= 0 or top_k <= 0:
        return {s: 0.0 for s in range(1, budget + 1)}

    curve: dict[int, float] = {}
    for s in range(1, budget + 1):
        if s > n_rem or (n_rem - top_k) < s:
            curve[s] = 1.0
            continue
        ways_to_miss = math.comb(n_rem - top_k, s)
        total_ways = math.comb(n_rem, s)
        prob = 1.0 - (ways_to_miss / total_ways)
        curve[s] = float(prob)
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
        decision_stage: ProcessStage = ProcessStage.COATING,
        dataset_id: str = "drakopoulos_graphite",
        dataset_fingerprint: str | None = None,
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
        if dataset_fingerprint is None:
            raw_bytes = pd.util.hash_pandas_object(self._candidate_pool, index=True).values.tobytes()
            self._dataset_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
        else:
            self._dataset_fingerprint = dataset_fingerprint

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

        # Reveal initial observations
        initial_values: list[float] = []
        for cid in sampled_initial:
            rec = oracle.reveal(cid)
            initial_values.append(rec[oracle.target_column])

        # Baseline best from initial design
        best_so_far = min(initial_values) if self._minimize else max(initial_values)
        initial_best_value = best_so_far

        canonical_strat = resolve_strategy(strategy) if strategy.lower() in ("random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement") else strategy

        is_process_surrogate = (
            strategy.upper() in (
                "AICOSCIENTIST_PROCESS_SURROGATE",
                "AICOSCIENTIST_PROCESS_ENGINE",
                "PRODUCTION_COORDINATOR",
                "COORDINATOR",
            )
            or strategy.lower().startswith("aicointel_")
            or self._coordinator is not None
        )
        is_direct_botorch = (
            strategy.upper().startswith("DIRECT_BOTORCH")
            or strategy.lower() in ("direct_botorch_baseline", "direct_botorch")
        )
        is_random = (canonical_strat == "random" and not is_process_surrogate)

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
                horizon = InformationHorizon(self._decision_stage, include_decision_stage_controls=True)
                horizon_str = f"InformationHorizon({self._decision_stage.value})"
                observable_ctrls = [
                    c for c in self._control_columns
                    if horizon.can_observe_stage(STAGE_OF_CONTROL.get(c, self._decision_stage))
                ]
                if not observable_ctrls:
                    observable_ctrls = list(self._control_columns)

                samples: list[ProcessSurrogateSample] = []
                for _, row in revealed.iterrows():
                    cid = str(row[oracle.candidate_id_column])
                    ctrls = {c: float(row[c]) for c in observable_ctrls if c in row and pd.notna(row[c])}
                    val = float(row[oracle.target_column])
                    samples.append(
                        ProcessSurrogateSample(
                            sample_id=cid,
                            run_id=cid,
                            recipe_id=cid,
                            source_dataset=self._dataset_id,
                            source_evidence_kind="PHYSICAL_HISTORICAL",
                            stage=self._decision_stage,
                            group_id=cid,
                            controls=ctrls,
                            targets={oracle.target_column: val},
                            fidelity="EXPERIMENTAL",
                            dataset_manifest_fingerprint=self._dataset_fingerprint,
                        )
                    )

                schema = SurrogateInputSchema.from_training_samples(samples, declared_fidelities=["EXPERIMENTAL"])
                preprocessor = TrainOnlyPreprocessor().fit(samples, schema)
                X_train = preprocessor.transform(samples)
                surrogate = ProcessSurrogate(model_type="gp", seed=seed * 1000 + step)
                surrogate.fit(X_train, targets={oracle.target_column: np.array([s.targets[oracle.target_column] for s in samples])})
                split_fp = hashlib.sha256(f"split_{seed}_{step}".encode()).hexdigest()
                artifact = SurrogateArtifact(
                    surrogate=surrogate,
                    preprocessor=preprocessor,
                    dataset_fingerprint=self._dataset_fingerprint,
                    split_fingerprint=split_fp,
                    target_names=(oracle.target_column,),
                    input_schema=schema,
                    target_units={oracle.target_column: "mAh/g"},
                    model_version="process-surrogate-v3",
                    training_config={"seed": seed * 1000 + step, "step": step},
                )
                artifact.verify_integrity()
                surrogate_artifact_fp = artifact.artifact_fingerprint
                training_view_summary = {
                    "num_revealed_samples": len(samples),
                    "observable_controls": observable_ctrls,
                    "decision_stage": self._decision_stage.value,
                    "features": list(preprocessor.output_names),
                }

                # Predict on unrevealed visible candidates
                visible_ids = visible[oracle.candidate_id_column].astype(str).tolist()
                cand_rows = []
                for _, row in visible.iterrows():
                    v_dict = {"fidelity::EXPERIMENTAL": 1.0}
                    for c in observable_ctrls:
                        if c in row and pd.notna(row[c]):
                            v_dict[f"control::{c}"] = float(row[c])
                    cand_rows.append(v_dict)

                X_cands = preprocessor.transform_values(cand_rows)
                dist = surrogate.predict_distribution(X_cands)
                means = dist[oracle.target_column][0]
                stds = dist[oracle.target_column][1]

                coord_strat = "noisy_expected_improvement"
                if strategy.startswith("aicointel_"):
                    coord_strat = strategy.replace("aicointel_", "")
                    if coord_strat == "nei":
                        coord_strat = "noisy_expected_improvement"
                    elif coord_strat == "ei":
                        coord_strat = "expected_improvement"
                    elif coord_strat == "ucb":
                        coord_strat = "gp_ucb"

                scored_candidates = []
                for i, cid in enumerate(visible_ids):
                    m = float(means[i])
                    s = max(float(stds[i]), 1e-8)
                    if coord_strat in ("ucb", "gp_ucb"):
                        acq = m - beta * s if self._minimize else m + beta * s
                    elif coord_strat == "greedy":
                        acq = -m if self._minimize else m
                    else:  # expected improvement
                        diff = (best_so_far - m) if self._minimize else (m - best_so_far)
                        z = diff / s
                        phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
                        acq = diff * phi + s * pdf
                    scored_candidates.append({
                        "id": cid,
                        "mean": m,
                        "std": s,
                        "acq": acq,
                    })

                scored_candidates.sort(key=lambda x: x["acq"], reverse=True)
                top_cand = scored_candidates[0]
                selected_id = top_cand["id"]
                pred_mean = top_cand["mean"]
                pred_std = top_cand["std"]
                acq_val = top_cand["acq"]

                prop_ids = [c["id"] for c in scored_candidates]
                if hidden_best in prop_ids:
                    hidden_rank = prop_ids.index(hidden_best) + 1
                else:
                    hidden_rank = None
            else:
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
    decision_stage: ProcessStage = ProcessStage.COATING,
    dataset_id: str = "drakopoulos_graphite",
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Runs full multi-policy, multi-seed offline closed-loop rediscovery benchmark."""
    replay = RediscoveryReplay(
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

    budget_for_analytic = max_steps if max_steps is not None else (len(candidate_pool) - initial_size)
    analytic_top1 = calculate_hypergeometric_baseline(len(candidate_pool), initial_size, budget_for_analytic, top_k=1)
    analytic_top3 = calculate_hypergeometric_baseline(len(candidate_pool), initial_size, budget_for_analytic, top_k=top_k_targets)

    return {
        "candidate_id_column": candidate_id_column,
        "target_column": target_column,
        "control_columns": replay._control_columns,
        "minimize": minimize,
        "candidate_pool_size": len(candidate_pool),
        "initial_size": initial_size,
        "seeds": list(seeds),
        "policies": list(policies),
        "analytic_hypergeometric": {
            "top1_hit_rate_by_step": analytic_top1,
            "top3_hit_rate_by_step": analytic_top3,
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
        decision_stage: ProcessStage = ProcessStage.COATING,
        dataset_id: str = "drakopoulos_graphite",
        dataset_fingerprint: str | None = None,
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
        )

