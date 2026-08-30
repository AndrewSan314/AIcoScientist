from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.optimization.closed_loop import ClosedLoopOptimizer, ExperimentProposal, ExperimentResult, OptimizerState
from src.optimization.search_space import ContinuousVariable, SearchSpace
from src.science.direct_baseline import DirectPerformanceModel
from src.science.ledger import ExperimentLedger
from src.science.model_bundle import ScientificModelBundle
from src.science.provenance import ScientificModelProvenance
from src.science.rationale import ScientificRationale, generate_scientific_rationale
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.two_stage import TwoStageScientificModel
from src.science.validation import validate_record_against_spec


class PendingExperimentError(RuntimeError):
    """Raised when attempting to propose a new experiment while an unresolved experiment is already pending."""


def _build_search_space(spec: DatasetSpec, candidate_pool: pd.DataFrame) -> SearchSpace:
    """Infers bounded search space from candidate pool or pre-experiment features."""
    cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features
    variables = []
    for col in cols:
        if col in candidate_pool.columns:
            c_min = float(candidate_pool[col].min())
            c_max = float(candidate_pool[col].max())
        else:
            c_min, c_max = 0.0, 1.0

        if c_min >= c_max:
            c_min = c_min - 0.5
            c_max = c_max + 0.5
        variables.append(ContinuousVariable(name=col, lower=c_min, upper=c_max))

    return SearchSpace(name=f"{spec.name}_space", variables=variables)


class ScientificClosedLoopCoordinator:
    """Orchestrates scientific closed-loop experimentation with validated optimizer, two-stage modeling, and tamper-evident ledger."""

    def __init__(
        self,
        spec: DatasetSpec,
        two_stage_spec: TwoStageModelSpec,
        optimizer: ClosedLoopOptimizer,
        optimizer_state: OptimizerState,
        model_bundle: ScientificModelBundle,
        ledger: ExperimentLedger,
        candidate_pool: pd.DataFrame,
        allow_parallel_experiments: bool = False,
        random_state: int = 42,
    ) -> None:
        self.spec = spec
        self.two_stage_spec = two_stage_spec
        self.optimizer = optimizer
        self.optimizer_state = optimizer_state
        self.model_bundle = model_bundle
        self.ledger = ledger
        self.candidate_pool = candidate_pool
        self.allow_parallel_experiments = allow_parallel_experiments
        self.random_state = random_state
        self.step_counter = len(self.ledger.list_completed_records())

    def propose_next(
        self,
        n_mc_samples: int = 64,
    ) -> tuple[ScientificExperimentRecord, ScientificRationale]:
        """Proposes the next experiment using validated optimizer, analyzes it with the two-stage model, and persists to ledger."""
        # 1. Pending experiment protection
        pending_records = self.ledger.list_pending_records()
        if pending_records and not self.allow_parallel_experiments:
            raise PendingExperimentError(
                f"Cannot propose a new experiment: experiment {pending_records[0].experiment_id!r} "
                f"is currently in stage {pending_records[0].stage.value}. "
                f"Record physical characterization/performance or mark failed before requesting next proposal."
            )

        # 2. Select next candidate using PRE-EXPERIMENT controllable variables only
        prop = self.optimizer.propose(self.optimizer_state)
        self._last_proposal = prop
        cand_dict = dict(prop.design_variables)
        cand_id = prop.candidate_id
        exp_id = f"EXP_{self.spec.name[:6].upper()}_{self.step_counter + 1:03d}"

        # Clean candidate process dictionary (only controllable pre-experiment variables)
        process_features = self.two_stage_spec.process_features
        cand_process = {k: float(cand_dict[k]) for k in process_features if k in cand_dict}

        # 3. Direct model prediction
        cand_df = pd.DataFrame([cand_process])
        dir_mean, dir_std = self.model_bundle.direct_model.predict(cand_df)
        dir_pred = (float(dir_mean[0]), float(dir_std[0]))

        # 4. Two-Stage model analysis (Monte Carlo uncertainty propagation)
        e2e_pred = self.model_bundle.two_stage_model.predict_end_to_end(
            cand_df,
            target_name=self.spec.target_column,
            n_mc_samples=n_mc_samples,
            seed=self.random_state + self.step_counter,
        )

        # 5. Generate structured, deterministic ScientificRationale
        history_df = self.ledger.to_dataframe()
        incumbent = self.optimizer_state.current_best

        rationale = generate_scientific_rationale(
            experiment_id=exp_id,
            candidate_id=cand_id,
            candidate_process=cand_process,
            direct_prediction=dir_pred,
            two_stage_prediction=e2e_pred,
            acquisition_method=prop.acquisition_method,
            acquisition_score=prop.acquisition_score,
            observed_history=history_df,
            process_features=process_features,
            incumbent_target=incumbent,
        )

        # 6. Create ScientificExperimentRecord & validate against DatasetSpec
        record = ScientificExperimentRecord(
            experiment_id=exp_id,
            candidate_id=cand_id,
            dataset_name=self.spec.name,
            stage=ExperimentStage.PROPOSED,
            pre_experiment_features=cand_process,
            candidate_variables=cand_process,
            proposal_metadata={
                "step": self.step_counter + 1,
                "strategy": prop.acquisition_method,
                "acquisition_score": prop.acquisition_score,
                "model_run_id": self.model_bundle.provenance.model_run_id,
                "learning_value_score": rationale.expected_learning_value,
                "model_disagreement_flag": rationale.model_disagreement_flag,
            },
        )
        validate_record_against_spec(record, self.spec)

        # 7. Persist proposal to append-only ledger
        self.ledger.record_proposal(record)
        return record, rationale

    def record_executed(
        self,
        experiment_id: str,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> ScientificExperimentRecord:
        """Records physical/simulated execution start."""
        return self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.EXECUTED,
            event_type="EXPERIMENT_EXECUTED",
            delta_payload={"execution_metadata": dict(execution_metadata or {})},
        )

    def record_characterization(
        self,
        experiment_id: str,
        characterization: Mapping[str, Any],
        measurement_uncertainty: Mapping[str, float] | None = None,
        quality_flags: list[str] | None = None,
    ) -> ScientificExperimentRecord:
        """Records post-experiment physical characterization (Structure/Morphology) and updates Stage A."""
        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.CHARACTERIZED,
            event_type="CHARACTERIZATION_RECORDED",
            delta_payload={
                "characterization": dict(characterization),
                "measurement_uncertainty": dict(measurement_uncertainty or {}),
                "quality_flags": list(quality_flags or []),
            },
        )
        validate_record_against_spec(rec, self.spec)

        # Online update of Stage A if enough data exists
        completed_df = self.ledger.to_dataframe()
        valid_char_df = completed_df[self.two_stage_spec.process_features + list(characterization.keys())].dropna()
        if len(valid_char_df) >= 2:
            self.model_bundle.two_stage_model.stage_a.fit(completed_df)

        return rec

    def record_performance(
        self,
        experiment_id: str,
        performance: Mapping[str, Any],
        measurement_uncertainty: Mapping[str, float] | None = None,
        quality_flags: list[str] | None = None,
    ) -> ScientificExperimentRecord:
        """Records downstream performance measurement, feeds optimizer, and updates all models."""
        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.COMPLETED,
            event_type="PERFORMANCE_RECORDED",
            delta_payload={
                "performance": dict(performance),
                "measurement_uncertainty": dict(measurement_uncertainty or {}),
                "quality_flags": list(quality_flags or []),
            },
        )
        validate_record_against_spec(rec, self.spec)
        self.step_counter += 1

        # Ingest into optimizer state
        target_val = float(performance[self.spec.target_column])
        exp_res = ExperimentResult(
            candidate_id=rec.candidate_id,
            design_variables=rec.pre_experiment_features,
            target_value=target_val,
        )
        prop = getattr(self, "_last_proposal", None)
        if prop is None or prop.candidate_id != rec.candidate_id:
            prop = ExperimentProposal(
                candidate_id=rec.candidate_id,
                design_variables=rec.pre_experiment_features,
                predicted_performance=0.0,
                prediction_uncertainty=1.0,
                acquisition_score=0.0,
                acquisition_method=self.optimizer.strategy,
                trust_region_center=None,
                trust_region_radius=None,
                recommendation_reason="",
                reason_code="OBSERVATION_RECORDED",
                distance_to_nearest_observed=0.0,
                step=self.step_counter,
            )
        self.optimizer.observe(self.optimizer_state, prop, exp_res)

        # Refit models with complete history
        completed_df = self.ledger.to_dataframe()
        valid_completed = completed_df[completed_df["stage"] == ExperimentStage.COMPLETED.value]
        if len(valid_completed) >= 2:
            self.model_bundle.direct_model.fit(valid_completed)
            self.model_bundle.two_stage_model.fit(valid_completed)

        return rec

    def record_failed(
        self,
        experiment_id: str,
        failure_reason: str,
        quality_flags: list[str] | None = None,
    ) -> ScientificExperimentRecord:
        """Records experimental failure in audit trail without fabricating fake low target numbers."""
        return self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.FAILED,
            event_type="EXPERIMENT_FAILED",
            delta_payload={
                "failure_reason": failure_reason,
                "quality_flags": list(quality_flags or ["FABRICATION_FAILURE"]),
            },
        )

    @classmethod
    def initialize_new(
        cls,
        spec: DatasetSpec,
        two_stage_spec: TwoStageModelSpec,
        initial_data: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        db_path: Path | str = ":memory:",
        strategy: str = "expected_improvement",
        allow_parallel_experiments: bool = False,
        random_state: int = 42,
    ) -> ScientificClosedLoopCoordinator:
        """Initializes a new coordinator from historical seed dataset and starts a fresh ledger."""
        ledger = ExperimentLedger(db_path)

        # Record initial historical experiments in ledger
        for i, row in initial_data.iterrows():
            eid = str(row.get("experiment_id", f"EXP_INIT_{i:03d}"))
            cid = str(row.get("candidate_id", f"CAND_INIT_{i:03d}"))
            pre_feats = {k: float(row[k]) for k in spec.pre_experiment_features if k in row and pd.notna(row[k])}
            char_feats = {k: float(row[k]) for k in spec.post_experiment_characterization if k in row and pd.notna(row[k])}
            perf_feats = {k: float(row[k]) for k in spec.targets if k in row and pd.notna(row[k])}

            rec = ScientificExperimentRecord(
                experiment_id=eid,
                candidate_id=cid,
                dataset_name=spec.name,
                stage=ExperimentStage.PROPOSED,
                pre_experiment_features=pre_feats,
                candidate_variables=pre_feats,
            )
            ledger.record_proposal(rec)
            ledger.append_transition(eid, ExperimentStage.EXECUTED, "INIT_EXECUTED", {})
            if char_feats:
                ledger.append_transition(eid, ExperimentStage.CHARACTERIZED, "INIT_CHAR", {"characterization": char_feats})
            if perf_feats:
                ledger.append_transition(eid, ExperimentStage.COMPLETED, "INIT_PERF", {"performance": perf_feats})

        # Fit initial models
        direct_model = DirectPerformanceModel(two_stage_spec.process_features, spec.target_column, random_state=random_state)
        direct_model.fit(initial_data)

        two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=random_state)
        two_stage_model.fit(initial_data)

        prov = ScientificModelProvenance.create(
            dataset_name=spec.name,
            dataset_fingerprint=hashlib.sha256(str(len(initial_data)).encode("utf-8")).hexdigest()[:12],
            spec_fingerprint=hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:12],
            training_experiment_ids=list(initial_data["experiment_id"]) if "experiment_id" in initial_data.columns else [f"EXP_INIT_{i:03d}" for i in range(len(initial_data))],
            feature_columns=spec.feature_columns,
            target_columns=spec.targets,
            random_seed=random_state,
            model_types={"direct": "GPR", "stage_a": "GPR", "stage_b": "GPR"},
        )

        bundle = ScientificModelBundle(
            direct_model=direct_model,
            two_stage_model=two_stage_model,
            spec=spec,
            two_stage_spec=two_stage_spec,
            provenance=prov,
        )

        search_space = _build_search_space(spec, candidate_pool)
        candidate_cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features
        optimizer = ClosedLoopOptimizer(
            search_space=search_space,
            feature_cols=candidate_cols,
            target_col=spec.target_column,
            strategy=strategy,
            objective=spec.objective,
            random_state=random_state,
        )
        opt_state = optimizer.initialize(initial_data)

        return cls(
            spec=spec,
            two_stage_spec=two_stage_spec,
            optimizer=optimizer,
            optimizer_state=opt_state,
            model_bundle=bundle,
            ledger=ledger,
            candidate_pool=candidate_pool,
            allow_parallel_experiments=allow_parallel_experiments,
            random_state=random_state,
        )

    @classmethod
    def resume_from_ledger(
        cls,
        db_path: Path | str,
        spec: DatasetSpec,
        two_stage_spec: TwoStageModelSpec,
        candidate_pool: pd.DataFrame,
        strategy: str = "expected_improvement",
        allow_parallel_experiments: bool = False,
        random_state: int = 42,
    ) -> ScientificClosedLoopCoordinator:
        """Recovers and reconstructs coordinator state deterministically from an existing SQLite ledger."""
        ledger = ExperimentLedger(db_path)
        valid, errors = ledger.verify_integrity()
        if not valid:
            raise ValueError(f"Ledger integrity verification failed: {errors}")

        completed_records = ledger.list_completed_records()
        completed_df = ledger.to_dataframe()
        valid_completed = completed_df[completed_df["stage"] == ExperimentStage.COMPLETED.value].reset_index(drop=True)

        if len(valid_completed) < 2:
            raise ValueError(f"Ledger at {db_path} contains fewer than 2 completed experiments ({len(valid_completed)} found).")

        # Refit models from ledger state
        direct_model = DirectPerformanceModel(two_stage_spec.process_features, spec.target_column, random_state=random_state)
        direct_model.fit(valid_completed)

        two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=random_state)
        two_stage_model.fit(valid_completed)

        prov = ScientificModelProvenance.create(
            dataset_name=spec.name,
            dataset_fingerprint=hashlib.sha256(str(len(valid_completed)).encode("utf-8")).hexdigest()[:12],
            spec_fingerprint=hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:12],
            training_experiment_ids=[r.experiment_id for r in completed_records],
            feature_columns=spec.feature_columns,
            target_columns=spec.targets,
            random_seed=random_state,
            model_types={"direct": "GPR", "stage_a": "GPR", "stage_b": "GPR"},
        )

        bundle = ScientificModelBundle(
            direct_model=direct_model,
            two_stage_model=two_stage_model,
            spec=spec,
            two_stage_spec=two_stage_spec,
            provenance=prov,
        )

        search_space = _build_search_space(spec, candidate_pool)
        candidate_cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features
        optimizer = ClosedLoopOptimizer(
            search_space=search_space,
            feature_cols=candidate_cols,
            target_col=spec.target_column,
            strategy=strategy,
            objective=spec.objective,
            random_state=random_state,
        )
        opt_state = optimizer.initialize(valid_completed)

        return cls(
            spec=spec,
            two_stage_spec=two_stage_spec,
            optimizer=optimizer,
            optimizer_state=opt_state,
            model_bundle=bundle,
            ledger=ledger,
            candidate_pool=candidate_pool,
            allow_parallel_experiments=allow_parallel_experiments,
            random_state=random_state,
        )
