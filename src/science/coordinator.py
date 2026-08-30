from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.optimization.closed_loop import ClosedLoopOptimizer, ExperimentProposal, ExperimentResult, OptimizerState
from src.optimization.search_space import ContinuousVariable, SearchSpace
from src.optimization.trust_region import TrustRegionState
from src.science.direct_baseline import DirectPerformanceModel
from src.science.ledger import ExperimentLedger
from src.science.model_bundle import ScientificModelBundle
from src.science.provenance import (
    ScientificModelProvenance,
    compute_dataset_fingerprint,
    compute_spec_fingerprint,
)
from src.science.rationale import ScientificRationale, generate_scientific_rationale
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.two_stage import MissingCharacterizationModelError, TwoStageScientificModel
from src.science.validation import InformationHorizonError, validate_record_against_spec

logger = logging.getLogger(__name__)


class PendingExperimentError(RuntimeError):
    """Raised when attempting to propose a new experiment while an unresolved experiment is already pending."""


def _build_fallback_search_space(spec: DatasetSpec, candidate_pool: pd.DataFrame) -> SearchSpace:
    """Infers bounded search space from candidate pool for numeric columns only as fallback."""
    logger.warning(
        f"No explicit SearchSpace provided for {spec.name!r}; falling back to continuous min/max range inference from candidate pool."
    )
    cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features
    variables = []
    for col in cols:
        if col not in candidate_pool.columns:
            raise ValueError(f"Candidate variable {col!r} not found in candidate pool.")

        # Check numeric
        if not pd.api.types.is_numeric_dtype(candidate_pool[col]):
            raise ValueError(
                f"Candidate variable {col!r} is non-numeric (type: {candidate_pool[col].dtype}). "
                f"Explicit SearchSpace with DiscreteVariable or CategoricalVariable is required."
            )

        c_min = float(candidate_pool[col].min())
        c_max = float(candidate_pool[col].max())
        if c_min >= c_max:
            c_min = c_min - 0.5
            c_max = c_max + 0.5
        variables.append(ContinuousVariable(name=col, lower=c_min, upper=c_max))

    return SearchSpace(name=f"{spec.name}_inferred_space", variables=variables)


class ScientificClosedLoopCoordinator:
    """Orchestrates scientific closed-loop experimentation with transactional proposals, two-stage modeling, and tamper-evident ledger."""

    def __init__(
        self,
        spec: DatasetSpec,
        two_stage_spec: TwoStageModelSpec,
        optimizer: ClosedLoopOptimizer,
        optimizer_state: OptimizerState,
        model_bundle: ScientificModelBundle,
        ledger: ExperimentLedger,
        candidate_pool: pd.DataFrame,
        search_space: SearchSpace,
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
        self.search_space = search_space
        self.allow_parallel_experiments = allow_parallel_experiments
        self.random_state = random_state
        self._last_proposal: ExperimentProposal | None = None

    @property
    def step_counter(self) -> int:
        """Returns the number of completed experiments in the ledger."""
        return len(self.ledger.list_completed_records())

    def build_stage_a_training_frame(self, char_col: str | None = None) -> pd.DataFrame:
        """Constructs the training view for Stage A (Process -> Characterization).

        Includes all records with required process features and valid characterization,
        regardless of whether downstream performance has arrived yet.
        """
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df

        process_cols = self.two_stage_spec.process_features
        # Must have process columns
        missing_proc = [c for c in process_cols if c not in all_df.columns]
        if missing_proc:
            return pd.DataFrame()

        if char_col is not None:
            if char_col not in all_df.columns:
                return pd.DataFrame()
            cols = process_cols + [char_col]
            if "experiment_id" in all_df.columns:
                cols = ["experiment_id"] + cols
            return all_df[cols].dropna().reset_index(drop=True)

        # Any characterization channel
        target_chars = [c for c in self.two_stage_spec.characterization_targets if c in all_df.columns]
        if not target_chars:
            return pd.DataFrame()

        # Rows where at least one characterization channel is present
        valid_proc = all_df[process_cols].notna().all(axis=1)
        valid_char = all_df[target_chars].notna().any(axis=1)
        return all_df[valid_proc & valid_char].reset_index(drop=True)

    def build_direct_training_frame(self) -> pd.DataFrame:
        """Constructs the training view for Direct Performance Model (Process -> Primary Performance).

        Includes all records with required process features and primary performance,
        even if characterization is pending or omitted.
        """
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df

        process_cols = self.two_stage_spec.process_features
        target_col = self.spec.target_column
        cols_needed = process_cols + [target_col]
        missing = [c for c in cols_needed if c not in all_df.columns]
        if missing:
            return pd.DataFrame()

        return all_df[all_df[cols_needed].notna().all(axis=1)].reset_index(drop=True)

    def build_stage_b_training_frame(self, target_name: str | None = None) -> pd.DataFrame:
        """Constructs the training view for Stage B (Process + Characterization -> Performance)."""
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df

        target = target_name or self.spec.target_column
        cols_needed = self.two_stage_spec.process_features + self.two_stage_spec.characterization_targets + [target]
        missing = [c for c in cols_needed if c not in all_df.columns]
        if missing:
            return pd.DataFrame()

        return all_df[all_df[cols_needed].notna().all(axis=1)].reset_index(drop=True)

    def build_optimizer_training_frame(self) -> pd.DataFrame:
        """Constructs the training view for the Bayesian Optimizer (Completed valid experiments)."""
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df
        completed_mask = all_df["stage"] == ExperimentStage.COMPLETED.value
        return all_df[completed_mask].reset_index(drop=True)

    def _refresh_model_provenance(self) -> None:
        """Refreshes the scientific model provenance capturing component-specific training datasets."""
        direct_df = self.build_direct_training_frame()
        stage_a_df = self.build_stage_a_training_frame()
        stage_b_df = self.build_stage_b_training_frame()

        direct_ids = list(direct_df["experiment_id"]) if ("experiment_id" in direct_df.columns and not direct_df.empty) else []
        stage_a_ids: dict[str, list[str]] = {}
        for c in self.two_stage_spec.characterization_targets:
            c_df = self.build_stage_a_training_frame(char_col=c)
            stage_a_ids[c] = list(c_df["experiment_id"]) if ("experiment_id" in c_df.columns and not c_df.empty) else []

        stage_b_ids: dict[str, list[str]] = {}
        for t in self.two_stage_spec.performance_targets:
            t_df = self.build_stage_b_training_frame(target_name=t)
            stage_b_ids[t] = list(t_df["experiment_id"]) if ("experiment_id" in t_df.columns and not t_df.empty) else []

        all_records_df = self.ledger.to_dataframe()
        ds_fp = compute_dataset_fingerprint(
            df=all_records_df,
            feature_cols=self.spec.feature_columns,
            target_cols=self.spec.targets,
            id_col=self.spec.id_column,
        )
        spec_fp = compute_spec_fingerprint(self.spec, self.two_stage_spec, self.search_space)
        all_training_ids = sorted(set(direct_ids + [eid for ids in stage_a_ids.values() for eid in ids]))

        new_prov = ScientificModelProvenance.create(
            dataset_name=self.spec.name,
            dataset_fingerprint=ds_fp,
            spec_fingerprint=spec_fp,
            training_experiment_ids=all_training_ids,
            feature_columns=self.spec.feature_columns,
            target_columns=self.spec.targets,
            random_seed=self.random_state,
            model_types={"direct": "GPR", "stage_a": "GPR", "stage_b": "GPR"},
            direct_training_experiment_ids=direct_ids,
            stage_a_training_experiment_ids_per_channel=stage_a_ids,
            stage_b_training_experiment_ids_per_target=stage_b_ids,
        )
        self.model_bundle.provenance = new_prov

    def propose_next(
        self,
        pre_experiment_context: Mapping[str, Any] | None = None,
        n_mc_samples: int = 64,
    ) -> tuple[ScientificExperimentRecord, ScientificRationale]:
        """Proposes the next experiment transactionally without mutating active optimizer state on failure.

        1. Pending experiment protection
        2. Deep-clones optimizer state for prospective evaluation
        3. Validates context merging and DatasetSpec information horizons
        4. Evaluates direct and two-stage scientific surrogate models
        5. Generates structured scientific rationale
        6. Atomically allocates proposal sequence and commits to tamper-evident ledger
        7. Anchors optimizer state snapshot into ledger and promotes prospective state
        """
        # 1. Pending experiment protection
        pending_records = self.ledger.list_pending_records()
        if pending_records and not self.allow_parallel_experiments:
            raise PendingExperimentError(
                f"Cannot propose a new experiment: experiment {pending_records[0].experiment_id!r} "
                f"is currently in stage {pending_records[0].stage.value}. "
                f"Record physical characterization/performance or mark failed before requesting next proposal."
            )

        # 2. Transactional prospective optimizer step
        prospective_opt_state = self.optimizer_state.clone()
        prop = self.optimizer.propose(prospective_opt_state)

        cand_dict = dict(prop.design_variables)
        cand_id = prop.candidate_id

        # 3. Merge non-controllable pre-experiment context with controllable candidate variables
        cand_vars = {k: float(cand_dict[k]) for k in (self.spec.candidate_variables or cand_dict.keys()) if k in cand_dict}
        full_process: dict[str, Any] = {}
        if pre_experiment_context:
            for k, v in pre_experiment_context.items():
                if k in cand_vars and float(v) != float(cand_vars[k]):
                    raise InformationHorizonError(
                        f"Conflict in pre_experiment_context: variable {k!r} is optimizer-controllable "
                        f"and cannot be overridden with static context ({v} vs {cand_vars[k]})."
                    )
                full_process[k] = float(v) if isinstance(v, (int, float, np.number)) else v

        full_process.update(cand_vars)

        # Ensure all required process features are present
        process_features = self.two_stage_spec.process_features
        missing_process = [k for k in process_features if k not in full_process]
        if missing_process:
            raise InformationHorizonError(
                f"Missing required pre-experiment process features: {missing_process}. "
                f"Provide them via pre_experiment_context at proposal time."
            )

        # 4. Direct model prediction
        cand_df = pd.DataFrame([full_process])
        dir_mean, dir_std = self.model_bundle.direct_model.predict(cand_df)
        dir_pred = (float(dir_mean[0]), float(dir_std[0]))

        # 5. Two-Stage model analysis (Monte Carlo uncertainty propagation)
        try:
            e2e_pred = self.model_bundle.two_stage_model.predict_end_to_end(
                cand_df,
                target_name=self.spec.target_column,
                n_mc_samples=n_mc_samples,
                seed=self.random_state + prospective_opt_state.step,
            )
        except (MissingCharacterizationModelError, RuntimeError) as exc:
            logger.warning(f"Two-stage prediction unavailable: {exc}")
            e2e_pred = None

        # 6. Generate structured, deterministic ScientificRationale
        history_df = self.ledger.to_dataframe()
        incumbent = self.optimizer_state.current_best

        # 7. Atomic sequence allocation and record construction
        proposal_seq = self.ledger.allocate_proposal_sequence(self.spec.name)
        exp_id = f"EXP_{self.spec.name[:6].upper()}_{proposal_seq:03d}"

        rationale = generate_scientific_rationale(
            experiment_id=exp_id,
            candidate_id=cand_id,
            candidate_process=full_process,
            direct_prediction=dir_pred,
            two_stage_prediction=e2e_pred,
            acquisition_method=prop.acquisition_method,
            acquisition_score=prop.acquisition_score,
            observed_history=history_df,
            process_features=process_features,
            incumbent_target=incumbent,
        )

        record = ScientificExperimentRecord(
            experiment_id=exp_id,
            candidate_id=cand_id,
            dataset_name=self.spec.name,
            stage=ExperimentStage.PROPOSED,
            pre_experiment_features=full_process,
            candidate_variables=cand_vars,
            proposal_metadata={
                "proposal_sequence": proposal_seq,
                "optimizer_step": prospective_opt_state.step,
                "strategy": prop.acquisition_method,
                "acquisition_score": prop.acquisition_score,
                "model_run_id": self.model_bundle.provenance.model_run_id,
                "learning_value_score": rationale.expected_learning_value,
                "model_disagreement_flag": rationale.model_disagreement_flag,
                "reason_code": prop.reason_code,
                "rationale_reason_code": rationale.reason_code,
                "proposal": prop.to_dict(),
            },
        )
        validate_record_against_spec(record, self.spec)

        # 8. Commit proposal to ledger and anchor optimizer snapshot
        self.ledger.record_proposal(record, spec=self.spec)
        self.ledger.save_optimizer_snapshot(prospective_opt_state.to_dict(), experiment_id=exp_id)

        # 9. Promote prospective state to active coordinator state
        self.optimizer_state = prospective_opt_state
        self._last_proposal = prop

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
            spec=self.spec,
        )

    def record_characterization(
        self,
        experiment_id: str,
        characterization: Mapping[str, Any],
        measurement_uncertainty: Mapping[str, float] | None = None,
        quality_flags: list[str] | None = None,
        allow_measurement_revision: bool = False,
    ) -> ScientificExperimentRecord:
        """Records post-experiment physical characterization (Structure/Morphology) with asynchronous lifecycle."""
        current_rec = self.ledger.get_record(experiment_id)
        if current_rec is None:
            raise KeyError(f"Experiment {experiment_id!r} not found in ledger.")

        # Check completion condition: all required characterization channels + primary performance present
        merged_chars = dict(current_rec.characterization)
        merged_chars.update(dict(characterization))
        required_chars = set(self.spec.post_experiment_characterization)
        has_req_chars = required_chars.issubset(merged_chars.keys()) if required_chars else True
        has_primary_perf = self.spec.target_column in current_rec.performance

        if has_req_chars and has_primary_perf:
            new_stage = ExperimentStage.COMPLETED
        elif has_primary_perf:
            new_stage = ExperimentStage.PERFORMANCE_MEASURED
        else:
            new_stage = ExperimentStage.CHARACTERIZED

        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=new_stage,
            event_type="CHARACTERIZATION_RECORDED",
            delta_payload={
                "characterization": dict(characterization),
                "measurement_uncertainty": dict(measurement_uncertainty or {}),
                "quality_flags": list(quality_flags or []),
                "allow_measurement_revision": allow_measurement_revision,
            },
            spec=self.spec,
        )

        # Online update of Stage A using all characterized records
        stage_a_df = self.build_stage_a_training_frame()
        if len(stage_a_df) >= 2:
            self.model_bundle.two_stage_model.stage_a.fit(stage_a_df)
            self._refresh_model_provenance()

        if rec.stage == ExperimentStage.COMPLETED:
            self._on_experiment_completed(rec)

        return rec

    def record_performance(
        self,
        experiment_id: str,
        performance: Mapping[str, Any],
        measurement_uncertainty: Mapping[str, float] | None = None,
        quality_flags: list[str] | None = None,
        allow_measurement_revision: bool = False,
    ) -> ScientificExperimentRecord:
        """Records downstream performance measurement with asynchronous lifecycle."""
        current_rec = self.ledger.get_record(experiment_id)
        if current_rec is None:
            raise KeyError(f"Experiment {experiment_id!r} not found in ledger.")

        merged_perfs = dict(current_rec.performance)
        merged_perfs.update(dict(performance))
        has_primary_perf = self.spec.target_column in merged_perfs

        # Check if characterization is required by dataset spec
        required_chars = set(self.spec.post_experiment_characterization)
        has_req_chars = current_rec.has_required_characterization(self.spec.post_experiment_characterization)

        if has_primary_perf and (not required_chars or has_req_chars):
            new_stage = ExperimentStage.COMPLETED
        elif has_primary_perf:
            new_stage = ExperimentStage.PERFORMANCE_MEASURED
        else:
            new_stage = current_rec.stage

        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=new_stage,
            event_type="PERFORMANCE_RECORDED",
            delta_payload={
                "performance": dict(performance),
                "measurement_uncertainty": dict(measurement_uncertainty or {}),
                "quality_flags": list(quality_flags or []),
                "allow_measurement_revision": allow_measurement_revision,
            },
            spec=self.spec,
        )

        # Online update of Direct Performance Model using all available process -> primary target data
        direct_df = self.build_direct_training_frame()
        if len(direct_df) >= 2:
            self.model_bundle.direct_model.fit(direct_df)
            self._refresh_model_provenance()

        if rec.stage == ExperimentStage.COMPLETED:
            self._on_experiment_completed(rec)

        return rec

    def _on_experiment_completed(self, rec: ScientificExperimentRecord) -> None:
        """Handles completion of an experiment: feeds optimizer, refits models, refreshes provenance, snapshots state."""
        target_val = float(rec.performance[self.spec.target_column])
        exp_res = ExperimentResult(
            candidate_id=rec.candidate_id,
            design_variables=rec.pre_experiment_features,
            target_value=target_val,
        )

        # Restore exact proposal semantics from record or cached proposal
        prop = getattr(self, "_last_proposal", None)
        if prop is None or prop.candidate_id != rec.candidate_id:
            if rec.proposal_metadata.get("proposal"):
                prop = ExperimentProposal.from_dict(rec.proposal_metadata["proposal"])
            else:
                prop = ExperimentProposal(
                    candidate_id=rec.candidate_id,
                    design_variables=rec.pre_experiment_features,
                    predicted_performance=0.0,
                    prediction_uncertainty=1.0,
                    acquisition_score=float(rec.proposal_metadata.get("acquisition_score", 0.0)),
                    acquisition_method=str(rec.proposal_metadata.get("strategy", self.optimizer.strategy)),
                    trust_region_center=None,
                    trust_region_radius=None,
                    recommendation_reason="",
                    reason_code=str(rec.proposal_metadata.get("reason_code", "OBSERVATION_RECORDED")),
                    distance_to_nearest_observed=0.0,
                    step=int(rec.proposal_metadata.get("optimizer_step", self.optimizer_state.step)),
                )

        self.optimizer.observe(self.optimizer_state, prop, exp_res)

        # Refit models with complete component views
        direct_df = self.build_direct_training_frame()
        if len(direct_df) >= 2:
            self.model_bundle.direct_model.fit(direct_df)

        stage_a_df = self.build_stage_a_training_frame()
        if len(stage_a_df) >= 2:
            self.model_bundle.two_stage_model.stage_a.fit(stage_a_df)

        stage_b_df = self.build_stage_b_training_frame()
        if len(stage_b_df) >= 2:
            self.model_bundle.two_stage_model.stage_b.fit(stage_b_df)

        self._refresh_model_provenance()

        # Save optimizer state snapshot into ledger
        self.ledger.save_optimizer_snapshot(self.optimizer_state.to_dict(), experiment_id=rec.experiment_id)

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
            spec=self.spec,
        )

    @classmethod
    def initialize_new(
        cls,
        spec: DatasetSpec,
        two_stage_spec: TwoStageModelSpec,
        initial_data: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        search_space: SearchSpace | None = None,
        db_path: Path | str = ":memory:",
        strategy: str = "expected_improvement",
        allow_parallel_experiments: bool = False,
        random_state: int = 42,
    ) -> ScientificClosedLoopCoordinator:
        """Initializes a new coordinator from historical seed dataset and starts a fresh ledger."""
        ledger = ExperimentLedger(db_path)

        # 1. Ingest initial seed experiments into ledger
        for i, row in initial_data.iterrows():
            eid = str(row.get("experiment_id", f"EXP_INIT_{i:03d}"))
            cid = str(row.get("candidate_id", f"CAND_INIT_{i:03d}"))
            pre_feats = {k: float(row[k]) for k in spec.pre_experiment_features if k in row and pd.notna(row[k])}
            cand_feats = {k: float(row[k]) for k in spec.candidate_variables if k in row and pd.notna(row[k])} or pre_feats
            char_feats = {k: float(row[k]) for k in spec.post_experiment_characterization if k in row and pd.notna(row[k])}
            perf_feats = {k: float(row[k]) for k in spec.targets if k in row and pd.notna(row[k])}

            rec = ScientificExperimentRecord(
                experiment_id=eid,
                candidate_id=cid,
                dataset_name=spec.name,
                stage=ExperimentStage.PROPOSED,
                pre_experiment_features=pre_feats,
                candidate_variables=cand_feats,
            )
            ledger.record_proposal(rec, spec=spec)
            ledger.append_transition(eid, ExperimentStage.EXECUTED, "INIT_EXECUTED", {}, spec=spec)
            if char_feats:
                ledger.append_transition(eid, ExperimentStage.CHARACTERIZED, "INIT_CHAR", {"characterization": char_feats}, spec=spec)
            if perf_feats:
                ledger.append_transition(eid, ExperimentStage.COMPLETED, "INIT_PERF", {"performance": perf_feats}, spec=spec)

        # 2. Fit initial models
        direct_model = DirectPerformanceModel(two_stage_spec.process_features, spec.target_column, random_state=random_state)
        direct_model.fit(initial_data)

        two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=random_state)
        two_stage_model.fit(initial_data)

        resolved_space = search_space if search_space is not None else _build_fallback_search_space(spec, candidate_pool)

        ds_fp = compute_dataset_fingerprint(
            initial_data,
            feature_cols=spec.feature_columns,
            target_cols=spec.targets,
            id_col=spec.id_column,
        )
        spec_fp = compute_spec_fingerprint(spec, two_stage_spec, resolved_space)
        exp_ids = (
            list(initial_data[spec.id_column])
            if spec.id_column in initial_data.columns
            else [f"EXP_INIT_{i:03d}" for i in range(len(initial_data))]
        )

        prov = ScientificModelProvenance.create(
            dataset_name=spec.name,
            dataset_fingerprint=ds_fp,
            spec_fingerprint=spec_fp,
            training_experiment_ids=exp_ids,
            feature_columns=spec.feature_columns,
            target_columns=spec.targets,
            random_seed=random_state,
            model_types={"direct": "GPR", "stage_a": "GPR", "stage_b": "GPR"},
            direct_training_experiment_ids=exp_ids,
            stage_a_training_experiment_ids_per_channel={c: exp_ids for c in two_stage_spec.characterization_targets},
            stage_b_training_experiment_ids_per_target={t: exp_ids for t in two_stage_spec.performance_targets},
        )

        bundle = ScientificModelBundle(
            direct_model=direct_model,
            two_stage_model=two_stage_model,
            spec=spec,
            two_stage_spec=two_stage_spec,
            provenance=prov,
        )

        # 3. Explicit or fallback SearchSpace
        candidate_cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features
        optimizer = ClosedLoopOptimizer(
            search_space=resolved_space,
            feature_cols=candidate_cols,
            target_col=spec.target_column,
            strategy=strategy,
            objective=spec.objective,
            random_state=random_state,
        )
        opt_state = optimizer.initialize(initial_data)
        ledger.save_optimizer_snapshot(opt_state.to_dict())

        return cls(
            spec=spec,
            two_stage_spec=two_stage_spec,
            optimizer=optimizer,
            optimizer_state=opt_state,
            model_bundle=bundle,
            ledger=ledger,
            candidate_pool=candidate_pool,
            search_space=resolved_space,
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
        search_space: SearchSpace | None = None,
        strategy: str = "expected_improvement",
        allow_parallel_experiments: bool = False,
        random_state: int = 42,
    ) -> ScientificClosedLoopCoordinator:
        """Recovers and reconstructs coordinator state deterministically from an existing SQLite ledger."""
        ledger = ExperimentLedger(db_path)
        valid, errors = ledger.verify_integrity()
        if not valid:
            raise ValueError(f"Ledger integrity verification failed: {errors}")

        resolved_space = search_space if search_space is not None else _build_fallback_search_space(spec, candidate_pool)

        # Direct model refit
        direct_model = DirectPerformanceModel(two_stage_spec.process_features, spec.target_column, random_state=random_state)
        # Two-stage model refit
        two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=random_state)

        all_df = ledger.to_dataframe()
        valid_completed = all_df[all_df["stage"] == ExperimentStage.COMPLETED.value].reset_index(drop=True)

        if len(valid_completed) >= 2:
            direct_model.fit(valid_completed)
            two_stage_model.fit(valid_completed)

        ds_fp = compute_dataset_fingerprint(
            all_df,
            feature_cols=spec.feature_columns,
            target_cols=spec.targets,
            id_col=spec.id_column,
        )
        spec_fp = compute_spec_fingerprint(spec, two_stage_spec, resolved_space)

        completed_records = ledger.list_completed_records()
        prov = ScientificModelProvenance.create(
            dataset_name=spec.name,
            dataset_fingerprint=ds_fp,
            spec_fingerprint=spec_fp,
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

        candidate_cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features
        optimizer = ClosedLoopOptimizer(
            search_space=resolved_space,
            feature_cols=candidate_cols,
            target_col=spec.target_column,
            strategy=strategy,
            objective=spec.objective,
            random_state=random_state,
        )

        # Restore optimizer state from snapshot if available, else initialize
        opt_state = optimizer.initialize(valid_completed)
        latest_snapshot = ledger.get_latest_optimizer_snapshot()
        if latest_snapshot is not None:
            opt_state.step = int(latest_snapshot.get("step", opt_state.step))
            opt_state.current_best = float(latest_snapshot.get("current_best", opt_state.current_best))
            if latest_snapshot.get("trust_region_state") and opt_state.trust_region:
                opt_state.trust_region.state = TrustRegionState.from_dict(latest_snapshot["trust_region_state"])
            if latest_snapshot.get("history"):
                opt_state.history = list(latest_snapshot["history"])
            optimizer._fit_surrogate(opt_state)

        coord = cls(
            spec=spec,
            two_stage_spec=two_stage_spec,
            optimizer=optimizer,
            optimizer_state=opt_state,
            model_bundle=bundle,
            ledger=ledger,
            candidate_pool=candidate_pool,
            search_space=resolved_space,
            allow_parallel_experiments=allow_parallel_experiments,
            random_state=random_state,
        )

        # Restore pending proposal if any
        pending_records = ledger.list_pending_records()
        if pending_records:
            pending_rec = pending_records[0]
            if pending_rec.proposal_metadata.get("proposal"):
                coord._last_proposal = ExperimentProposal.from_dict(pending_rec.proposal_metadata["proposal"])

        return coord
