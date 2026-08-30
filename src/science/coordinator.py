from __future__ import annotations

import copy
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.optimization.closed_loop import ClosedLoopOptimizer, ExperimentProposal, ExperimentResult, OptimizerState
from src.optimization.search_space import ContinuousVariable, SearchSpace
from src.optimization.trust_region import TrustRegionState, TuRBOTrustRegion
from src.science.direct_baseline import DirectPerformanceModel
from src.science.ledger import ExperimentLedger, _canonical_json
from src.science.model_bundle import ScientificModelBundle
from src.science.provenance import (
    ScientificModelProvenance,
    compute_dataset_fingerprint,
    compute_search_space_fingerprint,
    compute_spec_fingerprint,
)
from src.science.rationale import ScientificRationale, generate_scientific_rationale
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.two_stage import MissingCharacterizationModelError, TwoStageScientificModel
from src.science.validation import InformationHorizonError, validate_record_against_spec

logger = logging.getLogger(__name__)


class PendingExperimentError(RuntimeError):
    """Raised when attempting to propose a new experiment while an unresolved experiment is already pending."""


class ResumeStateMismatchError(ValueError):
    """Raised when ledger optimizer snapshot schema/state mismatches the current DatasetSpec or SearchSpace."""


class PrimaryTargetRevisionError(ValueError):
    """Raised when attempting to revise the primary target value on an already COMPLETED experiment."""


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
                f"Explicit SearchSpace with numeric DiscreteVariable or pre-encoded categorical features is required."
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

    def _is_experiment_observed_by_optimizer(self, experiment_id: str) -> bool:
        """Defensive idempotence check: returns True if experiment_id has already been observed by active optimizer state."""
        if self.optimizer_state is None:
            return False
        for r in self.optimizer_state.observed_records:
            if r.get("experiment_id") == experiment_id:
                return True
        for h in self.optimizer_state.history:
            if h.get("experiment_id") == experiment_id:
                return True
        return False

    def build_stage_a_training_frame(self, char_col: str | None = None) -> pd.DataFrame:
        """Constructs the training view for Stage A (Process -> Characterization).

        Includes all records with required process features and valid characterization,
        regardless of whether downstream performance has arrived yet.
        Excludes FAILED and CANCELLED records.
        """
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df

        valid_stage_mask = ~all_df["stage"].isin([ExperimentStage.FAILED.value, ExperimentStage.CANCELLED.value])
        all_df = all_df[valid_stage_mask].reset_index(drop=True)
        if all_df.empty:
            return all_df

        process_cols = self.two_stage_spec.process_features
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

        target_chars = [c for c in self.two_stage_spec.characterization_targets if c in all_df.columns]
        if not target_chars:
            return pd.DataFrame()

        valid_proc = all_df[process_cols].notna().all(axis=1)
        valid_char = all_df[target_chars].notna().any(axis=1)
        return all_df[valid_proc & valid_char].reset_index(drop=True)

    def build_direct_training_frame(self) -> pd.DataFrame:
        """Constructs the training view for Direct Performance Model (Process -> Primary Performance).

        Includes all records with required process features and primary performance,
        even if characterization is pending or omitted.
        Excludes FAILED and CANCELLED records.
        """
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df

        valid_stage_mask = ~all_df["stage"].isin([ExperimentStage.FAILED.value, ExperimentStage.CANCELLED.value])
        all_df = all_df[valid_stage_mask].reset_index(drop=True)
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
        """Constructs the training view for Stage B (Process + Characterization -> Performance).

        Excludes FAILED and CANCELLED records. Supports independent per-target missingness.
        """
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df

        valid_stage_mask = ~all_df["stage"].isin([ExperimentStage.FAILED.value, ExperimentStage.CANCELLED.value])
        all_df = all_df[valid_stage_mask].reset_index(drop=True)
        if all_df.empty:
            return all_df

        req_inputs = self.two_stage_spec.process_features + self.two_stage_spec.characterization_targets
        missing_inputs = [c for c in req_inputs if c not in all_df.columns]
        if missing_inputs:
            return pd.DataFrame()

        valid_inputs = all_df[req_inputs].notna().all(axis=1)

        if target_name is not None:
            if target_name not in all_df.columns:
                return pd.DataFrame()
            cols_needed = req_inputs + [target_name]
            if "experiment_id" in all_df.columns:
                cols_needed = ["experiment_id"] + cols_needed
            return all_df[valid_inputs & all_df[target_name].notna()][cols_needed].reset_index(drop=True)

        perf_targets = [t for t in self.two_stage_spec.performance_targets if t in all_df.columns]
        if not perf_targets:
            return pd.DataFrame()
        valid_any_perf = all_df[perf_targets].notna().any(axis=1)
        return all_df[valid_inputs & valid_any_perf].reset_index(drop=True)

    def build_optimizer_training_frame(self) -> pd.DataFrame:
        """Constructs the training view for the Bayesian Optimizer (Completed valid experiments).

        Excludes FAILED and CANCELLED records.
        """
        all_df = self.ledger.to_dataframe()
        if all_df.empty:
            return all_df
        completed_mask = all_df["stage"] == ExperimentStage.COMPLETED.value
        return all_df[completed_mask].reset_index(drop=True)

    def _refit_direct_model(self) -> None:
        """Refits the direct performance model or resets it if insufficient valid training data remains."""
        direct_df = self.build_direct_training_frame()
        self.model_bundle.direct_model.fit(direct_df)

    def _refit_stage_a_model(self) -> None:
        """Refits Stage A characterization channels or resets channels with insufficient valid data."""
        stage_a_df = self.build_stage_a_training_frame()
        self.model_bundle.two_stage_model.stage_a.fit(stage_a_df)

    def _refit_stage_b_model(self) -> None:
        """Refits Stage B performance targets or resets targets with insufficient valid data."""
        stage_b_df = self.build_stage_b_training_frame()
        self.model_bundle.two_stage_model.stage_b.fit(stage_b_df)

    def _refit_scientific_models(self) -> None:
        """Refits all scientific component models and refreshes end-to-end model status and provenance."""
        self._refit_direct_model()
        self._refit_stage_a_model()
        self._refit_stage_b_model()
        self._refresh_model_provenance()

    def _refresh_model_provenance(self) -> None:
        """Refreshes the scientific model provenance capturing component-specific training datasets."""
        direct_df = self.build_direct_training_frame()

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
        all_training_ids = sorted(
            set(
                direct_ids
                + [eid for ids in stage_a_ids.values() for eid in ids]
                + [eid for ids in stage_b_ids.values() for eid in ids]
            )
        )

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

    def _rebuild_optimizer_from_ledger(self) -> None:
        """Deterministically rebuilds optimizer state from valid completed observations in the ledger.

        Reconstructs observed_records, current_best, GP surrogate, history, and TuRBO trust region
        by replaying surviving completed online observations in chronological order
        using their original ExperimentProposal specifications.
        Persists a new authoritative optimizer snapshot to the ledger and refreshes provenance.
        """
        completed_records = self.ledger.list_completed_records()

        # Partition surviving completed records into initial seed records and online proposed experiments
        seed_records: list[ScientificExperimentRecord] = []
        online_records: list[ScientificExperimentRecord] = []

        for r in completed_records:
            if r.proposal_metadata and r.proposal_metadata.get("proposal"):
                online_records.append(r)
            else:
                seed_records.append(r)

        # Sort online records chronologically by proposal sequence
        online_records.sort(key=lambda x: int(x.proposal_metadata.get("proposal_sequence", 0)))

        candidate_cols = self.spec.candidate_columns or self.spec.candidate_variables or self.spec.pre_experiment_features

        if seed_records:
            seed_rows = []
            for r in seed_records:
                row_dict = {
                    "candidate_id": r.candidate_id,
                    "experiment_id": r.experiment_id,
                    **{k: float(r.pre_experiment_features[k]) for k in candidate_cols if k in r.pre_experiment_features},
                    self.spec.target_column: float(r.performance[self.spec.target_column]),
                }
                seed_rows.append(row_dict)
            new_opt_state = self.optimizer.initialize(seed_rows)
        elif online_records:
            first_rec = online_records.pop(0)
            first_row = {
                "candidate_id": first_rec.candidate_id,
                "experiment_id": first_rec.experiment_id,
                **{k: float(first_rec.pre_experiment_features[k]) for k in candidate_cols if k in first_rec.pre_experiment_features},
                self.spec.target_column: float(first_rec.performance[self.spec.target_column]),
            }
            new_opt_state = self.optimizer.initialize([first_row])
        else:
            raise RuntimeError("Cannot rebuild optimizer: no completed valid records exist in the ledger.")

        # Replay surviving online proposed observations
        for rec in online_records:
            prop_data = rec.proposal_metadata.get("proposal")
            if prop_data:
                prop = ExperimentProposal.from_dict(prop_data)
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
                    step=int(rec.proposal_metadata.get("optimizer_step", new_opt_state.step + 1)),
                )
            new_opt_state.step = int(rec.proposal_metadata.get("optimizer_step", prop.step))
            target_val = float(rec.performance[self.spec.target_column])
            exp_res = ExperimentResult(
                candidate_id=rec.candidate_id,
                design_variables=rec.pre_experiment_features,
                target_value=target_val,
                observations={"experiment_id": rec.experiment_id},
            )
            self.optimizer.observe(new_opt_state, prop, exp_res)

        self.optimizer._fit_surrogate(new_opt_state)
        self.optimizer_state = new_opt_state

        # Persist new authoritative snapshot
        self.ledger.save_optimizer_snapshot(self.optimizer_state.to_dict())

        # Refit scientific models and refresh provenance
        self._refit_scientific_models()

    def propose_next(
        self,
        pre_experiment_context: Mapping[str, Any] | None = None,
        n_mc_samples: int = 64,
    ) -> tuple[ScientificExperimentRecord, ScientificRationale]:
        """Proposes the next experiment transactionally without mutating active optimizer state on failure."""
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

        # 6. Compute prospective optimizer state hash and search space fingerprint
        history_df = self.ledger.to_dataframe()
        if not history_df.empty and "stage" in history_df.columns:
            history_df = history_df[~history_df["stage"].isin([ExperimentStage.FAILED.value, ExperimentStage.CANCELLED.value])].reset_index(drop=True)
        incumbent = self.optimizer_state.current_best
        prospective_snap = prospective_opt_state.to_dict()
        opt_state_hash = hashlib.sha256(_canonical_json(prospective_snap).encode("utf-8")).hexdigest()
        search_space_fp = compute_search_space_fingerprint(self.search_space)

        # 7. Atomic proposal transaction (sequence allocation + proposal record + optimizer snapshot)
        def prop_metadata_builder(final_exp_id: str, final_seq: int) -> dict[str, Any]:
            rationale = generate_scientific_rationale(
                experiment_id=final_exp_id,
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
            return {
                "proposal_sequence": final_seq,
                "optimizer_step": prospective_opt_state.step,
                "strategy": prop.acquisition_method,
                "acquisition_score": prop.acquisition_score,
                "model_run_id": self.model_bundle.provenance.model_run_id,
                "learning_value_score": rationale.expected_learning_value,
                "model_disagreement_flag": rationale.model_disagreement_flag,
                "reason_code": prop.reason_code,
                "rationale_reason_code": rationale.reason_code,
                "optimizer_state_hash": opt_state_hash,
                "search_space_fingerprint": search_space_fp,
                "proposal": prop.to_dict(),
                "rationale": rationale.to_dict(),
            }

        record = self.ledger.commit_proposal_transaction(
            dataset_name=self.spec.name,
            candidate_id=cand_id,
            pre_experiment_features=full_process,
            candidate_variables=cand_vars,
            proposal_metadata_builder=prop_metadata_builder,
            optimizer_snapshot=prospective_snap,
            spec=self.spec,
        )

        # Reconstruct rationale object
        rationale_data = record.proposal_metadata.get("rationale")
        if rationale_data:
            rationale = ScientificRationale(**rationale_data)
        else:
            rationale = generate_scientific_rationale(
                experiment_id=record.experiment_id,
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

        # 8. Promote prospective state to active coordinator state
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

        was_completed = (current_rec.stage == ExperimentStage.COMPLETED)

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

        # Online update of Stage A and Stage B models
        self._refit_stage_a_model()
        if was_completed:
            self._refit_stage_b_model()
        self._refresh_model_provenance()

        # Only observe if transitioned to COMPLETED for the first time
        if rec.stage == ExperimentStage.COMPLETED and not was_completed:
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

        if current_rec.stage == ExperimentStage.COMPLETED:
            return self.record_additional_performance(
                experiment_id=experiment_id,
                performance=performance,
                measurement_uncertainty=measurement_uncertainty,
                quality_flags=quality_flags,
                allow_measurement_revision=allow_measurement_revision,
            )

        was_completed = (current_rec.stage == ExperimentStage.COMPLETED)
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

        # Online update of Direct Performance Model
        self._refit_direct_model()
        self._refresh_model_provenance()

        if rec.stage == ExperimentStage.COMPLETED and not was_completed:
            self._on_experiment_completed(rec)

        return rec

    def record_additional_performance(
        self,
        experiment_id: str,
        performance: Mapping[str, Any],
        measurement_uncertainty: Mapping[str, float] | None = None,
        quality_flags: list[str] | None = None,
        allow_measurement_revision: bool = False,
    ) -> ScientificExperimentRecord:
        """Records additional secondary performance measurements on an already COMPLETED experiment without re-triggering optimizer observation."""
        current_rec = self.ledger.get_record(experiment_id)
        if current_rec is None:
            raise KeyError(f"Experiment {experiment_id!r} not found in ledger.")

        if current_rec.stage != ExperimentStage.COMPLETED:
            return self.record_performance(
                experiment_id=experiment_id,
                performance=performance,
                measurement_uncertainty=measurement_uncertainty,
                quality_flags=quality_flags,
                allow_measurement_revision=allow_measurement_revision,
            )

        # Enforce strict policy: Primary target column cannot be revised after experiment is COMPLETED
        if self.spec.target_column in performance:
            existing_primary = current_rec.performance.get(self.spec.target_column)
            new_primary = float(performance[self.spec.target_column])
            if existing_primary is not None and float(existing_primary) != new_primary:
                raise PrimaryTargetRevisionError(
                    f"Primary target column {self.spec.target_column!r} cannot be revised after experiment "
                    f"{experiment_id!r} is COMPLETED (existing: {existing_primary}, attempted: {new_primary})."
                )

        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.COMPLETED,
            event_type="ADDITIONAL_PERFORMANCE_RECORDED",
            delta_payload={
                "performance": dict(performance),
                "measurement_uncertainty": dict(measurement_uncertainty or {}),
                "quality_flags": list(quality_flags or []),
                "allow_measurement_revision": allow_measurement_revision,
            },
            spec=self.spec,
        )

        # Refit Stage B & Direct models
        self._refit_stage_b_model()
        self._refit_direct_model()
        self._refresh_model_provenance()
        return rec

    def _on_experiment_completed(self, rec: ScientificExperimentRecord) -> None:
        """Handles completion of an experiment: feeds optimizer, refits models, refreshes provenance, snapshots state."""
        # Defensive idempotence guard: do not re-observe if already recorded in optimizer state
        if self._is_experiment_observed_by_optimizer(rec.experiment_id):
            logger.warning(
                f"Experiment {rec.experiment_id!r} is already observed in optimizer state; skipping duplicate observation."
            )
            return

        target_val = float(rec.performance[self.spec.target_column])
        exp_res = ExperimentResult(
            candidate_id=rec.candidate_id,
            design_variables=rec.pre_experiment_features,
            target_value=target_val,
            observations={"experiment_id": rec.experiment_id},
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
        self._refit_scientific_models()

        # Save optimizer state snapshot into ledger
        self.ledger.save_optimizer_snapshot(self.optimizer_state.to_dict(), experiment_id=rec.experiment_id)

    def record_failed(
        self,
        experiment_id: str,
        failure_reason: str,
        quality_flags: list[str] | None = None,
    ) -> ScientificExperimentRecord:
        """Records experimental failure in audit trail and rebuilds optimizer/models if previously observed."""
        was_observed = self._is_experiment_observed_by_optimizer(experiment_id)
        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.FAILED,
            event_type="EXPERIMENT_FAILED",
            delta_payload={
                "failure_reason": failure_reason,
                "quality_flags": list(quality_flags or ["FABRICATION_FAILURE"]),
            },
            spec=self.spec,
        )

        if was_observed:
            self._rebuild_optimizer_from_ledger()
        else:
            self._refit_scientific_models()
        return rec

    def record_cancelled(
        self,
        experiment_id: str,
        reason: str,
    ) -> ScientificExperimentRecord:
        """Records cancellation of an experiment and rebuilds optimizer/models if previously observed."""
        was_observed = self._is_experiment_observed_by_optimizer(experiment_id)
        rec = self.ledger.append_transition(
            experiment_id=experiment_id,
            new_stage=ExperimentStage.CANCELLED,
            event_type="EXPERIMENT_CANCELLED",
            delta_payload={"failure_reason": reason},
            spec=self.spec,
        )

        if was_observed:
            self._rebuild_optimizer_from_ledger()
        else:
            self._refit_scientific_models()
        return rec

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
        """Initializes a new coordinator from historical seed dataset supporting varying measurement completeness."""
        ledger = ExperimentLedger(db_path)

        # 1. Ingest initial seed experiments into ledger respecting true information horizons
        cand_vars_keys = spec.candidate_variables or []
        for i, row in initial_data.iterrows():
            eid = str(row[spec.id_column]) if spec.id_column in row and pd.notna(row[spec.id_column]) else str(row.get("experiment_id", f"EXP_INIT_{i:03d}"))
            cid = str(row[spec.candidate_id_column]) if spec.candidate_id_column in row and pd.notna(row[spec.candidate_id_column]) else str(row.get("candidate_id", f"CAND_INIT_{i:03d}"))
            pre_feats = {k: float(row[k]) for k in spec.pre_experiment_features if k in row and pd.notna(row[k])}
            cand_feats = {k: float(row[k]) for k in cand_vars_keys if k in row and pd.notna(row[k])}
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

            has_char = bool(char_feats)
            required_chars = set(spec.post_experiment_characterization)
            has_req_chars = required_chars.issubset(char_feats.keys()) if required_chars else True
            has_primary_perf = spec.target_column in perf_feats

            if has_char:
                ledger.append_transition(eid, ExperimentStage.CHARACTERIZED, "INIT_CHAR", {"characterization": char_feats}, spec=spec)

            if perf_feats:
                if has_primary_perf and (not required_chars or has_req_chars):
                    target_stage = ExperimentStage.COMPLETED
                else:
                    target_stage = ExperimentStage.PERFORMANCE_MEASURED
                ledger.append_transition(eid, target_stage, "INIT_PERF", {"performance": perf_feats}, spec=spec)

        resolved_space = search_space if search_space is not None else _build_fallback_search_space(spec, candidate_pool)

        # 2. Fit initial models from component training views
        direct_model = DirectPerformanceModel(two_stage_spec.process_features, spec.target_column, random_state=random_state)
        two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=random_state)

        prov = ScientificModelProvenance.create(
            dataset_name=spec.name,
            dataset_fingerprint="ds_init",
            spec_fingerprint="spec_init",
            training_experiment_ids=[],
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

        # 3. Instantiate coordinator
        coord = cls(
            spec=spec,
            two_stage_spec=two_stage_spec,
            optimizer=optimizer,
            optimizer_state=None,  # type: ignore
            model_bundle=bundle,
            ledger=ledger,
            candidate_pool=candidate_pool,
            search_space=resolved_space,
            allow_parallel_experiments=allow_parallel_experiments,
            random_state=random_state,
        )

        # 4. Refit models using coordinator's pure training frame methods
        coord._refit_scientific_models()

        # 5. Initialize optimizer from completed valid observations
        opt_df = coord.build_optimizer_training_frame()
        init_rows = []
        for _, row in opt_df.iterrows():
            r_dict = {
                "candidate_id": str(row.get("candidate_id", "")),
                "experiment_id": str(row.get("experiment_id", "")),
                **{k: float(row[k]) for k in candidate_cols if k in row and pd.notna(row[k])},
                spec.target_column: float(row[spec.target_column]),
            }
            init_rows.append(r_dict)
        opt_state = optimizer.initialize(init_rows if init_rows else opt_df)
        coord.optimizer_state = opt_state
        ledger.save_optimizer_snapshot(opt_state.to_dict())

        return coord

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
        """Recovers and reconstructs coordinator state deterministically from an existing SQLite ledger, performing two-way observation reconciliation."""
        ledger = ExperimentLedger(db_path)
        valid, errors = ledger.verify_integrity()
        if not valid:
            raise ValueError(f"Ledger integrity verification failed: {errors}")

        resolved_space = search_space if search_space is not None else _build_fallback_search_space(spec, candidate_pool)
        candidate_cols = spec.candidate_columns or spec.candidate_variables or spec.pre_experiment_features

        # Retrieve authoritative snapshot from verified event history
        latest_snapshot = ledger.get_latest_verified_optimizer_snapshot()
        if latest_snapshot is not None:
            # Validate snapshot schema against current spec & candidate columns
            snap_target = latest_snapshot.get("target_col")
            if snap_target and snap_target != spec.target_column:
                raise ResumeStateMismatchError(
                    f"Snapshot target column {snap_target!r} does not match DatasetSpec target_column {spec.target_column!r}"
                )
            snap_obj = latest_snapshot.get("objective")
            if snap_obj and snap_obj != spec.objective:
                raise ResumeStateMismatchError(
                    f"Snapshot objective {snap_obj!r} does not match DatasetSpec objective {spec.objective!r}"
                )
            snap_feats = latest_snapshot.get("feature_cols")
            if snap_feats and list(snap_feats) != list(candidate_cols):
                raise ResumeStateMismatchError(
                    f"Snapshot feature columns {snap_feats} do not match coordinator candidate columns {candidate_cols}"
                )

        direct_model = DirectPerformanceModel(two_stage_spec.process_features, spec.target_column, random_state=random_state)
        two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=random_state)

        prov = ScientificModelProvenance.create(
            dataset_name=spec.name,
            dataset_fingerprint="ds_resumed",
            spec_fingerprint="spec_resumed",
            training_experiment_ids=[],
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

        optimizer = ClosedLoopOptimizer(
            search_space=resolved_space,
            feature_cols=candidate_cols,
            target_col=spec.target_column,
            strategy=strategy,
            objective=spec.objective,
            random_state=random_state,
        )

        coord = cls(
            spec=spec,
            two_stage_spec=two_stage_spec,
            optimizer=optimizer,
            optimizer_state=None,  # type: ignore
            model_bundle=bundle,
            ledger=ledger,
            candidate_pool=candidate_pool,
            search_space=resolved_space,
            allow_parallel_experiments=allow_parallel_experiments,
            random_state=random_state,
        )

        # Refit models using the pure refit helpers (handles reset if < 2 valid rows)
        coord._refit_scientific_models()

        # Optimizer State Reconstruction & Reconciliation
        opt_df = coord.build_optimizer_training_frame()
        completed_records = ledger.list_completed_records()

        if latest_snapshot is not None:
            # 1. Restore optimizer state from authoritative snapshot
            opt_state = OptimizerState(
                observed_records=[dict(r) for r in latest_snapshot.get("observed_records", [])],
                feature_cols=list(candidate_cols),
                target_col=spec.target_column,
                objective=spec.objective,
                step=int(latest_snapshot.get("step", 0)),
                current_best=float(latest_snapshot.get("current_best", -np.inf if spec.objective == "maximize" else np.inf)),
                trust_region=None,
                gp_model=None,
                scaler=None,
                history=[dict(h) for h in latest_snapshot.get("history", [])],
            )
            if "turbo" in strategy:
                opt_state.trust_region = TuRBOTrustRegion(search_space=resolved_space)
                if latest_snapshot.get("trust_region_state"):
                    opt_state.trust_region.state = TrustRegionState.from_dict(latest_snapshot["trust_region_state"])
                else:
                    if opt_state.observed_records:
                        targets = [float(r[spec.target_column]) for r in opt_state.observed_records]
                        b_idx = int(np.argmax(targets) if spec.objective == "maximize" else np.argmin(targets))
                        opt_state.trust_region.initialize(opt_state.observed_records[b_idx], opt_state.current_best)

            coord.optimizer_state = opt_state

            # 2. Two-way reconciliation against authoritative ledger completed records
            ledger_valid_exp_ids = {r.experiment_id for r in completed_records}
            snapshot_obs_exp_ids = {str(r["experiment_id"]) for r in opt_state.observed_records if r.get("experiment_id")}

            has_missing_exp_id = len(snapshot_obs_exp_ids) < len(opt_state.observed_records)

            if has_missing_exp_id:
                # Legacy snapshot migration: lacks experiment_id; perform one-time full rebuild from ledger and persist new snapshot
                coord._rebuild_optimizer_from_ledger()
            elif snapshot_obs_exp_ids == ledger_valid_exp_ids:
                # Snapshot exactly matches ledger valid observations
                optimizer._fit_surrogate(opt_state)
            elif snapshot_obs_exp_ids.issubset(ledger_valid_exp_ids):
                # Snapshot missing some completed observations (crash after completion before snapshot write)
                missing_completed = [rec for rec in completed_records if rec.experiment_id not in snapshot_obs_exp_ids]
                for rec in missing_completed:
                    if rec.proposal_metadata.get("proposal"):
                        prop = ExperimentProposal.from_dict(rec.proposal_metadata["proposal"])
                    else:
                        prop = ExperimentProposal(
                            candidate_id=rec.candidate_id,
                            design_variables=rec.pre_experiment_features,
                            predicted_performance=0.0,
                            prediction_uncertainty=1.0,
                            acquisition_score=float(rec.proposal_metadata.get("acquisition_score", 0.0)),
                            acquisition_method=str(rec.proposal_metadata.get("strategy", strategy)),
                            trust_region_center=None,
                            trust_region_radius=None,
                            recommendation_reason="",
                            reason_code=str(rec.proposal_metadata.get("reason_code", "OBSERVATION_RECORDED")),
                            distance_to_nearest_observed=0.0,
                            step=int(rec.proposal_metadata.get("optimizer_step", opt_state.step + 1)),
                        )
                    opt_state.step = int(rec.proposal_metadata.get("optimizer_step", prop.step))
                    target_val = float(rec.performance[spec.target_column])
                    exp_res = ExperimentResult(
                        candidate_id=rec.candidate_id,
                        design_variables=rec.pre_experiment_features,
                        target_value=target_val,
                        observations={"experiment_id": rec.experiment_id},
                    )
                    optimizer.observe(opt_state, prop, exp_res)

                ledger.save_optimizer_snapshot(opt_state.to_dict())
                optimizer._fit_surrogate(opt_state)
            else:
                # Snapshot contains stale/invalid observations that were later invalidated (crash before rebuilt snapshot save)
                coord._rebuild_optimizer_from_ledger()
        else:
            opt_state = optimizer.initialize(opt_df)
            coord.optimizer_state = opt_state
            ledger.save_optimizer_snapshot(opt_state.to_dict())

        # Restore pending proposal if any
        pending_records = ledger.list_pending_records()
        if pending_records:
            pending_rec = pending_records[0]
            if pending_rec.proposal_metadata.get("proposal"):
                coord._last_proposal = ExperimentProposal.from_dict(pending_rec.proposal_metadata["proposal"])

        return coord
