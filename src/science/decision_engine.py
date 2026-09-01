from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.backend import OptimizerBackend
from src.optimization.objective import OptimizationObjective
from src.science.actions import (
    ActionRecommendation,
    ActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
    HypothesisTrainingContext,
    MaterialDomainAdapter,
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)
from src.science.falsification.policy import (
    FalsificationFirstPolicy,
    FalsificationPolicyMode,
)
from src.science.hypothesis_models import (
    HypothesisEnsemble,
    PredictiveDistribution,
)
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage, ScientificExperimentRecord

logger = logging.getLogger(__name__)


def _to_optimizer_objective(obj: ObjectiveDefinition) -> OptimizationObjective:
    """Converts a science domain ObjectiveDefinition into an optimization OptimizationObjective."""
    return OptimizationObjective(
        target_name=obj.name,
        minimize=(obj.direction == ObjectiveDirection.MINIMIZE),
        units=obj.units,
        threshold=getattr(obj, "threshold", None),
        metadata=dict(obj.metadata),
    )


class ScientificDecisionEngine:
    """Universal, domain-agnostic scientific decision engine for materials discovery.

    Orchestrates the multimodal scientific cycle:
    1. Inspects visible candidate pool under strict offline information firewall.
    2. Obtains current valid experimental actions from the domain adapter.
    3. Fits/maintains competing scientific hypotheses across observed modalities.
    4. Evaluates Expected Hypothesis Information Gain (HIG) and discovery value.
    5. Recommends the optimal next experiment with contrastive counterfactuals.
    6. Pre-registers predictive distributions from all active hypotheses.
    7. Executes/reveals the chosen action via the domain connector/oracle.
    8. Updates hypothesis beliefs via Bayesian log-predictive evidence.
    9. Records complete provenance and lifecycle transitions in the ledger.
    """

    def __init__(
        self,
        domain: MaterialDomainAdapter,
        optimizer_backend: OptimizerBackend | None = None,
        ledger: ExperimentLedger | None = None,
        ensemble: HypothesisEnsemble | None = None,
        policy: FalsificationFirstPolicy | None = None,
        policy_mode: FalsificationPolicyMode | str = FalsificationPolicyMode.HYBRID,
        seed: int = 42,
        w_hig: float = 1.0,
        w_disc: float = 0.8,
        w_cost: float = 0.8,
        cost_exponent: float = 1.0,
        fast_mode: bool = False,
    ) -> None:
        self.domain = domain
        self.domain_id = domain.domain_id
        self.config: MaterialDomainConfig = domain.get_config()
        self.seed = seed
        self.fast_mode = fast_mode

        # Domain metadata
        self.candidate_pool_df: pd.DataFrame = domain.get_candidate_pool().copy()
        self.feature_cols: tuple[str, ...] = self.config.candidate_features
        self.modalities: Sequence[ModalityDefinition] = domain.get_modality_schema()
        self.objectives: Sequence[ObjectiveDefinition] = domain.get_objectives()

        # Map modality names to definitions
        self._modality_map: dict[str, ModalityDefinition] = {m.name: m for m in self.modalities}
        self._objective_map: dict[str, ObjectiveDefinition] = {o.name: o for o in self.objectives}

        # Positional candidate mapping (safe against arbitrary DataFrame index labels)
        cids = self.candidate_pool_df["candidate_id"].tolist()
        feat_matrix = self.candidate_pool_df[list(self.feature_cols)].to_numpy(dtype=np.float64)
        self._candidate_features_map: dict[str, np.ndarray] = {
            cid: feat_matrix[i] for i, cid in enumerate(cids)
        }

        # Hypotheses
        if ensemble is not None:
            self.ensemble = ensemble
        else:
            provider = domain.get_hypothesis_provider()
            if provider is not None:
                self.ensemble = HypothesisEnsemble(hypotheses=provider.build_hypotheses())
            else:
                self.ensemble = HypothesisEnsemble()

        # Policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = FalsificationFirstPolicy(
                mode=policy_mode,
                w_hig=w_hig,
                w_disc=w_disc,
                w_cost=w_cost,
                cost_exponent=cost_exponent,
            )

        # Optimization & Ledger
        self.optimizer_backend = optimizer_backend
        self.ledger = ledger or ExperimentLedger()
        self.last_optimizer_status: dict[str, Any] = {
            "used": False,
            "success": True,
            "reason": "Not yet evaluated",
            "num_scored": 0,
        }

        # State tracking
        self.step = 0
        self.observations_by_modality: dict[str, dict[str, Any]] = {
            m.name: {} for m in self.modalities
        }
        self.revealed_outcomes: dict[str, ExperimentOutcome] = {}
        self.recorded_experiments: list[ScientificExperimentRecord] = []
        self._last_recommendation: ActionRecommendation | None = None

    def _get_candidate_composition(self, candidate_id: str) -> np.ndarray:
        if candidate_id in self._candidate_features_map:
            return self._candidate_features_map[candidate_id]
        return np.asarray(list(self.domain.get_candidate_features(candidate_id).values()), dtype=np.float64)

    def _is_objective_action(self, action_type: ActionType) -> bool:
        norm_type = normalize_action_type(action_type)
        if norm_type in self._modality_map:
            return self._modality_map[norm_type].measures_objective()
        for obj in self.objectives:
            if obj.name == norm_type:
                return True
        return False

    def _extract_observation_data(
        self,
        action_type: ActionType,
        revealed_data: Mapping[str, Any],
        outcome: ExperimentOutcome | None = None,
    ) -> Any:
        """Extracts the primary numeric/vector payload from revealed measurement data driven by domain metadata."""
        if outcome is not None and getattr(outcome, "canonical_observation", None) is not None:
            return outcome.canonical_observation

        norm_type = normalize_action_type(action_type)
        m_def = self._modality_map.get(norm_type)

        if m_def is not None:
            # 1. Direct observation_key specified by modality definition
            if m_def.observation_key and m_def.observation_key in revealed_data:
                val = revealed_data[m_def.observation_key]
                return np.asarray(val, dtype=np.float64) if isinstance(val, (list, tuple, np.ndarray)) else val

            # 2. Key in metadata dictionary
            obs_key_meta = m_def.metadata.get("observation_key")
            if obs_key_meta and obs_key_meta in revealed_data:
                val = revealed_data[obs_key_meta]
                return np.asarray(val, dtype=np.float64) if isinstance(val, (list, tuple, np.ndarray)) else val

            # 3. Objective names specified by modality definition
            for obj_name in m_def.objective_names:
                if obj_name in revealed_data:
                    val = revealed_data[obj_name]
                    return float(val) if isinstance(val, (int, float, np.number)) else val

        # 4. Domain objective targets matching keys
        for obj in self.objectives:
            if obj.name in revealed_data:
                val = revealed_data[obj.name]
                return float(val) if isinstance(val, (int, float, np.number)) else val

        # 5. Metadata fallback: check generic payload keys if non-empty
        for k, v in revealed_data.items():
            if isinstance(v, (int, float, np.number)):
                return float(v)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) <= 64:
                return np.asarray(v, dtype=np.float64)

        return next(iter(revealed_data.values())) if revealed_data else 0.0

    def initialize(
        self,
        initial_actions: Sequence[ScientificAction],
    ) -> list[ExperimentOutcome]:
        """Executes a curated set of initial actions to establish baseline observations."""
        outcomes: list[ExperimentOutcome] = []
        for act in initial_actions:
            outcome = self.domain.execute_or_reveal(act)
            norm_act = normalize_action_type(act.action_type)
            if norm_act not in self.observations_by_modality:
                self.observations_by_modality[norm_act] = {}

            # Store revealed measurement
            data_val = self._extract_observation_data(act.action_type, outcome.revealed_data, outcome)
            self.observations_by_modality[norm_act][act.candidate_id] = data_val
            self.revealed_outcomes[act.action_id] = outcome
            outcomes.append(outcome)

            # Update representation if adapter supports it
            if hasattr(self.domain, "update_representation_after_evidence"):
                self.domain.update_representation_after_evidence(act.action_type, act.candidate_id, outcome.revealed_data)

            # Record baseline imported evidence in ledger with actual scientific observation payload
            cand_comp = self._get_candidate_composition(act.candidate_id)
            pre_features = {col: float(cand_comp[i]) for i, col in enumerate(self.feature_cols)}
            perf: dict[str, float] = {}
            char_data: dict[str, Any] = {}
            is_obj = self._is_objective_action(act.action_type)
            if is_obj:
                if len(self.objectives) > 0:
                    val_num = float(data_val) if isinstance(data_val, (int, float, np.number)) else 0.0
                    perf[self.objectives[0].name] = val_num
            else:
                char_data[norm_act] = data_val.tolist() if isinstance(data_val, np.ndarray) else data_val

            record = ScientificExperimentRecord(
                experiment_id=act.action_id,
                candidate_id=act.candidate_id,
                dataset_name=self.domain_id,
                stage=ExperimentStage.COMPLETED,
                pre_experiment_features=pre_features,
                characterization=char_data,
                performance=perf,
                proposal_metadata={
                    **act.to_dict(),
                    "observed_value": float(data_val) if isinstance(data_val, (int, float, np.number)) else None,
                    "canonical_observation": data_val.tolist() if isinstance(data_val, np.ndarray) else data_val,
                },
                provenance=dict(outcome.provenance),
            )
            if self.ledger is not None:
                try:
                    if hasattr(self.ledger, "record_baseline_evidence"):
                        self.ledger.record_baseline_evidence(record)
                    else:
                        self.ledger.record_proposal(record)
                except Exception as e:
                    logger.warning("Ledger baseline recording notice: %s", e)
            self.recorded_experiments.append(record)

        # Synchronize multi-modal representation state from domain adapter if supported
        if hasattr(self.domain, "get_observations_by_modality"):
            domain_obs = self.domain.get_observations_by_modality()
            for m_name, obs_dict in domain_obs.items():
                if m_name not in self.observations_by_modality:
                    self.observations_by_modality[m_name] = {}
                self.observations_by_modality[m_name].update(obs_dict)

        # Fit hypotheses on initial observations
        comp_by_id = {cid: self._get_candidate_composition(cid) for cid in self._candidate_features_map}
        self.ensemble.fit_all(
            composition_by_id=comp_by_id,
            observations_by_modality=self.observations_by_modality,
            modality_definitions=self.modalities,
            objective_definitions=self.objectives,
        )
        return outcomes

    def propose_next_experiment(
        self,
        fast_mode: bool | None = None,
    ) -> ActionRecommendation:
        """Evaluates all valid actions and recommends the optimal next scientific experiment."""
        use_fast = self.fast_mode if fast_mode is None else fast_mode
        valid_actions = self.domain.list_valid_actions()

        if not valid_actions:
            raise RuntimeError(f"All valid experimental actions have been exhausted in domain '{self.domain_id}'.")

        # Synchronize domain representation state before policy evaluation
        if hasattr(self.domain, "get_observations_by_modality"):
            domain_obs = self.domain.get_observations_by_modality()
            for m_name, obs_dict in domain_obs.items():
                if m_name not in self.observations_by_modality:
                    self.observations_by_modality[m_name] = {}
                self.observations_by_modality[m_name].update(obs_dict)

        # Extract discovery scores from optimizer backend if available
        disc_scores: dict[str, float] = {}
        if self.optimizer_backend is not None and len(self.objectives) > 0:
            primary_obj = self.objectives[0]
            obj_obs: dict[str, float] = {}
            for m in self.modalities:
                if m.measures_objective(primary_obj.name):
                    for cid, val in self.observations_by_modality.get(m.name, {}).items():
                        if isinstance(val, (int, float, np.number)):
                            obj_obs[cid] = float(val)

            if len(obj_obs) >= 3:
                obs_rows = []
                for cid, val in obj_obs.items():
                    feat_vec = self._get_candidate_composition(cid)
                    row = {"candidate_id": cid, primary_obj.name: float(val)}
                    for i, col in enumerate(self.feature_cols):
                        row[col] = float(feat_vec[i])
                    obs_rows.append(row)
                obs_df = pd.DataFrame(obs_rows)

                try:
                    opt_obj = _to_optimizer_objective(primary_obj)
                    disc_scores = self.optimizer_backend.score_candidates(
                        observations=obs_df,
                        candidate_pool=self.candidate_pool_df,
                        objective=opt_obj,
                        feature_columns=list(self.feature_cols),
                        candidate_id_column="candidate_id",
                        seed=self.seed + self.step,
                    )
                    self.last_optimizer_status = {
                        "used": True,
                        "success": True,
                        "reason": "OK",
                        "num_scored": len(disc_scores),
                    }
                except Exception as e:
                    logger.warning("Optimizer backend acquisition evaluation notice in domain '%s': %s", self.domain_id, e)
                    self.last_optimizer_status = {
                        "used": True,
                        "success": False,
                        "reason": str(e),
                        "num_scored": 0,
                    }
                    disc_scores = {}
            else:
                self.last_optimizer_status = {
                    "used": True,
                    "success": False,
                    "reason": f"Insufficient objective observations ({len(obj_obs)} < 3)",
                    "num_scored": 0,
                }

        # Policy recommendation
        rec = self.policy.recommend_next_experiment(
            candidate_pool_df=self.candidate_pool_df,
            ensemble=self.ensemble,
            property_discovery_scores=disc_scores,
            observed_modalities_map=self.observations_by_modality,
            fast_mode=use_fast,
            seed=self.seed + self.step,
            step=self.step,
            valid_actions=valid_actions,
            feature_cols=self.feature_cols,
            modality_definitions=self.modalities,
            objective_definitions=self.objectives,
            domain_id=self.domain_id,
        )
        self._last_recommendation = rec

        # Create proposal record in ledger with PROPOSED stage
        rec_action = rec.action
        cand_comp = self._get_candidate_composition(rec_action.candidate_id)
        proposal_rec = ScientificExperimentRecord(
            experiment_id=rec_action.action_id,
            candidate_id=rec_action.candidate_id,
            dataset_name=self.domain_id,
            stage=ExperimentStage.PROPOSED,
            pre_experiment_features={col: float(cand_comp[i]) for i, col in enumerate(self.feature_cols)},
            proposal_metadata=rec_action.to_dict(),
            provenance={"domain_id": self.domain_id, "engine": "ScientificDecisionEngine"},
        )
        if self.ledger is not None:
            try:
                self.ledger.record_proposal(proposal_rec)
            except Exception as e:
                logger.warning("Ledger record_proposal notice: %s", e)

        return rec

    def execute_recommendation(
        self,
        recommendation: ActionRecommendation | None = None,
    ) -> ExperimentOutcome:
        """Executes the recommended scientific action, updates evidence and ledger.

        INVARIANT ORDERING:
        1. Pre-experiment: capture frozen representation basis snapshot R_N.
        2. Generate pre-predictions under R_N.
        3. Execute/reveal raw measurement.
        4. Transform realized measurement using frozen basis R_N.
        5. Update Bayesian log-evidence and compute posterior under R_N (fail-closed if basis mismatch).
        6. ONLY AFTER evidence update: refit representation basis to R_N+1 and recompute all historical embeddings.
        7. Refit hypothesis predictive models under R_N+1.
        """
        rec = recommendation or self._last_recommendation
        if rec is None:
            raise RuntimeError("No active experiment recommendation to execute. Call propose_next_experiment() first.")

        action = rec.action
        cand_id = action.candidate_id
        cand_comp = self._get_candidate_composition(cand_id)
        norm_act = normalize_action_type(action.action_type)

        # 1. Pre-experiment representation basis snapshot (R_N)
        snapshot = None
        if hasattr(self.domain, "get_representation_snapshot"):
            snapshot = self.domain.get_representation_snapshot(norm_act)

        # 2. Pre-predictions prior to outcome revelation (under frozen basis R_N)
        raw_pre_preds = self.ensemble.predict_all(
            candidate_id=cand_id,
            action_type=action.action_type,
            composition=cand_comp,
            observed_modalities=self.observations_by_modality,
        )
        pre_preds: dict[str, PredictiveDistribution] = {}
        for hid, pred in raw_pre_preds.items():
            if snapshot is not None:
                pre_preds[hid] = PredictiveDistribution(
                    hypothesis_id=pred.hypothesis_id,
                    candidate_id=pred.candidate_id,
                    action_type=pred.action_type,
                    mean=pred.mean,
                    variance=pred.variance,
                    metadata=dict(pred.metadata),
                    representation_fingerprint=snapshot.fingerprint,
                    representation_version=snapshot.version,
                    representation_id=snapshot.representation_id,
                )
            else:
                pre_preds[hid] = pred

        # 3. Execute / reveal outcome via domain adapter
        outcome = self.domain.execute_or_reveal(action)
        if norm_act not in self.observations_by_modality:
            self.observations_by_modality[norm_act] = {}

        # 4. Transform realized measurement using frozen snapshot R_N
        if snapshot is not None and hasattr(self.domain, "transform_with_snapshot"):
            data_val = self.domain.transform_with_snapshot(action.action_type, outcome.revealed_data, snapshot)
        else:
            data_val = self._extract_observation_data(action.action_type, outcome.revealed_data, outcome)

        self.revealed_outcomes[action.action_id] = outcome
        obs_array = np.atleast_1d(np.asarray(data_val, dtype=np.float64))

        # 5. Update Bayesian likelihood & posterior strictly under frozen basis R_N
        evidence_update = self.ensemble.record_observation_and_update(
            action_id=action.action_id,
            candidate_id=cand_id,
            action_type=action.action_type,
            observation=obs_array,
            pre_predictions=pre_preds,
            observation_representation_fingerprint=snapshot.fingerprint if snapshot is not None else None,
        )

        # 6. ONLY AFTER evidence update: update representation basis to R_N+1 and re-embed all revealed data
        if hasattr(self.domain, "update_representation_after_evidence"):
            self.domain.update_representation_after_evidence(action.action_type, cand_id, outcome.revealed_data)

        if hasattr(self.domain, "get_observations_by_modality"):
            domain_obs = self.domain.get_observations_by_modality()
            for m_name, obs_dict in domain_obs.items():
                if m_name not in self.observations_by_modality:
                    self.observations_by_modality[m_name] = {}
                self.observations_by_modality[m_name].update(obs_dict)
        else:
            self.observations_by_modality[norm_act][cand_id] = data_val

        # 7. Refit hypothesis predictive models on the updated observations
        comp_by_id = {cid: self._get_candidate_composition(cid) for cid in self._candidate_features_map}
        self.ensemble.fit_all(
            composition_by_id=comp_by_id,
            observations_by_modality=self.observations_by_modality,
            modality_definitions=self.modalities,
            objective_definitions=self.objectives,
        )

        # 8. Record execution and completion transitions in ledger
        delta = {"performance" if self._is_objective_action(action.action_type) else "characterization": outcome.revealed_data}
        if self.ledger is not None:
            try:
                self.ledger.append_transition(
                    experiment_id=action.action_id,
                    new_stage=ExperimentStage.EXECUTED,
                    event_type="EXPERIMENT_EXECUTED",
                    delta_payload={},
                )
                self.ledger.append_transition(
                    experiment_id=action.action_id,
                    new_stage=ExperimentStage.COMPLETED,
                    event_type="EXPERIMENT_COMPLETED",
                    delta_payload=delta,
                )
            except Exception as e:
                logger.warning("Ledger transition notice: %s", e)

        record = ScientificExperimentRecord(
            experiment_id=action.action_id,
            candidate_id=action.candidate_id,
            dataset_name=self.domain_id,
            stage=ExperimentStage.COMPLETED,
            pre_experiment_features={col: float(cand_comp[i]) for i, col in enumerate(self.feature_cols)},
            proposal_metadata=action.to_dict(),
            provenance=dict(outcome.provenance),
        )
        self.recorded_experiments.append(record)

        self.step += 1
        self._last_recommendation = None
        return outcome

    def get_state(self) -> dict[str, Any]:
        """Returns a structured, demo-ready snapshot of the current decision engine state."""
        beliefs = self.ensemble.get_beliefs()
        entropy = self.ensemble.get_entropy()
        obs_counts = {m.name: len(self.observations_by_modality.get(m.name, {})) for m in self.modalities}

        primary_obj = self.objectives[0] if self.objectives else None
        best_obj_val = None
        if primary_obj is not None:
            all_obj_obs: list[float] = []
            for m in self.modalities:
                if m.measures_objective(primary_obj.name):
                    for v in self.observations_by_modality.get(m.name, {}).values():
                        if isinstance(v, (int, float, np.number)):
                            all_obj_obs.append(float(v))
            if all_obj_obs:
                best_obj_val = max(all_obj_obs) if primary_obj.direction == ObjectiveDirection.MAXIMIZE else min(all_obj_obs)

        return {
            "domain_id": self.domain_id,
            "step": self.step,
            "current_beliefs": beliefs,
            "current_entropy": entropy,
            "observations_count": obs_counts,
            "best_observed_objective": best_obj_val,
            "primary_objective": primary_obj.to_dict() if primary_obj else None,
            "active_hypotheses": list(self.ensemble.hypotheses.keys()),
            "evidence_history_length": len(self.ensemble.evidence_history),
            "ledger_records_count": len(self.recorded_experiments),
            "last_optimizer_status": dict(self.last_optimizer_status),
        }
