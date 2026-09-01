from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.backend import OptimizerBackend
from src.science.actions import (
    ActionRecommendation,
    ActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
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
        for m in self.modalities:
            if m.name == norm_type:
                return m.measures_objective()
        return norm_type in ("PROPERTY", "CAPACITY_TEST")

    def _extract_observation_data(self, action_type: ActionType, revealed_data: Mapping[str, Any]) -> Any:
        """Extracts the primary numeric/vector payload from revealed measurement data."""
        if "xrd_embedding" in revealed_data:
            return np.asarray(revealed_data["xrd_embedding"], dtype=np.float64)
        if "sem_features" in revealed_data:
            return np.asarray(revealed_data["sem_features"], dtype=np.float64)
        if "embedding" in revealed_data:
            return np.asarray(revealed_data["embedding"], dtype=np.float64)

        # Check domain objectives
        for obj in self.objectives:
            if obj.name in revealed_data:
                return float(revealed_data[obj.name])

        for key in ("k0", "capacity", "measured_k0", "measured_capacity", "value", "target"):
            if key in revealed_data:
                val = revealed_data[key]
                return float(val) if isinstance(val, (int, float)) else val

        for k, v in revealed_data.items():
            if k not in ("two_theta", "intensity", "downsampled_two_theta", "downsampled_intensity", "normalized_intensity"):
                return v
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
            data_val = self._extract_observation_data(act.action_type, outcome.revealed_data)
            self.observations_by_modality[norm_act][act.candidate_id] = data_val
            self.revealed_outcomes[act.action_id] = outcome
            outcomes.append(outcome)

            # Record to ledger
            cand_comp = self._get_candidate_composition(act.candidate_id)
            record = ScientificExperimentRecord(
                experiment_id=act.action_id,
                candidate_id=act.candidate_id,
                dataset_name=self.domain_id,
                stage=ExperimentStage.COMPLETED,
                pre_experiment_features={col: float(cand_comp[i]) for i, col in enumerate(self.feature_cols)},
                proposal_metadata=act.to_dict(),
                provenance=dict(outcome.provenance),
            )
            delta = {"performance" if self._is_objective_action(act.action_type) else "characterization": outcome.revealed_data}
            if self.ledger is not None:
                try:
                    self.ledger.record_proposal(record)
                    self.ledger.append_transition(
                        experiment_id=act.action_id,
                        new_stage=ExperimentStage.COMPLETED,
                        event_type="INITIAL_BASELINE_EXPERIMENT",
                        delta_payload=delta,
                    )
                except Exception as e:
                    logger.debug("Ledger recording notice: %s", e)
            self.recorded_experiments.append(record)

        # Fit hypotheses on initial observations
        comp_by_id = {cid: self._get_candidate_composition(cid) for cid in self._candidate_features_map}
        self.ensemble.fit_all(
            composition_by_id=comp_by_id,
            observations_by_modality=self.observations_by_modality,
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

        # Extract discovery scores from optimizer backend if available
        disc_scores: dict[str, float] = {}
        if self.optimizer_backend is not None and len(self.objectives) > 0:
            primary_obj = self.objectives[0]
            # Identify candidates with observed objective measurements
            obj_obs: dict[str, float] = {}
            for m in self.modalities:
                if m.measures_objective(primary_obj.name):
                    obj_obs.update(self.observations_by_modality.get(m.name, {}))

            if len(obj_obs) >= 3:
                try:
                    cand_pool_features = self.candidate_pool_df[list(self.feature_cols)]
                    acq_scores = self.optimizer_backend.score_candidates(
                        candidate_features=cand_pool_features,
                        observed_candidate_ids=set(obj_obs.keys()),
                        observed_values=obj_obs,
                    )
                    cids = self.candidate_pool_df["candidate_id"].tolist()
                    disc_scores = {cid: float(acq_scores[i]) for i, cid in enumerate(cids) if i < len(acq_scores)}
                except Exception as e:
                    logger.debug("Optimizer scoring skipped: %s", e)

        # Observed characterization embeddings map
        char_embs_map: dict[str, np.ndarray] = {}
        for m in self.modalities:
            if m.observation_kind in ("characterization", "spectrum", "embedding", "image_features"):
                for cid, emb in self.observations_by_modality.get(m.name, {}).items():
                    char_embs_map[cid] = np.asarray(emb, dtype=np.float64)

        # Policy recommendation
        rec = self.policy.recommend_next_experiment(
            candidate_pool_df=self.candidate_pool_df,
            ensemble=self.ensemble,
            property_discovery_scores=disc_scores,
            observed_xrd_embeddings_map=char_embs_map,
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

        # Create proposal record in ledger
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
                logger.debug("Ledger record_proposal notice: %s", e)

        return rec

    def execute_recommendation(
        self,
        recommendation: ActionRecommendation | None = None,
    ) -> ExperimentOutcome:
        """Executes the recommended scientific action, updates evidence and ledger."""
        rec = recommendation or self._last_recommendation
        if rec is None:
            raise RuntimeError("No active experiment recommendation to execute. Call propose_next_experiment() first.")

        action = rec.action
        cand_id = action.candidate_id
        cand_comp = self._get_candidate_composition(cand_id)
        norm_act = normalize_action_type(action.action_type)

        # 1. Pre-predictions prior to outcome revelation
        pre_preds = self.ensemble.predict_all(
            candidate_id=cand_id,
            action_type=action.action_type,
            composition=cand_comp,
            observed_modalities=self.observations_by_modality,
        )

        # 2. Execute / reveal outcome via domain adapter
        outcome = self.domain.execute_or_reveal(action)
        if norm_act not in self.observations_by_modality:
            self.observations_by_modality[norm_act] = {}

        data_val = self._extract_observation_data(action.action_type, outcome.revealed_data)
        self.observations_by_modality[norm_act][cand_id] = data_val
        self.revealed_outcomes[action.action_id] = outcome

        # Observation array for evidence calculation
        obs_array = np.atleast_1d(np.asarray(data_val, dtype=np.float64))

        # 3. Update hypothesis log-evidence and compute posterior
        evidence_update = self.ensemble.record_observation_and_update(
            action_id=action.action_id,
            candidate_id=cand_id,
            action_type=action.action_type,
            observation=obs_array,
            pre_predictions=pre_preds,
        )

        # 4. Refit hypothesis models with updated observations
        comp_by_id = {cid: self._get_candidate_composition(cid) for cid in self._candidate_features_map}
        self.ensemble.fit_all(
            composition_by_id=comp_by_id,
            observations_by_modality=self.observations_by_modality,
        )

        # 5. Record execution and completion in ledger
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
                logger.debug("Ledger transition notice: %s", e)

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
                        if isinstance(v, (int, float)):
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
        }
