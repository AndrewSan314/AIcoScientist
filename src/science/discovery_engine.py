from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.datasets.auirh_actions import AuIrRhMultimodalOracle
    from src.optimization.backend import OptimizerBackend

from src.science.actions import (
    ActionRecommendation,
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
)
from src.science.agents import AgentPerspective, MultiAgentPresentationLayer
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode
from src.science.hypotheses import HypothesisEngine, get_default_hypotheses
from src.science.hypothesis_models import HypothesisEnsemble, PredictiveDistribution
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.experiment_policy import NextBestExperimentPolicy
from src.science.scientific_models import PropertySurrogateModel, StructureSurrogateModel
from src.science.xrd_representation import XRDRepresentationExtractor

logger = logging.getLogger(__name__)


class AutonomousDiscoveryEngine:
    """Unified Falsification-First Multimodal Discovery Engine for the AI Scientist Discovery Console.

    Orchestrates the complete closed-loop scientific decision workflow:
    1. Observable candidate landscape & strict offline-oracle firewall.
    2. Dynamic XRD representation & ephemeral surrogate projections.
    3. Formal competing scientific hypotheses (H1, H2, H3) & predictive distributions.
    4. Expected Hypothesis Information Gain (HIG) Monte Carlo estimation.
    5. Adaptive multi-mode experiment selection policy with contrastive counterfactuals.
    6. Multi-agent role-based presentation layer.
    7. Authoritative tamper-evident ledger event logging with pre-registered predictions.
    """

    def __init__(
        self,
        oracle: AuIrRhMultimodalOracle | None = None,
        optimizer_backend: OptimizerBackend | None = None,
        db_path: Path | str = ":memory:",
        seed: int = 42,
        cost_xrd: float = 1.0,
        cost_property: float = 5.0,
        w_info: float = 1.0,
        w_disc: float = 0.8,
        w_cost: float = 0.8,
        policy_mode: FalsificationPolicyMode | str = FalsificationPolicyMode.HYBRID,
    ) -> None:
        self.seed = seed
        self._db_path = str(db_path)
        if oracle is None:
            from src.datasets.auirh_actions import AuIrRhMultimodalOracle

            self.oracle = AuIrRhMultimodalOracle()
        else:
            self.oracle = oracle

        if optimizer_backend is None:
            from src.optimization.botorch_backend import BoTorchBackend

            self.optimizer_backend: OptimizerBackend = BoTorchBackend(default_strategy="expected_improvement")
        else:
            self.optimizer_backend = optimizer_backend

        self.ledger = ExperimentLedger(db_path=self._db_path)
        self.xrd_extractor = XRDRepresentationExtractor()
        self.structure_model = StructureSurrogateModel(random_state=seed)
        self.property_model = PropertySurrogateModel(random_state=seed)
        self.hypothesis_engine = HypothesisEngine(get_default_hypotheses())
        self.ensemble = HypothesisEnsemble()

        self.policy = NextBestExperimentPolicy(
            cost_xrd=cost_xrd,
            cost_property=cost_property,
            w_info=w_info,
            w_disc=w_disc,
            w_cost=w_cost,
        )
        self.falsification_policy = FalsificationFirstPolicy(
            mode=policy_mode,
            cost_xrd=cost_xrd,
            cost_property=cost_property,
            w_hig=w_info,
            w_disc=w_disc,
            w_cost=w_cost,
        )
        self.agent_layer = MultiAgentPresentationLayer()

        self.current_step = 0
        self.total_budget_spent = 0.0
        self.timeline: list[dict[str, Any]] = []
        self._last_recommendation: ActionRecommendation | None = None
        self._last_perspectives: list[AgentPerspective] = []

        # Pre-measurement predictions for event-driven residual computation and pre-registration
        self._pre_pred_emb: np.ndarray | None = None
        self._pre_pred_mean: float | None = None
        self._pre_pred_std: float | None = None
        self._pre_predictions: dict[str, PredictiveDistribution] = {}

    def initialize_curated_scenario(
        self,
        n_init_prop: int = 6,
        n_init_xrd: int = 4,
        seed: int | None = None,
    ) -> None:
        """Sets up a deterministic, reproducible initial scientific campaign state.

        Initial seed measurements provide prior context and fit models without generating
        fabricated hypothesis evidence events or altering neutral baseline belief weights.
        """
        init_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(init_seed)

        self.reset()
        cand_df = self.oracle.get_candidate_pool()
        all_cids = cand_df["candidate_id"].tolist()

        # Randomly choose initial candidates using fixed seed
        shuffled_cids = list(all_cids)
        rng.shuffle(shuffled_cids)

        init_prop_cids = shuffled_cids[:n_init_prop]
        init_xrd_cids = shuffled_cids[n_init_prop : n_init_prop + n_init_xrd]

        # 1. Execute initial XRD actions (record to ledger and timeline only)
        for cid in init_xrd_cids:
            act = ScientificAction(
                action_id=f"init_xrd_{cid}",
                candidate_id=cid,
                action_type=ExperimentActionType.XRD,
                estimated_cost=self.policy.cost_xrd,
                requested_at_step=0,
            )
            outcome = self.oracle.execute(act)
            self._record_to_ledger(act, outcome)
            self.total_budget_spent += self.policy.cost_xrd

            self.timeline.append(
                {
                    "step": len(self.timeline) + 1,
                    "action_id": act.action_id,
                    "candidate_id": cid,
                    "action_type": "XRD",
                    "cost": self.policy.cost_xrd,
                    "hypothesis_tested": "Initial Baseline Context",
                    "revealed_summary": f"Peak 2theta: {outcome.revealed_data.get('peak_two_theta', 0.0):.1f}°",
                    "status": "COMPLETED",
                }
            )

        # 2. Execute initial Property actions (record to ledger and timeline only)
        for cid in init_prop_cids:
            act = ScientificAction(
                action_id=f"init_prop_{cid}",
                candidate_id=cid,
                action_type=ExperimentActionType.PROPERTY,
                estimated_cost=self.policy.cost_property,
                requested_at_step=0,
            )
            outcome = self.oracle.execute(act)
            self._record_to_ledger(act, outcome)
            self.total_budget_spent += self.policy.cost_property

            self.timeline.append(
                {
                    "step": len(self.timeline) + 1,
                    "action_id": act.action_id,
                    "candidate_id": cid,
                    "action_type": "PROPERTY",
                    "cost": self.policy.cost_property,
                    "hypothesis_tested": "Initial Baseline Context",
                    "revealed_summary": f"k0: {outcome.revealed_data.get('k0', 0.0):.5f} cm/s",
                    "status": "COMPLETED",
                }
            )

        self._refit_models()

    def _refit_models(self) -> float:
        """Refits surrogate models on revealed data and returns current structural predictive advantage."""
        cand_df = self.oracle.get_candidate_pool()

        # Build candidate_id keyed composition dictionary from full candidate pool
        comp_by_id: dict[str, np.ndarray] = {
            str(row["candidate_id"]): row[["Au", "Ir", "Rh"]].to_numpy(dtype=np.float64)
            for _, row in cand_df.iterrows()
        }

        # 1. Fit XRD Representation Extractor on revealed spectra
        revealed_xrd_map = self.oracle.get_revealed_xrd()
        revealed_spectra = [
            out.revealed_data["normalized_intensity"] for out in revealed_xrd_map.values()
        ]
        self.xrd_extractor.fit(revealed_spectra)

        # 2. Extract embeddings for candidates with revealed XRD
        observed_xrd_ids = set(revealed_xrd_map.keys())
        xrd_embedding_by_id: dict[str, np.ndarray] = {
            cid: self.xrd_extractor.transform(out.revealed_data["normalized_intensity"])
            for cid, out in revealed_xrd_map.items()
        }

        ordered_xrd_cids = sorted(observed_xrd_ids)
        if ordered_xrd_cids:
            xrd_comps = np.array([comp_by_id[cid] for cid in ordered_xrd_cids], dtype=np.float64)
            xrd_embs = np.array([xrd_embedding_by_id[cid] for cid in ordered_xrd_cids], dtype=np.float64)
            self.structure_model.fit(xrd_comps, xrd_embs)
        else:
            xrd_comps = np.empty((0, 3), dtype=np.float64)
            xrd_embs = np.empty((0, 8), dtype=np.float64)
            self.structure_model.fit(xrd_comps, xrd_embs)

        # 3. Fit Property Model on revealed properties
        revealed_prop_map = self.oracle.get_revealed_properties()
        observed_property_ids = set(revealed_prop_map.keys())
        property_by_id: dict[str, float] = {
            cid: float(out.revealed_data["k0"])
            for cid, out in revealed_prop_map.items()
        }

        ordered_prop_cids = sorted(observed_property_ids)
        if ordered_prop_cids:
            prop_comps = np.array([comp_by_id[cid] for cid in ordered_prop_cids], dtype=np.float64)
            prop_targets = np.array([property_by_id[cid] for cid in ordered_prop_cids], dtype=np.float64)

            # If candidates have both XRD and property, evaluate structural predictive advantage via LOO-CV
            joint_cids = sorted([cid for cid in ordered_prop_cids if cid in xrd_embedding_by_id])
            if len(joint_cids) >= 3:
                joint_comps = np.array([comp_by_id[cid] for cid in joint_cids], dtype=np.float64)
                joint_targets = np.array([property_by_id[cid] for cid in joint_cids], dtype=np.float64)
                joint_embs = np.array([xrd_embedding_by_id[cid] for cid in joint_cids], dtype=np.float64)
                self.property_model.fit(prop_comps, prop_targets, embeddings=None)
                eval_dict = self.property_model.evaluate_structure_predictive_advantage(
                    compositions=joint_comps,
                    targets=joint_targets,
                    embeddings=joint_embs,
                )
                struct_adv = float(eval_dict["structure_advantage_ratio"])
            else:
                self.property_model.fit(prop_comps, prop_targets)
                struct_adv = 0.0
        else:
            prop_comps = np.empty((0, 3), dtype=np.float64)
            prop_targets = np.empty((0,), dtype=np.float64)
            self.property_model.fit(prop_comps, prop_targets)
            struct_adv = 0.0

        # 4. Fit formal hypothesis models in ensemble using candidate_id as the ONLY join key
        self.ensemble.fit_all(
            composition_by_id=comp_by_id,
            property_by_id=property_by_id,
            xrd_embedding_by_id=xrd_embedding_by_id,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_property_ids,
        )

        # Synchronize legacy hypothesis engine
        self.hypothesis_engine.recalculate_current_scores(structure_advantage_ratio=struct_adv)
        return struct_adv

    def _record_to_ledger(self, action: ScientificAction, outcome: ExperimentOutcome) -> None:
        """Appends experiment record to authoritative ledger via valid lifecycle transitions."""
        cand_df = self.oracle.get_candidate_pool()
        row = cand_df[cand_df["candidate_id"] == action.candidate_id].iloc[0]

        proposal_rec = ScientificExperimentRecord(
            experiment_id=action.action_id,
            candidate_id=action.candidate_id,
            dataset_name="Au-Ir-Rh_Multimodal_Demo",
            stage=ExperimentStage.PROPOSED,
            pre_experiment_features={"Au": float(row["Au"]), "Ir": float(row["Ir"]), "Rh": float(row["Rh"])},
            candidate_variables={"Library": str(row["Library"]), "Area": int(row["Area"])},
            proposal_metadata=action.to_dict(),
            provenance=outcome.provenance,
        )
        self.ledger.record_proposal(proposal_rec)

        self.ledger.append_transition(
            experiment_id=action.action_id,
            new_stage=ExperimentStage.EXECUTED,
            event_type="EXPERIMENT_EXECUTED",
            delta_payload={},
        )

        delta = (
            {"characterization": outcome.revealed_data}
            if action.action_type == ExperimentActionType.XRD
            else {"performance": outcome.revealed_data}
        )
        self.ledger.append_transition(
            experiment_id=action.action_id,
            new_stage=ExperimentStage.COMPLETED,
            event_type="EXPERIMENT_COMPLETED",
            delta_payload=delta,
        )

    def propose_next_experiment(
        self,
        use_falsification_first: bool = False,
        fast_mode: bool = True,
    ) -> tuple[ActionRecommendation, list[AgentPerspective]]:
        """Evaluates policy and multi-agent layer to recommend the next scientific experiment."""
        cand_df = self.oracle.get_candidate_pool()
        observed_xrd = set(self.oracle.get_revealed_xrd_ids())
        observed_prop = set(self.oracle.get_revealed_property_ids())
        revealed_props = self.oracle.get_revealed_properties()
        revealed_xrds = self.oracle.get_revealed_xrd()

        # Build observations DataFrame for production optimizer backend
        if revealed_props:
            obs_rows = []
            for cid, outcome in revealed_props.items():
                match = cand_df[cand_df["candidate_id"] == cid]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    row["k0"] = float(outcome.revealed_data["k0"])
                    obs_rows.append(row)
            obs_df = pd.DataFrame(obs_rows)
        else:
            obs_df = pd.DataFrame(columns=["candidate_id", "Au", "Ir", "Rh", "k0"])

        # Score candidate property discovery potential via production optimizer backend
        try:
            property_discovery_scores = self.optimizer_backend.score_candidates(
                observations=obs_df,
                candidate_pool=cand_df[["candidate_id", "Au", "Ir", "Rh"]],
                objective="k0",
                strategy="expected_improvement",
                seed=self.seed,
            )
        except Exception as exc:
            logger.warning(f"Optimizer scoring fallback: {exc}")
            property_discovery_scores = {}

        if use_falsification_first:
            xrd_embs_map = {
                cid: self.xrd_extractor.transform(revealed_xrds[cid].revealed_data["normalized_intensity"])
                for cid in revealed_xrds
            }
            recommendation = self.falsification_policy.recommend_next_experiment(
                candidate_pool_df=cand_df,
                observed_xrd_ids=observed_xrd,
                observed_property_ids=observed_prop,
                ensemble=self.ensemble,
                property_discovery_scores=property_discovery_scores,
                observed_xrd_embeddings_map=xrd_embs_map,
                fast_mode=fast_mode,
                seed=self.seed + self.current_step,
                step=self.current_step,
            )
        else:
            recommendation = self.policy.recommend_next_experiment(
                candidate_pool_df=cand_df,
                observed_xrd_ids=observed_xrd,
                observed_property_ids=observed_prop,
                structure_model=self.structure_model,
                property_model=self.property_model,
                hypothesis_engine=self.hypothesis_engine,
                property_discovery_scores=property_discovery_scores,
                step=self.current_step,
            )

        perspectives = self.agent_layer.generate_perspectives(
            recommendation=recommendation,
            hypothesis_engine=self.hypothesis_engine,
            num_xrd_revealed=len(observed_xrd),
            num_property_revealed=len(observed_prop),
            total_candidates=len(cand_df),
        )

        # Pre-register predictions across all hypotheses before execution
        rec_cid = recommendation.action.candidate_id
        cand_row = cand_df[cand_df["candidate_id"] == rec_cid].iloc[0]
        cand_comp = np.array([[cand_row["Au"], cand_row["Ir"], cand_row["Rh"]]], dtype=np.float64)

        if self.structure_model.is_fitted:
            pred_emb, _ = self.structure_model.predict(cand_comp)
            self._pre_pred_emb = pred_emb[0]
        else:
            self._pre_pred_emb = None

        if self.property_model.is_fitted:
            pred_mean, pred_std = self.property_model.predict(cand_comp)
            self._pre_pred_mean = float(pred_mean[0])
            self._pre_pred_std = float(pred_std[0])
        else:
            self._pre_pred_mean = None
            self._pre_pred_std = None

        # Pre-register predictions in ensemble
        self._pre_predictions = self.ensemble.predict_all(
            candidate_id=rec_cid,
            action_type=recommendation.action.action_type,
            composition=cand_comp[0],
        )

        self._last_recommendation = recommendation
        self._last_perspectives = perspectives
        return recommendation, perspectives

    def execute_experiment(self, action: ScientificAction | None = None) -> dict[str, Any]:
        """Executes the recommended (or specified) scientific action via the oracle.

        Exact Sequential Flow:
        1. Capture before-state beliefs.
        2. Execute action on oracle & record to ledger.
        3. Derive empirical event metrics (residual/novelty) using pre-measurement predictions.
        4. Refit models on newly revealed observation.
        5. Compute current LOO-CV structural predictive advantage.
        6. Record EXACTLY ONE EvidenceEvent in HypothesisEngine & update HypothesisEnsemble.
        7. Recompute beliefs once from event history.
        """
        act = action or (self._last_recommendation.action if self._last_recommendation else None)
        if act is None:
            raise ValueError("No action provided or recommended to execute.")

        # Capture pre-measurement prediction if not cached
        cand_df = self.oracle.get_candidate_pool()
        cand_row = cand_df[cand_df["candidate_id"] == act.candidate_id].iloc[0]
        cand_comp = np.array([[cand_row["Au"], cand_row["Ir"], cand_row["Rh"]]], dtype=np.float64)

        if (
            (act.action_type == ExperimentActionType.XRD and self._pre_pred_emb is None)
            or (act.action_type == ExperimentActionType.PROPERTY and self._pre_pred_mean is None)
        ):
            if self.structure_model.is_fitted:
                p_emb, _ = self.structure_model.predict(cand_comp)
                self._pre_pred_emb = p_emb[0]
            if self.property_model.is_fitted:
                p_m, p_s = self.property_model.predict(cand_comp)
                self._pre_pred_mean = float(p_m[0])
                self._pre_pred_std = float(p_s[0])

        if not self._pre_predictions:
            self._pre_predictions = self.ensemble.predict_all(
                candidate_id=act.candidate_id,
                action_type=act.action_type,
                composition=cand_comp[0],
            )

        before_beliefs = {hid: h.belief_score for hid, h in self.hypothesis_engine.hypotheses.items()}

        outcome = self.oracle.execute(act)
        self._record_to_ledger(act, outcome)

        self.current_step += 1
        self.total_budget_spent += act.estimated_cost

        # 1. Compute empirical event metrics (strictly None if no pre-measurement model existed)
        if act.action_type == ExperimentActionType.XRD:
            norm_spec = outcome.revealed_data["normalized_intensity"]
            actual_emb = self.xrd_extractor.transform(norm_spec)

            if self._pre_pred_emb is not None:
                raw_res = float(np.linalg.norm(actual_emb - self._pre_pred_emb))
                struct_res = min(1.0, raw_res / np.sqrt(len(actual_emb) + 1e-12))
            else:
                struct_res = None

            revealed_xrd_map = self.oracle.get_revealed_xrd()
            if len(revealed_xrd_map) > 1:
                other_embs = [
                    self.xrd_extractor.transform(out.revealed_data["normalized_intensity"])
                    for cid, out in revealed_xrd_map.items()
                    if cid != act.candidate_id
                ]
                dists = [float(np.linalg.norm(actual_emb - oe)) for oe in other_embs]
                struct_nov = min(1.0, float(min(dists)) / np.sqrt(len(actual_emb) + 1e-12))
            else:
                struct_nov = None

            prop_res = None
            obs_for_ensemble = actual_emb

        elif act.action_type == ExperimentActionType.PROPERTY:
            y_obs = float(outcome.revealed_data["k0"])
            if self._pre_pred_mean is not None:
                denom = max(self._pre_pred_std or 1e-3, 1e-4)
                prop_res = min(5.0, abs(y_obs - self._pre_pred_mean) / denom)
            else:
                prop_res = None

            struct_res = None
            struct_nov = None
            obs_for_ensemble = y_obs

        # 2. Update sequential predictive evidence in ensemble
        ensemble_update = self.ensemble.record_observation_and_update(
            action_id=act.action_id,
            candidate_id=act.candidate_id,
            action_type=act.action_type,
            observation=obs_for_ensemble,
            pre_predictions=self._pre_predictions,
        )

        # 3. Refit models and compute current LOO-CV structural advantage
        struct_adv = self._refit_models()

        # 4. Record EXACTLY ONE evidence event in legacy HypothesisEngine
        self.hypothesis_engine.record_evidence_event(
            event_id=act.action_id,
            action_type=act.action_type.value,
            candidate_id=act.candidate_id,
            structure_residual=struct_res,
            structure_novelty=struct_nov,
            property_residual=prop_res,
            structure_advantage_ratio=struct_adv,
        )

        # Clear pre-prediction cache for next step
        self._pre_pred_emb = None
        self._pre_pred_mean = None
        self._pre_pred_std = None
        self._pre_predictions.clear()

        after_beliefs = {hid: h.belief_score for hid, h in self.hypothesis_engine.hypotheses.items()}
        belief_deltas = {hid: after_beliefs[hid] - before_beliefs[hid] for hid in after_beliefs}

        summary = {
            "step": self.current_step,
            "action": act.to_dict(),
            "outcome": outcome.to_dict(),
            "budget_spent": self.total_budget_spent,
            "before_beliefs": before_beliefs,
            "after_beliefs": after_beliefs,
            "belief_deltas": belief_deltas,
            "ensemble_beliefs": self.ensemble.get_beliefs(),
            "realized_entropy_reduction": ensemble_update["realized_entropy_reduction"],
            "best_observed_k0": self.oracle.get_revealed_state_summary()["best_observed_k0"],
        }

        # Append to timeline
        revealed_str = (
            f"XRD Peak 2theta: {outcome.revealed_data.get('peak_two_theta', 0.0):.1f}°"
            if act.action_type == ExperimentActionType.XRD
            else f"Measured k0: {outcome.revealed_data.get('k0', 0.0):.5f} cm/s"
        )
        self.timeline.append(
            {
                "step": len(self.timeline) + 1,
                "action_id": act.action_id,
                "candidate_id": act.candidate_id,
                "action_type": act.action_type.value,
                "cost": act.estimated_cost,
                "hypothesis_tested": act.metadata.get("hypothesis_id", "H1"),
                "revealed_summary": revealed_str,
                "status": "COMPLETED",
            }
        )

        return summary

    def get_landscape_dataframe(self) -> pd.DataFrame:
        """Returns observable landscape DataFrame for UI visualization."""
        cand_df = self.oracle.get_candidate_pool().copy()
        all_comps = cand_df[["Au", "Ir", "Rh"]].to_numpy(dtype=np.float64)

        # Predictions
        _, struct_stds = self.structure_model.predict(all_comps)
        prop_means, prop_stds = self.property_model.predict(all_comps)

        cand_df["struct_uncertainty"] = struct_stds
        cand_df["predicted_k0"] = prop_means
        cand_df["prop_uncertainty"] = prop_stds

        # Status categorization
        statuses = []
        measured_k0s = []
        revealed_props = self.oracle.get_revealed_properties()

        for _, row in cand_df.iterrows():
            cid = row["candidate_id"]
            has_xrd = self.oracle.is_xrd_observed(cid)
            has_prop = self.oracle.is_property_observed(cid)

            if has_xrd and has_prop:
                statuses.append("Both XRD & Property")
            elif has_xrd:
                statuses.append("XRD Characterized")
            elif has_prop:
                statuses.append("Property Tested")
            else:
                statuses.append("Unobserved")

            if has_prop and cid in revealed_props:
                measured_k0s.append(revealed_props[cid].revealed_data["k0"])
            else:
                measured_k0s.append(np.nan)

        cand_df["status"] = statuses
        cand_df["measured_k0"] = measured_k0s
        return cand_df

    def get_disagreement_landscape(self, fast_mode: bool = True) -> pd.DataFrame:
        """Computes hypothesis disagreement and Expected Information Gain landscape across all candidates."""
        cand_df = self.get_landscape_dataframe()
        comp_cols = ["Au", "Ir", "Rh"]
        comps = cand_df[comp_cols].to_numpy(dtype=np.float64)
        cids = cand_df["candidate_id"].tolist()

        prop_disagreements: list[float] = []
        struct_disagreements: list[float] = []
        max_higs: list[float] = []
        best_actions: list[str] = []

        beliefs = self.ensemble.get_beliefs()
        dominant_h = max(beliefs.keys(), key=lambda h: beliefs[h])

        for i, cid in enumerate(cids):
            has_xrd = self.oracle.is_xrd_observed(cid)
            has_prop = self.oracle.is_property_observed(cid)

            eval_prop = self.falsification_policy.hig_estimator.evaluate_action_discrimination(
                candidate_id=cid,
                action_type=ExperimentActionType.PROPERTY,
                composition=comps[i],
                ensemble=self.ensemble,
                fast_mode=fast_mode,
                seed=self.seed,
            )

            eval_xrd = self.falsification_policy.hig_estimator.evaluate_action_discrimination(
                candidate_id=cid,
                action_type=ExperimentActionType.XRD,
                composition=comps[i],
                ensemble=self.ensemble,
                fast_mode=fast_mode,
                seed=self.seed,
            )

            prop_disagreements.append(eval_prop.property_disagreement)
            struct_disagreements.append(eval_xrd.structure_disagreement)

            # Determine best unobserved action
            if not has_prop and not has_xrd:
                if eval_prop.hypothesis_information_gain >= eval_xrd.hypothesis_information_gain:
                    max_higs.append(eval_prop.hypothesis_information_gain)
                    best_actions.append("PROPERTY")
                else:
                    max_higs.append(eval_xrd.hypothesis_information_gain)
                    best_actions.append("XRD")
            elif not has_prop:
                max_higs.append(eval_prop.hypothesis_information_gain)
                best_actions.append("PROPERTY")
            elif not has_xrd:
                max_higs.append(eval_xrd.hypothesis_information_gain)
                best_actions.append("XRD")
            else:
                max_higs.append(0.0)
                best_actions.append("COMPLETED")

        cand_df["property_disagreement"] = prop_disagreements
        cand_df["structure_disagreement"] = struct_disagreements
        cand_df["hypothesis_information_gain"] = max_higs
        cand_df["best_action"] = best_actions
        cand_df["dominant_hypothesis"] = dominant_h
        return cand_df

    def close(self) -> None:
        """Closes underlying ledger database connection."""
        if hasattr(self, "ledger") and self.ledger is not None:
            self.ledger.close()

    def __del__(self) -> None:
        self.close()

    def reset(self) -> None:
        """Resets the engine and oracle for fresh replay, preserving configured persistence db_path."""
        if hasattr(self, "ledger") and self.ledger is not None:
            self.ledger.close()
        self.oracle.reset()
        self.ledger = ExperimentLedger(db_path=self._db_path)
        self.xrd_extractor = XRDRepresentationExtractor()
        self.structure_model = StructureSurrogateModel(random_state=self.seed)
        self.property_model = PropertySurrogateModel(random_state=self.seed)
        self.hypothesis_engine = HypothesisEngine(get_default_hypotheses())
        self.ensemble = HypothesisEnsemble()
        self.current_step = 0
        self.total_budget_spent = 0.0
        self.timeline.clear()
        self._last_recommendation = None
        self._last_perspectives.clear()
        self._pre_pred_emb = None
        self._pre_pred_mean = None
        self._pre_pred_std = None
        self._pre_predictions.clear()
