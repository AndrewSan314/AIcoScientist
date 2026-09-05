from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from src.science.actions import ScientificAction, normalize_action_type
from src.science.domain import ModalityDefinition
from src.science.multimodal.evidence import MultimodalEvidenceLedger, bayesian_update, entropy
from src.science.multimodal.hypotheses import MultimodalScientificHypothesis
from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.schemas import ScientificObservable


@dataclass(frozen=True)
class MultimodalRecommendation:
    action: ScientificAction
    score: float
    why: dict[str, Any]
    why_not: list[dict[str, Any]]
    preregistration: dict[str, Any]
    scored_actions: list[dict[str, Any]] = field(default_factory=list)


class MultimodalDecisionEngine:
    """Candidate×modality decision layer independent of any extractor or GP backend."""

    def __init__(
        self,
        candidate_features_by_id: Mapping[str, Any],
        modalities: Sequence[ModalityDefinition],
        hypotheses: Mapping[str, MultimodalScientificHypothesis],
        *,
        prior_beliefs: Mapping[str, float] | None = None,
        discovery_values: Mapping[Any, float] | None = None,
        w_hig: float = 1.0,
        w_discovery: float = 1.0,
        w_cost: float = 1.0,
        sample_state_by_candidate: Mapping[str, Mapping[str, bool]] | None = None,
        seed: int = 42,
        policy_name: str = "HYBRID",
    ) -> None:
        if not hypotheses:
            raise ValueError("at least one hypothesis is required")
        self.candidate_features_by_id = {
            str(cid): np.asarray(features, dtype=np.float64)
            for cid, features in candidate_features_by_id.items()
        }
        self.modalities = {m.name: m for m in modalities}
        self.hypotheses = dict(hypotheses)
        self.beliefs = self._normalize(prior_beliefs or {hid: 1.0 for hid in self.hypotheses})
        self.discovery_values = dict(discovery_values or {})
        self.w_hig = float(w_hig)
        self.w_discovery = float(w_discovery)
        self.w_cost = float(w_cost)
        self.policy_name = str(policy_name).upper()
        self.sample_state_by_candidate = sample_state_by_candidate
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.observed_by_modality: dict[str, dict[str, Any]] = {m: {} for m in self.modalities}
        self.preregistered: dict[str, dict[str, Any]] = {}
        self._hig_diagnostics_by_action: dict[str, dict[str, Any]] = {}
        self.ledger = MultimodalEvidenceLedger()

        for hypothesis in self.hypotheses.values():
            hypothesis.fit(self.candidate_features_by_id, self.observed_by_modality)

    @staticmethod
    def _normalize(values: Mapping[str, float]) -> dict[str, float]:
        total = sum(max(float(v), 0.0) for v in values.values())
        if total <= 0:
            total = float(len(values))
            return {str(k): 1.0 / total for k in values}
        return {str(k): max(float(v), 0.0) / total for k, v in values.items()}

    @property
    def current_entropy(self) -> float:
        return entropy(self.beliefs)

    @staticmethod
    def _hig_epsilon(standard_error: float) -> float:
        """Three Monte-Carlo standard errors plus a small numeric allowance."""
        return 3.0 * max(0.0, float(standard_error)) + 1e-8

    def _requires_satisfied(self, candidate_id: str, modality: ModalityDefinition) -> bool:
        for prerequisite in modality.requires:
            if candidate_id not in self.observed_by_modality.get(prerequisite, {}):
                return False
        if self.sample_state_by_candidate is not None:
            state = self.sample_state_by_candidate.get(candidate_id, {})
            if any(not state.get(name, False) for name in modality.required_existing_sample_state):
                return False
        return True

    def enumerate_actions(self) -> list[ScientificAction]:
        actions: list[ScientificAction] = []
        for candidate_id in self.candidate_features_by_id:
            for modality in self.modalities.values():
                if modality.metadata.get("supported", True) is False:
                    continue
                if candidate_id in self.observed_by_modality.get(modality.name, {}):
                    continue
                if not self._requires_satisfied(candidate_id, modality):
                    continue
                actions.append(
                    ScientificAction(
                        action_id=f"{modality.name}_{candidate_id}",
                        candidate_id=candidate_id,
                        action_type=modality.name,
                        estimated_cost=modality.cost,
                        requested_at_step=self.step + 1,
                        metadata={"cost_units": modality.cost_units, "prerequisites": list(modality.requires)},
                    )
                )
        return actions

    def _predictions(self, action: ScientificAction) -> dict[str, PredictiveObservableDistribution]:
        modality = normalize_action_type(action.action_type)
        return {
            hid: hypothesis.predict_observable_distribution(
                action.candidate_id,
                modality,
                self.observed_by_modality,
                candidate_features=self.candidate_features_by_id[action.candidate_id],
            )
            for hid, hypothesis in self.hypotheses.items()
        }

    def expected_hypothesis_information_gain(
        self,
        action: ScientificAction,
        predictions: Mapping[str, PredictiveObservableDistribution] | None = None,
        samples: int = 128,
    ) -> float:
        return float(self.expected_hypothesis_information_gain_diagnostics(action, predictions, samples)["clipped_hig_nats"])

    def expected_hypothesis_information_gain_diagnostics(
        self,
        action: ScientificAction,
        predictions: Mapping[str, PredictiveObservableDistribution] | None = None,
        samples: int = 128,
    ) -> dict[str, Any]:
        preds = dict(predictions or self._predictions(action))
        if len(preds) < 2:
            result = {
                "raw_hig_mc_nats": 0.0,
                "clipped_hig_nats": 0.0,
                "current_entropy_nats": self.current_entropy,
                "posterior_entropy_mc_mean": self.current_entropy,
                "posterior_entropy_mc_std": 0.0,
                "hig_mc_standard_error": 0.0,
                "mc_samples": 0,
                "hig_bound_k": 3.0,
                "hig_numeric_epsilon_nats": 1e-8,
                "hig_bound_epsilon_nats": 1e-8,
                "raw_hig_lower_bound_ok": True,
                "raw_hig_upper_bound_ok": True,
                "predictive_variance_by_hypothesis": {hid: np.asarray(pred.variance, dtype=float).tolist() for hid, pred in preds.items()},
            }
            self._hig_diagnostics_by_action[action.action_id] = result
            return result
        keys = list(preds)
        probs = np.asarray([self.beliefs[k] for k in keys], dtype=np.float64)
        before = self.current_entropy
        posterior_entropies: list[float] = []
        action_seed = hashlib.sha256(
            f"{self.seed}|{self.step}|{action.action_id}|HIG-v2".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(action_seed[:8], "big", signed=False))
        for _ in range(max(1, samples)):
            idx = int(rng.choice(len(keys), p=probs / probs.sum()))
            sample = preds[keys[idx]].sample(rng)
            log_likes = {k: preds[k].log_pdf(sample) for k in keys}
            posterior = bayesian_update(self.beliefs, log_likes)
            posterior_entropies.append(entropy(posterior))
        posterior_mean = float(np.mean(posterior_entropies))
        posterior_std = float(np.std(posterior_entropies, ddof=1)) if len(posterior_entropies) > 1 else 0.0
        standard_error = posterior_std / np.sqrt(max(1, len(posterior_entropies)))
        raw_hig = float(before - posterior_mean)
        epsilon = self._hig_epsilon(standard_error)
        result = {
            "raw_hig_mc_nats": raw_hig,
            "clipped_hig_nats": float(np.clip(raw_hig, 0.0, before)),
            "current_entropy_nats": float(before),
            "posterior_entropy_mc_mean": posterior_mean,
            "posterior_entropy_mc_std": posterior_std,
            "hig_mc_standard_error": float(standard_error),
            "mc_samples": len(posterior_entropies),
            "hig_bound_k": 3.0,
            "hig_numeric_epsilon_nats": 1e-8,
            "hig_bound_epsilon_nats": float(epsilon),
            "raw_hig_lower_bound_ok": bool(raw_hig >= -epsilon),
            "raw_hig_upper_bound_ok": bool(raw_hig <= before + epsilon),
            "predictive_variance_by_hypothesis": {hid: np.asarray(pred.variance, dtype=float).tolist() for hid, pred in preds.items()},
        }
        self._hig_diagnostics_by_action[action.action_id] = result
        return result

    def _discovery_value(self, action: ScientificAction) -> float:
        modality = normalize_action_type(action.action_type)
        return float(self.discovery_values.get((action.candidate_id, modality), self.discovery_values.get(action.candidate_id, 0.0)))

    def register_selected_action(
        self,
        action: ScientificAction,
        *,
        predictions: Mapping[str, PredictiveObservableDistribution] | None = None,
        hig: float | None = None,
        discovery: float | None = None,
    ) -> MultimodalRecommendation:
        """Register one externally selected action without scoring/registering its siblings."""
        if action.action_id not in {item.action_id for item in self.enumerate_actions()}:
            raise ValueError(f"action is not currently feasible: {action.action_id}")
        preds = dict(predictions or self._predictions(action))
        expected_hig = float(self.expected_hypothesis_information_gain(action, preds) if hig is None else hig)
        hig_diagnostics = dict(self._hig_diagnostics_by_action.get(action.action_id, {}))
        if not hig_diagnostics:
            hig_diagnostics = {
                "raw_hig_mc_nats": expected_hig,
                "clipped_hig_nats": expected_hig,
                "current_entropy_nats": self.current_entropy,
                "posterior_entropy_mc_mean": self.current_entropy - expected_hig,
                "posterior_entropy_mc_std": None,
                "hig_mc_standard_error": None,
                "mc_samples": 0,
                "hig_bound_k": 3.0,
                "hig_numeric_epsilon_nats": 1e-8,
                "hig_bound_epsilon_nats": None,
                "raw_hig_lower_bound_ok": None,
                "raw_hig_upper_bound_ok": None,
                "audit_source": "externally_supplied_hig",
            }
        discovery_value = float(self._discovery_value(action) if discovery is None else discovery)
        max_cost = max(max((item.estimated_cost for item in self.enumerate_actions()), default=1.0), 1.0)
        score = self._score(
            expected_hig,
            discovery_value,
            action.estimated_cost,
            max_hig=max(expected_hig, 1e-12),
            max_discovery=max(discovery_value, 1e-12),
            max_cost=max_cost,
        )
        registration = self._preregister(action, preds, expected_hig, discovery_value, score, hig_diagnostics)
        why = {
            "expected_hig_nats": expected_hig,
            "discovery_utility": discovery_value,
            "normalized_cost": action.estimated_cost / max_cost,
            "current_entropy_nats": self.current_entropy,
            "policy_name": self.policy_name,
            "selection_mode": "external_policy_selector",
            "hig_diagnostics": hig_diagnostics,
            "raw_hig_mc_nats": hig_diagnostics["raw_hig_mc_nats"],
            "clipped_hig_nats": hig_diagnostics["clipped_hig_nats"],
            "hig_bound_epsilon_nats": hig_diagnostics["hig_bound_epsilon_nats"],
            "hig_upper_bound_epsilon_nats": hig_diagnostics["hig_bound_epsilon_nats"],
            "hig_lower_bound_ok": hig_diagnostics["raw_hig_lower_bound_ok"],
            "hig_upper_bound_ok": hig_diagnostics["raw_hig_upper_bound_ok"],
        }
        return MultimodalRecommendation(action, score, why, [], registration, [])

    def _score(
        self,
        hig: float,
        discovery: float,
        cost: float,
        *,
        max_hig: float,
        max_discovery: float,
        max_cost: float,
    ) -> float:
        normalized_hig = hig / max(max_hig, 1e-12)
        normalized_discovery = discovery / max(max_discovery, 1e-12) if max_discovery > 0 else 0.0
        normalized_cost = cost / max(max_cost, 1e-12)
        if self.policy_name in {"PURE_HIG", "HIG"}:
            return normalized_hig
        if self.policy_name in {"DISCOVERY_ONLY", "PROPERTY_ONLY", "PROPERTY_ONLY_BO"}:
            return normalized_discovery
        if self.policy_name == "UNCERTAINTY_ONLY":
            return normalized_hig
        return self.w_hig * normalized_hig + self.w_discovery * normalized_discovery - self.w_cost * normalized_cost

    def _preregister(
        self,
        action: ScientificAction,
        predictions: Mapping[str, PredictiveObservableDistribution],
        hig: float,
        discovery: float,
        score: float,
        hig_diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        diagnostics = dict(hig_diagnostics or self._hig_diagnostics_by_action.get(action.action_id, {}))
        epsilon = diagnostics.get("hig_bound_epsilon_nats")
        record = {
            "event": "PREREGISTERED_SELECTED_ACTION",
            "step": self.step + 1,
            "event_sequence": len(self.ledger.events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "measurement_revealed": False,
            "action": action.to_dict(),
            "beliefs_before": dict(self.beliefs),
            "predictive_distributions": {hid: p.to_dict() for hid, p in predictions.items()},
            "expected_hig_nats": hig,
            "discovery_utility": discovery,
            "normalized_cost": action.estimated_cost,
            "total_action_score": score,
            "current_hypothesis_entropy_nats": self.current_entropy,
            "hig_diagnostics": diagnostics,
            "raw_hig_mc_nats": diagnostics.get("raw_hig_mc_nats", hig),
            "clipped_hig_nats": diagnostics.get("clipped_hig_nats", hig),
            "posterior_entropy_mc_mean": diagnostics.get("posterior_entropy_mc_mean"),
            "posterior_entropy_mc_std": diagnostics.get("posterior_entropy_mc_std"),
            "hig_mc_standard_error": diagnostics.get("hig_mc_standard_error"),
            "mc_samples": diagnostics.get("mc_samples"),
            "hig_bound_epsilon_nats": epsilon,
            "hig_upper_bound_epsilon_nats": epsilon,
            "raw_hig_lower_bound_ok": diagnostics.get("raw_hig_lower_bound_ok"),
            "raw_hig_upper_bound_ok": diagnostics.get("raw_hig_upper_bound_ok"),
            "hig_lower_bound_ok": diagnostics.get("raw_hig_lower_bound_ok"),
            "hig_upper_bound_ok": diagnostics.get("raw_hig_upper_bound_ok"),
        }
        self.preregistered[action.action_id] = record
        self.ledger.append(record)
        return record

    def recommend(self, *, samples: int = 128) -> MultimodalRecommendation:
        actions = self.enumerate_actions()
        if not actions:
            raise RuntimeError("No feasible candidate×modality actions remain")
        max_cost = max(max((a.estimated_cost for a in actions), default=1.0), 1.0)
        scored: list[dict[str, Any]] = []
        raw_scores: list[tuple[ScientificAction, dict[str, PredictiveObservableDistribution], float, float]] = []
        for action in actions:
            predictions = self._predictions(action)
            hig = self.expected_hypothesis_information_gain(action, predictions, samples=samples)
            discovery = self._discovery_value(action)
            raw_scores.append((action, predictions, hig, discovery))
        max_hig = max((row[2] for row in raw_scores), default=0.0)
        max_discovery = max((row[3] for row in raw_scores), default=0.0)
        for action, predictions, hig, discovery in raw_scores:
            hig_diagnostics = self._hig_diagnostics_by_action[action.action_id]
            score = self._score(
                hig,
                discovery,
                action.estimated_cost,
                max_hig=max_hig,
                max_discovery=max_discovery,
                max_cost=max_cost,
            )
            self.ledger.append({
                "event": "ACTION_SCORE_RECORD",
                "event_sequence": len(self.ledger.events) + 1,
                "step": self.step + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action.to_dict(),
                "expected_hig_nats": hig,
                "discovery_utility": discovery,
                "normalized_cost": action.estimated_cost / max_cost,
                "total_action_score": score,
                "policy_name": self.policy_name,
                "current_hypothesis_entropy_nats": self.current_entropy,
                "hig_diagnostics": hig_diagnostics,
                "raw_hig_mc_nats": hig_diagnostics["raw_hig_mc_nats"],
                "clipped_hig_nats": hig_diagnostics["clipped_hig_nats"],
                "posterior_entropy_mc_mean": hig_diagnostics["posterior_entropy_mc_mean"],
                "posterior_entropy_mc_std": hig_diagnostics["posterior_entropy_mc_std"],
                "hig_mc_standard_error": hig_diagnostics["hig_mc_standard_error"],
                "mc_samples": hig_diagnostics["mc_samples"],
                "hig_bound_epsilon_nats": hig_diagnostics["hig_bound_epsilon_nats"],
                "hig_upper_bound_epsilon_nats": hig_diagnostics["hig_bound_epsilon_nats"],
                "raw_hig_lower_bound_ok": hig_diagnostics["raw_hig_lower_bound_ok"],
                "raw_hig_upper_bound_ok": hig_diagnostics["raw_hig_upper_bound_ok"],
                "hig_lower_bound_ok": hig_diagnostics["raw_hig_lower_bound_ok"],
                "hig_upper_bound_ok": hig_diagnostics["raw_hig_upper_bound_ok"],
            })
            scored.append({
                "action": action.to_dict(),
                "expected_hig_nats": hig,
                "discovery_utility": discovery,
                "normalized_cost": action.estimated_cost / max_cost,
                "total_action_score": score,
                "dominant_hypothesis_disagreement": self._disagreement(predictions),
                "current_hypothesis_entropy_nats": self.current_entropy,
                "hig_diagnostics": hig_diagnostics,
                "raw_hig_mc_nats": hig_diagnostics["raw_hig_mc_nats"],
                "clipped_hig_nats": hig_diagnostics["clipped_hig_nats"],
                "posterior_entropy_mc_mean": hig_diagnostics["posterior_entropy_mc_mean"],
                "posterior_entropy_mc_std": hig_diagnostics["posterior_entropy_mc_std"],
                "hig_mc_standard_error": hig_diagnostics["hig_mc_standard_error"],
                "mc_samples": hig_diagnostics["mc_samples"],
                "hig_bound_epsilon_nats": hig_diagnostics["hig_bound_epsilon_nats"],
                "hig_upper_bound_epsilon_nats": hig_diagnostics["hig_bound_epsilon_nats"],
                "raw_hig_lower_bound_ok": hig_diagnostics["raw_hig_lower_bound_ok"],
                "raw_hig_upper_bound_ok": hig_diagnostics["raw_hig_upper_bound_ok"],
                "hig_lower_bound_ok": hig_diagnostics["raw_hig_lower_bound_ok"],
                "hig_upper_bound_ok": hig_diagnostics["raw_hig_upper_bound_ok"],
            })
        scored.sort(key=lambda row: row["total_action_score"], reverse=True)
        top = scored[0]
        action = ScientificAction.from_dict(top["action"])
        registration = self._preregister(
            action,
            next(predictions for candidate, predictions, _, _ in raw_scores if candidate.action_id == action.action_id),
            top["expected_hig_nats"],
            top["discovery_utility"],
            top["total_action_score"],
            top["hig_diagnostics"],
        )
        why = {
            "expected_hig_nats": top["expected_hig_nats"],
            "discovery_utility": top["discovery_utility"],
            "normalized_cost": top["normalized_cost"],
            "dominant_hypothesis_disagreement": top["dominant_hypothesis_disagreement"],
            "current_entropy_nats": self.current_entropy,
            "hig_diagnostics": top["hig_diagnostics"],
            "raw_hig_mc_nats": top["raw_hig_mc_nats"],
            "clipped_hig_nats": top["clipped_hig_nats"],
            "hig_bound_epsilon_nats": top["hig_bound_epsilon_nats"],
            "hig_lower_bound_ok": top["hig_lower_bound_ok"],
            "hig_upper_bound_ok": top["hig_upper_bound_ok"],
            "policy_name": self.policy_name,
            "score_components": {
                "normalized_hig": top["expected_hig_nats"] / max(max_hig, 1e-12),
                "normalized_discovery": top["discovery_utility"] / max(max_discovery, 1e-12) if max_discovery > 0 else 0.0,
                "normalized_cost": top["normalized_cost"],
            },
            "falsification_signatures": {
                hid: hypothesis.falsification_signature()
                for hid, hypothesis in self.hypotheses.items()
            },
        }
        why_not = [
            {
                "action": row["action"],
                "reason": "Lower preregistered action value under the same policy weights.",
                "expected_hig_nats": row["expected_hig_nats"],
                "discovery_utility": row["discovery_utility"],
                "normalized_cost": row["normalized_cost"],
                "score_gap": top["total_action_score"] - row["total_action_score"],
                "total_action_score": row["total_action_score"],
            }
            for row in scored[1:4]
        ]
        return MultimodalRecommendation(action, top["total_action_score"], why, why_not, registration, scored)

    @staticmethod
    def _disagreement(predictions: Mapping[str, PredictiveObservableDistribution]) -> list[str]:
        rows = []
        for hid, prediction in predictions.items():
            rows.append(f"{hid} predicts mean={np.round(prediction.mean, 4).tolist()}")
        return rows

    def observe(self, observable: ScientificObservable) -> dict[str, Any]:
        modality = normalize_action_type(observable.modality)
        action_id = f"{modality}_{observable.candidate_id}"
        registration = self.preregistered.get(action_id)
        if registration is None:
            raise ValueError("measurement must be preregistered before reveal")
        if registration["measurement_revealed"]:
            raise ValueError(f"measurement already revealed: {action_id}")
        if not observable.observable_names:
            predicted_names = tuple(next(iter(registration["predictive_distributions"].values()))["observable_names"])
            observable = replace(
                observable,
                observable_names=predicted_names,
                observable_type="vector" if np.asarray(observable.value).size > 1 else observable.observable_type,
                provenance={**observable.provenance, "legacy_schema_inferred": True},
            )
        before = dict(self.beliefs)
        log_likelihoods = {
            hid: hypothesis.log_likelihood(observable, self.observed_by_modality)
            for hid, hypothesis in self.hypotheses.items()
        }
        self.beliefs = bayesian_update(self.beliefs, log_likelihoods)
        self.observed_by_modality.setdefault(modality, {})[observable.candidate_id] = observable
        for hypothesis in self.hypotheses.values():
            hypothesis.fit(self.candidate_features_by_id, self.observed_by_modality)
        after = dict(self.beliefs)
        pairwise_log_bayes_factors = {}
        hypothesis_ids = list(log_likelihoods)
        for left_index, left in enumerate(hypothesis_ids):
            for right in hypothesis_ids[left_index + 1:]:
                pairwise_log_bayes_factors[f"{left}_vs_{right}"] = log_likelihoods[left] - log_likelihoods[right]
        modality_roles = {
            hid: registration["predictive_distributions"].get(hid, {}).get("metadata", {}).get("modality_role", "UNSPECIFIED")
            for hid in hypothesis_ids
        }
        likelihood_modes = {
            hid: registration["predictive_distributions"].get(hid, {}).get("metadata", {}).get("likelihood_mode", "UNSPECIFIED")
            for hid in hypothesis_ids
        }
        registration["measurement_revealed"] = True
        registration["reveal_timestamp"] = observable.timestamp or datetime.now(timezone.utc).isoformat()
        event = {
            "event": "MEASUREMENT_REVEALED",
            "step": self.step + 1,
            "event_sequence": len(self.ledger.events) + 1,
            "timestamp": observable.timestamp or datetime.now(timezone.utc).isoformat(),
            "action": registration["action"],
            "observed_measurement": observable.to_dict(),
            "likelihood_under_hypothesis": log_likelihoods,
            "log_bayes_factor_pairwise": pairwise_log_bayes_factors,
            "modality_diagnostic_role": modality_roles,
            "likelihood_mode": likelihood_modes,
            "beliefs_before": before,
            "beliefs_after": after,
            "posterior_delta": {hid: after[hid] - before[hid] for hid in after},
            "realized_entropy_reduction_nats": entropy(before) - entropy(after),
        }
        self.ledger.append(event)
        self.ledger.append({
            "event": "BELIEF_UPDATE",
            "step": self.step + 1,
            "event_sequence": len(self.ledger.events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": registration["action"],
            "beliefs_before": before,
            "beliefs_after": after,
            "likelihood_under_hypothesis": log_likelihoods,
        })
        self.step += 1
        return event

    def write_evidence_ledger(self, path: str) -> None:
        self.ledger.write_jsonl(path)


MultimodalScientificDecisionEngine = MultimodalDecisionEngine

__all__ = ["MultimodalDecisionEngine", "MultimodalRecommendation", "MultimodalScientificDecisionEngine"]
