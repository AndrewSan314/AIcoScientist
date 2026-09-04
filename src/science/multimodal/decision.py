from __future__ import annotations

from dataclasses import dataclass, field
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
        self.sample_state_by_candidate = sample_state_by_candidate
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.observed_by_modality: dict[str, dict[str, Any]] = {m: {} for m in self.modalities}
        self.preregistered: dict[str, dict[str, Any]] = {}
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
        preds = dict(predictions or self._predictions(action))
        if len(preds) < 2:
            return 0.0
        keys = list(preds)
        probs = np.asarray([self.beliefs[k] for k in keys], dtype=np.float64)
        before = self.current_entropy
        posterior_entropies = []
        for _ in range(max(1, samples)):
            idx = int(self.rng.choice(len(keys), p=probs / probs.sum()))
            sample = preds[keys[idx]].sample(self.rng)
            log_likes = {k: preds[k].log_pdf(sample) for k in keys}
            posterior = bayesian_update(self.beliefs, log_likes)
            posterior_entropies.append(entropy(posterior))
        return float(np.clip(before - float(np.mean(posterior_entropies)), 0.0, before))

    def _discovery_value(self, action: ScientificAction) -> float:
        modality = normalize_action_type(action.action_type)
        return float(self.discovery_values.get((action.candidate_id, modality), self.discovery_values.get(action.candidate_id, 0.0)))

    def _preregister(self, action: ScientificAction, predictions: Mapping[str, PredictiveObservableDistribution], hig: float, discovery: float, score: float) -> dict[str, Any]:
        record = {
            "event": "PREREGISTERED_PREDICTION",
            "step": self.step + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "measurement_revealed": False,
            "action": action.to_dict(),
            "beliefs_before": dict(self.beliefs),
            "predictive_distributions": {hid: p.to_dict() for hid, p in predictions.items()},
            "expected_hig_nats": hig,
            "discovery_utility": discovery,
            "normalized_cost": action.estimated_cost,
            "total_action_score": score,
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
        registrations: dict[str, dict[str, Any]] = {}
        for action in actions:
            predictions = self._predictions(action)
            hig = self.expected_hypothesis_information_gain(action, predictions, samples=samples)
            discovery = self._discovery_value(action)
            score = self.w_hig * hig + self.w_discovery * discovery - self.w_cost * (action.estimated_cost / max_cost)
            registrations[action.action_id] = self._preregister(action, predictions, hig, discovery, score)
            scored.append({
                "action": action.to_dict(),
                "expected_hig_nats": hig,
                "discovery_utility": discovery,
                "normalized_cost": action.estimated_cost / max_cost,
                "total_action_score": score,
                "dominant_hypothesis_disagreement": self._disagreement(predictions),
            })
        scored.sort(key=lambda row: row["total_action_score"], reverse=True)
        top = scored[0]
        action = ScientificAction.from_dict(top["action"])
        why = {
            "expected_hig_nats": top["expected_hig_nats"],
            "discovery_utility": top["discovery_utility"],
            "normalized_cost": top["normalized_cost"],
            "dominant_hypothesis_disagreement": top["dominant_hypothesis_disagreement"],
            "current_entropy_nats": self.current_entropy,
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
                "total_action_score": row["total_action_score"],
            }
            for row in scored[1:4]
        ]
        return MultimodalRecommendation(action, top["total_action_score"], why, why_not, registrations[action.action_id], scored)

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
        before = dict(self.beliefs)
        log_likelihoods = {
            hid: hypothesis.log_likelihood(observable, self.observed_by_modality)
            for hid, hypothesis in self.hypotheses.items()
        }
        self.beliefs = bayesian_update(self.beliefs, log_likelihoods)
        self.observed_by_modality.setdefault(modality, {})[observable.candidate_id] = observable.value
        for hypothesis in self.hypotheses.values():
            hypothesis.fit(self.candidate_features_by_id, self.observed_by_modality)
        after = dict(self.beliefs)
        registration["measurement_revealed"] = True
        registration["reveal_timestamp"] = observable.timestamp or datetime.now(timezone.utc).isoformat()
        event = {
            "event": "MEASUREMENT_REVEALED",
            "step": self.step + 1,
            "action": registration["action"],
            "observed_measurement": observable.to_dict(),
            "likelihood_under_hypothesis": log_likelihoods,
            "beliefs_before": before,
            "beliefs_after": after,
            "realized_entropy_reduction_nats": entropy(before) - entropy(after),
        }
        self.ledger.append(event)
        self.step += 1
        return event

    def write_evidence_ledger(self, path: str) -> None:
        self.ledger.write_jsonl(path)


MultimodalScientificDecisionEngine = MultimodalDecisionEngine

__all__ = ["MultimodalDecisionEngine", "MultimodalRecommendation", "MultimodalScientificDecisionEngine"]
