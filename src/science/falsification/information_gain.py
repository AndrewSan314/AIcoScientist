from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from src.science.actions import ActionType, ExperimentActionType
from src.science.hypothesis_models import HypothesisEnsemble, PredictiveDistribution

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscriminationEvaluation:
    """Quantitative summary of hypothesis discrimination value for a single candidate action."""

    candidate_id: str
    action_type: ActionType
    hypothesis_information_gain: float
    current_entropy: float
    expected_posterior_entropy: float
    property_disagreement: float
    structure_disagreement: float
    predictions: dict[str, PredictiveDistribution]
    metadata: dict[str, Any]
    observation_disagreement: float = 0.0
    disagreement_by_modality: dict[str, float] = field(default_factory=dict)


class HypothesisInformationGainEstimator:
    """Monte Carlo Estimator for Expected Hypothesis Information Gain (HIG).

    Calculates the mutual information between the discrete hypothesis identity H and
    future experimental measurement outcome Y_a for action a = (candidate, modality):

        HIG(a) = I(H ; Y_a | D) = H[P(H|D)] - E_{y ~ p(y|a,D)}[ H[P(H | D, y)] ]

    Properties:
    1. Non-negative: HIG(a) >= 0.
    2. Zero when all hypotheses make identical predictions (no discrimination).
    3. Upper-bounded by current hypothesis entropy H[P(H|D)].
    4. Supports scalar property k0 and multidimensional XRD embedding observations.
    """

    def __init__(self, n_samples_benchmark: int = 256, n_samples_demo: int = 64) -> None:
        self.n_samples_benchmark = n_samples_benchmark
        self.n_samples_demo = n_samples_demo

    def evaluate_action_discrimination(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        ensemble: HypothesisEnsemble,
        observed_xrd_embedding: np.ndarray | None = None,
        observed_modalities: Mapping[str, Any] | None = None,
        fast_mode: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> DiscriminationEvaluation:
        """Evaluates Expected Hypothesis Information Gain for a candidate action."""
        rng = np.random.default_rng(seed)
        n_samples = self.n_samples_demo if fast_mode else self.n_samples_benchmark

        # 1. Obtain current beliefs and current entropy
        current_beliefs = ensemble.get_beliefs()
        current_entropy = ensemble.get_entropy()
        hids = list(ensemble.hypotheses.keys())
        p_vec = np.array([current_beliefs[hid] for hid in hids], dtype=np.float64)
        if np.isnan(p_vec).any() or np.sum(p_vec) <= 0:
            p_vec = np.ones(len(hids), dtype=np.float64) / len(hids)
        else:
            p_vec = p_vec / np.sum(p_vec)

        # 2. Get predictive distributions from all hypotheses
        preds = ensemble.predict_all(
            candidate_id=candidate_id,
            action_type=action_type,
            composition=composition,
            observed_xrd_embedding=observed_xrd_embedding,
            observed_modalities=observed_modalities,
        )

        if not preds or len(preds) < 2:
            return DiscriminationEvaluation(
                candidate_id=candidate_id,
                action_type=action_type,
                hypothesis_information_gain=0.0,
                current_entropy=current_entropy,
                expected_posterior_entropy=current_entropy,
                property_disagreement=0.0,
                structure_disagreement=0.0,
                predictions=preds,
                metadata={"reason": "insufficient_hypotheses"},
                observation_disagreement=0.0,
                disagreement_by_modality={},
            )

        # 3. Compute pairwise disagreement metrics
        first_pred = next(iter(preds.values()))
        if len(first_pred.mean) == 1:
            means = [preds[hid].mean[0] for hid in hids if hid in preds]
            prop_disagreement = float(np.var(means)) if len(means) > 1 else 0.0
            struct_disagreement = 0.0
            obs_disagreement = prop_disagreement
        else:
            emb_means = [preds[hid].mean for hid in hids if hid in preds]
            if len(emb_means) >= 2:
                dists = [np.linalg.norm(emb_means[i] - emb_means[j]) for i in range(len(emb_means)) for j in range(i + 1, len(emb_means))]
                struct_disagreement = float(np.mean(dists))
            else:
                struct_disagreement = 0.0
            prop_disagreement = 0.0
            obs_disagreement = struct_disagreement

        mod_name = str(action_type.value if hasattr(action_type, "value") else action_type)
        disagreement_by_modality = {mod_name: obs_disagreement}

        # If all hypotheses have virtually identical predictions, HIG is analytically zero
        all_means = np.array([preds[hid].mean for hid in hids if hid in preds])
        all_vars = np.array([preds[hid].variance for hid in hids if hid in preds])
        mean_diff = np.max(np.abs(all_means - all_means[0:1]))
        var_diff = np.max(np.abs(all_vars - all_vars[0:1]))
        if mean_diff < 1e-9 and var_diff < 1e-9:
            return DiscriminationEvaluation(
                candidate_id=candidate_id,
                action_type=action_type,
                hypothesis_information_gain=0.0,
                current_entropy=current_entropy,
                expected_posterior_entropy=current_entropy,
                property_disagreement=prop_disagreement,
                structure_disagreement=struct_disagreement,
                predictions=preds,
                metadata={"analytic_zero": True},
                observation_disagreement=obs_disagreement,
                disagreement_by_modality=disagreement_by_modality,
            )

        # 4. Monte Carlo integration over the predictive mixture
        # Sample hypothesis indices according to current beliefs
        sampled_h_indices = rng.choice(len(hids), size=n_samples, p=p_vec)
        posterior_entropies = np.zeros(n_samples, dtype=np.float64)

        for s in range(n_samples):
            h_sampled_id = hids[sampled_h_indices[s]]
            # Draw hypothetical measurement outcome from the sampled hypothesis predictive distribution
            y_hypothetical = preds[h_sampled_id].sample(n_samples=1, rng=rng)[0]

            # Compute log-predictive density of y_hypothetical under each hypothesis
            log_likes = np.zeros(len(hids), dtype=np.float64)
            for j, hid in enumerate(hids):
                log_likes[j] = preds[hid].log_pdf(y_hypothetical)

            # Updated unnormalized log-posterior: log P(H_j) + log p(y | H_j)
            log_likes = np.clip(log_likes, -500.0, 500.0)
            unnorm_log_post = np.log(np.maximum(p_vec, 1e-12)) + log_likes
            unnorm_log_post = np.clip(unnorm_log_post, -500.0, 500.0)
            norm_log_post = unnorm_log_post - logsumexp(unnorm_log_post)
            post_probs = np.exp(norm_log_post)
            if np.isnan(post_probs).any() or np.sum(post_probs) <= 0:
                post_probs = np.ones(len(hids), dtype=np.float64) / len(hids)
            else:
                post_probs = post_probs / np.sum(post_probs)
            post_probs = np.maximum(post_probs, 1e-12)

            # Compute posterior entropy
            entropy_s = -np.sum(post_probs * np.log(post_probs))
            posterior_entropies[s] = entropy_s

        expected_posterior_entropy = float(np.mean(posterior_entropies))
        raw_hig = current_entropy - expected_posterior_entropy
        hig = float(np.clip(raw_hig, 0.0, current_entropy))

        return DiscriminationEvaluation(
            candidate_id=candidate_id,
            action_type=action_type,
            hypothesis_information_gain=hig,
            current_entropy=current_entropy,
            expected_posterior_entropy=expected_posterior_entropy,
            property_disagreement=prop_disagreement,
            structure_disagreement=struct_disagreement,
            predictions=preds,
            metadata={"n_mc_samples": n_samples},
            observation_disagreement=obs_disagreement,
            disagreement_by_modality=disagreement_by_modality,
        )
