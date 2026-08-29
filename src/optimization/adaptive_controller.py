from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.optimization.acquisition import compute_acquisition


@dataclass(frozen=True)
class ControllerDecision:
    """Decision output of the autonomous adaptive Bayesian Optimization controller."""
    chosen_method: str
    beta: float
    xi: float
    exploration_score: float
    exploitation_score: float
    controller_reason: str
    should_stop: bool
    stop_reason: str
    max_expected_improvement: float
    max_posterior_uncertainty: float
    mean_posterior_uncertainty: float


class AdaptiveBOController:
    """Autonomous adaptive controller that decides HOW to search at each experiment.

    Operates strictly on optimizer-visible posterior signals:
    - Posterior mean and epistemic variance from GP surrogate
    - Progress / stagnation of observed objective values
    - Remaining experimental budget fraction
    """

    def __init__(
        self,
        base_beta: float = 2.0,
        base_xi: float = 0.01,
        stagnation_threshold: int = 3,
        ei_high_threshold: float = 5.0,
    ) -> None:
        self.base_beta = base_beta
        self.base_xi = base_xi
        self.stagnation_threshold = stagnation_threshold
        self.ei_high_threshold = ei_high_threshold

    def compute_stagnation(self, observed_targets: Sequence[float]) -> int:
        """Calculates the number of query steps elapsed since the maximum observed target increased."""
        if len(observed_targets) <= 1:
            return 0
        best_so_far = observed_targets[0]
        steps_since_best = 0
        for val in observed_targets[1:]:
            if val > best_so_far + 1e-4:
                best_so_far = val
                steps_since_best = 0
            else:
                steps_since_best += 1
        return steps_since_best

    def decide(
        self,
        step: int,
        total_queries: int,
        observed_targets: Sequence[float],
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
    ) -> ControllerDecision:
        """Evaluates posterior state and returns the optimal acquisition strategy and scientific rationale."""
        m = np.asarray(pred_mean, dtype=float)
        s = np.asarray(pred_std, dtype=float)

        current_best = float(np.max(observed_targets)) if len(observed_targets) > 0 else 0.0
        stagnation = self.compute_stagnation(observed_targets)

        budget_rem_fraction = max(0.0, float(total_queries - step) / max(1, total_queries))

        max_unc = float(np.max(s))
        mean_unc = float(np.mean(s))
        max_mean = float(np.max(m))

        ei_scores = compute_acquisition("expected_improvement", m, s, best_observed=current_best, xi=self.base_xi)
        max_ei = float(np.max(ei_scores))

        # Dynamic beta schedule
        beta_t = self.base_beta * np.sqrt(max(0.05, budget_rem_fraction)) * (
            1.0 + 0.5 * min(1.0, stagnation / max(1, self.stagnation_threshold))
        )
        beta_t = float(np.clip(beta_t, 0.1, 4.0))

        # Exploration and exploitation scores
        expl_score = float(max_unc * (0.5 + 0.5 * budget_rem_fraction))
        explt_score = float(max(0.0, max_mean - current_best) + max_ei)

        # 1. Early exploration
        if step <= 2:
            chosen_method = "gp_ucb"
            reason = f"Early exploration phase (step {step}/{total_queries}); prioritizing epistemic coverage (beta={beta_t:.2f})."

        # 2. Late budget exhaustion: strictly exploit posterior incumbent
        elif budget_rem_fraction <= 0.15:
            chosen_method = "greedy"
            reason = f"Late budget phase ({budget_rem_fraction:.0%} remaining); switching to pure exploitation of posterior mean."

        # 3. Stagnation breakout with high epistemic variance
        elif stagnation >= self.stagnation_threshold and budget_rem_fraction > 0.2 and max_unc > 15.0:
            chosen_method = "gp_ucb"
            reason = f"Stagnation detected ({stagnation} steps without improvement) with high epistemic variance ({max_unc:.1f}); boosting exploration (beta={beta_t:.2f})."

        # 4. High-confidence Expected Improvement
        elif max_ei >= self.ei_high_threshold and max_mean >= current_best:
            chosen_method = "expected_improvement"
            reason = f"High Expected Improvement ({max_ei:.1f} cycles) detected; targeting high-probability improvement zone."

        # 5. High-confidence local peak
        elif max_mean > current_best + 10.0 and max_unc < 12.0:
            chosen_method = "greedy"
            reason = f"High-confidence local improvement identified ({max_mean:.1f} vs best {current_best:.1f}, std={max_unc:.1f}); focusing on exploitation."

        # 6. Standard balanced acquisition
        else:
            chosen_method = "expected_improvement"
            reason = f"Balancing predicted mean ({max_mean:.1f}) and posterior epistemic uncertainty ({max_unc:.1f})."

        # Stopping signal diagnostic (evaluator-independent)
        if max_ei < 0.5 and max_unc < 8.0 and stagnation >= 6:
            should_stop = True
            stop_reason = f"Max Expected Improvement ({max_ei:.2f} cycles) is below threshold with depleted uncertainty across candidates."
        else:
            should_stop = False
            stop_reason = f"Active search warranted (max EI: {max_ei:.2f} cycles, max std: {max_unc:.2f})."

        return ControllerDecision(
            chosen_method=chosen_method,
            beta=beta_t,
            xi=self.base_xi,
            exploration_score=expl_score,
            exploitation_score=explt_score,
            controller_reason=reason,
            should_stop=should_stop,
            stop_reason=stop_reason,
            max_expected_improvement=max_ei,
            max_posterior_uncertainty=max_unc,
            mean_posterior_uncertainty=mean_unc,
        )
