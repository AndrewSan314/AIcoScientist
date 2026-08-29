from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.stats import norm

AcquisitionMethod = Literal[
    "greedy",
    "gp_ucb",
    "expected_improvement",
    "probability_of_improvement",
    "noisy_expected_improvement",
    "nei",
    "turbo_nei",
]


def greedy_acquisition(mean: np.ndarray, objective: str = "maximize") -> np.ndarray:
    """Computes exploitation acquisition score (posterior mean)."""
    m = np.asarray(mean, dtype=float)
    if objective == "minimize":
        return -m
    return m


def ucb_acquisition(
    mean: np.ndarray,
    std: np.ndarray,
    beta: float = 1.0,
    objective: str = "maximize",
) -> np.ndarray:
    """Computes Upper Confidence Bound (GP-UCB) score.

    For maximization: mu(x) + beta * sigma(x)
    For minimization: -(mu(x) - beta * sigma(x))
    """
    m = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float)
    if objective == "minimize":
        return -(m - beta * s)
    return m + beta * s


# Backward-compatible alias
ucb = ucb_acquisition


def expected_improvement_acquisition(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
    objective: str = "maximize",
) -> np.ndarray:
    """Computes numerically stable Expected Improvement (EI).

    EI(x) = (mu(x) - y^* - xi) * Phi(gamma) + sigma(x) * phi(gamma)
    where gamma = (mu(x) - y^* - xi) / sigma(x).
    """
    m = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float)

    if objective == "minimize":
        # For minimization, improvement is best_observed - mu(x) - xi
        improvement = best_observed - m - xi
    else:
        improvement = m - best_observed - xi

    ei = np.zeros_like(m)
    mask = s > 1e-9

    if np.any(mask):
        gamma = improvement[mask] / s[mask]
        cdf_val = norm.cdf(gamma)
        pdf_val = norm.pdf(gamma)
        ei[mask] = improvement[mask] * cdf_val + s[mask] * pdf_val

    # For zero/near-zero uncertainty regions
    zero_mask = ~mask
    if np.any(zero_mask):
        ei[zero_mask] = np.maximum(0.0, improvement[zero_mask])

    return np.maximum(0.0, ei)


def noisy_expected_improvement_acquisition(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float | None = None,
    observed_posterior_means: np.ndarray | Sequence[float] | None = None,
    xi: float = 0.01,
    objective: str = "maximize",
) -> np.ndarray:
    """Computes Noisy Expected Improvement (NEI) under uncertain/noisy observations.

    Under noisy observations y_i = f(x_i) + eps_i, the true latent incumbent f* is uncertain.
    NEI computes expected improvement over the denoised posterior incumbent:
    f* = max_i mu(x_i_obs) (for maximization) or min_i mu(x_i_obs) (for minimization).

    If observed_posterior_means is provided, the denoised incumbent is derived from
    the GP posterior predictions at observed points. Otherwise, best_observed is used.
    """
    m = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float)

    if observed_posterior_means is not None and len(observed_posterior_means) > 0:
        obs_m = np.asarray(observed_posterior_means, dtype=float)
        incumbent = float(np.max(obs_m) if objective == "maximize" else np.min(obs_m))
    elif best_observed is not None:
        incumbent = float(best_observed)
    else:
        raise ValueError("Either best_observed or observed_posterior_means must be provided for NEI")

    if objective == "minimize":
        improvement = incumbent - m - xi
    else:
        improvement = m - incumbent - xi

    nei = np.zeros_like(m)
    mask = s > 1e-9

    if np.any(mask):
        gamma = improvement[mask] / s[mask]
        cdf_val = norm.cdf(gamma)
        pdf_val = norm.pdf(gamma)
        nei[mask] = improvement[mask] * cdf_val + s[mask] * pdf_val

    zero_mask = ~mask
    if np.any(zero_mask):
        nei[zero_mask] = np.maximum(0.0, improvement[zero_mask])

    return np.maximum(0.0, nei)


# Alias
nei_acquisition = noisy_expected_improvement_acquisition


def mc_noisy_expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    observed_means: np.ndarray | Sequence[float],
    observed_stds: np.ndarray | Sequence[float] | None = None,
    n_mc_samples: int = 500,
    xi: float = 0.01,
    objective: str = "maximize",
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo Noisy Expected Improvement drawing joint posterior samples over the incumbent distribution."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float)
    obs_m = np.asarray(observed_means, dtype=float)

    if observed_stds is not None:
        obs_s = np.asarray(observed_stds, dtype=float)
        # Draw samples of observed points: (n_mc_samples, n_obs)
        obs_samples = rng.normal(obs_m, np.maximum(1e-6, obs_s), size=(n_mc_samples, len(obs_m)))
    else:
        obs_samples = np.tile(obs_m, (n_mc_samples, 1))

    # Incumbent sample per MC replicate
    if objective == "maximize":
        f_star_samples = np.max(obs_samples, axis=1)  # shape: (n_mc_samples,)
    else:
        f_star_samples = np.min(obs_samples, axis=1)

    # Draw candidate samples: (n_mc_samples, n_candidates)
    cand_samples = rng.normal(m, np.maximum(1e-6, s), size=(n_mc_samples, len(m)))

    if objective == "maximize":
        improvements = np.maximum(0.0, cand_samples - f_star_samples[:, None] - xi)
    else:
        improvements = np.maximum(0.0, f_star_samples[:, None] - cand_samples - xi)

    return np.mean(improvements, axis=0)


def probability_of_improvement_acquisition(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
    objective: str = "maximize",
) -> np.ndarray:
    """Computes Probability of Improvement (PI): Phi(gamma)."""
    m = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float)

    if objective == "minimize":
        improvement = best_observed - m - xi
    else:
        improvement = m - best_observed - xi

    pi = np.zeros_like(m)
    mask = s > 1e-9

    if np.any(mask):
        gamma = improvement[mask] / s[mask]
        pi[mask] = norm.cdf(gamma)

    zero_mask = ~mask
    if np.any(zero_mask):
        pi[zero_mask] = np.where(improvement[zero_mask] > 0, 1.0, 0.0)

    return np.clip(pi, 0.0, 1.0)


def compute_acquisition(
    method: AcquisitionMethod | str,
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    beta: float = 1.0,
    xi: float = 0.01,
    objective: str = "maximize",
    observed_posterior_means: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Universal acquisition computation function supporting multiple exploration-exploitation formulations."""
    normalized_method = method.lower().strip()
    if normalized_method in {"greedy", "posterior_mean", "mean"}:
        return greedy_acquisition(mean, objective=objective)
    elif normalized_method in {"gp_ucb", "ucb", "upper_confidence_bound"}:
        return ucb_acquisition(mean, std, beta=beta, objective=objective)
    elif normalized_method in {"expected_improvement", "ei"}:
        return expected_improvement_acquisition(mean, std, best_observed=best_observed, xi=xi, objective=objective)
    elif normalized_method in {"noisy_expected_improvement", "nei", "turbo_nei"}:
        return noisy_expected_improvement_acquisition(
            mean,
            std,
            best_observed=best_observed,
            observed_posterior_means=observed_posterior_means,
            xi=xi,
            objective=objective,
        )
    elif normalized_method in {"probability_of_improvement", "pi"}:
        return probability_of_improvement_acquisition(mean, std, best_observed=best_observed, xi=xi, objective=objective)
    else:
        raise ValueError(
            f"Unknown acquisition method {method!r}. "
            "Supported methods: 'greedy', 'gp_ucb', 'expected_improvement', 'noisy_expected_improvement' (or 'nei'), 'probability_of_improvement'."
        )
