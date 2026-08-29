from __future__ import annotations

from typing import Any, Literal, Sequence
import numpy as np
from scipy.stats import norm

AcquisitionMethod = Literal[
    "greedy",
    "gp_ucb",
    "expected_improvement",
    "probability_of_improvement",
    "denoised_expected_improvement",
    "noisy_expected_improvement",
    "nei",
    "true_nei",
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


def denoised_expected_improvement_acquisition(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float | None = None,
    observed_posterior_means: np.ndarray | Sequence[float] | None = None,
    xi: float = 0.01,
    objective: str = "maximize",
) -> np.ndarray:
    """Computes Denoised-Incumbent Expected Improvement.

    Heuristic approximation: Evaluates analytic EI over the single posterior mean incumbent:
    f* = max_i mu(x_i_obs) (for maximization) or min_i mu(x_i_obs) (for minimization).
    Preserved for backward compatibility. For canonical joint-posterior NEI, use compute_true_mc_nei.
    """
    m = np.asarray(mean, dtype=float)

    if observed_posterior_means is not None and len(observed_posterior_means) > 0:
        obs_m = np.asarray(observed_posterior_means, dtype=float)
        incumbent = float(np.max(obs_m) if objective == "maximize" else np.min(obs_m))
    elif best_observed is not None:
        incumbent = float(best_observed)
    else:
        raise ValueError("Either best_observed or observed_posterior_means must be provided for Denoised EI")

    return expected_improvement_acquisition(
        mean=m,
        std=std,
        best_observed=incumbent,
        xi=xi,
        objective=objective,
    )


# Backward-compatible aliases
posterior_mean_incumbent_ei = denoised_expected_improvement_acquisition
noisy_expected_improvement_acquisition = denoised_expected_improvement_acquisition


def safe_cholesky(cov: np.ndarray, base_jitter: float = 1e-8, max_jitter: float = 1e-2) -> np.ndarray:
    """Computes Cholesky factor with adaptive diagonal jitter escalation and SVD fallback."""
    cov_reg = np.asarray(cov, dtype=float)
    if cov_reg.ndim != 2 or cov_reg.shape[0] != cov_reg.shape[1]:
        raise ValueError(f"Covariance matrix must be square 2D, got shape {cov_reg.shape}")

    n = cov_reg.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.array([[np.sqrt(max(cov_reg[0, 0], 1e-12))]], dtype=float)

    jitter = base_jitter
    while jitter <= max_jitter:
        try:
            return np.linalg.cholesky(cov_reg + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # Spectral fallback if Cholesky fails after max jitter
    evals, evecs = np.linalg.eigh((cov_reg + cov_reg.T) / 2.0)
    evals = np.maximum(evals, 1e-10)
    return evecs @ np.diag(np.sqrt(evals))


def compute_true_mc_nei(
    gp: Any,
    X_observed_scaled: np.ndarray,
    X_candidates_scaled: np.ndarray,
    n_fantasies: int = 256,
    xi: float = 0.01,
    objective: str = "maximize",
    seed: int = 42,
    base_jitter: float = 1e-8,
) -> np.ndarray:
    """Computes Canonical Joint-Posterior Monte Carlo Noisy Expected Improvement (True NEI).

    Mathematical Formulation:
    -------------------------
    Under noisy observations y_i = f(x_i) + eps_i, the true latent values f(X_obs) at previously
    evaluated points follow a joint Gaussian posterior distribution:
        f_obs ~ N(mu_obs, Sigma_obs)
    where Sigma_obs is the full posterior covariance matrix over all observed points.

    1. We draw K joint fantasy realizations of latent observed vectors:
        f_obs^(k) = mu_obs + L * z^(k), where L * L^T = Sigma_obs + jitter * I, z^(k) ~ N(0, I)
    2. For each fantasy k in {1, ..., K}, we compute the fantasy incumbent:
        f*^(k) = max_i f_obs,i^(k)  (or min for minimization)
    3. For each candidate x, we evaluate the Rao-Blackwellized expected improvement across all fantasy incumbents:
        alpha_NEI(x) = (1/K) * sum_{k=1}^K EI(mu(x), sigma(x), incumbent=f*^(k), xi=xi)

    This integrates over the full joint posterior uncertainty of the latent incumbent without
    overfitting to single noisy spikes or ignoring posterior covariances between observations.
    """
    rng = np.random.default_rng(seed)

    X_obs = np.asarray(X_observed_scaled, dtype=float)
    X_cand = np.asarray(X_candidates_scaled, dtype=float)

    if X_obs.ndim != 2 or X_cand.ndim != 2:
        raise ValueError("X_observed_scaled and X_candidates_scaled must be 2D arrays")

    n_obs = len(X_obs)
    n_cand = len(X_cand)

    if n_obs == 0 or n_cand == 0:
        return np.zeros(n_cand, dtype=float)

    # 1. Compute joint posterior mean and covariance over observed points
    mu_obs, cov_obs = gp.predict(X_obs, return_cov=True)
    mu_obs = np.asarray(mu_obs, dtype=float)
    cov_obs = np.asarray(cov_obs, dtype=float)

    # 2. Factorize joint covariance
    L_obs = safe_cholesky(cov_obs, base_jitter=base_jitter)

    # 3. Draw K joint fantasy samples: shape (n_fantasies, n_obs)
    std_normals = rng.standard_normal(size=(n_fantasies, n_obs))
    f_obs_fantasies = mu_obs + (std_normals @ L_obs.T)

    # 4. Compute fantasy incumbents: shape (n_fantasies,)
    if objective == "maximize":
        fantasy_incumbents = np.max(f_obs_fantasies, axis=1)
    else:
        fantasy_incumbents = np.min(f_obs_fantasies, axis=1)

    # 5. Predict candidate posterior marginals
    cand_mean, cand_std = gp.predict(X_cand, return_std=True)
    cand_mean = np.asarray(cand_mean, dtype=float)
    cand_std = np.asarray(cand_std, dtype=float)

    # 6. Evaluate Rao-Blackwellized EI across all fantasy incumbents: shape (n_fantasies, n_cand)
    fantasy_ei_matrix = np.zeros((n_fantasies, n_cand), dtype=float)
    for k in range(n_fantasies):
        fantasy_ei_matrix[k] = expected_improvement_acquisition(
            mean=cand_mean,
            std=cand_std,
            best_observed=float(fantasy_incumbents[k]),
            xi=xi,
            objective=objective,
        )

    # 7. Average across fantasies
    nei_scores = np.mean(fantasy_ei_matrix, axis=0)
    return np.maximum(0.0, nei_scores)


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
    gp: Any | None = None,
    X_observed_scaled: np.ndarray | None = None,
    X_candidates_scaled: np.ndarray | None = None,
    n_fantasies: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Universal acquisition computation function supporting both analytic and joint-posterior formulations."""
    normalized_method = method.lower().strip()

    if normalized_method in {"greedy", "posterior_mean", "mean"}:
        return greedy_acquisition(mean, objective=objective)

    elif normalized_method in {"gp_ucb", "ucb", "upper_confidence_bound"}:
        return ucb_acquisition(mean, std, beta=beta, objective=objective)

    elif normalized_method in {"expected_improvement", "ei"}:
        return expected_improvement_acquisition(mean, std, best_observed=best_observed, xi=xi, objective=objective)

    elif normalized_method in {"denoised_expected_improvement", "denoised_ei", "posterior_mean_incumbent_ei"}:
        return denoised_expected_improvement_acquisition(
            mean=mean,
            std=std,
            best_observed=best_observed,
            observed_posterior_means=observed_posterior_means,
            xi=xi,
            objective=objective,
        )

    elif normalized_method in {"noisy_expected_improvement", "nei", "true_nei", "turbo_nei"}:
        # If GP and scaled design matrices are provided, compute canonical True Joint-Posterior MC NEI
        if gp is not None and X_observed_scaled is not None and X_candidates_scaled is not None:
            return compute_true_mc_nei(
                gp=gp,
                X_observed_scaled=X_observed_scaled,
                X_candidates_scaled=X_candidates_scaled,
                n_fantasies=n_fantasies,
                xi=xi,
                objective=objective,
                seed=seed,
            )
        # Fallback to denoised incumbent EI if model objects are not passed
        return denoised_expected_improvement_acquisition(
            mean=mean,
            std=std,
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
            "Supported methods: 'greedy', 'gp_ucb', 'expected_improvement', 'noisy_expected_improvement' (or 'nei', 'true_nei'), 'denoised_ei', 'probability_of_improvement'."
        )
