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
    Preserved for explicit comparison. For canonical joint-posterior NEI, use compute_true_mc_nei.
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
denoised_ei = denoised_expected_improvement_acquisition


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
    candidate_chunk_size: int = 128,
    base_jitter: float = 1e-8,
) -> np.ndarray:
    """Computes Canonical Joint Observed+Candidate Monte Carlo Noisy Expected Improvement (True NEI).

    Mathematical Formulation:
    -------------------------
    Under noisy observations y_i = f(x_i) + eps_i, the true latent values at observed points
    and candidate points follow a full joint Gaussian Process posterior:
        [f(X_obs), f(X_cand)] ~ N(mu_joint, Sigma_joint)
    where Sigma_joint incorporates the full posterior cross-covariance between candidate
    locations and previously evaluated points.

    Algorithm:
    1. Evaluates candidates in memory-efficient chunks (e.g. 128 candidates per chunk).
    2. For each chunk, constructs the joint design matrix:
           X_joint = [X_obs; X_chunk]
    3. Predicts joint posterior mean mu_joint and full joint covariance Sigma_joint.
    4. Computes safe Cholesky factor L_joint (with adaptive jitter escalation).
    5. Draws K correlated joint fantasy samples:
           F_joint = mu_joint + Z * L_joint^T, where Z ~ N(0, I)
    6. Extracts correlated fantasy incumbents f*^(k) = max_i f_obs,i^(k) (or min)
       and candidate fantasy draws f_cand^(k).
    7. Evaluates exact Monte Carlo improvement per fantasy:
           I^(k)(x) = max(0, f_cand^(k)(x) - f*^(k) - xi)  [for maximization]
           I^(k)(x) = max(0, f*^(k) - f_cand^(k)(x) - xi)  [for minimization]
    8. Averages over K fantasies:
           alpha_NEI(x) = (1 / K) * sum_{k=1}^K I^(k)(x)
    """
    X_obs = np.asarray(X_observed_scaled, dtype=float)
    X_cand = np.asarray(X_candidates_scaled, dtype=float)

    if X_obs.ndim != 2 or X_cand.ndim != 2:
        raise ValueError("X_observed_scaled and X_candidates_scaled must be 2D arrays")

    n_obs = len(X_obs)
    n_cand = len(X_cand)

    if n_obs == 0 or n_cand == 0:
        return np.zeros(n_cand, dtype=float)

    all_scores = np.zeros(n_cand, dtype=float)
    chunk_size = max(1, int(candidate_chunk_size))

    for chunk_start in range(0, n_cand, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_cand)
        X_chunk = X_cand[chunk_start:chunk_end]
        n_chunk = len(X_chunk)

        # 1. Joint design matrix: (n_obs + n_chunk, d)
        X_joint = np.vstack([X_obs, X_chunk])

        # 2. Joint posterior mean and full joint covariance
        mu_joint, cov_joint = gp.predict(X_joint, return_cov=True)
        mu_joint = np.asarray(mu_joint, dtype=float)
        cov_joint = np.asarray(cov_joint, dtype=float)

        # 3. Factorize joint covariance
        L_joint = safe_cholesky(cov_joint, base_jitter=base_jitter)

        # 4. Draw K correlated joint fantasy samples: (n_fantasies, n_obs + n_chunk)
        rng_chunk = np.random.default_rng(seed + chunk_start * 1000 + 7)
        Z = rng_chunk.standard_normal(size=(n_fantasies, n_obs + n_chunk))
        F_joint = mu_joint + (Z @ L_joint.T)

        # 5. Extract observed and candidate fantasy slices
        F_obs = F_joint[:, :n_obs]
        F_cand = F_joint[:, n_obs:]

        # 6. Compute fantasy incumbents and joint improvements
        if objective == "maximize":
            fantasy_incumbents = np.max(F_obs, axis=1, keepdims=True)  # shape: (n_fantasies, 1)
            improvements = np.maximum(0.0, F_cand - fantasy_incumbents - xi)  # shape: (n_fantasies, n_chunk)
        else:
            fantasy_incumbents = np.min(F_obs, axis=1, keepdims=True)
            improvements = np.maximum(0.0, fantasy_incumbents - F_cand - xi)

        # 7. Average across fantasies for this chunk
        chunk_scores = np.mean(improvements, axis=0)
        all_scores[chunk_start:chunk_end] = chunk_scores

    return np.maximum(0.0, all_scores)


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
    candidate_chunk_size: int = 128,
) -> np.ndarray:
    """Universal acquisition computation function supporting both analytic and joint-posterior formulations.

    Strict Contract:
    - 'nei', 'true_nei', 'noisy_expected_improvement', 'turbo_nei' REQUIRE gp, X_observed_scaled, X_candidates_scaled.
      If missing, raises ValueError (zero silent fallbacks).
    - 'denoised_expected_improvement' / 'denoised_ei' uses single posterior mean incumbent heuristic.
    """
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
        if gp is None or X_observed_scaled is None or X_candidates_scaled is None:
            raise ValueError(
                f"Acquisition method {method!r} requires 'gp', 'X_observed_scaled', and 'X_candidates_scaled' "
                "to evaluate joint candidate-observed posterior covariance. "
                "Silent fallback is disabled. For heuristic denoised-incumbent EI without joint covariance, "
                "explicitly specify method='denoised_expected_improvement'."
            )
        return compute_true_mc_nei(
            gp=gp,
            X_observed_scaled=X_observed_scaled,
            X_candidates_scaled=X_candidates_scaled,
            n_fantasies=n_fantasies,
            xi=xi,
            objective=objective,
            seed=seed,
            candidate_chunk_size=candidate_chunk_size,
        )

    elif normalized_method in {"probability_of_improvement", "pi"}:
        return probability_of_improvement_acquisition(mean, std, best_observed=best_observed, xi=xi, objective=objective)

    else:
        raise ValueError(
            f"Unknown acquisition method {method!r}. "
            "Supported methods: 'greedy', 'gp_ucb', 'expected_improvement', 'noisy_expected_improvement' (or 'nei', 'true_nei'), 'denoised_expected_improvement', 'probability_of_improvement'."
        )
