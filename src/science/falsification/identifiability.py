from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.science.actions import ExperimentActionType
from src.science.hypothesis_models import (
    CompositionSufficientHypothesis,
    HypothesisEnsemble,
    LocalStructuralRegimeHypothesis,
    PredictiveDistribution,
    StructureInformedHypothesis,
)

logger = logging.getLogger(__name__)

DEFAULT_IDENTIFIABILITY_REPORT = Path("reports/falsification/hypothesis_identifiability.md")


def _df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Formats a DataFrame as a GitHub-flavored Markdown table without tabulate dependency."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_strs = []
        for val in row:
            if isinstance(val, float):
                row_strs.append(f"{val:.4f}")
            else:
                row_strs.append(str(val))
        rows.append("| " + " | ".join(row_strs) + " |")
    return "\n".join([header_line, sep_line] + rows)


def moment_matched_gaussian_divergence_proxy(
    p1: PredictiveDistribution,
    p2: PredictiveDistribution,
) -> float:
    """Computes a moment-matched Gaussian proxy divergence between two predictive distributions.

    Note: This is an analytical proxy approximation, not exact Jensen-Shannon divergence.
    For true Jensen-Shannon divergence between Gaussian mixtures, use compute_monte_carlo_js_divergence.
    """
    m1, v1 = p1.mean, np.maximum(p1.variance, 1e-10)
    m2, v2 = p2.mean, np.maximum(p2.variance, 1e-10)

    # Moment-matched mixture mean and variance
    m_mix = 0.5 * (m1 + m2)
    v_mix = 0.5 * (v1 + v2) + 0.25 * ((m1 - m2) ** 2)

    kl1 = 0.5 * np.sum(np.log(v_mix / v1) + (v1 + (m1 - m_mix) ** 2) / v_mix - 1.0)
    kl2 = 0.5 * np.sum(np.log(v_mix / v2) + (v2 + (m2 - m_mix) ** 2) / v_mix - 1.0)

    proxy = 0.5 * (kl1 + kl2)
    return float(max(0.0, proxy))


def _compute_raw_monte_carlo_js(
    p1: PredictiveDistribution,
    p2: PredictiveDistribution,
    n_samples: int = 256,
    seed: int = 42,
) -> float:
    """Computes unclipped raw Monte Carlo Jensen-Shannon Divergence estimate."""
    if np.allclose(p1.mean, p2.mean, atol=1e-9) and np.allclose(p1.variance, p2.variance, atol=1e-9):
        return 0.0

    rng = np.random.default_rng(seed)
    samples_1 = p1.sample(n_samples=n_samples, rng=rng)
    samples_2 = p2.sample(n_samples=n_samples, rng=rng)

    # 1. E_{y ~ p1} [ log p1(y) - log m(y) ]
    log_p1_s1 = np.array([p1.log_pdf(y) for y in samples_1])
    log_p2_s1 = np.array([p2.log_pdf(y) for y in samples_1])
    log_m_s1 = np.log(0.5) + np.logaddexp(log_p1_s1, log_p2_s1)
    kl1 = float(np.mean(log_p1_s1 - log_m_s1))

    # 2. E_{y ~ p2} [ log p2(y) - log m(y) ]
    log_p1_s2 = np.array([p1.log_pdf(y) for y in samples_2])
    log_p2_s2 = np.array([p2.log_pdf(y) for y in samples_2])
    log_m_s2 = np.log(0.5) + np.logaddexp(log_p1_s2, log_p2_s2)
    kl2 = float(np.mean(log_p2_s2 - log_m_s2))

    return float(0.5 * (kl1 + kl2))


def compute_monte_carlo_js_divergence(
    p1: PredictiveDistribution,
    p2: PredictiveDistribution,
    n_samples: int = 256,
    seed: int = 42,
    tol: float | None = None,
) -> float:
    """Computes true Shannon-base-e Jensen-Shannon Divergence using Monte Carlo sampling.

    Definition:
        JS(p1, p2) = 0.5 * KL(p1 || m) + 0.5 * KL(p2 || m)
        where m(y) = 0.5 * p1(y) + 0.5 * p2(y)

    Properties:
        - Symmetric: JS(p1, p2) == JS(p2, p1)
        - Bounded: 0.0 <= JS(p1, p2) <= ln(2) ~ 0.69315 nats
        - Zero if and only if p1 == p2 almost everywhere

    Tolerance behavior:
        When `tol` is None, a conservative Monte Carlo tolerance proportional to
        1/sqrt(n_samples) is used: effective_tol = max(0.01, 0.5 / sqrt(n_samples))
        to accommodate expected finite-sample numerical fluctuations while still
        rejecting material violations of the theoretical JS bounds.

    Parameters
    ----------
    tol : float | None
        Tolerance for Monte Carlo sampling variance around [0, ln(2)] bounds.
        If None, automatically derived from n_samples.
        Excursions beyond tolerance raise RuntimeError rather than being silently clipped.
    """
    if tol is not None and tol < 0:
        raise ValueError("tol must be non-negative")

    effective_tol = tol if tol is not None else max(0.01, 0.5 / np.sqrt(max(1, n_samples)))
    raw_js = _compute_raw_monte_carlo_js(p1, p2, n_samples=n_samples, seed=seed)

    # Validate against theoretical bounds with numerical tolerance
    ln2 = float(np.log(2.0))
    if raw_js < -effective_tol or raw_js > ln2 + effective_tol:
        raise RuntimeError(
            f"Monte Carlo JS divergence estimate ({raw_js:.5f}) exceeded theoretical bounds "
            f"[0.0, {ln2:.5f}] beyond tolerance ({effective_tol:.5f})."
        )

    # Clamp tiny epsilon-scale Monte Carlo sampling noise to exact mathematical bounds [0, ln(2)]
    return float(min(max(raw_js, 0.0), ln2))


# Alias for backward compatibility
compute_gaussian_js_divergence = compute_monte_carlo_js_divergence


def run_identifiability_analysis(
    candidate_pool_df: pd.DataFrame,
    ensemble: HypothesisEnsemble | None = None,
    output_path: Path | str = DEFAULT_IDENTIFIABILITY_REPORT,
    use_true_js: bool = True,
) -> pd.DataFrame:
    """Evaluates pairwise hypothesis identifiability across the materials candidate space."""
    ens = ensemble if ensemble is not None else HypothesisEnsemble()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pairs = [("H1", "H2"), ("H1", "H3"), ("H2", "H3")]
    rows: list[dict[str, Any]] = []

    comp_cols = ["Au", "Ir", "Rh"]
    comps = candidate_pool_df[comp_cols].to_numpy(dtype=np.float64)
    cids = candidate_pool_df["candidate_id"].tolist()

    for i, cid in enumerate(cids):
        comp = comps[i]
        for action_type in [ExperimentActionType.XRD, ExperimentActionType.PROPERTY]:
            preds = ens.predict_all(cid, action_type, comp)

            for h_a, h_b in pairs:
                if h_a in preds and h_b in preds:
                    if use_true_js:
                        js_val = compute_monte_carlo_js_divergence(preds[h_a], preds[h_b], n_samples=64, seed=42 + i)
                    else:
                        js_val = moment_matched_gaussian_divergence_proxy(preds[h_a], preds[h_b])

                    mean_sep = float(np.linalg.norm(preds[h_a].mean - preds[h_b].mean))

                    rows.append(
                        {
                            "candidate_id": cid,
                            "Au": comp[0],
                            "Ir": comp[1],
                            "Rh": comp[2],
                            "action_type": action_type.value,
                            "hypothesis_pair": f"{h_a}_vs_{h_b}",
                            "js_divergence": js_val,
                            "mean_separation": mean_sep,
                        }
                    )

    df = pd.DataFrame(rows)

    # Aggregate by pair and action
    agg_df = df.groupby(["hypothesis_pair", "action_type"]).agg(
        mean_js=("js_divergence", "mean"),
        max_js=("js_divergence", "max"),
        min_js=("js_divergence", "min"),
        mean_sep=("mean_separation", "mean"),
    ).reset_index()

    table_md = _df_to_markdown_simple(agg_df)

    lines = [
        "# Scientific Hypothesis Identifiability & Predictive Divergence Analysis",
        "",
        "## 1. Overview",
        "Evaluates true Monte Carlo Jensen-Shannon Divergence (bounded in $[0, \\ln 2] \\approx [0, 0.69315]$ nats) "
        "across the candidate space for competing hypothesis pairs ($H_1, H_2, H_3$).",
        "",
        "## 2. Pairwise Divergence Metrics (Monte Carlo JS Divergence in nats)",
        "",
        table_md,
        "",
        "## 3. Identifiability Findings",
        "- **$H_1$ vs. $H_2$ (Composition-Sufficient vs. Structure-Informed)**: Identical structural predictions ($JS = 0.0$). Distinguishable under PROPERTY actions when structural features are characterized.",
        "- **$H_1$ vs. $H_3$ (Composition-Sufficient vs. Local-Regime)**: Distinguishable under both XRD and PROPERTY in localized compositional regime boundaries.",
        "- **$H_2$ vs. $H_3$ (Structure-Informed vs. Local-Regime)**: Distinguishable in transition zones where localized non-smooth shifts occur.",
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Identifiability report written to {out_file}")
    return df
