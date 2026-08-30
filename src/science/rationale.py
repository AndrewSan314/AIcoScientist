from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class ScientificRationale:
    """Structured, deterministic scientific rationale object explaining an experimental proposal."""

    experiment_id: str
    candidate_id: str
    design_changes: dict[str, Any]
    predicted_performance_mean: float
    predicted_performance_latent_std: float
    predicted_characterization: dict[str, dict[str, float]]
    acquisition_method: str
    acquisition_score: float
    exploration_component: float | None = None
    exploitation_component: float | None = None
    nearest_observed_experiment_id: str | None = None
    distance_to_nearest_observed: float | None = None
    comparison_to_incumbent: dict[str, Any] = field(default_factory=dict)
    model_disagreement: float = 0.0
    model_disagreement_flag: str | None = None
    expected_learning_value: float = 0.0
    learning_value_components: dict[str, float] = field(default_factory=dict)
    uncertainty_sources: dict[str, float] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    reason_code: str = "BALANCED_EXPLORATION_EXPLOITATION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render_text(self) -> str:
        """Renders deterministic human-readable text grounded strictly in model predictions with cautious scientific phrasing."""
        lines = [
            f"=== Scientific Rationale for Proposal [{self.experiment_id} | {self.candidate_id}] ===",
            "",
            "1. WHAT SHOULD WE TEST? (Candidate Process Parameters)",
        ]
        for k, v in self.design_changes.items():
            if isinstance(v, float):
                lines.append(f"   - {k}: {v:.4f}")
            else:
                lines.append(f"   - {k}: {v}")

        lines.extend([
            "",
            "2. PREDICTED PERFORMANCE (Two-Stage Model with Uncertainty Propagation):",
            f"   - Expected Performance Target: {self.predicted_performance_mean:.2f} ± {self.predicted_performance_latent_std:.2f} (latent 1-sigma)",
        ])
        if self.model_disagreement_flag:
            lines.append(f"   - Model Disagreement: {self.model_disagreement:.2f} [{self.model_disagreement_flag}]")
        else:
            lines.append(f"   - Model Disagreement (Direct vs Two-Stage): {self.model_disagreement:.2f}")

        lines.extend([
            "",
            "3. EXPECTED STRUCTURE / CHARACTERIZATION (Stage A Estimates):",
        ])
        for char_name, char_stats in self.predicted_characterization.items():
            m = char_stats.get("mean", float("nan"))
            s = char_stats.get("latent_std", float("nan"))
            lines.append(f"   - {char_name}: {m:.4f} ± {s:.4f}")

        lines.extend([
            "",
            "4. WHY THIS EXPERIMENT? (Optimizer Strategy & Acquisition Rationale):",
            f"   - Strategy: {self.acquisition_method}",
            f"   - Acquisition Score: {self.acquisition_score:.4f}",
            f"   - Reason Code: {self.reason_code}",
        ])
        if self.nearest_observed_experiment_id:
            dist_str = f"{self.distance_to_nearest_observed:.4f}" if self.distance_to_nearest_observed is not None else "N/A"
            lines.append(f"   - Nearest Observed Experiment: {self.nearest_observed_experiment_id} (normalized distance = {dist_str})")

        lines.extend([
            "",
            "5. WHAT WILL WE LEARN? (Exploratory Information Value):",
            f"   - Overall Learning Value Score: {self.expected_learning_value:.4f}",
        ])
        for comp_name, comp_val in self.learning_value_components.items():
            lines.append(f"     * {comp_name}: {comp_val:.4f}")

        if self.caveats:
            lines.extend([
                "",
                "6. SCIENTIFIC CAVEATS & LIMITATIONS:",
            ])
            for cav in self.caveats:
                lines.append(f"   ! {cav}")

        return "\n".join(lines)


def generate_scientific_rationale(
    experiment_id: str,
    candidate_id: str,
    candidate_process: Mapping[str, Any],
    direct_prediction: tuple[float, float],
    two_stage_prediction: Any,
    acquisition_method: str,
    acquisition_score: float,
    observed_history: pd.DataFrame | None = None,
    process_features: Sequence[str] | None = None,
    incumbent_target: float | None = None,
) -> ScientificRationale:
    """Constructs a deterministic, structured ScientificRationale object from models and optimizer state."""
    dir_mean, dir_std = direct_prediction
    e2e_mean = float(two_stage_prediction.performance_mean[0]) if hasattr(two_stage_prediction.performance_mean, "__len__") else float(two_stage_prediction.performance_mean)
    e2e_std = float(two_stage_prediction.performance_latent_std[0]) if hasattr(two_stage_prediction.performance_latent_std, "__len__") else float(two_stage_prediction.performance_latent_std)

    # 1. Model disagreement
    disagreement = abs(dir_mean - e2e_mean)
    pooled_std = np.sqrt(dir_std**2 + e2e_std**2)
    disagreement_flag = "MODEL_DISAGREEMENT_HIGH" if (pooled_std > 1e-6 and disagreement > 2.0 * pooled_std) else None

    # 2. Stage A characterization summary
    char_summary: dict[str, dict[str, float]] = {}
    char_prop_var = 0.0
    for char_col, stats_dict in two_stage_prediction.characterization_predictions.items():
        m_val = float(stats_dict["mean"][0]) if hasattr(stats_dict["mean"], "__len__") else float(stats_dict["mean"])
        s_val = float(stats_dict["latent_std"][0]) if hasattr(stats_dict["latent_std"], "__len__") else float(stats_dict["latent_std"])
        obs_val = float(stats_dict["observation_std"][0]) if hasattr(stats_dict["observation_std"], "__len__") else float(stats_dict["observation_std"])
        char_summary[char_col] = {
            "mean": m_val,
            "latent_std": s_val,
            "observation_std": obs_val,
        }

    if hasattr(two_stage_prediction, "characterization_propagation_variance"):
        cpv = two_stage_prediction.characterization_propagation_variance
        char_prop_var = float(cpv[0]) if hasattr(cpv, "__len__") else float(cpv)

    # 3. Nearest neighbor and distance
    nearest_exp_id: str | None = None
    min_dist: float | None = None

    if observed_history is not None and not observed_history.empty and process_features:
        feat_cols = list(process_features)
        present_feats = [c for c in feat_cols if c in observed_history.columns]
        if present_feats:
            obs_mat = observed_history[present_feats].to_numpy(dtype=float)
            cand_mat = np.array([[float(candidate_process[k]) for k in present_feats]], dtype=float)

            ranges = np.ptp(obs_mat, axis=0)
            ranges[ranges == 0.0] = 1.0

            diffs = (obs_mat - cand_mat) / ranges
            dists = np.sqrt(np.sum(diffs**2, axis=1))

            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])
            if "experiment_id" in observed_history.columns:
                nearest_exp_id = str(observed_history["experiment_id"].iloc[min_idx])

    # 4. Learning value components
    # Uncertainty component: normalized by (1 + e2e_std)
    u_comp = float(e2e_std / (1.0 + e2e_std))
    # Novelty component: min_dist normalized
    nov_comp = float(min_dist / (1.0 + min_dist)) if min_dist is not None else 0.5
    # Disagreement component
    dis_comp = float(disagreement / (1.0 + pooled_std))

    total_learning_val = float(0.4 * u_comp + 0.4 * nov_comp + 0.2 * dis_comp)

    # 5. Caveats list
    caveats = [
        "Predicted characterization values are surrogate model estimates, not physically measured properties.",
        "The model assumes independent Stage-A characterization errors; potential cross-channel physical correlations are unmodeled.",
    ]
    if disagreement_flag:
        caveats.append(
            "High model disagreement detected between the direct process->performance model and the two-stage model."
        )

    # 6. Reason code
    if min_dist is not None and min_dist > 1.0:
        reason_code = "REGION_EXPLORATION"
    elif incumbent_target is not None and e2e_mean > incumbent_target:
        reason_code = "HIGH_POTENTIAL_EXPLOITATION"
    else:
        reason_code = "BALANCED_EXPLORATION_EXPLOITATION"

    return ScientificRationale(
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        design_changes=dict(candidate_process),
        predicted_performance_mean=e2e_mean,
        predicted_performance_latent_std=e2e_std,
        predicted_characterization=char_summary,
        acquisition_method=acquisition_method,
        acquisition_score=float(acquisition_score),
        nearest_observed_experiment_id=nearest_exp_id,
        distance_to_nearest_observed=min_dist,
        comparison_to_incumbent={"incumbent_target": incumbent_target},
        model_disagreement=disagreement,
        model_disagreement_flag=disagreement_flag,
        expected_learning_value=total_learning_val,
        learning_value_components={
            "uncertainty_component": u_comp,
            "novelty_component": nov_comp,
            "disagreement_component": dis_comp,
        },
        uncertainty_sources={
            "characterization_propagation_variance": char_prop_var,
            "performance_model_variance": (
                float(two_stage_prediction.performance_model_variance[0])
                if hasattr(two_stage_prediction, "performance_model_variance")
                else 0.0
            ),
        },
        caveats=caveats,
        reason_code=reason_code,
    )
