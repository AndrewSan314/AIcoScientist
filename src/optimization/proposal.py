from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateProposal:
    """Represents a proposed candidate for experimental execution or characterization."""

    candidate_id: str
    design_variables: dict[str, Any]
    predicted_mean: float
    predicted_std: float
    acquisition_name: str
    acquisition_value: float
    backend_name: str
    backend_version: str
    seed: int | None = None
    reason_code: str = "BALANCED_EXPLORATION_EXPLOITATION"
    recommendation_reason: str = ""
    distance_to_nearest_observed: float = 0.0
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_performance(self) -> float:
        """Alias for predicted_mean for compatibility with scientific ledger/rationale."""
        return self.predicted_mean

    @property
    def prediction_uncertainty(self) -> float:
        """Alias for predicted_std for compatibility with scientific ledger/rationale."""
        return self.predicted_std

    @property
    def acquisition_score(self) -> float:
        """Alias for acquisition_value for compatibility."""
        return self.acquisition_value

    @property
    def acquisition_method(self) -> str:
        """Alias for acquisition_name for compatibility."""
        return self.acquisition_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "design_variables": dict(self.design_variables),
            **self.design_variables,
            "predicted_mean": float(self.predicted_mean),
            "predicted_std": float(self.predicted_std),
            "predicted_performance": float(self.predicted_mean),
            "prediction_uncertainty": float(self.predicted_std),
            "acquisition_name": self.acquisition_name,
            "acquisition_value": float(self.acquisition_value),
            "acquisition_method": self.acquisition_name,
            "acquisition_score": float(self.acquisition_value),
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "seed": self.seed,
            "reason_code": self.reason_code,
            "recommendation_reason": self.recommendation_reason,
            "distance_to_nearest_observed": float(self.distance_to_nearest_observed),
            "step": int(self.step),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateProposal:
        d = dict(data)
        cand_id = str(d["candidate_id"])
        design_vars = dict(d.get("design_variables", {}))
        if not design_vars:
            std_keys = {
                "candidate_id",
                "design_variables",
                "predicted_mean",
                "predicted_std",
                "predicted_performance",
                "prediction_uncertainty",
                "acquisition_name",
                "acquisition_value",
                "acquisition_method",
                "acquisition_score",
                "backend_name",
                "backend_version",
                "seed",
                "reason_code",
                "recommendation_reason",
                "distance_to_nearest_observed",
                "step",
                "metadata",
            }
            design_vars = {k: v for k, v in d.items() if k not in std_keys}

        pred_mean = float(d.get("predicted_mean", d.get("predicted_performance", 0.0)))
        pred_std = float(d.get("predicted_std", d.get("prediction_uncertainty", 0.0)))
        acq_name = str(d.get("acquisition_name", d.get("acquisition_method", "expected_improvement")))
        acq_val = float(d.get("acquisition_value", d.get("acquisition_score", 0.0)))
        b_name = str(d.get("backend_name", "botorch"))
        b_ver = str(d.get("backend_version", "unknown"))

        return cls(
            candidate_id=cand_id,
            design_variables=design_vars,
            predicted_mean=pred_mean,
            predicted_std=pred_std,
            acquisition_name=acq_name,
            acquisition_value=acq_val,
            backend_name=b_name,
            backend_version=b_ver,
            seed=int(d["seed"]) if d.get("seed") is not None else None,
            reason_code=str(d.get("reason_code", "BALANCED_EXPLORATION_EXPLOITATION")),
            recommendation_reason=str(d.get("recommendation_reason", "")),
            distance_to_nearest_observed=float(d.get("distance_to_nearest_observed", 0.0)),
            step=int(d.get("step", 0)),
            metadata=dict(d.get("metadata", {})),
        )


# Backward-compatible alias for existing ledger and science coordinator records
ExperimentProposal = CandidateProposal
