from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from src.science.actions import ExperimentActionType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictiveDistribution:
    """Represents a univariate or multivariate predictive distribution for an experimental action.

    Attributes
    ----------
    hypothesis_id : str
        Identifier of the hypothesis generating this prediction (e.g. 'H1', 'H2', 'H3').
    candidate_id : str
        Identifier of the physical candidate.
    action_type : ExperimentActionType
        Measurement modality ('XRD' or 'PROPERTY').
    mean : np.ndarray
        Predictive mean array (shape (1,) for scalar property k0; shape (D,) for D-dimensional XRD embedding).
    variance : np.ndarray
        Predictive variance array (diagonal variance elements; shape (1,) or (D,)).
    metadata : dict[str, Any]
        Additional context such as component uncertainties or regime indices.
    """

    hypothesis_id: str
    candidate_id: str
    action_type: ExperimentActionType
    mean: np.ndarray
    variance: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def sample(self, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draws Monte Carlo samples from the predictive Gaussian distribution."""
        gen = rng if rng is not None else np.random.default_rng()
        std = np.sqrt(np.maximum(self.variance, 1e-8))
        return gen.normal(loc=self.mean, scale=std, size=(n_samples, len(self.mean)))

    def log_pdf(self, observation: np.ndarray | float) -> float:
        """Computes calibrated log-predictive density of a realized measurement.

        For scalar property (D=1), computes standard 1D Gaussian log-density.
        For multivariate characterization embeddings (D > 1), computes the normalized
        log-density per dimension to prevent multidimensional embeddings from artificially
        dominating scalar property observations by an arbitrary factor of D:
            log_pdf = -0.5 * [ log(2*pi) + 1/D * sum(log(var_d)) + 1/D * sum(diff_d^2 / var_d) ]
        """
        obs = np.atleast_1d(np.asarray(observation, dtype=np.float64))
        dim = len(self.mean)
        diff = obs - self.mean

        if dim == 1:
            var = np.maximum(self.variance, 1e-10)
            quad = float((diff[0] ** 2) / var[0])
            log_det = float(np.log(var[0]))
            log_prob = -0.5 * (np.log(2.0 * np.pi) + log_det + quad)
        else:
            var = np.maximum(self.variance, 1e-5)
            # Dimension-normalized log-density rate
            quad = float(np.mean((diff**2) / var))
            log_det = float(np.mean(np.log(var)))
            log_prob = -0.5 * (np.log(2.0 * np.pi) + log_det + quad)

        return float(log_prob)


class ScientificHypothesisModel(Protocol):
    """Formal protocol for testable predictive scientific hypotheses."""

    @property
    def hypothesis_id(self) -> str:
        """Unique identifier (e.g. 'H1')."""
        ...

    @property
    def title(self) -> str:
        """Human-readable hypothesis title."""
        ...

    @property
    def statement(self) -> str:
        """Formal scientific claim."""
        ...

    @property
    def assumptions(self) -> list[str]:
        """Underlying modeling assumptions."""
        ...

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Fits hypothesis surrogate models strictly indexed by candidate_id."""
        ...

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
    ) -> PredictiveDistribution:
        """Generates the hypothesis-specific predictive distribution for a candidate action."""
        ...

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        """Evaluates empirical log-likelihood of observation under this hypothesis."""
        ...

    def supports_action(self, action_type: ExperimentActionType) -> bool:
        """Returns whether this hypothesis can predict outcomes for action_type."""
        ...

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
    ) -> str:
        """Returns explicit quantitative pre-conditions that would refute this hypothesis."""
        ...


def _build_candidate_maps(
    compositions: np.ndarray | None = None,
    property_targets: np.ndarray | None = None,
    xrd_embeddings: np.ndarray | None = None,
    xrd_compositions: np.ndarray | None = None,
    candidate_ids: Sequence[str] | None = None,
    xrd_candidate_ids: Sequence[str] | None = None,
    property_candidate_ids: Sequence[str] | None = None,
    observed_xrd_ids: set[str] | None = None,
    observed_property_ids: set[str] | None = None,
    composition_by_id: Mapping[str, np.ndarray] | None = None,
    property_by_id: Mapping[str, float] | None = None,
    xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Helper to safely build candidate ID maps from either explicit maps or legacy arrays."""
    comp_map: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64) for k, v in (composition_by_id or {}).items()}
    prop_map: dict[str, float] = {k: float(v) for k, v in (property_by_id or {}).items()}
    xrd_map: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64) for k, v in (xrd_embedding_by_id or {}).items()}

    # Populate comp_map from legacy positional lists if not provided
    if candidate_ids is not None and compositions is not None and len(candidate_ids) == len(compositions):
        for cid, c in zip(candidate_ids, compositions):
            comp_map[cid] = np.asarray(c, dtype=np.float64)
    if property_candidate_ids is not None and compositions is not None and len(property_candidate_ids) == len(compositions):
        for cid, c in zip(property_candidate_ids, compositions):
            comp_map[cid] = np.asarray(c, dtype=np.float64)
    if xrd_candidate_ids is not None and xrd_compositions is not None and len(xrd_candidate_ids) == len(xrd_compositions):
        for cid, c in zip(xrd_candidate_ids, xrd_compositions):
            comp_map[cid] = np.asarray(c, dtype=np.float64)

    # Populate prop_map from legacy targets
    if not prop_map and property_targets is not None:
        p_cids = property_candidate_ids or candidate_ids
        if p_cids is not None and len(p_cids) == len(property_targets):
            for cid, y in zip(p_cids, property_targets):
                prop_map[cid] = float(y)

    # Populate xrd_map from legacy embeddings
    if not xrd_map and xrd_embeddings is not None:
        x_cids = xrd_candidate_ids or candidate_ids
        if x_cids is not None and len(x_cids) == len(xrd_embeddings):
            for cid, emb in zip(x_cids, xrd_embeddings):
                xrd_map[cid] = np.asarray(emb, dtype=np.float64)

    return comp_map, prop_map, xrd_map


# ---------------------------------------------------------------------------
# H1: Composition-Sufficient Hypothesis
# ---------------------------------------------------------------------------
class CompositionSufficientHypothesis:
    """H1: Composition-Sufficient Hypothesis.

    Scientific Claim:
    Observed kinetics k0 is determined by composition alone.
    Structural characterization (XRD) does not provide independent predictive information for k0:
        p(k0 | composition, structure, H1) = p(k0 | composition, H1).
    """

    def __init__(self, random_state: int = 42) -> None:
        self._id = "H1"
        self._title = "Composition-Sufficient Hypothesis"
        self._statement = "Electrocatalytic property is determined by nominal composition alone without structural mediation."
        self._assumptions = [
            "Smooth global Gaussian process mapping from (Au, Ir, Rh) composition to k0.",
            "XRD structure does not provide independent predictive information for k0.",
            "Structural variation follows a shared baseline mapping from composition.",
        ]
        self.random_state = random_state
        self._prop_gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[10.0, 10.0, 10.0], length_scale_bounds=(1.0, 100.0))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e-1)),
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=random_state,
        )
        self._struct_gps: list[GaussianProcessRegressor] = []
        self.is_fitted = False
        self._emb_dim = 8

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def statement(self) -> str:
        return self._statement

    @property
    def assumptions(self) -> list[str]:
        return list(self._assumptions)

    def supports_action(self, action_type: ExperimentActionType) -> bool:
        return True

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray] | None = None,
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        comp_map, prop_map, xrd_map = _build_candidate_maps(
            composition_by_id=composition_by_id,
            property_by_id=property_by_id,
            xrd_embedding_by_id=xrd_embedding_by_id,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_property_ids,
            **kwargs,
        )

        # Fit property GP strictly on candidates with observed property
        prop_cids = sorted([cid for cid in prop_map if cid in comp_map])
        if prop_cids:
            X_p = np.array([comp_map[cid] for cid in prop_cids], dtype=np.float64)
            y_p = np.array([prop_map[cid] for cid in prop_cids], dtype=np.float64)
            self._prop_gp.fit(X_p, y_p)
        elif comp_map:
            cids = sorted(comp_map.keys())
            X_dummy = np.array([comp_map[cid] for cid in cids], dtype=np.float64)
            self._prop_gp.fit(X_dummy, np.zeros(len(cids)))

        # Fit shared structure GP on candidates with observed XRD
        xrd_cids = sorted([cid for cid in xrd_map if cid in comp_map])
        if xrd_cids:
            X_x = np.array([comp_map[cid] for cid in xrd_cids], dtype=np.float64)
            Y_x = np.array([xrd_map[cid] for cid in xrd_cids], dtype=np.float64)
            self._emb_dim = Y_x.shape[1]
            self._struct_gps = []
            for d in range(self._emb_dim):
                gp = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[10.0, 10.0, 10.0], length_scale_bounds=(1.0, 100.0))
                    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-1)),
                    alpha=1e-4,
                    normalize_y=True,
                    random_state=self.random_state + d,
                )
                gp.fit(X_x, Y_x[:, d])
                self._struct_gps.append(gp)

        self.is_fitted = True

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
    ) -> PredictiveDistribution:
        comp = np.atleast_2d(composition)

        if action_type == ExperimentActionType.PROPERTY:
            if self.is_fitted:
                mean, std = self._prop_gp.predict(comp, return_std=True)
                mean_val = float(mean[0])
                var_val = float(max(std[0] ** 2, 1e-4))
            else:
                mean_val = 0.005
                var_val = 0.01

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model_type": "composition_gp"},
            )

        elif action_type == ExperimentActionType.XRD:
            if self.is_fitted and self._struct_gps:
                means = []
                vars_ = []
                for gp in self._struct_gps:
                    m, s = gp.predict(comp, return_std=True)
                    means.append(float(m[0]))
                    vars_.append(float(max(s[0] ** 2, 1e-3)))
                mean_vec = np.array(means, dtype=np.float64)
                var_vec = np.array(vars_, dtype=np.float64)
            else:
                mean_vec = np.zeros(self._emb_dim, dtype=np.float64)
                var_vec = np.ones(self._emb_dim, dtype=np.float64) * 0.25

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model_type": "baseline_structure_gp"},
            )

        raise ValueError(f"Unsupported action type: {action_type}")

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
    ) -> str:
        return "Measured k0 residual correlates strongly with XRD structural features, falsifying composition sufficiency."


# ---------------------------------------------------------------------------
# H2: Structure-Informed Hypothesis
# ---------------------------------------------------------------------------
class StructureInformedHypothesis:
    """H2: Structure-Informed Hypothesis.

    Scientific Claim:
    Electrocatalytic kinetics k0 is mediated by physical crystal structure z.
    When XRD is observed, k0 is predicted jointly from (composition, XRD embedding):
        p(k0 | composition, observed_structure, H2).
    When XRD is unobserved, structural uncertainty is integrated into property prediction:
        p(k0 | composition, H2) = \\int p(k0 | composition, z) p(z | composition) dz.
    """

    def __init__(self, random_state: int = 42) -> None:
        self._id = "H2"
        self._title = "Structure-Informed Hypothesis"
        self._statement = "Electrocatalytic property is mediated by crystal structure; structural characterization provides predictive information."
        self._assumptions = [
            "XRD structural embedding represents physically active crystal features.",
            "When XRD is observed, k0 is predicted jointly from (composition, XRD embedding).",
            "When XRD is unobserved, structural uncertainty is integrated into property variance.",
        ]
        self.random_state = random_state
        self._joint_gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e2)) * Matern(length_scale=[10.0] * 11, nu=2.5, length_scale_bounds=(1.0, 100.0))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e-1)),
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=random_state,
        )
        self._comp_gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[10.0, 10.0, 10.0], length_scale_bounds=(1.0, 100.0))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e-1)),
            alpha=1e-5,
            normalize_y=True,
            random_state=random_state,
        )
        self._struct_gps: list[GaussianProcessRegressor] = []
        self.is_fitted = False
        self._has_joint_data = False
        self._emb_dim = 8

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def statement(self) -> str:
        return self._statement

    @property
    def assumptions(self) -> list[str]:
        return list(self._assumptions)

    def supports_action(self, action_type: ExperimentActionType) -> bool:
        return True

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray] | None = None,
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        comp_map, prop_map, xrd_map = _build_candidate_maps(
            composition_by_id=composition_by_id,
            property_by_id=property_by_id,
            xrd_embedding_by_id=xrd_embedding_by_id,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_property_ids,
            **kwargs,
        )

        # 1. Fit structure surrogate strictly on candidates with observed XRD
        xrd_cids = sorted([cid for cid in xrd_map if cid in comp_map])
        if xrd_cids:
            X_x = np.array([comp_map[cid] for cid in xrd_cids], dtype=np.float64)
            Y_x = np.array([xrd_map[cid] for cid in xrd_cids], dtype=np.float64)
            self._emb_dim = Y_x.shape[1]
            self._struct_gps = []
            for d in range(self._emb_dim):
                gp = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[10.0, 10.0, 10.0], length_scale_bounds=(1.0, 100.0))
                    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-1)),
                    alpha=1e-4,
                    normalize_y=True,
                    random_state=self.random_state + d,
                )
                gp.fit(X_x, Y_x[:, d])
                self._struct_gps.append(gp)

        # 2. Fit composition property GP
        prop_cids = sorted([cid for cid in prop_map if cid in comp_map])
        if prop_cids:
            X_p = np.array([comp_map[cid] for cid in prop_cids], dtype=np.float64)
            y_p = np.array([prop_map[cid] for cid in prop_cids], dtype=np.float64)
            self._comp_gp.fit(X_p, y_p)

        # 3. Fit joint (composition + XRD embedding -> k0) model using candidate_id as ONLY join key
        joint_cids = sorted([cid for cid in prop_map if cid in xrd_map and cid in comp_map])
        if len(joint_cids) >= 2:
            X_joint = np.array([np.hstack([comp_map[cid], xrd_map[cid]]) for cid in joint_cids], dtype=np.float64)
            y_joint = np.array([prop_map[cid] for cid in joint_cids], dtype=np.float64)
            self._joint_gp.fit(X_joint, y_joint)
            self._has_joint_data = True
        else:
            self._has_joint_data = False

        self.is_fitted = True

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
    ) -> PredictiveDistribution:
        comp = np.atleast_2d(composition)

        if action_type == ExperimentActionType.PROPERTY:
            # Predict structure mean & variance
            struct_means = []
            struct_vars = []
            if self.is_fitted and self._struct_gps:
                for gp in self._struct_gps:
                    m, s = gp.predict(comp, return_std=True)
                    struct_means.append(float(m[0]))
                    struct_vars.append(float(max(s[0] ** 2, 1e-3)))
                pred_z = np.array(struct_means, dtype=np.float64)
                var_z = np.array(struct_vars, dtype=np.float64)
            else:
                pred_z = np.zeros(self._emb_dim, dtype=np.float64)
                var_z = np.ones(self._emb_dim, dtype=np.float64) * 0.25

            # If observed XRD is provided, condition directly on observed structure
            if observed_xrd_embedding is not None and self._has_joint_data:
                joint_feat = np.hstack([comp, np.atleast_2d(observed_xrd_embedding)])
                mean, std = self._joint_gp.predict(joint_feat, return_std=True)
                mean_val = float(mean[0])
                var_val = float(max(std[0] ** 2, 1e-4))
                mode = "joint_observed_structure"
            elif self._has_joint_data:
                # Structure unobserved: propagate predicted structure mean and inflate variance with structural uncertainty
                joint_feat = np.hstack([comp, np.atleast_2d(pred_z)])
                mean, std = self._joint_gp.predict(joint_feat, return_std=True)
                mean_val = float(mean[0])
                var_val = float(max(std[0] ** 2 + 0.3 * float(np.mean(var_z)), 1e-4))
                mode = "joint_predicted_structure_inflated"
            elif self.is_fitted:
                mean, std = self._comp_gp.predict(comp, return_std=True)
                mean_val = float(mean[0])
                var_val = float(max(std[0] ** 2 + 0.1 * float(np.mean(var_z)), 1e-4))
                mode = "comp_gp_with_structure_uncertainty"
            else:
                mean_val = 0.005
                var_val = 0.015
                mode = "unfitted_prior"

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model_type": mode},
            )

        elif action_type == ExperimentActionType.XRD:
            # Shared structure prediction
            if self.is_fitted and self._struct_gps:
                means = []
                vars_ = []
                for gp in self._struct_gps:
                    m, s = gp.predict(comp, return_std=True)
                    means.append(float(m[0]))
                    vars_.append(float(max(s[0] ** 2, 1e-3)))
                mean_vec = np.array(means, dtype=np.float64)
                var_vec = np.array(vars_, dtype=np.float64)
            else:
                mean_vec = np.zeros(self._emb_dim, dtype=np.float64)
                var_vec = np.ones(self._emb_dim, dtype=np.float64) * 0.25

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model_type": "structure_surrogate_gp"},
            )

        raise ValueError(f"Unsupported action type: {action_type}")

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
    ) -> str:
        return "Structure-informed joint model achieves zero residual variance reduction over composition on held-out samples."


# ---------------------------------------------------------------------------
# H3: Local Structural-Regime Hypothesis
# ---------------------------------------------------------------------------
class LocalStructuralRegimeHypothesis:
    """H3: Local Structural-Regime Hypothesis.

    Scientific Claim:
    The composition-structure space contains localized distinct structural regimes.
    Global smooth interpolation fails in transition regions where local regimes dictate properties.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        self._id = "H3"
        self._title = "Local Structural-Regime Hypothesis"
        self._statement = "Composition space exhibits localized structural regimes with sharp regime-dependent properties."
        self._assumptions = [
            "Discrete local compositional clusters / regimes dictate structural behavior.",
            "Local regime models capture non-linear transitions that smooth global models smooth over.",
            "Candidates near regime boundaries exhibit higher structural variance.",
        ]
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=5)
        self._regime_prop_gps: dict[int, GaussianProcessRegressor] = {}
        self._global_prop_gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[5.0, 5.0, 5.0], length_scale_bounds=(0.5, 50.0))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e-1)),
            alpha=1e-5,
            normalize_y=True,
            random_state=random_state,
        )
        self._struct_gps: list[GaussianProcessRegressor] = []
        self.is_fitted = False
        self._emb_dim = 8

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def statement(self) -> str:
        return self._statement

    @property
    def assumptions(self) -> list[str]:
        return list(self._assumptions)

    def supports_action(self, action_type: ExperimentActionType) -> bool:
        return True

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray] | None = None,
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        comp_map, prop_map, xrd_map = _build_candidate_maps(
            composition_by_id=composition_by_id,
            property_by_id=property_by_id,
            xrd_embedding_by_id=xrd_embedding_by_id,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_property_ids,
            **kwargs,
        )

        all_cids = sorted(comp_map.keys())
        if all_cids:
            all_comps = np.array([comp_map[cid] for cid in all_cids], dtype=np.float64)
            if len(all_comps) >= self.n_regimes:
                self._kmeans.fit(all_comps)

        # Fit regime property GPs
        prop_cids = sorted([cid for cid in prop_map if cid in comp_map])
        if prop_cids:
            X_p = np.array([comp_map[cid] for cid in prop_cids], dtype=np.float64)
            y_p = np.array([prop_map[cid] for cid in prop_cids], dtype=np.float64)
            self._global_prop_gp.fit(X_p, y_p)
            self._regime_prop_gps = {}

            if len(X_p) >= self.n_regimes:
                cluster_labels = self._kmeans.predict(X_p)
                for k in range(self.n_regimes):
                    idx = np.where(cluster_labels == k)[0]
                    if len(idx) >= 2:
                        gp = GaussianProcessRegressor(
                            kernel=ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=[5.0, 5.0, 5.0], length_scale_bounds=(0.5, 50.0))
                            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e-1)),
                            alpha=1e-5,
                            normalize_y=True,
                            random_state=self.random_state + k,
                        )
                        gp.fit(X_p[idx], y_p[idx])
                        self._regime_prop_gps[k] = gp

        # Fit localized structure surrogate with Matern kernel
        xrd_cids = sorted([cid for cid in xrd_map if cid in comp_map])
        if xrd_cids:
            X_x = np.array([comp_map[cid] for cid in xrd_cids], dtype=np.float64)
            Y_x = np.array([xrd_map[cid] for cid in xrd_cids], dtype=np.float64)
            self._emb_dim = Y_x.shape[1]
            self._struct_gps = []
            for d in range(self._emb_dim):
                gp = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (1e-3, 1e2)) * Matern(length_scale=[5.0, 5.0, 5.0], nu=1.5, length_scale_bounds=(0.5, 50.0))
                    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-1)),
                    alpha=1e-4,
                    normalize_y=True,
                    random_state=self.random_state + 10 + d,
                )
                gp.fit(X_x, Y_x[:, d])
                self._struct_gps.append(gp)

        self.is_fitted = True

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
    ) -> PredictiveDistribution:
        comp = np.atleast_2d(composition)

        if action_type == ExperimentActionType.PROPERTY:
            if self.is_fitted:
                try:
                    regime_idx = int(self._kmeans.predict(comp)[0])
                except Exception:
                    regime_idx = 0

                if regime_idx in self._regime_prop_gps:
                    mean, std = self._regime_prop_gps[regime_idx].predict(comp, return_std=True)
                    mean_val = float(mean[0])
                    var_val = float(max(std[0] ** 2, 1e-4))
                    mode = f"regime_{regime_idx}_gp"
                else:
                    mean, std = self._global_prop_gp.predict(comp, return_std=True)
                    mean_val = float(mean[0])
                    var_val = float(max(std[0] ** 2 * 1.2, 1e-4))
                    mode = "global_regime_gp"
            else:
                mean_val = 0.005
                var_val = 0.02
                mode = "unfitted"

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model_type": mode},
            )

        elif action_type == ExperimentActionType.XRD:
            if self.is_fitted and self._struct_gps:
                means = []
                vars_ = []
                for gp in self._struct_gps:
                    m, s = gp.predict(comp, return_std=True)
                    means.append(float(m[0]))
                    vars_.append(float(max(s[0] ** 2, 1e-3)))
                mean_vec = np.array(means, dtype=np.float64)
                var_vec = np.array(vars_, dtype=np.float64)
            else:
                mean_vec = np.zeros(self._emb_dim, dtype=np.float64)
                var_vec = np.ones(self._emb_dim, dtype=np.float64) * 0.35

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model_type": "local_structure_matern_gp"},
            )

        raise ValueError(f"Unsupported action type: {action_type}")

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
    ) -> str:
        return "Measured structural features follow a uniform smooth global gradient across regime boundaries without localized transitions."


# ---------------------------------------------------------------------------
# Hypothesis Ensemble & Sequential Predictive Evidence Engine
# ---------------------------------------------------------------------------
class HypothesisEnsemble:
    """Manages the set of competing scientific hypotheses and performs sequential evidence updates."""

    def __init__(
        self,
        hypotheses: Sequence[ScientificHypothesisModel] | None = None,
        prior_beliefs: Mapping[str, float] | None = None,
    ) -> None:
        if hypotheses is None:
            self.hypotheses: dict[str, ScientificHypothesisModel] = {
                "H1": CompositionSufficientHypothesis(),
                "H2": StructureInformedHypothesis(),
                "H3": LocalStructuralRegimeHypothesis(),
            }
        else:
            self.hypotheses = {h.hypothesis_id: h for h in hypotheses}

        K = len(self.hypotheses)
        if prior_beliefs is None:
            self.prior_beliefs = {hid: 1.0 / K for hid in self.hypotheses}
        else:
            total_p = sum(prior_beliefs.values())
            self.prior_beliefs = {hid: float(prior_beliefs.get(hid, 1.0 / K)) / total_p for hid in self.hypotheses}

        self.cumulative_log_evidence: dict[str, float] = {hid: np.log(self.prior_beliefs[hid] + 1e-12) for hid in self.hypotheses}
        self.evidence_history: list[dict[str, Any]] = []

    def get_beliefs(self) -> dict[str, float]:
        """Returns normalized posterior hypothesis probabilities P(H_i | D_t) via Log-Sum-Exp."""
        log_evs = np.array([self.cumulative_log_evidence[hid] for hid in self.hypotheses], dtype=np.float64)
        lse = logsumexp(log_evs)
        norm_log_p = log_evs - lse
        probs = np.exp(norm_log_p)
        return {hid: float(p) for hid, p in zip(self.hypotheses.keys(), probs)}

    def get_entropy(self) -> float:
        """Returns current hypothesis entropy H[P(H|D)]."""
        beliefs = self.get_beliefs()
        probs = np.array(list(beliefs.values()), dtype=np.float64)
        probs = np.maximum(probs, 1e-12)
        return float(-np.sum(probs * np.log(probs)))

    def fit_all(
        self,
        composition_by_id: Mapping[str, np.ndarray] | None = None,
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Fits all underlying hypothesis models strictly on candidate ID maps."""
        for h in self.hypotheses.values():
            h.fit(
                composition_by_id=composition_by_id,
                property_by_id=property_by_id,
                xrd_embedding_by_id=xrd_embedding_by_id,
                observed_xrd_ids=observed_xrd_ids,
                observed_property_ids=observed_property_ids,
                **kwargs,
            )

    def predict_all(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
    ) -> dict[str, PredictiveDistribution]:
        """Generates predictive distributions for all hypotheses for a given action."""
        return {
            hid: h.predict_observation(
                candidate_id=candidate_id,
                action_type=action_type,
                composition=composition,
                observed_xrd_embedding=observed_xrd_embedding,
            )
            for hid, h in self.hypotheses.items()
            if h.supports_action(action_type)
        }

    def record_observation_and_update(
        self,
        action_id: str,
        candidate_id: str,
        action_type: ExperimentActionType,
        observation: np.ndarray | float,
        pre_predictions: dict[str, PredictiveDistribution],
    ) -> dict[str, Any]:
        """Updates sequential predictive log-evidence and computes new hypothesis beliefs."""
        before_beliefs = self.get_beliefs()
        before_entropy = self.get_entropy()

        log_scores: dict[str, float] = {}
        for hid, pred in pre_predictions.items():
            h = self.hypotheses[hid]
            score = h.log_predictive_density(observation=observation, prediction=pred)
            log_scores[hid] = score
            self.cumulative_log_evidence[hid] += score

        after_beliefs = self.get_beliefs()
        after_entropy = self.get_entropy()
        realized_entropy_reduction = before_entropy - after_entropy

        record = {
            "step": len(self.evidence_history) + 1,
            "action_id": action_id,
            "candidate_id": candidate_id,
            "action_type": action_type.value,
            "log_predictive_scores": log_scores,
            "before_beliefs": before_beliefs,
            "after_beliefs": after_beliefs,
            "before_entropy": before_entropy,
            "after_entropy": after_entropy,
            "realized_entropy_reduction": realized_entropy_reduction,
        }
        self.evidence_history.append(record)
        return record

    def reset(self) -> None:
        """Resets cumulative log-evidence to prior beliefs and clears history."""
        self.cumulative_log_evidence = {hid: np.log(self.prior_beliefs[hid] + 1e-12) for hid in self.hypotheses}
        self.evidence_history.clear()
