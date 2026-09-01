from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.auirh_actions import AuIrRhMultimodalOracle
from src.domains.auirh.config import (
    AUIRH_DOMAIN_CONFIG,
    AUIRH_MODALITY_PROPERTY,
    AUIRH_MODALITY_XRD,
    AUIRH_OBJECTIVE_K0,
)
from src.domains.auirh.hypotheses import AuIrRhHypothesisProvider
from src.science.actions import (
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
    HypothesisProvider,
    MaterialDomainAdapter,
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
)
from src.science.representation import (
    ObservationRepresentationManager,
    RepresentationSnapshot,
)
from src.science.xrd_representation import XRDRepresentationExtractor


class AuIrRhDomainAdapter(ObservationRepresentationManager):
    """Material domain adapter for the Au-Ir-Rh thin-film catalyst discovery system.

    Implements the generic `MaterialDomainAdapter` contract wrapping the offline
    `AuIrRhMultimodalOracle` with strict information-horizon firewall enforcement
    and frozen-basis representation lifecycle management.
    """

    def __init__(
        self,
        oracle: AuIrRhMultimodalOracle | None = None,
        hypothesis_provider: HypothesisProvider | None = None,
    ) -> None:
        self._oracle = oracle if oracle is not None else AuIrRhMultimodalOracle()
        self._hypothesis_provider = (
            hypothesis_provider
            if hypothesis_provider is not None
            else AuIrRhHypothesisProvider()
        )
        self._xrd_extractor = XRDRepresentationExtractor(min_pca_samples=3, representation_id="auirh_xrd_pca")
        self._revealed_xrd_spectra: dict[str, np.ndarray] = {}
        self._xrd_embeddings: dict[str, np.ndarray] = {}

    @property
    def domain_id(self) -> str:
        """Unique identifier for this material domain."""
        return "auirh"

    def get_config(self) -> MaterialDomainConfig:
        """Returns the typed configuration contract for this domain."""
        return AUIRH_DOMAIN_CONFIG

    @property
    def oracle(self) -> AuIrRhMultimodalOracle:
        """Underlying multimodal oracle instance."""
        return self._oracle

    def get_default_initial_actions(
        self,
        n_property: int = 5,
        n_characterization: int = 5,
        n_prop: int | None = None,
        n_xrd: int | None = None,
        pairing_strategy: str = "joint",
        seed: int = 42,
    ) -> list[ScientificAction]:
        """Generates a reproducible initial seed action plan for campaign bootstrapping.

        Parameters
        ----------
        n_property : int
            Number of initial property measurement actions.
        n_characterization : int
            Number of initial XRD characterization actions.
        pairing_strategy : str
            'joint' (default): candidate IDs for property and characterization overlap.
            'disjoint': property and characterization use mutually disjoint candidates.
            'mixed': partial overlap.
        seed : int
            Random seed for shuffling candidate pool.
        """
        if n_prop is not None:
            n_property = n_prop
        if n_xrd is not None:
            n_characterization = n_xrd

        rng = np.random.default_rng(seed)
        cids = list(self._oracle.get_candidate_pool()["candidate_id"].tolist())
        rng.shuffle(cids)

        if pairing_strategy == "joint":
            init_prop_cids = cids[:n_property]
            init_xrd_cids = cids[:n_characterization]
        elif pairing_strategy == "disjoint":
            init_prop_cids = cids[:n_property]
            init_xrd_cids = cids[n_property : n_property + n_characterization]
        elif pairing_strategy == "mixed":
            n_overlap = min(n_property, n_characterization) // 2
            init_prop_cids = cids[:n_property]
            init_xrd_cids = cids[:n_overlap] + cids[n_property : n_property + (n_characterization - n_overlap)]
        else:
            raise ValueError(f"Unknown pairing_strategy: '{pairing_strategy}'. Expected 'joint', 'disjoint', or 'mixed'.")

        actions: list[ScientificAction] = []
        for cid in init_prop_cids:
            actions.append(
                ScientificAction(
                    action_id=f"init_prop_{cid}",
                    candidate_id=cid,
                    action_type=ExperimentActionType.PROPERTY,
                    estimated_cost=AUIRH_MODALITY_PROPERTY.cost,
                )
            )
        for cid in init_xrd_cids:
            actions.append(
                ScientificAction(
                    action_id=f"init_xrd_{cid}",
                    candidate_id=cid,
                    action_type=ExperimentActionType.XRD,
                    estimated_cost=AUIRH_MODALITY_XRD.cost,
                )
            )
        return actions

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns visible candidate pool containing composition features ONLY.

        STRICT FIREWALL: Contains zero target values and zero XRD measurements.
        """
        return self._oracle.get_candidate_pool()

    def get_candidate_features(self, candidate_id: str) -> Mapping[str, float]:
        """Returns the pre-experiment composition features for a candidate."""
        pool_df = self._oracle.get_candidate_pool()
        matching = pool_df[pool_df["candidate_id"] == candidate_id]
        if matching.empty:
            raise KeyError(f"Candidate '{candidate_id}' not found in candidate pool.")
        row = matching.iloc[0]
        return {
            "Au": float(row["Au"]),
            "Ir": float(row["Ir"]),
            "Rh": float(row["Rh"]),
        }

    def list_valid_actions(
        self,
        state: Any = None,
    ) -> Sequence[ScientificAction]:
        """Lists all currently valid measurement actions (unobserved XRD and PROPERTY)."""
        pool_df = self._oracle.get_candidate_pool()
        cids = pool_df["candidate_id"].tolist()
        actions: list[ScientificAction] = []

        for cid in cids:
            if not self._oracle.is_xrd_observed(cid):
                actions.append(
                    ScientificAction(
                        action_id=f"act_xrd_{cid}",
                        candidate_id=cid,
                        action_type=ExperimentActionType.XRD,
                        estimated_cost=AUIRH_MODALITY_XRD.cost,
                    )
                )
            if not self._oracle.is_property_observed(cid):
                actions.append(
                    ScientificAction(
                        action_id=f"act_prop_{cid}",
                        candidate_id=cid,
                        action_type=ExperimentActionType.PROPERTY,
                        estimated_cost=AUIRH_MODALITY_PROPERTY.cost,
                    )
                )
        return actions

    def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Executes a scientific measurement action via the offline oracle."""
        outcome = self._oracle.execute(action)
        norm_type = normalize_action_type(action.action_type)
        if norm_type == "XRD" and "normalized_intensity" in outcome.revealed_data:
            cid = action.candidate_id
            norm_spec = np.asarray(outcome.revealed_data["normalized_intensity"], dtype=np.float64)
            emb = self._xrd_extractor.transform(norm_spec)
            outcome.revealed_data["xrd_embedding"] = emb.tolist()
            self._revealed_xrd_spectra[cid] = norm_spec
            self._xrd_embeddings[cid] = emb

        return outcome

    def get_representation_snapshot(self, modality_name: str) -> RepresentationSnapshot | None:
        """Returns an immutable snapshot of the current representation basis."""
        if normalize_action_type(modality_name) == "XRD":
            return self._xrd_extractor.get_snapshot(representation_id="auirh_xrd_pca", modality_name="XRD")
        return None

    def transform_with_snapshot(
        self,
        modality_name: str,
        raw_observation: Any,
        snapshot: RepresentationSnapshot,
    ) -> Any:
        """Transforms a raw measurement using the specific frozen representation snapshot basis."""
        if normalize_action_type(modality_name) == "XRD":
            if isinstance(raw_observation, dict) and "normalized_intensity" in raw_observation:
                spec = raw_observation["normalized_intensity"]
            else:
                spec = raw_observation
            return self._xrd_extractor.transform_with_snapshot(spec, snapshot)
        return raw_observation

    def update_representation_after_evidence(
        self,
        modality_name: str,
        candidate_id: str,
        raw_observation: Any,
    ) -> None:
        """Updates the representation basis using newly confirmed evidence (called strictly after Bayesian update)."""
        if normalize_action_type(modality_name) == "XRD":
            if isinstance(raw_observation, dict) and "normalized_intensity" in raw_observation:
                spec = raw_observation["normalized_intensity"]
            else:
                spec = raw_observation
            self._revealed_xrd_spectra[candidate_id] = np.asarray(spec, dtype=np.float64)
            self._xrd_extractor.fit(list(self._revealed_xrd_spectra.values()))
            self._xrd_embeddings = self._xrd_extractor.transform_batch(self._revealed_xrd_spectra)

    def get_revealed_xrd_embeddings(self) -> dict[str, np.ndarray]:
        """Returns current embeddings for all revealed XRD candidates under the unified current basis."""
        return dict(self._xrd_embeddings)

    def get_observations_by_modality(self) -> dict[str, dict[str, Any]]:
        """Returns all currently revealed observations grouped by modality name."""
        rev_xrds = dict(self._xrd_embeddings)
        rev_props: dict[str, float] = {}
        for cid, out in self._oracle.get_revealed_properties().items():
            if "k0" in out.revealed_data:
                rev_props[cid] = float(out.revealed_data["k0"])
        return {
            "XRD": rev_xrds,
            "PROPERTY": rev_props,
        }

    def get_current_observations(self) -> Mapping[str, Mapping[str, Any]]:
        """Returns all current observations in the latest representation basis."""
        return self.get_observations_by_modality()

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        """Returns the domain objective definitions."""
        return [AUIRH_OBJECTIVE_K0]

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        """Returns the domain modality definitions."""
        return [AUIRH_MODALITY_XRD, AUIRH_MODALITY_PROPERTY]

    def get_hypothesis_provider(self) -> HypothesisProvider:
        """Returns the domain-specific hypothesis factory."""
        return self._hypothesis_provider
