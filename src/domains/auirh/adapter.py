from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from src.datasets.auirh_actions import AuIrRhMultimodalOracle
from src.domains.auirh.config import (
    AUIRH_MODALITY_PROPERTY,
    AUIRH_MODALITY_XRD,
    AUIRH_OBJECTIVE_K0,
)
from src.domains.auirh.hypotheses import AuIrRhHypothesisProvider
from src.science.actions import (
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
)
from src.science.domain import (
    HypothesisProvider,
    MaterialDomainAdapter,
    ModalityDefinition,
    ObjectiveDefinition,
)


class AuIrRhDomainAdapter:
    """Material domain adapter for the Au-Ir-Rh thin-film catalyst discovery system.

    Implements the generic `MaterialDomainAdapter` contract wrapping the offline
    `AuIrRhMultimodalOracle` with strict information-horizon firewall enforcement.
    """

    def __init__(
        self,
        oracle: AuIrRhMultimodalOracle | None = None,
    ) -> None:
        if oracle is None:
            self._oracle = AuIrRhMultimodalOracle()
        else:
            self._oracle = oracle
        self._hypothesis_provider = AuIrRhHypothesisProvider()

    @property
    def domain_id(self) -> str:
        """Unique identifier for the Au-Ir-Rh domain."""
        return "auirh"

    @property
    def oracle(self) -> AuIrRhMultimodalOracle:
        """Underlying multimodal oracle instance."""
        return self._oracle

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the visible candidate pool containing pre-experiment features only.

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
        return self._oracle.execute(action)

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        """Returns the domain objective definitions."""
        return [AUIRH_OBJECTIVE_K0]

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        """Returns the domain modality definitions."""
        return [AUIRH_MODALITY_XRD, AUIRH_MODALITY_PROPERTY]

    def get_hypothesis_provider(self) -> HypothesisProvider:
        """Returns the domain-specific hypothesis factory."""
        return self._hypothesis_provider
