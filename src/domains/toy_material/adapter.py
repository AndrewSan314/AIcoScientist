from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.domains.toy_material.config import (
    TOY_MODALITY_CAPACITY,
    TOY_MODALITY_SEM,
    TOY_OBJECTIVE_CAPACITY,
)
from src.domains.toy_material.hypotheses import ToyMaterialHypothesisProvider
from src.science.actions import (
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
    HypothesisProvider,
    MaterialDomainAdapter,
    ModalityDefinition,
    ObjectiveDefinition,
)


class ToyMaterialDomainAdapter:
    """Synthetic battery cathode material domain adapter for architecture validation.

    STRICT FIREWALL:
    - Hidden SEM morphologies and true capacities are stored privately in `_ground_truth_map`.
    - `get_candidate_pool()` exposes ONLY pre-experiment formulation parameters (Li_ratio, doping_conc, sintering_temp).
    - Unrevealed values are strictly inaccessible until `execute_or_reveal()` is invoked.
    """

    def __init__(self, n_candidates: int = 30, seed: int = 42) -> None:
        self.n_candidates = n_candidates
        self.seed = seed
        self._hypothesis_provider = ToyMaterialHypothesisProvider()

        rng = np.random.default_rng(seed)
        self._candidate_pool_records: list[dict[str, Any]] = []
        self._ground_truth_map: dict[str, dict[str, Any]] = {}

        for i in range(n_candidates):
            cid = f"BAT_{i+1:03d}"
            li_ratio = float(np.round(rng.uniform(0.85, 1.15), 3))
            doping = float(np.round(rng.uniform(0.01, 0.08), 3))
            temp = float(np.round(rng.uniform(700.0, 920.0), 1))

            self._candidate_pool_records.append(
                {
                    "candidate_id": cid,
                    "Li_ratio": li_ratio,
                    "doping_conc": doping,
                    "sintering_temp": temp,
                }
            )

            # Ground truth generation (hidden behind firewall)
            # True underlying physical law: Thermal sintering optimum near 820 C + Li stoichiometry
            temp_opt = np.exp(-0.5 * ((temp - 820.0) / 45.0) ** 2)
            grain_size = float(np.clip((temp - 650.0) / 300.0, 0.1, 1.0))
            sem_emb = np.array([grain_size, 0.7 * li_ratio, 0.5 * (1.0 - doping), grain_size * 0.9])

            true_cap = float(135.0 + 45.0 * temp_opt + 30.0 * (li_ratio - 1.0) - 20.0 * doping + rng.normal(0, 0.5))

            self._ground_truth_map[cid] = {
                "candidate_id": cid,
                "Li_ratio": li_ratio,
                "doping_conc": doping,
                "sintering_temp": temp,
                "sem_features": sem_emb,
                "capacity": true_cap,
            }

        self._revealed_sem: dict[str, ExperimentOutcome] = {}
        self._revealed_capacity: dict[str, ExperimentOutcome] = {}
        self._action_history: list[ExperimentOutcome] = []

    @property
    def domain_id(self) -> str:
        return "toy_material"

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns visible candidate pool containing pre-experiment features ONLY."""
        return pd.DataFrame(self._candidate_pool_records).copy()

    def get_candidate_features(self, candidate_id: str) -> Mapping[str, Any]:
        """Returns pre-experiment composition and process parameters for a candidate."""
        pool_df = self.get_candidate_pool()
        matching = pool_df[pool_df["candidate_id"] == candidate_id]
        if matching.empty:
            raise KeyError(f"Candidate '{candidate_id}' not found in candidate pool.")
        row = matching.iloc[0]
        return {
            "Li_ratio": float(row["Li_ratio"]),
            "doping_conc": float(row["doping_conc"]),
            "sintering_temp": float(row["sintering_temp"]),
        }

    def is_sem_observed(self, candidate_id: str) -> bool:
        return candidate_id in self._revealed_sem

    def is_capacity_observed(self, candidate_id: str) -> bool:
        return candidate_id in self._revealed_capacity

    def list_valid_actions(
        self,
        state: Any = None,
    ) -> Sequence[ScientificAction]:
        """Lists unobserved SEM and CAPACITY_TEST actions."""
        pool_df = self.get_candidate_pool()
        cids = pool_df["candidate_id"].tolist()
        actions: list[ScientificAction] = []

        for cid in cids:
            if not self.is_sem_observed(cid):
                actions.append(
                    ScientificAction(
                        action_id=f"act_sem_{cid}",
                        candidate_id=cid,
                        action_type="SEM",
                        estimated_cost=TOY_MODALITY_SEM.cost,
                    )
                )
            if not self.is_capacity_observed(cid):
                actions.append(
                    ScientificAction(
                        action_id=f"act_cap_{cid}",
                        candidate_id=cid,
                        action_type="CAPACITY_TEST",
                        estimated_cost=TOY_MODALITY_CAPACITY.cost,
                    )
                )
        return actions

    def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Executes a measurement action and reveals the ground-truth outcome."""
        cid = action.candidate_id
        if cid not in self._ground_truth_map:
            raise KeyError(f"Candidate ID '{cid}' not found in toy material space.")

        gt = self._ground_truth_map[cid]
        act_norm = normalize_action_type(action.action_type)

        if act_norm == "SEM":
            if cid in self._revealed_sem:
                raise ValueError(f"SEM characterization already executed for candidate '{cid}'.")
            outcome = ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type="SEM",
                revealed_data={"sem_features": gt["sem_features"].tolist()},
                provenance={"domain_id": self.domain_id, "technique": "SEM Morphology"},
            )
            self._revealed_sem[cid] = outcome
        elif act_norm == "CAPACITY_TEST":
            if cid in self._revealed_capacity:
                raise ValueError(f"Capacity measurement already executed for candidate '{cid}'.")
            outcome = ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type="CAPACITY_TEST",
                revealed_data={"capacity": float(gt["capacity"])},
                provenance={"domain_id": self.domain_id, "technique": "Galvanostatic Cycling", "rate": "0.1C"},
            )
            self._revealed_capacity[cid] = outcome
        else:
            raise ValueError(f"Unsupported action type for toy material domain: {action.action_type}")

        self._action_history.append(outcome)
        return outcome

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        return [TOY_OBJECTIVE_CAPACITY]

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        return [TOY_MODALITY_SEM, TOY_MODALITY_CAPACITY]

    def get_hypothesis_provider(self) -> HypothesisProvider:
        return self._hypothesis_provider
