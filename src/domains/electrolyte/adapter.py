from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.domains.electrolyte.config import (
    ELECTROLYTE_DOMAIN_CONFIG,
    ELECTROLYTE_DOMAIN_ID,
    ELECTROLYTE_MODALITY_CAPACITY,
    ELECTROLYTE_OBJECTIVE_CAPACITY,
    ELECTROLYTE_SOLVENT_FEATURES,
)
from src.domains.electrolyte.data import (
    DEFAULT_COMPATIBLE_DERIVED_PATH,
    DEFAULT_CONTRACT_PATH,
    extract_candidate_pool_from_derived,
    generate_candidate_id,
    load_derived_historical_outcomes,
    load_electrolyte_data_contract,
)
from src.domains.electrolyte.hypotheses import ElectrolyteHypothesisProvider
from src.domains.electrolyte.oracle import (
    HistoricalElectrolyteOracle,
    SurrogateElectrolyteOracle,
)
from src.science.actions import (
    ActionType,
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

logger = logging.getLogger(__name__)


class ElectrolyteDomainAdapter(MaterialDomainAdapter):
    """Universal MaterialDomainAdapter implementation for the anode-free electrolyte discovery domain.

    Supports:
    1. Retrospective finite-pool replay over experimentally measured de-expanded historical outcomes (N=75).
    2. Zero target leakage through the strict offline candidate pool firewall.
    3. Three competing predictive structural hypotheses (H1, H2, H3).
    4. Deterministic candidate identities (ELEC_<hash>).
    5. Native compatibility with ScientificDecisionEngine and BoTorchBackend.
    """

    def __init__(
        self,
        derived_outcomes_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
        contract_path: str = DEFAULT_CONTRACT_PATH,
        config: MaterialDomainConfig | None = None,
        use_surrogate_oracle: bool = False,
        surrogate_train_path: str | None = None,
        candidate_pool_df: pd.DataFrame | None = None,
        oracle: Any | None = None,
    ) -> None:
        self._config = config or ELECTROLYTE_DOMAIN_CONFIG
        self._hypothesis_provider = ElectrolyteHypothesisProvider()

        # Load historical derived data
        if derived_outcomes_path and os.path.exists(derived_outcomes_path):
            self._derived_df = load_derived_historical_outcomes(derived_outcomes_path)
        else:
            self._derived_df = pd.DataFrame()

        # Candidate pool: working set takes precedence if explicitly passed
        if candidate_pool_df is not None:
            # Enforce candidate firewall: drop forbidden columns if any
            clean_cols = ["candidate_id"] + [f for f in self._config.candidate_features if f in candidate_pool_df.columns]
            self._candidate_pool_df = candidate_pool_df[clean_cols].copy().reset_index(drop=True)
        elif len(self._derived_df) > 0:
            self._candidate_pool_df = extract_candidate_pool_from_derived(
                self._derived_df,
                feature_cols=self._config.candidate_features,
            )
        else:
            self._candidate_pool_df = pd.DataFrame(columns=["candidate_id"] + list(self._config.candidate_features))

        # Build feature cache
        self._feature_cache: dict[str, dict[str, Any]] = {}
        for _, row in self._candidate_pool_df.iterrows():
            cid = str(row["candidate_id"])
            self._feature_cache[cid] = {f: float(row[f]) for f in self._config.candidate_features}

        # Initialize Oracle
        if oracle is not None:
            self._oracle = oracle
            self._is_surrogate = isinstance(oracle, SurrogateElectrolyteOracle) or getattr(oracle, "_is_surrogate", False)
        elif use_surrogate_oracle:
            train_df = self._derived_df if surrogate_train_path is None else pd.read_csv(surrogate_train_path)
            self._oracle = SurrogateElectrolyteOracle(df_train=train_df, feature_cols=self._config.candidate_features)
            self._is_surrogate = True
        else:
            self._oracle = HistoricalElectrolyteOracle(self._derived_df)
            self._is_surrogate = False

        self._revealed_cids: set[str] = set()
        self._revealed_capacity_obs: dict[str, float] = {}

    @property
    def domain_id(self) -> str:
        return self._config.domain_id

    def get_config(self) -> MaterialDomainConfig:
        return self._config

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the visible candidate pool.

        STRICT FIREWALL: Contains zero ground-truth labels and zero future batch indicators.
        """
        return self._candidate_pool_df.copy()

    def get_candidate_features(self, candidate_id: str) -> Mapping[str, Any]:
        if candidate_id not in self._feature_cache:
            raise KeyError(f"Candidate '{candidate_id}' not found in electrolyte domain candidate pool.")
        return dict(self._feature_cache[candidate_id])

    def get_observations_by_modality(self) -> Mapping[str, Mapping[str, Any]]:
        return {"CAPACITY_TEST": dict(self._revealed_capacity_obs)}

    def list_valid_actions(self, state: Any = None) -> Sequence[ScientificAction]:
        """Lists valid unexecuted CAPACITY_TEST measurement actions."""
        observed_cids: set[str] = set(self._revealed_cids)
        if state is not None:
            if hasattr(state, "observed_candidate_ids"):
                observed_cids.update(state.observed_candidate_ids)
            elif hasattr(state, "executed_actions"):
                observed_cids.update(act.candidate_id for act in state.executed_actions)
            elif hasattr(state, "observations_by_modality"):
                for m_dict in state.observations_by_modality.values():
                    observed_cids.update(m_dict.keys())
            elif isinstance(state, dict):
                if "observed_candidate_ids" in state:
                    observed_cids.update(state["observed_candidate_ids"])
                elif "observations_by_modality" in state:
                    for m_dict in state["observations_by_modality"].values():
                        observed_cids.update(m_dict.keys())
            elif isinstance(state, (set, list, tuple)):
                observed_cids.update(state)

        actions = []
        for cid in self._candidate_pool_df["candidate_id"]:
            if cid in observed_cids:
                continue
            actions.append(
                ScientificAction(
                    action_id=f"CAPACITY_TEST_{cid}",
                    candidate_id=cid,
                    action_type="CAPACITY_TEST",
                    estimated_cost=ELECTROLYTE_MODALITY_CAPACITY.cost,
                    metadata={"modality": "CAPACITY_TEST", "domain": self.domain_id},
                )
            )
        return actions

    def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Reveals historical experimental measurement or in-silico simulation outcome."""
        act_norm = normalize_action_type(action.action_type)
        if act_norm != "CAPACITY_TEST":
            raise ValueError(f"Electrolyte domain only supports 'CAPACITY_TEST', got '{act_norm}'.")

        if self._is_surrogate:
            feats = np.array([self._feature_cache[action.candidate_id][f] for f in self._config.candidate_features])
            outcome = self._oracle.reveal(action, feats)  # type: ignore[union-attr]
        else:
            outcome = self._oracle.reveal(action)  # type: ignore[union-attr]

        self._revealed_cids.add(action.candidate_id)
        if outcome.revealed_data and "C_norm_20" in outcome.revealed_data:
            self._revealed_capacity_obs[action.candidate_id] = float(outcome.revealed_data["C_norm_20"])

        return outcome

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        return list(self._config.objectives)

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        return list(self._config.modalities)

    def get_hypothesis_provider(self) -> HypothesisProvider | None:
        return self._hypothesis_provider

    def get_default_initial_actions(
        self,
        n_seed: int = 3,
        seed: int = 42,
    ) -> Sequence[ScientificAction]:
        """Returns the natural initial evidence actions without future leakage.

        In historical mode, selects the Batch-0 compatible seed cells (N=3) which were
        tested in the initial exploratory seed library under 1.0 M LiFSI Cu||LFP protocol.
        """
        if len(self._candidate_pool_df) == 0:
            return []

        seed_cids = []
        if not self._is_surrogate and len(self._derived_df) > 0 and "batch" in self._derived_df.columns:
            b0_cids = set(self._derived_df[self._derived_df["batch"] == 0]["candidate_id"].unique())
            pool_b0 = [cid for cid in self._candidate_pool_df["candidate_id"].unique() if cid in b0_cids]
            if len(pool_b0) >= n_seed:
                seed_cids = pool_b0[:n_seed]

        if not seed_cids:
            rng = np.random.default_rng(seed)
            all_cids = list(self._candidate_pool_df["candidate_id"].unique())
            seed_cids = list(rng.choice(all_cids, size=min(n_seed, len(all_cids)), replace=False))

        actions = [
            ScientificAction(
                action_id=f"CAPACITY_TEST_{cid}",
                candidate_id=cid,
                action_type="CAPACITY_TEST",
                estimated_cost=ELECTROLYTE_MODALITY_CAPACITY.cost,
                metadata={"seed_init": True, "batch": 0},
            )
            for cid in seed_cids
        ]

        return actions

    def get_candidate_features(self, candidate_id: str) -> dict[str, float]:
        """Public read-only accessor for candidate features."""
        return dict(self._feature_cache.get(candidate_id, {}))

    def get_historical_outcomes_for_evaluation(self) -> pd.DataFrame:
        """Public read-only accessor for historical outcomes (EVALUATION-ONLY)."""
        return self._derived_df.copy()
