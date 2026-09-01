from __future__ import annotations

import io
import json
import logging
import os
import pickle
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class _SafeALabUnpickler(pickle.Unpickler):
    """Custom unpickler that gracefully mocks external dara/pymatgen classes without requiring dependencies."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except Exception:
            class _DummyRefinementObject:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass

                def __setstate__(self, state: Any) -> None:
                    if isinstance(state, dict):
                        self.__dict__.update(state)

            _DummyRefinementObject.__name__ = name
            _DummyRefinementObject.__module__ = module
            return _DummyRefinementObject

from src.domains.alab.artifact_index import ALabArtifactIndex, ArtifactRef
from src.domains.alab.config import (
    ALAB_DOMAIN_CONFIG,
    ALAB_MODALITY_OUTCOME_TEST,
    ALAB_MODALITY_REFINEMENT,
    ALAB_MODALITY_XRD,
    ALAB_OBJECTIVE_REACTION_CONVERSION,
)
from src.domains.alab.feature_encoder import ALabFeatureEncoder
from src.domains.alab.hypotheses import ALabHypothesisProvider
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

logger = logging.getLogger(__name__)


class ALabDomainAdapter(ObservationRepresentationManager):
    """Multimodal domain adapter for the A-Lab Precursor Genome solid-state synthesis dataset.

    Implements the generic `MaterialDomainAdapter` and `ObservationRepresentationManager`
    contracts over real external offline experimental archives.
    """

    def __init__(
        self,
        data_dir: str = "data/external/precursor_genome_2026",
        cache_dir: str = "data/derived/alab",
        samples: list[dict[str, Any]] | None = None,
        hypothesis_provider: HypothesisProvider | None = None,
        min_pca_samples: int = 3,
    ) -> None:
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self._samples_list = samples
        self._hypothesis_provider = (
            hypothesis_provider
            if hypothesis_provider is not None
            else ALabHypothesisProvider()
        )

        # Load samples from ledger if not explicitly provided
        if self._samples_list is None:
            ledger_path = os.path.join(self.data_dir, "ledger_precursor_genome.json")
            if os.path.exists(ledger_path):
                with open(ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._samples_list = data.get("samples", [])
            else:
                self._samples_list = []

        self._samples_map: dict[str, dict[str, Any]] = {
            s["sample_id"]: s for s in self._samples_list if "sample_id" in s
        }

        # Initialize feature encoder
        self._encoder = ALabFeatureEncoder()
        self._encoder.fit(self._samples_list)

        # Initialize artifact index
        self._artifact_index = ALabArtifactIndex(data_dir=self.data_dir, cache_dir=self.cache_dir)
        self._artifact_index.build_or_load(samples=self._samples_list)

        # XRD representation manager (leakage-free basis lifecycle)
        self._xrd_extractor = XRDRepresentationExtractor(
            n_components=8,
            min_pca_samples=min_pca_samples,
            representation_id="alab_xrd_pca",
        )
        self._revealed_xrd_spectra: dict[str, np.ndarray] = {}
        self._xrd_embeddings: dict[str, np.ndarray] = {}
        self._revealed_refinements: dict[str, np.ndarray] = {}
        self._revealed_outcomes: dict[str, float] = {}

        # Build candidate pool DataFrame (strictly firewalled: pre-experiment features ONLY)
        pool_rows = []
        for s in self._samples_list:
            sid = s["sample_id"]
            feat_vec = self._encoder.encode_candidate(s)
            raw_meta = self._encoder.extract_raw_metadata(s)
            row = {
                "candidate_id": sid,
                "reaction_energy_ev_per_atom": float(feat_vec[0]),
                "heating_temperature_c": float(raw_meta.get("heating_temperature_c") or 200.0),
                "heating_time_minutes": float(raw_meta.get("heating_time_minutes") or 60.0),
                "precursor_1_idx": float(feat_vec[3]),
                "precursor_2_idx": float(feat_vec[4]),
                "target_compound": str(raw_meta.get("target_compound") or ""),
                "precursor_1": str(raw_meta.get("precursor_1") or ""),
                "precursor_2": str(raw_meta.get("precursor_2") or ""),
            }
            pool_rows.append(row)

        self._candidate_pool_df = pd.DataFrame(pool_rows)

    @property
    def domain_id(self) -> str:
        return "alab_precursor_genome"

    def get_config(self) -> MaterialDomainConfig:
        return ALAB_DOMAIN_CONFIG

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the pre-experiment candidate pool (firewalled against post-experiment observations)."""
        return self._candidate_pool_df.copy()

    def get_candidate_features(self, candidate_id: str) -> Mapping[str, float]:
        """Returns numeric candidate feature vector."""
        if candidate_id not in self._samples_map:
            raise KeyError(f"Candidate '{candidate_id}' not found in A-Lab dataset.")
        sample = self._samples_map[candidate_id]
        feat_vec = self._encoder.encode_candidate(sample)
        return {
            "reaction_energy_ev_per_atom": float(feat_vec[0]),
            "heating_temperature_c": float(feat_vec[1]),
            "heating_time_minutes": float(feat_vec[2]),
            "precursor_1_idx": float(feat_vec[3]),
            "precursor_2_idx": float(feat_vec[4]),
        }

    def list_valid_actions(self, state: Any = None) -> Sequence[ScientificAction]:
        """Lists valid unrevealed measurement actions.

        PREREQUISITE CONTRACT:
        - OUTCOME_TEST: available if not yet revealed.
        - XRD: available if unrevealed and raw scan exists in artifact index.
        - REFINEMENT: available if unrevealed, refinement artifact exists, AND XRD is already observed.
        """
        actions: list[ScientificAction] = []
        for sid in self._samples_map:
            # 1. OUTCOME_TEST
            if sid not in self._revealed_outcomes:
                actions.append(
                    ScientificAction(
                        action_id=f"act_outcome_{sid}",
                        candidate_id=sid,
                        action_type="OUTCOME_TEST",
                        estimated_cost=ALAB_MODALITY_OUTCOME_TEST.cost,
                        metadata={"modality": "OUTCOME_TEST"},
                    )
                )

            # 2. XRD
            if sid not in self._revealed_xrd_spectra and self._artifact_index.has_artifact(sid, "XRD"):
                actions.append(
                    ScientificAction(
                        action_id=f"act_xrd_{sid}",
                        candidate_id=sid,
                        action_type="XRD",
                        estimated_cost=ALAB_MODALITY_XRD.cost,
                        metadata={"modality": "XRD"},
                    )
                )

            # 3. REFINEMENT (Requires XRD prerequisite)
            if (
                sid in self._revealed_xrd_spectra
                and sid not in self._revealed_refinements
                and self._artifact_index.has_artifact(sid, "REFINEMENT")
            ):
                actions.append(
                    ScientificAction(
                        action_id=f"act_ref_{sid}",
                        candidate_id=sid,
                        action_type="REFINEMENT",
                        estimated_cost=ALAB_MODALITY_REFINEMENT.cost,
                        metadata={"modality": "REFINEMENT"},
                    )
                )

        return actions

    def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Reveals historical experimental evidence for the requested candidate and modality."""
        cid = action.candidate_id
        if cid not in self._samples_map:
            raise KeyError(f"Candidate '{cid}' not found in A-Lab dataset.")

        sample = self._samples_map[cid]
        norm_type = normalize_action_type(action.action_type)
        mod_hint = action.metadata.get("modality", norm_type)

        if mod_hint == "OUTCOME_TEST" or norm_type in ("PROPERTY", "OUTCOME_TEST"):
            # Quantitative reaction conversion from ledger outcome
            cat = sample.get("outcome", {}).get("reaction_category")
            if cat == "completely_reacted":
                score = 1.0
            elif cat == "transformed":
                score = 0.75
            elif cat == "partially_reacted":
                score = 0.5
            else:
                score = 0.0

            self._revealed_outcomes[cid] = score
            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type=action.action_type,
                revealed_data={
                    "reaction_conversion": score,
                    "reaction_category": str(cat),
                },
                canonical_observation=score,
                provenance={
                    "dataset": "A-Lab Precursor Genome",
                    "execution_mode": "offline_replay",
                    "sample_id": cid,
                },
            )

        elif mod_hint == "XRD" or norm_type == "XRD":
            ref = self._artifact_index.get_artifact_ref(cid, "XRD")
            if ref is None:
                raise RuntimeError(f"No XRD artifact found for candidate '{cid}'.")

            # Parse XRD spectrum from XML bytes
            raw_bytes = self._artifact_index.read_artifact_bytes(ref)
            root = ET.fromstring(raw_bytes)
            intensities: list[float] = []
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag in ("intensities", "counts") and elem.text:
                    intensities = [float(x) for x in elem.text.split()]
                    break

            if not intensities:
                intensities = [0.0] * 450

            # Resample / standardize to 450 grid points
            raw_arr = np.asarray(intensities, dtype=np.float64)
            if len(raw_arr) != 450:
                grid_indices = np.linspace(0, len(raw_arr) - 1, 450)
                norm_spec = np.interp(grid_indices, np.arange(len(raw_arr)), raw_arr)
            else:
                norm_spec = raw_arr

            # Normalize intensity
            max_val = np.max(norm_spec)
            if max_val > 0:
                norm_spec = norm_spec / max_val

            emb = self._xrd_extractor.transform(norm_spec)
            self._revealed_xrd_spectra[cid] = norm_spec
            self._xrd_embeddings[cid] = emb

            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type=action.action_type,
                revealed_data={
                    "normalized_intensity": norm_spec.tolist(),
                    "xrd_embedding": emb.tolist(),
                },
                canonical_observation=emb,
                provenance={
                    "dataset": "A-Lab Precursor Genome",
                    "execution_mode": "offline_replay",
                    "sample_id": cid,
                    "artifact_ref": ref.to_dict(),
                },
            )

        elif mod_hint == "REFINEMENT" or norm_type in ("REFINEMENT", "CHARACTERIZATION"):
            ref = self._artifact_index.get_artifact_ref(cid, "REFINEMENT")
            if ref is None:
                raise RuntimeError(f"No Refinement artifact found for candidate '{cid}'.")

            raw_bytes = self._artifact_index.read_artifact_bytes(ref)
            try:
                pkl_data = _SafeALabUnpickler(io.BytesIO(raw_bytes)).load()
            except Exception:
                pkl_data = {}

            phase_weights = {}
            if isinstance(pkl_data, dict):
                phase_weights = pkl_data.get("phase_weights", {})
                rwp = float(pkl_data.get("best_rwp", 5.0))
            else:
                phase_weights = getattr(pkl_data, "phase_weights", {})
                rwp = float(getattr(pkl_data, "best_rwp", 5.0))

            if not isinstance(phase_weights, dict):
                phase_weights = {}

            # Target fraction estimate from non-precursor phases
            target_frac = 0.0
            precursor_frac = 0.0
            for pname, w in phase_weights.items():
                if any(p in str(pname) for p in ["Ag2O", "BaO", "Al", "Fe", "Mn", "Co", "Ti", "precursor"]):
                    precursor_frac += float(w)
                else:
                    target_frac += float(w)

            target_frac = float(np.clip(target_frac, 0.0, 1.0))
            ref_vec = np.array([target_frac, precursor_frac, float(len(phase_weights)), rwp], dtype=np.float64)
            self._revealed_refinements[cid] = ref_vec

            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type=action.action_type,
                revealed_data={
                    "refinement_features": ref_vec.tolist(),
                    "phase_weights": {str(k): float(v) for k, v in phase_weights.items()},
                    "rwp": rwp,
                    "target_fraction": target_frac,
                },
                canonical_observation=ref_vec,
                provenance={
                    "dataset": "A-Lab Precursor Genome",
                    "execution_mode": "offline_replay",
                    "sample_id": cid,
                    "artifact_ref": ref.to_dict(),
                },
            )

        raise ValueError(f"Unsupported action modality '{mod_hint}' in A-Lab domain.")

    def get_representation_snapshot(self, modality_name: str) -> RepresentationSnapshot | None:
        """Returns frozen representation basis snapshot."""
        if normalize_action_type(modality_name) == "XRD":
            return self._xrd_extractor.get_snapshot(representation_id="alab_xrd_pca", modality_name="XRD")
        return None

    def transform_with_snapshot(
        self,
        modality_name: str,
        raw_observation: Any,
        snapshot: RepresentationSnapshot,
    ) -> Any:
        """Transforms a raw measurement under the frozen snapshot basis."""
        if normalize_action_type(modality_name) == "XRD":
            if isinstance(raw_observation, dict) and "normalized_intensity" in raw_observation:
                spec = raw_observation["normalized_intensity"]
            elif isinstance(raw_observation, (list, tuple, np.ndarray)):
                spec = raw_observation
            else:
                return raw_observation
            return self._xrd_extractor.transform_with_snapshot(spec, snapshot)
        return raw_observation

    def update_representation_after_evidence(
        self,
        modality_name: str,
        candidate_id: str,
        raw_observation: Any,
    ) -> None:
        """Refits representation basis strictly after Bayesian evidence update."""
        if normalize_action_type(modality_name) == "XRD":
            if isinstance(raw_observation, dict) and "normalized_intensity" in raw_observation:
                spec = raw_observation["normalized_intensity"]
            elif isinstance(raw_observation, (list, tuple, np.ndarray)):
                spec = raw_observation
            else:
                return
            self._revealed_xrd_spectra[candidate_id] = np.asarray(spec, dtype=np.float64)
            self._xrd_extractor.fit(list(self._revealed_xrd_spectra.values()))
            self._xrd_embeddings = self._xrd_extractor.transform_batch(self._revealed_xrd_spectra)

    def get_observations_by_modality(self) -> dict[str, dict[str, Any]]:
        """Returns all revealed observations grouped by modality name."""
        return {
            "OUTCOME_TEST": dict(self._revealed_outcomes),
            "XRD": dict(self._xrd_embeddings),
            "REFINEMENT": dict(self._revealed_refinements),
        }

    def get_current_observations(self) -> Mapping[str, Mapping[str, Any]]:
        return self.get_observations_by_modality()

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        return [ALAB_OBJECTIVE_REACTION_CONVERSION]

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        return [ALAB_MODALITY_XRD, ALAB_MODALITY_REFINEMENT, ALAB_MODALITY_OUTCOME_TEST]

    def get_hypothesis_provider(self) -> HypothesisProvider:
        return self._hypothesis_provider

    def get_default_initial_actions(
        self,
        n_candidates: int = 3,
        pairing_strategy: str = "joint",
        seed: int = 42,
    ) -> list[ScientificAction]:
        """Generates reproducible bootstrap actions for A-Lab campaign initialization."""
        rng = np.random.default_rng(seed)
        cids = [s for s in self._samples_map if self._artifact_index.has_artifact(s, "XRD")]
        rng.shuffle(cids)
        chosen_cids = cids[:n_candidates]

        actions: list[ScientificAction] = []
        for cid in chosen_cids:
            actions.append(
                ScientificAction(
                    action_id=f"init_outcome_{cid}",
                    candidate_id=cid,
                    action_type="OUTCOME_TEST",
                    estimated_cost=ALAB_MODALITY_OUTCOME_TEST.cost,
                    metadata={"modality": "OUTCOME_TEST"},
                )
            )
            if pairing_strategy == "joint":
                actions.append(
                    ScientificAction(
                        action_id=f"init_xrd_{cid}",
                        candidate_id=cid,
                        action_type="XRD",
                        estimated_cost=ALAB_MODALITY_XRD.cost,
                        metadata={"modality": "XRD"},
                    )
                )
        return actions
