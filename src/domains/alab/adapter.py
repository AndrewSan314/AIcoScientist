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

from src.domains.alab.artifact_index import ALabArtifactIndex, ArtifactRef
from src.domains.alab.chemistry import parse_chemical_formula, parse_refinement_phases
from src.domains.alab.canonical import get_canonical_refinement_case, get_canonical_scan
from src.domains.alab.config import (
    ALAB_CANONICAL_PRECURSORS,
    ALAB_CANDIDATE_FEATURE_NAMES,
    ALAB_DOMAIN_CONFIG,
    ALAB_OBJECTIVE_REACTION_OUTCOME,
)
from src.domains.alab.feature_encoder import ALabFeatureEncoder
from src.domains.alab.hypotheses import ALabHypothesisProvider
from src.science.actions import (
    ActionType,
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
    HypothesisProvider,
    HypothesisTrainingContext,
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


class ALabDomainAdapter(MaterialDomainAdapter, ObservationRepresentationManager):
    """Scientific domain adapter for the autonomous A-Lab Precursor Genome solid-state synthesis campaign.

    Implements:
    1. Offline candidate pool firewall (strictly pre-experiment features).
    2. Modality prerequisite enforcement (REFINEMENT strictly requires completed XRD).
    3. Physical 2theta axis extraction and interpolation for powder XRD scans.
    4. Deterministic chemical phase matching for Rietveld refinements.
    5. ObservationRepresentationManager protocol with frozen PCA snapshots.
    """

    def __init__(
        self,
        data_dir: str = "data/external/precursor_genome_2026",
        cache_dir: str = "data/derived/alab",
        xrd_embedding_dim: int = 8,
        config: MaterialDomainConfig | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._config = config or ALAB_DOMAIN_CONFIG

        self._encoder = ALabFeatureEncoder(ALAB_CANONICAL_PRECURSORS)
        self._artifact_index = ALabArtifactIndex(data_dir=self._data_dir, cache_dir=self._cache_dir)
        self._hypothesis_provider = ALabHypothesisProvider()

        # State collections
        self._samples_by_id: dict[str, dict[str, Any]] = {}
        self._candidate_features: dict[str, np.ndarray] = {}
        self._candidate_pool_df: pd.DataFrame | None = None

        # Revealed observations (offline simulation state)
        self._revealed_xrd_spectra: dict[str, np.ndarray] = {}
        self._xrd_embeddings: dict[str, np.ndarray] = {}
        self._revealed_refinements: dict[str, np.ndarray] = {}
        self._revealed_outcomes: dict[str, float] = {}

        # Representation manager for powder XRD
        self._xrd_extractor = XRDRepresentationExtractor(
            n_components=xrd_embedding_dim,
            num_grid_points=450,
            representation_id="alab_xrd_pca",
        )

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Loads and indexes samples from ledger_precursor_genome.json or test fixtures."""
        ledger_path = os.path.join(self._data_dir, "ledger_precursor_genome.json")
        if not os.path.exists(ledger_path):
            fixture_path = os.path.join(self._data_dir, "samples.json")
            if os.path.exists(fixture_path):
                ledger_path = fixture_path
            else:
                logger.warning("No A-Lab ledger found at '%s' or fixture '%s'. Empty domain.", ledger_path, fixture_path)
                self._candidate_pool_df = pd.DataFrame()
                return

        with open(ledger_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples: list[dict[str, Any]] = []
        if isinstance(raw_data, dict):
            samples = raw_data.get("samples", [])
        elif isinstance(raw_data, list):
            samples = raw_data

        # Build or load streaming artifact index
        self._artifact_index.build_or_load(samples=samples)

        rows = []
        for s in samples:
            cid = str(s.get("sample_id"))
            self._samples_by_id[cid] = s
            feat_vec = self._encoder.encode_candidate(s)
            self._candidate_features[cid] = feat_vec

            raw_meta = self._encoder.extract_raw_metadata(s)
            row_dict = {
                "candidate_id": cid,
                "target_compound": raw_meta.get("target_compound", ""),
                "precursor_1": raw_meta.get("precursor_1", ""),
                "precursor_2": raw_meta.get("precursor_2", ""),
                "heating_temperature_c": raw_meta.get("heating_temperature_c", 200.0),
                "heating_time_minutes": raw_meta.get("heating_time_minutes", 60.0),
            }
            # Populate exact canonical feature columns
            for i, col_name in enumerate(self._encoder.feature_names):
                row_dict[col_name] = float(feat_vec[i])

            rows.append(row_dict)

        self._candidate_pool_df = pd.DataFrame(rows)
        logger.info("Loaded A-Lab candidate pool with %d candidates.", len(self._candidate_pool_df))

    @property
    def domain_id(self) -> str:
        return self._config.domain_id

    def get_config(self) -> MaterialDomainConfig:
        return self._config

    @property
    def config(self) -> MaterialDomainConfig:
        return self._config

    @property
    def candidate_features(self) -> tuple[str, ...]:
        return self._config.candidate_features

    @property
    def objectives(self) -> tuple[ObjectiveDefinition, ...]:
        return self._config.objectives

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        return self._config.objectives

    @property
    def modalities(self) -> tuple[ModalityDefinition, ...]:
        return self._config.modalities

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        return self._config.modalities

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns candidate pool containing strictly pre-experiment features."""
        if self._candidate_pool_df is None:
            return pd.DataFrame()
        return self._candidate_pool_df.copy()

    def get_candidate_features(self, candidate_id: str) -> np.ndarray:
        """Returns pre-experiment feature vector for candidate_id matching candidate_features."""
        if candidate_id not in self._candidate_features:
            raise KeyError(f"Candidate '{candidate_id}' not found in A-Lab domain.")
        return np.copy(self._candidate_features[candidate_id])

    def get_hypothesis_provider(self) -> HypothesisProvider:
        return self._hypothesis_provider

    def has_revealable_outcome(self, candidate_id: str) -> bool:
        """Determines whether a candidate has an experimentally classified reaction outcome."""
        sample = self._samples_by_id.get(str(candidate_id), {})
        outcome = sample.get("outcome") or {}
        cat = outcome.get("reaction_category")
        return cat is not None and str(cat).strip().lower() != "none"

    def list_valid_actions(self) -> list[ScientificAction]:
        """Lists currently eligible actions enforcing prerequisites and outcome availability."""
        actions: list[ScientificAction] = []
        if self._candidate_pool_df is None or self._candidate_pool_df.empty:
            return actions

        for cid in self._candidate_pool_df["candidate_id"]:
            # 1. XRD action (if not yet measured and artifact exists)
            if cid not in self._revealed_xrd_spectra and self._artifact_index.get_artifact_ref(cid, "XRD") is not None:
                actions.append(
                    ScientificAction(
                        action_id=f"XRD_{cid}",
                        candidate_id=cid,
                        action_type="XRD",
                        estimated_cost=1.0,
                        metadata={"description": f"Powder XRD 2theta scan for {cid}", "modality_hint": "XRD"},
                    )
                )

            # 2. REFINEMENT action (strictly requires completed XRD on cid and canonical/artifact refinement data)
            sample_obj = self._samples_by_id.get(cid, {})
            can_scan, _, _ = get_canonical_scan(sample_obj)
            can_case, _, _ = get_canonical_refinement_case(can_scan)
            has_ref = (
                can_case is not None
                or self._artifact_index.get_artifact_ref(cid, "REFINEMENT") is not None
            )
            if cid in self._revealed_xrd_spectra and cid not in self._revealed_refinements and has_ref:
                actions.append(
                    ScientificAction(
                        action_id=f"REFINEMENT_{cid}",
                        candidate_id=cid,
                        action_type="REFINEMENT",
                        estimated_cost=0.5,
                        metadata={"description": f"Rietveld phase refinement for {cid}", "modality_hint": "REFINEMENT"},
                    )
                )

            # 3. OUTCOME_TEST action (strictly if outcome is classified in ledger and not yet tested)
            if cid not in self._revealed_outcomes and self.has_revealable_outcome(cid):
                actions.append(
                    ScientificAction(
                        action_id=f"OUTCOME_{cid}",
                        candidate_id=cid,
                        action_type="OUTCOME_TEST",
                        estimated_cost=2.0,
                        metadata={"description": f"Synthesis outcome test for {cid}", "modality_hint": "OUTCOME_TEST"},
                    )
                )

        return actions

    def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Executes experimental observation action in offline replay mode with strict physical parsing."""
        cid = str(action.candidate_id)
        if cid not in self._samples_by_id:
            raise KeyError(f"Candidate '{cid}' not found in A-Lab domain.")

        sample = self._samples_by_id[cid]
        norm_type = normalize_action_type(action.action_type)
        mod_hint = str(action.metadata.get("modality_hint") or action.action_type).upper()

        if mod_hint == "OUTCOME_TEST" or norm_type in ("OUTCOME_TEST", "PROPERTY"):
            outcome_dict = sample.get("outcome") or {}
            cat = outcome_dict.get("reaction_category")

            category_utility_map = {
                "completely_reacted": 1.0,
                "transformed": 0.75,
                "partially_reacted": 0.5,
                "unreacted": 0.0,
            }

            if cat is None or str(cat).strip().lower() == "none":
                utility_val = None
                is_labeled = False
                canonical_obs = None
            else:
                utility_val = float(category_utility_map.get(cat, 0.0))
                is_labeled = True
                self._revealed_outcomes[cid] = utility_val
                canonical_obs = utility_val

            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type=action.action_type,
                revealed_data={
                    "reaction_outcome_utility": utility_val,
                    "reaction_category": cat,
                    "is_labeled": is_labeled,
                    "category_notes": outcome_dict.get("category_notes"),
                    "physical_failure": sample.get("physical_failure"),
                },
                canonical_observation=canonical_obs,
                provenance={
                    "dataset": "A-Lab Precursor Genome",
                    "dataset_key": "precursor_genome_2026",
                    "execution_mode": "offline_replay",
                    "sample_id": cid,
                    "is_labeled": is_labeled,
                    "reaction_category": cat,
                    "reaction_outcome_utility": utility_val,
                    "domain_version": "1.0.0",
                },
            )

        elif mod_hint == "XRD" or norm_type == "XRD":
            ref = self._artifact_index.get_artifact_ref(cid, "XRD")
            if ref is None:
                raise RuntimeError(f"No XRD artifact found for candidate '{cid}'.")

            raw_bytes = self._artifact_index.read_artifact_bytes(ref)
            try:
                root = ET.fromstring(raw_bytes)
            except Exception as e:
                raise ValueError(f"Malformed XRD XML file for candidate '{cid}': {e}") from e

            # Extract physical 2theta axis start and end positions
            start_pos: float | None = None
            end_pos: float | None = None
            for elem in root.iter():
                if elem.tag.endswith("positions") and elem.attrib.get("axis") == "2Theta":
                    for child in elem:
                        if child.tag.endswith("startPosition") and child.text:
                            start_pos = float(child.text.strip())
                        elif child.tag.endswith("endPosition") and child.text:
                            end_pos = float(child.text.strip())
                elif elem.tag.endswith("startPosition") and elem.text and start_pos is None:
                    try:
                        start_pos = float(elem.text.strip())
                    except (ValueError, TypeError):
                        pass
                elif elem.tag.endswith("endPosition") and elem.text and end_pos is None:
                    try:
                        end_pos = float(elem.text.strip())
                    except (ValueError, TypeError):
                        pass

            can_scan, can_scan_idx, scan_method = get_canonical_scan(sample)
            # If 2theta axis missing in XML, check explicit scan metadata from canonical scan
            if start_pos is None or end_pos is None:
                xrd_settings = (can_scan or {}).get("xrd_settings", {})
                r2t = xrd_settings.get("range_2theta")
                if isinstance(r2t, (list, tuple)) and len(r2t) == 2:
                    start_pos = float(r2t[0])
                    end_pos = float(r2t[1])

            if start_pos is None or end_pos is None:
                raise ValueError(
                    f"Missing physical 2Theta axis metadata for XRD scan of candidate '{cid}'. "
                    f"Neither XML positions nor xrd_settings.range_2theta provide axis limits."
                )

            # Extract raw intensities / counts
            intensities: list[float] = []
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag in ("intensities", "counts") and elem.text:
                    intensities = [float(x) for x in elem.text.split()]
                    break

            if not intensities:
                raise ValueError(f"Empty or missing XRD intensity counts in scan for candidate '{cid}'.")

            raw_arr = np.asarray(intensities, dtype=np.float64)
            # Physical 2theta axis interpolation onto canonical 450-point 10-100 deg grid
            phys_2theta = np.linspace(start_pos, end_pos, len(raw_arr))
            canonical_grid = np.linspace(10.0, 100.0, 450)
            norm_spec = np.interp(canonical_grid, phys_2theta, raw_arr)

            # Normalize intensity
            max_val = float(np.max(norm_spec))
            if max_val > 0:
                norm_spec = norm_spec / max_val

            emb = self._xrd_extractor.transform(norm_spec)
            self._revealed_xrd_spectra[cid] = norm_spec
            self._xrd_embeddings[cid] = emb

            snapshot = self.get_representation_snapshot("XRD")
            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type=action.action_type,
                revealed_data={
                    "normalized_intensity": norm_spec.tolist(),
                    "xrd_embedding": emb.tolist(),
                    "two_theta_start": start_pos,
                    "two_theta_end": end_pos,
                },
                canonical_observation=emb,
                provenance={
                    "dataset": "A-Lab Precursor Genome",
                    "dataset_key": "precursor_genome_2026",
                    "execution_mode": "offline_replay",
                    "sample_id": cid,
                    "source_archive": ref.archive_path,
                    "archive_member_path": ref.member_path,
                    "member_size_bytes": ref.size_bytes,
                    "archive_checksum": ref.checksum,
                    "preprocessing_version": "physical_2theta_v1",
                    "selected_scan_index": can_scan_idx,
                    "selection_method": scan_method,
                    "representation_id": "alab_xrd_pca",
                    "representation_version": snapshot.version if snapshot else 0,
                    "representation_fingerprint": snapshot.fingerprint if snapshot else None,
                    "domain_version": "1.0.0",
                },
            )

        elif mod_hint == "REFINEMENT" or norm_type in ("REFINEMENT", "CHARACTERIZATION"):
            # Enforce prerequisite
            if cid not in self._revealed_xrd_spectra:
                raise RuntimeError(f"Prerequisite not met: REFINEMENT on '{cid}' requires completed XRD observation.")

            target_compound = str(sample.get("target_compound", ""))
            precursors = [p.get("formula", "") if isinstance(p, dict) else str(p) for p in sample.get("precursors", [])]

            # 1. Prefer structured refinement data from canonical scan & case in ledger
            can_scan, can_scan_idx, scan_method = get_canonical_scan(sample)
            can_case, can_case_idx, case_method = get_canonical_refinement_case(can_scan)

            phase_weights: dict[str, float] = {}
            rwp = 5.0
            rank = None
            used_source = "ledger"

            if can_case is not None:
                phase_weights = {str(k): float(v) for k, v in can_case.get("phase_weights", {}).items()}
                rwp = float(can_case.get("rwp", 5.0) or 5.0)
                rank = can_case.get("rank")
            else:
                # 2. Fallback to reading pickle artifact
                ref = self._artifact_index.get_artifact_ref(cid, "REFINEMENT")
                if ref is None:
                    raise RuntimeError(f"No Refinement artifact found for candidate '{cid}'.")

                raw_bytes = self._artifact_index.read_artifact_bytes(ref)
                try:
                    pkl_data = _SafeALabUnpickler(io.BytesIO(raw_bytes)).load()
                except Exception as e:
                    raise ValueError(f"Failed to parse refinement pickle for candidate '{cid}': {e}") from e

                if isinstance(pkl_data, dict):
                    phase_weights = pkl_data.get("phase_weights", {})
                    rwp = float(pkl_data.get("best_rwp", 5.0))
                    rank = pkl_data.get("rank")
                else:
                    phase_weights = getattr(pkl_data, "phase_weights", {})
                    rwp = float(getattr(pkl_data, "best_rwp", 5.0))
                    rank = getattr(pkl_data, "rank", None)
                used_source = "pickle_artifact"

            if not isinstance(phase_weights, dict):
                phase_weights = {}

            # Perform exact chemical phase matching
            ref_parsed = parse_refinement_phases(
                phase_weights=phase_weights,
                target_formula=target_compound,
                precursor_formulas=precursors,
                rwp=rwp,
            )

            ref_vec = np.asarray(ref_parsed["feature_vector"], dtype=np.float64)
            self._revealed_refinements[cid] = ref_vec

            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=cid,
                action_type=action.action_type,
                revealed_data=ref_parsed,
                canonical_observation=ref_vec,
                provenance={
                    "dataset": "A-Lab Precursor Genome",
                    "dataset_key": "precursor_genome_2026",
                    "execution_mode": "offline_replay",
                    "sample_id": cid,
                    "refinement_source": used_source,
                    "selected_scan_index": can_scan_idx,
                    "selected_case_index": can_case_idx,
                    "refinement_rank": rank,
                    "r_wp": rwp,
                    "selection_method": case_method,
                    "domain_version": "1.0.0",
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
        """Transforms a raw measurement using the specific frozen representation snapshot basis."""
        if normalize_action_type(modality_name) == "XRD":
            if isinstance(raw_observation, dict) and "normalized_intensity" in raw_observation:
                spec = raw_observation["normalized_intensity"]
            else:
                spec = raw_observation
            raw_arr = np.asarray(spec, dtype=np.float64)
            return self._xrd_extractor.transform_with_snapshot(raw_arr, snapshot)
        elif normalize_action_type(modality_name) == "REFINEMENT":
            if isinstance(raw_observation, dict) and "feature_vector" in raw_observation:
                return np.asarray(raw_observation["feature_vector"], dtype=np.float64)
            return np.asarray(raw_observation, dtype=np.float64)
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
            raw_arr = np.asarray(spec, dtype=np.float64)
            self._revealed_xrd_spectra[candidate_id] = raw_arr
            if len(self._revealed_xrd_spectra) >= 2:
                self._xrd_extractor.fit(list(self._revealed_xrd_spectra.values()))
                self._xrd_embeddings = self._xrd_extractor.transform_batch(self._revealed_xrd_spectra)

    def get_revealed_xrd_embeddings(self) -> dict[str, np.ndarray]:
        """Returns current embeddings for all revealed XRD candidates under the unified current basis."""
        return dict(self._xrd_embeddings)

    def get_current_observations(self) -> dict[str, dict[str, Any]]:
        """Returns all currently revealed candidate observations in the latest representation basis."""
        return self.get_observations_by_modality()

    def get_default_initial_actions(
        self,
        n_candidates: int = 4,
        pairing_strategy: str = "joint",
        seed: int = 42,
    ) -> list[ScientificAction]:
        """Generates reproducible bootstrap initial actions."""
        if self._candidate_pool_df is None or self._candidate_pool_df.empty:
            return []

        rng = np.random.default_rng(seed)
        all_cids = list(self._candidate_pool_df["candidate_id"])
        # Filter for candidates that have an XRD artifact available and revealable outcome for joint pairing
        valid_cids = [
            cid for cid in all_cids
            if self._artifact_index.get_artifact_ref(cid, "XRD") is not None
            and (pairing_strategy != "joint" or self.has_revealable_outcome(cid))
        ]
        if not valid_cids:
            valid_cids = all_cids
        chosen_indices = rng.choice(len(valid_cids), size=min(n_candidates, len(valid_cids)), replace=False)
        chosen_cids = [valid_cids[i] for i in chosen_indices]

        actions = []
        for cid in chosen_cids:
            actions.append(
                ScientificAction(
                    action_id=f"BOOTSTRAP_XRD_{cid}",
                    candidate_id=cid,
                    action_type="XRD",
                    estimated_cost=1.0,
                    metadata={"modality_hint": "XRD"},
                )
            )
            if pairing_strategy == "joint":
                actions.append(
                    ScientificAction(
                        action_id=f"BOOTSTRAP_OUTCOME_{cid}",
                        candidate_id=cid,
                        action_type="OUTCOME_TEST",
                        estimated_cost=2.0,
                        metadata={"modality_hint": "OUTCOME_TEST"},
                    )
                )
        return actions

    def get_observations_by_modality(self) -> dict[str, dict[str, Any]]:
        """Returns revealed observations structured by modality name."""
        return {
            "XRD": dict(self._xrd_embeddings),
            "REFINEMENT": dict(self._revealed_refinements),
            "OUTCOME_TEST": dict(self._revealed_outcomes),
        }
