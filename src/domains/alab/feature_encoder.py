from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

from src.domains.alab.config import (
    ALAB_CANONICAL_PRECURSORS,
    ALAB_CANDIDATE_FEATURE_NAMES,
)


class ALabFeatureEncoder:
    """Encodes pre-experiment A-Lab candidate metadata into reproducible numeric feature vectors.

    Uses deterministic, continuous, and multi-hot categorical chemistry representations:
    - reaction_energy_ev_per_atom: Float thermodynamic energy (eV/atom)
    - heating_temperature_scaled: Float scaled synthesis temperature in [0, 1]
    - heating_time_scaled: Float scaled synthesis time in [0, 1]
    - prec_<precursor>: Multi-hot binary indicator (1.0 if precursor present, 0.0 otherwise) for 46 precursors.
    """

    def __init__(self, precursor_vocab: Sequence[str] | None = None) -> None:
        self.precursor_vocab: list[str] = (
            list(precursor_vocab) if precursor_vocab is not None else list(ALAB_CANONICAL_PRECURSORS)
        )
        self._feature_names = tuple(
            ["reaction_energy_ev_per_atom", "heating_temperature_scaled", "heating_time_scaled"]
            + [f"prec_{p}" for p in self.precursor_vocab]
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def fit(self, samples: Sequence[Mapping[str, Any]]) -> ALabFeatureEncoder:
        """Ensures precursor vocabulary covers all dataset precursors in deterministic sorted order."""
        vocab_set = set(self.precursor_vocab)
        for s in samples:
            for p in s.get("precursors", []):
                formula = p.get("formula") if isinstance(p, dict) else str(p)
                if formula:
                    vocab_set.add(formula)
        self.precursor_vocab = sorted(list(vocab_set))
        self._feature_names = tuple(
            ["reaction_energy_ev_per_atom", "heating_temperature_scaled", "heating_time_scaled"]
            + [f"prec_{p}" for p in self.precursor_vocab]
        )
        return self

    def encode_candidate(self, sample: Mapping[str, Any]) -> np.ndarray:
        """Converts a single sample's pre-experiment features to a canonical numeric array."""
        syn = sample.get("synthesis") or {}
        if not isinstance(syn, dict):
            syn = {}

        energy = float(sample.get("reaction_energy_ev_per_atom", 0.0) or 0.0)
        temp = float(syn.get("heating_temperature", 200.0) or 200.0)
        time_min = float(syn.get("heating_time", 60.0) or 60.0)

        temp_scaled = float(np.clip((temp - 200.0) / 1000.0, 0.0, 1.0))
        time_scaled = float(np.clip((time_min - 60.0) / 180.0, 0.0, 1.0))

        # Multi-hot precursor representation
        sample_precursors = set()
        for p in sample.get("precursors", []):
            f = p.get("formula") if isinstance(p, dict) else str(p)
            if f:
                sample_precursors.add(f.strip())

        prec_indicators = [1.0 if p in sample_precursors else 0.0 for p in self.precursor_vocab]

        feature_values = [energy, temp_scaled, time_scaled] + prec_indicators
        return np.array(feature_values, dtype=np.float64)

    def extract_raw_metadata(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Returns clean, human-readable candidate attributes for explanation and provenance."""
        precursors = sample.get("precursors", [])
        p1 = (
            precursors[0].get("formula", "Unknown")
            if len(precursors) > 0 and isinstance(precursors[0], dict)
            else (str(precursors[0]) if len(precursors) > 0 else "Unknown")
        )
        p2 = (
            precursors[1].get("formula", "Unknown")
            if len(precursors) > 1 and isinstance(precursors[1], dict)
            else (str(precursors[1]) if len(precursors) > 1 else "Unknown")
        )
        syn = sample.get("synthesis") or {}
        if not isinstance(syn, dict):
            syn = {}

        return {
            "sample_id": sample.get("sample_id"),
            "target_compound": sample.get("target_compound"),
            "precursor_1": p1,
            "precursor_2": p2,
            "heating_temperature_c": float(syn.get("heating_temperature", 200.0) or 200.0),
            "heating_time_minutes": float(syn.get("heating_time", 60.0) or 60.0),
            "reaction_energy_ev_per_atom": float(sample.get("reaction_energy_ev_per_atom", 0.0) or 0.0),
        }
