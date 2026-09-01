from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np


class ALabFeatureEncoder:
    """Encodes pre-experiment A-Lab candidate metadata into reproducible numeric feature vectors."""

    def __init__(self, precursor_vocab: Sequence[str] | None = None) -> None:
        self.precursor_vocab: list[str] = list(precursor_vocab) if precursor_vocab is not None else []
        self._precursor_to_idx: dict[str, int] = {p: i for i, p in enumerate(self.precursor_vocab)}

    def fit(self, samples: Sequence[Mapping[str, Any]]) -> ALabFeatureEncoder:
        """Builds vocabulary of unique precursor chemical formulas."""
        vocab_set = set()
        for s in samples:
            for p in s.get("precursors", []):
                formula = p.get("formula") or p.get("formula_clean")
                if formula:
                    vocab_set.add(formula)
        self.precursor_vocab = sorted(list(vocab_set))
        self._precursor_to_idx = {p: i for i, p in enumerate(self.precursor_vocab)}
        return self

    def encode_candidate(self, sample: Mapping[str, Any]) -> np.ndarray:
        """Converts a single sample's pre-experiment features to a numeric array."""
        precursors = sample.get("precursors", [])
        p1 = precursors[0].get("formula", "") if len(precursors) > 0 else ""
        p2 = precursors[1].get("formula", "") if len(precursors) > 1 else ""

        p1_idx = float(self._precursor_to_idx.get(p1, -1))
        p2_idx = float(self._precursor_to_idx.get(p2, -1))

        # Synthesis parameters
        syn = sample.get("synthesis") or {}
        if not isinstance(syn, dict):
            syn = {}
        temp = float(syn.get("heating_temperature", 200.0) or 200.0)
        time_min = float(syn.get("heating_time", 60.0) or 60.0)
        energy = float(sample.get("reaction_energy_ev_per_atom", 0.0) or 0.0)

        # 5-dimensional numeric descriptor vector
        return np.array(
            [
                energy,
                (temp - 200.0) / 1000.0,
                time_min / 120.0,
                p1_idx,
                p2_idx,
            ],
            dtype=np.float64,
        )

    def extract_raw_metadata(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Returns clean, human-readable candidate attributes for explanation and provenance."""
        precursors = sample.get("precursors", [])
        p1 = precursors[0].get("formula", "Unknown") if len(precursors) > 0 else "Unknown"
        p2 = precursors[1].get("formula", "Unknown") if len(precursors) > 1 else "Unknown"
        syn = sample.get("synthesis") or {}
        if not isinstance(syn, dict):
            syn = {}

        return {
            "sample_id": sample.get("sample_id"),
            "target_compound": sample.get("target_compound"),
            "precursor_1": p1,
            "precursor_2": p2,
            "heating_temperature_c": syn.get("heating_temperature"),
            "heating_time_minutes": syn.get("heating_time"),
            "reaction_energy_ev_per_atom": sample.get("reaction_energy_ev_per_atom"),
        }
