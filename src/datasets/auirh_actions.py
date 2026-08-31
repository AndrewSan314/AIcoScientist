from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.auirh import (
    AUIRH_CANDIDATE_ID_COLUMN,
    AUIRH_DEFAULT_EDX_DIR,
    AUIRH_DEFAULT_SECCM_DIR,
    AUIRH_DEFAULT_XRD_DIR,
    AUIRH_FALLBACK_EDX_DIR,
    AUIRH_FALLBACK_SECCM_DIR,
    AUIRH_FALLBACK_XRD_DIR,
    EXPECTED_TOTAL_MEASURED_COUNT,
    _resolve_dir,
    generate_auirh_candidate_id,
    load_raw_auirh_dataset,
)
from src.science.actions import (
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
)

logger = logging.getLogger(__name__)


def parse_xrd_xy_file(filepath: Path | str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Parses a real Au-Ir-Rh .xy diffractogram file.

    Line 1 contains key-value metadata headers.
    Subsequent lines contain '2theta intensity' pairs.

    Returns:
        (two_theta_array, intensity_array, metadata_dict)
    """
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"XRD diffractogram file not found at: {p.resolve()}")

    meta: dict[str, str] = {}
    two_theta_list: list[float] = []
    intensity_list: list[float] = []

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
        # Parse simple key-value quotes if present
        meta["raw_header"] = first_line

        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    tt = float(parts[0])
                    inte = float(parts[1])
                    two_theta_list.append(tt)
                    intensity_list.append(inte)
                except ValueError:
                    continue

    tt_arr = np.array(two_theta_list, dtype=np.float64)
    int_arr = np.array(intensity_list, dtype=np.float64)
    return tt_arr, int_arr, meta


class AuIrRhMultimodalOracle:
    """Offline Multimodal Oracle for the real Au-Ir-Rh experimental dataset.

    STRICT OFFLINE FIREWALL CONTRACT:
    - Encapsulates exact ground-truth measurements (XRD diffractograms & SECCM k0)
      for all 966 physical library samples.
    - Decision policies have access ONLY to:
      1. Candidate identities and composition features (Au, Ir, Rh).
      2. Previously executed and revealed XRD spectra.
      3. Previously executed and revealed electrochemical properties (k0).
    - Hidden spectra and k0 values are inaccessible until an explicit ScientificAction
      is executed via `execute_xrd()` or `execute_property()`.
    - Exact physical candidate identity: no continuous extrapolation or synthesis interpolation.
    """

    def __init__(
        self,
        edx_dir: Path | str = AUIRH_DEFAULT_EDX_DIR,
        seccm_dir: Path | str = AUIRH_DEFAULT_SECCM_DIR,
        xrd_dir: Path | str = AUIRH_DEFAULT_XRD_DIR,
        allow_duplicate_actions: bool = False,
    ) -> None:
        self.xrd_dir = _resolve_dir(Path(xrd_dir), AUIRH_FALLBACK_XRD_DIR)
        self.allow_duplicate_actions = allow_duplicate_actions

        # Load complete ground truth records
        self._raw_records_df = load_raw_auirh_dataset(
            edx_dir=edx_dir,
            seccm_dir=seccm_dir,
        )
        if len(self._raw_records_df) != EXPECTED_TOTAL_MEASURED_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_TOTAL_MEASURED_COUNT} Au-Ir-Rh records, got {len(self._raw_records_df)}"
            )

        # Index ground truth by candidate_id
        self._ground_truth_map: dict[str, dict[str, Any]] = {}
        for _, row in self._raw_records_df.iterrows():
            cid = str(row[AUIRH_CANDIDATE_ID_COLUMN])
            lib = str(row["Library"])
            area = int(row["Area"])
            xrd_filename = f"Au-Ir-Rh_{lib}_XRD_area_{area:03d}_diffractogram.xy"
            xrd_path = self.xrd_dir / xrd_filename

            self._ground_truth_map[cid] = {
                "candidate_id": cid,
                "Library": lib,
                "Area": area,
                "Au": float(row["Au"]),
                "Ir": float(row["Ir"]),
                "Rh": float(row["Rh"]),
                "k0": float(row["k0"]),
                "i_lim": float(row["i_lim"]),
                "alpha": float(row["alpha"]),
                "xrd_filename": xrd_filename,
                "xrd_path": xrd_path,
            }

        # Track revealed state
        self._revealed_xrd: dict[str, ExperimentOutcome] = {}
        self._revealed_property: dict[str, ExperimentOutcome] = {}
        self._action_history: list[ExperimentOutcome] = []

        # Find global best k0 for benchmark/debug audit only
        best_cid = max(self._ground_truth_map.keys(), key=lambda k: self._ground_truth_map[k]["k0"])
        self.global_best_candidate_id = best_cid
        self.global_best_k0 = float(self._ground_truth_map[best_cid]["k0"])

    @property
    def total_candidates(self) -> int:
        return len(self._ground_truth_map)

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the visible candidate pool table (Composition features ONLY).

        STRICT FIREWALL: Contains zero target values and zero XRD measurements.
        """
        cols = ["candidate_id", "Library", "Area", "Au", "Ir", "Rh"]
        records = [
            {
                "candidate_id": v["candidate_id"],
                "Library": v["Library"],
                "Area": v["Area"],
                "Au": v["Au"],
                "Ir": v["Ir"],
                "Rh": v["Rh"],
            }
            for v in self._ground_truth_map.values()
        ]
        return pd.DataFrame(records)[cols].copy()

    def is_xrd_observed(self, candidate_id: str) -> bool:
        return candidate_id in self._revealed_xrd

    def is_property_observed(self, candidate_id: str) -> bool:
        return candidate_id in self._revealed_property

    def execute_xrd(self, candidate_id: str, action_id: str | None = None) -> ExperimentOutcome:
        """Executes an XRD characterization action for an exact physical candidate.

        Retrieves and reveals the real measured diffractogram.
        """
        if candidate_id not in self._ground_truth_map:
            raise KeyError(f"Candidate ID '{candidate_id}' does not exist in Au-Ir-Rh candidate space.")

        if not self.allow_duplicate_actions and candidate_id in self._revealed_xrd:
            raise ValueError(f"XRD characterization already executed for candidate '{candidate_id}'.")

        entry = self._ground_truth_map[candidate_id]
        xrd_path = entry["xrd_path"]
        if not xrd_path.is_file():
            raise FileNotFoundError(f"Diffractogram file missing: {xrd_path}")

        tt, intensity, header_meta = parse_xrd_xy_file(xrd_path)

        # Standardized downsampling for fast visualization and representations (e.g. 450 points)
        # Uniform linear interpolation across [10.0, 100.0] 2theta
        tt_grid = np.linspace(10.0, 100.0, 450)
        int_downsampled = np.interp(tt_grid, tt, intensity)

        # Standard min-max normalization
        i_min = float(np.min(int_downsampled))
        i_max = float(np.max(int_downsampled))
        norm_intensity = (int_downsampled - i_min) / (i_max - i_min + 1e-12)

        act_id = action_id or f"act_xrd_{candidate_id}_{len(self._action_history)+1:04d}"
        outcome = ExperimentOutcome(
            action_id=act_id,
            candidate_id=candidate_id,
            action_type=ExperimentActionType.XRD,
            revealed_data={
                "two_theta": tt.tolist(),
                "intensity": intensity.tolist(),
                "downsampled_two_theta": tt_grid.tolist(),
                "downsampled_intensity": int_downsampled.tolist(),
                "normalized_intensity": norm_intensity.tolist(),
                "peak_two_theta": float(tt[np.argmax(intensity)]),
                "peak_intensity": float(np.max(intensity)),
            },
            provenance={
                "library": entry["Library"],
                "area": entry["Area"],
                "xrd_filename": entry["xrd_filename"],
                "num_points": len(tt),
            },
            metadata={"header": header_meta},
        )

        self._revealed_xrd[candidate_id] = outcome
        self._action_history.append(outcome)
        return outcome

    def execute_property(self, candidate_id: str, action_id: str | None = None) -> ExperimentOutcome:
        """Executes an electrochemical performance measurement action for an exact physical candidate.

        Retrieves and reveals the real measured k0, i_lim, and alpha.
        """
        if candidate_id not in self._ground_truth_map:
            raise KeyError(f"Candidate ID '{candidate_id}' does not exist in Au-Ir-Rh candidate space.")

        if not self.allow_duplicate_actions and candidate_id in self._revealed_property:
            raise ValueError(f"Property measurement already executed for candidate '{candidate_id}'.")

        entry = self._ground_truth_map[candidate_id]
        act_id = action_id or f"act_prop_{candidate_id}_{len(self._action_history)+1:04d}"

        outcome = ExperimentOutcome(
            action_id=act_id,
            candidate_id=candidate_id,
            action_type=ExperimentActionType.PROPERTY,
            revealed_data={
                "k0": float(entry["k0"]),
                "i_lim": float(entry["i_lim"]),
                "alpha": float(entry["alpha"]),
            },
            provenance={
                "library": entry["Library"],
                "area": entry["Area"],
                "source_metric": "k^0 [cm/s]",
            },
        )

        self._revealed_property[candidate_id] = outcome
        self._action_history.append(outcome)
        return outcome

    def execute(self, action: ScientificAction) -> ExperimentOutcome:
        """Executes a generic scientific action."""
        if action.action_type == ExperimentActionType.XRD:
            return self.execute_xrd(action.candidate_id, action_id=action.action_id)
        elif action.action_type == ExperimentActionType.PROPERTY:
            return self.execute_property(action.candidate_id, action_id=action.action_id)
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

    def get_revealed_xrd(self) -> dict[str, ExperimentOutcome]:
        """Returns a defensive copy of all revealed XRD experiment outcomes."""
        return {cid: copy.deepcopy(outcome) for cid, outcome in self._revealed_xrd.items()}

    def get_revealed_properties(self) -> dict[str, ExperimentOutcome]:
        """Returns a defensive copy of all revealed property experiment outcomes."""
        return {cid: copy.deepcopy(outcome) for cid, outcome in self._revealed_property.items()}

    def get_revealed_xrd_ids(self) -> list[str]:
        """Returns list of candidate IDs with revealed XRD measurements."""
        return list(self._revealed_xrd.keys())

    def get_revealed_property_ids(self) -> list[str]:
        """Returns list of candidate IDs with revealed property measurements."""
        return list(self._revealed_property.keys())

    def get_revealed_state_summary(self) -> dict[str, Any]:
        """Returns observable campaign summary without leaking unobserved hidden data."""
        best_observed_k0 = None
        best_observed_cid = None
        if self._revealed_property:
            best_observed_cid = max(
                self._revealed_property.keys(),
                key=lambda cid: self._revealed_property[cid].revealed_data["k0"],
            )
            best_observed_k0 = self._revealed_property[best_observed_cid].revealed_data["k0"]

        return {
            "total_candidates": len(self._ground_truth_map),
            "num_xrd_observed": len(self._revealed_xrd),
            "num_property_observed": len(self._revealed_property),
            "best_observed_k0": best_observed_k0,
            "best_observed_candidate_id": best_observed_cid,
            "observed_xrd_ids": list(self._revealed_xrd.keys()),
            "observed_property_ids": list(self._revealed_property.keys()),
        }

    def get_observable_dataset(self) -> pd.DataFrame:
        """Builds the observable dataset frame combining composition and revealed measurements."""
        records = []
        for cid, entry in self._ground_truth_map.items():
            has_xrd = cid in self._revealed_xrd
            has_prop = cid in self._revealed_property
            rec: dict[str, Any] = {
                "candidate_id": cid,
                "Library": entry["Library"],
                "Area": entry["Area"],
                "Au": entry["Au"],
                "Ir": entry["Ir"],
                "Rh": entry["Rh"],
                "xrd_observed": has_xrd,
                "property_observed": has_prop,
                "k0": self._revealed_property[cid].revealed_data["k0"] if has_prop else np.nan,
                "i_lim": self._revealed_property[cid].revealed_data["i_lim"] if has_prop else np.nan,
            }
            records.append(rec)
        return pd.DataFrame(records)

    def reset(self) -> None:
        """Resets revealed action state for trajectory replay."""
        self._revealed_xrd.clear()
        self._revealed_property.clear()
        self._action_history.clear()
