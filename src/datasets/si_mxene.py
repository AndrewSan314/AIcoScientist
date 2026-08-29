from __future__ import annotations

from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.chemistry_rules import validate_candidate as validate_chemistry_candidate
from src.edx_features import add_edx_features
from src.experiment_store import ingest_csvs, load_source_tables
from src.utils import (
    ensure_sample_data,
)

from .base import DatasetAdapter
from .si_mxene_spec import (
    CHEMISTRY_ALPHA,
    MODEL_FEATURES,
    PROCESS_FEATURES,
    SEARCH_SPACE,
    SI_MXENE_SPEC,
)


class SiMxeneAdapter(DatasetAdapter):
    @property
    def spec(self) -> DatasetSpec:
        return SI_MXENE_SPEC

    def load(self) -> pd.DataFrame:
        ensure_sample_data()
        ingest_csvs()
        sources = load_source_tables()
        df = sources["process_data"]
        for name in ("sem_features", "edx_data", "electrochem_data"):
            df = df.merge(sources[name], on="sample_id", validate="one_to_one")
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["si_mxene_ratio"] = result["si_content"] / result["mxene_content"]
        result = add_edx_features(result)
        result["capacity_fade"] = result["initial_capacity"] - result["capacity_100"]
        return result

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        rows = []
        keys = list(SEARCH_SPACE)
        for values in product(*(SEARCH_SPACE[key] for key in keys)):
            candidate = dict(zip(keys, values))
            candidate["carbon_content"] = (
                100
                - candidate["si_content"]
                - candidate["mxene_content"]
                - candidate["alginate_content"]
            )
            rows.append(candidate)
        return pd.DataFrame(rows)

    def validate_candidate(self, candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
        result = validate_chemistry_candidate(dict(candidate))
        return result.valid, result.violations

    def candidate_metadata(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        result = validate_chemistry_candidate(dict(candidate))
        return {
            "chemistry_score": result.chemistry_score,
            "volume_expansion_risk": result.volume_expansion_risk,
            "violations": "; ".join(result.violations),
        }

    def build_candidate_features(
        self,
        candidates: pd.DataFrame,
        observed: pd.DataFrame,
        fill_values: Mapping[str, Any],
    ) -> pd.DataFrame:
        result = super().build_candidate_features(candidates, observed, fill_values)
        result["pressing_pressure"] = float(observed["pressing_pressure"].median())
        result["si_mxene_ratio"] = result["si_content"] / result["mxene_content"]
        result["si_ti_ratio"] = result["si_percent"] / result["ti_percent"]
        result["c_o_ratio"] = result["c_percent"] / result["o_percent"]
        return result[self.spec.feature_columns]

    def distance_columns(self) -> list[str]:
        return PROCESS_FEATURES

    def adjust_acquisition_score(
        self,
        candidates: pd.DataFrame,
        acquisition_score: pd.Series,
        predicted_mean: np.ndarray,
    ) -> pd.Series:
        scale = pd.Series(predicted_mean).std()
        return acquisition_score + CHEMISTRY_ALPHA * candidates["chemistry_score"] * scale

    def format_recommendations(
        self,
        candidates: pd.DataFrame,
        observed: pd.DataFrame,
    ) -> pd.DataFrame:
        result = candidates.copy()

        def reason(row: pd.Series) -> str:
            parts = []
            if row["confidence"] == "low":
                parts.append("Exploratory recipe with higher GP uncertainty.")
            elif row["confidence"] == "high":
                parts.append("Recipe near well-characterized process window.")

            si = row["si_content"]
            mxene = row["mxene_content"]
            alginate = row["alginate_content"]
            carbon = row["carbon_content"]
            conductive = mxene + carbon
            if si >= 70:
                parts.append(f"High Si ({si}%) boosts capacity but increases volume expansion risk.")
            elif si <= 50:
                parts.append(f"Lower Si ({si}%) reduces capacity but improves cycling stability.")
            if mxene >= 20 and alginate >= 10:
                parts.append("Balanced MXene network and alginate binder level.")
            if conductive >= 25:
                parts.append(f"Strong conductive network (MXene+C={conductive}%).")
            elif conductive < 18:
                parts.append(f"Conductive phase near percolation threshold ({conductive}%).")
            ratio = si / max(mxene, 1)
            if 3.0 <= ratio <= 5.0:
                parts.append(f"Si/MXene ratio ({ratio:.1f}) in optimal encapsulation window.")
            if row.get("chemistry_score", 0) >= 0.7:
                parts.append("Excellent chemistry score — recipe in physical sweet spot.")
            return " ".join(parts) or "Conservative recipe near existing stable process window."

        result["predicted_retention"] = result["predicted_mean"]
        result["predicted_retention_mean"] = result["predicted_mean"]
        result["predicted_retention_std"] = result["predicted_std"]
        result["reason"] = result.apply(reason, axis=1)
        output_columns = [
            "si_content",
            "mxene_content",
            "alginate_content",
            "carbon_content",
            "drying_temp",
            "mixing_time",
            "pressing_pressure",
            "predicted_retention",
            "predicted_retention_mean",
            "predicted_retention_std",
            "acquisition_score",
            "chemistry_score",
            "volume_expansion_risk",
            "final_score",
            "confidence",
            "reason",
        ]
        result = result[["rank", *output_columns]]
        for column in output_columns:
            if column not in {"confidence", "reason"}:
                result[column] = result[column].round(2)
        return result
