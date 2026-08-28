from itertools import product

import joblib
import numpy as np
import pandas as pd

from src.build_dataset import build_master_dataset
from src.chemistry_rules import validate_candidate, score_recipe_quality
from src.train_model import train_model
from src.utils import MASTER_FILE, MODEL_FILE, PROCESS_FEATURES, RECOMMENDATIONS_FILE


SEARCH_SPACE = {
    "si_content": range(40, 81, 5),
    "mxene_content": range(5, 36, 5),
    "alginate_content": range(5, 19, 1),
    "drying_temp": range(60, 121, 10),
    "mixing_time": range(30, 61, 15),
}
UCB_BETA = 1.0
CHEMISTRY_ALPHA = 0.5  # Weight of chemistry score in final ranking


def _nearest_distance(candidate, existing):
    ranges = existing[PROCESS_FEATURES].max() - existing[PROCESS_FEATURES].min()
    ranges = ranges.replace(0, 1)
    point = pd.Series(candidate)[PROCESS_FEATURES]
    return float(((existing[PROCESS_FEATURES] - point) / ranges).pow(2).sum(axis=1).pow(0.5).min())


def _confidence(distance, std):
    if distance <= 0.7 and std <= 2.0:
        return "high"
    if distance <= 1.2 and std <= 5.0:
        return "medium"
    return "low"


def _reason(row):
    """Generate a human-readable reason that includes chemistry insights."""
    parts = []

    # Confidence-based opening
    if row["confidence"] == "low":
        parts.append("Exploratory recipe with higher GP uncertainty.")
    elif row["confidence"] == "high":
        parts.append("Recipe near well-characterized process window.")

    # Chemistry-informed commentary
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

    si_mxene_ratio = si / max(mxene, 1)
    if 3.0 <= si_mxene_ratio <= 5.0:
        parts.append(f"Si/MXene ratio ({si_mxene_ratio:.1f}) in optimal encapsulation window.")

    # Volume expansion risk from chemistry_score
    chem_score = row.get("chemistry_score", 0)
    if chem_score >= 0.7:
        parts.append("Excellent chemistry score — recipe in physical sweet spot.")

    if not parts:
        parts.append("Conservative recipe near existing stable process window.")

    return " ".join(parts)


def recommend_top(n=3):
    if not MASTER_FILE.exists():
        build_master_dataset()
    if not MODEL_FILE.exists():
        train_model()

    bundle = joblib.load(MODEL_FILE)
    if "gp_model" not in bundle or "scaler" not in bundle:
        train_model()
        bundle = joblib.load(MODEL_FILE)

    gp_model = bundle["gp_model"]
    scaler = bundle["scaler"]
    features = bundle["features"]
    fill_values = bundle["fill_values"]
    master = pd.read_csv(MASTER_FILE)
    seen = set(
        master[["si_content", "mxene_content", "alginate_content", "drying_temp", "mixing_time"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )

    rows = []
    keys = list(SEARCH_SPACE)
    for values in product(*(SEARCH_SPACE[key] for key in keys)):
        candidate = dict(zip(keys, values))
        recipe_key = tuple(candidate[key] for key in ["si_content", "mxene_content", "alginate_content", "drying_temp", "mixing_time"])
        if recipe_key in seen:
            continue

        carbon = 100 - candidate["si_content"] - candidate["mxene_content"] - candidate["alginate_content"]
        candidate["carbon_content"] = carbon

        # ── Chemistry rules filter ──────────────────────────────
        result = validate_candidate(candidate)
        if not result.valid:
            continue
        # ────────────────────────────────────────────────────────

        row = {feature: fill_values[feature] for feature in features}
        row.update(candidate)
        row["pressing_pressure"] = float(master["pressing_pressure"].median())
        row["si_mxene_ratio"] = row["si_content"] / row["mxene_content"]
        row["si_ti_ratio"] = row["si_percent"] / row["ti_percent"]
        row["c_o_ratio"] = row["c_percent"] / row["o_percent"]

        # Attach chemistry metadata
        row["chemistry_score"] = result.chemistry_score
        row["volume_expansion_risk"] = result.volume_expansion_risk
        row["violations"] = ""  # Empty because valid candidates have no violations
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No valid candidates found. All recipes were filtered by chemistry rules. "
            "Consider widening the search space or relaxing constraints."
        )

    candidates = pd.DataFrame(rows)
    mean, std = gp_model.predict(scaler.transform(candidates[features]), return_std=True)
    candidates["predicted_retention_mean"] = mean
    candidates["predicted_retention_std"] = std
    candidates["acquisition_score"] = (
        candidates["predicted_retention_mean"]
        + UCB_BETA * candidates["predicted_retention_std"]
    )

    # ── Final ranking: GP acquisition + chemistry quality boost ──
    candidates["final_score"] = (
        candidates["acquisition_score"]
        + CHEMISTRY_ALPHA * candidates["chemistry_score"] * candidates["predicted_retention_mean"].std()
    )

    candidates["predicted_retention"] = candidates["predicted_retention_mean"]
    candidates["nearest_distance"] = candidates.apply(lambda row: _nearest_distance(row, master), axis=1)
    candidates["confidence"] = candidates.apply(
        lambda row: _confidence(row["nearest_distance"], row["predicted_retention_std"]),
        axis=1,
    )
    candidates["reason"] = candidates.apply(_reason, axis=1)

    output_cols = [
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
    top = candidates.sort_values("final_score", ascending=False).head(n).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    for col in ["predicted_retention", "predicted_retention_mean", "predicted_retention_std",
                 "acquisition_score", "chemistry_score", "volume_expansion_risk", "final_score"]:
        top[col] = top[col].round(2)
    RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    top[["rank"] + output_cols].to_csv(RECOMMENDATIONS_FILE, index=False)
    return top[["rank"] + output_cols]


def main():
    top = recommend_top()
    print(f"Saved: {RECOMMENDATIONS_FILE}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
