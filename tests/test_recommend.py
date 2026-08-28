import pandas as pd

from run_pipeline import main
from src.chemistry_rules import validate_candidate
from src.utils import MASTER_FILE, RECOMMENDATIONS_FILE


def test_recommendations_are_top_three_new_recipes():
    main()
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    master = pd.read_csv(MASTER_FILE)
    cols = ["si_content", "mxene_content", "alginate_content", "drying_temp", "mixing_time"]
    seen = set(master[cols].astype(int).itertuples(index=False, name=None))
    recommended = set(recs[cols].astype(int).itertuples(index=False, name=None))

    assert len(recs) == 3
    assert pd.api.types.is_numeric_dtype(recs["predicted_retention"])
    assert pd.api.types.is_numeric_dtype(recs["predicted_retention_mean"])
    assert pd.api.types.is_numeric_dtype(recs["predicted_retention_std"])
    assert pd.api.types.is_numeric_dtype(recs["acquisition_score"])
    assert (recs["predicted_retention_std"] >= 0).all()
    assert (recs["carbon_content"] >= 3).all()
    assert (
        recs[["si_content", "mxene_content", "alginate_content", "carbon_content"]]
        .sum(axis=1)
        .eq(100)
        .all()
    )
    assert not recommended & seen
    assert recs["confidence"].isin(["high", "medium", "low"]).all()


def test_recommendations_pass_all_chemistry_rules():
    """Every recommendation must satisfy all hard chemistry rules."""
    main()
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    for _, row in recs.iterrows():
        recipe = {
            "si_content": row["si_content"],
            "mxene_content": row["mxene_content"],
            "alginate_content": row["alginate_content"],
            "carbon_content": row["carbon_content"],
            "drying_temp": row["drying_temp"],
        }
        result = validate_candidate(recipe)
        assert result.valid, (
            f"Recommendation rank={row['rank']} violates chemistry rules: "
            f"{result.violations}"
        )


def test_recommendations_have_percolation_threshold():
    """MXene + Carbon must be >= 15 wt% for conductive percolation."""
    main()
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    conductive = recs["mxene_content"] + recs["carbon_content"]
    assert (conductive >= 15).all(), (
        f"Some recommendations below percolation threshold: "
        f"{conductive[conductive < 15].tolist()}"
    )


def test_recommendations_have_binder_si_ratio():
    """Alginate / Si must be >= 0.08 for mechanical buffering."""
    main()
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    ratio = recs["alginate_content"] / recs["si_content"]
    assert (ratio >= 0.08).all(), (
        f"Some recommendations have insufficient binder/Si ratio: "
        f"{ratio[ratio < 0.08].tolist()}"
    )


def test_recommendations_have_chemistry_score():
    """Output must contain chemistry_score column with positive values."""
    main()
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    assert "chemistry_score" in recs.columns
    assert (recs["chemistry_score"] > 0).all()


def test_recommendations_have_volume_expansion_risk():
    """Output must contain volume_expansion_risk in [0, 1]."""
    main()
    recs = pd.read_csv(RECOMMENDATIONS_FILE)
    assert "volume_expansion_risk" in recs.columns
    assert (recs["volume_expansion_risk"] >= 0).all()
    assert (recs["volume_expansion_risk"] <= 1).all()
