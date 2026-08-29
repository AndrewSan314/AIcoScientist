import json

import joblib
import pandas as pd

from run_pipeline import main
from src.chemistry_rules import validate_candidate
from src.utils import MASTER_FILE, METRICS_FILE, MODEL_FILE, RECOMMENDATIONS_FILE


def test_legacy_pipeline_contract():
    main()
    assert MASTER_FILE.is_file()
    assert MODEL_FILE.is_file()
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    assert {"rf_metrics", "gp_metrics"} <= set(metrics)

    recommendations = pd.read_csv(RECOMMENDATIONS_FILE)
    assert len(recommendations) == 3
    assert {
        "si_content",
        "mxene_content",
        "alginate_content",
        "carbon_content",
        "drying_temp",
        "mixing_time",
        "predicted_retention",
        "predicted_retention_mean",
        "predicted_retention_std",
        "acquisition_score",
        "chemistry_score",
        "volume_expansion_risk",
        "confidence",
        "reason",
    } <= set(recommendations)
    for _, row in recommendations.iterrows():
        result = validate_candidate(row.to_dict())
        assert result.valid

    bundle = joblib.load(MODEL_FILE)
    assert bundle["dataset"] == "si_mxene"
