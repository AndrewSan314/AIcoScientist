import json

import joblib
import pandas as pd

from run_pipeline import main
from src.utils import MASTER_FILE, METRICS_FILE, MODEL_FILE


def test_model_outputs_and_predicts():
    main()
    bundle = joblib.load(MODEL_FILE)
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    df = pd.read_csv(MASTER_FILE)

    assert {"mae", "rmse", "r2"} <= set(metrics)
    assert {"gp_model", "scaler", "rf_metrics", "gp_metrics", "model_versions"} <= set(bundle)
    pred = bundle["model"].predict(df[bundle["features"]].head(1))
    gp_mean, gp_std = bundle["gp_model"].predict(
        bundle["scaler"].transform(df[bundle["features"]].head(1)),
        return_std=True,
    )
    assert len(pred) == 1
    assert len(gp_mean) == 1
    assert len(gp_std) == 1
    assert gp_std[0] >= 0
