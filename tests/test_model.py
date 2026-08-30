import json

import joblib
import pandas as pd

from run_pipeline import main
from src.utils import MASTER_FILE, METRICS_FILE, MODEL_FILE


def test_rf_model_baseline_hyperparameters():
    from src.train_model import _rf_model
    rf = _rf_model()
    assert rf.n_estimators == 200
    assert rf.min_samples_leaf == 2
    assert rf.random_state == 42

