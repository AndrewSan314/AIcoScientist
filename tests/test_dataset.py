import sqlite3

import pandas as pd

from run_pipeline import main
from src.utils import DATABASE_FILE, MASTER_FILE, MODEL_FEATURES, TARGET


def test_master_dataset_created():
    main()
    assert DATABASE_FILE.is_file()
    with sqlite3.connect(DATABASE_FILE) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM ingestion_runs WHERE status = 'succeeded'"
        ).fetchone()[0] >= 1
    df = pd.read_csv(MASTER_FILE)
    assert "sample_id" in df.columns
    assert TARGET in df.columns
    assert len(df) >= 30
    assert not df[[TARGET, *MODEL_FEATURES]].isna().any().any()
    assert {"si_mxene_ratio", "si_ti_ratio", "c_o_ratio", "impurity_score"} <= set(df)
    assert (
        df[["si_content", "mxene_content", "alginate_content", "carbon_content"]]
        .sum(axis=1)
        .eq(100)
        .all()
    )
