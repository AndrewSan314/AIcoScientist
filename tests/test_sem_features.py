import numpy as np
from matplotlib import pyplot as plt

from src.sem_features import FEATURE_COLUMNS, extract_sem_features


def test_sem_features_include_bad_image_without_crashing(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    plt.imsave(image_dir / "S001.png", np.eye(16), cmap="gray")
    plt.imsave(image_dir / "S002.png", np.indices((16, 16)).sum(axis=0) % 2, cmap="gray")
    (image_dir / "broken.png").write_text("not an image", encoding="utf-8")
    output = tmp_path / "features.csv"

    df = extract_sem_features(image_dir, output)

    assert output.exists()
    assert set(df["sample_id"]) == {"S001", "S002", "broken"}
    assert all(np.issubdtype(df[column].dtype, np.number) for column in FEATURE_COLUMNS)
    broken = df.loc[df["sample_id"] == "broken"].iloc[0]
    assert df.loc[df["sample_id"] == "S001", "status"].iloc[0] == "ok"
    assert df.loc[df["sample_id"] == "S002", "status"].iloc[0] == "ok"
    assert broken["status"].startswith("error:")
    assert broken[FEATURE_COLUMNS].isna().all()
