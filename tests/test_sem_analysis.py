import numpy as np

from src.sem_analysis import CRACK_METRIC_COLUMNS, _elongated_crack_mask, analyze_sem_image, crack_mask


def test_sem_analysis_detects_crack_metrics():
    image = np.full((96, 96, 3), 210, dtype=np.uint8)
    image[24:72, 18:78] = 235
    image[45:50, 25:70] = 20
    image[84:, :] = 0

    result = analyze_sem_image(image, enhance=True)
    metrics = result["metrics"]

    assert result["overlay"].shape == image.shape
    assert result["crack_mask"].any()
    assert not result["crack_mask"][84:, :].any()
    assert all(metrics[column] >= 0 for column in CRACK_METRIC_COLUMNS)
    assert metrics["crack_count"] >= 1


def test_crack_shape_filter_rejects_wide_blob_and_small_scratch():
    yy, xx = np.ogrid[:128, :128]
    wide_blob = ((xx - 64) / 22) ** 2 + ((yy - 64) / 11) ** 2 <= 1
    small_scratch = np.zeros((128, 128), dtype=bool)
    small_scratch[24:28, 24:44] = True
    borderline_scratch = np.zeros((128, 128), dtype=bool)
    borderline_scratch[88:93, 48:78] = True

    assert not _elongated_crack_mask(wide_blob).any()
    assert not _elongated_crack_mask(small_scratch).any()
    assert not _elongated_crack_mask(borderline_scratch).any()


def test_crack_mask_accepts_tunable_detection_parameters():
    image = np.full((128, 128, 3), 220, dtype=np.uint8)
    image[60:65, 24:104] = 10
    material = np.ones((128, 128), dtype=bool)

    detected = crack_mask(
        image,
        material,
        top_hat_radius=3,
        response_percentile=92,
        min_component_area=160,
        min_elongation=2.2,
    )

    assert detected[60:65, 24:104].any()
