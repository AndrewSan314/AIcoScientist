import numpy as np
import pandas as pd
from skimage import measure

from src.sem_analysis import CRACK_METRIC_COLUMNS, analyze_sem_image, image_from_bytes
from src.utils import SEM_EXTRACTED_FILE, SEM_IMAGE_DIR


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FEATURE_COLUMNS = [
    "particle_count",
    "mean_particle_area",
    "texture_score",
    "porosity_score",
    "edge_density",
    "particle_area_fraction",
    *CRACK_METRIC_COLUMNS,
]


def _gray_image(img):
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(float)
    img -= img.min()
    return img / img.max() if img.max() else img


def _component_areas(mask):
    labels = measure.label(mask.astype(bool), connectivity=1)
    if labels.max() == 0:
        return np.array([], dtype=float)
    return measure.regionprops_table(labels, properties=("area",))["area"]


def _features(path):
    image = image_from_bytes(path.read_bytes())
    img = _gray_image(image)
    analysis = analyze_sem_image(image, enhance=True)
    mask = analysis["particle_mask"]
    areas = _component_areas(mask)
    dy = np.abs(np.diff(img, axis=0)).mean()
    dx = np.abs(np.diff(img, axis=1)).mean()
    crack = analysis["metrics"]
    return {
        "sample_id": path.stem,
        "particle_count": len(areas),
        "mean_particle_area": float(np.mean(areas)) if areas.size else 0.0,
        "texture_score": float(img.std()),
        "porosity_score": float(1.0 - mask.mean()),
        "edge_density": float(dx + dy),
        "particle_area_fraction": crack["particle_area_fraction"],
        **{col: crack[col] for col in CRACK_METRIC_COLUMNS},
        "status": "ok",
    }


def extract_sem_features(image_dir=SEM_IMAGE_DIR, output_file=SEM_EXTRACTED_FILE):
    rows = []
    for path in sorted(image_dir.iterdir()) if image_dir.exists() else []:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rows.append(_features(path))
        except Exception as exc:
            rows.append({"sample_id": path.stem, **{col: np.nan for col in FEATURE_COLUMNS}, "status": f"error: {exc}"})

    df = pd.DataFrame(rows, columns=["sample_id", *FEATURE_COLUMNS, "status"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


def main():
    df = extract_sem_features()
    print(f"Saved: {SEM_EXTRACTED_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
