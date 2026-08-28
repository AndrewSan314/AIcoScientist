"""Compare Otsu vs SAM segmentation on demo SEM images."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg

from src.sem_analysis import analyze_sem_image, overlay_masks

ROOT = Path(__file__).resolve().parents[1]
SEM_DIR = ROOT / "data" / "raw" / "sem_images"
SAM_CKPT = ROOT / "models" / "sam_vit_b_01ec64.pth"
OUT_DIR = ROOT / "outputs" / "sem_segmentation_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(image_path: Path, enhance: bool = True):
    image = mpimg.imread(image_path)
    otsu = analyze_sem_image(image, enhance=enhance, sam_checkpoint=None)
    sam = analyze_sem_image(image, enhance=enhance, sam_checkpoint=SAM_CKPT)
    return image, otsu, sam


def save_panel(image_path: Path, enhance: bool = True):
    image, otsu, sam = run(image_path, enhance=enhance)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    title = image_path.name
    axes[0, 0].imshow(otsu["working_image"])
    axes[0, 0].set_title("CLAHE input" if enhance else "Input")
    axes[0, 1].imshow(otsu["overlay"])
    axes[0, 1].set_title(f"Otsu | area={otsu['metrics']['particle_area_fraction']:.3f}")
    axes[0, 2].imshow(otsu["crack_mask"], cmap="gray")
    axes[0, 2].set_title(f"Otsu cracks | n={otsu['metrics']['crack_count']}")

    axes[1, 0].imshow(sam["working_image"])
    axes[1, 0].set_title(sam["metrics"]["segmentation_method"])
    axes[1, 1].imshow(sam["overlay"])
    axes[1, 1].set_title(f"SAM | area={sam['metrics']['particle_area_fraction']:.3f}")
    axes[1, 2].imshow(sam["crack_mask"], cmap="gray")
    axes[1, 2].set_title(f"SAM cracks | n={sam['metrics']['crack_count']}")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / f"{image_path.stem}_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return otsu["metrics"], sam["metrics"], out


def main():
    print(f"SAM checkpoint: {SAM_CKPT.exists()} ({SAM_CKPT.stat().st_size / 1e6:.1f} MB)")
    rows = []
    for path in sorted(SEM_DIR.glob("*.jpg")):
        print(f"\n=== {path.name} ===")
        otsu_m, sam_m, out = save_panel(path, enhance=True)
        print(f"Otsu: method={otsu_m['segmentation_method']} particle={otsu_m['particle_area_fraction']:.4f} cracks={otsu_m['crack_count']}")
        print(f"SAM:  method={sam_m['segmentation_method']} particle={sam_m['particle_area_fraction']:.4f} cracks={sam_m['crack_count']}")
        rows.append((path.name, otsu_m, sam_m, out))
        print(f"Saved: {out}")

    # Summary deltas
    print("\n=== SUMMARY ===")
    for name, otsu_m, sam_m, _ in rows:
        delta_area = sam_m["particle_area_fraction"] - otsu_m["particle_area_fraction"]
        delta_cracks = sam_m["crack_count"] - otsu_m["crack_count"]
        print(
            f"{name}: particle_area {otsu_m['particle_area_fraction']:.3f} -> {sam_m['particle_area_fraction']:.3f} "
            f"(delta {delta_area:+.3f}), cracks {otsu_m['crack_count']} -> {sam_m['crack_count']} (delta {delta_cracks:+d})"
        )


if __name__ == "__main__":
    main()
