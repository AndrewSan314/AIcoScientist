from pathlib import Path
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, util


CRACK_METRIC_COLUMNS = [
    "crack_area_fraction",
    "crack_count",
    "crack_length_density",
    "mean_crack_width",
]

# Faster automatic-mask settings for MVP demo (reload model once, balanced grid points).
SAM_FAST_POINTS_PER_SIDE = 16
SAM_INFERENCE_MAX_DIM = 768
_SAM_GENERATOR_CACHE: dict[tuple[str, str, str], object] = {}


def _sam_device():
    requested = os.environ.get("GTIP_SAM_DEVICE", "cuda").lower()
    if requested == "cuda":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _resize_for_sam(rgb, max_dim=SAM_INFERENCE_MAX_DIM):
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return rgb, 1.0
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale


def _upscale_mask(mask, shape):
    h, w = shape[:2]
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _downscale_mask(mask, shape):
    h, w = shape[:2]
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def image_from_bytes(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _uint8_rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = util.img_as_ubyte((image - image.min()) / (image.max() - image.min() + 1e-12))
    return image


def _gray_float(image):
    rgb = _uint8_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def _footer_cutoff(image):
    """Exclude SEM instrument annotation bars (dark or colored footers)."""
    rgb = _uint8_rgb(image)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    candidates = [h]

    bottom_start = int(h * 0.62)
    dark_fraction = (gray < 0.08).mean(axis=1)
    dark_rows = np.where((np.arange(h) >= bottom_start) & (dark_fraction > 0.35))[0]
    if dark_rows.size:
        candidates.append(int(dark_rows[0]))

    for row in range(bottom_start, h):
        red = rgb[row, :, 0].astype(np.int16)
        green = rgb[row, :, 1].astype(np.int16)
        blue = rgb[row, :, 2].astype(np.int16)
        colored = ((red > green + 18) & (red > blue + 12) & (red > 70)).mean()
        if colored > 0.3:
            candidates.append(row)
            break

    row_std = gray.std(axis=1)
    baseline = float(np.median(row_std[: max(1, int(h * 0.72))]))
    for row in range(int(h * 0.68), h):
        if row_std[row] < baseline * 0.3 and dark_fraction[row] > 0.12:
            candidates.append(row)
            break

    cutoff = min(candidates)
    return max(int(h * 0.55), cutoff)


def _content_roi(image):
    h, w = _uint8_rgb(image).shape[:2]
    cutoff = _footer_cutoff(image)
    roi = np.zeros((h, w), dtype=bool)
    roi[:cutoff, :] = True
    return roi


def enhance_sem_image(image):
    rgb = _uint8_rgb(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def _otsu_material_mask(gray, roi):
    roi_values = gray[roi]
    threshold = filters.threshold_otsu(roi_values) if roi_values.size else filters.threshold_otsu(gray)
    mask = gray > threshold
    mask &= roi
    mask = morphology.remove_small_objects(mask, max_size=max(15, mask.size // 2500))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def _filter_sam_masks(masks, gray, roi):
    h, w = gray.shape
    area_total = h * w
    min_area = max(64, int(area_total * 0.00004))
    max_area = int(area_total * 0.12)
    roi_values = gray[roi]
    bright_threshold = float(np.percentile(roi_values, 38)) if roi_values.size else float(np.median(gray))

    filtered = []
    for item in masks:
        area = int(item["area"])
        if area < min_area or area > max_area:
            continue
        seg = item["segmentation"].astype(bool)
        if not seg[roi].any():
            continue
        if float(gray[seg].mean()) < bright_threshold:
            continue
        score = float(item.get("stability_score", 0.0)) * float(item.get("predicted_iou", 0.0))
        filtered.append((score, seg))

    filtered.sort(key=lambda pair: pair[0], reverse=True)
    return [seg for _, seg in filtered]


def _refine_material_mask(seed_mask, gray, roi):
    mask = seed_mask & roi
    if not mask.any():
        return mask

    mask = morphology.closing(mask, morphology.disk(4))
    mask = ndi.binary_fill_holes(mask)

    otsu = _otsu_material_mask(gray, roi)
    grown = mask | (otsu & morphology.dilation(mask, morphology.disk(9)))
    grown = morphology.closing(grown, morphology.disk(5))
    grown = ndi.binary_fill_holes(grown)
    grown &= roi
    grown = morphology.remove_small_objects(grown, max_size=max(40, grown.size // 6000))
    return grown.astype(bool)


def load_sam_generator(checkpoint, model_type="vit_b", device=None):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return None

    device = device or _sam_device()
    cache_key = (str(checkpoint.resolve()), model_type, device)
    if cache_key in _SAM_GENERATOR_CACHE:
        return _SAM_GENERATOR_CACHE[cache_key]

    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=SAM_FAST_POINTS_PER_SIDE,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.82,
        min_mask_region_area=64,
        crop_n_layers=0,
    )
    _SAM_GENERATOR_CACHE[cache_key] = generator
    return generator


def _sam_particle_mask(image, checkpoint, model_type, sam_generator=None):
    checkpoint = Path(checkpoint) if checkpoint else None
    generator = sam_generator or (load_sam_generator(checkpoint, model_type) if checkpoint else None)
    if generator is None:
        return None, "threshold"

    rgb_full = _uint8_rgb(image)
    gray_full = _gray_float(image)
    roi_full = _content_roi(image)
    rgb_small, _ = _resize_for_sam(rgb_full)
    gray_small = cv2.resize(
        (gray_full * 255).astype(np.uint8),
        (rgb_small.shape[1], rgb_small.shape[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    roi_small = _downscale_mask(roi_full, rgb_small.shape)

    masks = generator.generate(rgb_small)
    if not masks:
        return None, "sam-empty"

    segments = _filter_sam_masks(masks, gray_small, roi_small)
    if not segments:
        return None, "sam-empty"

    h, w = gray_full.shape
    seed = np.zeros((h, w), dtype=bool)
    for seg in segments:
        seed |= _upscale_mask(seg, (h, w))

    refined = _refine_material_mask(seed, gray_full, roi_full)
    if refined[roi_full].mean() < 0.22:
        otsu = _otsu_material_mask(gray_full, roi_full)
        refined = refined | (otsu & morphology.dilation(refined, morphology.disk(12)))
        refined = _refine_material_mask(refined, gray_full, roi_full)
        method = f"SAM {model_type} + morphology + Otsu fill"
    else:
        method = f"SAM {model_type} + morphology refine"

    if not refined.any():
        return None, "sam-empty"
    return refined, method


def particle_mask(image, sam_checkpoint=None, sam_model_type="vit_b", sam_source=None, sam_generator=None):
    roi = _content_roi(image)
    segmentation_image = sam_source if sam_source is not None else image
    if sam_checkpoint or sam_generator is not None:
        mask, method = _sam_particle_mask(
            segmentation_image,
            sam_checkpoint or "",
            sam_model_type,
            sam_generator=sam_generator,
        )
        if mask is not None and mask.any():
            mask &= roi
            return mask, method

    gray = _gray_float(image)
    return _otsu_material_mask(gray, roi), "Otsu threshold"


def _elongated_crack_mask(candidate, roi=None, min_area=160, min_elongation=2.2):
    labels = measure.label(candidate)
    if labels.max() == 0:
        return candidate

    h, w = labels.shape
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        ymin, xmin, ymax, xmax = region.bbox
        bbox_width = xmax - xmin
        if ymin >= int(h * 0.72) and bbox_width >= int(w * 0.35):
            continue
        major = float(region.axis_major_length)
        minor = max(float(region.axis_minor_length), 1.0)
        if major / minor >= min_elongation:
            keep[region.label] = True
    filtered = keep[labels]
    if roi is not None:
        filtered &= roi
    return filtered


def crack_mask(
    image,
    material_mask=None,
    *,
    top_hat_radius=3,
    response_percentile=92,
    min_component_area=160,
    min_elongation=2.2,
):
    gray = _gray_float(image)
    roi = _content_roi(image)
    response = morphology.black_tophat(gray, morphology.disk(top_hat_radius))
    positive = response[roi & (response > 0)]
    if positive.size:
        cutoff = float(np.percentile(positive, response_percentile))
        candidate = response >= cutoff
    else:
        candidate = gray < np.quantile(gray[roi], 0.1)
    candidate &= roi
    if material_mask is not None and material_mask.any():
        candidate &= morphology.dilation(material_mask, morphology.disk(3))
    candidate = morphology.remove_small_objects(candidate, max_size=max(0, min_component_area - 1))
    candidate = _elongated_crack_mask(candidate, roi, min_component_area, min_elongation)
    return morphology.closing(candidate, morphology.disk(1)).astype(bool)


def crack_metrics(mask, roi=None):
    mask = mask.astype(bool)
    if roi is not None:
        mask &= roi
    skeleton = morphology.skeletonize(mask)
    distance = ndi.distance_transform_edt(mask)
    widths = distance[skeleton] * 2.0
    labels = measure.label(mask)
    area = mask.size or 1
    return {
        "crack_area_fraction": float(mask.mean()),
        "crack_count": int(labels.max()),
        "crack_length_density": float(skeleton.sum() / area),
        "mean_crack_width": float(widths.mean()) if widths.size else 0.0,
    }


def overlay_masks(image, material_mask, cracks, roi=None):
    rgb = _uint8_rgb(image).astype(np.float32) / 255.0
    if roi is None:
        roi = np.ones(rgb.shape[:2], dtype=bool)
    material_mask = material_mask & roi
    cracks = cracks & roi
    overlay = rgb.copy()
    overlay[material_mask] = overlay[material_mask] * 0.55 + np.array([0.1, 0.65, 0.35]) * 0.45
    overlay[cracks] = np.array([1.0, 0.1, 0.05])
    return np.clip(overlay, 0, 1)


def analyze_sem_image(
    image,
    enhance=False,
    sam_checkpoint=None,
    sam_model_type="vit_b",
    sam_generator=None,
):
    original = _uint8_rgb(image)
    working = enhance_sem_image(image) if enhance else original
    roi = _content_roi(working)
    material_mask, method = particle_mask(
        working,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        sam_source=original,
        sam_generator=sam_generator,
    )
    cracks = crack_mask(working, material_mask)
    metrics = crack_metrics(cracks, roi)
    roi_pixels = int(roi.sum()) or 1
    metrics.update(
        {
            "particle_area_fraction": float(material_mask[roi].sum() / roi_pixels),
            "segmentation_method": method,
            "enhancement": "CLAHE" if enhance else "none",
        }
    )
    return {
        "image": original,
        "working_image": working,
        "particle_mask": material_mask,
        "crack_mask": cracks,
        "overlay": overlay_masks(working, material_mask, cracks, roi),
        "metrics": metrics,
    }
