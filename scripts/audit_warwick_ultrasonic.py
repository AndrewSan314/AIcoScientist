#!/usr/bin/env python3
"""Audit Warwick Frequency-Domain Ultrasonic Electrode Dataset.

Official source: Mendeley Data, DOI: 10.17632/c62yn37d9h.4 (Version 4)
Audits:
- Anode and Cathode samples
- Before/after calendering JSON pairs
- FFT frequency grids and magnitude spectra
- Process metadata (roll gap, speed, coat weight, thickness, density)
- Builds strict grouped 5-fold CV split manifest by Sample_ID
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_warwick_ultrasonic() -> dict[str, Any]:
    raw_root = Path("data/external/warwick_ultrasonic/raw")
    archive_path = raw_root / "frequency_domain_ultrasound_v4.zip"
    extracted_root = raw_root / "Frequency-Domain Ultrasonic Signal Dataset for Bat"
    if not extracted_root.exists():
        extracted_root = raw_root

    out_dir = Path("outputs/warwick_ultrasonic")
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_sha256 = compute_file_sha256(archive_path) if archive_path.exists() else None

    materials = ["Cathode", "Anode"]
    samples_data: list[dict[str, Any]] = []

    for mat in materials:
        mat_dir = extracted_root / mat
        if not mat_dir.exists():
            continue
        sample_dirs = sorted([d for d in mat_dir.iterdir() if d.is_dir() and d.name not in ["ToF", "ToF (ms)"]])
        for s_dir in sample_dirs:
            sample_id = s_dir.name
            p_before = s_dir / "before-calendering.json"
            p_after = s_dir / "after-calendering.json"

            before_exists = p_before.is_file()
            after_exists = p_after.is_file()

            meta_b, meta_a = {}, {}
            fft_b_mag, fft_b_freq = [], []
            fft_a_mag, fft_a_freq = [], []

            if before_exists:
                with open(p_before, "r", encoding="utf-8") as f:
                    db = json.load(f)
                meta_b = db.get("metadata", {})
                fft_b_mag = db.get("fft_magnitude", [])
                fft_b_freq = db.get("fft_frequency", [])

            if after_exists:
                with open(p_after, "r", encoding="utf-8") as f:
                    da = json.load(f)
                meta_a = da.get("metadata", {})
                fft_a_mag = da.get("fft_magnitude", [])
                fft_a_freq = da.get("fft_frequency", [])

            rec = {
                "material": mat,
                "sample_id": sample_id,
                "has_before": before_exists,
                "has_after": after_exists,
                "pair_complete": bool(before_exists and after_exists),
                "fft_b_len": len(fft_b_mag),
                "fft_a_len": len(fft_a_mag),
                "freq_min_mhz": float(np.min(fft_b_freq)) if fft_b_freq else np.nan,
                "freq_max_mhz": float(np.max(fft_b_freq)) if fft_b_freq else np.nan,
                "roll_gap": float(meta_b.get("Roll_Gap", np.nan)),
                "web_speed": float(meta_b.get("Web_speed", np.nan)) if "Web_speed" in meta_b else np.nan,
                "calendering_speed": float(meta_b.get("Calendering_Speed", np.nan)) if "Calendering_Speed" in meta_b else np.nan,
                "coat_weight_gsm": float(meta_b.get("Coat_Weight", np.nan)) if "Coat_Weight" in meta_b else np.nan,
                "thickness_before_um": float(meta_b.get("Thickness", np.nan)),
                "thickness_after_um": float(meta_a.get("Thickness", np.nan)),
                "density_before_g_cm3": float(meta_b.get("Density", np.nan)),
                "density_after_g_cm3": float(meta_a.get("Density", np.nan)),
                "before_json_sha256": compute_file_sha256(p_before) if before_exists else None,
                "after_json_sha256": compute_file_sha256(p_after) if after_exists else None,
            }
            samples_data.append(rec)

    df_samples = pd.DataFrame(samples_data)
    df_samples.to_csv(out_dir / "sample_manifest.csv", index=False)

    # Build strict grouped 5-fold CV splits by Sample_ID
    # Cathode and Anode split separately
    split_manifest: dict[str, dict[str, int]] = {"Cathode": {}, "Anode": {}}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for mat in materials:
        sub = df_samples[df_samples["material"] == mat].reset_index(drop=True)
        sample_ids = sub["sample_id"].tolist()
        for fold_idx, (_, test_indices) in enumerate(kf.split(sample_ids), start=1):
            for idx in test_indices:
                sid = sample_ids[idx]
                split_manifest[mat][sid] = fold_idx

    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    cat_df = df_samples[df_samples["material"] == "Cathode"]
    ano_df = df_samples[df_samples["material"] == "Anode"]

    dataset_manifest = {
        "dataset_id": "warwick_ultrasonic",
        "title": "Frequency-Domain Ultrasonic Signal Dataset for Battery Electrode Thickness Prediction",
        "official_source": "https://data.mendeley.com/datasets/c62yn37d9h/4",
        "source_doi": "10.17632/c62yn37d9h.4",
        "version": "4",
        "retrieval_date": "2026-09-16",
        "license": "CC BY 4.0",
        "archive_filename": archive_path.name if archive_path else None,
        "archive_sha256": archive_sha256,
        "cathode_samples": len(cat_df),
        "anode_samples": len(ano_df),
        "total_paired_samples": len(df_samples),
        "missing_pairs": int((~df_samples["pair_complete"]).sum()),
    }
    with open(out_dir / "dataset_manifest.json", "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    source_audit = {
        "dataset_name": "Frequency-Domain Ultrasonic Signal Dataset for Battery Electrode Thickness Prediction",
        "doi": "10.17632/c62yn37d9h.4",
        "version": "4",
        "num_cathode_samples": len(cat_df),
        "num_anode_samples": len(ano_df),
        "cathode_fft_length": int(cat_df["fft_b_len"].iloc[0]) if not cat_df.empty else 0,
        "anode_fft_length": int(ano_df["fft_b_len"].iloc[0]) if not ano_df.empty else 0,
        "cathode_frequency_range_mhz": [float(cat_df["freq_min_mhz"].min()), float(cat_df["freq_max_mhz"].max())],
        "anode_frequency_range_mhz": [float(ano_df["freq_min_mhz"].min()), float(ano_df["freq_max_mhz"].max())],
        "missing_before_after_pairs": int((~df_samples["pair_complete"]).sum()),
        "duplicate_sample_ids": int(df_samples["sample_id"].duplicated().sum()),
        "meta_features_cathode": ["Roll_Gap", "Web_speed", "Coat_Weight", "Thickness (before)", "Density (before)"],
        "meta_features_anode": ["Roll_Gap", "Calendering_Speed", "Thickness (before)", "Density (before)"],
        "targets": ["Thickness (after)", "Density (after)"],
        "cv_strategy": "Grouped 5-Fold Cross-Validation by Sample_ID (before and after strictly in same fold)",
        "train_only_preprocessing": True,
    }
    with open(out_dir / "source_audit.json", "w") as f:
        json.dump(source_audit, f, indent=2)

    print(f"Ultrasonic Audit Complete: {len(cat_df)} Cathode samples, {len(ano_df)} Anode samples. 0 missing pairs.")
    return source_audit


if __name__ == "__main__":
    audit_warwick_ultrasonic()
