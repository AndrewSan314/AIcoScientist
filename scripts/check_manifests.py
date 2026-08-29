from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def generate_manifest(
    dataset_name: str,
    dataset_dir: Path,
    source: str,
    source_version: str,
    output_path: Path,
    include_subdirs: list[str] | None = None,
) -> dict:
    files = []
    dataset_dir = dataset_dir.resolve()
    search_dirs = [dataset_dir / sub for sub in include_subdirs] if include_subdirs else [dataset_dir]

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for root, _, filenames in os.walk(sdir):
            for fname in sorted(filenames):
                if fname.startswith(".") or fname == ".DS_Store":
                    continue
                if fname.endswith(".json") and "manifest" in fname:
                    continue
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(dataset_dir).as_posix()
                size = fpath.stat().st_size
                sha = hashlib.sha256()
                with open(fpath, "rb") as f:
                    while chunk := f.read(1024 * 1024 * 8):
                        sha.update(chunk)
                files.append(
                    {
                        "path": rel_path,
                        "size_bytes": size,
                        "sha256": sha.hexdigest(),
                    }
                )

    files.sort(key=lambda x: x["path"])
    manifest = {
        "dataset": dataset_name,
        "source": source,
        "source_version": source_version,
        "files": files,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_manifest(manifest_path: Path) -> tuple[bool, list[str]]:
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_dir = manifest_path.parent
    errors = []

    for file_info in manifest.get("files", []):
        rel_path = file_info["path"]
        expected_size = file_info["size_bytes"]
        expected_sha = file_info["sha256"]

        fpath = base_dir / rel_path
        if not fpath.exists():
            errors.append(f"Missing file: {rel_path}")
            continue

        actual_size = fpath.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}"
            )
            continue

        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(1024 * 1024 * 8):
                sha.update(chunk)
        actual_sha = sha.hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"SHA256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or generate dataset manifests")
    parser.add_argument("--mode", choices=["generate", "verify"], default="verify")
    parser.add_argument("--dataset", choices=["severson", "dynamic_cycling", "all"], default="all")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    datasets = {
        "severson": {
            "name": "severson_2019",
            "dir": project_root / "data" / "external" / "severson_2019",
            "source": "https://data.matr.io/1/projects/5c488730b7d30d0001550c60",
            "version": "nature_energy_2019",
            "manifest": project_root / "data" / "external" / "severson_2019" / "manifest.json",
            "subdirs": ["raw"],
        },
        "dynamic_cycling": {
            "name": "dynamic_cycling_2024",
            "dir": project_root / "data" / "external" / "dynamic_cycling_2024",
            "source": "https://purl.stanford.edu/td676xr4322",
            "code_repository": "https://github.com/chueh-ermon-group/dynamic-cycling",
            "version": "nat_energy_2024",
            "manifest": project_root / "data" / "external" / "dynamic_cycling_2024" / "manifest.json",
            "subdirs": ["paper_code/data"],
        },
    }

    target_keys = [args.dataset] if args.dataset != "all" else ["severson", "dynamic_cycling"]

    for key in target_keys:
        cfg = datasets[key]
        if args.mode == "generate":
            print(f"Generating manifest for {cfg['name']}...")
            generate_manifest(
                dataset_name=cfg["name"],
                dataset_dir=cfg["dir"],
                source=cfg["source"],
                source_version=cfg["version"],
                output_path=cfg["manifest"],
                include_subdirs=cfg.get("subdirs"),
            )
            print(f"Saved: {cfg['manifest']}")
        else:
            print(f"Verifying manifest for {cfg['name']}...")
            ok, errors = verify_manifest(cfg["manifest"])
            if ok:
                print(f"[OK] {cfg['name']} manifest matches files on disk.")
            else:
                print(f"[FAIL] {cfg['name']} manifest errors:")
                for err in errors:
                    print(f"  - {err}")



if __name__ == "__main__":
    main()
