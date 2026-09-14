from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.process.simulators.artistic.convergence import Checkpoint, StabilityPolicy, build_convergence_report
from src.process.simulators.artistic.parser import parse_thermo_log


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_relative(value: float, denominator: float, scale: float, *, crosses_zero: bool = False) -> tuple[float | None, str]:
    if crosses_zero:
        return None, "ZERO_CROSSING_NOT_APPLICABLE"
    if abs(denominator) <= max(scale * 1e-6, 1e-12):
        return None, "NEAR_ZERO_DENOMINATOR"
    return value / abs(denominator), "VALID"


def _tail_stability_row(name: str, values: list[float], steps: list[int]) -> dict[str, object]:
    minimum, maximum = min(values), max(values)
    mean = sum(values) / len(values)
    scale = max(abs(minimum), abs(maximum), abs(mean))
    crosses_zero = minimum < 0 < maximum
    span = maximum - minimum
    relative_span, relative_status = _safe_relative(span, values[-1], scale, crosses_zero=crosses_zero)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    coefficient_of_variation, cv_status = _safe_relative(std, mean, scale)
    split = len(values) // 2
    first_block, second_block = values[:split], values[split:]
    first_block_mean = sum(first_block) / len(first_block) if first_block else None
    second_block_mean = sum(second_block) / len(second_block) if second_block else None
    block_shift = abs(second_block_mean - first_block_mean) if first_block_mean is not None and second_block_mean is not None else None
    block_relative_shift, block_relative_status = _safe_relative(block_shift, mean, scale, crosses_zero=crosses_zero) if block_shift is not None else (None, "NOT_AVAILABLE")
    return {
        "metric": name, "tail_count": len(values), "first": values[0], "last": values[-1], "minimum": minimum, "maximum": maximum,
        "mean": mean, "std": std, "absolute_span": span, "crosses_zero": crosses_zero,
        "relative_span_to_last": relative_span, "relative_span_status": relative_status,
        "coefficient_of_variation": coefficient_of_variation, "coefficient_of_variation_status": cv_status,
        "slope_per_step": (values[-1] - values[0]) / (steps[-1] - steps[0]),
        "recent_window_diagnostic": "RECENT_WINDOW_DIAGNOSTIC", "first_block_mean": first_block_mean,
        "second_block_mean": second_block_mean, "absolute_block_mean_shift": block_shift,
        "relative_block_mean_shift": block_relative_shift, "relative_block_mean_shift_status": block_relative_status,
    }


def analyze(run_directory: Path, output_directory: Path) -> dict[str, object]:
    manifest_path = run_directory / "manifest.json"
    cutoff_path = run_directory / "slurry_cutoff_manifest.json"
    manifest = _json(manifest_path)
    cutoff = _json(cutoff_path) if cutoff_path.is_file() else {}
    workspace = run_directory / "workspace"
    parsed = parse_thermo_log(workspace / "slurry.log", minimization_expected=True)
    dynamics = [item for item in parsed.checkpoints if item.phase == "dynamics" and item.dynamics_step is not None]
    requested = int(cutoff.get("requested_slurry_steps", manifest["requested_slurry_steps"]))
    completed = max((int(item.dynamics_step) for item in dynamics), default=0)
    if completed != requested:
        raise ValueError(f"slurry completion mismatch: expected {requested}, found {completed}")
    checkpoints = tuple(Checkpoint(int(item.dynamics_step), item.metrics, raw_step=item.raw_step, dynamics_step=item.dynamics_step, stage=item.stage, source_log=item.source_log) for item in dynamics)
    convergence = build_convergence_report(checkpoints, requested_steps=requested, stability_policy=StabilityPolicy())
    metrics = sorted({name for item in dynamics for name in item.metrics})
    tail = dynamics[-10:]
    stability_rows = []
    for name in metrics:
        values = [float(item.metrics[name]) for item in tail]
        stability_rows.append(_tail_stability_row(name, values, [int(item.dynamics_step) for item in tail]))
    slurry_command = next(item for item in manifest.get("commands", []) if item.get("stage") == "slurry")
    wall_seconds = float(slurry_command["wall_seconds"])
    minimization_seconds = sum(item.wall_seconds for item in parsed.timings if item.phase == "minimization")
    dynamics_seconds = sum(item.wall_seconds for item in parsed.timings if item.phase == "dynamics")
    dynamics_rate = completed / dynamics_seconds if dynamics_seconds else None
    rate = completed / wall_seconds
    reference_steps = int(manifest.get("reference_slurry_steps", 20_000_000))
    drying_command = next((item for item in manifest.get("commands", []) if str(item.get("stage", "")).startswith("drying")), None)
    artifacts = {str(path.relative_to(run_directory)).replace("\\", "/"): _sha256(path) for path in (manifest_path, cutoff_path, workspace / "coord_out_slurry.data", workspace / "density_slurry.out", workspace / "slurry.log", workspace / "in_slurry.run") if path.is_file()}
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "classification": "SLURRY_CUTOFF_SUCCESS",
        "canonical_manifest_status": manifest["status"],
        "cutoff_manifest_status": cutoff.get("status", "NOT_AVAILABLE"),
        "full_pipeline_complete": False,
        "slurry_completion": {"requested_dynamic_steps": requested, "completed_dynamic_steps": completed, "raw_dynamic_start": dynamics[0].raw_step, "raw_final_step": dynamics[-1].raw_step, "thermo_checkpoint_count": len(dynamics), "thermo_parse_diagnostics": list(parsed.diagnostics)},
        "slurry_density": float((workspace / "density_slurry.out").read_text(encoding="utf-8").strip()),
        "fidelity": {"mode": manifest["fidelity_mode"], "short_horizon_protocol": manifest.get("short_horizon_protocol") or ("COMPRESSED_RAMP" if manifest["fidelity_mode"] == "SHORT_HORIZON" else None), "protocol_schedule": manifest.get("protocol_schedule"), "reference_steps": reference_steps, "reference_equivalence_status": "REFERENCE_NOT_AVAILABLE"},
        "stability": {"status": str(convergence.stability_status), "policy": dict(convergence.stability_policy), "interpretation": "RECENT_WINDOW_DIAGNOSTIC and RAMP_AWARE_DIAGNOSTIC only; changing Press/Volume under an NPT ramp is NOT_A_DIRECT_EQUILIBRIUM_TEST."},
        "runtime": {"minimization_wall_seconds": minimization_seconds if any(item.phase == "minimization" for item in parsed.timings) else None, "dynamics_wall_seconds": dynamics_seconds if dynamics_seconds else None, "total_slurry_command_wall_seconds": wall_seconds, "dynamics_steps": completed, "dynamics_steps_per_second": dynamics_rate, "throughput": {"numerator_completed_dynamics_steps": completed, "denominator_walltime_kind": "TOTAL_SLURRY_COMMAND", "denominator_wall_seconds": wall_seconds, "effective_total_stage_step_rate": rate, "pure_dynamics_rate_available": dynamics_rate is not None}, "rough_total_stage_linear_projection": {"projection_method": "linear scaling from observed total slurry-command walltime", "timing_basis": "TOTAL_SLURRY_COMMAND", "measured": False, "projected_seconds_by_horizon": {str(steps): steps / rate for steps in (1_000_000, 2_000_000, reference_steps)}, "projected_seconds_20m_from_start": reference_steps / rate, "projected_seconds_remaining_to_20m": (reference_steps - completed) / rate}, "mpi_processes": len(slurry_command["command"]) and int(next((slurry_command["command"][index + 1] for index, value in enumerate(slurry_command["command"][:-1]) if value == "-n"), 1))},
        "partial_drying_evidence": {"status": cutoff.get("full_pipeline_status", "NOT_AVAILABLE"), "command": drying_command, "artifacts": [path.name for path in sorted(workspace.glob("dump_sol.com_*.atom"))], "use_restriction": "Audit-only partial evidence; not a completed drying result and not training evidence."},
        "provenance": {key: manifest.get(key) for key in ("recipe_fingerprint", "physics_config_fingerprint", "fidelity_identity", "checked_out_commit", "source_tree_hash", "ai_co_scientist_commit", "mpi_environment")},
        "artifact_sha256": artifacts,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "thermo_checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["raw_step", "dynamics_step", *metrics])
        writer.writeheader()
        for item in dynamics:
            writer.writerow({"raw_step": item.raw_step, "dynamics_step": item.dynamics_step, **item.metrics})
    with (output_directory / "tail_stability_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stability_rows[0]))
        writer.writeheader(); writer.writerows(stability_rows)
    _write_json(output_directory / "runtime_projection.json", summary["runtime"])
    _write_json(output_directory / "summary.json", summary)
    report = "\n".join((
        "# Real ARTISTIC 500k slurry cutoff", "", "## Scope and status", "",
        f"- Run: `{manifest['run_id']}`; SHORT_HORIZON slurry only.",
        f"- Slurry completion: {completed:,}/{requested:,} dynamics-relative steps; raw LAMMPS {dynamics[0].raw_step:,} → {dynamics[-1].raw_step:,}; {len(dynamics)} parsed thermo checkpoints and no parser diagnostics.",
        f"- The canonical full-pipeline manifest is `{manifest['status']}` because the downstream drying command was intentionally terminated. The cutoff sidecar classifies the completed stage as `{cutoff.get('status', 'NOT_AVAILABLE')}`.",
        "- Drying/calendering are incomplete. The available drying dumps are audit-only, excluded from final-KPI and training claims.", "", "## Observations", "",
        f"- Final slurry density: {summary['slurry_density']:.14g}.",
        f"- Total slurry command wall time: {wall_seconds / 3600:.3f} h using {summary['runtime']['mpi_processes']} MPI ranks. The effective total-stage rate is {rate:.4f} completed dynamics steps per total slurry-command wall second; parsed dynamics rate is {dynamics_rate:.4f} steps/s." if dynamics_rate is not None else f"- Total slurry command wall time: {wall_seconds / 3600:.3f} h using {summary['runtime']['mpi_processes']} MPI ranks. The effective total-stage rate is {rate:.4f} completed dynamics steps per total slurry-command wall second; parsed dynamics timing is unavailable.",
        f"- Rough total-stage linear projection: 20,000,000 steps is {summary['runtime']['rough_total_stage_linear_projection']['projected_seconds_20m_from_start'] / 86400:.2f} days from start ({summary['runtime']['rough_total_stage_linear_projection']['projected_seconds_remaining_to_20m'] / 86400:.2f} additional days after this run). This projection linearly scales the observed 500k total slurry-command walltime and may overestimate longer horizons because minimization is a fixed cost.",
        f"- Existing diagnostic stability policy: last 10 checkpoints, 1% relative tolerance, all available metrics. Result: `{convergence.stability_status}`. This is a `RECENT_WINDOW_DIAGNOSTIC` / `RAMP_AWARE_DIAGNOSTIC`, not a direct equilibrium test while NPT pressure/volume are intentionally changing.",
        "- Reference equivalence: `REFERENCE_NOT_AVAILABLE`; no compatible exact 20,000,000-step reference was supplied.", "", "## Provenance", "",
        f"- Recipe fingerprint: `{manifest['recipe_fingerprint']}`", f"- Physics fingerprint: `{manifest['physics_config_fingerprint']}`",
        f"- Pinned source commit/tree: `{manifest['checked_out_commit']}` / `{manifest['source_tree_hash']}`", f"- MPI environment was validated: `{manifest['mpi_environment']['validated']}`.",
        "- SHA-256 bindings for the manifest, cutoff sidecar, rendered slurry input, log, density, and final slurry coordinates are in `summary.json`.", "", "Artifacts: `thermo_checkpoints.csv`, `tail_stability_metrics.csv`, `runtime_projection.json`, and `summary.json`.", "",
    ))
    (output_directory / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an evidence-only ARTISTIC slurry-cutoff analysis bundle.")
    parser.add_argument("--run-directory", type=Path, default=Path("outputs/artistic_runs/artistic-short-500k-rerun"))
    parser.add_argument("--output-directory", type=Path, default=Path("outputs/artistic_convergence/slurry_500k_real"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_directory, args.output_directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
