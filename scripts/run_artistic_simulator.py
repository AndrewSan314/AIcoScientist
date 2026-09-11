from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.battery_process.artistic import ArtisticSimulationAdapter
from src.process.simulators.artistic import ArtisticRecipe, ArtisticRunConfig, ArtisticSimulator, CalenderingRecipe, DryingMode, ExecutionMode, HeterogeneousDryingRecipe, SlurryRecipe
from src.process.simulators.artistic.config import PINNED_COMMIT


def _recipe(path: Path) -> ArtisticRecipe:
    raw = json.loads(path.read_text(encoding="utf-8"))
    slurry = dict(raw["slurry"])
    if "dry_mass_mg" in slurry:
        raise ValueError("legacy dry_mass_mg is unsupported; migrate the recipe to electrode_mass_ug (the upstream source divides it by 1E6)")
    slurry["diameter_am_um"] = tuple(slurry["diameter_am_um"])
    slurry["percent_am"] = tuple(slurry["percent_am"])
    return ArtisticRecipe(
        slurry=SlurryRecipe(**slurry), drying_mode=DryingMode(raw["drying_mode"]) if raw.get("drying_mode") else None,
        heterogeneous_drying=HeterogeneousDryingRecipe(**raw["heterogeneous_drying"]) if raw.get("heterogeneous_drying") else None,
        calendering=CalenderingRecipe(**raw["calendering"]) if raw.get("calendering") else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pinned public ARTISTIC source in an isolated workspace.")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--source-root", type=Path, default=ArtisticRunConfig().source_root)
    parser.add_argument("--output-root", type=Path, default=ArtisticRunConfig().output_root)
    parser.add_argument("--lammps-command", default="lmp")
    parser.add_argument("--mode", choices=[mode.value for mode in ExecutionMode], default=ExecutionMode.LOCAL.value)
    parser.add_argument("--mpi-processes", type=int, default=ArtisticRunConfig().mpi_processes)
    parser.add_argument("--mpi-launcher", default=ArtisticRunConfig().mpi_launcher)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Report predicted ARTISTIC particle and memory requirements without rendering or running.")
    parser.add_argument("--max-particle-count", type=int, default=ArtisticRunConfig().max_particle_count)
    parser.add_argument("--allow-unsafe-particle-count", action="store_true", help="Explicitly override the particle-count safety threshold.")
    parser.add_argument("--normalize-root", type=Path, default=Path("data/external/artistic") / PINNED_COMMIT)
    parser.add_argument("--no-normalize", action="store_true", help="Do not write a successful, validated run to the ARTISTIC adapter cache.")
    args = parser.parse_args()
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=args.source_root, output_root=args.output_root, lammps_command=args.lammps_command, execution_mode=ExecutionMode(args.mode), mpi_processes=args.mpi_processes, mpi_launcher=args.mpi_launcher, max_particle_count=args.max_particle_count, allow_unsafe_particle_count=args.allow_unsafe_particle_count))
    recipe = _recipe(args.recipe)
    if args.preflight:
        print(json.dumps(simulator.preflight(recipe).as_dict(), indent=2))
        return 0
    if args.prepare_only:
        print(simulator.prepare(recipe, run_id=args.run_id))
        return 0
    result = simulator.execute(recipe, run_id=args.run_id)
    cache = None
    if result.status.value == "Success" and not args.no_normalize:
        cache = ArtisticSimulationAdapter.normalize_successful(result, recipe, root=args.normalize_root)
    print(json.dumps({"run_id": result.run_id, "status": result.status, "directory": str(result.run_directory), "normalized_cache": str(cache) if cache else None, "diagnostics": result.diagnostics}, indent=2))
    return 0 if result.status.value == "Success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
