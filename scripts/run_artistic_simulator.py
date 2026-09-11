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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--normalize-root", type=Path, default=Path("data/external/artistic") / PINNED_COMMIT)
    parser.add_argument("--no-normalize", action="store_true", help="Do not write a successful, validated run to the ARTISTIC adapter cache.")
    args = parser.parse_args()
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=args.source_root, output_root=args.output_root, lammps_command=args.lammps_command, execution_mode=ExecutionMode(args.mode)))
    recipe = _recipe(args.recipe)
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
