from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.battery_process import DrakopoulosGraphiteAdapter, WarwickNMC622Adapter
from src.datasets.battery_process.base import ProcessOptimizationTask
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.evaluation import reveal_one
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.stages import ProcessStage


ADAPTERS = {"drakopoulos_graphite": DrakopoulosGraphiteAdapter, "warwick_nmc622": WarwickNMC622Adapter}


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-backed no-lookahead process recipe replay")
    parser.add_argument("--dataset", choices=ADAPTERS, required=True)
    parser.add_argument("--task", choices=["recipe_optimization"], default="recipe_optimization")
    parser.add_argument("--target", required=True, help="Verified adapter target name; never invent a source column.")
    parser.add_argument("--stage", choices=[stage.value for stage in ProcessStage], default=ProcessStage.COATING.value)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/process_demo"))
    args = parser.parse_args()

    adapter = ADAPTERS[args.dataset]()
    task = ProcessOptimizationTask(args.target, ProcessStage(args.stage))
    replay = adapter.build_replay_frame(task)
    if len(replay) <= args.initial_count:
        raise ValueError("source data needs more rows than initial-count for replay")
    space = adapter.build_optimization_space(task)
    observed, hidden = replay.iloc[:args.initial_count].copy(), replay.iloc[args.initial_count:].copy()
    objective = ProcessOptimizationObjective([ObjectiveSpec(args.target, "maximize")])
    proposal = ProcessOptimizationCoordinator().propose_recipes(observed, space, objective, seed=args.seed)[0]
    observed, hidden, revealed = reveal_one(observed, hidden, recipe_id=proposal.source_recipe_id or "", id_column="recipe_id", target=args.target)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": adapter.metadata().__dict__, "target": args.target, "stage": args.stage, "seed": args.seed,
        "proposal": {"recipe_id": proposal.source_recipe_id, "controls": proposal.controls, "prediction": {key: value.__dict__ for key, value in proposal.predicted_outputs.items()}},
        "revealed": revealed.__dict__, "best_so_far": float(observed[args.target].max()), "remaining_hidden": int(len(hidden)),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
