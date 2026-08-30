from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.science.coordinator import ScientificClosedLoopCoordinator
from src.science.evaluation import evaluate_two_stage_model
from src.science.provenance import get_environment_provenance, get_git_provenance
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter


def run_synthetic_demo(
    seed: int = 42,
    steps: int = 5,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Runs a complete 5-step synthetic closed-loop experimentation demonstration."""
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "outputs" / "scientific_demo"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "experiment_ledger.db"
    # Remove existing db if fresh run
    if db_path.exists():
        db_path.unlink()

    oracle = SyntheticExperimentOracle()
    adapter = SyntheticScienceAdapter(oracle=oracle)

    print("\n========================================================")
    print(f"  AIcoScientist Scientific Closed-Loop Demo (Seed={seed}, Steps={steps})")
    print("========================================================")

    # 1. Generate initial historical seed data
    print("[1/5] Generating initial historical seed dataset...")
    init_df = adapter.load_initial_dataset(n_samples=10, seed=seed)
    print(f"      Loaded {len(init_df)} initial completed experiments.")

    # 2. Candidate pool
    candidate_pool = adapter.candidate_space(observed=init_df, n_candidates=100, seed=seed)
    print(f"      Prepared candidate space of {len(candidate_pool)} viable process candidates.")

    # 3. Initialize coordinator with ledger & models
    print("[2/5] Initializing ScientificClosedLoopCoordinator & SQLite Ledger...")
    coordinator = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=candidate_pool,
        db_path=db_path,
        strategy="expected_improvement",
        random_state=seed,
    )

    proposal_history: list[dict[str, Any]] = []

    print("\n[3/5] Executing Autonomous Closed-Loop Cycle (Process -> Structure -> Property -> Feedback)...")
    for step in range(1, steps + 1):
        print(f"\n--- Closed-Loop Step {step} / {steps} ---")

        # A. Propose next candidate
        rec, rationale = coordinator.propose_next(n_mc_samples=64)
        print(f"-> Proposed: {rec.experiment_id} ({rec.candidate_id})")
        print(f"   Process: x1={rec.pre_experiment_features['x1']:.2f}, x2={rec.pre_experiment_features['x2']:.2f}, x3={rec.pre_experiment_features['x3']:.2f}")
        print(f"   Expected Target: {rationale.predicted_performance_mean:.2f} ± {rationale.predicted_performance_latent_std:.2f}")
        print(f"   Expected Structure: z1={rationale.predicted_characterization['z1']['mean']:.3f}, z2={rationale.predicted_characterization['z2']['mean']:.3f}")
        print(f"   Reason Code: {rationale.reason_code} | Learning Value: {rationale.expected_learning_value:.4f}")

        proposal_history.append({
            "step": step,
            "experiment_id": rec.experiment_id,
            "candidate_id": rec.candidate_id,
            "process": rec.pre_experiment_features,
            "rationale": rationale.to_dict(),
            "rendered_rationale": rationale.render_text(),
        })

        # B. Physical execution & Characterization feedback
        coordinator.record_executed(rec.experiment_id)
        chars = oracle.evaluate_characterization(rec.pre_experiment_features, seed=seed * 1000 + step)
        coordinator.record_characterization(rec.experiment_id, chars)
        print(f"   Measured Structure (Stage A): z1={chars['z1']:.4f}, z2={chars['z2']:.4f}")

        # C. Performance measurement feedback
        perf = oracle.evaluate_performance(rec.pre_experiment_features, chars, seed=seed * 1000 + step + 50)
        coordinator.record_performance(rec.experiment_id, perf)
        print(f"   Measured Performance (Stage B): y={perf['y']:.3f}")

    # 4. Verify Ledger Integrity
    print("\n[4/5] Verifying Ledger SHA-256 Hash Chain Integrity...")
    valid, errors = coordinator.ledger.verify_integrity()
    print(f"      Ledger Valid: {valid} (Errors: {len(errors)})")
    if not valid:
        raise RuntimeError(f"Ledger integrity verification failed: {errors}")

    # 5. Evaluate models on a held-out test set
    print("\n[5/5] Generating Honest Model Evaluation Report...")
    test_df = adapter.load_initial_dataset(n_samples=25, seed=seed + 999)
    eval_report = evaluate_two_stage_model(
        two_stage_model=coordinator.model_bundle.two_stage_model,
        direct_model=coordinator.model_bundle.direct_model,
        test_df=test_df,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        n_mc_samples=64,
        seed=seed,
    )

    # Save artifacts
    proposal_jsonl = output_dir / "proposal_history.jsonl"
    with open(proposal_jsonl, "w", encoding="utf-8") as f:
        for p in proposal_history:
            f.write(json.dumps(p) + "\n")

    report_json = output_dir / "model_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    prov_json = output_dir / "run_provenance.json"
    run_prov = {
        "dataset": "synthetic_science",
        "seed": seed,
        "steps": steps,
        "completed_experiments": len(coordinator.ledger.list_completed_records()),
        "ledger_verified": valid,
        "git": get_git_provenance(),
        "environment": get_environment_provenance(),
    }
    with open(prov_json, "w", encoding="utf-8") as f:
        json.dump(run_prov, f, indent=2)

    print(f"\nDemo successfully completed. Artifacts saved to: {output_dir}/")
    print(f"  - {proposal_jsonl.name}")
    print(f"  - {report_json.name}")
    print(f"  - {prov_json.name}")
    print("========================================================\n")

    return {
        "status": "SUCCESS",
        "steps": steps,
        "ledger_valid": valid,
        "evaluation": eval_report,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AIcoScientist Generic Scientific Closed-Loop CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run synthetic domain-generic closed-loop demonstration")
    demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    demo_parser.add_argument("--steps", type=int, default=5, help="Number of closed-loop experimentation steps")
    demo_parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    if args.command == "demo":
        run_synthetic_demo(seed=args.seed, steps=args.steps, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
