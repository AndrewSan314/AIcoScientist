from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.feconi_benchmark import run_feconi_aicoscientist_benchmark
from src.evaluation.feconi_reproduction_benchmark import run_feconi_reproduction_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fe-Co-Ni benchmark suite.")
    parser.add_argument("--mode", choices=["dev", "smoke", "full_reproduction", "aicoscientist", "all"], default="dev")
    parser.add_argument("--reproduction-seeds", type=int, default=100)
    parser.add_argument("--aicoscientist-seeds", type=int, default=30)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    base_out = Path("outputs/feconi")
    base_out.mkdir(parents=True, exist_ok=True)

    if args.mode == "dev":
        seeds = range(5)
        logger.info("=== RUNNING 5-SEED DEV REPRODUCTION (KERR & COERCIVITY) ===")
        t0 = time.time()
        run_feconi_reproduction_benchmark(
            target_name="Kerr",
            seeds=seeds,
            total_budget=50,
            output_dir=base_out / "reproduction" / "kerr_dev",
        )
        run_feconi_reproduction_benchmark(
            target_name="Coer",
            seeds=seeds,
            total_budget=50,
            output_dir=base_out / "reproduction" / "coercivity_dev",
        )
        logger.info("Dev benchmark finished in %.2fs", time.time() - t0)

    elif args.mode in {"full_reproduction", "all"}:
        seeds = range(args.reproduction_seeds)
        logger.info("=== RUNNING %d-SEED PAPER REPRODUCTION (KERR) ===", len(seeds))
        t0 = time.time()
        run_feconi_reproduction_benchmark(
            target_name="Kerr",
            seeds=seeds,
            total_budget=args.budget,
            output_dir=base_out / "reproduction" / "kerr",
        )
        logger.info("Kerr reproduction finished in %.2fs", time.time() - t0)

        logger.info("=== RUNNING %d-SEED PAPER REPRODUCTION (COERCIVITY) ===", len(seeds))
        t0 = time.time()
        run_feconi_reproduction_benchmark(
            target_name="Coer",
            seeds=seeds,
            total_budget=args.budget,
            output_dir=base_out / "reproduction" / "coercivity",
        )
        logger.info("Coercivity reproduction finished in %.2fs", time.time() - t0)

    if args.mode in {"aicoscientist", "all"}:
        seeds = range(args.aicoscientist_seeds)
        logger.info("=== RUNNING %d-SEED AICOSCIENTIST BENCHMARK (KERR) ===", len(seeds))
        t0 = time.time()
        run_feconi_aicoscientist_benchmark(
            target_name="Kerr",
            seeds=seeds,
            total_budget=args.budget,
            output_dir=base_out / "aicoscientist" / "kerr",
        )
        logger.info("Kerr AIcoScientist benchmark finished in %.2fs", time.time() - t0)

        logger.info("=== RUNNING %d-SEED AICOSCIENTIST BENCHMARK (COERCIVITY) ===", len(seeds))
        t0 = time.time()
        run_feconi_aicoscientist_benchmark(
            target_name="Coer",
            seeds=seeds,
            total_budget=args.budget,
            output_dir=base_out / "aicoscientist" / "coercivity",
        )
        logger.info("Coercivity AIcoScientist benchmark finished in %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
