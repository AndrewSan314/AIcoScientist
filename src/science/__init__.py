"""AIcoScientist Scientific Closed-Loop Experimentation Framework.

Provides domain-generic abstractions for closed-loop experimentation:
- Process (Controllable variables) -> Characterization (Structure) -> Performance (Properties)
- Append-only experiment ledger with tamper-evident SHA-256 hash chaining
- Multi-channel Stage A (Process -> Characterization) and Stage B (Process + Characterization -> Performance)
- Monte Carlo uncertainty propagation via Law of Total Variance
- Structured deterministic ScientificRationale
- Resumable closed-loop coordinator
"""

from __future__ import annotations

from src.science.provenance import (
    ScientificModelProvenance,
    build_benchmark_run_manifest,
    get_environment_provenance,
    get_git_provenance,
)

__all__ = [
    "ScientificModelProvenance",
    "build_benchmark_run_manifest",
    "get_environment_provenance",
    "get_git_provenance",
]
