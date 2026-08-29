from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.attia import AttiaAdapter, compute_expected_c4
from src.evaluation.oracle import OracleResponse

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SEEDS: list[int] = list(range(1000, 1050))  # 50 fixed evaluation seeds
SIMULATOR_VERSION: str = "1.0.0"
ATTIA_SOURCE_COMMIT_SHA: str = "0068fd0136bcd65884f5cd94b2b967c1ba73a668"


def generate_attia_simulator_seed(benchmark_seed: int, policy_id: str) -> int:
    """Generates a deterministic integer seed for the Attia simulator from benchmark_seed and policy_id.

    Fair Stochastic Seeding Invariant:
    For a given benchmark_seed, querying a specific policy_id always receives the exact same stochastic
    noise draw, regardless of strategy, query order, or query step.
    """
    key = f"attia_seed:{benchmark_seed}:{policy_id}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    # 31-bit positive integer
    return int(h[:8], 16) % (2**31 - 1)


_SIMULATOR_CACHE: dict[tuple[float, float, float, str, bool, int], int] = {}


def simulate_attia_policy(
    c1: float,
    c2: float,
    c3: float,
    mode: str = "hi",
    variance: bool = True,
    seed: int = 0,
) -> int:
    """Simulates battery lifetime under 4-step fast charging using Attia et al. 2020 thermal-degradation PDE model.

    Mathematical Model:
    - 1D radial heat conduction PDE in cylindrical coordinates for 18650 cell (R=9mm, L=65mm).
    - Joule heating from internal resistance: e_gen = (I^2 * R_int) / V.
    - 4 successive 20% SOC charge steps at rates C1, C2, C3, C4.
    - Arrhenius degradation rate accumulation: deg_rates = sum(A * exp(-Ea / (k_B * T(r, t))) * dt).
    - True latent lifetime: lifetime_true = int(1 / deg_rates / 2e10) + 500 (for mode 'hi').
    - Measured lifetime with cell-to-cell stochastic variation: lifetime_meas = int(Gauss(lifetime_true, sigma=164)).

    NOTE: Outputs are simulated lifetimes from a numerical model with stochastic noise, NOT physical experiments.
    """
    cache_key = (round(float(c1), 5), round(float(c2), 5), round(float(c3), 5), str(mode), bool(variance), int(seed))
    if cache_key in _SIMULATOR_CACHE:
        return _SIMULATOR_CACHE[cache_key]

    # Deterministic RNG state given seed and policy coordinates
    rng_seed = int((seed * 1000 + c1 * 10 + c2 * 20 + c3 * 30) % (2**31 - 1))
    sim_rng = random.Random(rng_seed)

    sigma = 164 if variance else 0

    c4 = float(compute_expected_c4(c1, c2, c3))
    if not math.isfinite(c4) or c4 <= 0:
        raise ValueError(f"Invalid policy coordinates: C1={c1}, C2={c2}, C3={c3} yields invalid C4={c4}")

    # Physical parameters (A123 APR18650M1A cell)
    r_cell = 0.009       # [m] radius
    l_cell = 0.065       # [m] length
    r_int = 0.017        # [Ohms] internal resistance
    t_init = 30.0        # [deg C] initial temperature
    t_inf = 30.0         # [deg C] ambient temperature
    v_cell = math.pi * r_cell**2 * l_cell  # [m^3] volume

    k_therm = 0.20       # [W/m-K] thermal conductivity
    c_p = 1000.0         # [J/kg-K] specific heat capacity
    rho = 2362.0         # [kg/m^3] density
    h_conv = 10.0        # [W/m^2-K] heat transfer coefficient (air)

    if mode == "lo":
        a_arrh = 5e21
        ea_arrh = 0.25
    elif mode == "med":
        a_arrh = 4.0
        ea_arrh = 0.10
    elif mode == "hi":
        a_arrh = 136.0
        ea_arrh = 0.122
    else:
        raise ValueError(f"Unrecognized simulator mode {mode!r}. Must be 'lo', 'med', or 'hi'.")

    kb = 8.617e-5        # [eV/K] Boltzmann constant

    # Current and step times for 1.1 Ah nominal cell, 20% SOC step = 0.22 Ah
    q_step = 1.1 * 0.20
    i1, i2, i3, i4 = c1 * 1.1, c2 * 1.1, c3 * 1.1, c4 * 1.1
    t1, t2, t3, t4 = (q_step / i1) * 3600.0, (q_step / i2) * 3600.0, (q_step / i3) * 3600.0, (q_step / i4) * 3600.0

    alpha = k_therm / (rho * c_p)
    e_gen1 = (i1**2 * r_int) / v_cell
    e_gen2 = (i2**2 * r_int) / v_cell
    e_gen3 = (i3**2 * r_int) / v_cell
    e_gen4 = (i4**2 * r_int) / v_cell

    # Spatial and temporal discretization
    dr = 0.001           # [m] spatial step
    dt = 20.0            # [s] temporal step
    n_nodes = int(np.round(r_cell / dr + 1))
    r_coords = np.arange(-dr, r_cell + 2 * dr, dr)
    temp_grid = t_init * np.ones((n_nodes + 2, 1))

    time_elapsed = 0.0
    deg_rates = 0.0

    def _solve_pde_step(tin: np.ndarray, e_gen: float) -> np.ndarray:
        mat = np.zeros((n_nodes + 2, n_nodes + 2))
        rhs = np.zeros((n_nodes + 2, 1))

        # Internal domain nodes
        for i in range(2, n_nodes + 1):
            mat[i, i] = (r_coords[i] * dr**2 * dt) * (1.0 / (alpha * dt) + 2.0 / (dr**2))
            mat[i, i + 1] = (r_coords[i] * dr**2 * dt) * (-1.0 / (2.0 * r_coords[i] * dr) - 1.0 / (dr**2))
            mat[i, i - 1] = (r_coords[i] * dr**2 * dt) * (1.0 / (2.0 * r_coords[i] * dr) - 1.0 / (dr**2))
            rhs[i, 0] = (r_coords[i] * dr**2 * dt) * (tin[i, 0] / (alpha * dt) + e_gen / k_therm)

        # Symmetry BC at r = 0
        mat[0, 0] = 1.0
        mat[0, 2] = -1.0

        # Cartesian singularity at r = 0
        mat[1, 1] = (dr**2 * dt) * (4.0 / (dr**2))
        mat[1, 2] = (dr**2 * dt) * (1.0 / dt - 4.0 / (dr**2))
        rhs[1, 0] = (dr**2 * dt) * (tin[1, 0] / dt + e_gen / k_therm)

        # Convective BC at r = R
        mat[n_nodes + 1, n_nodes] = -h_conv
        mat[n_nodes + 1, n_nodes + 1] = -k_therm / (2.0 * dr)
        mat[n_nodes + 1, n_nodes - 1] = k_therm / (2.0 * dr)
        rhs[n_nodes + 1, 0] = -h_conv * t_inf

        return np.linalg.lstsq(mat, rhs, rcond=None)[0]

    # Integrate across all 4 charge steps
    steps = [(t1, e_gen1), (t2, e_gen2), (t3, e_gen3), (t4, e_gen4)]
    for t_step, e_gen in steps:
        step_end = time_elapsed + t_step
        while time_elapsed < step_end:
            temp_grid = _solve_pde_step(temp_grid, e_gen)
            time_elapsed += dt
            # Accumulate Arrhenius degradation rate across spatial nodes
            deg_rates += float(np.sum(a_arrh * np.exp(-ea_arrh / (kb * temp_grid))))

    if mode == "lo":
        lifetime_true = int(1.0 / deg_rates / 1e10) + 900
    elif mode == "med":
        lifetime_true = int(1.0 / deg_rates / 4e9) + 900
    elif mode == "hi":
        lifetime_true = int(1.0 / deg_rates / 2e10) + 500

    lifetime_meas = int(sim_rng.gauss(lifetime_true, sigma))
    if lifetime_meas < 1:
        lifetime_meas = 1

    _SIMULATOR_CACHE[cache_key] = lifetime_meas
    return lifetime_meas


class AttiaSimulatorOracle:
    """Stochastic simulator oracle wrapping Attia et al. 2020 fast-charging PDE simulation.

    Strict Contract:
    - Accepts ONLY candidates from the canonical 224-policy universe.
    - Requires 'policy_id'.
    - Rejects missing, unknown, or coordinate-conflicting queries.
    """

    def __init__(
        self,
        policies_df: pd.DataFrame,
        mode: str = "hi",
        variance: bool = True,
    ) -> None:
        self.policies_df = policies_df.copy()
        self.mode = mode
        self.variance = variance

        if "policy_id" not in self.policies_df.columns:
            raise ValueError("policies_df must contain 'policy_id' column")

        self._policy_lookup: dict[str, dict[str, float]] = {}
        for _, row in self.policies_df.iterrows():
            pid = str(row["policy_id"])
            self._policy_lookup[pid] = {
                "C1": float(row["C1"]),
                "C2": float(row["C2"]),
                "C3": float(row["C3"]),
                "C4": float(row["C4"]),
            }

    def query(
        self,
        candidate: Mapping[str, Any] | pd.Series,
        seed: int = 0,
    ) -> OracleResponse:
        """Queries the simulator oracle for a canonical candidate policy and stochastic experiment seed."""
        cand_dict = candidate.to_dict() if isinstance(candidate, pd.Series) else dict(candidate)

        if "policy_id" not in cand_dict or cand_dict["policy_id"] is None:
            raise ValueError("AttiaSimulatorOracle requires 'policy_id' in candidate query.")

        pid = str(cand_dict["policy_id"])
        if pid not in self._policy_lookup:
            raise KeyError(
                f"Unknown policy_id {pid!r}. AttiaSimulatorOracle only accepts canonical policies "
                f"({len(self._policy_lookup)} valid policies)."
            )

        stored = self._policy_lookup[pid]
        # Validate coordinates if provided (suitable for 3-decimal author policy file)
        for col in ["C1", "C2", "C3", "C4"]:
            if col in cand_dict and cand_dict[col] is not None:
                if not np.isclose(float(cand_dict[col]), stored[col], atol=1e-3):
                    raise ValueError(
                        f"Candidate coordinate {col}={cand_dict[col]} conflicts with canonical policy {pid} definition ({stored[col]})."
                    )

        c1, c2, c3, c4 = stored["C1"], stored["C2"], stored["C3"], stored["C4"]

        sim_lifetime = simulate_attia_policy(
            c1=c1,
            c2=c2,
            c3=c3,
            mode=self.mode,
            variance=self.variance,
            seed=seed,
        )

        return OracleResponse(
            candidate={"policy_id": pid, "C1": c1, "C2": c2, "C3": c3, "C4": c4},
            observations={},
            target=float(sim_lifetime),
            metadata={
                "candidate_id": pid,
                "simulated": True,
                "simulator_mode": self.mode,
                "simulator_variance": self.variance,
                "simulator_seed": seed,
                "data_type": "simulated_lifetime",
            },
        )


def compute_or_load_reference_landscape(
    adapter: AttiaAdapter,
    eval_seeds: Sequence[int] | None = None,
    mode: str = "hi",
    output_path: Path | str | None = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Computes or loads the reference landscape across all policies with manifest validation.

    Benchmark Reference Objective:
    - Primary Latent Objective: `reference_true_lifetime` = simulator(policy, variance=False)
      (representing the exact underlying deterministic physical PDE solution).
    - Diagnostic Metrics: `reference_mean_lifetime` and `reference_std_lifetime` across 50 stochastic seeds.

    IMPORTANT: This reference table is strictly evaluator-only and must never be exposed to candidate generation,
    surrogates, or acquisition/optimization loops.
    """
    if eval_seeds is None:
        eval_seeds = DEFAULT_EVAL_SEEDS

    if output_path is None:
        project_root = Path(__file__).resolve().parents[2]
        output_path = project_root / "outputs" / "attia" / "reference_landscape.csv"
    else:
        output_path = Path(output_path)

    manifest_path = output_path.parent / "reference_landscape_manifest.json"

    # Compute hash of raw policies file for manifest verification
    policies_raw_file = adapter.raw_policies_file
    policies_file_sha256 = ""
    if policies_raw_file.exists():
        policies_file_sha256 = hashlib.sha256(policies_raw_file.read_bytes()).hexdigest()

    # Validate cache against manifest
    cache_valid = False
    if output_path.exists() and manifest_path.exists() and not force_recompute:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            if (
                manifest_data.get("attia_source_commit_sha") == ATTIA_SOURCE_COMMIT_SHA
                and manifest_data.get("policies_file_sha256") == policies_file_sha256
                and manifest_data.get("simulator_mode") == mode
                and manifest_data.get("simulator_version") == SIMULATOR_VERSION
                and manifest_data.get("evaluation_seed_list") == list(eval_seeds)
                and manifest_data.get("n_policies") == len(adapter.load_policies())
                and manifest_data.get("reference_objective") == "reference_true_lifetime"
            ):
                cache_valid = True
        except Exception:
            cache_valid = False

    if cache_valid:
        ref_df = pd.read_csv(output_path)
        if (
            len(ref_df) == len(adapter.load_policies())
            and "reference_true_lifetime" in ref_df.columns
            and "reference_mean_lifetime" in ref_df.columns
        ):
            global_max = float(ref_df["reference_true_lifetime"].max())
            top_10_pct = float(np.percentile(ref_df["reference_true_lifetime"], 90))
            top_5_pct = float(np.percentile(ref_df["reference_true_lifetime"], 95))
            meta = {
                "global_max": global_max,
                "top_10_pct_val": top_10_pct,
                "top_5_pct_val": top_5_pct,
                "reference_objective": "reference_true_lifetime",
                "n_eval_seeds": len(eval_seeds),
                "eval_seeds": list(eval_seeds),
                "mode": mode,
                "cache_reused": True,
            }
            return ref_df, meta

    policies_df = adapter.load_policies()
    logger.info(
        "Computing reference simulator landscape across %d policies with %d evaluation seeds...",
        len(policies_df),
        len(eval_seeds),
    )

    rows: list[dict[str, Any]] = []
    for _, row in policies_df.iterrows():
        pid = str(row["policy_id"])
        c1, c2, c3, c4 = float(row["C1"]), float(row["C2"]), float(row["C3"]), float(row["C4"])

        # Compute deterministic true latent lifetime (sigma=0)
        true_life = simulate_attia_policy(c1, c2, c3, mode=mode, variance=False, seed=0)

        # Compute stochastic draws over fixed evaluation seeds
        noisy_draws = [
            simulate_attia_policy(c1, c2, c3, mode=mode, variance=True, seed=s)
            for s in eval_seeds
        ]

        mean_life = float(np.mean(noisy_draws))
        std_life = float(np.std(noisy_draws, ddof=1)) if len(noisy_draws) > 1 else 0.0

        rows.append(
            {
                "policy_id": pid,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "reference_true_lifetime": int(true_life),
                "reference_mean_lifetime": mean_life,
                "reference_std_lifetime": std_life,
            }
        )

    ref_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(output_path, index=False)

    # Write reference landscape manifest
    csv_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest_data = {
        "attia_source_commit_sha": ATTIA_SOURCE_COMMIT_SHA,
        "policies_file_sha256": policies_file_sha256,
        "simulator_mode": mode,
        "simulator_version": SIMULATOR_VERSION,
        "n_policies": len(ref_df),
        "evaluation_seed_list": list(eval_seeds),
        "reference_objective": "reference_true_lifetime",
        "csv_sha256": csv_hash,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "notes": (
            "reference_true_lifetime is the deterministic latent PDE thermal-degradation objective (variance=False). "
            "reference_mean_lifetime is the sample mean across 50 stochastic Gaussian variation draws (sigma=164)."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    global_max = float(ref_df["reference_true_lifetime"].max())
    top_10_pct = float(np.percentile(ref_df["reference_true_lifetime"], 90))
    top_5_pct = float(np.percentile(ref_df["reference_true_lifetime"], 95))

    meta = {
        "global_max": global_max,
        "top_10_pct_val": top_10_pct,
        "top_5_pct_val": top_5_pct,
        "reference_objective": "reference_true_lifetime",
        "n_eval_seeds": len(eval_seeds),
        "eval_seeds": list(eval_seeds),
        "mode": mode,
        "cache_reused": False,
    }

    return ref_df, meta
