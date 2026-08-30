from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.feconi import FeCoNiAdapter


def ternary_to_cartesian(co: np.ndarray, fe: np.ndarray, ni: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts ternary compositions (summing to ~100) to 2D Cartesian coordinates for plotting."""
    # Standard equilateral triangle:
    # Bottom left: (0, 0) -> 100% Fe
    # Bottom right: (1, 0) -> 100% Ni
    # Top apex: (0.5, sqrt(3)/2) -> 100% Co
    total = np.maximum(co + fe + ni, 1e-6)
    c_norm = co / total
    f_norm = fe / total
    n_norm = ni / total

    x = 0.5 * c_norm + n_norm
    y = (np.sqrt(3) / 2.0) * c_norm
    return x, y


def plot_ternary_landscapes(output_dir: Path) -> None:
    adapter = FeCoNiAdapter(target="Kerr")
    df = adapter.load_data()

    co = df["Co"].to_numpy()
    fe = df["Fe"].to_numpy()
    ni = df["Ni"].to_numpy()
    kerr = df["Kerr"].to_numpy()
    coer = df["Coer"].to_numpy()

    x, y = ternary_to_cartesian(co, fe, ni)

    # Triangle boundary
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, np.sqrt(3) / 2.0, 0.0]

    # 1. Kerr Map
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri_x, tri_y, "k-", lw=1.5)
    sc = ax.scatter(x, y, c=kerr, cmap="viridis", s=35, edgecolors="none", alpha=0.9)
    cb = plt.colorbar(sc, ax=ax, label="Kerr Rotation [mrad]")
    ax.text(-0.02, -0.05, "100% Fe", fontsize=11, ha="center", weight="bold")
    ax.text(1.02, -0.05, "100% Ni", fontsize=11, ha="center", weight="bold")
    ax.text(0.5, np.sqrt(3)/2.0 + 0.04, "100% Co", fontsize=11, ha="center", weight="bold")
    # Mark global max
    max_kerr_idx = np.argmax(kerr)
    ax.scatter(x[max_kerr_idx], y[max_kerr_idx], c="red", marker="*", s=200, edgecolors="black", label=f"Global Max ({kerr[max_kerr_idx]:.4f} mrad)")
    ax.set_title("NIST Fe-Co-Ni: Experimental Kerr Rotation Landscape\n(921 Measured Samples)", fontsize=12, pad=15)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "kerr_ternary_scatter_map.png", dpi=300)
    plt.close(fig)

    # 2. Coercivity Map
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri_x, tri_y, "k-", lw=1.5)
    sc = ax.scatter(x, y, c=coer, cmap="plasma", s=35, edgecolors="none", alpha=0.9)
    cb = plt.colorbar(sc, ax=ax, label="Coercivity [mT]")
    ax.text(-0.02, -0.05, "100% Fe", fontsize=11, ha="center", weight="bold")
    ax.text(1.02, -0.05, "100% Ni", fontsize=11, ha="center", weight="bold")
    ax.text(0.5, np.sqrt(3)/2.0 + 0.04, "100% Co", fontsize=11, ha="center", weight="bold")
    # Mark global max
    max_coer_idx = np.argmax(coer)
    ax.scatter(x[max_coer_idx], y[max_coer_idx], c="cyan", marker="*", s=200, edgecolors="black", label=f"Global Max ({coer[max_coer_idx]:.2f} mT)")
    ax.set_title("NIST Fe-Co-Ni: Experimental Magnetic Coercivity Landscape\n(921 Measured Samples)", fontsize=12, pad=15)
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "coercivity_ternary_scatter_map.png", dpi=300)
    plt.close(fig)


def plot_benchmark_curves(per_step_csv: Path, output_dir: Path, target_title: str, prefix: str) -> None:
    df = pd.read_csv(per_step_csv)
    strategies = df["strategy"].unique()

    # 3. Mean Best-so-Far vs Samples & 5. CI bands
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "random": "#7f7f7f",
        "greedy": "#1f77b4",
        "gp_ucb": "#ff7f0e",
        "gp_ucb_1": "#aec7e8",
        "thompson_sampling": "#2ca02c",
        "expected_improvement": "#d62728",
        "ei": "#d62728",
        "noisy_expected_improvement": "#9467bd",
        "nei": "#9467bd",
        "turbo_nei": "#8c564b",
    }

    for strat in strategies:
        strat_df = df[df["strategy"] == strat]
        grouped = strat_df.groupby("iteration")["best_observed"]
        iters = grouped.mean().index.to_numpy()
        means = grouped.mean().to_numpy()
        stds = grouped.std().to_numpy()
        ns = grouped.count().to_numpy()
        sems = stds / np.maximum(np.sqrt(ns), 1)

        c = colors.get(strat, "#333333")
        label = strat.replace("_", " ").title()
        ax.plot(iters, means, label=label, color=c, lw=2)
        ax.fill_between(iters, means - 1.96 * sems, means + 1.96 * sems, color=c, alpha=0.15)

    global_best = df["global_best"].iloc[0]
    ax.axhline(global_best, color="k", linestyle="--", alpha=0.7, label=f"Global Max ({global_best:.4f})")
    ax.set_xlabel("Number of Materials Evaluated (Queries)", fontsize=11)
    ax.set_ylabel(f"Best Observed {target_title}", fontsize=11)
    ax.set_title(f"Optimization Trajectory: Best Observed {target_title} (95% CI)", fontsize=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_best_so_far_vs_samples.png", dpi=300)
    plt.close(fig)

    # 4. Mean Regret vs Samples
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat in strategies:
        strat_df = df[df["strategy"] == strat]
        grouped = strat_df.groupby("iteration")["regret"]
        iters = grouped.mean().index.to_numpy()
        means = grouped.mean().to_numpy()
        stds = grouped.std().to_numpy()
        ns = grouped.count().to_numpy()
        sems = stds / np.maximum(np.sqrt(ns), 1)

        c = colors.get(strat, "#333333")
        label = strat.replace("_", " ").title()
        ax.plot(iters, means, label=label, color=c, lw=2)
        ax.fill_between(iters, np.maximum(0, means - 1.96 * sems), means + 1.96 * sems, color=c, alpha=0.15)

    ax.set_xlabel("Number of Materials Evaluated (Queries)", fontsize=11)
    ax.set_ylabel("Simple Maximization Regret ($y^* - y_{best}$)", fontsize=11)
    ax.set_title(f"Regret vs Budget: {target_title} Optimization", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_regret_vs_samples.png", dpi=300)
    plt.close(fig)


def plot_threshold_comparison(summary_json: Path, output_dir: Path, target_title: str, prefix: str) -> None:
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    strategies = list(data["strategies"].keys())
    sr_10 = [data["strategies"][s]["success_rate_10pct"] * 100 for s in strategies]
    sr_5 = [data["strategies"][s]["success_rate_5pct"] * 100 for s in strategies]
    sr_1 = [data["strategies"][s]["success_rate_1pct"] * 100 for s in strategies]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, sr_10, width, label="Within 10% of Optimum", color="#4daf4a")
    ax.bar(x, sr_5, width, label="Within 5% of Optimum", color="#377eb8")
    ax.bar(x + width, sr_1, width, label="Within 1% of Optimum", color="#e41a1c")

    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title(f"Success Rate by Target Proximity: {target_title}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_threshold_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    out_dir = Path("outputs/feconi/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ternary landscape plots...")
    plot_ternary_landscapes(out_dir)

    # If reproduction results exist
    kerr_repro_csv = Path("outputs/feconi/reproduction/kerr/per_step.csv")
    kerr_repro_json = Path("outputs/feconi/reproduction/kerr/summary.json")
    if kerr_repro_csv.is_file() and kerr_repro_json.is_file():
        print("Generating reproduction plots for Kerr...")
        plot_benchmark_curves(kerr_repro_csv, out_dir, "Kerr Rotation", "reproduction_kerr")
        plot_threshold_comparison(kerr_repro_json, out_dir, "Kerr Rotation", "reproduction_kerr")

    coer_repro_csv = Path("outputs/feconi/reproduction/coercivity/per_step.csv")
    coer_repro_json = Path("outputs/feconi/reproduction/coercivity/summary.json")
    if coer_repro_csv.is_file() and coer_repro_json.is_file():
        print("Generating reproduction plots for Coercivity...")
        plot_benchmark_curves(coer_repro_csv, out_dir, "Coercivity", "reproduction_coercivity")
        plot_threshold_comparison(coer_repro_json, out_dir, "Coercivity", "reproduction_coercivity")

    # If AIcoScientist results exist
    kerr_ai_csv = Path("outputs/feconi/aicoscientist/kerr/per_step.csv")
    kerr_ai_json = Path("outputs/feconi/aicoscientist/kerr/summary.json")
    if kerr_ai_csv.is_file() and kerr_ai_json.is_file():
        print("Generating AIcoScientist plots for Kerr...")
        plot_benchmark_curves(kerr_ai_csv, out_dir, "Kerr Rotation", "aicoscientist_kerr")
        plot_threshold_comparison(kerr_ai_json, out_dir, "Kerr Rotation", "aicoscientist_kerr")

    coer_ai_csv = Path("outputs/feconi/aicoscientist/coercivity/per_step.csv")
    coer_ai_json = Path("outputs/feconi/aicoscientist/coercivity/summary.json")
    if coer_ai_csv.is_file() and coer_ai_json.is_file():
        print("Generating AIcoScientist plots for Coercivity...")
        plot_benchmark_curves(coer_ai_csv, out_dir, "Coercivity", "aicoscientist_coercivity")
        plot_threshold_comparison(coer_ai_json, out_dir, "Coercivity", "aicoscientist_coercivity")

    print("Plotting complete! Saved to:", out_dir)


if __name__ == "__main__":
    main()
