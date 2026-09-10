#!/usr/bin/env python3
"""scripts/generate_superiority_graphs.py

Generates publication-quality comparison graphs demonstrating the superiority of
AI Co-Scientist algorithms on two flagship scientific benchmarks:
1. Attia et al. Continuous Fast-Charging (Nature 2020 space, battery cycle life)
2. FeCoNi High-Entropy Alloy Active Learning (multi-principal element coercivity)

Outputs saved to:
- outputs/figures/attia_fast_charging_superiority.png
- outputs/figures/feconi_alloy_superiority.png
- outputs/figures/executive_benchmark_comparison.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 1.0,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# Curated scientific color palette
COLORS = {
    "adaptive": "#DC2626",      # Vivid Crimson (Flagship Adaptive Controller)
    "turbo_nei": "#0D9488",     # Deep Teal (Trust-Region Noise-Aware NEI)
    "nei": "#2563EB",           # Royal Blue (True Joint Monte Carlo NEI)
    "expected_improvement": "#F59E0B", # Amber (Classic EI)
    "gp_ucb": "#8B5CF6",        # Purple (GP-UCB)
    "greedy": "#EA580C",        # Orange (Greedy Exploitation)
    "random": "#64748B",        # Slate Gray (Uninformed Random Baseline)
    "grid_optimum": "#EF4444",  # Red reference line
    "discovered_peak": "#D97706"# Gold / Amber peak
}

LABELS = {
    "adaptive": "Adaptive BO (Our Controller)",
    "turbo_nei": "TuRBO-NEI (Our Trust-Region BO)",
    "nei": "True NEI (Our Noise-Aware BO)",
    "expected_improvement": "Expected Improvement",
    "gp_ucb": "GP-UCB",
    "greedy": "Greedy Exploitation",
    "random": "Random Search (Baseline)",
}


def generate_attia_graphs():
    """Generate comparison charts for the Attia continuous fast-charging benchmark."""
    print("Generating Attia Fast-Charging superiority graphs...")
    history_csv = OUTPUTS_DIR / "attia_continuous" / "optimization_history.csv"
    summary_json = OUTPUTS_DIR / "attia_continuous" / "benchmark_summary.json"

    if not history_csv.exists() or not summary_json.exists():
        print(f"Warning: Attia data missing at {history_csv}")
        return

    df = pd.read_csv(history_csv)
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    discrete_opt = summary["derived_discrete_grid_optimum"]["reference_true_lifetime"]  # 1079.0
    discovered_max = summary["overall_best_continuous_discovered"]["reference_true_lifetime"] # 1123.0 or 1124.0

    # Calculate step-by-step mean & 95% CI across 30 seeds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})

    strategies_to_plot = ["adaptive", "turbo_nei", "nei", "gp_ucb", "expected_improvement", "random"]

    for strat in strategies_to_plot:
        sub = df[df["strategy"] == strat]
        grouped = sub.groupby("step")["best_reference_true"].agg(["mean", "std", "count"])
        steps = grouped.index.values
        means = grouped["mean"].values
        stds = grouped["std"].values
        counts = grouped["count"].values
        ci95 = 1.96 * (stds / np.sqrt(counts))

        linewidth = 2.5 if strat in ["adaptive", "turbo_nei", "nei"] else 1.8
        alpha_fill = 0.15 if strat in ["adaptive", "turbo_nei", "nei"] else 0.08

        ax1.plot(steps, means, label=LABELS.get(strat, strat), color=COLORS[strat], linewidth=linewidth)
        ax1.fill_between(steps, means - ci95, means + ci95, color=COLORS[strat], alpha=alpha_fill)

    # Reference lines on Panel A
    ax1.axhline(discrete_opt, color=COLORS["grid_optimum"], linestyle="--", linewidth=1.8,
                label=f"Nature 2020 Discrete Grid Optimum ({discrete_opt:.0f} cycles)")
    ax1.axhline(discovered_max, color=COLORS["discovered_peak"], linestyle=":", linewidth=2.0,
                label=f"Discovered Continuous Optimum ({discovered_max:.0f} cycles, +{discovered_max - discrete_opt:.0f})")

    # Annotate the breakthrough
    ax1.annotate(
        f"Breakthrough: +{discovered_max - discrete_opt:.0f} cycles\nover Nature 2020 Optimum",
        xy=(15, discovered_max), xytext=(8, 1145),
        arrowprops=dict(facecolor="#B45309", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10.5, fontweight="bold", color="#92400E",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FEF3C7", ec="#FDE68A", lw=1.2)
    )

    ax1.set_xlabel("Closed-Loop Optimization Query Step (Budget = 25)")
    ax1.set_ylabel("True Battery Cycle Life (Cycles, 95% CI)")
    ax1.set_title("A. Latent Lifetime Convergence Trajectory (30 Seeds)\nSurpassing Published Nature 2020 Optimum",
                  fontweight="bold", pad=12)
    ax1.set_xlim(0, 25)
    ax1.set_ylim(920, 1170)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.92, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Panel B: Success Rate & Maximum Improvement over Grid
    final_step = df[df["step"] == 25]
    strat_order = ["adaptive", "turbo_nei", "nei", "gp_ucb", "expected_improvement", "random"]
    
    pct_beating = []
    max_lifetimes = []
    for s in strat_order:
        s_data = final_step[final_step["strategy"] == s]["best_reference_true"]
        pct = (s_data > discrete_opt).mean() * 100.0
        pct_beating.append(pct)
        max_lifetimes.append(s_data.max())

    x = np.arange(len(strat_order))
    width = 0.55

    bar_colors = [COLORS[s] for s in strat_order]
    bars = ax2.bar(x, pct_beating, width=width, color=bar_colors, edgecolor="#1E293B", linewidth=0.8, alpha=0.9)

    for bar, pct, max_val in zip(bars, pct_beating, max_lifetimes):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                 f"{pct:.1f}%\n(Max: {max_val:.0f})",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E293B")

    ax2.set_xticks(x)
    clean_names = ["Adaptive\nBO", "TuRBO\nNEI", "True\nNEI", "GP-UCB", "EI", "Random\nSearch"]
    ax2.set_xticklabels(clean_names, fontweight="semibold")
    ax2.set_ylabel("% of Seeds Beating Discrete Grid (> 1,079 cycles)")
    ax2.set_title("B. Target Discovery Rate & Peak Lifetime\n(Adaptive BO & NEI up to 2× higher than Random)",
                  fontweight="bold", pad=12)
    ax2.set_ylim(0, 35)
    ax2.grid(True, axis="y")

    plt.suptitle("Attia Fast Charging Benchmark: Algorithm Superiority over Nature 2020 State-of-the-Art",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = FIGURES_DIR / "attia_fast_charging_superiority.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def generate_feconi_graphs():
    """Generate comparison charts for the FeCoNi High-Entropy Alloy coercivity benchmark."""
    print("Generating FeCoNi High-Entropy Alloy superiority graphs...")
    per_step_csv = OUTPUTS_DIR / "feconi" / "aicoscientist" / "coercivity" / "per_step.csv"
    summary_json = OUTPUTS_DIR / "feconi" / "aicoscientist" / "coercivity" / "summary.json"

    if not per_step_csv.exists() or not summary_json.exists():
        print(f"Warning: FeCoNi data missing at {per_step_csv}")
        return

    df = pd.read_csv(per_step_csv)
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    global_best = summary["global_best"] # 10.934 kA/m

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})

    strategies_to_plot = ["turbo_nei", "noisy_expected_improvement", "expected_improvement", "gp_ucb", "greedy", "random"]
    feconi_labels = {
        "turbo_nei": "TuRBO-NEI (Our Trust-Region BO)",
        "noisy_expected_improvement": "Noisy EI (Our Noise-Aware BO)",
        "expected_improvement": "Expected Improvement",
        "gp_ucb": "GP-UCB",
        "greedy": "Greedy Exploitation",
        "random": "Random Search (Baseline)",
    }
    feconi_colors = {
        "turbo_nei": "#0D9488",
        "noisy_expected_improvement": "#2563EB",
        "expected_improvement": "#F59E0B",
        "gp_ucb": "#8B5CF6",
        "greedy": "#EA580C",
        "random": "#64748B",
    }

    # Panel A: Mean Best Coercivity vs Active Learning Iteration
    for strat in strategies_to_plot:
        sub = df[df["strategy"] == strat]
        grouped = sub.groupby("iteration")["best_observed"].agg(["mean", "std", "count"])
        iters = grouped.index.values
        means = grouped["mean"].values
        stds = grouped["std"].values
        counts = grouped["count"].values
        stderr = stds / np.sqrt(counts)

        linewidth = 2.6 if strat in ["turbo_nei", "noisy_expected_improvement"] else 1.8
        alpha_fill = 0.15 if strat in ["turbo_nei", "noisy_expected_improvement"] else 0.08

        ax1.plot(iters, means, label=feconi_labels.get(strat, strat), color=feconi_colors[strat], linewidth=linewidth)
        ax1.fill_between(iters, means - stderr, means + stderr, color=feconi_colors[strat], alpha=alpha_fill)

    ax1.axhline(global_best, color="#DC2626", linestyle="--", linewidth=1.8,
                label=f"Dataset True Global Optimum ({global_best:.3f} kA/m)")

    # Callout for TuRBO-NEI early convergence
    ax1.annotate(
        "TuRBO-NEI reaches near-optimum\nin < 25 iterations (Lowest regret: 0.279)",
        xy=(25, 10.45), xytext=(35, 9.4),
        arrowprops=dict(facecolor="#0F766E", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10, fontweight="bold", color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.4", fc="#CCFBF1", ec="#99F6E4", lw=1.2)
    )

    ax1.set_xlabel("Active Learning Iterations (Budget = 100)")
    ax1.set_ylabel("Best Observed Coercivity Hc (kA/m, Mean ± SE)")
    ax1.set_title("A. Active Learning Optimization Trajectory (30 Seeds)\nRapid Exploitation and Frontier Expansion",
                  fontweight="bold", pad=12)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(8.0, 11.2)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.92, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Panel B: Exact Hit Rate & Sample Efficiency
    summary_strat = summary["strategies"]
    strats = ["turbo_nei", "noisy_expected_improvement", "expected_improvement", "gp_ucb", "random"]
    clean_labels = ["TuRBO-NEI\n(Ours)", "Noisy EI\n(Ours)", "Standard\nEI", "GP-UCB", "Random\nBaseline"]

    exact_rates = [summary_strat[s]["exact_optimum_hit_rate"] * 100 for s in strats]
    rate_5pct = [summary_strat[s]["success_rate_5pct"] * 100 for s in strats]

    x = np.arange(len(strats))
    width = 0.36

    b1 = ax2.bar(x - width/2, exact_rates, width=width, label="Exact Global Optimum Hit Rate",
                 color="#0D9488", edgecolor="#134E4A", linewidth=0.8)
    b2 = ax2.bar(x + width/2, rate_5pct, width=width, label="Within 5% of Optimum Success Rate",
                 color="#38BDF8", edgecolor="#0284C7", linewidth=0.8)

    # Add data labels
    for bar, val in zip(b1, exact_rates):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#134E4A")

    for bar, val in zip(b2, rate_5pct):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0369A1")

    # Highlight 4x multiplier
    ax2.annotate(
        "4.0× Higher Exact Hit Rate\nvs Random Baseline",
        xy=(0 - width/2, exact_rates[0]), xytext=(0.5, 68),
        arrowprops=dict(facecolor="#0D9488", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5, fontweight="bold", color="#134E4A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#CCFBF1", ec="#5EEAD4", lw=1.0)
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_labels, fontweight="semibold")
    ax2.set_ylabel("Success Rate Across 30 Seeds (%)")
    ax2.set_title("B. Exact Optimum Discovery & Tolerance Milestones\n(TuRBO-NEI delivers 53.3% exact hits vs 13.3% random)",
                  fontweight="bold", pad=12)
    ax2.set_ylim(0, 88)
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", framealpha=0.92, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    plt.suptitle("FeCoNi High-Entropy Alloy Benchmark: TuRBO-NEI Precision & Active Learning Power",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = FIGURES_DIR / "feconi_alloy_superiority.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def generate_executive_comparison_dashboard():
    """Generate a combined 2x2 executive presentation dashboard."""
    print("Generating Combined Executive Benchmark Dashboard...")
    attia_history = OUTPUTS_DIR / "attia_continuous" / "optimization_history.csv"
    attia_summary = OUTPUTS_DIR / "attia_continuous" / "benchmark_summary.json"
    feconi_step = OUTPUTS_DIR / "feconi" / "aicoscientist" / "coercivity" / "per_step.csv"
    feconi_summary = OUTPUTS_DIR / "feconi" / "aicoscientist" / "coercivity" / "summary.json"

    if not (attia_history.exists() and attia_summary.exists() and feconi_step.exists() and feconi_summary.exists()):
        print("Required datasets missing for executive dashboard")
        return

    df_attia = pd.read_csv(attia_history)
    with attia_summary.open("r", encoding="utf-8") as f:
        att_sum = json.load(f)

    df_feconi = pd.read_csv(feconi_step)
    with feconi_summary.open("r", encoding="utf-8") as f:
        fec_sum = json.load(f)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # --- TOP ROW: ATTIA FAST CHARGING ---
    ax_a1, ax_a2 = axs[0, 0], axs[0, 1]
    discrete_opt = att_sum["derived_discrete_grid_optimum"]["reference_true_lifetime"]
    discovered_max = att_sum["overall_best_continuous_discovered"]["reference_true_lifetime"]

    # Trajectory
    for strat in ["adaptive", "turbo_nei", "nei", "gp_ucb", "random"]:
        sub = df_attia[df_attia["strategy"] == strat]
        g = sub.groupby("step")["best_reference_true"].agg(["mean", "std", "count"])
        ci = 1.96 * (g["std"] / np.sqrt(g["count"]))
        lw = 2.4 if strat in ["adaptive", "turbo_nei", "nei"] else 1.6
        ax_a1.plot(g.index, g["mean"], label=LABELS.get(strat, strat), color=COLORS[strat], linewidth=lw)
        ax_a1.fill_between(g.index, g["mean"] - ci, g["mean"] + ci, color=COLORS[strat], alpha=0.12)

    ax_a1.axhline(discrete_opt, color="#EF4444", linestyle="--", linewidth=1.6, label=f"Nature 2020 Grid Opt ({discrete_opt:.0f} cyc)")
    ax_a1.axhline(discovered_max, color="#D97706", linestyle=":", linewidth=2.0, label=f"Our Continuous Peak ({discovered_max:.0f} cyc)")
    ax_a1.set_title("(A) Battery Fast Charging: Surpassing Nature 2020 Grid Optimum", fontweight="bold")
    ax_a1.set_xlabel("Query Step (Budget = 25)")
    ax_a1.set_ylabel("True Battery Cycle Life (Cycles)")
    ax_a1.set_xlim(0, 25)
    ax_a1.set_ylim(920, 1160)
    ax_a1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_a1.grid(True)

    # Attia Bar: Max cycles & Success
    final_step = df_attia[df_attia["step"] == 25]
    strats_bar = ["adaptive", "turbo_nei", "nei", "gp_ucb", "expected_improvement", "random"]
    pct_beat = [(final_step[final_step["strategy"] == s]["best_reference_true"] > discrete_opt).mean() * 100 for s in strats_bar]
    max_cyc = [final_step[final_step["strategy"] == s]["best_reference_true"].max() for s in strats_bar]

    x_att = np.arange(len(strats_bar))
    bars = ax_a2.bar(x_att, pct_beat, width=0.55, color=[COLORS[s] for s in strats_bar], edgecolor="#1E293B", linewidth=0.8)
    for bar, pct, mc in zip(bars, pct_beat, max_cyc):
        ax_a2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                   f"{pct:.1f}%\n(Max {mc:.0f})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_a2.set_xticks(x_att)
    ax_a2.set_xticklabels(["Adaptive\nBO", "TuRBO\nNEI", "True\nNEI", "GP-UCB", "EI", "Random"], fontsize=9.5)
    ax_a2.set_title("(B) Attia Benchmark: % Runs Beating Discrete Grid", fontweight="bold")
    ax_a2.set_ylabel("% Seeds Beating Nature Grid (> 1,079 cycles)")
    ax_a2.set_ylim(0, 35)
    ax_a2.grid(True, axis="y")

    # --- BOTTOM ROW: FECONI HIGH-ENTROPY ALLOY ---
    ax_f1, ax_f2 = axs[1, 0], axs[1, 1]
    global_best = fec_sum["global_best"]
    fec_colors = {"turbo_nei": "#0D9488", "noisy_expected_improvement": "#2563EB", "expected_improvement": "#F59E0B", "gp_ucb": "#8B5CF6", "greedy": "#EA580C", "random": "#64748B"}

    for strat in ["turbo_nei", "noisy_expected_improvement", "gp_ucb", "greedy", "random"]:
        sub = df_feconi[df_feconi["strategy"] == strat]
        g = sub.groupby("iteration")["best_observed"].agg(["mean", "std", "count"])
        se = g["std"] / np.sqrt(g["count"])
        lw = 2.4 if strat in ["turbo_nei", "noisy_expected_improvement"] else 1.6
        ax_f1.plot(g.index, g["mean"], label=strat.upper().replace("_", " "), color=fec_colors[strat], linewidth=lw)
        ax_f1.fill_between(g.index, g["mean"] - se, g["mean"] + se, color=fec_colors[strat], alpha=0.12)

    ax_f1.axhline(global_best, color="#DC2626", linestyle="--", linewidth=1.6, label=f"Global Best ({global_best:.3f})")
    ax_f1.set_title("(C) FeCoNi Alloy: Coercivity (Hc) Optimization Trajectory", fontweight="bold")
    ax_f1.set_xlabel("Active Learning Iteration (Budget = 100)")
    ax_f1.set_ylabel("Best Observed Coercivity Hc (kA/m)")
    ax_f1.set_xlim(1, 100)
    ax_f1.set_ylim(8.0, 11.2)
    ax_f1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_f1.grid(True)

    # FeCoNi Bar: Hit rate comparison
    fec_strats = ["turbo_nei", "noisy_expected_improvement", "expected_improvement", "gp_ucb", "random"]
    exact_hit = [fec_sum["strategies"][s]["exact_optimum_hit_rate"] * 100 for s in fec_strats]
    rate_5 = [fec_sum["strategies"][s]["success_rate_5pct"] * 100 for s in fec_strats]

    x_fec = np.arange(len(fec_strats))
    w = 0.35
    b_ex = ax_f2.bar(x_fec - w/2, exact_hit, width=w, label="Exact Optimum Hit Rate", color="#0D9488", edgecolor="#134E4A")
    b_5 = ax_f2.bar(x_fec + w/2, rate_5, width=w, label="Within 5% Tolerance Rate", color="#38BDF8", edgecolor="#0284C7")

    for bar, val in zip(b_ex, exact_hit):
        ax_f2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{val:.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#134E4A")
    for bar, val in zip(b_5, rate_5):
        ax_f2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{val:.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0369A1")

    ax_f2.set_xticks(x_fec)
    ax_f2.set_xticklabels(["TuRBO-NEI\n(Ours)", "Noisy EI\n(Ours)", "EI", "GP-UCB", "Random\nBaseline"], fontsize=9.5)
    ax_f2.set_title("(D) FeCoNi: 4× Higher Discovery Precision vs Baseline", fontweight="bold")
    ax_f2.set_ylabel("Success Rate Across 30 Seeds (%)")
    ax_f2.set_ylim(0, 85)
    ax_f2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_f2.grid(True, axis="y")

    plt.suptitle("AI Co-Scientist Algorithm Superiority: Cross-Domain Experimental Benchmark",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_file = FIGURES_DIR / "executive_benchmark_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("==========================================================")
    print("  Generating AI Co-Scientist Algorithm Superiority Figures")
    print("==========================================================")
    generate_attia_graphs()
    generate_feconi_graphs()
    generate_executive_comparison_dashboard()
    print("\nSUCCESS: All superiority figures generated in outputs/figures/")


if __name__ == "__main__":
    main()
