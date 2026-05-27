"""
Convergence plots: visualizes how the best fitness evolves over iterations.
Includes single run comparison, averaged curves, boxplots and speedup charts.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def plot_comparison(
    results_list: list[dict],
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots convergence curves of multiple PSO runs on the same axes.
    Each run gets its own color and label showing evaluator, time and seed.

    Args:
        results_list: List of results dictionaries to compare.
        title:        Custom plot title. Auto-generated if None.
        save_path:    If provided, saves the plot to this path.
        show:         If True, displays the plot interactively.
    """
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    styles = [
        {"linewidth": 2.5, "alpha": 1.0,  "linestyle": "-"},
        {"linewidth": 2.0, "alpha": 0.85, "linestyle": "--"},
        {"linewidth": 2.0, "alpha": 0.85, "linestyle": "-."},
        {"linewidth": 1.5, "alpha": 0.7,  "linestyle": ":"},
        {"linewidth": 1.5, "alpha": 0.7,  "linestyle": "-"},
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, results in enumerate(results_list):
        curve      = results["convergence_curve"]
        iterations = [entry["iter"] for entry in curve]
        fitness    = [entry["best_fitness"] for entry in curve]
        objective  = results["config"]["objective"]
        dim        = results["config"]["dim"]
        evaluator  = results["config"]["evaluator"].split("(")[0]
        total_time = results["timing"]["total_s"]
        seed       = results["metadata"]["seed"]

        label = f"{objective} d={dim} | {evaluator} | {total_time:.3f}s | seed={seed}"
        ax.plot(
            iterations, fitness,
            color=colors[i % len(colors)],
            label=label,
            **styles[i % len(styles)],
        )

    ref       = results_list[0]
    objective = ref["config"]["objective"]
    dim       = ref["config"]["dim"]

    ax.set_title(title or "PSO Convergence Comparison", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


def plot_averaged_convergence(
    results_list: list[dict],
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots averaged convergence curves across multiple seeds per strategy.
    Shows mean curve with shaded standard deviation band.
    Groups results by evaluator and objective automatically.

    Args:
        results_list: List of results dictionaries from multiple seeds and strategies.
        title:        Custom plot title. Auto-generated if None.
        save_path:    If provided, saves the plot to this path.
        show:         If True, displays the plot interactively.
    """
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]

    # Group results by (evaluator, objective, dim)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results_list:
        evaluator = r["config"]["evaluator"].split("(")[0]
        objective = r["config"]["objective"]
        dim       = r["config"]["dim"]
        key       = f"{evaluator} | {objective} d={dim}"
        groups[key].append(r)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, group) in enumerate(groups.items()):
        # Find minimum curve length across seeds
        min_len = min(len(r["convergence_curve"]) for r in group)

        # Stack fitness curves — shape (n_seeds, min_len)
        curves = np.array([
            [entry["best_fitness"] for entry in r["convergence_curve"][:min_len]]
            for r in group
        ])

        iterations = list(range(min_len))
        mean_curve = curves.mean(axis=0)
        std_curve  = curves.std(axis=0)

        color = colors[i % len(colors)]

        ax.plot(iterations, mean_curve, color=color, linewidth=2.0,
                label=f"{label} (n={len(group)} seeds)")
        ax.fill_between(
            iterations,
            np.maximum(mean_curve - std_curve, 1e-12),
            mean_curve + std_curve,
            color=color, alpha=0.15,
        )

    ax.set_title(title or "PSO Averaged Convergence (mean ± std)", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


def plot_boxplots(
    results_list: list[dict],
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Boxplots of final best fitness grouped by evaluator strategy.
    Shows distribution of results across multiple seeds.

    Args:
        results_list: List of results dictionaries from multiple seeds and strategies.
        title:        Custom plot title. Auto-generated if None.
        save_path:    If provided, saves the plot to this path.
        show:         If True, displays the plot interactively.
    """
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]

    # Group final fitness by evaluator
    groups: dict[str, list[float]] = defaultdict(list)
    for r in results_list:
        evaluator = r["config"]["evaluator"].split("(")[0]
        groups[evaluator].append(r["results"]["best_fitness"])

    labels   = list(groups.keys())
    data     = [groups[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
    ax.set_title(title or "Final Fitness Distribution by Strategy", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


def plot_speedup(
    results_list: list[dict],
    baseline_strategy: str = "SequentialEvaluator",
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Bar chart showing speedup of each strategy relative to the baseline.
    Speedup = baseline_time / strategy_time. Values > 1 are faster than baseline.

    Args:
        results_list:       List of results dictionaries.
        baseline_strategy:  Evaluator name to use as baseline (default: SequentialEvaluator).
        title:              Custom plot title. Auto-generated if None.
        save_path:          If provided, saves the plot to this path.
        show:               If True, displays the plot interactively.
    """
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]

    # Group average total time by evaluator
    groups: dict[str, list[float]] = defaultdict(list)
    for r in results_list:
        evaluator = r["config"]["evaluator"].split("(")[0]
        groups[evaluator].append(r["timing"]["total_s"])

    avg_times = {k: np.mean(v) for k, v in groups.items()}

    if baseline_strategy not in avg_times:
        print(f"Baseline '{baseline_strategy}' not found in results. Available: {list(avg_times.keys())}")
        return

    baseline_time = avg_times[baseline_strategy]
    labels   = list(avg_times.keys())
    speedups = [baseline_time / avg_times[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, speedups, color=colors[:len(labels)], alpha=0.8, edgecolor="black")

    # Add value labels on top of bars
    for bar, speedup in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{speedup:.2f}x",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Baseline reference line
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Baseline (1.0x)")

    ax.set_ylabel("Speedup vs Sequential (higher is faster)", fontsize=12)
    ax.set_title(title or "Speedup Comparison vs Sequential Baseline", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)