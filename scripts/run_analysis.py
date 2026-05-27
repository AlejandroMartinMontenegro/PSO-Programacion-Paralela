"""
run_analysis.py: loads benchmark results and generates summary tables.
Compares strategies across functions and dimensions.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --benchmarks_dir results/benchmarks
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso.io.storage import load_all_results


def load_benchmarks(benchmarks_dir: str) -> list[dict]:
    """Loads all results from all strategy subdirectories."""
    all_results = []
    path = Path(benchmarks_dir)
    for strategy_dir in sorted(path.iterdir()):
        if strategy_dir.is_dir():
            results = load_all_results(str(strategy_dir))
            all_results.extend(results)
    return all_results


def summarize_by_strategy_function_dim(results: list[dict]) -> dict:
    """
    Groups results by (evaluator, objective, dim) and computes:
    - mean and std of best fitness across seeds
    - mean total time
    - convergence rate (% of runs that converged before max_iter)
    """
    groups = defaultdict(list)
    for r in results:
        evaluator = r["config"]["evaluator"].split("(")[0]
        objective = r["config"]["objective"]
        dim       = r["config"]["dim"]
        key       = (evaluator, objective, dim)
        groups[key].append(r)

    summary = {}
    for key, group in groups.items():
        fitnesses  = [r["results"]["best_fitness"] for r in group]
        times      = [r["timing"]["total_s"] for r in group]
        converged  = [r["results"]["stop_reason"] != "max_iterations" for r in group]

        n = len(fitnesses)
        mean_fitness = sum(fitnesses) / n
        std_fitness  = (sum((f - mean_fitness) ** 2 for f in fitnesses) / n) ** 0.5
        mean_time    = sum(times) / n
        conv_rate    = sum(converged) / n * 100

        summary[key] = {
            "evaluator":    key[0],
            "objective":    key[1],
            "dim":          key[2],
            "n_seeds":      n,
            "mean_fitness": mean_fitness,
            "std_fitness":  std_fitness,
            "mean_time_s":  mean_time,
            "conv_rate_pct": conv_rate,
        }

    return summary


def compute_speedup(summary: dict) -> dict:
    """Computes speedup of each strategy vs SequentialEvaluator per (objective, dim)."""
    # Find baseline times
    baseline = {}
    for key, row in summary.items():
        evaluator, objective, dim = key
        if evaluator == "SequentialEvaluator":
            baseline[(objective, dim)] = row["mean_time_s"]

    speedups = {}
    for key, row in summary.items():
        evaluator, objective, dim = key
        base_time = baseline.get((objective, dim))
        if base_time and base_time > 0:
            speedups[key] = base_time / row["mean_time_s"]
        else:
            speedups[key] = None

    return speedups


def print_table(summary: dict, speedups: dict) -> None:
    """Prints a formatted summary table to stdout."""
    # Header
    print("\n" + "=" * 100)
    print("PSO BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Evaluator':<25} {'Function':<12} {'d':>4} {'Seeds':>6} "
          f"{'Mean Fitness':>14} {'Std Fitness':>13} {'Mean Time':>10} "
          f"{'Conv%':>7} {'Speedup':>8}")
    print("-" * 100)

    # Group by objective and dim for readability
    prev_obj_dim = None
    for key in sorted(summary.keys(), key=lambda k: (k[1], k[2], k[0])):
        row     = summary[key]
        speedup = speedups.get(key)
        obj_dim = (key[1], key[2])

        if obj_dim != prev_obj_dim:
            if prev_obj_dim is not None:
                print()
            prev_obj_dim = obj_dim

        speedup_str = f"{speedup:.2f}x" if speedup else "—"

        print(
            f"{row['evaluator']:<25} {row['objective']:<12} {row['dim']:>4} "
            f"{row['n_seeds']:>6} {row['mean_fitness']:>14.3e} "
            f"{row['std_fitness']:>13.3e} {row['mean_time_s']:>10.3f}s "
            f"{row['conv_rate_pct']:>6.0f}% {speedup_str:>8}"
        )

    print("=" * 100)


def save_csv(summary: dict, speedups: dict, output_path: str) -> None:
    """Saves summary table as CSV."""
    rows = []
    for key, row in sorted(summary.items(), key=lambda k: (k[0][1], k[0][2], k[0][0])):
        speedup = speedups.get(key)
        rows.append({
            "evaluator":      row["evaluator"],
            "objective":      row["objective"],
            "dim":            row["dim"],
            "n_seeds":        row["n_seeds"],
            "mean_fitness":   row["mean_fitness"],
            "std_fitness":    row["std_fitness"],
            "mean_time_s":    row["mean_time_s"],
            "conv_rate_pct":  row["conv_rate_pct"],
            "speedup_vs_seq": round(speedup, 4) if speedup else "",
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV summary saved to {output_path}")


def print_top_performers(summary: dict) -> None:
    """Prints best strategy per (objective, dim) by mean fitness."""
    print("\n" + "=" * 60)
    print("BEST STRATEGY PER FUNCTION AND DIMENSION (by mean fitness)")
    print("=" * 60)

    obj_dims = sorted(set((k[1], k[2]) for k in summary.keys()))
    for obj, dim in obj_dims:
        candidates = {
            k: v for k, v in summary.items()
            if k[1] == obj and k[2] == dim
        }
        best_key = min(candidates, key=lambda k: candidates[k]["mean_fitness"])
        best     = candidates[best_key]
        print(
            f"  {obj:<12} d={dim:<4} → {best['evaluator']:<25} "
            f"fitness={best['mean_fitness']:.3e}"
        )

    print("=" * 60)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyse PSO benchmark results.")
    parser.add_argument(
        "--benchmarks_dir", type=str, default="results/benchmarks",
        help="Directory with benchmark results.",
    )
    parser.add_argument(
        "--output_csv", type=str, default="results/analysis_summary.csv",
        help="Path to save CSV summary.",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.benchmarks_dir}...")
    results = load_benchmarks(args.benchmarks_dir)

    if not results:
        print("No results found. Run scripts/run_benchmarks.py first.")
        return

    print(f"Loaded {len(results)} runs.")

    summary  = summarize_by_strategy_function_dim(results)
    speedups = compute_speedup(summary)

    print_table(summary, speedups)
    print_top_performers(summary)
    save_csv(summary, speedups, args.output_csv)


if __name__ == "__main__":
    main()