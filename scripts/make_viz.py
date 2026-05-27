"""
make_viz.py: generates all PSO visualizations.
Convergence comparison, averaged curves, boxplots, speedup chart,
and swarm animations in 2D and 3D.

Usage:
    python scripts/make_viz.py
    python scripts/make_viz.py --save
    python scripts/make_viz.py --animate --objective sphere
    python scripts/make_viz.py --benchmarks_dir results/benchmarks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso.io.storage import load_all_results
from pso.viz.convergence import (
    plot_comparison,
    plot_averaged_convergence,
    plot_boxplots,
    plot_speedup,
)
from pso.viz.graph import plot_swarm_animation_2d, plot_swarm_animation_3d
from pso.experiments.runner import run_pso
from pso.parallel.sequential_eval import SequentialEvaluator

VALID_OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PSO visualizations.")
    parser.add_argument("--results_dir",    type=str, default="results",
                        help="Directory with single-run JSON results.")
    parser.add_argument("--benchmarks_dir", type=str, default=None,
                        help="Directory with benchmark results for averaged plots.")
    parser.add_argument("--save",           action="store_true",
                        help="Save all plots to results/plots/.")
    parser.add_argument("--animate",        action="store_true",
                        help="Generate swarm animations in 2D and 3D.")
    parser.add_argument("--objective",      type=str, default="sphere",
                        help="Objective for animation.")
    args = parser.parse_args()

    save_dir = None
    if args.save:
        save_dir = Path(args.results_dir) / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

    # Single run convergence comparison
    results_list = load_all_results(args.results_dir)

    if results_list:
        print(f"\nFound {len(results_list)} result files in {args.results_dir}:")
        for r in results_list:
            evaluator  = r["config"]["evaluator"].split("(")[0]
            objective  = r["config"]["objective"]
            dim        = r["config"]["dim"]
            total_time = r["timing"]["total_s"]
            fitness    = r["results"]["best_fitness"]
            print(f"  {evaluator} | {objective} | d={dim} | {total_time:.3f}s | fitness={fitness:.3e}")

        print("\nPlotting convergence comparison...")
        plot_comparison(
            results_list,
            save_path=save_dir / "convergence_comparison.png" if save_dir else None,
            show=not args.save,
        )

        print("Plotting speedup chart...")
        plot_speedup(
            results_list,
            save_path=save_dir / "speedup.png" if save_dir else None,
            show=not args.save,
        )

        print("Plotting boxplots...")
        plot_boxplots(
            results_list,
            save_path=save_dir / "boxplots.png" if save_dir else None,
            show=not args.save,
        )

    # Averaged convergence from benchmarks directory
    if args.benchmarks_dir:
        all_benchmark_results = []
        benchmarks_path = Path(args.benchmarks_dir)

        for strategy_dir in benchmarks_path.iterdir():
            if strategy_dir.is_dir():
                strategy_results = load_all_results(str(strategy_dir))
                all_benchmark_results.extend(strategy_results)

        if all_benchmark_results:
            print(f"\nFound {len(all_benchmark_results)} benchmark result files.")
            print("Plotting averaged convergence curves...")
            plot_averaged_convergence(
                all_benchmark_results,
                save_path=save_dir / "averaged_convergence.png" if save_dir else None,
                show=not args.save,
            )
            print("Plotting benchmark boxplots...")
            plot_boxplots(
                all_benchmark_results,
                title="Final Fitness Distribution by Strategy (benchmarks)",
                save_path=save_dir / "benchmark_boxplots.png" if save_dir else None,
                show=not args.save,
            )
            print("Plotting benchmark speedup...")
            plot_speedup(
                all_benchmark_results,
                save_path=save_dir / "benchmark_speedup.png" if save_dir else None,
                show=not args.save,
            )

    # Swarm animations
    if args.animate:
        if args.objective not in VALID_OBJECTIVES:
            print(f"Invalid objective. Choose from {VALID_OBJECTIVES}")
            return

        print(f"\nGenerating swarm animations for {args.objective} d=2...")

        results = run_pso(
            objective=args.objective,
            dim=2,
            n_particles=30,
            max_iter=80,
            w=0.7, c1=1.5, c2=1.5,
            seed=42,
            evaluator=SequentialEvaluator(),
            tolerance=1e-6,
            tolerance_window=20,
            stagnation_window=30,
            save_trajectory=True,
        )

        trajectory   = results["results"]["trajectory"]
        global_bests = results["results"]["global_bests_traj"]

        if not trajectory:
            print("No trajectory data.")
            return

        print("  Generating 2D animation...")
        plot_swarm_animation_2d(
            objective=args.objective,
            trajectory=trajectory,
            global_bests=global_bests,
            save_path=save_dir / f"animation_2d_{args.objective}.gif" if save_dir else None,
            show=not args.save,
        )

        print("  Generating 3D animation...")
        plot_swarm_animation_3d(
            objective=args.objective,
            trajectory=trajectory,
            global_bests=global_bests,
            save_path=save_dir / f"animation_3d_{args.objective}.gif" if save_dir else None,
            show=not args.save,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()