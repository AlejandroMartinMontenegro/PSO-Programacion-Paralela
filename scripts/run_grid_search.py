"""
run_grid_search.py: entry point for PSO hyperparameter grid search.
Reads grid config from configs/grid_search.yaml.
Objective, dimension and strategy can be selected interactively or via CLI.

Usage:
    python scripts/run_grid_search.py
    python scripts/run_grid_search.py --objective sphere --dim 10
    python scripts/run_grid_search.py --objective sphere --dim 10 --strategies sequential vectorized
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso.experiments.grid_search import run_grid_search
from pso.parallel.sequential_eval import SequentialEvaluator, BaseEvaluator
from pso.parallel.threading_eval import ThreadingEvaluator
from pso.parallel.multiprocessing_eval import MultiprocessingEvaluator
from pso.parallel.async_eval import AsyncEvaluator
from pso.parallel.vector_eval import VectorizedEvaluator


VALID_OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]
VALID_DIMS       = [2, 10, 30]
VALID_STRATEGIES = ["sequential", "threading", "multiprocessing", "async", "vectorized"]


def build_evaluator(strategy: str, n_workers: int) -> BaseEvaluator:
    if strategy == "sequential":
        return SequentialEvaluator()
    if strategy == "threading":
        return ThreadingEvaluator(n_workers=n_workers)
    if strategy == "multiprocessing":
        batch_size = max(1, 30 // n_workers)
        return MultiprocessingEvaluator(n_workers=n_workers, batch_size=batch_size)
    if strategy == "async":
        return AsyncEvaluator(simulated_latency=0.0)
    if strategy == "vectorized":
        return VectorizedEvaluator()
    raise ValueError(f"Unknown strategy '{strategy}'.")


def ask_objective() -> str:
    print("\nSelect benchmark function:")
    for i, name in enumerate(VALID_OBJECTIVES, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input("Enter number (1-4): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(VALID_OBJECTIVES):
            return VALID_OBJECTIVES[int(choice) - 1]
        print("Invalid choice.")


def ask_dim() -> int:
    print("\nSelect dimensions:")
    for i, d in enumerate(VALID_DIMS, 1):
        print(f"  {i}. d={d}")
    while True:
        choice = input("Enter number (1-3): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(VALID_DIMS):
            return VALID_DIMS[int(choice) - 1]
        print("Invalid choice.")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PSO hyperparameter grid search.")
    parser.add_argument("--config",      type=str, default="configs/grid_search.yaml")
    parser.add_argument("--objective",   type=str, default=None)
    parser.add_argument("--dim",         type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results/grid_search")
    parser.add_argument(
        "--strategies", nargs="+",
        default=["sequential", "vectorized"],
        help=f"Strategies to run. Options: {VALID_STRATEGIES}",
    )
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging("INFO")

    objective = args.objective or ask_objective()
    dim       = args.dim       or ask_dim()

    if objective not in VALID_OBJECTIVES:
        print(f"Invalid objective '{objective}'.")
        sys.exit(1)

    print(f"\nRunning grid search | objective={objective} dim={dim} strategies={args.strategies}")

    all_summaries = []

    for strategy in args.strategies:
        print(f"\n--- Strategy: {strategy} ---")
        evaluator = build_evaluator(strategy, args.n_workers)

        summary = run_grid_search(
            objective=objective,
            dim=dim,
            grid=cfg["grid"],
            fixed=cfg["fixed"],
            n_seeds=cfg["experiment"]["n_seeds"],
            base_seed=42,
            evaluator=evaluator,
            results_dir=args.results_dir,
        )
        all_summaries.extend(summary)

    # Print top 5 per strategy
    for strategy in args.strategies:
        evaluator_name = build_evaluator(strategy, args.n_workers).__class__.__name__
        strategy_rows = [r for r in all_summaries if r["evaluator"] == evaluator_name]
        strategy_rows_sorted = sorted(strategy_rows, key=lambda x: x["avg_best_fitness"])

        print(f"\nTop 5 — {evaluator_name}:")
        print(f"{'w':>6} {'c1':>6} {'c2':>6} {'n':>6} {'avg_fitness':>14} {'avg_time':>10}")
        print("-" * 55)
        for row in strategy_rows_sorted[:5]:
            print(
                f"{row['w']:>6} {row['c1']:>6} {row['c2']:>6} "
                f"{int(row['n_particles']):>6} "
                f"{row['avg_best_fitness']:>14.3e} "
                f"{row['avg_time_s']:>10.3f}s"
            )

    print(f"\nResults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()