"""
run_grid_search.py: entry point for PSO hyperparameter grid search.
Reads grid config from configs/grid_search.yaml.
Objective and dimension can be selected interactively or via CLI.

Usage:
    python scripts/run_grid_search.py
    python scripts/run_grid_search.py --objective sphere --dim 10
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso.experiments.grid_search import run_grid_search


VALID_OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]
VALID_DIMS       = [2, 10, 30]


def ask_objective() -> str:
    """Interactively asks the user to select a benchmark function."""
    print("\nSelect benchmark function:")
    for i, name in enumerate(VALID_OBJECTIVES, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input("Enter number (1-4): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(VALID_OBJECTIVES):
            return VALID_OBJECTIVES[int(choice) - 1]
        print("Invalid choice. Enter a number between 1 and 4.")


def ask_dim() -> int:
    """Interactively asks the user to select the number of dimensions."""
    print("\nSelect dimensions:")
    for i, d in enumerate(VALID_DIMS, 1):
        print(f"  {i}. d={d}")
    while True:
        choice = input("Enter number (1-3): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(VALID_DIMS):
            return VALID_DIMS[int(choice) - 1]
        print("Invalid choice. Enter a number between 1 and 3.")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PSO hyperparameter grid search.")
    parser.add_argument("--config", type=str, default="configs/grid_search.yaml")
    parser.add_argument(
        "--objective", type=str, default=None,
        help="Benchmark function. If not provided, will ask interactively.",
    )
    parser.add_argument(
        "--dim", type=int, default=None,
        help="Dimensions. If not provided, will ask interactively.",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/grid_search",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging("INFO")

    # Interactive selection if not provided via CLI
    objective = args.objective or ask_objective()
    dim       = args.dim       or ask_dim()

    if objective not in VALID_OBJECTIVES:
        print(f"Invalid objective '{objective}'. Choose from {VALID_OBJECTIVES}")
        sys.exit(1)

    print(f"\nRunning grid search | objective={objective} dim={dim}")

    summary = run_grid_search(
        objective=objective,
        dim=dim,
        grid=cfg["grid"],
        fixed=cfg["fixed"],
        n_seeds=cfg["experiment"]["n_seeds"],
        base_seed=42,
        results_dir=args.results_dir,
    )

    # Print top 5 combinations by avg fitness
    summary_sorted = sorted(summary, key=lambda x: x["avg_best_fitness"])
    print("\nTop 5 hyperparameter combinations:")
    print(f"{'w':>6} {'c1':>6} {'c2':>6} {'n':>6} {'avg_fitness':>14} {'avg_time':>10}")
    print("-" * 55)
    for row in summary_sorted[:5]:
        print(
            f"{row['w']:>6} {row['c1']:>6} {row['c2']:>6} "
            f"{int(row['n_particles']):>6} "
            f"{row['avg_best_fitness']:>14.3e} "
            f"{row['avg_time_s']:>10.3f}s"
        )

    print(f"\nFull summary saved to results/grid_search/summary_{objective}_d{dim}.csv")


if __name__ == "__main__":
    main()