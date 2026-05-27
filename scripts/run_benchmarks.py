"""
run_benchmarks.py: runs PSO across all benchmark functions, dimensions,
and strategies. Uses multiple seeds for statistical validity.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --strategies sequential threading multiprocessing async vectorized
    python scripts/run_benchmarks.py --dims 2 10 30
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso.experiments.runner import run_pso
from pso.io.storage import save_results
from pso.parallel.sequential_eval import SequentialEvaluator, BaseEvaluator
from pso.parallel.threading_eval import ThreadingEvaluator
from pso.parallel.multiprocessing_eval import MultiprocessingEvaluator
from pso.parallel.async_eval import AsyncEvaluator
from pso.parallel.vector_eval import VectorizedEvaluator


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


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
    if strategy == "async_simulated":
        return AsyncEvaluator(simulated_latency=0.01)
    if strategy == "vectorized":
        return VectorizedEvaluator()
    raise ValueError(f"Unknown strategy '{strategy}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSO benchmarks.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["sequential", "threading", "multiprocessing", "async", "vectorized"],
        help="Strategies to benchmark.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Benchmark functions to use.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[2, 10, 30],
        help="Dimensions to test.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of seeds per combination.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed. Seeds will be base_seed + i for i in range(n_seeds).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/benchmarks",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["logging"]["level"])
    logger = logging.getLogger(__name__)

    pso_cfg   = cfg["pso"]
    seeds     = [args.base_seed + i for i in range(args.n_seeds)]
    n_workers = cfg["parallel"]["n_workers"]

    total     = len(args.strategies) * len(args.objectives) * len(args.dims) * len(seeds)
    completed = 0

    logger.info(
        f"Starting benchmarks | "
        f"strategies={args.strategies} objectives={args.objectives} "
        f"dims={args.dims} seeds={seeds} total_runs={total}"
    )

    for strategy in args.strategies:
        evaluator = build_evaluator(strategy, n_workers)

        for objective in args.objectives:
            for dim in args.dims:
                for seed in seeds:
                    results = run_pso(
                        objective=objective,
                        dim=dim,
                        n_particles=pso_cfg["n_particles"],
                        max_iter=pso_cfg["max_iter"],
                        w=pso_cfg["w"],
                        c1=pso_cfg["c1"],
                        c2=pso_cfg["c2"],
                        seed=seed,
                        evaluator=evaluator,
                        tolerance=pso_cfg["tolerance"],
                        tolerance_window=pso_cfg["tolerance_window"],
                        stagnation_window=pso_cfg["stagnation_window"],
                    )

                    save_results(
                        results=results,
                        results_dir=f"{args.results_dir}/{strategy}",
                        overwrite=False,
                    )

                    completed += 1
                    print(
                        f"[{completed}/{total}] "
                        f"{strategy} | {objective} | d={dim} | seed={seed} | "
                        f"fitness={results['results']['best_fitness']:.3e} | "
                        f"time={results['timing']['total_s']:.3f}s"
                    )

    logger.info(f"Benchmarks complete. Results saved to {args.results_dir}")
    print(f"\nDone. {completed} runs saved to {args.results_dir}/")


if __name__ == "__main__":
    main()