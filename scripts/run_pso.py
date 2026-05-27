"""
run_pso.py: entry point for a single PSO run.
Reads config from configs/default.yaml. CLI arguments override YAML values.
If --interactive is passed, asks for objective, dim and strategy from a menu.

Usage:
    python scripts/run_pso.py --strategy sequential --objective sphere --dim 10
    python scripts/run_pso.py --interactive
    python scripts/run_pso.py --strategy threading --dim 30 --objective rastrigin
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

VALID_OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]
VALID_STRATEGIES = [
    "sequential",
    "threading",
    "multiprocessing",
    "multiprocessing_batched",
    "async",
    "async_simulated",
    "vectorized",
]
VALID_DIMS = [2, 10, 30]


def setup_logging(level: str, log_to_file: bool, log_dir: str) -> None:
    """
    Configures the root logger for the entire run.

    Args:
        level:       Logging level string (DEBUG, INFO, WARNING).
        log_to_file: If True, also writes logs to a file in log_dir.
        log_dir:     Directory for log files.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / "run.log"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def build_evaluator(strategy: str, n_workers: int) -> BaseEvaluator:
    """
    Creates the correct evaluator based on the strategy name.

    Args:
        strategy:  Evaluator strategy name.
        n_workers: Number of workers for parallel strategies.

    Returns: Evaluator instance.
    """
    if strategy == "sequential":
        return SequentialEvaluator()
    if strategy == "threading":
        return ThreadingEvaluator(n_workers=n_workers)
    if strategy == "multiprocessing":
        return MultiprocessingEvaluator(n_workers=n_workers, batch_size=None)
    if strategy == "multiprocessing_batched":
        batch_size = max(1, 30 // n_workers)
        return MultiprocessingEvaluator(n_workers=n_workers, batch_size=batch_size)
    if strategy == "async":
        return AsyncEvaluator(simulated_latency=0.0)
    if strategy == "async_simulated":
        return AsyncEvaluator(simulated_latency=0.01)
    if strategy == "vectorized":
        return VectorizedEvaluator()

    raise ValueError(
        f"Unknown strategy '{strategy}'. Available: {VALID_STRATEGIES}"
    )


def ask_choice(prompt: str, options: list) -> int:
    """
    Generic interactive selection. Returns the selected index.

    Args:
        prompt:  Text shown before the options.
        options: List of option labels to display.

    Returns: Index of selected option (0-based).
    """
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input(f"Enter number (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        print(f"Invalid choice. Enter a number between 1 and {len(options)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single PSO optimization.")
    parser.add_argument("--config",      type=str, default="configs/default.yaml")
    parser.add_argument("--strategy",    type=str, default=None,
                        help=f"Evaluator strategy. Options: {VALID_STRATEGIES}")
    parser.add_argument("--dim",         type=int, default=None,
                        help=f"Search space dimensions. Options: {VALID_DIMS}")
    parser.add_argument("--objective",   type=str, default=None,
                        help=f"Benchmark function. Options: {VALID_OBJECTIVES}")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite existing results file.")
    parser.add_argument("--interactive", action="store_true",
                        help="Ask for objective, dim and strategy interactively.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Interactive mode — ask only for values not provided via CLI
    if args.interactive:
        if not args.objective:
            idx = ask_choice("Select benchmark function:", VALID_OBJECTIVES)
            args.objective = VALID_OBJECTIVES[idx]
        if not args.dim:
            idx = ask_choice("Select dimensions:", [f"d={d}" for d in VALID_DIMS])
            args.dim = VALID_DIMS[idx]
        if not args.strategy:
            idx = ask_choice("Select strategy:", VALID_STRATEGIES)
            args.strategy = VALID_STRATEGIES[idx]

    # CLI overrides YAML
    if args.strategy:
        cfg["parallel"]["strategy"] = args.strategy
    if args.dim:
        cfg["objective"]["dim"] = args.dim
    if args.objective:
        cfg["objective"]["name"] = args.objective

    # Setup logging
    log_cfg = cfg["logging"]
    setup_logging(
        level=log_cfg["level"],
        log_to_file=log_cfg["log_to_file"],
        log_dir=log_cfg["log_dir"],
    )

    # Build evaluator
    par_cfg   = cfg["parallel"]
    evaluator = build_evaluator(
        strategy=par_cfg["strategy"],
        n_workers=par_cfg["n_workers"],
    )

    # Run PSO
    pso_cfg = cfg["pso"]
    results = run_pso(
        objective=cfg["objective"]["name"],
        dim=cfg["objective"]["dim"],
        n_particles=pso_cfg["n_particles"],
        max_iter=pso_cfg["max_iter"],
        w=pso_cfg["w"],
        c1=pso_cfg["c1"],
        c2=pso_cfg["c2"],
        seed=cfg["reproducibility"]["seed"],
        evaluator=evaluator,
        tolerance=pso_cfg["tolerance"],
        tolerance_window=pso_cfg["tolerance_window"],
        stagnation_window=pso_cfg["stagnation_window"],
    )

    # Save results
    overwrite = args.overwrite or cfg["output"]["overwrite"]
    filepath  = save_results(
        results=results,
        results_dir=cfg["output"]["results_dir"],
        overwrite=overwrite,
    )

    print(f"\nResults saved to: {filepath}")
    print(f"Best fitness:     {results['results']['best_fitness']:.6e}")
    print(f"Iterations:       {results['results']['n_iterations']}")
    print(f"Stop reason:      {results['results']['stop_reason']}")
    print(f"Total time:       {results['timing']['total_s']:.3f}s")
    print(f"Eval time:        {results['timing']['eval_fitness_s']:.3f}s")
    print(f"Update time:      {results['timing']['update_particles_s']:.3f}s")


if __name__ == "__main__":
    main()