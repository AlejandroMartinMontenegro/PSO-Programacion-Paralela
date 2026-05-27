"""
Grid search: hyperparameter optimization for PSO.
Runs PSO across combinations of w, c1, c2, and n_particles.
Results are aggregated across multiple seeds and saved to CSV.
"""

import csv
import itertools
import logging
from pathlib import Path
from pso.experiments.runner import run_pso
from pso.io.storage import save_results
from pso.parallel.sequential_eval import SequentialEvaluator

logger = logging.getLogger(__name__)


def run_grid_search(objective: str, dim: int, grid: dict, fixed: dict, n_seeds: int,
    base_seed: int, results_dir: str = "results/grid_search") -> list[dict]:
    
    """
    Runs PSO for all combinations of hyperparameters in the grid.
    Each combination is evaluated across multiple seeds and averaged.

    Args:
        objective: Benchmark function name.
        dim: Search space dimensions.
        grid: Dict with lists of values for w, c1, c2, n_particles.
        fixed: Dict with fixed PSO parameters (max_iter, tolerance, etc).
        n_seeds: Number of seeds per combination.
        base_seed: Base seed. Seeds = [base_seed + i for i in range(n_seeds)].
        results_dir: Directory to save individual run JSONs.

    Returns: List of summary dicts, one per hyperparameter combination.
    """
    seeds = [base_seed + i for i in range(n_seeds)]
    param_names  = ["w", "c1", "c2", "n_particles"]
    param_values = [grid[p] for p in param_names]
    combinations = list(itertools.product(*param_values))
    total = len(combinations) * len(seeds)

    logger.info(f"Grid search | objective={objective} dim={dim} "
                 f"combinations={len(combinations)} seeds={seeds} total_runs={total}")

    output_dir = Path(results_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    summary_rows: list[dict] = []
    completed = 0

    for combo in combinations:
        w, c1, c2, n_particles = combo
        seed_fitnesses: list[float] = []
        seed_times: list[float] = []
        seed_iters: list[int] = []

        for seed in seeds:
            results = run_pso(
                objective=objective,
                dim=dim,
                n_particles=int(n_particles),
                max_iter=fixed["max_iter"],
                w=w, c1=c1, c2=c2,
                seed=seed,
                evaluator=SequentialEvaluator(),
                tolerance=fixed["tolerance"],
                tolerance_window=fixed["tolerance_window"],
                stagnation_window=fixed["stagnation_window"])

            save_results( results=results, results_dir=str(output_dir / "runs"), overwrite=False)

            seed_fitnesses.append(results["results"]["best_fitness"])
            seed_times.append(results["timing"]["total_s"])
            seed_iters.append(results["results"]["n_iterations"])
            completed += 1

        avg_fitness = sum(seed_fitnesses) / len(seed_fitnesses)
        avg_time = sum(seed_times) / len(seed_times)
        avg_iters = sum(seed_iters) / len(seed_iters)

        row = {
            "w": w, "c1": c1, "c2": c2, "n_particles": n_particles,
            "avg_best_fitness": avg_fitness,
            "avg_time_s": avg_time,
            "avg_n_iterations": avg_iters,
            "n_seeds": n_seeds,
            "objective": objective,
            "dim": dim,
        }
        
        summary_rows.append(row)

        logger.info(
            f"[{completed}/{total}] w={w} c1={c1} c2={c2} n={n_particles} | "
            f"avg_fitness={avg_fitness:.3e} avg_time={avg_time:.3f}s"
        )

    # Save CSV summary
    csv_path = output_dir / f"summary_{objective}_d{dim}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info(f"Summary saved to {csv_path}")
    return summary_rows