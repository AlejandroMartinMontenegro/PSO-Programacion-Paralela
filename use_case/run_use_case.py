"""
run_use_case.py: Portfolio optimization using PSO with all parallel strategies.
Compares V0-V4 on a real financial optimization problem.

Usage:
    python use_case/run_use_case.py
"""

import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pso.core.swarm import Swarm
from pso.core.equations import update_velocity, update_position
from pso.core.stopping_criteria import StoppingCriteria
from pso.parallel.sequential_eval import SequentialEvaluator
from pso.parallel.threading_eval import ThreadingEvaluator
from pso.parallel.async_eval import AsyncEvaluator
from pso.parallel.vector_eval import VectorizedEvaluator, vectorized_update
from use_case.portfolio_optimization import (
    load_returns, make_fitness_fn, fitness_fn_mp, _worker_init, TICKERS,
)

# ── Configuration ─────────────────────────────────────────────────────────────

PSO_CONFIG = dict(
    n_particles      = 30,
    max_iter         = 200,
    w                = 0.7,
    c1               = 1.5,
    c2               = 1.5,
    seed             = 42,
    tolerance        = 1e-6,
    tolerance_window = 30,
    stagnation_window= 50,
)

N_WORKERS  = 4
BATCH_SIZE = 8

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Multiprocessing evaluator with initializer ────────────────────────────────

def _eval_batch_mp(args):
    indices, positions = args
    return [(i, fitness_fn_mp(pos)) for i, pos in zip(indices, positions)]


def evaluate_multiprocessing(particles, mean_returns, cov_matrix):
    """V2 evaluator with initializer — passes data explicitly to workers."""
    start = time.perf_counter()
    n = len(particles)
    n_batches = math.ceil(n / BATCH_SIZE)
    batches = []
    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, n)
        idxs = list(range(s, e))
        poss = [particles[i].position for i in idxs]
        batches.append((idxs, poss))

    fitness_values = [0.0] * n
    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_worker_init,
        initargs=(mean_returns, cov_matrix),
    ) as executor:
        futures = {executor.submit(_eval_batch_mp, b): b for b in batches}
        for future in as_completed(futures):
            for i, v in future.result():
                fitness_values[i] = v

    return fitness_values, time.perf_counter() - start


# ── PSO runner ────────────────────────────────────────────────────────────────

def run_portfolio_pso(fitness_fn, evaluator, config, mean_returns=None, cov_matrix=None):
    dim    = len(TICKERS)
    bounds = (0.0, 1.0)

    swarm   = Swarm(n_particles=config["n_particles"], dim=dim, bounds=bounds, seed=config["seed"])
    stopper = StoppingCriteria(
        max_iter         = config["max_iter"],
        tolerance        = config["tolerance"],
        tolerance_window = config["tolerance_window"],
        stagnation_window= config["stagnation_window"],
    )

    total_eval_time = 0.0; total_update_time = 0.0
    fitness_history = []; convergence_curve = []
    stop_reason = "running"
    total_start = time.perf_counter()

    for iteration in range(config["max_iter"]):

        # V2 uses its own evaluator with initializer
        if evaluator == "multiprocessing":
            fitness_values, eval_time = evaluate_multiprocessing(
                swarm.particles, mean_returns, cov_matrix
            )
        else:
            fitness_values, eval_time = evaluator.evaluate(swarm.particles, fitness_fn)
        total_eval_time += eval_time

        for particle, fitness in zip(swarm.particles, fitness_values):
            particle.update_personal_best(fitness)
        swarm.update_global_best()

        update_start = time.perf_counter()
        if isinstance(evaluator, VectorizedEvaluator):
            positions      = np.array([p.position for p in swarm.particles])
            velocities     = np.array([p.velocity for p in swarm.particles])
            personal_bests = np.array([p.best_position for p in swarm.particles])
            new_positions, new_velocities = vectorized_update(
                positions, velocities, personal_bests, swarm.best_position,
                config["w"], config["c1"], config["c2"], swarm.rng, bounds,
            )
            for i, particle in enumerate(swarm.particles):
                particle.position = new_positions[i]
                particle.velocity = new_velocities[i]
        else:
            for particle in swarm.particles:
                r1 = swarm.rng.random(dim); r2 = swarm.rng.random(dim)
                nv = update_velocity(
                    particle.velocity, particle.position,
                    particle.best_position, swarm.best_position,
                    config["w"], config["c1"], config["c2"], r1, r2,
                )
                np_, nv = update_position(particle.position, nv, bounds)
                particle.velocity = nv; particle.position = np_
        total_update_time += time.perf_counter() - update_start

        fitness_history.append(swarm.best_fitness)
        convergence_curve.append({"iter": iteration, "best_fitness": swarm.best_fitness})

        should_stop, stop_reason = stopper.should_stop(iteration, fitness_history)
        if should_stop:
            break

    total_time = time.perf_counter() - total_start
    best_w = np.abs(swarm.best_position); best_w = best_w / best_w.sum()

    return {
        "stop_reason":      stop_reason,
        "n_iterations":     len(fitness_history),
        "best_sharpe":      round(-swarm.best_fitness, 6),
        "best_weights":     best_w.tolist(),
        "total_s":          round(total_time, 4),
        "eval_s":           round(total_eval_time, 4),
        "update_s":         round(total_update_time, 4),
        "convergence_curve": convergence_curve,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_returns  = load_returns()
    fitness_fn   = make_fitness_fn(log_returns)
    mean_returns = log_returns.mean(axis=0)
    cov_matrix   = np.cov(log_returns.T)

    strategies = [
        ("V0 Sequential",      SequentialEvaluator()),
        ("V1 Threading",       ThreadingEvaluator(n_workers=N_WORKERS)),
        ("V2 Multiprocessing", "multiprocessing"),
        ("V3 Async",           AsyncEvaluator(simulated_latency=0.0)),
        ("V4 Vectorized",      VectorizedEvaluator()),
    ]

    print("\n" + "=" * 70)
    print("PORTFOLIO OPTIMIZATION — PSO STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Assets:     {TICKERS}")
    print(f"Dimensions: {len(TICKERS)}")
    print(f"Particles:  {PSO_CONFIG['n_particles']}  |  Max iter: {PSO_CONFIG['max_iter']}")
    print("=" * 70)

    all_results = {}

    for name, evaluator in strategies:
        print(f"\nRunning {name}...")
        result = run_portfolio_pso(
            fitness_fn, evaluator, PSO_CONFIG, mean_returns, cov_matrix
        )
        all_results[name] = result
        print(f"  Sharpe ratio:  {result['best_sharpe']:.4f}")
        print(f"  Iterations:    {result['n_iterations']}  ({result['stop_reason']})")
        print(f"  Total time:    {result['total_s']:.4f}s")
        print(f"  Eval time:     {result['eval_s']:.4f}s")

        out_path = RESULTS_DIR / f"result_{name.replace(' ', '_').lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<22} {'Sharpe':>8} {'Time':>10} {'Speedup':>9} {'Iters':>7}")
    print("-" * 70)

    baseline_time = all_results["V0 Sequential"]["total_s"]
    for name, result in all_results.items():
        speedup = baseline_time / result["total_s"] if result["total_s"] > 0 else 0
        print(
            f"{name:<22} {result['best_sharpe']:>8.4f} "
            f"{result['total_s']:>9.4f}s "
            f"{speedup:>8.2f}x "
            f"{result['n_iterations']:>7}"
        )

    print("\n" + "=" * 70)
    print("BEST PORTFOLIO (V0 Sequential)")
    print("=" * 70)
    for ticker, weight in zip(TICKERS, all_results["V0 Sequential"]["best_weights"]):
        bar = "█" * int(weight * 40)
        print(f"  {ticker:<6} {weight:>6.1%}  {bar}")


if __name__ == "__main__":
    main()