"""
Runner: orchestrates a full PSO run from the start to the end.
Creates the swarm, runs the main loop, tracks the timing and convergence, 
and returns a results ready to be saved or analysed.
"""

import os
import subprocess
import time
import logging
import platform
import numpy as np
from pso.core.swarm import Swarm
from pso.core.equations import update_velocity, update_position
from pso.core.stopping_criteria import StoppingCriteria
from pso.parallel.sequential_eval import BaseEvaluator
from pso.objectives.functions import FUNCTIONS
from pso.parallel.vector_eval import VectorizedEvaluator, vectorized_update

logger = logging.getLogger(__name__)

def run_pso( objective: str, dim: int, n_particles: int, max_iter: int, w: float, c1: float, c2: float,
    seed: int, evaluator: BaseEvaluator, tolerance: float = 1e-6, tolerance_window: int = 30,
    stagnation_window: int = 50, save_trajectory: bool = False) -> dict:
    
    """
    Runs a complete PSO optimization and returns all results.

    Args:
        objective: Name of the benchmark function (e.g. "sphere").
        dim: Number of dimensions of the search space.
        n_particles: Number of particles in the swarm.
        max_iter: Maximum number of iterations.
        w: Inertia weight.
        c1: Individual memory coefficient.
        c2: Social influence coefficient.
        seed: Random seed for reproducibility.
        evaluator: Evaluator instance (V0, V1, V2, V3, V4).
        tolerance: Minimum fitness improvement to avoid tolerance stop.
        tolerance_window: Iterations window for tolerance check.
        stagnation_window: Iterations window for stagnation check.

    Returns: Dictionary with metadata, config, results, timing, and convergence curve.
    """

    fn_entry = FUNCTIONS[objective]
    fitness_fn = fn_entry["fn"]
    bounds = fn_entry["bounds"]

    # If V4, inject the vectorized fitness function
    if isinstance(evaluator, VectorizedEvaluator):
        fn_vec = fn_entry.get("fn_vec")
        if fn_vec is not None:
            evaluator.fitness_fn_vec = fn_vec

    swarm = Swarm(n_particles = n_particles, dim = dim, bounds = bounds, seed = seed)
    stopper = StoppingCriteria(
        max_iter = max_iter,
        tolerance = tolerance,
        tolerance_window = tolerance_window,
        stagnation_window = stagnation_window,
    )

    total_eval_time: float = 0.0
    total_update_time: float = 0.0
    fitness_history: list[float] = []
    convergence_curve: list[dict] = []
    stop_reason: str = "running"
    total_start = time.perf_counter()
    trajectory: list[list[list[float]]]  = []
    global_bests_traj: list[list[float]] = []

    logger.info(
        f"Starting PSO | objective = {objective} dim = {dim} "
        f"n_particles = {n_particles} seed = {seed} "
        f"evaluator = {evaluator!r}"
    )

    for iteration in range(max_iter):

        fitness_values, eval_time = evaluator.evaluate(swarm.particles, fitness_fn)
        total_eval_time += eval_time

        for particle, fitness in zip(swarm.particles, fitness_values):
            particle.update_personal_best(fitness)

        swarm.update_global_best()

        update_start = time.perf_counter()

        if isinstance(evaluator, VectorizedEvaluator):
            positions = np.array([p.position for p in swarm.particles])
            velocities = np.array([p.velocity for p in swarm.particles])
            personal_bests = np.array([p.best_position for p in swarm.particles])

            new_positions, new_velocities = vectorized_update(
                positions=positions,
                velocities=velocities,
                personal_bests=personal_bests,
                global_best=swarm.best_position,
                w=w, c1=c1, c2=c2,
                rng=swarm.rng,
                bounds=bounds,
            )

            for i, particle in enumerate(swarm.particles):
                particle.position = new_positions[i]
                particle.velocity = new_velocities[i]
        else:
            for particle in swarm.particles:
                r1 = swarm.rng.random(dim)
                r2 = swarm.rng.random(dim)

                new_velocity = update_velocity(
                    velocity=particle.velocity,
                    position=particle.position,
                    personal_best=particle.best_position,
                    global_best=swarm.best_position,
                    w=w, c1=c1, c2=c2,
                    r1=r1, r2=r2,
                )
                new_position, new_velocity = update_position(
                    position=particle.position,
                    velocity=new_velocity,
                    bounds=bounds,
                )
                particle.velocity = new_velocity
                particle.position = new_position

        total_update_time += time.perf_counter() - update_start

        fitness_history.append(swarm.best_fitness)
        convergence_curve.append({
            "iter": iteration,
            "best_fitness": swarm.best_fitness,
        })
        
        if save_trajectory and dim == 2:
            trajectory.append([p.position.tolist() for p in swarm.particles])
            global_bests_traj.append(swarm.best_position.tolist())
        
        logger.debug(
            f"iter={iteration:4d} | best_fitness={swarm.best_fitness:.6e}"
        )

        should_stop, stop_reason = stopper.should_stop(iteration, fitness_history)
        if should_stop:
            logger.info(f"Stopping at iter={iteration} | reason={stop_reason}")
            break

    total_time = time.perf_counter() - total_start

    logger.info(
        f"Finished | best_fitness={swarm.best_fitness:.6e} "
        f"iters = {len(fitness_history)} reason = {stop_reason} "
        f"total_time = {total_time:.3f}s"
    )

    # Git commit hash (best effort — fails gracefully if no git available)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    return {
        "metadata": {
            "seed": seed,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "git_hash": git_hash,
        },
        
        "config": {"objective": objective, "dim": dim, "n_particles": n_particles, "max_iter": max_iter,
            "w": w, "c1": c1, "c2": c2, "evaluator": repr(evaluator), "tolerance": tolerance,
            "stagnation_window": stagnation_window},
        
        "results": {"best_fitness": swarm.best_fitness, "best_position": swarm.best_position.tolist(),
            "converged": stop_reason in ("tolerance", "stagnation"), "n_iterations": len(fitness_history),
            "stop_reason": stop_reason, "trajectory": trajectory if save_trajectory else [],
            "global_bests_traj": global_bests_traj if save_trajectory else []},
        
        "timing": {"total_s": round(total_time, 6), "eval_fitness_s": round(total_eval_time, 6),
            "update_particles_s": round(total_update_time, 6), 
            "overhead_s": round(total_time - total_eval_time - total_update_time, 6),
        },
        
        "convergence_curve": convergence_curve,
    }