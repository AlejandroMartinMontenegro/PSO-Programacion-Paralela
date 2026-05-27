"""
V4 - Vectorized evaluator (implicit parallelism via NumPy).
Evaluates all particles at once using matrix operations instead of Python loops.
NumPy internally uses BLAS/LAPACK and SIMD instructions for fast computation.

For simple math functions (Sphere, Rastrigin) this is typically the fastest strategy.
No explicit parallelism — speed comes from NumPy's optimized C/Fortran kernels.

Also provides vectorized velocity and position updates for the full swarm,
replacing the per-particle loop in runner.py.

All speedup measurements are relative to SequentialEvaluator (V0).
"""

import time
import numpy as np

from pso.core.particle import Particle
from pso.parallel.sequential_eval import BaseEvaluator


class VectorizedEvaluator(BaseEvaluator):
    """
    V4 - Vectorized evaluator (implicit parallelism via NumPy).
    Evaluates all particles at once by passing the full position matrix (n, d)
    to a vectorized fitness function that returns a fitness vector (n,) in one call.
    NumPy internally uses BLAS/LAPACK and SIMD instructions for fast computation.

    For simple math functions (Sphere, Rastrigin) this is typically the fastest strategy.
    No explicit parallelism — speed comes from NumPy's optimized C/Fortran kernels.

    Also provides vectorized velocity and position updates for the full swarm,
    replacing the per-particle loop in runner.py.

    All speedup measurements are relative to SequentialEvaluator (V0).
    """

    def __init__(self, fitness_fn_vec=None) -> None:
        """
        Args:
            fitness_fn_vec: Vectorized fitness function. If None, will be set
                            by the runner from FUNCTIONS[objective]["fn_vec"].
        """
        self.fitness_fn_vec: callable | None = fitness_fn_vec

    def evaluate(
        self,
        particles: list[Particle],
        fitness_fn: callable,
    ) -> tuple[list[float], float]:
        start = time.perf_counter()
        positions = np.array([p.position for p in particles])

        if self.fitness_fn_vec is not None:
            fitness_values = self.fitness_fn_vec(positions)
        else:
            # Fallback to per-row evaluation
            fitness_values = np.array([fitness_fn(p) for p in positions])

        elapsed = time.perf_counter() - start
        return list(fitness_values.astype(float)), elapsed

    def __repr__(self) -> str:
        return "VectorizedEvaluator(strategy=V4_vectorized)"


# ── Vectorized PSO update equations ──────────────────────────────────────────

def vectorized_update(
    positions: np.ndarray,
    velocities: np.ndarray,
    personal_bests: np.ndarray,
    global_best: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Updates all particle velocities and positions in a single vectorized step.
    Replaces the per-particle loop in runner.py for V4.

    Args:
        positions:      Current positions. Shape: (n_particles, d)
        velocities:     Current velocities. Shape: (n_particles, d)
        personal_bests: Personal best positions. Shape: (n_particles, d)
        global_best:    Global best position. Shape: (d,)
        w:              Inertia weight.
        c1:             Cognitive coefficient.
        c2:             Social coefficient.
        rng:            NumPy random generator for reproducibility.
        bounds:         (min, max) search space limits.

    Returns: Tuple of (new_positions, new_velocities). Both Shape: (n_particles, d)
    """
    n_particles, d = positions.shape

    # Generate r1 and r2 for all particles at once — shape (n_particles, d)
    r1 = rng.random((n_particles, d))
    r2 = rng.random((n_particles, d))

    # Velocity update — full matrix operation
    inertia           = w * velocities
    individual_memory = c1 * r1 * (personal_bests - positions)
    social_influence  = c2 * r2 * (global_best - positions)
    new_velocities    = inertia + individual_memory + social_influence

    # Position update
    new_positions = positions + new_velocities

    low, high = bounds

    # Clamp — find violations
    below = new_positions < low
    above = new_positions > high

    # Clip positions to bounds
    new_positions = np.clip(new_positions, low, high)

    # Zero velocity on violated dimensions
    new_velocities[below | above] = 0.0

    return new_positions, new_velocities