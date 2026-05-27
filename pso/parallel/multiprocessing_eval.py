"""
V2 - Multiprocessing evaluator.
Uses ProcessPoolExecutor to evaluate particles in true parallel across CPU cores.
Each process has its own Python interpreter and GIL — real CPU parallelism.
Overhead comes from IPC (inter-process communication) and pickling of data.
On Windows, processes are spawned (not forked) which adds significant overhead.

Two modes:
  batch_size=None: one task per particle (maximum granularity, maximum IPC overhead)
  batch_size=N:    N particles per task (reduces IPC overhead, less granularity)

All speedup measurements are relative to SequentialEvaluator (V0).
"""

import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from pso.core.particle import Particle
from pso.parallel.sequential_eval import BaseEvaluator


def _evaluate_single(args: tuple) -> tuple[int, float]:
    """
    Evaluates a single particle position.
    Defined at module level so it can be pickled by multiprocessing.

    Args:
        args: Tuple of (index, position_array, fitness_fn).

    Returns: Tuple of (index, fitness_value).
    """
    index, position, fitness_fn = args
    return index, fitness_fn(position)


def _evaluate_batch(args: tuple) -> list[tuple[int, float]]:
    """
    Evaluates a batch of particles in a single process call.
    Reduces IPC overhead by sending multiple particles per task.
    Defined at module level so it can be pickled by multiprocessing.

    Args:
        args: Tuple of (indices, positions, fitness_fn).
            indices:    List of particle indices.
            positions:  List of position arrays.
            fitness_fn: Objective function.

    Returns: List of (index, fitness_value) tuples.
    """
    indices, positions, fitness_fn = args
    return [(i, fitness_fn(pos)) for i, pos in zip(indices, positions)]


class MultiprocessingEvaluator(BaseEvaluator):
    """
    V2 - Parallel evaluator using processes.
    Spawns multiple processes to evaluate particles in true parallel.

    batch_size=None: one particle per task, maximum IPC overhead.
    batch_size=N:    N particles per task, reduced IPC overhead.
    On Windows, batching is especially important due to spawn overhead.
    """

    def __init__(self, n_workers: int = 4, batch_size: int | None = None) -> None:
        """
        Args:
            n_workers:  Number of worker processes.
            batch_size: Particles per task sent to each process.
                        None means one particle per task (no batching).
                        Recommended: n_particles // n_workers for best IPC efficiency.
        """
        self.n_workers: int        = n_workers
        self.batch_size: int | None = batch_size

    def evaluate(
        self,
        particles: list[Particle],
        fitness_fn: callable,
    ) -> tuple[list[float], float]:
        """
        Evaluates all particles in parallel using separate processes.
        Uses batching if batch_size is set to reduce IPC overhead.
        Results are returned in the same order as the input particles.

        Args:
            particles:  List of particles to evaluate.
            fitness_fn: Objective function. Takes np.ndarray, returns float.

        Returns: Tuple of (fitness_values, elapsed_time_seconds).
        """
        start = time.perf_counter()

        fitness_values: list[float] = [0.0] * len(particles)

        if self.batch_size is None:
            # No batching — one task per particle
            tasks = [
                (i, particle.position, fitness_fn)
                for i, particle in enumerate(particles)
            ]
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_evaluate_single, task): task[0]
                    for task in tasks
                }
                for future in as_completed(futures):
                    index, fitness = future.result()
                    fitness_values[index] = fitness
        else:
            # Batching — group particles into chunks
            n = len(particles)
            n_batches = math.ceil(n / self.batch_size)
            batches = []
            for b in range(n_batches):
                start_idx = b * self.batch_size
                end_idx   = min(start_idx + self.batch_size, n)
                indices   = list(range(start_idx, end_idx))
                positions = [particles[i].position for i in indices]
                batches.append((indices, positions, fitness_fn))

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_evaluate_batch, batch): batch[0][0]
                    for batch in batches
                }
                for future in as_completed(futures):
                    results = future.result()
                    for index, fitness in results:
                        fitness_values[index] = fitness

        elapsed = time.perf_counter() - start
        return fitness_values, elapsed

    def __repr__(self) -> str:
        return (
            f"MultiprocessingEvaluator(strategy=V2_multiprocessing, "
            f"n_workers={self.n_workers}, batch_size={self.batch_size})"
        )