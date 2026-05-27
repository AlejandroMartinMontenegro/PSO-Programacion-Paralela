"""
V1: Is the threading evaluator.
Uses ThreadPoolExecutor to evaluate particles concurrently across multiple threads.
All speedup measurements are relative to SequentialEvaluator (V0).
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pso.core.particle import Particle
from pso.parallel.sequential_eval import BaseEvaluator


class ThreadingEvaluator(BaseEvaluator):
    
    """
    V1: Concurrent evaluator using threads.
    Submits one evaluation task per particle to a thread pool.
    The number of worker threads is configurable.
    """

    def __init__(self, n_workers: int = 4) -> None:
        
        """
        Args:
            n_workers: Number of threads in the pool.
            Recommended: number of logical CPU cores.
        """
        self.n_workers: int = n_workers

    def evaluate( self, particles: list[Particle], fitness_fn: callable) -> tuple[list[float], float]:
        
        """
        Evaluates all particles concurrently using a thread pool.
        The results are returned in the same order as the input particles.

        Args:
            particles: List of particles to evaluate.
            fitness_fn: Objective function. Takes np.ndarray, returns float.

        Returns: Tuple of (fitness_values, elapsed_time_seconds).
        """
        
        start = time.perf_counter()

        # Submit one task per particle — each thread evaluates one particle
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Map future -> index to preserve original particle order
            future_to_index = {
                executor.submit(fitness_fn, particle.position): i
                for i, particle in enumerate(particles)}

            # Collect results preserving order
            fitness_values: list[float] = [0.0] * len(particles)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                fitness_values[index] = future.result()

        elapsed = time.perf_counter() - start
        return fitness_values, elapsed

    def __repr__(self) -> str:
        return f"ThreadingEvaluator(strategy = V1_threading, n_workers = {self.n_workers})"