"""
V3 - Async evaluator (cooperative concurrency).
Uses asyncio and gather() to evaluate particles concurrently.
Asyncio runs in a single thread — no true parallelism, no GIL issues.
It shines when evaluation involves I/O or latency (e.g. external service calls).
For pure CPU-bound math functions it offers no speedup over V0.

Two modes:
  simulated_latency=False: pure function evaluation (shows asyncio overhead)
  simulated_latency=True:  adds async sleep to simulate I/O latency (shows asyncio strength)
"""

import asyncio
import time

import numpy as np

from pso.core.particle import Particle
from pso.parallel.sequential_eval import BaseEvaluator


async def _evaluate_single_async(
    index: int,
    position: np.ndarray,
    fitness_fn: callable,
    latency: float,
) -> tuple[int, float]:
    """
    Evaluates a single particle asynchronously.
    If latency > 0, simulates an async I/O wait before evaluation.

    Args:
        index:      Particle index to preserve order.
        position:   Particle position to evaluate.
        fitness_fn: Objective function.
        latency:    Seconds to sleep before evaluating (simulates I/O).

    Returns: Tuple of (index, fitness_value).
    """
    if latency > 0.0:
        await asyncio.sleep(latency)
    return index, fitness_fn(position)


class AsyncEvaluator(BaseEvaluator):
    """
    V3 - Concurrent evaluator using asyncio.
    Launches all particle evaluations concurrently with asyncio.gather().
    All coroutines run in a single thread — cooperative multitasking.

    Best suited for I/O-bound or latency-bound fitness evaluations.
    For CPU-bound math functions, expect similar or worse performance vs V0.
    """

    def __init__(
        self,
        simulated_latency: float = 0.0,
    ) -> None:
        """
        Args:
            simulated_latency: Seconds of async sleep added to each evaluation.
                               0.0 means pure function evaluation, no artificial latency.
                               Use > 0 to demonstrate asyncio efficiency on I/O-bound tasks.
        """
        self.simulated_latency: float = simulated_latency

    def evaluate(
        self,
        particles: list[Particle],
        fitness_fn: callable,
    ) -> tuple[list[float], float]:
        """
        Evaluates all particles concurrently using asyncio.gather().
        Runs the async loop synchronously so it fits the BaseEvaluator interface.

        Args:
            particles:  List of particles to evaluate.
            fitness_fn: Objective function. Takes np.ndarray, returns float.

        Returns: Tuple of (fitness_values, elapsed_time_seconds).
        """
        start = time.perf_counter()
        fitness_values = asyncio.run(self._evaluate_all(particles, fitness_fn))
        elapsed = time.perf_counter() - start
        return fitness_values, elapsed

    async def _evaluate_all(
        self,
        particles: list[Particle],
        fitness_fn: callable,
    ) -> list[float]:
        """
        Internal coroutine that launches all evaluations concurrently.
        asyncio.gather() starts all coroutines at once and waits for all to finish.
        Total time ≈ max(individual times) instead of sum(individual times).

        Args:
            particles:  List of particles to evaluate.
            fitness_fn: Objective function.

        Returns: Fitness values in the same order as input particles.
        """
        coroutines = [
            _evaluate_single_async(i, p.position, fitness_fn, self.simulated_latency)
            for i, p in enumerate(particles)
        ]

        # gather() launches all coroutines concurrently and collects results
        results = await asyncio.gather(*coroutines)

        # Results come back as (index, fitness) tuples — restore original order
        fitness_values = [0.0] * len(particles)
        for index, fitness in results:
            fitness_values[index] = fitness

        return fitness_values

    def __repr__(self) -> str:
        return (
            f"AsyncEvaluator(strategy=V3_async, "
            f"simulated_latency={self.simulated_latency})"
        )