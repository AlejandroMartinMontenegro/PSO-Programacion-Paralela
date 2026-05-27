"""
Tests that all parallel evaluation strategies produce valid PSO results.
Verifies convergence on Sphere and result consistency across V0-V4.
"""

from pso.experiments.runner import run_pso
from pso.parallel.sequential_eval import SequentialEvaluator
from pso.parallel.threading_eval import ThreadingEvaluator
from pso.parallel.multiprocessing_eval import MultiprocessingEvaluator
from pso.parallel.async_eval import AsyncEvaluator
from pso.parallel.vector_eval import VectorizedEvaluator

_CFG = dict(
    objective="sphere", dim=5, n_particles=20, max_iter=100,
    w=0.7, c1=1.5, c2=1.5, seed=42, tolerance=1e-6,
    tolerance_window=20, stagnation_window=30,
)


def test_threading_converges() -> None:
    """V1 threading must converge to a reasonable fitness on Sphere."""
    r = run_pso(**_CFG, evaluator=ThreadingEvaluator(n_workers=2))
    assert r["results"]["best_fitness"] < 1e-2


def test_multiprocessing_converges() -> None:
    """V2 multiprocessing must converge to a reasonable fitness on Sphere."""
    r = run_pso(**_CFG, evaluator=MultiprocessingEvaluator(n_workers=2, batch_size=10))
    assert r["results"]["best_fitness"] < 1e-2


def test_async_converges() -> None:
    """V3 async must converge to a reasonable fitness on Sphere."""
    r = run_pso(**_CFG, evaluator=AsyncEvaluator(simulated_latency=0.0))
    assert r["results"]["best_fitness"] < 1e-2


def test_vectorized_converges() -> None:
    """V4 vectorized must converge to a reasonable fitness on Sphere."""
    r = run_pso(**_CFG, evaluator=VectorizedEvaluator())
    assert r["results"]["best_fitness"] < 1e-2


def test_v0_v1_same_fitness() -> None:
    """V0 and V1 use the same per-particle loop — must produce identical results."""
    r0 = run_pso(**_CFG, evaluator=SequentialEvaluator())
    r1 = run_pso(**_CFG, evaluator=ThreadingEvaluator(n_workers=2))
    assert r0["results"]["best_fitness"] == r1["results"]["best_fitness"]


def test_v0_v3_same_fitness() -> None:
    """V0 and V3 async use the same per-particle evaluation — must produce identical results."""
    r0 = run_pso(**_CFG, evaluator=SequentialEvaluator())
    r3 = run_pso(**_CFG, evaluator=AsyncEvaluator(simulated_latency=0.0))
    assert r0["results"]["best_fitness"] == r3["results"]["best_fitness"]