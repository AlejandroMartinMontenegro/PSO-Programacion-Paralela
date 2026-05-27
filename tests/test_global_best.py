"""
Tests for global best monotonicity: the best fitness must never get worse.
The convergence curve must be monotonically non-increasing.
"""

from pso.experiments.runner import run_pso
from pso.parallel.sequential_eval import SequentialEvaluator


def _run(objective: str, seed: int = 42) -> list[float]:
    """Helper: runs PSO and returns the convergence curve as a list of floats."""
    result = run_pso( objective = objective, dim = 5, n_particles = 20, max_iter = 100, w = 0.7, c1 = 1.5, c2 = 1.5,
        seed = seed,  evaluator = SequentialEvaluator(), tolerance = 1e-6, tolerance_window = 20, stagnation_window = 30)
    return [entry["best_fitness"] for entry in result["convergence_curve"]]


def test_global_best_never_worsens_sphere() -> None:
    """Global best must be non-increasing on Sphere."""
    curve = _run("sphere")
    for i in range(1, len(curve)):
        assert curve[i] <= curve[i - 1], (f"Global best worsened at iter {i}: {curve[i-1]:.6e} -> {curve[i]:.6e}")


def test_global_best_never_worsens_rastrigin() -> None:
    """Global best must be non-increasing on Rastrigin."""
    curve = _run("rastrigin")
    for i in range(1, len(curve)):
        assert curve[i] <= curve[i - 1], (f"Global best worsened at iter {i}: {curve[i-1]:.6e} -> {curve[i]:.6e}")


def test_global_best_never_worsens_ackley() -> None:
    """Global best must be non-increasing on Ackley."""
    curve = _run("ackley")
    for i in range(1, len(curve)):
        assert curve[i] <= curve[i - 1], (f"Global best worsened at iter {i}: {curve[i-1]:.6e} -> {curve[i]:.6e}")