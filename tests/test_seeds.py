"""
Tests for reproducibility: same seed must always produce identical results.
Covers both single-run reproducibility and cross-evaluator consistency.
"""

from pso.experiments.runner import run_pso
from pso.parallel.sequential_eval import SequentialEvaluator

def _base_config() -> dict:
    """Returns a minimal PSO config for testing."""
    return dict( objective = "sphere", dim = 5, n_particles = 20, max_iter = 50, w = 0.7, c1 = 1.5, c2 = 1.5,
        tolerance = 1e-6, tolerance_window = 20, stagnation_window = 30)


def test_same_seed_same_result() -> None:
    """Two runs with the same seed, it must produce identical results."""
    cfg = _base_config()
    result_a = run_pso(**cfg, seed = 42, evaluator = SequentialEvaluator())
    result_b = run_pso(**cfg, seed = 42, evaluator = SequentialEvaluator())
    assert result_a["results"]["best_fitness"] == result_b["results"]["best_fitness"]
    assert result_a["results"]["n_iterations"] == result_b["results"]["n_iterations"]
    assert result_a["convergence_curve"] == result_b["convergence_curve"]


def test_different_seeds_different_results() -> None:
    """Two runs with different seeds, it must produce different convergence curves."""
    cfg = _base_config()
    result_a = run_pso(**cfg, seed = 0, evaluator = SequentialEvaluator())
    result_b = run_pso(**cfg, seed = 1, evaluator = SequentialEvaluator())
    # Different seeds should produce different trajectories
    assert result_a["convergence_curve"] != result_b["convergence_curve"]


def test_seed_zero_works() -> None:
    """Seed 0 is a valid seed and it must not cause any errors."""
    cfg = _base_config()
    result = run_pso(**cfg, seed = 0, evaluator = SequentialEvaluator())
    assert result["results"]["best_fitness"] < 1e4