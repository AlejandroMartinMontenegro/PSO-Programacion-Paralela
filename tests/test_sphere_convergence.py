"""
Tests for correctness: PSO must converge to near-zero on Sphere.
Sphere is the simplest benchmark — if PSO fails here there is a bug.
"""
from pso.experiments.runner import run_pso
from pso.parallel.sequential_eval import SequentialEvaluator


def test_sphere_converges_to_near_zero() -> None:
    """PSO must find a fitness below 1e-4 on Sphere in d=10."""
    result = run_pso( objective = "sphere", dim = 10, n_particles = 30, max_iter = 300, w = 0.7, c1 = 1.5, c2 = 1.5,
        seed = 42, evaluator = SequentialEvaluator(), tolerance = 1e-6, tolerance_window = 30, stagnation_window = 50)
    
    assert result["results"]["best_fitness"] < 1e-4, (
        f"PSO did not converge on Sphere: {result['results']['best_fitness']:.6e}"
    )


def test_sphere_stop_reason_is_not_max_iter() -> None:
    """PSO should converge before hitting max_iter on Sphere in d=5."""
    result = run_pso(objective = "sphere", dim = 5, n_particles = 30, max_iter = 500, w = 0.7, c1 = 1.5, c2 = 1.5,
        seed = 42, evaluator = SequentialEvaluator(), tolerance = 1e-6, tolerance_window = 30, stagnation_window = 50)
    
    assert result["results"]["stop_reason"] != "max_iterations", (
        "PSO hit max_iter on Sphere — it should have converged earlier")


def test_sphere_convergence_multiple_seeds() -> None:
    """PSO must converge on Sphere for several different seeds."""
    for seed in range(5):
        result = run_pso( objective="sphere", dim=10, n_particles=30, max_iter=300, w=0.7, c1=1.5, c2=1.5,
            seed=seed, evaluator=SequentialEvaluator(), tolerance=1e-6, tolerance_window=30,
            stagnation_window=50)
        
        assert result["results"]["best_fitness"] < 1e-4, (f"PSO failed on Sphere with seed={seed}: "
            f"{result['results']['best_fitness']:.6e}")