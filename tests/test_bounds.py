"""
Tests for bounds enforcement: no particle should ever leave the search space.
Verifies the clamp strategy works correctly across all dimensions.
"""

import numpy as np
from pso.core.swarm import Swarm
from pso.core.equations import update_velocity, update_position
from pso.objectives.functions import FUNCTIONS


def test_clamp_keeps_position_in_bounds() -> None:
    """After update_position, all positions must be within bounds."""
    bounds = (-5.12, 5.12)
    low, high = bounds
    rng = np.random.default_rng(42)
    # Create a position that is already out of bounds
    position = rng.uniform(-10.0, 10.0, size=(10,))
    velocity = rng.uniform(-5.0, 5.0, size=(10,))
    new_position, new_velocity = update_position(position, velocity, bounds)
    assert np.all(new_position >= low), "Position below lower bound after clamp"
    assert np.all(new_position <= high), "Position above upper bound after clamp"


def test_clamp_zeros_velocity_on_violation() -> None:
    """Velocity must be zeroed on dimensions that violated the bounds."""
    bounds = (-5.0, 5.0)
    low, high = bounds
    # Force a violation: position + velocity goes out of bounds
    position = np.array([4.9])
    velocity = np.array([1.0])   # will push position to 5.9, above high
    _, new_velocity = update_position(position, velocity, bounds)
    assert new_velocity[0] == 0.0, "Velocity not zeroed on violated dimension"


def test_swarm_positions_in_bounds_after_init() -> None:
    """All particle positions must be within bounds right after swarm creation."""
    bounds = (-5.12, 5.12)
    low, high = bounds
    swarm = Swarm(n_particles=30, dim=10, bounds=bounds, seed=42)

    for particle in swarm.particles:
        assert np.all(particle.position >= low)
        assert np.all(particle.position <= high)


def test_positions_stay_in_bounds_during_run() -> None:
    """No particle should leave bounds during a full PSO run."""
    from pso.experiments.runner import run_pso
    from pso.parallel.sequential_eval import SequentialEvaluator
    bounds = FUNCTIONS["rastrigin"]["bounds"]
    low, high = bounds
    result = run_pso( objective="rastrigin", dim=5, n_particles=20, max_iter=100, w=0.7, c1=1.5, c2=1.5,
        seed=42, evaluator=SequentialEvaluator(), tolerance=1e-6, tolerance_window=20, stagnation_window=30)

    # Best position must be within bounds
    best_pos = np.array(result["results"]["best_position"])
    assert np.all(best_pos >= low)
    assert np.all(best_pos <= high)