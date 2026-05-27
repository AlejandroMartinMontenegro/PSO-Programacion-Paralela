"""
Equations: core PSO velocity and position update formulas.
Bounds strategy: clamp (clip position to bounds, zero velocity on violated dimensions). 

"""

import numpy as np


def update_velocity( velocity: np.ndarray, position: np.ndarray, personal_best: np.ndarray, global_best: np.ndarray, w: float,
    c1: float, c2: float, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    
    """
    Computes new velocity for one particle using the canonical PSO equation.

    v_new = w * v_old + c1 * r1 * (personal_best - position) + c2 * r2 * (global_best  - position)

    Args:
        velocity: Current velocity. Shape: (d,)
        position: Current position. Shape: (d,)
        personal_best: Best position found by this particle. Shape: (d,)
        global_best: Best position found by the entire swarm. Shape: (d,)
        w: Inertia weight, how much of the old velocity is kept.
        c1: Cognitive coefficient, attraction to personal best.
        c2: Social coefficient, attraction to global best.
        r1: Random vector in [0, 1]. Shape: (d,)
        r2: Random vector in [0, 1]. Shape: (d,)

    Returns: New velocity. Shape: (d,)
    """
    
    inertia = w * velocity
    individual_memory = c1 * r1 * (personal_best - position)
    social_influence = c2 * r2 * (global_best - position)

    return inertia + individual_memory + social_influence


def update_position( position: np.ndarray, velocity: np.ndarray, bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Moves the particle by adding velocity to position, then clamps to bounds.
    If a particle goes out of the search space, the position exceeds the bounds and the velocity in that dimension goes to zero.
    (Clamp strategy)

    Args:
        position: Current position. Shape: (d,)
        velocity: Current velocity. Shape: (d,)
        bounds: (min, max) limits for all dimensions.

    Returns: Tuple of (new_position, new_velocity) both Shape: (d,)
    """
    
    new_position = position + velocity
    new_velocity = velocity.copy()
    low, high = bounds

    # Find dimensions that violated the bounds
    below = new_position < low
    above = new_position > high

    # Clamp position to bounds
    new_position = np.clip(new_position, low, high)

    # Zero out velocity on violated dimensions
    new_velocity[below | above] = 0.0

    return new_position, new_velocity