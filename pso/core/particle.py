"""
Particle: 
Holds position, velocity, and personal best memory.
There is no PSO logic here, just state storage and personal best tracking.
It can only do one active thing, update its best personal position. When there is a value of fitness (X) better, it updates in its memory.
"""

import numpy as np


class Particle:
    """
    Represents one particle in the swarm. A particle knows only about itself: where it is,
    how fast it is moving, and the best place it has visited. It does not know anything about other particles or the global best.
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray, initial_fitness: float) -> None:
        """
        Args: 
         position: Starting position in the search space. Shape: (d,)
         velocity: Starting velocity. Shape: (d,)
         initial_fitness: Fitness value at the starting position.
        """
        # Current state, updated every iteration that is made
        self.position: np.ndarray = position.copy()
        self.velocity: np.ndarray = velocity.copy()

        # Personal best memory, only updated when we find something better
        self.best_position: np.ndarray = position.copy()
        self.best_fitness: float = initial_fitness

    def update_personal_best(self, current_fitness: float) -> bool:
        """
        Compares the current fitness against the personal best at that moment.
        If the current fitness value is better (lower), it updates its personal best memory.

        Args: current_fitness: Fitness value at the current position.

        Returns: True if personal best was updated, False if not.
        """
        
        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position.copy()
            return True
        return False

    def __repr__(self) -> str:
        return (f"Particle(best_fitness = {self.best_fitness:.6f}, "
                f"pos = {self.position})")