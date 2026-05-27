"""
Swarm: collection of particles and global best memory.
The swarm is the central object. Contains all the particles and the global best position founded until the moment. 
Handles initialization of all particles at once using NumPy matrix operations and owns the random number generator.
"""

import numpy as np
from pso.core.particle import Particle


class Swarm:
    """
    Holds all particles and tracks the global best position found by any particle.
    """

    def __init__(self, n_particles: int, dim: int, bounds: tuple[float, float], seed: int) -> None:
        """
        Args:
            n_particles: Number of particles in the swarm.
            dim: Number of dimensions of the search space.
            bounds: (min, max) applied to all dimensions.
            seed: Random seed for full reproducibility.
        """
        # The source of randomness for the algorithm
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.n_particles: int = n_particles
        self.dim: int = dim
        self.bounds: tuple[float, float] = bounds

        # Initializes all positions and velocities at once using matrix operations
        low, high = bounds
        positions = self.rng.uniform(low, high, size=(n_particles, dim))
        velocities = self.rng.uniform(-(high - low), (high - low), size=(n_particles, dim))

        # Each particle starts with inf fitness that it will be evaluated on the first iteration
        self.particles: list[Particle] = [
            Particle(positions[i], velocities[i], initial_fitness=np.inf)
            for i in range(n_particles)]

        # Global best, updated whenever any particle finds a better position
        self.best_position: np.ndarray = np.zeros(dim)
        self.best_fitness: float = np.inf

    def update_global_best(self) -> bool:
        """
        Scans all particles and updates global best if any personal best is better.
        Returns: True if the global best was updated, False if not.
        """
        improved = False
        for particle in self.particles:
            if particle.best_fitness < self.best_fitness:
                self.best_fitness = particle.best_fitness
                self.best_position = particle.best_position.copy()
                improved = True
        return improved

    def __repr__(self) -> str:
        return (f" Swarm (n_particles = {self.n_particles}, dim = {self.dim}, "
            f" best_fitness = {self.best_fitness:.6f})")