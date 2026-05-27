"""
Topology: defines how particles share information in the swarm.
The topology determines which particles each particle can "see" when
updating its velocity toward the best known position.

Global-best (star topology): every particle sees the single best position
found by any particle in the entire swarm. This is the canonical PSO topology
and the one implemented in this project.

Local-best (ring topology): each particle only sees its k nearest neighbors.
This is optional and not implemented here — it is left as a future extension.
"""

from abc import ABC, abstractmethod
import numpy as np
from pso.core.particle import Particle


class BaseTopology(ABC):
    """
    Abstract base class for PSO topologies.
    A topology defines which particles are visible to each particle
    when computing the social component of the velocity update.
    """

    @abstractmethod
    def get_best_position(
        self,
        particles: list[Particle],
        global_best_position: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the best known position visible to a particle.

        Args:
            particles:             All particles in the swarm.
            global_best_position:  Current global best position.

        Returns: Best position to use in velocity update. Shape: (d,)
        """


class GlobalBestTopology(BaseTopology):
    """
    Star topology — global best.
    Every particle sees the single best position found by the entire swarm.
    This is the simplest and most common PSO topology.
    Converges fast but can get trapped in local minima on multimodal functions.
    """

    def get_best_position(
        self,
        particles: list[Particle],
        global_best_position: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the global best position — same for all particles.

        Args:
            particles:             All particles in the swarm (not used here).
            global_best_position:  Current global best position.

        Returns: Global best position. Shape: (d,)
        """
        return global_best_position

    def __repr__(self) -> str:
        return "GlobalBestTopology(type=star)"