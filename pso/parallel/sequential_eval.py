"""
V0 - Sequential evaluator (baseline).
Defines the base interface that all parallel strategies must follow. 
Evaluates particles one by one in a simple Python loop, no parallelism.
All speedup measurements are relative to this version.
"""

import time
from abc import ABC, abstractmethod
import numpy as np
from pso.core.particle import Particle


class BaseEvaluator(ABC):
    
    """
    Abstract base class for all evaluation strategies (V0 to V4).
    Any evaluator must implement the evaluate method.
    The PSO runner only talks to this interface, it never knows what strategy is running underneath.
    """

    @abstractmethod
    def evaluate( self, particles: list[Particle], fitness_fn: callable) -> tuple[list[float], float]:
        """
        Evaluates all particles and returns their fitness values.
        Args:
            particles: List of particles to evaluate.
            fitness_fn: Objective function. Takes np.ndarray, returns float.
            
        Returns: Tuple of (fitness_values, elapsed_time_seconds).
            fitness_values: Fitness for each particle. Same order as input.
            elapsed_time: Time spent on evaluation in seconds.
        """


class SequentialEvaluator(BaseEvaluator):
    
    """
    V0 - Baseline evaluator. Evaluates each particle one by one.
    Simple Python loop, no parallelism. It is a reference point for all speedup calculations.
    """

    def evaluate( self, particles: list[Particle], fitness_fn: callable) -> tuple[list[float], float]:
        
        """
        Evaluates all particles sequentially.
        Args:
            particles: List of particles to evaluate.
            fitness_fn: Objective function. Takes np.ndarray, returns float.

        Returns: Tuple of (fitness_values, elapsed_time_seconds).
        """
        
        start = time.perf_counter()
        fitness_values = [
            fitness_fn(particle.position)
            for particle in particles
        ]
        
        elapsed = time.perf_counter() - start
        return fitness_values, elapsed

    def __repr__(self) -> str:
        return "SequentialEvaluator(strategy = V0_sequential)"