"""
Stopping criteria: It decides when the PSO loop should stop.
There are three conditions: max iterations, tolerance (convergence), stagnation.
It will always returns the stop reason, therefore it can be saved in results.
"""


class StoppingCriteria:
    
    """
    Evaluates whether the PSO main loop should stop based on iteration count,
    convergence tolerance, or stagnation of the global best fitness.
    """

    def __init__( self, max_iter: int, tolerance: float, tolerance_window: int, stagnation_window: int) -> None:
        
        """
        Args:
            max_iter: Maximum number of iterations.
            tolerance: Minimum improvement in fitness to not be considered stagnant.
            tolerance_window: Number of iterations to check for tolerance.
            stagnation_window: Number of iterations with zero improvement before stopping.
        """
        
        self.max_iter: int = max_iter
        self.tolerance: float = tolerance
        self.tolerance_window: int = tolerance_window
        self.stagnation_window: int = stagnation_window

    def should_stop( self, iteration: int, fitness_history: list[float]) -> tuple[bool, str]:
        
        """
        It checks all the stopping conditions in order of priority.

        Args:
            iteration: Current iteration number.
            fitness_history: List of the best fitness values recorded of each iteration.

        Returns: Tuple of (should_stop, reason).
            should_stop: True if any condition is achieved, False otherwise.
            reason: One of "max_iterations", "tolerance", "stagnation", or "running".
        """
        
        # Condition 1, max iterations always are checked first
        if iteration >= self.max_iter - 1:
            return True, "max_iterations"

        # Need enough history to check the other two conditions
        if len(fitness_history) < self.stagnation_window:
            return False, "running"
        recent = fitness_history[-self.stagnation_window:]

        # Condition 2, stagnation: no change at all in the last window
        if max(recent) - min(recent) == 0.0:
            return True, "stagnation"

        # Condition 3, tolerance: improvement is smaller than epsilon
        if len(fitness_history) >= self.tolerance_window:
            recent_tol = fitness_history[-self.tolerance_window:]
            improvement = recent_tol[0] - recent_tol[-1]
            if improvement < self.tolerance:
                return True, "tolerance"
        return False, "running"

    def __repr__(self) -> str:
        return (f" StoppingCriteria (max_iter = {self.max_iter}," f"tolerance = {self.tolerance}, "
                f"stagnation_window = {self.stagnation_window})")
        