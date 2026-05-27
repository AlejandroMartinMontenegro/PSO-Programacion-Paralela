"""
Contains the four function benchmarks for PSO evaluation.
Each function receives a position vector and returns a scalar fitness value.
Lower is better, all functions have a global minimum of 0.

Functions: Sphere, Rosenbrock, Rastrigin, Ackley.
Default bounds and known global minimum are stored alongside each function.
"""

import numpy as np


def sphere(x: np.ndarray) -> float:
    
    """
    Sphere function. Simplest benchmark, convex and unimodal.
    Global minimum: f(0, 0, ..., 0) = 0
    Default bounds: [-5.12, 5.12]
    Args: x: Position vector. Shape: (d,)
    Returns: Scalar fitness value.
    """
    
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    
    """
    Rosenbrock function. Narrow curved valley leading to the minimum.
    Global minimum: f(1, 1, ..., 1) = 0
    Default bounds: [-2.048, 2.048]
    Args: x: Position vector. Shape: (d,)
    Returns: Scalar fitness value.
    """
    
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    
    """
    Rastrigin function. Highly multimodal with many local minima.
    Global minimum: f(0, 0, ..., 0) = 0
    Default bounds: [-5.12, 5.12]
    Args: x: Position vector. Shape: (d,)
    Returns: Scalar fitness value.
    """
    
    d = len(x)
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    
    """
    Ackley function. Multimodal with a nearly flat outer region and deep global minimum.
    Global minimum: f(0, 0, ..., 0) = 0
    Default bounds: [-32.768, 32.768]
    Args: x: Position vector. Shape: (d,)
    Returns: Scalar fitness value.
    """
    
    d = len(x)
    sum_sq   = np.sum(x ** 2)
    sum_cos  = np.sum(np.cos(2 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return float(term1 + term2 + 20.0 + np.e)


# ── Vectorized versions (accept matrix input) ──────────────────────────────

def sphere_vec(X: np.ndarray) -> np.ndarray:
    return np.sum(X ** 2, axis=1)

def rosenbrock_vec(X: np.ndarray) -> np.ndarray:
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)

def rastrigin_vec(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    return 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)

def ackley_vec(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    sum_sq = np.sum(X ** 2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * X), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20.0 + np.e

FUNCTIONS: dict[str, dict] = {
    "sphere": {
        "fn":     sphere,
        "fn_vec": sphere_vec,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
    },
    "rosenbrock": {
        "fn":     rosenbrock,
        "fn_vec": rosenbrock_vec,
        "bounds": (-2.048, 2.048),
        "optimum": 0.0,
    },
    "rastrigin": {
        "fn":     rastrigin,
        "fn_vec": rastrigin_vec,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
    },
    "ackley": {
        "fn":     ackley,
        "fn_vec": ackley_vec,
        "bounds": (-32.768, 32.768),
        "optimum": 0.0,
    },
}