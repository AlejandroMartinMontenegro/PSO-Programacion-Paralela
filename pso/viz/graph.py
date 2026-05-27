"""
Graph: visualizes the swarm movement over the search space in d=2 and d=3.
Shows particle positions on top of the function contour (2D) or surface (3D).
Generates animations (GIF) or static frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from pso.objectives.functions import FUNCTIONS


def _compute_grid_2d(
    fitness_fn: callable,
    bounds: tuple[float, float],
    resolution: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the fitness function over a 2D grid for contour plotting.

    Args:
        fitness_fn:  Objective function.
        bounds:      (min, max) search space limits.
        resolution:  Number of points per axis.

    Returns: Tuple of (X, Y, Z) grids.
    """
    low, high = bounds
    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [fitness_fn(np.array([xi, yi])) for xi in x]
        for yi in y
    ])
    return X, Y, Z


def _compute_grid_3d(
    fitness_fn: callable,
    bounds: tuple[float, float],
    resolution: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the fitness function over a coarser 3D grid for surface plotting.
    Lower resolution than 2D to keep rendering fast.

    Args:
        fitness_fn:  Objective function.
        bounds:      (min, max) search space limits.
        resolution:  Number of points per axis.

    Returns: Tuple of (X, Y, Z) grids.
    """
    low, high = bounds
    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [fitness_fn(np.array([xi, yi])) for xi in x]
        for yi in y
    ])
    return X, Y, Z


# ─── 2D functions ─────────────────────────────────────────────────────────────

def plot_swarm_animation_2d(
    objective: str,
    trajectory: list[list[list[float]]],
    global_bests: list[list[float]],
    save_path: str | Path | None = None,
    show: bool = True,
    interval: int = 150,
) -> None:
    """
    Animates the swarm movement over the function contour in d=2.

    Args:
        objective:    Name of the benchmark function.
        trajectory:   Particle positions per iteration. Shape: (n_iter, n_particles, 2)
        global_bests: Global best position per iteration. Shape: (n_iter, 2)
        save_path:    If provided, saves as GIF.
        show:         If True, displays interactively.
        interval:     Milliseconds between frames.
    """
    fn_entry   = FUNCTIONS[objective]
    fitness_fn = fn_entry["fn"]
    bounds     = fn_entry["bounds"]

    X, Y, Z = _compute_grid_2d(fitness_fn, bounds)
    log_Z    = np.log1p(Z)

    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(X, Y, log_Z, levels=40, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="log(1 + fitness)")

    particles_scatter = ax.scatter([], [], c="#ffffff", s=20, zorder=5, label="Particles")
    best_scatter      = ax.scatter([], [], c="#ff3333", s=80, marker="*", zorder=6, label="Global best")

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    title = ax.set_title("", fontsize=13, fontweight="bold")

    def update(frame: int):
        positions = np.array(trajectory[frame])
        best      = np.array(global_bests[frame])
        particles_scatter.set_offsets(positions)
        best_scatter.set_offsets(best.reshape(1, -1))
        title.set_text(
            f"{objective.capitalize()} d=2 — Iteration {frame + 1}/{len(trajectory)}"
        )
        return particles_scatter, best_scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=True,
    )

    if save_path:
        save_path = Path(save_path)
        anim.save(save_path, writer="pillow", fps=10)
        print(f"2D animation saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_swarm_frame_2d(
    objective: str,
    positions: list[list[float]],
    global_best: list[float],
    iteration: int,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots a single 2D frame of the swarm over the function contour.

    Args:
        objective:   Name of the benchmark function.
        positions:   Particle positions. Shape: (n_particles, 2)
        global_best: Global best position. Shape: (2,)
        iteration:   Current iteration number for the title.
        save_path:   If provided, saves as PNG.
        show:        If True, displays interactively.
    """
    fn_entry   = FUNCTIONS[objective]
    fitness_fn = fn_entry["fn"]
    bounds     = fn_entry["bounds"]

    X, Y, Z = _compute_grid_2d(fitness_fn, bounds)
    log_Z    = np.log1p(Z)

    positions   = np.array(positions)
    global_best = np.array(global_best)

    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(X, Y, log_Z, levels=40, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="log(1 + fitness)")

    ax.scatter(positions[:, 0], positions[:, 1],
               c="#ffffff", s=20, zorder=5, label="Particles")
    ax.scatter(global_best[0], global_best[1],
               c="#ff3333", s=120, marker="*", zorder=6, label="Global best")

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(
        f"{objective.capitalize()} d=2 — Iteration {iteration}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


# ─── 3D functions ─────────────────────────────────────────────────────────────

def plot_swarm_animation_3d(
    objective: str,
    trajectory: list[list[list[float]]],
    global_bests: list[list[float]],
    save_path: str | Path | None = None,
    show: bool = True,
    interval: int = 150,
) -> None:
    """
    Animates the swarm movement over the function surface in d=2 plotted in 3D.
    The Z axis shows the fitness value at each (x1, x2) position.
    Particles are shown floating above the surface at their fitness height.

    Args:
        objective:    Name of the benchmark function.
        trajectory:   Particle positions per iteration. Shape: (n_iter, n_particles, 2)
        global_bests: Global best position per iteration. Shape: (n_iter, 2)
        save_path:    If provided, saves as GIF.
        show:         If True, displays interactively.
        interval:     Milliseconds between frames.
    """
    fn_entry   = FUNCTIONS[objective]
    fitness_fn = fn_entry["fn"]
    bounds     = fn_entry["bounds"]

    X, Y, Z = _compute_grid_3d(fitness_fn, bounds)
    log_Z    = np.log1p(Z)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, log_Z, cmap="viridis", alpha=0.6, zorder=0)

    # Initialize empty scatter plots
    particles_scatter = ax.scatter([], [], [], c="#ffffff", s=20, zorder=5, label="Particles")
    best_scatter      = ax.scatter([], [], [], c="#ff3333", s=80, marker="*", zorder=6, label="Global best")

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("x₁", fontsize=10)
    ax.set_ylabel("x₂", fontsize=10)
    ax.set_zlabel("log(1 + fitness)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    title = ax.set_title("", fontsize=12, fontweight="bold")

    def update(frame: int):
        positions = np.array(trajectory[frame])
        best      = np.array(global_bests[frame])

        # Compute fitness height for each particle
        z_particles = np.array([
            np.log1p(fitness_fn(p)) for p in positions
        ])
        z_best = np.log1p(fitness_fn(best))

        particles_scatter._offsets3d = (
            positions[:, 0], positions[:, 1], z_particles
        )
        best_scatter._offsets3d = (
            np.array([best[0]]), np.array([best[1]]), np.array([z_best])
        )
        title.set_text(
            f"{objective.capitalize()} d=2 (3D view) — "
            f"Iteration {frame + 1}/{len(trajectory)}"
        )
        return particles_scatter, best_scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False,
    )

    if save_path:
        save_path = Path(save_path)
        anim.save(save_path, writer="pillow", fps=10)
        print(f"3D animation saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_swarm_frame_3d(
    objective: str,
    positions: list[list[float]],
    global_best: list[float],
    iteration: int,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots a single 3D frame of the swarm over the function surface.

    Args:
        objective:   Name of the benchmark function.
        positions:   Particle positions. Shape: (n_particles, 2)
        global_best: Global best position. Shape: (2,)
        iteration:   Current iteration number for the title.
        save_path:   If provided, saves as PNG.
        show:        If True, displays interactively.
    """
    fn_entry   = FUNCTIONS[objective]
    fitness_fn = fn_entry["fn"]
    bounds     = fn_entry["bounds"]

    X, Y, Z = _compute_grid_3d(fitness_fn, bounds)
    log_Z    = np.log1p(Z)

    positions   = np.array(positions)
    global_best = np.array(global_best)

    z_particles = np.array([np.log1p(fitness_fn(p)) for p in positions])
    z_best      = np.log1p(fitness_fn(global_best))

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, log_Z, cmap="viridis", alpha=0.6, zorder=0)
    ax.scatter(positions[:, 0], positions[:, 1], z_particles,
               c="#ffffff", s=20, zorder=5, label="Particles")
    ax.scatter(global_best[0], global_best[1], z_best,
               c="#ff3333", s=120, marker="*", zorder=6, label="Global best")

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("x₁", fontsize=10)
    ax.set_ylabel("x₂", fontsize=10)
    ax.set_zlabel("log(1 + fitness)", fontsize=10)
    ax.set_title(
        f"{objective.capitalize()} d=2 (3D view) — Iteration {iteration}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)