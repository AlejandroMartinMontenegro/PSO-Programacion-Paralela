# PSO Design Document

## Architecture

The project follows a modular architecture with a clear separation of concerns.
The core PSO engine lives in pso/core/ and is completely independent of any
parallel strategy. The parallel strategies in pso/parallel/ implement a common
interface defined by BaseEvaluator in sequential_eval.py. The runner in
pso/experiments/runner.py orchestrates the full PSO loop and only interacts
with BaseEvaluator — it never knows which strategy is running underneath.

This design follows the Strategy Pattern: swapping V0 for V1, V2, V3, or V4
requires changing a single line in the configuration file or passing a different
CLI argument.

Dependency flow:

    scripts/run_pso.py
        reads configs/default.yaml
        calls experiments/runner.py
            creates core/swarm.py
                contains N x core/particle.py
            uses parallel/sequential_eval.py (or V1, V2, V3, V4)
                calls objectives/functions.py
            uses core/equations.py
            uses core/stopping_criteria.py
            returns results dictionary
        calls io/storage.py         saves JSON to results/
        calls viz/convergence.py    convergence plot
        calls viz/graph.py          swarm animation in d=2

---

## Key Decisions

### Bounds strategy — clamp

When a particle exits the search space its position is clipped to the nearest
boundary and its velocity in that dimension is set to zero. This was chosen over
reflect or penalty because it is simple, predictable, and easy to verify in tests.

### Seed management

All randomness flows from a single np.random.default_rng(seed) instance owned
by the Swarm. This guarantees full reproducibility: same seed always produces
identical results across runs and across machines. Seeds for multi-run experiments
are generated as base_seed + i for i in range(n_seeds).

### Results format — JSON and CSV

Results are saved as JSON rather than CSV because the data is hierarchical.
Metadata, config, timing, and the convergence curve cannot be naturally represented
in a flat table. For the grid search summary, CSV is used because that data is
tabular by nature.

### Trajectory saved only for d=2

Full particle trajectories are only recorded when save_trajectory=True and dim=2.
For higher dimensions the trajectories are large and not visualizable, so saving
them would waste disk space without benefit.

---

## Parallel Strategies

### V0 — Sequential (baseline)

Simple Python list comprehension. One particle evaluated at a time.
All speedup measurements are relative to this version.

### V1 — Threading

Uses ThreadPoolExecutor. Due to Python's GIL, true CPU parallelism is not
achieved for pure math functions. The overhead of thread creation and task
dispatch makes V1 slower than V0 for cheap evaluations. For NumPy-heavy
functions that release the GIL internally, V1 can show some improvement.

### V2 — Multiprocessing

Uses ProcessPoolExecutor. Each process has its own GIL so true CPU parallelism
is achieved. On Windows, processes are created with spawn rather than fork,
which means each worker starts a full Python interpreter from scratch. This
overhead dominates for cheap fitness functions like Sphere, making V2 much slower
than V0 in practice on Windows. On Linux/Mac with fork, V2 would be significantly
faster for expensive evaluations. Batching is used by default (batch_size =
n_particles // n_workers) to reduce the number of IPC round-trips per iteration.

### V3 — Async

Uses asyncio.gather(). All coroutines run in a single thread — no parallelism,
just cooperative concurrency. For CPU-bound math functions V3 adds event loop
overhead with no benefit. The async_simulated mode adds asyncio.sleep() to each
evaluation, simulating I/O latency. In this mode, 30 particles each waiting 10ms
sequentially would take 0.3s per iteration. With asyncio they all wait concurrently
and the total is approximately 10ms per iteration — a 30x improvement.

### V4 — Vectorized (NumPy)

Passes all particle positions as a matrix of shape (n_particles, d) to a
vectorized fitness function that returns a fitness vector of shape (n_particles,)
in a single NumPy call. Each benchmark function has a dedicated vectorized
version (sphere_vec, rosenbrock_vec, rastrigin_vec, ackley_vec) that uses
axis-wise matrix operations with no Python loops. Velocity and position updates
are also fully vectorized using matrix operations, replacing the per-particle
loop entirely. NumPy dispatches these operations to BLAS/LAPACK and SIMD
instructions internally, making V4 the fastest strategy for simple CPU-bound
functions. V4 produces slightly different convergence trajectories than V0-V3
because r1 and r2 are generated as full matrices at once rather than
per-particle sequentially — both approaches are mathematically valid.

---

## Timing Results — Sphere d=10, seed=42, 30 particles

    V0 sequential:           0.087s   stop: tolerance
    V1 threading:            0.310s   stop: tolerance
    V2 multiprocessing:     54.779s   stop: tolerance
    V3 async pure:           0.196s   stop: tolerance
    V3 async simulated 10ms: 2.658s   stop: tolerance
    V4 vectorized:           0.041s   stop: tolerance

All strategies converge to the same result (best fitness ~2.44e-08).
Only timing differs — the algorithm is identical across all versions.

---

## Limitations

V2 multiprocessing is not practical on Windows for cheap fitness functions
due to spawn overhead. Batching reduces the number of IPC round-trips from
n_particles to n_particles // n_workers per iteration, but cannot eliminate
the spawn cost entirely. On Linux with fork-based process creation, V2 would
be significantly faster for expensive evaluations.

The stopping criteria tolerance check compares improvement over a window,
which means very slowly converging algorithms may stop early on difficult
functions like Rastrigin in high dimensions.

V4 vectorized fitness functions require a dedicated implementation per benchmark.
Functions that cannot be expressed as matrix operations must fall back to the
sequential per-row evaluation path.