# PSO Parallel Programming — Final Report

**Course:** Parallel Programming, 4th year Computer Science  
**University:** CUNEF Universidad  
**Academic year:** 2025/2026  

---

## 1. Introduction

This report presents the design, implementation, and experimental evaluation of a
Particle Swarm Optimization (PSO) system in Python. The central objective is to
compare different parallel and concurrent evaluation strategies applied to the PSO
fitness evaluation step, and to analyze when and why each strategy outperforms or
underperforms the sequential baseline.

PSO is a population-based metaheuristic that mimics the collective movement of
bird flocks. A swarm of particles explores a continuous search space, each updating
its velocity based on its own best known position and the best position found by
the entire swarm.

---

## 2. Implementation

### 2.1 Architecture

The system follows a modular architecture based on the Strategy Pattern. The core
PSO engine in pso/core/ is completely decoupled from the evaluation strategy. The
runner in pso/experiments/runner.py only interacts with the BaseEvaluator interface
and never knows which strategy is running underneath.

Modules:

    pso/core/           PSO engine: Swarm, Particle, equations, stopping criteria, topology
    pso/objectives/     Benchmark functions: Sphere, Rosenbrock, Rastrigin, Ackley
    pso/parallel/       Evaluation strategies: V0 to V4
    pso/experiments/    Runner, grid search, analysis
    pso/io/             JSON persistence and loading
    pso/viz/            Convergence plots, boxplots, speedup charts, swarm animations

### 2.2 PSO Algorithm

The canonical PSO velocity update equation is:

    v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))

where w is the inertia weight, c1 the cognitive coefficient, c2 the social
coefficient, and r1, r2 are random vectors drawn uniformly from [0, 1].

Bounds enforcement uses the clamp strategy: if a particle exits the search space,
its position is clipped to the boundary and its velocity in that dimension is set
to zero. This was chosen over reflect or penalty for its simplicity and
predictability.

### 2.3 Stopping Criteria

Three conditions are checked at each iteration in order of priority:

    1. max_iterations: hard limit, always respected
    2. stagnation: no change at all in the last stagnation_window iterations
    3. tolerance: improvement smaller than epsilon over tolerance_window iterations

### 2.4 Reproducibility

All randomness flows from a single np.random.default_rng(seed) instance owned by
the Swarm. This guarantees that the same seed always produces identical results
across runs and machines. Seeds for multi-run experiments are generated as:

    seeds = [base_seed + i for i in range(n_seeds)]

Default configuration: base_seed=42, n_seeds=5 → seeds [42, 43, 44, 45, 46].

---

## 3. Benchmark Functions

| Function    | Global minimum      | Difficulty | Bounds            |
|-------------|---------------------|------------|-------------------|
| Sphere      | f(0,...,0) = 0      | Easy       | [-5.12, 5.12]     |
| Rosenbrock  | f(1,...,1) = 0      | Medium     | [-2.048, 2.048]   |
| Rastrigin   | f(0,...,0) = 0      | Hard       | [-5.12, 5.12]     |
| Ackley      | f(0,...,0) = 0      | Hard       | [-32.768, 32.768] |

Sphere is convex and unimodal — PSO converges reliably. Rosenbrock has a narrow
curved valley that is difficult to follow. Rastrigin and Ackley are highly
multimodal with many local minima, making convergence to the global optimum
challenging especially at high dimensions.

---

## 4. Parallel Strategies

### V0 — Sequential (baseline)

Simple Python list comprehension. One particle evaluated at a time. All speedup
measurements are computed relative to this version.

### V1 — Threading (ThreadPoolExecutor)

Submits one evaluation task per particle to a thread pool. Due to Python's Global
Interpreter Lock (GIL), only one thread executes Python bytecode at a time. For
pure math functions, V1 is slower than V0 because the overhead of thread creation
and task dispatch dominates. NumPy releases the GIL internally for array
operations, so V1 would benefit for more complex NumPy-heavy evaluations.

### V2 — Multiprocessing (ProcessPoolExecutor)

Each worker process has its own Python interpreter and its own GIL, achieving true
CPU parallelism. However, data must be serialized (pickled) to cross process
boundaries, adding IPC overhead. On Windows, processes are created with spawn
rather than fork, meaning each worker starts a full Python interpreter from
scratch. This dominates for cheap fitness functions like Sphere.

Batching reduces the number of IPC round-trips by grouping particles into chunks.
With n_workers=4 and n_particles=30, batch_size=7 produces 4 tasks instead of 30,
reducing overhead significantly. Despite this, spawn overhead on Windows still
makes V2 the slowest strategy for cheap functions.

### V3 — Async (asyncio.gather)

All coroutines run in a single thread using cooperative multitasking. For CPU-bound
math functions, V3 adds event loop overhead with no benefit. Its strength appears
in latency-bound scenarios: with asyncio_simulated mode adding 10ms of sleep per
particle, 30 particles would take 300ms sequentially but only ~10ms with asyncio,
demonstrating a 30x improvement in I/O-bound cases.

### V4 — Vectorized (NumPy)

Passes all particle positions as a matrix (n_particles × d) to a dedicated
vectorized fitness function that returns all fitness values in a single NumPy call,
with no Python loops. Each benchmark has a dedicated implementation (sphere_vec,
rosenbrock_vec, rastrigin_vec, ackley_vec) that uses axis-wise matrix operations.
Velocity and position updates are also fully vectorized, eliminating the
per-particle loop entirely. NumPy dispatches these operations to BLAS/LAPACK and
SIMD instructions internally, making V4 the fastest strategy for simple math
functions.

Note: V4 produces slightly different fitness values than V0-V3 because r1 and r2
are generated as full matrices at once rather than per-particle sequentially. Both
approaches are mathematically valid — they simply explore different trajectories
from the same seed.

---

## 5. Experimental Setup

    PSO parameters:    n_particles=30, max_iter=200, w=0.7, c1=1.5, c2=1.5
    Stopping:          tolerance=1e-6, tolerance_window=30, stagnation_window=50
    Functions:         Sphere, Rosenbrock, Rastrigin, Ackley
    Dimensions:        d = 2, 10, 30
    Seeds:             5 per combination (base_seed=42: seeds 42-46)
    Total runs:        5 strategies × 4 functions × 3 dimensions × 5 seeds = 300

Hardware: Windows 11, Python 3.12.7, NumPy 2.4.4.

---

## 6. Results

### 6.1 Timing — Sphere d=10 (mean over 5 seeds)

| Strategy             | Mean time | Speedup vs V0 |
|----------------------|-----------|---------------|
| V0 Sequential        | 0.085s    | 1.00x         |
| V1 Threading         | 0.323s    | 0.26x         |
| V3 Async             | 0.206s    | 0.41x         |
| V4 Vectorized        | 0.041s    | 2.07x         |
| V2 Multiprocessing   | ~55s      | ~0.002x       |

V4 is the clear winner for simple CPU-bound functions. V1 and V3 are slower than
V0 due to overhead. V2 is impractical on Windows for cheap evaluations.

### 6.2 Solution Quality — Sphere

All strategies (V0, V1, V3) converge to the same fitness values with the same
seeds, confirming that the strategy does not affect the mathematical result. V4
converges to similar but not identical values due to the different r1/r2 generation
order, which is expected and correct.

### 6.3 Convergence by Function and Dimension

Sphere converges reliably across all dimensions and seeds. In d=2 PSO reaches
values below 1e-9. In d=10 it reaches ~1e-8. In d=30 it fails to converge within
200 iterations, showing the curse of dimensionality.

Rosenbrock fails to converge in d=10 and d=30 within 200 iterations — the narrow
valley makes it difficult for the swarm to follow the gradient toward the optimum.

Rastrigin converges only in d=2. In d=10 and d=30 the many local minima trap the
swarm, producing fitness values in the range 4-130 depending on dimension and seed.

Ackley converges in d=2. In d=10 it reaches values around 1e-5. In d=30 it gets
trapped with fitness values around 2-6.

### 6.4 Grid Search Results — Sphere d=10

Top 5 hyperparameter combinations by average fitness over 5 seeds:

| w   | c1  | c2  | n_particles | avg_fitness |
|-----|-----|-----|-------------|-------------|
| 0.4 | 1.5 | 1.5 | 50          | 5.66e-11    |
| 0.4 | 2.0 | 1.5 | 50          | 6.37e-11    |
| 0.4 | 1.0 | 2.0 | 50          | 1.45e-10    |
| 0.4 | 1.5 | 2.0 | 50          | 6.02e-10    |
| 0.7 | 1.0 | 1.0 | 50          | 7.42e-10    |

Lower inertia (w=0.4) and larger swarm (n=50) consistently produce better results
on Sphere, suggesting that exploitation is more important than exploration for this
unimodal function.

---

## 7. Discussion

### 7.1 GIL and Threading

The GIL prevents true parallel execution of Python bytecode, making V1 slower than
V0 for CPU-bound evaluations. Threading would be beneficial if the fitness function
called external services, performed file I/O, or used NumPy operations that release
the GIL internally. For the benchmarks used here, threading adds overhead without
benefit.

### 7.2 IPC Overhead and Multiprocessing

V2 achieves true parallelism but the cost of spawning processes on Windows
completely dominates for cheap functions. In a Linux environment with fork-based
process creation, V2 would be significantly faster for expensive fitness evaluations
(e.g., running simulations or calling external models). Batching reduces the number
of IPC round-trips from 30 to 4 per iteration but cannot eliminate spawn overhead.

### 7.3 Async and Cooperative Concurrency

V3 is best suited for I/O-bound or latency-bound evaluations. When each evaluation
involves waiting for a network response or an external service, asyncio allows all
30 evaluations to wait concurrently rather than sequentially. For pure math
functions it adds no value.

### 7.4 Vectorization

V4 is the most practical strategy for simple math functions. By eliminating Python
loops and delegating computation to NumPy's optimized C/Fortran kernels, it
achieves 2x speedup over V0. This advantage grows with larger swarms and higher
dimensions where the matrix operations become increasingly efficient relative to
the Python loop overhead.

### 7.5 Scalability

All strategies show degrading solution quality as dimension increases. This is the
curse of dimensionality: the search space grows exponentially while the swarm size
remains fixed at 30 particles. Increasing n_particles and max_iter would improve
results but at a linear cost in computation time.

---

## 8. Use Case — PSO Applied to a Real Optimization Problem

To demonstrate PSO beyond synthetic benchmarks, we define a fitness function that
optimizes the hyperparameters of a support vector classifier on the Iris dataset.
The two PSO dimensions are the regularization parameter C and the kernel coefficient
gamma of an RBF SVM, both searched in log scale. The fitness function trains the
model with 5-fold cross-validation and returns 1 - accuracy as the value to
minimize, so PSO naturally seeks the combination that maximizes classification
accuracy.

    Objective:   minimize 1 - CV_accuracy(SVC(C, gamma))
    Dimensions:  d=2  (log10(C) in [-2, 3],  log10(gamma) in [-4, 0])
    PSO config:  n_particles=20, max_iter=50, w=0.7, c1=1.5, c2=1.5, seed=42

This case illustrates two important points. First, PSO works as a black-box
optimizer for any callable fitness function — it requires no gradient information
and no modification to the core algorithm. Second, the fitness evaluation here
takes ~20ms per particle (cross-validation overhead), which makes V2 multiprocessing
genuinely competitive: with 4 workers each handling 5 particles per batch, the
expected speedup on Linux would be close to 4x, whereas V4 vectorization offers
no advantage because the function cannot be expressed as a matrix operation.

This use case demonstrates that the choice of parallel strategy depends entirely
on the cost and structure of the fitness function, not on the PSO algorithm itself.

---

## 9. Recommendations

For simple CPU-bound fitness functions on a single machine, V4 vectorization is
the recommended strategy. It is the fastest, simplest, and most portable option
with no additional dependencies.

For expensive fitness evaluations (simulations, ML model inference, external APIs),
the recommendation depends on the bottleneck. If the evaluation is CPU-bound and
the function can be compiled or uses C extensions, V2 multiprocessing on Linux
would be appropriate. If it involves network latency or I/O, V3 async is the
natural choice.

V1 threading is generally not recommended for pure Python or NumPy math functions.
It is only beneficial when the function releases the GIL internally or performs
significant I/O.

For hyperparameter tuning on simple functions, lower inertia (w around 0.4) and
larger swarms (n=50) tend to perform better. For multimodal functions like
Rastrigin, increasing max_iter and n_particles is more important than tuning w.

---

## 10. Conclusions

This project implements a complete PSO system with five evaluation strategies and
demonstrates their practical trade-offs on standard benchmark functions. The main
findings are:

- Vectorization (V4) is the most effective strategy for cheap CPU-bound functions,
  achieving 2x speedup over sequential baseline.
- Threading (V1) and async (V3) are slower than sequential for CPU-bound math due
  to overhead exceeding any concurrency benefit.
- Multiprocessing (V2) achieves true parallelism but is impractical on Windows for
  cheap functions due to spawn overhead. On Linux it would be preferable for
  expensive evaluations such as the SVM use case described in section 8.
- PSO converges reliably on simple unimodal functions (Sphere) but struggles with
  multimodal functions (Rastrigin) at high dimensions within 200 iterations.
- The Strategy Pattern allows clean comparison of all approaches with a single
  shared PSO core, confirming the value of modular architecture.