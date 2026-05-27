<<<<<<< HEAD
# PSO — Particle Swarm Optimization with Parallel Strategies

Final year project for the Parallel Programming course at CUNEF Universidad.
Implements canonical PSO in Python with multiple parallel and concurrent evaluation
strategies, benchmarked across standard optimization functions.

---

## Project Structure

    pso/
    ├── core/            PSO engine: particle, swarm, equations, stopping criteria
    ├── objectives/      Benchmark functions: Sphere, Rosenbrock, Rastrigin, Ackley
    ├── parallel/        Evaluation strategies: V0 sequential, V1 threading,
    │                    V2 multiprocessing, V3 async, V4 vectorized
    ├── experiments/     Runner and grid search orchestration
    ├── io/              Save and load results (JSON/CSV)
    └── viz/             Convergence plots and swarm animations
    scripts/             Entry points: run_pso, run_benchmarks, run_grid_search,
                         make_viz, run_analysis
    tests/               Unit tests (pytest)
    configs/             YAML configuration files
    results/             Generated results (JSON, CSV, plots)

---

## Installation

Clone the repository and install dependencies:

    git clone <repo-url>
    cd pso-project
    python -m venv .venv

Activate the virtual environment:

    # Windows PowerShell
    .venv\Scripts\Activate.ps1

    # Windows Git Bash / Mac / Linux
    source .venv/Scripts/activate

Install the package in editable mode:

    pip install -e ".[dev,viz]"

---

## Usage

### Run a single PSO optimization

With CLI arguments:

    python scripts/run_pso.py --strategy sequential --objective sphere --dim 10
    python scripts/run_pso.py --strategy threading --objective rastrigin --dim 30
    python scripts/run_pso.py --strategy vectorized --objective sphere --dim 30

Interactive mode — select function, dimensions and strategy from a menu:

    python scripts/run_pso.py --interactive

Overwrite existing results file:

    python scripts/run_pso.py --strategy sequential --objective sphere --dim 10 --overwrite

### Run benchmarks across all functions and strategies

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --strategies sequential threading multiprocessing async vectorized
    python scripts/run_benchmarks.py --dims 2 10 --n_seeds 3

### Run hyperparameter grid search

Interactive — asks for function and dimensions:

    python scripts/run_grid_search.py

With CLI arguments:

    python scripts/run_grid_search.py --objective rastrigin --dim 10

### Generate convergence plots and animations

    python scripts/make_viz.py
    python scripts/make_viz.py --save
    python scripts/make_viz.py --animate --objective sphere --save
    python scripts/make_viz.py --benchmarks_dir results/benchmarks --save

### Analyse benchmark results

    python scripts/run_analysis.py
    python scripts/run_analysis.py --benchmarks_dir results/benchmarks

### Run tests

    pytest tests/ -v

---

## Parallel Strategies

| Version | File | Method | Best use case |
|---------|------|--------|---------------|
| V0 | sequential_eval.py | Python loop | Baseline, all cases |
| V1 | threading_eval.py | ThreadPoolExecutor | I/O-bound or NumPy-heavy evaluations |
| V2 | multiprocessing_eval.py | ProcessPoolExecutor | Expensive CPU-bound evaluations |
| V3 | async_eval.py | asyncio.gather() | Latency-bound or I/O-bound evaluations |
| V4 | vector_eval.py | NumPy matrix operations | Simple CPU-bound math functions |

### Performance notes

V1 threading is slower than V0 for pure math functions because the GIL prevents
true parallel execution. For NumPy-heavy functions it may help as NumPy releases
the GIL internally.

V2 multiprocessing achieves real CPU parallelism but on Windows uses spawn to
create processes, which adds significant overhead per task. Batching is used by
default to reduce IPC round-trips. For cheap fitness functions this still makes
V2 much slower than V0 on Windows.

V3 async runs in a single thread. It adds no value for CPU-bound functions but
is highly efficient when evaluation involves latency. The async_simulated mode
demonstrates this with asyncio.sleep().

V4 vectorized passes all particle positions as a matrix (n_particles, d) to a
dedicated NumPy function that returns all fitness values in one call, with no
Python loops. Velocity and position updates are also fully vectorized. This is
the fastest strategy for simple math functions.

### Timing results — Sphere d=10, seed=42, 30 particles

| Strategy | Total time | Stop reason |
|----------|-----------|-------------|
| V0 sequential | 0.087s | tolerance |
| V1 threading | 0.310s | tolerance |
| V2 multiprocessing | 54.779s | tolerance |
| V3 async pure | 0.196s | tolerance |
| V3 async simulated 10ms | 2.658s | tolerance |
| V4 vectorized | 0.041s | tolerance |

All strategies converge to the same result (best fitness ~2.44e-08).
Only timing differs — the algorithm is identical across all versions.

---

## Benchmark Functions

| Function | Global minimum | Difficulty | Default bounds |
|----------|---------------|------------|----------------|
| Sphere | f(0,...,0) = 0 | Easy, convex and unimodal | [-5.12, 5.12] |
| Rosenbrock | f(1,...,1) = 0 | Medium, narrow curved valley | [-2.048, 2.048] |
| Rastrigin | f(0,...,0) = 0 | Hard, highly multimodal | [-5.12, 5.12] |
| Ackley | f(0,...,0) = 0 | Hard, multimodal with flat outer region | [-32.768, 32.768] |

---

## Reproducibility

All runs accept a seed parameter. The seed controls all randomness:
initial positions, initial velocities, and the r1 and r2 random vectors
used in the velocity update equation at each iteration.

Seeds for multi-run experiments are generated as:

    seeds = [base_seed + i for i in range(n_seeds)]
    # Example with base_seed=42, n_seeds=5: [42, 43, 44, 45, 46]

---

## Configuration

Default PSO parameters are in configs/default.yaml.
Grid search parameters are in configs/grid_search.yaml.
CLI arguments always override YAML values when provided.

Key parameters in default.yaml:

    pso:
      n_particles: 30
      max_iter: 200
      w: 0.7        # inertia weight
      c1: 1.5       # cognitive coefficient (attraction to personal best)
      c2: 1.5       # social coefficient (attraction to global best)
      tolerance: 1.0e-6
      stagnation_window: 50

    reproducibility:
      seed: 42

---

## Results format

Each run saves a JSON file to results/ with full metadata, config, timing,
and the convergence curve. Grid search saves a CSV summary.

JSON structure:

    metadata:          seed, platform, python version, cpu_count, git_hash
    config:            all PSO parameters used
    results:           best fitness, best position, stop reason, n iterations
    timing:            total time, eval time, update time, overhead
    convergence_curve: list of {iter, best_fitness} for every iteration

---

## Tests

| File | What it verifies |
|------|-----------------|
| test_seeds.py | Same seed always produces identical results |
| test_bounds.py | No particle ever leaves the search space |
| test_global_best.py | Global best never worsens across iterations |
| test_sphere_convergence.py | PSO converges to near-zero on Sphere |
| test_strategies.py | All parallel strategies converge and V0/V1/V3 are consistent |

Run all tests:

    pytest tests/ -v

---

## Environment

- Python 3.12.7
- NumPy 2.4.4
- Windows 11
- pytest 9.0.2

---

## Author

Alejandro Martín Montenegro
=======
# PSO-Programacion-Paralela
This repository presents a modular and reproducible implementation of Particle Swarm Optimization (PSO designed as an experimental testbed for evaluating parallel and concurrent programming strategies in Python.  
>>>>>>> 8227b458bb4834e6469ff2bb6af41706cf811cae
