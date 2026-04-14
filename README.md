# Cvxium

[![CI](https://github.com/rwilson4/Cvxium/actions/workflows/ci.yml/badge.svg)](https://github.com/rwilson4/Cvxium/actions/workflows/ci.yml)

Cvxium (pronounced "Calcium") is a Python framework for building fast
Interior Point Method (IPM) solvers for convex optimization problems of
the form:

```
minimize    f0(x)
subject to  A x = b
            fi(x) <= 0,  i = 1, ..., n
```

The framework's distinguishing feature is a clean interface for
exploiting **Hessian structure** to accelerate Newton steps. A generic
solver inverts an n×n dense matrix at each iteration — O(n³). By
encoding the Hessian's structure (diagonal, low-rank update, arrow
sparsity pattern, etc.), the same iteration can run in O(n). This
allows problems to scale to dimensions in the thousands or even higher.

## Why Cvxium?

Most convex optimization needs are well-served by existing tools. Here
is how Cvxium compares:

| Tool               | When to use it                                               | Why not Cvxium                                                                            |
|--------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **scipy.optimize** | General-purpose unconstrained/constrained optimization       | Handles arbitrary problems with minimal setup; no structural speedups needed              |
| **Cvxpy**          | Rapid prototyping of convex programs; standard problem forms | Modeling-layer convenience; dispatches to mature solvers (OSQP, SCS, ECOS) under the hood |
| **Gurobi / CPLEX** | LP, QP, MIP at industrial scale                              | Commercial license; exceptional performance on problems they support, including integers  |
| **OSQP / SCS**     | Large-scale QPs and conic programs                           | Fast first-order methods; good default choice when the problem fits their form            |

**Use Cvxium when:**

- Your problem has a custom convex structure that does not map cleanly
  onto a standard QP/LP/SOCP form — e.g., KL-divergence objectives,
  Huber loss with non-standard constraints, or specialized barrier
  functions.
- The Hessian has exploitable structure (diagonal, diagonal plus
  low-rank, arrow sparsity) that off-the-shelf solvers cannot leverage.
- You need predictable, low-overhead performance without a commercial
  license or a large solver dependency.

**Do not use Cvxium when:**

- Your problem fits a standard form that Cvxpy or Gurobi handles well.
  Those tools are mature, well-tested, and require far less code.
- You need integer variables. Cvxium is strictly continuous convex
  optimization.
- You want a solver you can just call. Cvxium's value is in the
  framework: you implement the math, it handles the IPM loop. If you
  are not willing to derive gradients and Hessians (or have an AI
  agent do this for you), use Cvxpy.

## Installation

```bash
pip install cvxium
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add cvxium
```

## Quick start: ready-made solvers

For the most common problem types, Cvxium ships concrete solvers that
require no subclassing.

### Find x satisfying Ax = b, x ≥ lb

`EqualityWithBoundsSolver` finds a feasible point and, if requested,
minimizes `‖x‖₂²` subject to the constraints:

```python
import numpy as np
from cvxium import EqualityWithBoundsSolver, OptimizationSettings

A = np.random.randn(20, 100)   # p=20 equality constraints, M=100 variables
w_true = np.random.rand(100) + 0.1
b = A @ w_true
lb = 0.01

solver = EqualityWithBoundsSolver(A=A, b=b, lb=lb)

# Feasibility: find any strictly feasible point
result = solver.solve()
assert np.all(result.solution > lb)
assert np.allclose(A @ result.solution, b)

# Optimize: minimize ‖x‖₂²
result = solver.solve(fully_optimize=True)
```

The solver detects infeasibility via the dual certificate and raises
`ProblemCertifiablyInfeasibleError` when the problem has no solution.

### Find x satisfying Ax = b, x ≥ lb, ‖Bx − c‖∞ ≤ ψ

`EqualityWithBoundsAndImbalanceConstraintSolver` adds an L∞ imbalance
constraint, useful when exact balance on a subset of covariates is
required alongside a bound on a larger set:

```python
from cvxium import EqualityWithBoundsAndImbalanceConstraintSolver

solver = EqualityWithBoundsAndImbalanceConstraintSolver(
    A=A, b=b, lb=lb,
    B=B, c=c, psi=0.05,  # ‖Bx − c‖∞ ≤ 0.05
)
result = solver.solve()
```

### Solve a quadratic program with equality and bound constraints

`QuadraticProgramEqualityBoundsSolver` solves:

```
minimize    x^T Q x + c^T x
subject to  A x = b
            x >= xl
```

It accepts optional `Q_vector_multiply` and `Q_solve` callables to
exploit structure in Q:

```python
from cvxium import QuadraticProgramEqualityBoundsSolver

solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
result = solver.solve()

print(result.solution)        # optimal x
print(result.objective_value) # primal objective
print(result.dual_value)      # dual lower bound (duality gap = objective - dual)
print(result.nits)            # outer IPM iterations
print(result.inner_nits)      # Newton iterations per centering step
```

When Q has structure (e.g., diagonal plus rank-one), passing
`Q_vector_multiply` and `Q_solve` callables can yield a further ~12×
speedup over the dense path. See USAGE.md for the full pattern.

## Building a custom solver

Cvxium's real power is its framework for new problem types. You
subclass one of the base classes, implement a handful of methods
(objective, gradient, Hessian multiply, Newton step, dual), and the
IPM loop is handled for you. A library of composable structured linear
algebra operations (e.g. matrix multiplication and solving systems of
linear equations) makes it straightforward to go from a mathematical
description of the Hessian to a fast Newton step.

Full guidance — including worked examples, the class hierarchy, and
the numerical helpers reference — is in USAGE.md and can be retrieved
at runtime:

```python
import cvxium
cvxium.usage()
```

An AI agent can implement a custom solver from an existing codebase
with a prompt like:

> Look at the optimization problem being solved in `<function>`. Learn
> how to use Cvxium by running `python -c 'import cvxium;
> print(cvxium.usage())'`. Make a plan to refactor `<function>` using
> Cvxium.

## Exception hierarchy

```
BacktrackingLineSearchError
├── ConstraintBoundaryError       — step would violate a constraint
├── InvalidDescentDirectionError  — Newton step is not a descent direction
└── SevereCurvatureError          — backtracking condition never satisfied

OptimizationError
├── CenteringStepError            — inner Newton loop failed
└── InteriorPointMethodError      — outer IPM loop failed

ProblemInfeasibleError            — no feasible point exists
ProblemCertifiablyInfeasibleError — dual certificate proves infeasibility
ProblemMarginallyFeasibleError    — feasible set is non-empty but has no interior
```

## Optimization settings

`OptimizationSettings` controls the IPM:

```python
from cvxium import OptimizationSettings

settings = OptimizationSettings(
    barrier_multiplier=10.0,    # factor by which t increases each outer iteration
    outer_tolerance=1e-8,       # duality gap threshold for convergence
    outer_tolerance_soft=None,  # looser threshold for feasibility-only problems
    max_outer_iterations=100,
    max_inner_iterations=100,
    verbose=False,
)
```

## Pareto frontier tracing for multi-objective problems

When a problem involves two or more competing objectives, Cvxium can
trace the full **Pareto frontier** — the curve of solutions where no
objective can be improved without worsening another.

The MultiObjectiveOptimizer (and classes that inherit from it)
facilitates trading off between competing objectives. For example, we
might have a tradeoff between speed and cost in some application. We
can often find a "knee in the curve" that provides most of the speed
benefit, with only a fraction of the cost, giving an "80/20 rule" kind
of performance. This framework provides functionality for visualizing
the tradeoff curve and automatically identifying a good tradeoff.

See USAGE.md for full guidance.

## References

- Boyd, Stephen and Vandenberghe, Lieven. *Convex Optimization*.
  Cambridge University Press, 2004.
