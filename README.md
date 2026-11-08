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

## Quick start: compose a problem, pick a solver

A `Problem` states the mathematics: an `Objective` (or none, for a
feasibility problem), a list of inequality constraint layers, and an
optional equality constraint `A x = b`. A `Solver` runs the algorithm.
Each layer and objective reports its Hessian as a structured
`MatrixSystem`, and the solver sums those contributions while keeping
their structure, so the Newton step stays fast without any
subclassing.

### Find x satisfying Ax = b, x ≥ lb, then minimize ‖x‖₂²

```python
import numpy as np
from cvxium import (
    BarrierSolver,
    BoundsConstraintLayer,
    DiagonalSystem,
    EqualityConstraintLayer,
    Problem,
    QuadraticObjective,
)

rng = np.random.default_rng(0)
A = rng.standard_normal((20, 100))  # p = 20 equality constraints, M = 100 variables
b = A @ (rng.random(100) + 0.1)
lb = 0.01

eq = EqualityConstraintLayer(A=A, b=b)
bounds = BoundsConstraintLayer(lb=lb, dimension=100)

# Feasibility: with no objective, the solver stops at a strictly feasible point.
feasible = BarrierSolver().solve(Problem(eq=eq, ineq=[bounds]))
assert np.all(feasible.solution > lb)
assert np.allclose(A @ feasible.solution, b)

# Optimization: minimize (1/2) ‖x‖₂² subject to the same constraints.
objective = QuadraticObjective(
    c=np.zeros(100), hessian_contribution=DiagonalSystem(np.ones(100))
)
problem = Problem.minimize(objective).subject_to(eq=eq, ineq=[bounds])
result = BarrierSolver().solve(problem)

print(result.status)           # SolveStatus.OPTIMAL
print(result.objective_value)  # primal objective
print(result.suboptimality)    # duality gap bound
```

When no strictly feasible `x0` is supplied, the solver builds and
solves a Phase I problem to find one. An infeasible problem raises
`ProblemCertifiablyInfeasibleError` when the dual certificate proves
infeasibility; pass `on_infeasible=...` to change that policy.

### Add an imbalance constraint ‖Bx − c‖∞ ≤ ψ

Layers stack. `ImbalanceConstraintLayer` bounds the imbalance on a set
of covariates, useful when exact balance on a few covariates (the
equality) is required alongside approximate balance on many more:

```python
from cvxium import ImbalanceConstraintLayer

B = rng.standard_normal((5, 100))
c = B @ feasible.solution
imbalance = ImbalanceConstraintLayer(B=B, c=c, psi=0.05)

problem = Problem.minimize(objective).subject_to(eq=eq, ineq=[bounds, imbalance])
result = BarrierSolver().solve(problem, x0=feasible.solution)
```

### Choose an algorithm

The same `Problem` can go to different solvers:

- `BarrierSolver`: the barrier method (Boyd and Vandenberghe, 2004,
  section 11.3), with nested centering steps.
- `PrimalDualSolver`: the primal-dual interior-point method (section
  11.7), carrying the inequality duals explicitly.
- `NewtonSolver`: Newton's method, for problems with no inequality
  constraints.
- `HsdSolver`: a homogeneous self-dual method for linear conic
  programs over products of symmetric cones; needs no Phase I.

```python
from cvxium import PrimalDualSolver

result = PrimalDualSolver().solve(problem)
nu = result.equality_multipliers      # shape (p,)
lmbda = result.inequality_multipliers  # stacked across layers, in order
```

A `SolveResult` round-trips through JSON with `serialize()` and
`SolveResult.deserialize(...)`.

## Building a custom solver

Cvxium's real power is its framework for new problem types. You
implement a new `Objective` or `ConstraintLayer` (values, gradients,
and a structured Hessian contribution), and the solvers handle the
interior-point loop. A library of composable structured linear algebra
(`DiagonalSystem`, `BandedSystem`, `ArrowSystem`, low-rank updates,
block-diagonal and permuted systems, and saddle systems for equality
constraints) makes it straightforward to go from a mathematical
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
> cvxium.usage()'`. Make a plan to refactor `<function>` using Cvxium.

## Exception hierarchy

```
OptimizationError                  — solve stopped before reaching its tolerance
├── CenteringStepError             — inner Newton loop failed
├── InteriorPointMethodError       — outer loop failed
└── StationarityConvergenceError   — two-block saddle solve failed

ProblemInfeasibleError             — no feasible point (or none found)
├── ProblemCertifiablyInfeasibleError — dual certificate proves infeasibility
└── ProblemMarginallyFeasibleError    — may be infeasible; cannot certify

BacktrackingLineSearchError
├── ConstraintBoundaryError        — even tiny steps leave the feasible region
├── InvalidDescentDirectionError   — Newton step is not a descent direction
└── SevereCurvatureError           — sufficient decrease never satisfied

NewtonStepError                    — Newton step could not be computed
```

Exceptions carry diagnostic data (the last iterate, suboptimality), so
a caller can recover a partial solution.

## Optimization settings

`OptimizationSettings` groups the knobs by phase:

```python
from cvxium import (
    InnerSettings,
    ObservabilitySettings,
    OptimizationSettings,
    OuterSettings,
)

settings = OptimizationSettings(
    outer=OuterSettings(
        tolerance=1e-8,           # duality gap threshold for convergence
        barrier_multiplier=10.0,  # factor by which t increases each outer iteration
        max_outer_iterations=30,
    ),
    inner=InnerSettings(max_iterations=1_000),
    observability=ObservabilitySettings(verbose=True),
)
result = BarrierSolver(settings=settings).solve(problem, x0=feasible.solution)
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
