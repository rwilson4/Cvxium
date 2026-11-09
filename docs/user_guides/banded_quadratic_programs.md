# Banded Quadratic Programs

Many estimation problems ask for a long sequence of numbers that fits
noisy data, varies smoothly, and respects a sign: a demand rate, a
concentration, a power output. Penalizing the difference between
neighbors makes the objective a quadratic whose matrix is tridiagonal,
and the sign constraint makes the problem a quadratic program. The
tridiagonal structure lets an interior-point method take each Newton
step in O(M) time, and Cvxium does exactly that.

This guide works one example from start to finish. We smooth a
solar panel's power readings, sampled each minute, subject to the
power being nonnegative. We solve it first with cvxpy as most people
would write it, then steel-man cvxpy with a better formulation and
every open-source QP solver it supports, then solve it with Cvxium. On
100,000 readings Cvxium takes about 0.08 seconds: about two times
faster than the fastest cvxpy configuration we found, six times faster
than Clarabel (cvxpy's default conic solver), and about a hundred
times faster than the obvious formulation. We close with how the comparison changes
with problem size. Every `python` block runs, in order, as one script,
except the one block marked as skipped.

## Smoothing a power meter

*Example 1.* A solar installation logs its power output y_i, in
kilowatts, once a minute for M = 100,000 minutes (about ten weeks).
The meter is noisy, with a standard deviation of half a kilowatt, so
at night, when the true output is zero, about half of the readings are
negative. We want an estimate x that stays close to the readings,
changes smoothly from minute to minute, and is never negative:

$$
\begin{array}{ll}
\mbox{minimize}   & \frac{1}{2} \sum_{i=1}^M (x_i - y_i)^2
                    + \frac{\lambda}{2} \sum_{i=1}^{M-1} (x_{i+1} - x_i)^2 \\
\mbox{subject to} & x \succeq 0.
\end{array}
$$

The smoothing parameter λ trades fidelity for smoothness; λ = 50
averages over roughly √λ ≈ 7 minutes. Writing D for the (M - 1) × M
first-difference matrix, the objective is (1/2) x^T P x - y^T x plus a
constant, with

$$
P = I + \lambda D^T D =
\begin{bmatrix}
1 + \lambda & -\lambda    &             &        &             \\
-\lambda    & 1 + 2\lambda & -\lambda   &        &             \\
            & \ddots      & \ddots      & \ddots &             \\
            &             & -\lambda    & 1 + 2\lambda & -\lambda \\
            &             &             & -\lambda    & 1 + \lambda
\end{bmatrix},
$$

symmetric, positive definite, and tridiagonal: bandwidth 1.

```python
import time

import numpy as np

M, lam = 100_000, 50.0
rng = np.random.default_rng(0)
minute = np.arange(M)
sun = np.maximum(0.0, -np.cos(2 * np.pi * minute / 1440.0))  # zero at night
day = minute // 1440
cloud = rng.uniform(0.4, 1.0, day[-1] + 1)[day]  # day-to-day weather
y = 5.0 * sun * cloud + rng.normal(0.0, 0.5, M)  # kW; shape (M,)

# P in banded storage: the superdiagonal (with an unused leading entry) above
# the main diagonal, as scipy.linalg.cholesky_banded expects.
main_diagonal = np.full(M, 1.0 + 2.0 * lam)
main_diagonal[[0, -1]] = 1.0 + lam
superdiagonal = np.full(M - 1, -lam)
ab = np.vstack([np.concatenate([[0.0], superdiagonal]), main_diagonal])
```

Without the constraint, the minimizer solves P x = y, a single
tridiagonal solve. A `BandedSystem` does it in linear time, and shows
why the constraint matters: the unconstrained estimate dips below zero
for a quarter of the minutes, which is half of every night.

```python
from cvxium import BandedSystem

P_system = BandedSystem(ab)
x_unconstrained = P_system.solve(y)
print(f"Unconstrained estimate negative {np.mean(x_unconstrained < 0):.0%} of the time")
```

## A baseline with cvxpy

The model reads naturally in cvxpy:

```{code-block} python
import cvxpy as cp

x = cp.Variable(M)
objective = 0.5 * cp.sum_squares(x - y) + 0.5 * lam * cp.sum_squares(cp.diff(x))
naive = cp.Problem(cp.Minimize(objective), [x >= 0])
naive.solve()
```

(The documentation tests skip this block, since it takes several
seconds.) cvxpy recognizes a QP and hands it to OSQP, its default QP
solver. On our machine the solve takes about 9 seconds, nearly all of
it inside OSQP, which needs about 1,700 iterations. (All timings in
this guide come from an Apple M2 Pro with cvxpy 1.9, Clarabel 0.11,
OSQP 1.1, SCS 3.3, PIQP 0.7, and QOCO 0.3; expect different absolute
numbers elsewhere, but similar ratios.)

## Steel-manning cvxpy

That number is not a fair picture of what cvxpy can do, so before
comparing we give it every advantage we know of.

**Formulate the quadratic directly.** `cp.sum_squares(cp.diff(x))`
canonicalizes through M - 1 auxiliary variables, one per difference,
which doubles the size of the problem the solver sees. Passing P as a
sparse matrix to `cp.quad_form` keeps the problem at M variables;
`cp.psd_wrap` tells cvxpy that P is positive semidefinite, sparing it an
eigenvalue check. For OSQP this alone cuts the iteration count from
about 1,700 to 75.

**Compile once.** cvxpy spends a fair share of a first solve
compiling the problem. Making y a `cp.Parameter` keeps the problem
DPP-compliant, so later solves with new readings skip compilation. We
time those later solves, and report compilation separately.

**No warm starts.** OSQP and SCS warm-start from the previous solution
by default. Solving the same data twice then takes almost no time,
which says nothing about solving new data, so we pass
`warm_start=False`. (In a sequence of similar problems, warm starts
are a real advantage of first-order methods; it is just not the
question here.)

**Try every solver.** cvxpy supports five open-source QP solvers we
can install from PyPI: OSQP and SCS (first-order methods), and Clarabel, PIQP, and
QOCO (interior-point methods). We time them all, and check each
answer against Cvxium's.

```python
import cvxpy as cp
import scipy.sparse

P = scipy.sparse.diags(
    [main_diagonal, superdiagonal, superdiagonal], [0, 1, -1], format="csc"
)
y_parameter = cp.Parameter(M)
x_cvxpy = cp.Variable(M)
steel_man = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x_cvxpy, cp.psd_wrap(P)) - y_parameter @ x_cvxpy),
    [x_cvxpy >= 0],
)


def solve_with_cvxpy(solver: str, readings: np.ndarray) -> float:
    """Solve the steel-manned problem on new readings; return seconds."""
    y_parameter.value = readings
    start = time.perf_counter()
    steel_man.solve(solver=solver, warm_start=False)
    return time.perf_counter() - start


cvxpy_seconds = {}
for solver in ["CLARABEL", "PIQP", "QOCO", "OSQP", "SCS"]:
    solve_with_cvxpy(solver, y + 0.01)  # compile, and warm up the solver
    cvxpy_seconds[solver] = solve_with_cvxpy(solver, y)
    print(f"{solver}: {cvxpy_seconds[solver]:.3f} s")
cvxpy_value = steel_man.value  # SCS's, the last solved
```

| Solver   | Method         | Solve time (s) | Iterations |
|----------|----------------|---------------:|-----------:|
| QOCO     | interior point |           0.17 |         11 |
| PIQP     | interior point |           0.19 |         17 |
| OSQP     | ADMM           |           0.21 |         75 |
| SCS      | ADMM           |           0.28 |         75 |
| Clarabel | interior point |           0.50 |         17 |

A first solve, including compilation, adds 0.3 to 0.4 seconds to each.
The fastest configuration, QOCO, is 50 times faster than the naive
formulation. (PIQP and QOCO are optional; install them with
`pip install piqp qoco`.) The first-order methods' answers are also less accurate:
OSQP's estimate dips to -3 × 10⁻⁵ kW, a small violation of the
constraint, where every interior-point method's estimate stays
strictly positive.

## The structure Cvxium exploits

Every solver above that uses an interior-point method replaces the
constraint with a barrier, or with complementarity conditions, and
takes Newton steps. For the primal-dual method, the Newton system is

$$
\big(P + \operatorname{diag}(\lambda_i / x_i)\big)\, \Delta x = -r,
$$

where λ_i ≥ 0 is the multiplier on x_i ≥ 0 and r is a residual. A
diagonal added to a tridiagonal matrix is still tridiagonal, so each
step is a tridiagonal factorization and solve: O(M) time, with no
fill-in. A sparse solver arrives at the same asymptotic cost, but it
builds the larger KKT system in the constraint multipliers too, orders
it, and runs a general sparse LDL^T factorization at every step.
Cvxium skips all of that: `QuadraticObjective` reports P as the
`BandedSystem` we built, `BoundsConstraintLayer` reports a
`DiagonalSystem`, and their sum is a `BandedSystem`, which a
tridiagonal matrix factors with LAPACK's dedicated routine (`dpttrf`).

## Solving with Cvxium

*Example 1, cont'd.* The problem is a composition of two library
components. We drop the constant (1/2) y^T y, as the cvxpy problem
did:

```python
from cvxium import (
    BoundsConstraintLayer,
    PrimalDualSolver,
    Problem,
    QuadraticObjective,
    SolveStatus,
)

problem = Problem.minimize(
    QuadraticObjective(c=-y, hessian_contribution=P_system)
).subject_to(ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)])

x0 = np.ones(M)  # strictly feasible: skips Phase I
problem.solve(solver=PrimalDualSolver(), x0=x0)  # warm up, as for cvxpy
start = time.perf_counter()
result = problem.solve(solver=PrimalDualSolver(), x0=x0)
cvxium_seconds = time.perf_counter() - start

x_star = result.solution
assert result.status == SolveStatus.OPTIMAL
assert np.all(x_star > 0.0)
assert abs(result.objective_value - cvxpy_value) <= 1e-6 * abs(cvxpy_value)
fastest = min(cvxpy_seconds, key=cvxpy_seconds.__getitem__)
print(
    f"Cvxium: {cvxium_seconds:.3f} s ({result.n_iterations} iterations), "
    f"{cvxpy_seconds[fastest] / cvxium_seconds:.1f}x faster than {fastest}, "
    f"{cvxpy_seconds['CLARABEL'] / cvxium_seconds:.1f}x faster than Clarabel"
)
```

Cvxium takes 14 iterations and about 0.08 seconds: about 2 times
faster than QOCO and PIQP, 6 times faster than Clarabel, and over 100
times faster than the naive formulation. Its duality-gap bound is
below 10⁻⁶, in the objective's units, and it agrees with the cvxpy
solvers on the optimal value to within their tolerances.

The constraint does real work. The estimate rests on zero for about
8% of the minutes, the dark middle of each night; elsewhere at night
it hovers just above zero, where the smoothing penalty outweighs the
pull of a few positive readings.

```python
at_zero = x_star < 1e-3
print(f"Estimate at zero {np.mean(at_zero):.0%} of the time")
```

## Getting the settings right

The defaults serve this problem well. Three points, which also
appeared in the guide to large linear programs, are worth checking.

**Use the primal-dual solver.** `BarrierSolver` solves the same
problem, but it centers to high accuracy at each value of the barrier
parameter and takes several times as many Newton steps. It is about
nine times slower than `PrimalDualSolver` here, and at M = 100,000 it
stops at `SolveStatus.ACCEPTABLE`, short of the default tolerance.

**Mind the units of the tolerance.** The stopping tolerance,
`OuterSettings.tolerance`, bounds the duality gap in the objective's
units. Here the optimal objective is of order 10⁵ (it is a sum over
100,000 minutes), so the default 10⁻⁶ asks for a relative accuracy
near 10⁻¹¹, far beyond what the cvxpy solvers' default tolerances
deliver. It is affordable, since the iterations converge quickly at
the end, but not free: loosening the tolerance to 1.0 (a relative
accuracy near 10⁻⁵) cuts the iterations from 14 to 10 and the time by
about a quarter. The timings in this guide keep the default, so
Cvxium's answers are the more accurate ones in every comparison.

**Start inside.** Any positive vector is strictly feasible for x ≥ 0,
so `x0 = np.ones(M)` skips Phase I.

## How the comparison scales

The same comparison at three sizes, timing the fastest of three solves
of new readings with each solver (and, for cvxpy, of three problems
compiled once). The fastest cvxpy solver is QOCO or PIQP at every
size; "first solve" is cvxpy's fastest time including compilation.

|         M | Cvxium (s) | fastest cvxpy (s) | Clarabel (s) | first solve (s) | vs. fastest | vs. Clarabel | vs. first solve |
|----------:|-----------:|------------------:|-------------:|----------------:|------------:|-------------:|----------------:|
|    10,000 |      0.010 |             0.017 |        0.046 |           0.041 |        1.7× |         4.6× |              4× |
|   100,000 |      0.078 |             0.173 |        0.500 |           0.507 |        2.2× |         6.4× |              7× |
| 1,000,000 |       1.17 |              1.99 |         5.33 |            13.5 |        1.7× |         4.6× |             12× |

Everything grows linearly in M, as the analysis predicts for every
interior-point solver here, so the ratios hold roughly steady: Cvxium
is about twice as fast as the best sparse QP solver and about five
times as fast as Clarabel. What does not hold steady is compilation:
at M = 1,000,000 cvxpy takes over 11 seconds to compile the problem
before any solver runs, so a one-off solve is twelve times slower than
Cvxium's.

The comparison has limits worth naming. The sparse solvers handle any
sparsity pattern, and they win wherever the pattern is irregular, or
where Cvxium has no `MatrixSystem` for it. And a factor of two over
QOCO or PIQP is a constant, not a change in complexity: those solvers
exploit the same tridiagonal structure, only through general
machinery. Cvxium's advantage is that it never pays for generality the
problem does not need.

## Summary

A quadratic program whose matrix is tridiagonal, subject to
nonnegativity, has a Newton system that stays tridiagonal at every
step. Cvxium assembles it from `QuadraticObjective` with a
`BandedSystem` Hessian and `BoundsConstraintLayer`, factors it with
LAPACK's tridiagonal routine, and solves the power-meter problem in
about 0.08 seconds at M = 100,000. We gave cvxpy every advantage we
could find, a direct quadratic form, one-time compilation, no warm
starts, and five solvers, and its fastest configuration remains about
twice as slow; Clarabel is six times slower, and the formulation
most of us would write first is a hundred times slower.

To get that speed on a problem of your own:

1. Build the quadratic's matrix in banded storage and pass it to
   `QuadraticObjective` as a `BandedSystem`; any band structure
   carries through, and bandwidth 1 gets the fastest path.
2. State bounds with `BoundsConstraintLayer`, whose diagonal
   contribution leaves the band intact.
3. Solve with `PrimalDualSolver`, from a strictly feasible `x0`.
4. When benchmarking against cvxpy, write the quadratic with
   `cp.quad_form` and a sparse matrix, compile once with parameters,
   and try more than one solver.
