# Large Linear Programs

A linear program with many variables and few constraints is a common
shape in applied optimization: allocate a budget across thousands of
options, subject to a handful of requirements. General LP solvers see
only a sparse matrix, which is the right view when nothing more is
known. Here more is known: the constraint matrix has only p dense
rows, and an interior-point method can exploit that to make every
iteration cost O(p² M).

This guide works one example from start to finish. We state a media
plan as a linear program, solve it with scipy's `linprog` (HiGHS) to
establish ground truth and a time to beat, explain the structure
Cvxium exploits, and then solve it with Cvxium: about 20 times faster
than `linprog`'s default method, and about 4 times faster than the
fastest method HiGHS offers. Three settings decide whether that
speedup appears at all: the choice of solver, the scale of the
objective, and the starting point. We cover each, read the shadow
prices off the solution, and close with how the speedup grows with
problem size. Every `python` block runs, in order, as one script.

## The media plan

*Example 1.* An advertiser buys impressions from M = 20,000 placements
(sites, apps, time slots). Placement i sells up to u_i thousand
impressions at c_i dollars per thousand. The advertiser cares about
p = 50 audience segments, and a fraction R_ki of placement i's
impressions reaches segment k. Buying x_i thousand impressions from
each placement delivers (R x)_k thousand impressions to segment k, and
the campaign must deliver at least r_k to each. The cheapest plan
solves

$$
\begin{array}{ll}
\mbox{minimize}   & c^T x \\
\mbox{subject to} & R x \succeq r \\
                  & 0 \preceq x \preceq u.
\end{array}
$$

There are M variables, p dense inequality rows, and 2M bounds. Each
placement's audience concentrates on a few segments, so we draw each
column of R from a Dirichlet distribution with parameter 0.5, and we
ask for 30% of the impressions available to each segment.

```python
import time

import numpy as np
import scipy.optimize

M, p = 20_000, 50
rng = np.random.default_rng(1)
R = rng.dirichlet(np.full(p, 0.5), size=M).T  # shape (p, M); columns sum to 1
u = rng.uniform(50.0, 150.0, M)  # capacity, thousands of impressions
c = rng.lognormal(np.log(5.0), 0.5, M)  # dollars per thousand impressions
r = 0.3 * (R @ u)  # thousands of impressions required per segment
```

## A baseline with scipy

`scipy.optimize.linprog` takes inequalities as A_ub x ≤ b_ub, so we
negate the reach constraints. Its default method, `"highs"`, lets HiGHS
choose an algorithm, and on this problem it chooses the dual simplex.
HiGHS's interior-point method, `"highs-ipm"`, is faster here, so to
steel-man the alternative we time both.

```python
bounds = np.column_stack([np.zeros(M), u])
c_millions = c / 1e6  # see "Scale the objective" below

start = time.perf_counter()
reference = scipy.optimize.linprog(c_millions, A_ub=-R, b_ub=-r, bounds=bounds)
scipy_seconds = time.perf_counter() - start
assert reference.status == 0

start = time.perf_counter()
reference_ipm = scipy.optimize.linprog(
    c_millions, A_ub=-R, b_ub=-r, bounds=bounds, method="highs-ipm"
)
scipy_ipm_seconds = time.perf_counter() - start
assert reference_ipm.status == 0
print(
    f"HiGHS: ${reference.fun:.4f}M in {scipy_seconds:.2f} s (default), "
    f"{scipy_ipm_seconds:.2f} s (interior point)"
)
```

The cheapest plan costs about $1.74 million. The default method finds
it in about 4.3 seconds, the interior-point method in about 0.9. (All timings in this guide come from an Apple M2 Pro
with scipy 1.17 and NumPy 2.4; expect different absolute numbers
elsewhere, but similar ratios.)

## The structure Cvxium exploits

An interior-point method replaces the inequalities with a logarithmic
barrier and takes Newton steps on

$$
t\, c^T x - \sum_{i=1}^M \log x_i - \sum_{i=1}^M \log(u_i - x_i)
         - \sum_{k=1}^p \log\big((R x)_k - r_k\big).
$$

The objective is linear, so it contributes nothing to the Hessian. The
bounds touch one variable each, so they contribute a diagonal matrix
D. The reach constraints contribute R^T S^{-2} R, where
S = diag(R x - r), a matrix of rank p. The Newton system is therefore

$$
\big(D + R^T S^{-2} R\big)\, \Delta x = -g,
$$

a diagonal plus a rank-p update. A dense solve would cost O(M³), about
8 × 10¹² operations at M = 20,000. The Woodbury identity instead
reduces it to a p × p system:

$$
\big(D + R^T S^{-2} R\big)^{-1}
  = D^{-1} - D^{-1} R^T \big(S^2 + R D^{-1} R^T\big)^{-1} R D^{-1}.
$$

Forming R D^{-1} R^T costs O(p² M), factoring it O(p³), and everything
else O(p M). The primal-dual method, which we use below, solves a
system with the same pattern, only with D and S weighted by the dual
variables. Interior-point methods need a few dozen iterations, nearly
independent of M, so the whole solve costs O(p² M) times a small
constant.

Cvxium assembles this structure without our help. `LinearObjective`
reports a `ZeroSystem` as its Hessian, `BoundsConstraintLayer` a
`DiagonalSystem`, and `LinearInequalityConstraintLayer` a low-rank
update; summing them yields a `LowRankUpdatedSystem` over a diagonal
base, whose `solve` applies the Woodbury identity.

## Solving with Cvxium

*Example 1, cont'd.* The problem is a composition of three library
components:

```python
from cvxium import (
    BoundsConstraintLayer,
    LinearInequalityConstraintLayer,
    LinearObjective,
    PrimalDualSolver,
    Problem,
    SolveStatus,
)

problem = Problem.minimize(LinearObjective(c_millions)).subject_to(
    ineq=[
        BoundsConstraintLayer(lb=0.0, ub=u),
        LinearInequalityConstraintLayer(A=R, cl=r),
    ]
)

x0 = 0.5 * u  # buys half of every placement: strictly feasible
start = time.perf_counter()
result = problem.solve(solver=PrimalDualSolver(), x0=x0)
cvxium_seconds = time.perf_counter() - start

x = result.solution
assert result.status == SolveStatus.OPTIMAL
assert np.all(R @ x >= r) and np.all(x >= 0.0) and np.all(x <= u)
assert abs(result.objective_value - reference.fun) <= 1e-5
print(
    f"Cvxium: ${result.objective_value:.4f}M in {cvxium_seconds:.2f} s "
    f"({result.n_iterations} iterations), "
    f"{scipy_seconds / cvxium_seconds:.0f}x faster than linprog's default, "
    f"{scipy_ipm_seconds / cvxium_seconds:.1f}x faster than HiGHS's IPM"
)
```

Cvxium takes 19 iterations and about 0.2 seconds: roughly 20 times
faster than `linprog`'s default and 4 times faster than HiGHS's
interior-point method, and it agrees with both on the optimal cost to
better than one part in a million. The solution reports a duality-gap bound,
`result.suboptimality`, of about 6 × 10⁻⁷: the plan costs at most
sixty cents more than the cheapest possible plan.

The plans themselves agree less closely than their costs. HiGHS's
simplex returns a vertex: with 50 binding reach constraints, at most
50 placements are bought partially, and every other placement is
bought fully or not at all. An interior-point method stops just inside
the feasible region, so Cvxium's plan sits a duality gap's worth from
that vertex, with each placement's purchase near, but not exactly at,
its bound. Counting a placement as fractional only when it sits more
than 0.1% of its capacity from either bound finds 51, nearly the
vertex's 50; where the optimum is not unique, the two
methods may also settle on different optimal plans of equal cost.

## Shadow prices

The multipliers on the reach constraints answer the question the
advertiser will ask next: what does one more thousand impressions in
segment k cost? `inequality_multipliers` stacks the layers' multipliers
in order, so the last p belong to the reach layer. HiGHS reports the
same quantities as `ineqlin.marginals`, negated because we negated the
constraints.

```python
shadow_price = 1e6 * result.inequality_multipliers[-p:]  # $ per thousand
np.testing.assert_allclose(
    shadow_price, -1e6 * reference.ineqlin.marginals, rtol=1e-3
)
cheapest, dearest = np.argmin(shadow_price), np.argmax(shadow_price)
print(
    f"An extra thousand impressions costs ${shadow_price[cheapest]:.2f} in "
    f"segment {cheapest} and ${shadow_price[dearest]:.2f} in segment {dearest}"
)
```

Every segment's constraint binds, and the marginal costs range from
about $1.70 to about $5.50 per thousand impressions. These
numbers often matter more than the plan itself: they tell the
advertiser which targets are expensive to raise and which are nearly
free.

## Getting the settings right

The speedup above depends on three choices. Each default is reasonable
for some problems, and each is wrong for this one.

### Use the primal-dual solver

`BarrierSolver`, the more general default elsewhere in Cvxium, solves a
sequence of centering problems, one per value of the barrier parameter
t, each to high accuracy. On a linear program that costs far more
Newton steps than the primal-dual method, which updates t at every
step:

```python
from cvxium import BarrierSolver

start = time.perf_counter()
barrier_result = problem.solve(solver=BarrierSolver(), x0=x0)
barrier_seconds = time.perf_counter() - start
newton_steps = sum(barrier_result.barrier_diagnostics.inner_nits)
assert abs(barrier_result.objective_value - reference.fun) <= 1e-5
print(f"Barrier: {newton_steps} Newton steps in {barrier_seconds:.2f} s")
```

The barrier method takes 88 Newton steps over 11 centering problems,
about 1.1 seconds: faster than `linprog`'s default, slower than
HiGHS's interior-point method, and five times slower than the
primal-dual method. For an LP, start with `PrimalDualSolver`.

### Scale the objective

Cvxium's stopping rule is absolute: `OuterSettings.tolerance`, 10⁻⁶ by
default, bounds the duality gap in the objective's own units. We
expressed costs in millions of dollars, so the default asks for a plan
within a dollar of optimal. Had we left them in dollars, the default
would ask for a millionth of a dollar on a $1.74 million plan, a
relative accuracy of 10⁻¹², beyond what double precision can deliver
through the barrier. The primal-dual solver then grinds to its
iteration cap: in our measurement it ran all 1,000 iterations, took
about 12 seconds, and returned `SolveStatus.ACCEPTABLE`.

Either rescale the objective so its optimal value is of order one, or
state the tolerance in the objective's units:

```python
from cvxium import OptimizationSettings, OuterSettings

dollars = Problem.minimize(LinearObjective(c)).subject_to(
    ineq=[
        BoundsConstraintLayer(lb=0.0, ub=u),
        LinearInequalityConstraintLayer(A=R, cl=r),
    ]
)
within_a_dollar = OptimizationSettings(outer=OuterSettings(tolerance=1.0))
dollar_result = dollars.solve(solver=PrimalDualSolver(within_a_dollar), x0=x0)
assert dollar_result.status == SolveStatus.OPTIMAL
assert abs(dollar_result.objective_value - 1e6 * reference.fun) <= 10.0
```

This solve takes the same 19 or so iterations as before.

### Start from a strictly feasible point

An interior-point method needs a starting point strictly inside every
constraint. Without one, Cvxium first solves a Phase I problem to find
it, which here adds about a quarter to the solve time. Problems like
this one usually have an obvious interior point: buying half of every
placement delivers 50% of each segment's available impressions,
comfortably above the 30% target, and stays strictly inside the
bounds. Passing it as `x0` skips Phase I. (Cvxium checks `x0` and
falls back to Phase I if it is not strictly feasible.)

```python
start = time.perf_counter()
phase1_result = problem.solve(solver=PrimalDualSolver())
phase1_seconds = time.perf_counter() - start
assert abs(phase1_result.objective_value - reference.fun) <= 1e-5
print(f"Without x0: {phase1_seconds:.2f} s; with x0: {cvxium_seconds:.2f} s")
```

## How the speedup scales

The table compares `linprog`'s default method, HiGHS's interior-point
method, and Cvxium (`PrimalDualSolver` from x0 = u/2) on media plans
generated as above, timing the fastest of three runs of each. The last
two columns are Cvxium's speedups.

|      M |   p | default (s) | HiGHS IPM (s) | Cvxium (s) | iterations | vs. default | vs. IPM |
|-------:|----:|------------:|--------------:|-----------:|-----------:|------------:|--------:|
|  2,000 |  50 |       0.094 |         0.076 |      0.023 |         18 |          4× |    3.3× |
|  5,000 |  50 |        0.43 |          0.19 |      0.045 |         19 |         10× |    4.3× |
| 20,000 |  50 |         4.2 |          0.92 |       0.21 |         19 |         20× |    4.4× |
| 50,000 |  50 |        19.2 |          2.44 |       0.57 |         20 |         34× |    4.3× |
| 20,000 |  10 |        0.17 |          0.19 |      0.091 |         17 |          2× |    2.1× |
| 20,000 | 100 |        28.1 |          2.14 |       0.43 |         22 |         65× |    5.0× |
| 20,000 | 200 |         265 |          4.43 |       1.14 |         28 |        233× |    3.9× |

Cvxium's iteration count stays between 17 and 28 across every size,
and its time grows linearly in M, as the analysis above predicts. The
default simplex grows much faster, in both M and p, because the number
of pivots grows with the problem as well as the cost of each one; at
p = 200 it takes over four minutes. HiGHS's interior-point method
scales like Cvxium's, so against it the advantage holds steady at a
factor of about 4, less when p is small and fixed costs dominate.

The default deserves the attention because it is what `linprog` runs
unless told otherwise, and nothing about this problem warns that the
simplex will be slow. Whichever solver we compare with, the lesson is
the same: time the alternatives on the problem at hand.

The comparison has limits worth naming. HiGHS solves problems Cvxium
does not, including sparse LPs with many constraints, where a
diagonal-plus-low-rank Hessian is no longer cheap, and mixed-integer
programs; it also returns a vertex solution, which some applications
need. The advantage here comes from the shape of the problem: many
variables, few dense constraints, and bounds.

## Summary

A linear program with M variables, p dense constraints, and bounds
has a barrier Hessian that is a diagonal plus a rank-p update, so each
interior-point iteration costs O(p² M). Cvxium assembles that
structure from library components, `LinearObjective`,
`BoundsConstraintLayer`, and `LinearInequalityConstraintLayer`, with no
code of our own, and on the media plan solves the problem about 20
times faster than `linprog`'s default and 4 times faster than HiGHS's
interior-point method, while reporting the same optimal cost and the
same shadow prices.

To get that speed on a problem of your own:

1. Express the problem with library layers: bounds as a
   `BoundsConstraintLayer`, dense rows as a
   `LinearInequalityConstraintLayer` (or an `EqualityConstraintLayer`
   for equalities).
2. Solve it with `PrimalDualSolver`.
3. Scale the objective so its optimal value is of order one, or set
   `OuterSettings(tolerance=...)` in the objective's units.
4. Pass a strictly feasible `x0` when one is obvious.
5. Check the answer against `linprog` once, and read the shadow
   prices from `inequality_multipliers`.
