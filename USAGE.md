# Implementing Fast Solvers in `Cvxium`

This document describes how to implement a new solver using the
building blocks Cvxium provides. The intended audience is AI agents,
though humans implementing new problem types may find the same
guidance useful.

## 0. Start with a test case

Before writing any solver code, establish ground truth, e.g. with
`scipy.optimize.minimize`. This also gives a "time to beat", after
all, if Cvxium isn't any faster than more popular libraries, why
bother? Spend some time making this ground truth calculation as fast
as possible, e.g. by passing an explicit Hessian callable. That way we
are steel-manning the more popular libraries.

For example, we can solve:
```
minimize    x^T Q x + c^T x
subject to  A x = b
x >= 0
```
where Q is symmetric positive semi-definite, via the `minimize`
function in scipy's optimize module:
```python
scipy_result = scipy.optimize.minimize(
    fun=lambda x: float(x @ Q @ x + c @ x),
    x0=x0_feas,
    method="trust-constr",
    jac=lambda x: 2.0 * Q @ x + c,
    hess=lambda x: 2.0 * Q,
    constraints=scipy.optimize.LinearConstraint(A, b, b),
    bounds=scipy.optimize.Bounds(lb=0),
    options={"gtol": 1e-10, "maxiter": 2000},
)
if not scipy_result.success:
    pytest.skip(f"scipy reference solver failed: {scipy_result.message}")
```

After implementing the solver using Cvxium, compare the solution
against `scipy_result.x`. Set tolerances that are meaningful for your
application. Cvxium will likely differ in the 8th decimal point, but
this level of precision is rarely important for many practical
applications. Verify the constraints are respected.

Time both solvers and print the ratio. This makes performance
regressions visible in CI without requiring a separate benchmark
suite.

## 1. Write out the math before writing any code

Work through the following four items in order. Mistakes in the math
propagate invisibly into the code; catching them on paper is much
cheaper.

### 1a. Write problem in standard form and sanity check convexity.

Standard form is:
```
minimize    f0(x)
subject to  A x = b
fi(x) <= 0, i=1, ..., n
```

Note the direction of the inequality on that last line: inequality
constraints are supposed to be less than 0. Maximizations can be
turned into minimizations by multiplying f0 by -1. Verify that f0 and
fi are all convex functions of x.

### 1b. Lagrangian

Introduce one multiplier per constraint type:
- `lmbda >= 0` for each inequality `fi(x) <= 0`
- `nu` for each equality `a_i^T x = b_i`

For the QP in standard form:
```
minimize    x^T Q x + c^T x
subject to  A x = b
-x <= 0,
```
the Lagrangian is:
```
L(x, lmbda, nu) = x^T Q x + c^T x + lmbda^T (-x) + nu^T (A x - b)
```

### 1c. Dual function

The Lagrangian dual function is:
```
g(lmbda, nu) = inf_x L(x, lmbda, nu).
```

Evaluating the dual function is itself an optimization problem, but
it's an unconstrained problem so it's sometimes possible to evaluate
it in closed form. It's often instructive to at least try to identify
such a closed form solution. Set `grad_x L = 0` and try to solve for
`x*` in terms of the multipliers. For the QP:

```
2 Q x + c - lmbda + A^T nu = 0
x* = -(1/2) Q^{-1} (c - lmbda + A^T nu)
```

Substituting back gives the dual function:


```
g(lmbda, nu) = -(1/4) v^T Q^{-1} v - nu^T b,
```
where `v = c - lmbda + A^T nu`.

The dual provides a lower bound on the primal objective at every
iteration, giving a **duality gap** that can certify convergence or
infeasibility. Implement it as the `evaluate_dual` method.

It won't always be possible to evaluate the dual function for general
lmbda, nu, but the dual function evaluated at the optimal x, lmbda, nu
is simply the Lagrangian function evaluated there. The dual function
doesn't provide a valid lower bound in that case, but will satisfy the
code linter.

### 1d. Barrier problem (inequality → objective)

Write out the barrier problem, which incorporates the inequality
constraints into the objective:
```
minimize    ft(x) := t * f0(x) - sum_i log(-fi(x))
subject to  A x = b
```

Write out the gradient and Hessian of ft. Pay especial attention to
any special properties of the Hessian. A common pattern is that f0 is
additive in the elements of x (e.g. `f0(x) = f01(x1) + f02(x2) + ...`)
and the constraints apply to individual elements of x (e.g. `fi(x) = fi(xi)`).
This structure leads to diagonal Hessians. In many practical
applications, the Hessian is either diagonal or closely related to
diagonal, such as banded, diagonal plus a low rank matrix, or having
an arrow sparsity pattern (non-zero only along the diagonal, and on
the last row/column).

## 2. Initial implementation of key methods

Using Cvxium means subclassing one of the base solvers. Ask: is this
an optimization or feasibility problem? Optimization problems minimize
some objective, with or without constraints. Feasibility problems find
a solution that satisfies constraints, but don't have an objective.
What kind of constraints are there: equality, inequality, both, or
neither? Look in the table below and create a class that inherits from
the class(es) specified.

| Constraints           | Optimization                                                             | Feasibility                                                                   |
|-----------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| None                  | UnconstrainedNewtonSolver                                                | n/a                                                                           |
| Equality              | EqualityConstrainedNewtonSolver                                          | EqualitySolver                                                                |
| Inequality            | InteriorPointMethodSolver                                                | FeasibilityInteriorPointSolver                                                |
| Equality + Inequality | InteriorPointMethodSolver + EqualityConstrainedInteriorPointMethodSolver | FeasibilityInteriorPointSolver + EqualityConstrainedInteriorPointMethodSolver |

Depending on what kind of solver you're creating, you'll need to
implement some or all of:
- evaluate_objective: f0(x)
- constraints: fi(x)
- num_eq_constraints
- num_ineq_constraints
- gradient: ∇f0(x)
- grad_constraints: ∇fi(x)
- hessian_multiply: ∇²ft(x) @ y
- calculate_newton_step: either the solution to ∇²ft(x) @ dx = -∇ft(x)
  (no equality constraints) or the solution to the KKT system (yes
  equality constraints):
      _           _   _  _     _        _
     | ∇²ft(x)   A^T | | dx |   | - ∇ft(x) |
     |   A        0  | | nu | = |     0    |
      -             -   -  -     -        -
  - For problems without equality constraints, return 0 as nu.
  - Also return the Newton decrement, dx^T ∇²ft(x) dx.
- evaluate_dual (from step 1c)

We recommend starting with simple implementations of these that can be
verified as "obviously correct", and update the test case from step 0
to compare to ground truth. Don't worry about making these methods
fast; get a correct implementation first, and then streamline. Time
the Cvxium code: it will probably be slower than scipy, but that's
before we have incorporated any special structure that makes Cvxium
shine.

### 2a. Dude, you need a FeasibilitySolver.
For optimization problems with constraints, you need a feasibility
solver that can find a feasible point. It's turtles all the way down;
Cvxium can be used to create fast feasibility solvers. Still, you may want
to implement the stack in pieces, using scipy.optimize for the feasibility
solver as an initial implementation, get good test cases in place, and
then implement a Cvxium version. It's often helpful to apply
constraints in layers, to chain feasibility solvers.

## 3. Construction-time precomputation

Expensive work that does not depend on the iterate `x` or barrier parameter
`t` should be done once in `__init__`, not per Newton step.

| Precomputation                    | Where                                       | Cost   |
|-----------------------------------|---------------------------------------------|--------|
| Matrix factorizations: `U_r, s_r` | `__init__`                                  | O(p²n) |
| Feasibility point                 | `__init__` (via `EqualityWithBoundsSolver`) | O(p²n) |

## 4. Exploit special structure in the...

### 4a. Constraint-gradient matrix
Cvxium typically interacts with the constraint gradient matrix,
∇fi(x), via matrix-vector multiplies, either ∇fi(x) y or y^T ∇fi(x).
The base classes provide naive implementations to get you started, but
often this matrix has special structure that permits linear-time
matrix-vector multiplies (which in general are quadratic-time
operations).

More generally, examine the Jacobian of your constraints analytically.
Sparse or structured Jacobians (diagonal, banded, rank-deficient)
should always be exploited here rather than stored as dense matrices.

### 4b. Hessian-vector multiply

Often H = ∇²ft(x) has some special structure that means evaluating
H @ y is a linear-time operation, rather than quadratic (in fact, if
this is not true, Cvxium will probably not be useful to you).

Here are some examples:
- H = diag(eta) -> H @ y = eta * y (elementwise multiply)
- H = diag(eta) + kappa @ kappa^T, where kappa is a vector
  - Then H is "diagonal plus rank-1), and
    H @ y = eta * y + (kappa^T y) * kappa (elementwise multiply, dot
    product, scalar multiply, vector add).

The same small number of patterns (diagonal, diagonal plus low rank,
block pattern) show up over and over.

The `hessian_multiply` method should work with matrices, too, e.g.
H @ Y, where Y is a matrix. This avoids the need to run multiple
multiplies in a for loop.

### 4c. Newton step calculation
As in the previous section, special structure in the Hessian often
permits calculating the Newton step in linear time, rather than the
naive n³ time. If this is *not* true, Cvxium probably isn't going to
deliver any value to you. The numerical_helpers.py file contains
helpers for the major patterns.

As in the previous section, the Hessian solve should be able to handle
multiple right-hand-sides, e.g. solve H * X = B, where B is a matrix.

For problems with equality constraints, we need to solve the KKT
system:
      _             _   _  _     _        _
     | ∇²ft(x)   A^T | | dx |   | - ∇ft(x) |
     |   A        0  | | nu | = |     0    |
      -             -   -  -     -        -

This is just a block structure. The `solve_kkt_system` function in
`numerical_helpers.py` is tailor-made to exploit structure in H when
solving this system. It will be most useful when the number of
equality constraints (represented by A) is small relative to the
problem dimension.

### 4d. Backtracking line search

The backtracking line search identifies a step modifier s, such that
x + s * dx is feasible and represents a "good enough" improvement over
x. The `btls_keep_feasible` method addresses the "feasible" part.
Equality constraints are automatically feasible for any s (assuming
the Newton step was calculated correctly). When inequality constraints
are present, a naive implementation just keeps reducing s until the
constraints hold. It is often possible to calculate analytically the
largest s (<= 1.0) for which x + s * dx is feasible.

## 5. Initialize the barrier parameter to match the problem
The key to fast solvers is two-fold: (1) make Newton steps faster; (2)
use fewer Newton steps. Initializing the barrier parameter, t, at a
higher value results in fewer Newton steps but can prevent the
algorithm from converging.

A strategy tailored to the specific problem can result in fast but
stable solvers.

## 6. Caching expensive constraint evaluations

When `constraints(x)` is expensive — because it calls a model function
like a log-likelihood, a simulation, or an iterative procedure — the
centering step can evaluate it many more times than necessary.

Within a single Newton iteration, several framework methods may call
`constraints(x)` at the same point: `calculate_newton_step` (via
`gradient_barrier`), `evaluate_barrier_objective`, and
`inequality_multipliers`. If each call recomputes the model from
scratch, the cost multiplies accordingly.

A simple cache keyed on the iterate `x` prevents this. A byte snapshot
(`x.data.tobytes()`) is a reliable key that detects in-place mutation:

  ```python
  def _ensure_cached(self, x):
      key = x.data.tobytes()
      if key != self._cache_key:
          self._cache_key = key
          self._cache_value = expensive_model_fn(x)
      return self._cache_value
  ```

Then `constraints`, `grad_constraints_multiply`, and any other method
that needs the model output calls `_ensure_cached(x)` instead of
recomputing.

### 6a. Multi-level caching

A subtlety arises when the Newton step needs *more* from the model
than `constraints` does. For example, `constraints` may need only the
log-likelihood, while `calculate_newton_step` also needs its gradient
(score) and Hessian. If these are bundled into a single cache, then
every call to `constraints` — including the repeated calls during the
backtracking line search at trial points `x + s * delta_x` — pays for
gradient and Hessian evaluations that are never used.

The fix is to split the cache into levels: a cheap level that computes
only what `constraints` needs, and a full level that adds the
quantities needed by the Newton step. The cheap level is called by
`constraints`; the full level is called by `calculate_newton_step`.
Since the full level builds on the cheap level, there is no redundant
work at the current iterate. But BTLS trial evaluations — which only
need constraints — avoid computing gradient and Hessian entirely.

As an illustration, a constraint on the binomial log-likelihood
involves a relatively expensive log calculation, while the Newton step
also needs the score and the diagonal of the Hessian. Without
multi-level caching, BTLS trial evaluations computed all three,
roughly doubling total solver time.

More generally, look at the call graph from
`evaluate_barrier_objective` → `constraints` → your model, and ensure
that this path does not trigger computation that is only needed by
`calculate_newton_step`.

## 7. Logging and debugging

Cvxium emits structured log records through Python's standard `logging`
module, so you can observe solver behavior without changing any solver
code.

### 7a. Basic usage: verbose=True

The quickest way to enable output is to pass `verbose=True` in
`OptimizationSettings`. This attaches a console handler to the
`cvxium` logger and produces output that looks like:

```
  Starting IPM (MyQPSolver)
  01 Beginning centering step with t=12.5
    01 Newton step in 1.234 ms; sub-opt 3.2e-02 = 0.5 * 0.253^2
    01 btls_s=0.5000, improvement=1.23e-04, expected improvement=9.87e-05
    02 Newton step in 0.987 ms; sub-opt 4.1e-03 = 0.5 * 0.091^2
    02 btls_s=1.0000, improvement=3.11e-03, expected improvement=2.84e-03
    03 Newton step in 0.991 ms; sub-opt 8.7e-08 = 0.5 * 4.2e-04^2 (quadratic convergence threshold)
    03 btls_s=1.0000, improvement=3.28e-03, expected improvement=3.28e-03
  01 Centering step completed in 3.212 ms
  02 Beginning centering step with t=125.0
  ...
  IPM completed in 31.4 ms (MyQPSolver)
```

Two-space indentation marks outer IPM events (centering steps); four
spaces mark inner Newton/BTLS events. `INFO`-level records cover the
outer loop; `DEBUG`-level records cover the inner loop. With
`verbose=True`, both levels are shown.

To enable only outer-loop visibility:

```python
import logging
logging.getLogger("cvxium").setLevel(logging.INFO)
logging.getLogger("cvxium").addHandler(logging.StreamHandler())
```

### 7b. Writing logs to a file

Attach a `FileHandler` to get output on disk while keeping the console
quiet, or combine both:

```python
import logging

file_handler = logging.FileHandler("solver.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

logging.getLogger("cvxium").setLevel(logging.DEBUG)
logging.getLogger("cvxium").addHandler(file_handler)
```

### 7c. Avoiding noise from other libraries

If you have configured the root logger at DEBUG level (e.g. via
`logging.basicConfig(level=logging.DEBUG)`), third-party libraries
such as scipy may emit their own debug records. Suppress them while
keeping Cvxium verbose by configuring at the specific logger level
rather than at root:

```python
import logging
logging.basicConfig(level=logging.WARNING)          # everything quiet by default
logging.getLogger("cvxium").setLevel(logging.DEBUG) # cvxium at full verbosity
```

### 7d. Collecting per-iteration data with IterationHandler

`IterationHandler` accumulates every log record into a list of dicts.
This is useful for plotting convergence, profiling individual steps, or
writing post-hoc analysis scripts.

```python
import logging
from cvxium import IterationHandler, OptimizationSettings

handler = IterationHandler()
logging.getLogger("cvxium").addHandler(handler)
logging.getLogger("cvxium").setLevel(logging.DEBUG)

# verbose=False so we don't also get console output
solver = MySolver(settings=OptimizationSettings(verbose=False))
solver.solve()

df = handler.to_dataframe()   # requires pandas

# Per-centering-step timing
centering = df[df["event"] == "centering_done"]
print(centering[["outer_nit", "elapsed_ms"]])

# Per-Newton-step suboptimality
newton = df[df["event"] == "newton_step"]
print(newton[["outer_nit", "inner_nit", "elapsed_ms", "suboptimality"]])
```

Available `event` values: `ipm_start`, `ipm_done`, `centering_start`,
`centering_done`, `early_stop`, `phase1_feasible`, `newton_step`,
`btls`, `feasible_initial_guess`, `svd_done`.

The structured extra fields available on each record are:

| Field | Events where present |
|---|---|
| `solver` | `ipm_start`, `ipm_done`, `phase1_feasible` |
| `outer_nit` | all outer-loop and inner-loop events |
| `inner_nit` | `newton_step`, `btls` |
| `elapsed_ms` | `centering_done`, `ipm_done`, `newton_step`, `svd_done` |
| `suboptimality` | `newton_step` |
| `t` | `centering_start` |
| `btls_s` | `btls` |
| `improvement` | `btls` |
| `expected_improvement` | `btls` |

### 7e. What to look for: debugging tips

**Backtracking line search step size after quadratic convergence.**
Once the log marks `(quadratic convergence threshold)`, the Newton
decrement has crossed into the quadratic phase and the step modifier
returned by the backtracking line search should always be `s=1.0000`.
A step shorter than 1 at this point means the Armijo sufficient-decrease
condition is not being satisfied, which indicates a bug — most likely
in `hessian_multiply` or `calculate_newton_step`. Check those against a
naive finite-difference implementation.

**First centering step is slow to converge.**
The first centering step runs with the initial barrier parameter `t`.
If it takes many Newton iterations to converge, `t` is likely too large
for the starting point `x0`: the barrier objective is sharply curved
near the constraint boundaries, and the Hessian is ill-conditioned. Two
levers to try:
- Override `initialize_barrier_parameter` to return a smaller `t` that
  is a better match for the curvature at `x0`.
- Pass a better initial guess `x0` that is farther from the constraint
  boundaries, so that the barrier terms are less dominant.

**Later centering steps are slow to converge.**
Ill-conditioning of the Hessian worsens as `t` grows large, because
the barrier terms shrink and the problem approaches its unconstrained
limit. This is exacerbated when there are many inequality constraints.
If convergence stalls on later centering steps, consider relaxing the
tolerances:

- `outer_tolerance_soft` (default `1e-3`) is the practical stopping
  threshold. The solver tries to do better, but exits gracefully if it
  cannot. Raising this value (e.g. to `1e-2`) gives the solver more
  room to fail gracefully. Many practical applications do not require
  high precision.
- `outer_tolerance` (default `1e-6`) is the aspirational threshold. If
  the solver is exiting early because it cannot meet `outer_tolerance`
  but *can* meet `outer_tolerance_soft`, that is the intended behavior.
  Raising `outer_tolerance` closer to `outer_tolerance_soft` causes the
  solver to stop sooner and avoids the ill-conditioned late steps
  entirely.

## 8. The full checklist for a new solver

1. **Write the Lagrangian.** One multiplier per constraint.
2. **Derive the dual function.** Verify `g <= f0` at feasible points.
3. **Write the barrier problem.** One `-log(-fi(x))` term per inequality.
4. **Compute `H_ft`.** Identify structure.
5. **Implement required methods.** Get test cases in place and perform
   initial timing.
6. **Exploit special structure to speed up key methods.** Use
   functions in numerical_helpers.py. Re-test and re-time.
7. **Initialize the barrier parameter.** Re-test and re-time.
8. **Implement FeasibilitySolver(s) in Cvxium.** Re-test and re-time.
9. **Evaluate full Cvxium implementation against ground truth.** Enjoy
   your orders-of-magnitude speed improvements, especially on large
   problem sizes!
