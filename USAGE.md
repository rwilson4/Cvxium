# Building Fast Solvers with `Cvxium`

This guide describes how to solve a new convex optimization problem
with Cvxium: how to state it, how to supply the pieces Cvxium does not
already have, and how to make the result fast. The intended audience is
AI agents, though humans adding new problem types may find the same
guidance useful.

Cvxium separates the *statement* of a problem from the *algorithm*
that solves it. A `Problem` holds an objective, a list of inequality
constraint layers, and optional equality constraints `A w = b`; a
`Solver` runs an interior-point method on it:

```text
result = (
    Problem.minimize(objective)
    .subject_to(eq=EqualityConstraintLayer(A=A, b=b), ineq=[layer_1, layer_2])
    .solve(solver=BarrierSolver())
)
```

Nothing here requires subclassing a solver, deriving a Newton step, or
writing a Phase I problem. When the objective and constraints are
already in the library (linear and quadratic objectives; bounds,
linear, imbalance, and norm constraints), composing them is the whole
job. When they are not, we write a new `Objective` or `ConstraintLayer`
that reports its value, gradient, and Hessian, and the solvers do the
rest. Speed comes from reporting each Hessian with its *structure*
(diagonal, low rank, banded, arrow) intact, which lets a Newton step on
a problem with a million variables run in linear rather than cubic
time.

The guide threads one example throughout. Sections 0 and 1 set up the
problem and its mathematics; section 2 explains how Cvxium assembles
structure; sections 3 and 4 write a new objective and a new constraint
layer; section 5 solves the problem; sections 6 through 8 tune and
debug the solve; section 9 is a checklist; and sections 10 and 11
cover multi-objective tradeoffs and conic and saddle-point problems.
Every `python` block in this guide runs, in order, as one script.

## 0. Start with a test case

*Example 1.* A survey of M respondents arrives with base weights q_i
(design weights, normalized to sum to one). We want calibrated weights
w that reproduce known population means of p covariates exactly, keep
the imbalance on q further covariates below a tolerance ψ, stay
positive, and otherwise stay as close to the base weights as possible,
measured by Kullback-Leibler divergence. Let A be the (p + 1) × M
matrix whose first row is all ones and whose other rows hold the p
covariates, b the vector [1; population means], B the q × M matrix of
the further covariates, and c their population means.

Before writing any Cvxium code, establish ground truth with a general
solver such as `scipy.optimize.minimize`. This also sets a time to
beat: if Cvxium is not faster than the popular libraries, why bother?
Make the reference as fast as it can reasonably be, for example by
passing explicit Jacobians and Hessians, so the comparison steel-mans
the alternative.

```python
import math
import time

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.sparse

rng = np.random.default_rng(0)
M, p, q = 1_000, 3, 5
X = rng.standard_normal((M, p + q))
q_base = np.full(M, 1.0 / M)
A = np.vstack([np.ones(M), X[:, :p].T])  # shape (p + 1, M)
b = np.concatenate([[1.0], np.full(p, 0.1)])  # shape (p + 1,)
B = X[:, p:].T  # shape (q, M)
c = np.full(q, 0.1)  # shape (q,)
psi = 0.05
lb = 1e-6

start = time.perf_counter()
scipy_result = scipy.optimize.minimize(
    fun=lambda w: float(np.sum(w * np.log(w / q_base) - w + q_base)),
    x0=q_base,
    jac=lambda w: np.log(w / q_base),
    hess=lambda w: scipy.sparse.diags(1.0 / w),
    method="trust-constr",
    constraints=[
        scipy.optimize.LinearConstraint(A, b, b),
        scipy.optimize.NonlinearConstraint(
            lambda w: float(np.sum(np.square(B @ w - c))),
            -np.inf,
            psi**2,
            jac=lambda w: 2.0 * (B.T @ (B @ w - c)),
            hess=lambda w, v: 2.0 * v[0] * (B.T @ B),
        ),
    ],
    bounds=scipy.optimize.Bounds(lb=lb),
    options={"gtol": 1e-10, "maxiter": 5_000},
)
scipy_seconds = time.perf_counter() - start
assert scipy_result.success, scipy_result.message
```

After implementing the problem in Cvxium, compare the solution against
`scipy_result.x`, with tolerances meaningful for the application;
expect agreement to about six digits, which is rarely the binding
concern in practice. Verify that the constraints hold. Time
both solvers and print the ratio, which makes performance regressions
visible in CI without a separate benchmark suite.

## 1. Write out the math before writing any code

Work through the following four items in order. Mistakes in the math
propagate invisibly into the code; catching them on paper is much
cheaper.

### 1a. Write the problem in standard form and check convexity

Standard form is
```text
minimize    f0(w)
subject to  A w = b
            fi(w) <= 0,  i = 1, ..., n.
```
Note the direction of the inequalities: each constraint function must
be at most zero. A maximization is a minimization of -f0, and
`Problem.maximize(objective)` does that negation for us. Verify that f0
and every fi is convex in w.

*Example 1, cont'd.* In standard form,
```text
minimize    f0(w) = sum_i [ w_i log(w_i / q_i) - w_i + q_i ]
subject to  A w = b
            lb - w_i <= 0,                 i = 1, ..., M
            || B w - c ||_2^2 - psi^2 <= 0.
```
The objective is convex, with Hessian diag(1 / w), positive definite on
w > 0; the bounds are affine; and the squared norm is convex. (The
-w_i + q_i terms do not change the minimizer; they make f0 the
generalized KL divergence, which is nonnegative and zero at w = q, a
fact section 6 exploits.)

### 1b. Lagrangian

Introduce one multiplier per constraint: ν for the equalities and
λ_i >= 0 for each inequality. For Example 1, with μ the multiplier of
the imbalance constraint,
```text
L(w, nu, lmbda, mu) = f0(w) + nu^T (A w - b) + lmbda^T (lb - w)
                      + mu (|| B w - c ||^2 - psi^2).
```

### 1c. Dual function

The dual function g(ν, λ) = inf_w L(w, ν, λ) is a lower bound on the
optimal value, which gives the duality gap that certifies convergence
or infeasibility. The solvers evaluate it for us: `BarrierSolver`
minimizes the Lagrangian with a few Newton steps, using the curvature
that each component reports (sections 3 and 4). Deriving it by hand is
still worthwhile, because a closed form exposes the problem's structure
and gives an independent check on the solver's bound.

*Example 1, cont'd.* Without the imbalance constraint (μ = 0), setting
∇_w L = 0 gives log(w_i / q_i) + (A^T ν)_i - λ_i = 0, so
w_i* = q_i exp(λ_i - (A^T ν)_i). Substituting back,
```text
g(nu, lmbda) = sum_i q_i (1 - exp(lmbda_i - (A^T nu)_i)) - b^T nu + lb sum_i lmbda_i.
```
With μ > 0 the term μ ||B w - c||^2 couples the coordinates of w and
there is no closed form, which is exactly the case where the solver's
numerical evaluation earns its keep.

### 1d. Barrier problem

Interior-point methods fold the inequalities into the objective with a
logarithmic barrier,
```text
minimize    ft(w) := t f0(w) + phi(w),   phi(w) = - sum_i log(-fi(w))
subject to  A w = b,
```
and solve a sequence of these problems as t grows. Write out the
Hessian of ft and look hard at its structure. Each constraint
contributes
```text
grad^2 phi_i = (1 / fi^2) grad fi grad fi^T - (1 / fi) grad^2 fi,
```
a rank-one term from its gradient plus a multiple of its own curvature.
A common pattern is that f0 is additive across the elements of w and
most constraints touch one element each, which makes the Hessian
diagonal; the remaining constraints often add a low-rank term, a band,
or a dense border (an arrow pattern).

*Example 1, cont'd.* With f = ||B w - c||^2 - ψ^2 < 0 and
g = 2 B^T (B w - c),
```text
grad^2 ft = t diag(1 / w) + diag(1 / (w - lb)^2)          (diagonal)
            + (-2 / f) B^T B + (1 / f^2) g g^T             (rank q + 1)
```
So the Hessian is diagonal plus a rank-(q + 1) update. A dense solve
costs O(M^3); exploiting the structure, a Newton step costs roughly
O(M (p + q)^2), linear in M.

## 2. How Cvxium assembles problem structure

Section 1d identified the structure on paper. This section explains how
Cvxium carries that structure through the solve, so that when we write
new components (sections 3 and 4) we report it in a form the solver can
use.

### 2a. Every component reports a structured Hessian

Each piece of a `Problem` reports its own second derivatives as a
`MatrixSystem`, an object that knows how to `solve` H x = b and
`multiply` H y in its own structure:
- The objective reports ∇²f0(w) through
  `Objective.hessian_contribution(w)`.
- Each inequality layer reports the weighted sum of its curvature
  atoms
  ```text
  sum_i lmbda_i C_i,   C_i = grad^2 fi - (1 / fi) grad fi grad fi^T,
  ```
  through `ConstraintLayer.hessian_contribution(w, lmbda, shift)`, and
  its curvature alone, sum_i λ_i ∇²f_i, through
  `ConstraintLayer.hessian_curvature(w, lmbda)`.

`Problem.build_hessian_system` sums these contributions,
```text
H = s grad^2 f0(w) + sum_i lmbda_i C_i(w),
```
and one contract serves both interior-point methods. The barrier
method passes s = t and λ_i = -1 / f_i, which turns the sum into
∇²ft from section 1d; the primal-dual method passes s = 1 and its dual
variables. (The two weightings agree on the central path.) The dual
evaluation of section 1c uses `hessian_curvature`, which is why every
layer must declare its curvature even when it is zero.

### 2b. The MatrixSystem catalog

| System                 | Structure                                           | Solve cost                       |
|------------------------|-----------------------------------------------------|----------------------------------|
| `ZeroSystem`           | the zero matrix (an additive identity)               | not solvable on its own          |
| `DiagonalSystem`       | diag(η)                                             | O(M)                             |
| `BandedSystem`         | symmetric banded, bandwidth k (LAPACK band storage) | O(M k^2) factor, O(M k) solve    |
| `DenseSystem`          | a general dense matrix                              | O(M^3) factor                    |
| `BlockDiagonalSystem`  | blocks, each any `MatrixSystem`                     | sum of the block costs           |
| `PermutedSystem`       | another system under a permutation                  | the inner system's cost          |
| `ArrowSystem`          | a core system plus a dense border of k rows/columns | k + 1 core solves, plus O(k^3)   |
| `.low_rank_update(κ, d)` | any system plus κ diag(d) κ^T, rank r            | base solves plus O(M r^2 + r^3)  |

A low-rank update is a method, not a constructor: any system can carry
one, solved through the Woodbury identity, and updates can stack.

### 2c. Summing keeps structure, or refuses

`MatrixSystem.plus` sums two systems without losing structure, under
these rules:
- `ZeroSystem` is the additive identity, and a `DiagonalSystem`
  absorbs into any base (diagonal plus banded is banded, diagonal plus
  arrow is arrow).
- Two bases of the same shape combine into that shape (diagonal plus
  diagonal is diagonal; banded plus banded is banded).
- A low-rank update rides on top of whatever the bases sum to.
- Any other pairing (banded plus dense, say) raises
  `IncompatibleStructureError` rather than silently densifying.

The last rule is deliberate: a silent densification would turn an
O(M) Newton step into an O(M^3) one without a word. When it fires,
re-express one contribution, usually as a low-rank update on a
`ZeroSystem`, which sums with anything.

*Example 1, cont'd.* The objective reports a `DiagonalSystem`, the
bounds layer reports a `DiagonalSystem`, and the imbalance layer of
section 4 reports a `ZeroSystem` carrying a rank-(q + 1) update. Their
sum is a `DiagonalSystem` carrying that update, the structure section
1d predicted. We can check the arithmetic directly:

```python
from cvxium import DiagonalSystem, ZeroSystem

w = rng.uniform(0.5, 1.5, size=M) / M
H = DiagonalSystem(1.0 / w).plus(
    ZeroSystem(M).low_rank_update(B.T, 2.0)
)
y = rng.standard_normal(M)
H_dense = np.diag(1.0 / w) + 2.0 * B.T @ B
np.testing.assert_allclose(H.multiply(y), H_dense @ y, rtol=1e-10)
np.testing.assert_allclose(H.solve(y), np.linalg.solve(H_dense, y), rtol=1e-8)
print(type(H).__name__)  # LowRankUpdatedSystem, on a DiagonalSystem base
```

### 2d. Equality constraints: saddle systems

With equality constraints, each Newton step solves the KKT system
```text
 _          _   _    _     _       _
|  H    A^T  | |  dw  |   | -grad ft |
|  A     0   | |  nu  | = |    0     |
 -          -   -    -     -       -
```
This matrix is indefinite, so it is not a `MatrixSystem`; a
`SaddleSystem` solves it, eliminating H through the Schur complement
A H^{-1} A^T and touching H only through `H.solve`. The
`EqualityConstraintLayer` chooses the saddle system through an injected
`SaddleFactory`:

| Factory                           | When                                                                 |
|-----------------------------------|----------------------------------------------------------------------|
| `SchurComplementSaddleFactory`    | the default: dense A with p small relative to M; O(p^2 M + p^3)      |
| `SparseSaddleFactory`             | sparse A (`scipy.sparse`), never forming the dense p × M matrix      |
| `BandedSaddleFactory`             | sparse A whose rows are ordered so A H^{-1} A^T is banded            |
| `ArrowSaddleFactory(n_border=k)`  | a banded core of rows plus k dense rows that touch many variables    |

For example, `EqualityConstraintLayer(A=A_sparse, b=b,
saddle_factory=SparseSaddleFactory())` keeps a constraint matrix with
millions of rows and columns, but only a few nonzeros per column, in
sparse form throughout.

## 3. Writing an Objective

When the objective is not already in the library (`LinearObjective`,
`QuadraticObjective`, and the wrappers `NegatedObjective` and
`LiftedObjective`), subclass `Objective` and implement:

| Method                       | Returns                                                       |
|------------------------------|---------------------------------------------------------------|
| `dimension` (property)       | M, the length of w                                            |
| `evaluate(w)`                | f0(w)                                                         |
| `gradient(w)`                | ∇f0(w), shape (M,)                                            |
| `hessian_contribution(w)`    | ∇²f0(w) as a `MatrixSystem`, with its structure intact        |

Optional overrides:
- `btls_keep_in_domain(w, dw)`: the largest step that keeps w + α dw
  in the domain of f0 (default inf). The solver's dual evaluation
  minimizes the Lagrangian over unconstrained w, and this bound is the
  only thing keeping those steps inside the domain. (The main iterate
  is kept in the domain by the constraint layers, so an objective with
  a restricted domain needs a layer that enforces it, such as bounds.)
- `is_linear = True` for an affine objective, which lets the barrier
  solver evaluate the dual in closed form.
- `at(w)`: a per-iterate evaluation context, for caching (section 7).

*Example 1, cont'd.* The generalized KL divergence has a diagonal
Hessian and a domain of w > 0:

```python
from cvxium import MatrixSystem, Objective


class KLDivergenceObjective(Objective):
    r"""Generalized KL divergence from base weights.

    f0(w) = \sum_i w_i \log(w_i / q_i) - w_i + q_i, for w > 0.

    """

    def __init__(self, q: npt.NDArray[np.float64]) -> None:
        if np.any(q <= 0.0):
            raise ValueError("Base weights q must be positive.")
        self.q = q

    @property
    def dimension(self) -> int:
        return len(self.q)

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        return float(np.sum(w * np.log(w / self.q) - w + self.q))

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.log(w / self.q)

    def hessian_contribution(self, w: npt.NDArray[np.float64]) -> MatrixSystem:
        return DiagonalSystem(1.0 / w, eta_inverse=w)

    def btls_keep_in_domain(
        self, w: npt.NDArray[np.float64], dw: npt.NDArray[np.float64]
    ) -> float:
        mask = dw < 0
        if not np.any(mask):
            return math.inf
        return float(np.min(-w[mask] / dw[mask]))
```

Passing `eta_inverse` spares the `DiagonalSystem` a division per solve.
Before going further, check the derivatives against finite
differences; a wrong gradient or Hessian is the most common bug, and it
surfaces later as a confusing convergence failure:

```python
objective = KLDivergenceObjective(q_base)
v = rng.standard_normal(M)
eps = 1e-6
fd_grad = (objective.evaluate(w + eps * v) - objective.evaluate(w - eps * v)) / (2 * eps)
np.testing.assert_allclose(
    fd_grad, np.dot(objective.gradient(w), v), rtol=1e-5
)
fd_hess = (objective.gradient(w + eps * v) - objective.gradient(w - eps * v)) / (2 * eps)
np.testing.assert_allclose(
    fd_hess, objective.hessian_contribution(w).multiply(v), rtol=1e-5
)
```

## 4. Writing a ConstraintLayer

A constraint layer is a block of m inequalities f_i(w) <= 0. The
library has bounds (`BoundsConstraintLayer`), general linear
inequalities (`LinearInequalityConstraintLayer`, with banded and arrow
variants), an L∞ imbalance bound (`ImbalanceConstraintLayer`), norm
bounds (`NormConstraintLayer`, `SquaredNormConstraintLayer`), and a
bound on any objective (`ObjectiveConstraintLayer`). When none fits,
write a new one.

### 4a. Slack form or direct form

Every layer subclasses one of two bases, and the choice decides how
Cvxium finds a strictly feasible starting point for it (Phase I):
- `SlackConstraintLayer`: Phase I adds a slack variable s, relaxes the
  layer to f_i(w) <= s, and minimizes s. This works for any layer. All
  of the library's linear layers use it.
- `DirectConstraintLayer`: for a layer with a *single* constraint.
  Phase I minimizes f(w) itself, with no slack variable, so the layer
  doubles as an `Objective` (the base class supplies that face). The
  norm layers and `ObjectiveConstraintLayer` use it.

`SlackConstraintLayer` sets `is_linear = True`, because every slack
layer in the library is affine. A curved slack layer must override it
with `is_linear = False`; otherwise the solver takes the closed-form
dual for linear programs and computes a wrong bound.

### 4b. What to implement

| Method                                         | Returns                                                                 |
|------------------------------------------------|-------------------------------------------------------------------------|
| `dimension` (property)                         | M                                                                       |
| `num_constraints(M)`                           | m (fixed at 1 by `DirectConstraintLayer`)                               |
| `constraints(w, shift=0.0)`                    | f_i(w) - shift, shape (m,); all negative when strictly feasible          |
| `grad_constraints(w)`                          | the Jacobian G, shape (m, M), row i = ∇f_i(w)                           |
| `hessian_contribution(w, lmbda, shift=0.0)`    | sum_i λ_i C_i as a `MatrixSystem`, with f_i the shifted values           |
| `hessian_curvature(w, lmbda)`                  | sum_i λ_i ∇²f_i as a `MatrixSystem` (a `ZeroSystem` for affine layers)   |

The `shift` relaxes every constraint to f_i(w) - shift <= 0; slack-form
Phase I passes its slack variable there. Optional overrides, each worth
writing once the layer is correct:
- `grad_constraints_multiply(w, v)` and
  `grad_constraints_transpose_multiply(w, u)`: G v and G^T u without
  forming G. The defaults form G, which is quadratic in M for a dense
  Jacobian.
- `btls_max_step(w, dw, shift, ds, line_search)`: the largest α with
  f_i(w + α dw) - (shift + α ds) < 0 for every i. The default
  backtracks numerically; a closed form is usually available and is
  both faster and exact (return inf when no constraint binds).
- `is_satisfied(w)`: a cheap strict-feasibility test.
- For direct layers: `interior_point(equality)`, a strictly feasible
  point known in closed form (it lets a feasibility solve skip Phase I),
  and `barrier_parameter_hint`, a floor on the initial barrier
  parameter for the direct-form Phase I.

### 4c. Example: a squared-imbalance layer

*Example 1, cont'd.* The imbalance constraint
f(w) = ||B w - c||^2 - ψ^2 <= 0 is a single constraint, so it is a
direct layer. With r = B w - c, its gradient is g = 2 B^T r and its
curvature is 2 B^T B, so its contribution λ (2 B^T B - (1 / f) g g^T) is
a rank-q update plus a rank-one update, both on a `ZeroSystem`. Along a
step, f(w + α dw) - (shift + α ds) is a quadratic in α, so the step
bound has a closed form. (Cvxium ships `NormConstraintLayer` for the
unsquared bound ||B w - c|| <= φ; we write the squared version to show
the mechanics.)

```python
from cvxium import DirectConstraintLayer, LineSearchSettings


class SquaredImbalanceLayer(DirectConstraintLayer):
    r"""Bound the squared imbalance: \| B w - c \|_2^2 - \psi^2 <= 0."""

    def __init__(
        self, B: npt.NDArray[np.float64], c: npt.NDArray[np.float64], psi: float
    ) -> None:
        if psi <= 0.0:
            raise ValueError(f"psi must be positive (got {psi}).")
        self.B = B
        self.c = c
        self.psi = psi

    @property
    def dimension(self) -> int:
        return self.B.shape[1]

    def constraints(
        self, w: npt.NDArray[np.float64], shift: float = 0.0
    ) -> npt.NDArray[np.float64]:
        r = self.B @ w - self.c
        return np.array([float(np.dot(r, r)) - self.psi**2 - shift])

    def grad_constraints(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r = self.B @ w - self.c
        return (2.0 * (self.B.T @ r)).reshape(1, -1)

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        if lmbda is None:
            raise ValueError("hessian_contribution requires lmbda.")
        lam = float(lmbda[0])
        f = float(self.constraints(w, shift)[0])  # < 0 at a strictly feasible w
        g = self.grad_constraints(w)[0]  # shape (M,)
        # lam * (2 B^T B - (1 / f) g g^T): rank q curvature plus a rank-one term,
        # both with positive weights, on a zero base that sums with anything.
        return (
            ZeroSystem(self.dimension)
            .low_rank_update(self.B.T, 2.0 * lam)
            .low_rank_update(g, -lam / f)
        )

    def hessian_curvature(
        self, w: npt.NDArray[np.float64], lmbda: npt.NDArray[np.float64]
    ) -> MatrixSystem:
        return ZeroSystem(self.dimension).low_rank_update(
            self.B.T, 2.0 * float(lmbda[0])
        )

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        # ||r + alpha u||^2 - psi^2 - shift - alpha ds = a alpha^2 + 2 b alpha + e.
        r = self.B @ w - self.c
        u = self.B @ dw
        a = float(np.dot(u, u))
        b = float(np.dot(r, u)) - 0.5 * ds
        e = float(np.dot(r, r)) - self.psi**2 - shift  # < 0 at a feasible w
        if a <= 0.0:
            return -e / (2.0 * b) if b > 0.0 else math.inf
        # e < 0 < a, so the larger root is positive.
        return (-b + math.sqrt(b * b - a * e)) / a
```

Check the contribution against the formula and the curvature against
finite differences of the gradient:

```python
imbalance = SquaredImbalanceLayer(B, c, psi)
w_bal = scipy_result.x  # a strictly feasible point
lam = np.array([0.7])
f = imbalance.constraints(w_bal)[0]
g = imbalance.grad_constraints(w_bal)[0]
expected = lam[0] * (2.0 * B.T @ (B @ v) - (1.0 / f) * g * np.dot(g, v))
np.testing.assert_allclose(
    imbalance.hessian_contribution(w_bal, lam).multiply(v), expected, rtol=1e-10
)
fd_curv = (
    imbalance.grad_constraints(w_bal + eps * v) - imbalance.grad_constraints(w_bal - eps * v)
)[0] / (2 * eps)
np.testing.assert_allclose(
    imbalance.hessian_curvature(w_bal, np.ones(1)).multiply(v), fd_curv, rtol=1e-5
)
```

### 4d. Layer order and Phase I

When no strictly feasible `x0` is supplied, the solver builds a Phase I
problem for the *last* layer in `ineq` (the active layer), keeping the
others as background constraints, and solves it through the same
solver. That solve needs a start that satisfies the background layers,
so it recurses: layers are made feasible one at a time, from the first
to the last. Put layers that are easy to satisfy (bounds) first and the
hardest last. A known strictly feasible point, passed as `x0`, skips the
cascade entirely.

## 5. Solving the problem and reading the result

*Example 1, cont'd.* With both components written, the problem is a
composition:

```python
from cvxium import (
    BarrierSolver,
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    PrimalDualSolver,
    Problem,
)

eq = EqualityConstraintLayer(A=A, b=b)
bounds = BoundsConstraintLayer(lb=lb, dimension=M)
problem = Problem.minimize(objective).subject_to(eq=eq, ineq=[bounds, imbalance])

start = time.perf_counter()
result = problem.solve(solver=BarrierSolver())
cvxium_seconds = time.perf_counter() - start

w_star = result.solution
np.testing.assert_allclose(A @ w_star, b, atol=1e-10)
assert np.all(w_star > lb)
assert np.linalg.norm(B @ w_star - c) <= psi
np.testing.assert_allclose(w_star, scipy_result.x, atol=1e-6)
print(f"{result.status}; {scipy_seconds / cvxium_seconds:.0f}x faster than scipy")
```

On this problem Cvxium agrees with scipy to within 1e-6 in every
weight and runs about 100 times faster; the gap widens with M (about
600 times at M = 2,000), since scipy's cost grows much faster than
linearly.

### 5a. Choosing a solver

The same `Problem` goes to any solver that supports its shape:

| Solver             | Method                                                                                 |
|--------------------|----------------------------------------------------------------------------------------|
| `BarrierSolver`    | barrier method (Boyd and Vandenberghe, 2004, section 11.3), nested centering steps     |
| `PrimalDualSolver` | primal-dual interior-point method (section 11.7), carrying the duals explicitly        |
| `NewtonSolver`     | Newton's method, for problems with no inequality constraints                           |
| `HsdSolver`        | homogeneous self-dual method for linear conic programs (section 11); needs no Phase I  |

`BarrierSolver` is the most general and the best-instrumented default
(only it supports `on_infeasible="continue"`, section 5c). Neither
interior-point method dominates the other, so time both on the
problem at hand.

```python
pd_result = problem.solve(solver=PrimalDualSolver())
np.testing.assert_allclose(pd_result.solution, w_star, atol=1e-6)
```

### 5b. The result

A `SolveResult` is a frozen value assembled from parts, each present
exactly when the problem has that structure:
- Always: `solution`, `status` (a `SolveStatus`: `OPTIMAL`,
  `ACCEPTABLE`, `FEASIBLE`, `MAX_ITERATIONS`, `INFEASIBLE`,
  `UNBOUNDED`), `succeeded`, `message`, `n_iterations`.
- With an objective: `objective_value`, `dual_value`, `suboptimality`
  (the duality-gap bound). A feasibility problem has none, and reading
  them raises `NoObjectiveError`.
- With constraints: `equality_multipliers` (shape (p,)) and
  `inequality_multipliers`, one flat array stacked across the layers in
  order. Reading an absent one raises `NoDualsError`.
- Per solver: `diagnostics`, narrowed with `barrier_diagnostics`,
  `primal_dual_diagnostics`, or `hsd_diagnostics` (the wrong one raises
  `WrongDiagnosticsError`).

The multipliers carry the marginal value of each constraint. Here the
last entry of `inequality_multipliers` belongs to the imbalance layer:
it is the rate at which the optimal divergence would fall per unit
increase in ψ^2, which section 10 uses to trace the tradeoff.

```python
from cvxium import SolveResult

mu_star = result.inequality_multipliers[-1]
print(f"KL = {result.objective_value:.5f}, d KL / d psi^2 = {-mu_star:.4f}")
restored = SolveResult.deserialize(result.serialize())
np.testing.assert_allclose(restored.solution, w_star)
```

### 5c. Infeasibility

A solve with no feasible point raises `ProblemCertifiablyInfeasibleError`
when the Phase I dual proves infeasibility, or
`ProblemMarginallyFeasibleError` when no strictly feasible point could
be found but infeasibility could not be certified (both subclass
`ProblemInfeasibleError`). Each carries the best-effort `result`.
Passing `on_infeasible="continue"` to `solve` makes `BarrierSolver`
drive Phase I to its optimum before raising, so that `result` holds the
least-infeasible point; the other solvers raise `NotImplementedError`
for it.

## 6. Initialize the barrier parameter to match the problem

Fast solvers come from two things: cheap Newton steps (sections 2
through 4) and few of them. The initial barrier parameter t0 matters
for the second. Too small, and the solver spends early centering steps
on a problem that barely sees the objective; too large, and the first
centering step starts from a sharply curved, ill-conditioned barrier.
Choose the strategy with `OuterSettings(barrier_init=...)`:

| `BarrierInitStrategy`  | t0                                                                                                   |
|------------------------|------------------------------------------------------------------------------------------------------|
| `UNIT`                 | 1                                                                                                    |
| `GRADIENT_BALANCE`     | ‖∇φ‖ / ‖∇f0‖ at x0; the default without equality constraints                                         |
| `EQUALITY_SCHUR`       | gradient balance through the Schur complement of A (section 11.3.1); the default with equalities     |
| `OBJECTIVE_VALUE_GAP`  | m / (μ (f0(x0) - p*)), given an under-estimate p* via `OuterSettings(objective_value_estimate=...)`  |

*Example 1, cont'd.* The KL divergence is nonnegative, so p* = 0 is a
valid under-estimate for free:

```python
from cvxium import BarrierInitStrategy, OptimizationSettings, OuterSettings

gap_settings = OptimizationSettings(
    outer=OuterSettings(
        barrier_init=BarrierInitStrategy.OBJECTIVE_VALUE_GAP,
        objective_value_estimate=0.0,
    )
)
gap_result = problem.solve(solver=BarrierSolver(gap_settings))
print(f"{result.n_iterations} -> {gap_result.n_iterations} outer iterations")
```

Here that cuts the outer iterations from 11 to 7. A closed-form
under-estimate, a relaxation's optimal value, or the previous point in
a parameter sweep all serve as p*.

## 7. Caching expensive evaluations

When an objective or constraint calls an expensive model (a
likelihood, a simulation, a forward pass), the solver may evaluate it
several times at the same point: the value, the gradient, and the
Hessian are separate calls, and the line search evaluates values at
trial points.

### 7a. Objectives: a cached `ObjectivePoint`

The solver acquires one `ObjectivePoint` per iterate from
`Objective.at(w)` and routes `evaluate`, `gradient`, and
`hessian_contribution` through it. The default point forwards each call
to the objective; overriding `at` to return a point that computes the
shared work once is the supported way to cache:

```python
from cvxium import ObjectivePoint


class _KLPoint(ObjectivePoint):
    """KL divergence at one w, with log(w / q) computed once."""

    def __init__(self, objective: KLDivergenceObjective, w: npt.NDArray[np.float64]) -> None:
        self._q = objective.q
        self._w = w
        self._log_ratio = np.log(w / objective.q)  # shape (M,)

    def evaluate(self) -> float:
        return float(np.sum(self._w * self._log_ratio - self._w + self._q))

    def gradient(self) -> npt.NDArray[np.float64]:
        return self._log_ratio

    def hessian_contribution(self) -> MatrixSystem:
        return DiagonalSystem(1.0 / self._w, eta_inverse=self._w)


class CachedKLDivergenceObjective(KLDivergenceObjective):
    def at(self, w: npt.NDArray[np.float64]) -> ObjectivePoint:
        return _KLPoint(self, w)


cached_problem = Problem.minimize(CachedKLDivergenceObjective(q_base)).subject_to(
    eq=eq, ineq=[bounds, imbalance]
)
np.testing.assert_allclose(
    cached_problem.solve(solver=BarrierSolver()).solution, w_star, atol=1e-10
)
```

For the KL divergence the saving is one logarithm per call, which is
not worth the code; for an objective whose value requires a model fit,
it can halve the solve time.

### 7b. Constraint layers: key on the iterate, and split by need

A layer has no per-point context, so cache inside it, keyed on a byte
snapshot of the iterate (which also detects in-place mutation):

```text
def _ensure_cached(self, w):
    key = w.data.tobytes()
    if key != self._cache_key:
        self._cache_key = key
        self._cache_value = expensive_model_fn(w)
    return self._cache_value
```

A subtlety arises when the Newton step needs more from the model than
`constraints` does. The line search calls only `constraints`, at trial
points w + α dw, while `grad_constraints` and `hessian_contribution`
need gradients and curvature at the accepted iterate. Bundling all of
them in one cache makes every trial evaluation pay for derivatives it
never uses; in one application (a constraint on a binomial
log-likelihood) that roughly doubled the solve time. Split the cache
into a cheap level that `constraints` uses and a full level, built on
it, that the derivative methods use.

## 8. Logging, tracing, and debugging

### 8a. Console output

`ObservabilitySettings(verbose=True)` attaches a console handler to the
`cvxium` logger:

```text
  Starting IPM (_BarrierAdapter)
  01 Beginning centering step with t=1.0
    01 Newton step in 0.386 ms; sub-opt 2.36219 = 0.5 * 2.17357^2
    01 btls_s=1.0000, improvement=2.62252, expected improvement=0.00744335
    02 Newton step in 0.287 ms; sub-opt 0.148928 = 0.5 * 0.545761^2
    03 Newton step in 0.279 ms; sub-opt 0.00398847 = 0.5 * 0.0893138^2 (quadratic convergence threshold)
    ...
  01 Evaluating Lagrange dual function
  01 Lagrangian dual function evaluated in 0.093 ms
  01 Centering step completed in 2.659 ms (duality gap = ...)
  02 Beginning centering step with t=10.0
  ...
  IPM completed in 8.994 ms (_BarrierAdapter)
```

Two-space indentation marks outer-loop events; four spaces mark inner
Newton and line-search events. Outer-loop records are `INFO`, inner
ones `DEBUG`. The library is silent by default (the `cvxium` logger
sits at `WARNING`), so configuring the logger directly also works:
`enable_logging(path, level=logging.DEBUG)` sends the `cvxium`
namespace (and any others named in `namespaces`) to a file or the
console. Configure the `cvxium` logger rather than the root logger, so
that third-party libraries stay quiet.

### 8b. Collecting per-iteration data

A `Trace` receives every solver event as a structured record, without
going through logging. `ListTrace` keeps them in memory, and
`DataFrameTrace` adds `to_dataframe()` (which requires pandas):

```python
from cvxium import ListTrace

trace = ListTrace()
problem.solve(solver=BarrierSolver(OptimizationSettings(trace=trace)))
newton = [r for r in trace.records() if r["event"] == "newton_step"]
print(len(newton), "Newton steps;", newton[0]["suboptimality"], "first sub-optimality")
```

The barrier solver emits `ipm_start`, `ipm_done`, `centering_start`,
`centering_done`, `newton_step`, `btls`, `dual_eval_start`,
`dual_eval_done`, `dual_newton_step`, `early_stop`, `converged`,
`phase1_feasible`, `feasible_initial_guess`, and `svd_done`; the
primal-dual solver emits `pd_start`, `pd_iteration`, `pd_done`, and
`pd_damping_engaged`; the HSD solver `hsd_start`, `hsd_iteration`, and
`hsd_done`. Each record carries its event's numbers (for `newton_step`:
`outer_nit`, `inner_nit`, `elapsed_ms`, `suboptimality`; for `btls`:
`btls_s`, `improvement`, `expected_improvement`); print a few records to
see the full set. `IterationHandler` collects the same fields from log
records, for code that already routes through logging.

### 8c. What to look for

**Short steps after the quadratic convergence threshold.** Once the
log marks `(quadratic convergence threshold)`, the Newton decrement is
in the quadratic phase and every line-search step should be
`btls_s=1.0000`. A shorter step there means the sufficient-decrease
condition is failing, which almost always points to a wrong gradient or
Hessian contribution. Check them against finite differences, as in
sections 3 and 4.

**A slow first centering step.** If the first centering step takes many
Newton iterations, t0 is likely too large for x0: the barrier is
sharply curved near the boundary and the Newton system is
ill-conditioned. Try another `barrier_init` (section 6), or a starting
point farther from the constraint boundaries.

**Slow later centering steps.** Conditioning worsens as t grows,
especially with many inequalities. `outer.tolerance_soft` (default
1e-3) is the practical stopping threshold, and `outer.tolerance`
(default 1e-6) the aspirational one: the solver tries for the latter
and exits gracefully at the former. Many applications do not need high
precision; raising `tolerance` toward `tolerance_soft` stops the solver
before the ill-conditioned late steps.

**`IncompatibleStructureError`.** Two contributions cannot be summed
without densifying (section 2c). Re-express one, usually as a low-rank
update on a `ZeroSystem`.

## 9. The checklist

1. **Establish ground truth** with scipy, and time it (section 0).
2. **Write the problem in standard form**, check convexity, write the
   Lagrangian and, where possible, the dual (sections 1a to 1c).
3. **Write the barrier Hessian and identify its structure** (section
   1d). If it has none, Cvxium will not be faster than a general
   solver.
4. **Compose from library components** where they exist.
5. **Write new objectives and layers**, reporting each Hessian as a
   structured `MatrixSystem`; check derivatives against finite
   differences (sections 3 and 4).
6. **Solve, verify against ground truth, and time** (section 5).
7. **Speed up the rest**: closed-form `btls_max_step`, structured
   Jacobian products, a sparse or banded `SaddleFactory`, a better
   `barrier_init`, caching (sections 2d, 4b, 6, and 7). Re-test and
   re-time after each.

## 10. Multi-objective optimization and Pareto frontier tracing

When a problem has two or more competing objectives,
`MultiObjectiveOptimizer` traces the Pareto frontier, the set of
solutions where no objective can improve without another worsening.
The framework handles corner detection, grid sweeping, deduplication,
knee finding, interpolation, and plotting; we supply the solves. It is
designed for a small number of objectives (two to four).

### 10a. Concept

Designate one objective as *primary* and the others as *auxiliary*.
The framework sweeps upper bounds on the auxiliaries and minimizes the
primary at each, tracing the curve from the corner that minimizes the
primary alone to the corners that minimize each auxiliary alone. For
two objectives the frontier answers: "if I allow f1 up to a budget φ,
what is the best f0 I can achieve?"

*Example 1, cont'd.* How much divergence from the base weights does
balance cost? Make the KL divergence primary and the squared imbalance
||B w - c||^2 auxiliary; the frontier shows the price of each
increment of balance.

### 10b. Subclassing MultiObjectiveOptimizer

Implement three methods:
- **`solve_with_bounds(bounds, x0=None)`**: minimize the primary
  objective subject to `auxiliary_j(w) <= bounds[j]`, returning the
  `SolveResult`. `bounds` has shape (N - 1,), in objective order,
  skipping the primary. With Cvxium this is one extra constraint layer
  per auxiliary: `ObjectiveConstraintLayer(objective, upper_bound)`
  bounds any `Objective`, and a custom layer (like section 4's) works
  as well.
- **`minimize_objective(objective_index, x0=None)`**: minimize one
  objective with no bound on the others, for the corners. Raise
  `OptimizationError` or `ProblemInfeasibleError` on failure, and the
  framework omits that corner.
- **`evaluate_objectives(w)`**: all N objective values at w, shape (N,).

`x0` is a warm start, by default the previous solve's solution; the
generic parameter `WarmStartT` and the `_extract_warm_start` hook let
a subclass thread a richer payload between solves.

One optional override is worth the effort: **`tangent_slopes(results)`**
returns the frontier slope d(primary) / d(auxiliary) at each point,
aligned with `results.points` (nan where unknown). For a bound
f1(w) <= φ with multiplier λ, the slope is -λ by the envelope theorem,
so it costs nothing to report. The slopes certify
`FrontierResults.envelope()` and enable `refine_knee`.

```python
from cvxium import FrontierResults, MultiObjectiveOptimizer

# solve_with_bounds takes a parameter named `bounds`, so name the layer apart.
bounds_layer = bounds


class CalibrationFrontier(MultiObjectiveOptimizer):
    """Trade KL divergence (primary) against squared imbalance (auxiliary)."""

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        layer = SquaredImbalanceLayer(B, c, psi=math.sqrt(float(bounds[0])))
        bounded = Problem.minimize(objective).subject_to(eq=eq, ineq=[bounds_layer, layer])
        return bounded.solve(solver=BarrierSolver(), x0=x0)

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        if objective_index == 0:
            unbounded = Problem.minimize(objective).subject_to(eq=eq, ineq=[bounds_layer])
            return unbounded.solve(solver=BarrierSolver())
        # Zero imbalance has many minimizers; the useful corner is the one with the
        # least divergence among them, so impose exact balance as equalities.
        exact = EqualityConstraintLayer(A=np.vstack([A, B]), b=np.concatenate([b, c]))
        balanced = Problem.minimize(objective).subject_to(eq=exact, ineq=[bounds_layer])
        return balanced.solve(solver=BarrierSolver())

    def evaluate_objectives(
        self, w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r = B @ w - c
        return np.array([objective.evaluate(w), float(np.dot(r, r))])

    def tangent_slopes(self, results: FrontierResults) -> npt.NDArray[np.float64]:
        slopes = []
        for point in results.points:
            if np.all(np.isinf(point.bounds)):
                # A corner. At the primary corner the bound is slack (slope 0); at
                # the exact-balance corner the tangent is vertical (unknown).
                is_primary = point is results.corners[0]
                slopes.append(0.0 if is_primary else math.nan)
            else:
                # The imbalance layer is the last, so its multiplier is the last.
                slopes.append(-float(point.ipm_result.inequality_multipliers[-1]))
        return np.array(slopes)


frontier = CalibrationFrontier(primary_objective=0)
fr = frontier.trace(num_points=20)
```

### 10c. Algorithm inside `trace()`

1. Calls `minimize_objective(k)` for each objective to compute the N
   corners and the sweep range of each auxiliary (its own corner's
   value up to its value at the primary corner).
2. If every corner's objective vector agrees to within 1e-8, the
   frontier is a single point; the sweep is skipped and logged at
   `INFO`.
3. Builds a grid of `num_points` bounds per auxiliary strictly between
   the corner values (the corners are already in hand, and at an exact
   corner bound an inequality can leave the feasible set with an empty
   interior).
4. Calls `solve_with_bounds` at each grid point, skipping (and
   counting in `n_skipped`) any that raise `OptimizationError` or
   `ProblemInfeasibleError`.
5. Deduplicates by objective vector (1e-8 absolute tolerance), sorts by
   the primary objective, and records `tangent_slopes`.

The sweep costs at most N + num_points^(N - 1) solves. `trace` also
takes `callback(point, k, total)`, called after each successful solve,
which can return True to stop early, and `n_jobs` to run the grid in
parallel processes (the subclass must then be picklable).

### 10d. Working with FrontierResults

- **`points`**: every distinct point, corners and knee probes
  included, sorted by the primary objective. Each `FrontierPoint` has
  `objectives`, `solution`, `bounds` (inf for corners), and
  `ipm_result`, the `SolveResult`.
- **`corners`**: `corners[0]` minimizes the primary; the rest minimize
  the auxiliaries in order. A failed corner is dropped, so later
  entries shift down.
- **`knee(knee_space="primary")`**: the point farthest from the
  hyperplane through the corners (for two objectives, the elbow of the
  curve). `knee_space="root"` measures in square-root coordinates,
  useful when the primary objective is a variance. Raises `ValueError`
  when corners are missing or degenerate.
- **`envelope(num=200)`**: a `FrontierEnvelope` interpolating the
  two-objective frontier, with a `chord` (an upper bound on a convex
  frontier), a `tangent` (a lower bound built from verified slopes), a
  smooth `curve` clipped between them, and `max_band_width`, the widest
  gap as a fraction of the primary objective's range. Without
  `tangent_slopes` there is no lower bound and the band is infinite.
- **`MultiObjectiveOptimizer.refine_knee(results)`**: solves for the
  knee rather than picking the nearest grid point. On a convex frontier
  f(a) the knee is where f'(a) equals the chord slope, a monotone
  root-find in which every probe is a real solve; it usually takes a
  handful of solves. Requires `tangent_slopes`; records its answer in
  `results.refined_knee`.
- **`plot(annotate_knee=True, ax=None, x_label=..., y_label=...)`**:
  plots a two-objective frontier with the primary objective on the
  x-axis, drawing the certified interpolant and its error band when
  slopes are known (else the polyline), and marks the sampled knee.
  Returns a matplotlib `Axes`; raises `NotImplementedError` for N > 2.
- **`n_attempted`, `n_skipped`**: grid and probe solves attempted and
  failed.

```python
knee = fr.knee()
refined = frontier.refine_knee(fr)
assert refined is not None
envelope = fr.envelope()
print(f"knee: KL = {knee.objectives[0]:.4f}, imbalance = {knee.objectives[1]:.4f}")
print(f"refined knee: imbalance = {refined.objectives[1]:.4f}")
print(f"envelope pinned to within {envelope.max_band_width:.0%} of the KL range")

import matplotlib

matplotlib.use("Agg")
ax = fr.plot(x_label="KL divergence", y_label="Squared imbalance")
```

### 10e. Subclassing FrontierResults

Override `trace()` to wrap the result in a domain-specific subclass with
named properties and a `plot()` with sensible defaults. Pass
`tangent_slopes` through the constructor: it is the one piece of state
not implied by the points, and dropping it silently downgrades every
envelope downstream.

```python
class CalibrationFrontierResults(FrontierResults):
    @property
    def divergences(self) -> list[float]:
        return [float(point.objectives[0]) for point in self.points]


class NamedCalibrationFrontier(CalibrationFrontier):
    def trace(self, num_points: int = 50, **kwargs) -> CalibrationFrontierResults:
        fr = super().trace(num_points=num_points, **kwargs)
        return CalibrationFrontierResults(
            points=fr.points,
            corners=fr.corners,
            primary_objective=fr.primary_objective,
            n_attempted=fr.n_attempted,
            n_skipped=fr.n_skipped,
            tangent_slopes=fr.tangent_slopes,
        )


named = NamedCalibrationFrontier().trace(num_points=10)
assert named.tangent_slopes is not None
```

### 10f. Practical notes

**Grid density.** `num_points=50` (the default) costs 52 solves for two
objectives. Use 20 to 30 for exploration; with `tangent_slopes`,
`envelope()` reports how tight the frontier is pinned, a better guide
than judging the plot by eye. Read the band where it is widest, though:
in Example 1 it is 22% with 20 points, and the widest gap sits at the
exact-balance corner, where the tangent is vertical and no slope bounds
the curve. Doubling the grid narrows it only to 16%, but away from that corner
the frontier is pinned tightly: past the first tenth of the imbalance
range, the band is under 1%.

**Many skipped points.** A high `n_skipped / n_attempted` usually
signals a problem in `solve_with_bounds`, since the sweep range comes
from the corners.

**More than two objectives.** `knee()` still works; `plot()`,
`envelope()`, and `refine_knee` are two-objective features. The grid
grows as num_points^(N - 1), so keep `num_points` small (5 to 10) for
N = 3.

**Choosing the primary objective.** It defaults to 0 and decides which
objective is minimized and which are bounded. Pick the one whose bound
is harder to impose as a constraint as the primary.

## 11. Conic and saddle-point problems

### 11a. Linear conic programs

`HsdSolver` solves
```text
minimize c^T x   subject to   A x = b,   x in K = K_1 x ... x K_q,
```
over a product of symmetric cones, by the homogeneous self-dual method
(Andersen, Roos, and Terlaky, 2003). It needs no strictly feasible
start, and reports infeasibility or unboundedness through the result's
status and certificate rather than by raising. The `Problem` supplies c
(a `LinearObjective`) and the equalities; the solver's `cones` argument
supplies K. `solve_second_order_cone_program` handles the common
special case of one second-order cone constraint with bounds and
equalities:

```python
from cvxium import solve_second_order_cone_program

# maximize 1^T x subject to ||x||_2 <= 1, x >= 0: the answer is x = 1 / sqrt(n).
n = 4
socp = solve_second_order_cone_program(
    f=-np.ones(n),
    A_soc=np.eye(n),
    b_soc=np.zeros(n),
    c_soc=np.zeros(n),
    d_soc=1.0,
    F=np.zeros((0, n)),
    g_eq=np.zeros(0),
)
np.testing.assert_allclose(socp.solution, np.full(n, 0.5), atol=1e-5)
```

A new cone (the positive semidefinite cone, say) is added by
subclassing `Cone`, without changing the solver.

### 11b. Saddle-point problems

Some problems seek a saddle point
```text
V = max_x min_y zeta(x, y)
```
of an objective concave in a large, structured x-block and convex in a
small y-block, each confined to its own domain: robust optimization
over a confidence region is the typical case. A `StationarityProblem`
bundles a `TwoBlockObjective` (its value, block gradients, and curvature
blocks; `QuadraticSaddleObjective` is a concrete example) with
constraint layers for each block's domain and optional equalities on
the x-block. `StationaritySolver` (barrier) and
`PrimalDualStationaritySolver` solve it, returning a
`TwoBlockSolveResult` with x- and y-block views. The x-block's
constraint layers and Hessian structure work exactly as in sections 2
through 4. See the docstrings in `cvxium/saddle/` for the contract.

## References

- Andersen, Erling D., Roos, Cornelis and Terlaky, Tamás. On
  Implementing a Primal-Dual Interior-Point Method for Conic Quadratic
  Optimization. *Mathematical Programming* 95, 2003.
- Boyd, Stephen and Vandenberghe, Lieven. *Convex Optimization*.
  Cambridge University Press, 2004.
