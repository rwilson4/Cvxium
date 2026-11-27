# Writing Your Own Structure: Radiotherapy QPs

Cvxium ships the structures that recur across applications: diagonal,
banded, low-rank, block diagonal, arrow, sparse equality constraints.
Real problems often have one more that is particular to their field.
Cvxium is designed so that you can describe that structure yourself,
as a new `MatrixSystem` and a new `ConstraintLayer`, and every solver
then exploits it with no changes of its own.

This guide works one such problem from start to finish. It comes from
[RadOptLIB](https://github.com/Radiotherapy-Optimization/RadOptLIB), a
library of quadratic programs from clinical radiotherapy treatment
planning at Memorial Sloan Kettering. We take its smallest instance,
rt-qp-0062: 8,665 variables, 27,313 inequality rows, and 2.5 million
nonzeros. We solve it with Clarabel for a time to beat, then with
Cvxium as most people would first write it, then study the structure
of its Newton system, write a matrix system and a constraint layer
that capture it, and measure the payoff:

| Formulation | Solver | Time |
|---|---|---|
| Clarabel (sparse LDL interior point) | Clarabel | 67 s |
| Cvxium, stock layers | — | runs out of memory |
| Cvxium, a 10-line custom layer (dense Newton system) | `PrimalDualSolver` | 272 s |
| Cvxium, a custom sparse arrow system and layer | `PrimalDualSolver` | 60 s |

Every solve that finishes matches RadOptLIB's reference solution, to
a relative duality gap of about 2e-9. Writing the structure down makes
Cvxium about 4.5 times faster than its first working formulation, and
puts it level with Clarabel. (All timings come from a 4-core
Intel Xeon at 2.1 GHz with 15 GB of memory, NumPy 2.4, SciPy 1.17, and
Clarabel 0.11; expect different absolute numbers elsewhere.)

The instance file is 15 MB and is downloaded separately, so the
documentation tests cannot run on it. Instead, every `python` block
below runs, in order, as one script on a *miniature* instance with the
same structure, a few hundred variables, and checks itself; the
blocks that load and solve the real instance are marked as skipped.
To run the guide on rt-qp-0062 itself, run the skipped blocks in place
of the miniature one.

## The problem

Radiotherapy shapes and weights hundreds of small radiation beams
(beamlets) around the patient so that the tumor receives its
prescribed dose and the healthy organs as little as possible. Every
RadOptLIB QP has the form

$$
\begin{array}{ll}
\mbox{minimize}   & \frac{1}{2} x^T Q x + q^T x \\
\mbox{subject to} & G x \preceq h \\
                  & l \preceq x \preceq u,
\end{array}
$$

with Q positive semidefinite. In rt-qp-0062 the first 3,135 variables
are beamlet intensities. The quadratic penalizes the deviation of the
delivered dose from the intended one, so Q is a dense-ish 3,135 ×
3,135 block (12.6% nonzero, full rank) and zero elsewhere. The other
5,530 variables are *epigraph slacks*, t_j ≥ 0, one for each
dose-limit penalty: a row g_j^T x - t_j ≤ h_j, where g_j is a row of
the dose-influence matrix, makes t_j at least the overdose at one
point in the body. The rows of G fall into four kinds:

- dose limits over beamlets only (21,717 rows, a few to a few
  thousand nonzeros each);
- penalty rows g_j^T x - t_j ≤ h_j, each with one slack (5,595 rows);
- one dose-volume budget, a sum over 3,853 slacks;
- one pair of rows w^T x ≤ c and -w^T x ≤ -c, which fixes the result
  of an earlier planning stage.

The file stores the matrices in CSR form:

```{code-block} python
import h5py
import scipy.sparse as sp


def load_instance(path):
    """Read a RadOptLIB QP instance (an HDF5 file) into a dict of arrays."""
    with h5py.File(path, "r") as f:
        def csr(key):
            return sp.csr_array(
                (f[key + "_data"][()], f[key + "_indices"][()], f[key + "_indptr"][()]),
                shape=tuple(f[key + "_shape"][()]),
            )
        return dict(Q=csr("Q"), q=f["q"][()], G=csr("G"), h=f["h"][()],
                    l=f["l"][()], u=f["u"][()], c=float(f["c"][()]))


data = load_instance("rt-qp-0062.h5")  # from RadOptLIB's examples/ directory
```

The miniature instance below has the same four kinds of rows, the
same split into beamlets (first) and slacks, and a strict interior
apart from the fixed pair.

```python
import time

import numpy as np
import scipy.sparse as sp


def miniature_instance(n_beam=40, n_slack=80, n_limit=120, seed=0):
    """A small QP with rt-qp-0062's structure; beamlets first, then slacks."""
    rng = np.random.default_rng(seed)
    n = n_beam + n_slack
    x_beam = rng.uniform(0.3, 0.7, n_beam)  # an interior point to build around
    t_slack = rng.uniform(0.2, 0.5, n_slack)
    target = sp.random_array((200, n_beam), density=0.3, rng=rng)  # dose to tumor
    Q = sp.block_diag([(target.T @ target).tocsr(), sp.csr_array((n_slack, n_slack))])
    q = np.r_[-(target.T @ rng.uniform(0.5, 1.0, 200)), np.full(n_slack, 0.1)]
    limit = sp.random_array((n_limit, n_beam), density=0.2, rng=rng)
    penalty = sp.random_array((n_slack, n_beam), density=0.3, rng=rng)
    w = rng.uniform(0.0, 1.0, n_beam)
    G = sp.vstack([
        sp.hstack([limit, sp.csr_array((n_limit, n_slack))]),  # dose limits
        sp.hstack([penalty, -sp.eye_array(n_slack)]),  # g_j^T x - t_j <= h_j
        sp.hstack([sp.csr_array((1, n_beam)), sp.csr_array(np.ones((1, n_slack)))]),
        sp.csr_array(np.r_[w, np.zeros(n_slack)][None, :]),  # w^T x <= c
        sp.csr_array(np.r_[-w, np.zeros(n_slack)][None, :]),  # -w^T x <= -c
    ]).tocsr()
    h = np.r_[
        limit @ x_beam + 0.1,
        penalty @ x_beam - t_slack + 0.1,
        t_slack.sum() + 1.0,
        w @ x_beam,
        -(w @ x_beam),
    ]
    l = np.zeros(n)
    u = np.r_[np.ones(n_beam), np.full(n_slack, np.inf)]
    return dict(Q=Q, q=q, G=G, h=h, l=l, u=u, c=0.0)


data = miniature_instance()
n = data["G"].shape[1]
```

## An equality in disguise

An interior-point method needs a point that satisfies every inequality
strictly, and this problem has none: no x satisfies w^T x < c and
-w^T x < -c at once. Given the pair as written, Phase I can only
discover that the best achievable margin is zero, at great expense
(tens of minutes on the full instance). The remedy is a small presolve
step: find each pair of rows that are exact negatives of each other,
and state the pair as one equality.

```python
def split_equalities(G, h):
    """Move each pair of rows G_j = -G_i, h_j = -h_i into equalities A x = b."""
    G = sp.csr_array(G)
    G.sort_indices()
    rows = {}
    for i in range(G.shape[0]):
        start, stop = G.indptr[i], G.indptr[i + 1]
        rows[(G.indices[start:stop].tobytes(), G.data[start:stop].tobytes(), h[i])] = i
    equal, dropped = [], set()
    for i in range(G.shape[0]):
        start, stop = G.indptr[i], G.indptr[i + 1]
        key = (G.indices[start:stop].tobytes(), (-G.data[start:stop]).tobytes(), -h[i])
        j = rows.get(key)
        if j is not None and i not in dropped and j not in dropped:
            equal.append(i)
            dropped |= {i, j}
    keep = np.array(sorted(set(range(G.shape[0])) - dropped))
    return G[keep], h[keep], G[equal].toarray(), h[equal]


G, h, A, b = split_equalities(data["G"], data["h"])
assert A.shape[0] == 1 and G.shape[0] == data["G"].shape[0] - 2
```

On rt-qp-0062 this finds the one pair (rows 27311 and 27312), and the
remaining inequalities then have a strict margin of 0.016.

## A baseline with Clarabel

[Clarabel](https://clarabel.org) is an interior-point solver built on
a sparse LDL factorization of the KKT matrix, and a natural reference
here: it needs no strictly feasible start and treats every nonzero
pattern alike. It takes the problem in the form A x + s = b, s ∈ K,
so we stack the inequalities, the bounds and the equality.

```python
import clarabel


def solve_with_clarabel(data, tol=1e-8):
    Q, q, G, h, l, u = (data[k] for k in ("Q", "q", "G", "h", "l", "u"))
    n = G.shape[1]
    lower, upper = np.flatnonzero(np.isfinite(l)), np.flatnonzero(np.isfinite(u))
    eye = sp.eye_array(n, format="csr")
    A_cl = sp.vstack([G, -eye[lower], eye[upper]]).tocsc()
    b_cl = np.r_[h, -l[lower], u[upper]]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = tol
    solver = clarabel.DefaultSolver(
        sp.triu(Q).tocsc(), q, A_cl, b_cl,
        [clarabel.NonnegativeConeT(A_cl.shape[0])], settings,
    )
    solution = solver.solve()
    return np.array(solution.x), solution


x_clarabel, clarabel_solution = solve_with_clarabel(data)
objective_clarabel = 0.5 * x_clarabel @ (data["Q"] @ x_clarabel) + data["q"] @ x_clarabel
```

On rt-qp-0062 Clarabel takes 26 iterations and 67 seconds, and its
solution matches the reference to a relative duality gap of 2e-9.
(Clarabel handles the fixed pair of rows without help, since it does
not need a strictly feasible start.)

## Cvxium out of the box

The stock formulation uses three library components: a
`QuadraticObjective` with Q as a `DenseSystem`, a
`BoundsConstraintLayer` for l ≤ x ≤ u, and a
`LinearInequalityConstraintLayer` for G x ≤ h, plus the equality.

```python
from cvxium import (
    BoundsConstraintLayer,
    DenseSystem,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
    OptimizationSettings,
    OuterSettings,
    PrimalDualSolver,
    Problem,
    QuadraticObjective,
)

settings = OptimizationSettings(
    outer=OuterSettings(objective_scale="relative", tolerance=1e-8)
)


def stock_problem(data, G, h, A, b):
    return Problem.minimize(
        QuadraticObjective(c=data["q"], hessian_contribution=DenseSystem(data["Q"].toarray()))
    ).subject_to(
        eq=EqualityConstraintLayer(A=A, b=b),
        ineq=[
            BoundsConstraintLayer(lb=data["l"], ub=data["u"]),
            LinearInequalityConstraintLayer(G, cu=h),
        ],
    )


stock = stock_problem(data, G, h, A, b).solve(solver=PrimalDualSolver(settings))
assert stock.succeeded
np.testing.assert_allclose(stock.objective_value, objective_clarabel, rtol=1e-6)
```

(We ask for a duality gap of 1e-8 relative to the objective with
`objective_scale="relative"`. The objective of rt-qp-0062 is about
10^4, so the default, an absolute gap of 1e-6, would ask for ten
significant digits, near the limit of double precision for a barrier
method, while the default tolerance made relative would stop at a gap
of 10^-2.)

That solves the miniature, but on rt-qp-0062 it fails with a
`MemoryError` at the first Newton step:

```{code-block} python
stock_problem(data, G, h, A, b).solve(solver=PrimalDualSolver(settings))
# numpy._core._exceptions._ArrayMemoryError: Unable to allocate 5.56 GiB
# for an array with shape (27311, 27311) and data type float64
```

`LinearInequalityConstraintLayer` reports its barrier Hessian
G^T D G (D the diagonal of barrier weights) as a low-rank update of
rank m, solved through the Woodbury identity. That is the right
choice for a few dense rows over many variables, as in the
[large linear programs guide](linear_programs.md), and the wrong one
here: with m = 27,311 rows over n = 8,665 variables, the update's
capacitance matrix alone is m × m.

## The simplest custom structure

When the update is wider than the matrix it updates, it is cheaper to
form the matrix itself: G^T D G is n × n. A constraint layer that does
so is a ten-line subclass. It inherits everything else, the constraint
values, Jacobian products, step-length bounds and Phase I support,
from `LinearInequalityConstraintLayer`, and overrides only the Hessian
contribution. For a one-sided layer (every row of the form
(G x)_i ≤ h_i), the barrier weight of row i is d_i = -λ_i / f_i, with
f_i = (G x - h)_i < 0.

```python
class DenseHessianInequalityLayer(LinearInequalityConstraintLayer):
    """G x <= h, reporting G^T diag(d) G as a dense matrix instead of an update."""

    def __init__(self, G, cu):
        super().__init__(G, cu=cu)
        self._G_sparse = sp.csr_array(G)

    def hessian_contribution(self, w, lmbda=None, shift=0.0):
        d = -lmbda / self.constraints(w, shift)
        GT_D_G = self._G_sparse.T @ sp.diags_array(d) @ self._G_sparse
        return DenseSystem(GT_D_G.toarray())


def dense_problem(data, G, h, A, b):
    return Problem.minimize(
        QuadraticObjective(c=data["q"], hessian_contribution=DenseSystem(data["Q"].toarray()))
    ).subject_to(
        eq=EqualityConstraintLayer(A=A, b=b),
        ineq=[
            BoundsConstraintLayer(lb=data["l"], ub=data["u"]),
            DenseHessianInequalityLayer(G, cu=h),
        ],
    )


dense = dense_problem(data, G, h, A, b).solve(solver=PrimalDualSolver(settings))
np.testing.assert_allclose(dense.solution, stock.solution, atol=1e-5)
```

The sum of the objective's `DenseSystem`, the bounds' `DiagonalSystem`
and this layer's `DenseSystem` is one `DenseSystem`, factored by
Cholesky at every Newton step. On rt-qp-0062 `PrimalDualSolver` solves
it in 26 iterations after Phase I and 272 seconds (`BarrierSolver`:
981 seconds), about 4 seconds per Newton step, of which the
8,665 × 8,665 Cholesky factorization is the largest part. That
is the cost of treating the Newton system as dense. To do better, we
look at its structure.

## The structure in the Newton system

The Newton system's matrix is H = Q + G^T D G + D_bounds. Order the
variables as slacks first, then beamlets, and write it in blocks:

$$
H = \begin{bmatrix} H_{tt} & H_{tx} \\ H_{xt} & H_{xx} \end{bmatrix}.
$$

On rt-qp-0062:

- **H_tt, slack × slack (5,530 × 5,530), is a diagonal plus a rank-one
  term.** A penalty row touches one slack, so its contribution d_i e_j
  e_j^T is diagonal. Only the budget row touches many slacks, and it
  contributes one rank-one term. (Measured on the whole matrix, H
  looks 23% dense; almost all of that density is this one rank-one
  term, 3,853² entries.)
- **H_tx, slack × beamlet, is sparse**: 43,738 nonzeros out of 17
  million. Each penalty row couples its slack to the few beamlets in
  its row.
- **H_xx, beamlet × beamlet (3,135 × 3,135), is dense.** Every dose
  row couples many beamlets: each beamlet deposits dose across much of
  the body. This is intrinsic to the physics, and no structure helps
  here.

This is an *arrow* matrix: a large, cheap core (H_tt) bordered by a
dense block of p = 3,135 rows and columns. Eliminating the core leaves
the p × p Schur complement

$$
S = H_{xx} - H_{xt} H_{tt}^{-1} H_{tx},
$$

and a Newton solve costs a Cholesky factorization of S instead of all
of H: about p³/3 = 10 GFLOP instead of n³/3 = 217 GFLOP.

Cvxium's `ArrowSystem` has this shape, but it stores the border
H_tx densely and forms H_xt H_tt⁻¹ H_tx by a dense matrix product,
2 M p² ≈ 108 GFLOP, which gives away most of the gain. The border is
99.7% zeros, so the right system keeps it sparse.

## Writing a MatrixSystem

A `MatrixSystem` needs `dimension`, `solve`, `multiply` and `scaled`.
To sum with other contributions it also implements `_absorb`, which
`plus` calls to fold another system into this one; and since the
default `is_positive_definite` would densify the matrix, we override
that too. The solve forms S once per Newton step and caches its
Cholesky factor, since a step solves several right-hand sides with the
same matrix.

```python
from scipy import linalg

from cvxium import ArrowSystem, DiagonalSystem, IncompatibleStructureError, MatrixSystem, ZeroSystem
from cvxium.exceptions import NewtonStepError


class SparseArrowSystem(MatrixSystem):
    """[[core, border], [border^T, corner]]: any MatrixSystem core, a sparse border,
    and a dense corner. Solves through the Schur complement on the corner."""

    def __init__(self, core, border, corner):
        self._core = core
        self._border = sp.csr_array(border)
        self._corner = np.asarray(corner, dtype=np.float64)
        self._factor = None  # Cholesky factor of S, computed on the first solve

    @property
    def dimension(self):
        return self._core.dimension + self._corner.shape[0]

    def _schur_factor(self):
        if self._factor is None:
            # S = corner - border^T core^{-1} border. Solve the core against the
            # border a block of columns at a time; the border stays sparse, so the
            # product border^T Y costs nnz(border) per column.
            B, p = self._border, self._corner.shape[0]
            S = self._corner.copy()
            for start in range(0, p, 256):
                block = slice(start, min(start + 256, p))
                S[:, block] -= B.T @ self._core.solve(B[:, block].toarray())
            try:
                self._factor = linalg.cho_factor(0.5 * (S + S.T), lower=True)
            except np.linalg.LinAlgError:
                raise NewtonStepError("H is not positive definite") from None
        return self._factor

    def solve(self, b):
        M = self._core.dimension
        b_core, b_corner = b[:M], b[M:]
        y = self._core.solve(b_core)
        x_corner = linalg.cho_solve(self._schur_factor(), b_corner - self._border.T @ y)
        x_core = self._core.solve(b_core - self._border @ x_corner)
        return np.concatenate([x_core, x_corner], axis=0)

    def multiply(self, y):
        M = self._core.dimension
        top = self._core.multiply(y[:M]) + self._border @ y[M:]
        bottom = self._border.T @ y[:M] + self._corner @ y[M:]
        return np.concatenate([top, bottom], axis=0)

    def scaled(self, t):
        return SparseArrowSystem(self._core.scaled(t), t * self._border, t * self._corner)

    def is_positive_definite(self):
        try:
            self._schur_factor()
        except NewtonStepError:
            return False
        return self._core.is_positive_definite()

    def _absorb(self, other):
        M = self._core.dimension
        if isinstance(other, SparseArrowSystem) and other._core.dimension == M:
            return SparseArrowSystem(
                self._core.plus(other._core),
                self._border + other._border,
                self._corner + other._corner,
            )
        if isinstance(other, DiagonalSystem):  # e.g. the bounds layer
            return SparseArrowSystem(
                self._core.plus(DiagonalSystem(other.eta[:M])),
                self._border,
                self._corner + np.diag(other.eta[M:]),
            )
        raise IncompatibleStructureError(f"Cannot combine with {type(other).__name__}.")
```

Before using it in a solver, we check it against the dense matrix it
represents, on a core with the shape it will have here: a diagonal
plus a rank-one term.

```python
rng = np.random.default_rng(1)
core = DiagonalSystem(rng.uniform(1.0, 2.0, 30)).low_rank_update(rng.standard_normal(30), 1.0)
border = sp.random_array((30, 6), density=0.2, rng=rng)
corner = 10.0 * np.eye(6)
H = SparseArrowSystem(core, border, corner)
H_dense = H.multiply(np.eye(36))
b_test = rng.standard_normal(36)
np.testing.assert_allclose(H.solve(b_test), np.linalg.solve(H_dense, b_test), rtol=1e-10)
np.testing.assert_allclose(
    H.plus(DiagonalSystem(np.ones(36))).multiply(np.eye(36)), H_dense + np.eye(36)
)
```

## Writing a ConstraintLayer

The constraint layer is again a subclass of
`LinearInequalityConstraintLayer` that overrides
`hessian_contribution`. With the core variables (the slacks) first, it
splits G's columns into a core part G_c and a border part G_b and
assembles

- the core block G_c^T D G_c: a row with at most one core nonzero
  adds only to its diagonal, and a row with two or more (the budget
  row) becomes a low-rank update, so the core is
  `DiagonalSystem(...).low_rank_update(...)`;
- the border G_c^T D G_b, kept sparse;
- the corner G_b^T D G_b, formed densely.

All three products keep their sparsity patterns from step to step, so
the splitting happens once, in the constructor.

```python
class SparseArrowInequalityLayer(LinearInequalityConstraintLayer):
    """G x <= h with the first core_dimension columns as an arrow's core."""

    def __init__(self, G, cu, core_dimension):
        super().__init__(G, cu=cu)
        G = sp.csr_array(G)
        G_core = G[:, :core_dimension].tocsr()
        self._G_border = G[:, core_dimension:].tocsc()
        self._G_core = G_core.tocsc()
        core_nonzeros = np.diff(G_core.indptr)
        self._light = np.flatnonzero(core_nonzeros <= 1)  # diagonal contributions
        self._heavy = np.flatnonzero(core_nonzeros > 1)  # low-rank contributions
        light = G_core[self._light]
        self._light_squares = light.multiply(light).T.tocsr()
        self._heavy_factor = G_core[self._heavy].T.toarray()

    def hessian_contribution(self, w, lmbda=None, shift=0.0):
        d = -lmbda / self.constraints(w, shift)
        core = DiagonalSystem(self._light_squares @ d[self._light])
        if self._heavy.size:
            core = core.low_rank_update(self._heavy_factor, d[self._heavy])
        D_border = sp.diags_array(d) @ self._G_border
        border = self._G_core.T @ D_border
        corner = (self._G_border.T @ D_border).toarray()
        return SparseArrowSystem(core, border, corner)
```

The objective must report its Hessian in the same shape, so that the
sum keeps the arrow: Q lives on the beamlets only, so it is a
`SparseArrowSystem` whose core is a `ZeroSystem` and whose border is
empty. (A zero core is fine, since it is never solved on its own: the
layers' contributions fill it in.) Finally, the layer needs the core
variables first, so we reorder the variables before building the
problem, and undo the permutation on the solution.

```python
def arrow_problem(data, G, h, A, b):
    n = G.shape[1]
    n_beam = int(np.flatnonzero(np.diff(sp.csr_array(data["Q"]).indptr)).max()) + 1
    n_core = n - n_beam
    order = np.r_[np.arange(n_beam, n), np.arange(n_beam)]  # slacks, then beamlets
    Q = sp.csr_array(data["Q"])[order][:, order]
    objective_hessian = SparseArrowSystem(
        ZeroSystem(n_core), sp.csr_array((n_core, n_beam)), Q[n_core:, n_core:].toarray()
    )
    problem = Problem.minimize(
        QuadraticObjective(c=data["q"][order], hessian_contribution=objective_hessian)
    ).subject_to(
        eq=EqualityConstraintLayer(A=A[:, order], b=b),
        ineq=[
            BoundsConstraintLayer(lb=data["l"][order], ub=data["u"][order]),
            SparseArrowInequalityLayer(G[:, order], cu=h, core_dimension=n_core),
        ],
    )
    return problem, order


problem, order = arrow_problem(data, G, h, A, b)
arrow = problem.solve(solver=PrimalDualSolver(settings))
x_arrow = np.empty(n)
x_arrow[order] = arrow.solution
np.testing.assert_allclose(x_arrow, stock.solution, atol=1e-5)
np.testing.assert_allclose(arrow.objective_value, objective_clarabel, rtol=1e-6)
```

Nothing else changes. Phase I lifts the problem with one extra slack
variable and wraps each layer's contribution in an `ArrowSystem` with
a border of width one; that wrapper solves through our system's
`solve`, so Phase I inherits the structure too. `BarrierSolver` uses
the same two classes unchanged.

## The payoff

On rt-qp-0062:

| Formulation | Solver | Iterations (after Phase I) | Time | Rel. duality gap |
|---|---|---|---|---|
| Clarabel | Clarabel | 26 | 67 s | 2.2e-9 |
| Stock layers | — | — | out of memory | — |
| Dense custom layer | `PrimalDualSolver` | 26 | 272 s | 1.8e-9 |
| Dense custom layer | `BarrierSolver` | 10 outer | 981 s | 3.8e-9 |
| Sparse arrow system | `PrimalDualSolver` | 26 | 60 s | 1.7e-9 |
| Sparse arrow system | `BarrierSolver` | 10 outer | 272 s | 3.8e-9 |

For each solver the two Cvxium formulations take the same number of
iterations to the same objective value, to all printed digits: the
arrow system solves the same Newton system as the dense one, only
faster, so the speedup is the linear algebra alone (4.5 times for
`PrimalDualSolver`, 3.6 times for `BarrierSolver`). The relative
duality gaps follow RadOptLIB's definition; its reference solution
scores 6.2e-9. `PrimalDualSolver` is the faster of Cvxium's solvers
here: it needs far fewer Newton steps than the barrier method's
nested centering.

Where the remaining time goes, about one second per Newton step: mostly
forming the dense corner G_b^T D G_b, a sparse product over the
21,717 dose rows with a 3,135 × 3,135 dense result, and then the
3,135 × 3,135 Cholesky factorization. Both are the intrinsically dense
beamlet block, so this is close to the floor for a direct method.

## Limits

- **Larger instances.** RadOptLIB's prostate instances have, by an
  estimate from their Q nonzeros, 15,000 to 30,000 beamlets. There
  the dense corner itself costs teraflops per Newton step, for Cvxium
  and for any solver that factors it. A matrix system that solves the
  corner iteratively (conjugate gradients with a low-rank
  preconditioner, using the decay of the dose matrix's singular
  values) would be the next structure to write.
- **Phase I** took about 40% of the primal-dual time in our runs. A
  cheaper strictly feasible start would help more than further linear
  algebra.
- **One instance.** These numbers are for rt-qp-0062 alone.

## Summary

A Newton system that is dense in a few thousand variables and cheap in
the rest is an arrow, and when its border is sparse, the arrow should
keep it so. The library does not need to know about radiotherapy:

1. **Presolve** pairs of opposed rows into equalities, or Phase I has
   no interior to find.
2. **When m ≫ n**, report G^T D G as a matrix, not as a rank-m update:
   a ten-line subclass of `LinearInequalityConstraintLayer`.
3. **Read the structure** of the Newton system off its blocks: which
   variables couple only diagonally, which rows couple many variables,
   and which block is irreducibly dense.
4. **Write a `MatrixSystem`** for that structure (`solve`, `multiply`,
   `scaled`, `_absorb`) and a **layer** that reports it, and report the
   objective's Hessian in the same shape. The solvers, Phase I
   included, need no changes.
