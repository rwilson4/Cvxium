# Implementing Fast Solvers in `pyrake.optimization`

This document describes how to implement a new solver using the building
blocks in this submodule.  The intended audience is AI agents, though humans
implementing new problem types will find the same guidance useful.

---

## 0. Start with a test case

Before writing any solver code, establish ground truth with `scipy.optimize.minimize`.
Use method `"trust-constr"` — it handles equality constraints and lower bounds
robustly and accepts an explicit Hessian callable.

```python
scipy_result = scipy.optimize.minimize(
    fun=lambda x: float(x @ Q @ x + c @ x),
    x0=x0_feas,
    method="trust-constr",
    jac=lambda x: 2.0 * Q @ x + c,
    hess=lambda x: 2.0 * Q,
    constraints=scipy.optimize.LinearConstraint(A, b, b),
    bounds=scipy.optimize.Bounds(lb=xl),
    options={"gtol": 1e-10, "maxiter": 2000},
)
if not scipy_result.success:
    pytest.skip(f"scipy reference solver failed: {scipy_result.message}")
```

Compare your solver's output against `scipy_result.x` with tolerances of
`rtol=1e-3, atol=1e-3`.  Tighter tolerances are achievable but these are
reliable across the full problem diversity in the test suite.  Additionally
verify:
- Equality constraints: `np.testing.assert_allclose(A @ result.solution, b, atol=1e-5)`
- Bound constraints: `assert np.all(result.solution >= xl - 1e-6)`
- Weak duality: `result.dual_value <= result.objective_value + 1e-6`

Time both solvers and print the ratio.  This makes performance regressions
visible in CI without requiring a separate benchmark suite.

---

## 1. Write out the math before writing any code

Work through the following four items in order.  Mistakes in the math
propagate invisibly into the code; catching them on paper is much cheaper.

### 1a. Lagrangian

Introduce one multiplier per constraint type:
- `lmbda >= 0` for each inequality `fi(x) <= 0`
- `nu` (free) for each equality `a_i^T x = b_i`

For the QP `minimize x^T Q x + c^T x  s.t.  A x = b, x >= xl`:

```
L(x, lmbda, nu) = x^T Q x + c^T x + lmbda^T (xl - x) + nu^T (A x - b)
```

### 1b. Dual function

Set `grad_x L = 0` and solve for `x*` in terms of the multipliers.  For the QP:

```
2 Q x + c - lmbda + A^T nu = 0
x* = -(1/2) Q^{-1} (c - lmbda + A^T nu)
```

Let `v = c - lmbda + A^T nu`.  Substituting back gives the dual function:

```
g(lmbda, nu) = -(1/4) v^T Q^+ v + lmbda^T xl - nu^T b
```

where `Q^+` is the Moore-Penrose pseudoinverse.  `g = -inf` when any
`lmbda_i < 0` or when `v` is outside `range(Q)` (unbounded Lagrangian).

The dual provides a lower bound on the primal objective at every iteration,
giving a **duality gap** that can certify convergence or infeasibility.

### 1c. Barrier problem (inequality → objective)

Each inequality `fi(x) <= 0` is replaced by `-log(-fi(x))`, scaled by `1/t`
(or equivalently the objective is scaled by `t`).  For bound constraints
`fi(x) = xl_i - x_i`:

```
ft(x) = t * f0(x) - sum_i log(x_i - xl_i)
subject to  A x = b
```

Gradient:
```
grad_ft_j = t * grad_f0_j - 1 / (x_j - xl_j)
```

Hessian:
```
H_ft = t * H_f0 + D,   D = diag(1 / (x - xl)^2)
```

**Key structural observation:** `D` is strictly positive definite whenever
`x` is strictly feasible (`x > xl`).  Therefore `H_ft` is strictly positive
definite even when `H_f0` (equivalently `Q`) is only PSD.  Newton's method
always has a unique step.

The outer loop increases `t` by `barrier_multiplier` (default 10) at each
centering step.  As `t → ∞`, the barrier terms shrink and the solution
approaches the true constrained optimum.

### 1d. Hessian and its structure

For a QP with `f0(x) = x^T Q x + c^T x`, `H_f0 = 2Q` and thus:

```
H_ft = 2t Q + D
```

Examine `Q` carefully for exploitable structure:

| Q structure | H_ft structure | Newton step cost |
|---|---|---|
| Diagonal `diag(q)` | `diag(2t q + d)` | O(n) |
| Low-rank `kappa kappa^T` | `D + kappa' kappa'^T`, `kappa'=sqrt(2t) kappa` | O(r²n + r³) via Woodbury |
| Diagonal + low-rank | `diag(2t q_d + d) + kappa' kappa'^T` | O(r²n + r³) via Woodbury |
| Full rank | Dense PD matrix | O(n³) Cholesky |

---

## 2. Implementing Q_solve and Q_vector_multiply

The solver accepts two optional callables to exploit Q's structure without
the framework needing to know what that structure is.

### Interface contract

```python
def Q_vector_multiply(
    v: npt.NDArray[np.float64],
    scale: float = 1.0,
    diag_add: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute (scale * Q + diag(diag_add)) @ v.

    Preconditions (guaranteed by the barrier framework):
      scale > 0
      diag_add >= 0 entry-wise (when provided)
    These together ensure (scale * Q + diag(diag_add)) is PSD.
    Implementing methods may assert these preconditions.
    """

def Q_solve(
    v: npt.NDArray[np.float64],
    scale: float = 1.0,
    diag_add: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Solve (scale * Q + diag(diag_add)) x = v.

    Preconditions (guaranteed by the barrier framework):
      scale > 0
      diag_add >= 0 entry-wise (when provided)
    """
```

When both `scale` and `diag_add` appear in the signature, the framework
detects this via `inspect.signature` and routes the Newton step through a
single call:

```python
hessian_solve(rhs) = Q_solve(rhs, scale=2t, diag_add=d)
```

This is the **highest-priority path** — it folds `Q` and the barrier
diagonal `D` into one structured solve, giving the maximum speedup.

### Why `scale` + `diag_add` matters

Without this interface, a `diag(q) + u u^T` callable would only get ~4x
speedup at `n=200` (the framework still forms `H_ft` separately and falls
through to the diagonal-probe path).  With the interface, the same callable
achieves ~12x speedup because it can absorb the barrier diagonal directly:

```
(2t * diag(q) + 2t * u u^T + D) x = v
(diag(2t*q + d) + (sqrt(2t)*u)(sqrt(2t)*u)^T) x = v   ← one Sherman-Morrison call
```

### Example: diagonal + rank-1 Q

```python
# Q = diag(q) + u @ u.T
alpha_fn = lambda eff_q: 1.0 / (1.0 + np.dot(u, u / eff_q))

def Q_vector_multiply(v, scale=1.0, diag_add=None):
    e = diag_add if diag_add is not None else 0.0
    if v.ndim == 1:
        return (scale * q + e) * v + scale * np.dot(u, v) * u
    return (scale * q + e)[:, np.newaxis] * v + scale * np.outer(u, u @ v)

def Q_solve(v, scale=1.0, diag_add=None):
    e = diag_add if diag_add is not None else 0.0
    eff_diag = scale * q + e          # D_e = diag(scale*q + e)
    coeff = scale / (1.0 + scale * np.dot(u, u / eff_diag))  # Sherman-Morrison
    if v.ndim == 1:
        v_d = v / eff_diag
        return v_d - coeff * np.dot(u, v_d) * (u / eff_diag)
    v_d = v / eff_diag[:, np.newaxis]
    return v_d - coeff * np.outer(u / eff_diag, u @ v_d)
```

Note that both callables must support 2-D `v` (matrix of right-hand sides)
for the KKT system solver, which solves for `A^T` all at once.

### Backward-compatible callables (no scale/diag_add)

If your callables lack `scale` and `diag_add`, the framework falls back to
existing paths:
1. If `Q_solve` is detected as diagonal (probed once with `e_0`): O(n) Newton step.
2. If Q is low-rank (detected via SVD at construction): O(r²n + r³) Woodbury.
3. Otherwise: O(n³) Cholesky fallback.

Always prefer the `scale`+`diag_add` interface for new code.

---

## 3. Exploit structure in the constraint-gradient matrix

When implementing `grad_constraints_multiply` and
`grad_constraints_transpose_multiply`, avoid forming the full `G` matrix.

For bound constraints `fi(x) = xl_i - x_i`, the full gradient matrix is
`G = -I` (n-by-n identity, negated).  Storing or multiplying by this is
O(n²).  Since `G @ y = -y` and `G^T @ y = -y`, both operations are O(n):

```python
def grad_constraints_multiply(self, x, y):
    return -y

def grad_constraints_transpose_multiply(self, x, y):
    return -y
```

More generally, examine the Jacobian of your constraints analytically.
Sparse or structured Jacobians (diagonal, banded, rank-deficient) should
always be exploited here rather than stored as dense matrices.

---

## 4. Exploit structure in the backtracking line search

`btls_keep_feasible` must return the largest step `s` keeping `x + s*delta_x`
strictly feasible.  For bound constraints:

```
xl_j - (x_j + s * delta_x_j) < 0
```

Only components with `delta_x_j < 0` can become infeasible.  The rest are
non-binding regardless of `s`.  Use a masked minimum rather than examining
all components:

```python
def btls_keep_feasible(self, x, delta_x):
    mask = delta_x < 0
    if not np.any(mask):
        return 1.0
    gaps = x - self.xl           # > 0 by strict feasibility
    ratios = gaps[mask] / (-delta_x[mask])
    return float(np.min(ratios))
```

This is O(n) and avoids allocating the full `(n,)` ratio array when most
components are non-binding.

For other constraint types, derive the maximum feasible step analytically
from the constraint structure before writing code.

---

## 5. The KKT system and the Schur complement

The equality-constrained Newton step solves:

```
| H_ft   A^T | | delta_x |   | -grad_ft |
|  A      0  | |   nu    | = |    0     |
```

`solve_kkt_system` in `numerical_helpers.py` implements the Schur complement
approach from Boyd & Vandenberghe Algorithm C.4:

1. Solve `H_ft B = A^T` (p solves with H_ft) and `H_ft b = g` (1 solve).
2. Form `S = -A B` (p×p Schur complement) and `c = -A b`.
3. Cholesky-factor S and solve `S nu = c` in O(p³).
4. Solve `H_ft delta_x = g - A^T nu`.

Total cost: **O(p²n + p³)** when `hessian_solve` costs O(n).  With p << n
(few equality constraints, many variables), this is far cheaper than the
naive O(n³) dense solve.  The framework provides this automatically; you
only need to supply a fast `hessian_solve`.

---

## 6. Construction-time precomputation

Expensive work that does not depend on the iterate `x` or barrier parameter
`t` should be done once in `__init__`, not per Newton step.

| Precomputation | Where | Cost |
|---|---|---|
| SVD of Q: `U_r, s_r` | `__init__` | O(n³), amortized |
| Square-root factor `kappa_cache = U_r * sqrt(s_r)` | `__init__` | O(rn) |
| Probe `Q_solve` with `e_0` to detect diagonal | `__init__` | O(n) |
| Detect `scale`/`diag_add` in callable signature | `__init__` | O(1) via `inspect.signature` |
| Phase I feasible point | `__init__` (via `EqualityWithBoundsSolver`) | O(p²n) |

Per-step work in `calculate_newton_step` should be limited to:
- Compute `d = 1/(x-xl)²` — O(n)
- Rescale `kappa = sqrt(2t) * kappa_cache` — O(rn)
- Call `hessian_solve` — O(n) or O(r²n) depending on path

---

## 7. The full checklist for a new solver

1. **Write the Lagrangian.** One multiplier per constraint.
2. **Derive the dual function.** Verify `g <= f0` at feasible points.
3. **Write the barrier problem.** One `-log(-fi(x))` term per inequality.
4. **Compute `H_ft`.** It is `H_f0 * t + D`. Identify structure.
5. **Implement `Q_vector_multiply` and `Q_solve`** with `scale`+`diag_add`
   support. Handle both 1-D and 2-D `v`.
6. **Implement `grad_constraints_multiply`** analytically. Avoid forming G.
7. **Implement `btls_keep_feasible`** with analytical per-component analysis.
8. **Write the ground-truth test** with scipy first, then add the PyRake solver.
9. **Time both** and verify a material speedup at `n ≥ 100`.
