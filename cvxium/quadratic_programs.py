"""Quadratic program solvers."""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .linear_programs import EqualityWithBoundsSolver
from .numerical_helpers import (
    multiply_diagonal,
    multiply_rank_p_update,
    solve_diagonal_eta_inverse,
    solve_rank_p_update,
)
from .optimization import (
    EqualityConstrainedNewtonProblem,
    EqualityConstrainedProblem,
    InteriorPointMethodResult,
    InteriorPointSolver,
    OptimizationSettings,
    UnconstrainedNewtonProblem,
)
from .systems import DenseSystem, KKTSystem, System


class QuadraticNewtonSolver(UnconstrainedNewtonProblem):
    r"""Solve min 0.5 * x^T Q x + c^T x, where Q is PSD.

    Optimal solution: x* = -Q^{-1} c.
    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
    ) -> None:
        """Build an unconstrained quadratic program."""
        super().__init__()
        self.Q = Q
        self.c = c

    def centering_system(self, x: npt.NDArray[np.float64], t: float) -> System:
        """The Hessian of the barrier objective is ``t * Q``."""
        return DenseSystem(t * self.Q)

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.Q @ x + self.c


class QuadraticEqualityConstrainedNewtonSolver(EqualityConstrainedNewtonProblem):
    r"""Solve min 0.5 * x^T Q x + c^T x subject to A x = b, where Q is PD.

    Optimal solution satisfies the KKT conditions:
        Q x* + c + A^T nu* = 0
        A x* = b.
    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> None:
        """Build an equality-constrained quadratic program."""
        super().__init__()
        self.Q = Q
        self.c = c
        self._A = A
        self._b = b
        self._Q_factor = linalg.cho_factor(Q)

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Equality constraint matrix."""
        return self._A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Equality constraint right-hand side."""
        return self._b

    def centering_system(self, x: npt.NDArray[np.float64], t: float) -> System:
        """The KKT system: Hessian ``t * Q`` bordered by the equality matrix A."""
        return KKTSystem(DenseSystem(t * self.Q), self._A)

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.Q @ x + self.c

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate the Lagrangian dual function g(nu).

        The Lagrangian is L(x, nu) = 0.5 x^T Q x + c^T x + nu^T (A x - b).
        Minimizing over x gives x*(nu) = -Q^{-1}(c + A^T nu), and:

            g(nu) = -0.5 (c + A^T nu)^T Q^{-1} (c + A^T nu) - nu^T b.

        """
        v = self.c + self._A.T @ nu
        Q_inv_v = linalg.cho_solve(self._Q_factor, v)
        return float(-0.5 * np.dot(v, Q_inv_v) - np.dot(self._b, nu))


def _callable_has_structured_params(fn: Callable[..., Any] | None) -> bool:
    """Return True if *fn* declares both ``scale`` and ``diag_add`` parameters.

    Used to detect whether a user-supplied Q_solve or Q_vector_multiply
    supports the structured ``(scale * Q + diag(diag_add))`` interface.
    When both parameters are present the Newton step can call the callable
    directly with the barrier diagonal, avoiding any separate treatment of D.
    """
    if fn is None:
        return False
    sig = inspect.signature(fn)
    return "scale" in sig.parameters and "diag_add" in sig.parameters


class _QPBarrierHessian(System):
    """The barrier Hessian ``H_ft = 2t Q + D`` of a bound-constrained QP.

    ``Q`` is reached only through whatever representation the solver holds -- a
    dense matrix, a low-rank SVD factor, or opaque ``Q_solve`` /
    ``Q_vector_multiply`` callables -- so this bespoke System dispatches over
    those representations rather than composing generic System primitives.
    """

    def __init__(
        self,
        solver: "QuadraticProgramEqualityBoundsSolver",
        x: npt.NDArray[np.float64],
        t: float,
    ) -> None:
        """Build the barrier Hessian System at ``(x, t)``."""
        self._solver = solver
        self._t = t
        d = 1.0 / np.square(x - solver.xl)
        self._d = d
        two_t = 2.0 * t
        q_solve = solver._Q_solve

        if q_solve is not None and solver._Q_solve_has_structured_params:

            def hessian_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return q_solve(rhs, scale=two_t, diag_add=d)  # type: ignore[call-arg]

        elif q_solve is not None and solver._Q_solve_is_diagonal:
            inv_2t_plus_m = 1.0 / (two_t + q_solve(d))

            def hessian_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                q_solved = q_solve(rhs)
                if q_solved.ndim == 1:
                    return q_solved * inv_2t_plus_m
                return q_solved * inv_2t_plus_m[:, np.newaxis]

        elif (
            solver._kappa_cache is not None and solver._kappa_cache.shape[1] < solver._n
        ):
            eta_inverse = 1.0 / d
            kappa = np.sqrt(two_t) * solver._kappa_cache

            def hessian_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return solve_rank_p_update(
                    rhs, kappa, solve_diagonal_eta_inverse, eta_inverse=eta_inverse
                )

        else:
            factor = linalg.cho_factor(two_t * solver.Q + np.diag(d))

            def hessian_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                solved: npt.NDArray[np.float64] = linalg.cho_solve(factor, rhs)
                return solved

        self._hessian_solve = hessian_solve

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._solver._n

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``H_ft @ x = b``."""
        return self._hessian_solve(b)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``H_ft @ y = (2t Q + D) y``."""
        solver = self._solver
        t = self._t
        d = self._d
        q_multiply = solver._Q_vector_multiply
        if q_multiply is not None:
            if solver._Q_vector_multiply_has_structured_params:
                return q_multiply(y, scale=2.0 * t, diag_add=d)  # type: ignore[call-arg]
            return 2.0 * t * q_multiply(y) + multiply_diagonal(y, d)
        if solver._kappa_cache is not None:
            kappa = np.sqrt(2.0 * t) * solver._kappa_cache
            return multiply_rank_p_update(y, kappa, multiply_diagonal, eta=d)
        return multiply_diagonal(y, d) + 2.0 * t * (solver.Q @ y)


class QuadraticProgramEqualityBoundsSolver(EqualityConstrainedProblem):
    r"""Solve a quadratic program with equality and bound constraints.

    Solves:
      minimize   x^T Q x + c^T x
      subject to A x = b
                 x >= xl

    where Q is positive semi-definite (PSD).

    Parameters
    ----------
     Q : (n, n) array
        Positive semi-definite quadratic cost matrix.
     c : (n,) array
        Linear cost vector.
     A : (p, n) array
        Equality constraint matrix.
     b : (p,) array
        Equality constraint right-hand side.
     xl : float or (n,) array, optional
        Lower bounds on x. Defaults to 0.
     settings : OptimizationSettings, optional
        Optimization settings.
     Q_vector_multiply : callable, optional
        If provided, called as ``Q_vector_multiply(v)`` and must return ``Q @ v``
        for any 1-D or 2-D array ``v``.  Replaces the dense matrix-vector product
        ``Q @ v`` in the gradient, objective, and barrier Hessian-vector product,
        enabling O(n) cost when Q has exploitable structure (e.g. diagonal).
        When supplied together with ``Q_solve`` the O(n³) SVD precomputation is
        skipped entirely.
     Q_solve : callable, optional
        If provided, called as ``Q_solve(v)`` and must return ``Q^{-1} v`` for
        any 1-D or 2-D array ``v``.  Implies Q is positive definite.  Used in
        ``evaluate_dual`` (replaces the SVD pseudoinverse) and in
        ``calculate_newton_step`` via the identity

            H_ft^{-1} z = (I + Q^{-1} D / (2t))^{-1} Q^{-1} z / (2t),

        where D = diag(1/(x - xl)^2).  For diagonal Q the inner factor reduces
        to a scalar elementwise divide, giving an O(n) Newton step.

    Notes
    -----

    **Lagrangian**

    The Lagrangian (with lmbda >= 0 for bound constraints and nu free for
    equality constraints) is:

       L(x, lmbda, nu) = x^T Q x + c^T x + lmbda^T (xl - x) + nu^T (A x - b)

    **Lagrangian dual function**

    Taking the gradient wrt x and setting to zero:

       2 Q x + c - lmbda + A^T nu = 0
       => x* = -(1/2) Q^{-1} (c - lmbda + A^T nu)

    Let v = c - lmbda + A^T nu. Since Q is PSD, the infimum over x is
    finite only when v lies in the range of Q; otherwise g = -inf.
    When v is in range(Q) the minimizer is unique along the row space of Q
    and:

       g(lmbda, nu) = -(1/4) v^T Q^+ v + lmbda^T xl - nu^T b

    where Q^+ is the Moore-Penrose pseudoinverse of Q. The dual is also
    -inf when any lmbda_i < 0.

    **Barrier problem**

    The inequality constraints x_i >= xl_i are absorbed via a log barrier:

       ft(x) = t * (x^T Q x + c^T x) - sum_i log(x_i - xl_i)
       subject to A x = b

    Gradient of ft:
       grad_ft_j = t * (2(Qx)_j + c_j) - 1 / (x_j - xl_j)

    Hessian of ft:
       H_ft = 2t Q + D,   D = diag(1 / (x - xl)^2)

    The Hessian is strictly positive definite whenever x is strictly feasible
    (x > xl), regardless of whether Q is PD or only PSD, because D alone is
    already strictly positive definite.

    **Newton step**

    The equality-constrained Newton step solves the KKT system:

       | H_ft   A^T | | delta_x |   | -grad_ft |
       |  A      0  | |   nu    | = |    0     |

    Writing Q = U_r S_r U_r^T (precomputed rank-r SVD) and defining the
    cached factor kappa_cache = U_r * sqrt(S_r) (n-by-r, computed once),
    we have Q = kappa_cache @ kappa_cache^T.  Setting kappa = sqrt(2t) *
    kappa_cache (recomputed cheaply at each step, O(rn)) gives

       H_ft = D + kappa @ kappa^T.

    The Hessian system (D + kappa kappa^T) y = z is then solved via the
    Woodbury identity using `solve_rank_p_update`:

       H_ft^{-1} z = D^{-1} z
                     - D^{-1} kappa (I_r + kappa^T D^{-1} kappa)^{-1} kappa^T D^{-1} z

    Cost per Newton step: O(r^2 n + r^3).  For full-rank PD Q (r = n) this
    is the same O(n^3) as a direct Cholesky of H_ft.  For low-rank PSD Q
    (r << n) it is substantially cheaper.

    **Phase I**

    `EqualityWithBoundsSolver` finds the initial strictly feasible point
    satisfying A x = b and x > xl.

    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        xl: float | list[float] | npt.NDArray[np.float64] = 0.0,
        Q_vector_multiply: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
        ) = None,
        Q_solve: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
        ) = None,
    ) -> None:
        super().__init__(phase1_problem=EqualityWithBoundsSolver(A=A, b=b, lb=xl))
        self.Q = Q
        self.c = c
        self._A = A
        self._b = b
        self.xl = xl
        self._n = Q.shape[0]
        self._Q_vector_multiply = Q_vector_multiply
        self._Q_solve = Q_solve
        self._Q_solve_is_diagonal = False
        self._Q_solve_has_structured_params = _callable_has_structured_params(Q_solve)
        self._Q_vector_multiply_has_structured_params = _callable_has_structured_params(
            Q_vector_multiply
        )

        if Q_vector_multiply is not None and Q_solve is not None:
            # Both callables supplied: skip the O(n³) SVD entirely.
            # Q_solve implies Q is PD, so every v is in range(Q) and the
            # pseudoinverse equals the true inverse — no range-check needed.
            self._Q_svd_U_r: npt.NDArray[np.float64] | None = None
            self._Q_svd_s_r: npt.NDArray[np.float64] | None = None
            self._kappa_cache: npt.NDArray[np.float64] | None = None

            # Probe Q_solve with e₀ to detect diagonal structure (one O(n) call).
            # For diagonal Q, Q_solve(e₀) = e₀/q₀, which is proportional to e₀.
            e0 = np.zeros(self._n)
            e0[0] = 1.0
            probe = Q_solve(e0)
            self._Q_solve_is_diagonal = bool(np.allclose(probe[1:], 0.0, atol=1e-10))
        else:
            # Precompute the economy SVD of Q (Q is PSD so U == V).
            # We retain only the rank-r subspace, which also handles PSD Q where
            # some singular values are numerically zero.
            U, s, _ = linalg.svd(Q, full_matrices=False)
            rank_tol = max(Q.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
            rank = int(np.sum(s > rank_tol))
            self._Q_svd_U_r = U[:, :rank]
            self._Q_svd_s_r = s[:rank]

            # Cached square-root factor: kappa_cache @ kappa_cache^T == Q.
            # Shape: n-by-r.  Used in hessian_multiply and calculate_newton_step
            # so that we never re-process Q at runtime; only a cheap O(r*n) scalar
            # multiply (to absorb sqrt(2t)) is needed per Newton step.
            self._kappa_cache = U[:, :rank] * np.sqrt(s[:rank])

            if Q_solve is not None:
                # Q_solve without Q_vector_multiply: probe once for diagonal detection.
                e0 = np.zeros(self._n)
                e0[0] = 1.0
                probe = Q_solve(e0)
                self._Q_solve_is_diagonal = bool(
                    np.allclose(probe[1:], 0.0, atol=1e-10)
                )

    # ------------------------------------------------------------------
    # Properties required by EqualityConstrainedProblem
    # ------------------------------------------------------------------

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Equality constraint matrix."""
        return self._A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Equality constraint right-hand side."""
        return self._b

    # ------------------------------------------------------------------
    # Properties required by InteriorPointProblem
    # ------------------------------------------------------------------

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self._A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality (bound) constraints."""
        return self._n

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _q_multiply(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return Q @ v, using the callable when available."""
        if self._Q_vector_multiply is not None:
            return self._Q_vector_multiply(v)
        return self.Q @ v

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate f0(x) = x^T Q x + c^T x."""
        return float(x @ self._q_multiply(x) + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of f0: grad_f0 = 2 Q x + c."""
        return 2.0 * self._q_multiply(x) + self.c

    # ------------------------------------------------------------------
    # Constraints: fi(x) = xl_i - x_i <= 0
    # ------------------------------------------------------------------

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate fi(x) = xl_i - x_i (each <= 0 when x >= xl)."""
        return self.xl - x  # broadcasts for scalar or array xl

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient matrix of constraints: row i is grad fi = -e_i, so G = -I."""
        return -np.eye(self._n)

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Compute G @ y in O(n) time, where G is the constraint-gradient matrix.

        The n inequality constraints are fi(x) = xl_i - x_i, so

            grad fi(x) = -e_i   (i-th standard basis vector),

        giving the n-by-n gradient matrix G = -I. Thus G @ y = -y.
        This avoids forming the n-by-n identity and doing a full matrix-vector
        multiply (which would be O(n²)).

        """
        return -y

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Compute G^T @ y in O(n) time.

        Since G = -I we have G^T = -I, so G^T @ y = -y.

        """
        return -y

    def centering_system(self, x: npt.NDArray[np.float64], t: float) -> System:
        """Assemble the KKT System for the Newton step.

        The barrier Hessian ``H_ft = 2t Q + D`` is wrapped in a bespoke System
        (:class:`_QPBarrierHessian`) and bordered by the equality matrix A.
        """
        return KKTSystem(_QPBarrierHessian(self, x, t), self.A)

    # ------------------------------------------------------------------
    # Backtracking line search feasibility
    # ------------------------------------------------------------------

    def btls_keep_feasible(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        settings: OptimizationSettings,
    ) -> float:
        r"""Return the largest step s keeping x + s * delta_x strictly feasible.

        Strict feasibility requires fi(x + s * delta_x) < 0 for all i, i.e.

            xl_j - (x_j + s * delta_x_j) < 0
            ⟺  s * delta_x_j > xl_j - x_j    for all j.

        Three cases:

        * delta_x_j > 0: the left side grows with s, and since x_j > xl_j
          (strict feasibility of the current iterate) the inequality already
          holds at s = 0 and remains satisfied for all s > 0. No constraint.

        * delta_x_j = 0: reduces to 0 > xl_j - x_j, which holds by strict
          feasibility. No constraint.

        * delta_x_j < 0: dividing by delta_x_j flips the inequality:

              s < (xl_j - x_j) / delta_x_j
                = (x_j - xl_j) / (-delta_x_j).

          Both numerator and denominator are positive, giving a positive
          upper bound on s.

        The tightest bound is therefore

            s_max = min_{j : delta_x_j < 0}  (x_j - xl_j) / (-delta_x_j).

        This is computed in O(n) time with a single masked minimum.

        """
        mask = delta_x < 0
        if not np.any(mask):
            return 1.0

        gaps = x - self.xl  # (x_j - xl_j) > 0 for strictly feasible x
        ratios = gaps[mask] / (-delta_x[mask])
        return float(np.min(ratios))

    # ------------------------------------------------------------------
    # Dual function
    # ------------------------------------------------------------------

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate the Lagrangian dual function.

        The dual function (derived by minimizing the Lagrangian over x) is:

            g(lmbda, nu) = -(1/4) v^T Q^+ v + lmbda^T xl - nu^T b

        where v = c - lmbda + A^T nu and Q^+ is the Moore-Penrose pseudoinverse
        of Q. The dual is -infinity when any lmbda_i < 0, or when v does not
        lie in the range of Q (so the infimum over x is -infinity).

        When ``Q_solve`` is provided Q is PD so Q^+ = Q^{-1} and every v is in
        range(Q); the range check and SVD pseudoinverse are both skipped.

        """
        if np.any(lmbda < 0):
            return -np.inf

        v = self.c - lmbda + self.A.T @ nu
        base = np.sum(lmbda * self.xl) - np.dot(self.b, nu)

        if self._Q_solve is not None:
            # Q is PD: Q^+ = Q^{-1}, every v is in range(Q).
            Q_inv_v = self._Q_solve(v)
            return float(-0.25 * np.dot(v, Q_inv_v) + base)

        # General PSD path: check range and use SVD pseudoinverse.
        # Project v onto the range of Q.  If v has a non-trivial component in
        # the null space of Q, the Lagrangian is unbounded below and g = -inf.
        assert self._Q_svd_U_r is not None and self._Q_svd_s_r is not None
        v_proj = self._Q_svd_U_r @ (self._Q_svd_U_r.T @ v)
        if not np.allclose(v_proj, v, atol=1e-6):
            return -np.inf

        # Q^+ v = U_r diag(1/s_r) U_r^T v  (Q is symmetric so U == V)
        Q_pinv_v = self._Q_svd_U_r @ ((self._Q_svd_U_r.T @ v) / self._Q_svd_s_r)
        return float(-0.25 * np.dot(v, Q_pinv_v) + base)

    # ------------------------------------------------------------------
    # solve: narrow the result type to InteriorPointMethodResult
    # ------------------------------------------------------------------

    def solve(
        self,
        solver: InteriorPointSolver | None = None,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: object,
    ) -> InteriorPointMethodResult:
        """Solve the QP to optimality."""
        result = super().solve(
            solver=solver, x0=x0, fully_optimize=fully_optimize, **kwargs
        )
        assert isinstance(result, InteriorPointMethodResult)
        return result
