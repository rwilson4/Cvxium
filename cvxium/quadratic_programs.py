"""Quadratic program solvers."""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .linear_programs import EqualityWithBoundsSolver
from .numerical_helpers import (
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_p_update,
)
from .optimization import (
    EqualityConstrainedInteriorPointMethodSolver,
    InteriorPointMethodResult,
    InteriorPointMethodSolver,
    OptimizationSettings,
    UnconstrainedNewtonSolver,
)


class QuadraticNewtonSolver(UnconstrainedNewtonSolver):
    r"""Solve min 0.5 * x^T Q x + c^T x, where Q is PSD.

    Optimal solution: x* = -Q^{-1} c.
    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        settings: OptimizationSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.Q = Q
        self.c = c
        self._Q_factor = linalg.cho_factor(Q)

    def newton_step(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve Q * delta_x = -grad_f0 = -(Q*x + c)."""
        return linalg.cho_solve(self._Q_factor, -(self.Q @ x + self.c))

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.Q @ x + self.c

    def hessian_vector_product(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.Q @ y


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


class QuadraticProgramEqualityBoundsSolver(
    EqualityConstrainedInteriorPointMethodSolver, InteriorPointMethodSolver
):
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
        settings: OptimizationSettings | None = None,
        Q_vector_multiply: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
        ) = None,
        Q_solve: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
        ) = None,
    ) -> None:
        phase1_solver = EqualityWithBoundsSolver(A=A, b=b, lb=xl, settings=settings)
        super().__init__(phase1_solver=phase1_solver, settings=settings)
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
    # Properties required by EqualityConstrainedInteriorPointMethodSolver
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
    # Properties required by BaseInteriorPointMethodSolver
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

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------

    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Multiply H_ft * y = (2t Q + D) y.

        Three paths, in priority order:

        1. ``Q_vector_multiply`` provided → ``2t * Q_vector_multiply(y) + d * y``.
           O(n) when Q_vector_multiply is O(n) (e.g. diagonal Q).
        2. kappa_cache available → ``2t * kappa_cache @ (kappa_cache^T @ y) + d * y``.
           O(rn) for rank-r Q.
        3. Fallback → ``2t * Q @ y + d * y``.  O(n^2).
        """
        d = 1.0 / np.square(x - self.xl)
        if self._Q_vector_multiply is not None:
            if self._Q_vector_multiply_has_structured_params:
                # Q_vector_multiply handles (scale * Q + diag(diag_add)) directly.
                # scale=2t > 0 and diag_add=d >= 0 are guaranteed by the barrier.
                return self._Q_vector_multiply(y, scale=2.0 * t, diag_add=d)  # type: ignore[call-arg]
            return 2.0 * t * self._Q_vector_multiply(y) + d * y
        if self._kappa_cache is not None:
            return 2.0 * t * (self._kappa_cache @ (self._kappa_cache.T @ y)) + d * y
        return 2.0 * t * (self.Q @ y) + d * y

    # ------------------------------------------------------------------
    # Newton step
    # ------------------------------------------------------------------

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step for the barrier sub-problem.

        Solves the KKT system:

           | H_ft   A^T | | delta_x |   | -grad_ft |
           |  A      0  | |   nu    | = |    0     |

        where H_ft = 2t Q + D, D = diag(1 / (x - xl)^2).

        Five solve paths, in priority order:

        **Q_solve supports scale + diag_add (highest priority)**
            Solves H_ft x = b as Q_solve(b, scale=2t, diag_add=d) where
            ``Q_solve(v, scale=s, diag_add=e)`` solves ``(s Q + diag(e)) x = v``.
            ``scale`` is guaranteed positive; ``diag_add`` entries are guaranteed
            non-negative.  Implementing methods may assert these preconditions.
            Q's internal structure is entirely opaque to the solver.

        **Q_solve provided AND diagonal Q**
            H_ft = 2t Q + D = diag(2t q + d), so the Hessian system reduces to
            elementwise division.  Using the identity

                H_ft^{-1} z = (I + M)^{-1} Q^{-1}(z) / (2t),
                M = Q^{-1} D / (2t),

            and the fact that M = diag(Q_solve(d) / (2t)) when Q is diagonal,
            (I + M)^{-1} is also diagonal → O(n) solve.

            For non-diagonal Q, Q^{-1}D is non-symmetric so (I+M) cannot be
            Cholesky-factored.  Those cases fall through to the existing paths.

        **Low-rank Q (r < n)**
            Woodbury identity with kappa_cache.  O(r^2 n + r^3) per step.

        **Full-rank Q (r = n) or non-diagonal Q_solve**
            Form H_ft explicitly and Cholesky-factor.  O(n^3) per step.

        """
        d = 1.0 / np.square(x - self.xl)

        _q_solve = self._Q_solve  # local alias so mypy can narrow the type
        if _q_solve is not None and self._Q_solve_has_structured_params:
            # Q_solve handles (scale * Q + diag(diag_add)) directly.
            # scale=2t > 0 and diag_add=d >= 0 are guaranteed by the barrier.
            two_t = 2.0 * t

            def hessian_solve(
                rhs: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return _q_solve(rhs, scale=two_t, diag_add=d)  # type: ignore[call-arg]

        elif _q_solve is not None and self._Q_solve_is_diagonal:
            # Q is diagonal → H_ft = diag(2tq + d).  O(n) solve.
            # From H_ft^{-1} z = (I+M)^{-1} Q_solve(z)/(2t) with M diagonal:
            #   (I+M)^{-1}_{ii} = 1 / (1 + Q_solve(d)_i / (2t))
            #   H_ft^{-1} z = Q_solve(z) / (2t + Q_solve(d))  (elementwise)
            inv_2t_plus_m = 1.0 / (2.0 * t + _q_solve(d))

            def hessian_solve(
                rhs: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                q_solved = _q_solve(rhs)
                if q_solved.ndim == 1:
                    return q_solved * inv_2t_plus_m
                return q_solved * inv_2t_plus_m[:, np.newaxis]

        elif self._kappa_cache is not None and self._kappa_cache.shape[1] < self._n:
            # Low-rank Q (r < n): Woodbury identity.  O(r^2 n + r^3) per step.
            eta_inverse = 1.0 / d  # (x_j - xl_j)^2; diagonal D^{-1}
            kappa = np.sqrt(2.0 * t) * self._kappa_cache  # n-by-r

            def hessian_solve(
                rhs: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return solve_rank_p_update(
                    rhs, kappa, solve_diagonal_eta_inverse, eta_inverse=eta_inverse
                )

        else:
            # Full-rank Q: form H_ft explicitly and Cholesky-factor.  O(n^3).
            H = 2.0 * t * self.Q + np.diag(d)
            H_factor = linalg.cho_factor(H)

            def hessian_solve(
                rhs: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return linalg.cho_solve(H_factor, rhs)

        return solve_kkt_system(
            A=self.A,
            g=-self.gradient_barrier(x, t),
            hessian_solve=hessian_solve,
        )

    # ------------------------------------------------------------------
    # Backtracking line search feasibility
    # ------------------------------------------------------------------

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
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
    # solve: force fully_optimize=True (inherited from InteriorPointMethodSolver)
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: object,
    ) -> InteriorPointMethodResult:
        """Solve the QP to optimality."""
        result = super().solve(x0=x0, fully_optimize=True, **kwargs)
        assert isinstance(result, InteriorPointMethodResult)
        return result
