"""Linear program solvers."""

import time
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import NewtonStepError, ProblemInfeasibleError
from .numerical_helpers import (
    solve_block_plus_one,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_p_update,
)
from .optimization import (
    EqualityConstrainedInteriorPointMethodSolver,
    NewtonResult,
    OptimizationResult,
    OptimizationSettings,
    PhaseIInteriorPointSolver,
    PhaseISolver,
    ProblemCertifiablyInfeasibleError,
)


class EqualitySolver(PhaseISolver):
    """Find x satisfying A * x = b."""

    def __init__(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        settings: OptimizationSettings | None = None,
    ) -> None:
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

        p, _ = A.shape
        assert len(b.shape) == 1
        assert b.shape[0] == p
        self.A = A
        self.b = b

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return 0

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> OptimizationResult:
        r"""Solve A * x = b.

        Parameters
        ----------
         x0 : vector, optional
            Initial guess. See Notes.

        Returns
        -------
         res : OptimizationResult
            The solution.

        Notes
        -----
        If x0 is feasible (that is, A * x0 = b), we simply return it.

        Otherwise, if x0 is passed, we solve:
           minimize    \| x - x0 \|_2^2
           subject to  A * x = b.
        There is an analytic solution to this problem. The Lagrangian is:
            L(x, nu) = \| x - x0 \|_2^2 + nu^T * (A * x - b).
        The dual function is:
            g(nu) = inf_x {L(x, nu)}.
        The x that achieves this infimum satisfies:
            2 * (x - x0) + A^T * nu = 0, or
            x = x0 - (1/2) * A^T * nu.
        Thus, the dual function is:
            -(1/4) * nu^T * A * A^T * nu + nu^T * (A * x0 - b).
        The dual problem is:
            maximize g(nu),
        whose solution satisfies:
            (A * A^T) * nu_star = 2 * (A * x0 - b).
        We do *not* assume A is full rank, but instead use the SVD to find such a
        nu_star. Then the solution is:
            x_star = x0 - (1/2) * A^T * nu_star.

        When x0 is not passed, we use the SVD to find the minimum norm solution to A * x
        = b.

        When x0 is passed, it takes a little longer to calculate a feasible x (less than
        twice as long), but this part is super fast anyway, and lets us stay true to the
        point passed.

        """
        if x0 is not None and np.all(np.abs(self.A @ x0 - self.b) < 1e-10):
            if self.settings.verbose:
                print("  Initial guess was feasible.")
            return OptimizationResult(solution=x0)

        # If A = U * s * Vh, then we seek to solve:
        # U * s * Vh * x = b
        #
        # Then s * Vh * x = U^T * b
        # Vh * x = (U^T * b) / s
        # x = Vh^T * (U^T * b) / s
        if self.settings.verbose:
            start_time = time.time()

        U, s, Vh = self.svd_A()
        rank = int(np.sum(s > 1e-10))
        U_r = U[:, 0:rank]
        if not np.allclose(U_r @ (U_r.T @ self.b), self.b):
            raise ProblemInfeasibleError("b was not in the range of A.")

        if x0 is not None:
            # Step 1: find nu such that
            # (A * A^T) nu = 2 * (A * x0 - b) =: rhs
            # Since A = U * diag(s) * V^T, we have:
            #   U * diag(s^2) * U^T * nu = rhs
            #   U^T * nu = diag(s^{-2}) * (U * rhs).
            # In general, U * U^T does not equal the identity matrix (unless A is full rank),
            # but if we define nu_star = U * diag(s^{-2}) * (U * rhs), then
            #   U^T * nu_star = diag(s^{-2}) * (U * rhs).
            # This works even when A is not full rank.
            rhs = 2.0 * (self.A @ x0 - self.b)
            nu_star = U_r @ ((U_r.T @ rhs) / np.square(s[0:rank]))

            # Step 2: Calculate x as x0 - (1/2) * A^T nu_star.
            x = x0 - 0.5 * (self.A.T @ nu_star)
        else:
            s_inv = np.zeros_like(s)
            s_inv[0:rank] = 1.0 / s[0:rank]
            x = Vh.T @ (s_inv * (U.T @ self.b))

        if self.settings.verbose:
            end_time = time.time()
            print(f"  SVD calculated in {1000 * (end_time - start_time):.03f} ms")

        return OptimizationResult(solution=x)

    @cache  # noqa: B019
    def svd_A(  # noqa: N802
        self,
    ) -> tuple[
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
    ]:
        """Calculate and cache SVD of A.

        When A is large with a high "aspect ratio", that is a huge number of weights but
        a modest number of constraints, it pays to first calculate the QR factorization
        of A^T, then compute the SVD of R. R will have dimension corresponding to the
        number of constraints, so computing the SVD of R is much faster than computing
        the SVD of A. But empirically, this only starts to pay off when we have around
        10,000 weights. When the number of weights is small (e.g. 100), just directly
        calculating the SVD of A is faster. We check the shape of A and choose
        appropriately.

        """
        if self.A.shape[1] < 1_000 or self.A.shape[0] > 0.1 * self.A.shape[1]:
            return linalg.svd(self.A, full_matrices=False)

        # Compute the QR factorization of A^T = Q*R
        Q, R = linalg.qr(self.A.T, mode="economic")

        # Compute the SVD of R = U * diag(s) * Vh
        U, s, Vh = linalg.svd(R)

        # SVD of A is R^T * Q^T = V * diag(s) * U^T * Q^T
        #                       = V * diag(s) * (Q * U)^T
        QU = Q @ U
        return Vh.T, s, QU.T


class EqualityWithBoundsSolver(PhaseIInteriorPointSolver):
    r"""Find x satisfying A * x = b and x > lb.

    We do this by solving:
      minimize   s
      subject to A * x = b
                 lb - x_i <= s
                 s <= s0 + eps,

    equivalent to the standard form:
      minimize   s
      subject to A * x = b
                 -x_i - (s - lb) <= 0
                 s - (s0 + eps) <= 0.

    We initialize s as s0 := -x.min() + lb + eps, guaranteeing the starting point is
    strictly feasible. The last constraint is needed to make the Hessian of the inner
    problem strictly positive definite. If the resulting s^\star is > 0, the original
    problem is infeasible.

    Parameters
    ----------
     phase1_solver : EqualitySolver, optional
        This class requires a Phase I solver for the equality constraints, and there are
        two ways of generating this. The user can either pass an EqualitySolver, or pass
        A and b and an EqualitySolver will be initialized.
     A, b : npt.NDArray
        Equality constraints: A * x = b. Required when phase1_solver is None.
     lb : float or list[float], optional
        Lower bound on elements of x. Defaults to 0.
     settings : OptimizationSettings
        Optimization settings.

    """

    def __init__(
        self,
        phase1_solver: EqualitySolver | None = None,
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        lb: float | list[float] | npt.NDArray[np.float64] = 0.0,
        settings: OptimizationSettings | None = None,
    ) -> None:
        if phase1_solver is not None:
            super().__init__(phase1_solver=phase1_solver, settings=settings)
        elif A is not None and b is not None:
            super().__init__(
                phase1_solver=EqualitySolver(A, b, settings=settings), settings=settings
            )
        else:
            raise ValueError("Must specify `A` and `b`.")

        self.lb = lb
        self.s0_plus_eps = 0.0

    @property
    def dimension(self) -> int:
        """Problem dimension."""
        return self.A.shape[1]

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return self.A.shape[1] + 1

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wrap A."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualitySolver):
            raise ValueError("PhaseISolver must be an EqualitySolver.")

        return self.phase1_solver.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Wrap b."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualitySolver):
            raise ValueError("PhaseISolver must be an EqualitySolver.")

        return self.phase1_solver.b

    def svd_A(  # noqa: N802
        self,
    ) -> tuple[
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
    ]:
        """Calculate and cache SVD of A."""
        if self.phase1_solver is None:
            raise ValueError("phase1_solver is required.")

        if not isinstance(self.phase1_solver, EqualitySolver):
            raise ValueError("phase1_solver must be an EqualitySolver")

        return self.phase1_solver.svd_A()

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""
        return bool(np.all(x[0 : self.dimension] > self.lb))

    def check_for_infeasibility(self, result: NewtonResult) -> None:
        """Check if infeasible."""
        if result.dual_value > 0:
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    "Problem certifiably infeasible: dual value was "
                    f"{result.dual_value} > 0"
                ),
                result=result,
            )

    def augment_previous_solution(
        self,
        phase1_res: OptimizationResult,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Initialize variable based on Phase I result."""
        x = np.zeros((len(phase1_res.solution) + 1,))
        x[0 : self.dimension] = phase1_res.solution
        eps = 1.0
        x[self.dimension] = np.max(self.lb - phase1_res.solution) + eps
        self.s0_plus_eps = x[self.dimension] + eps
        return x

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """De-augment solution."""
        return x[0 : self.dimension]

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        r"""Initialize barrier parameter.

        We try 2 different approaches to initializing the barrier parameter, based on
        the discussion in Boyd and Vandenberghe, section 11.3.1.
          1. Choose t0 so that m / t0 is of the same order as f0(x0) - p_star,
             where m is the number of inequality constraints. Since f0 = s, and the
             happy path is p_star < 0 for a feasible solution, f0(x0) - p_star > s0.
             Thus, we want m / t0 > s0, or t0 < m / s0.
          2. Choose t0 to minimize the deviation of x0 from x^\star(t0). This ends up
             being a least squares equation (Eq 11.12), and is quite a bit more
             complicated, but ends up having a fairly simple formula.

        """
        M = self.dimension
        t1 = self.num_ineq_constraints / max(self.settings.outer_tolerance, x0[M])
        t2 = np.sum(1.0 / (x0[0:M] - self.lb + x0[M])) - M / 1.0
        return max(1.0, t1, t2)

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(w) := t * -s - \sum_i log(w_i - s)
          subject to A * w = b.

        Parameters
        ----------
         x : vector
            Current estimate, [w; s].
         t : float
            Barrier parameter.

        Returns
        -------
         delta_x : vector
            Newton step.
         nu : vector
            Lagrange multiplier associated with equality constraints.

        Notes
        -----
        The Newton step, delta_x, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_x |   | - grad_ft |
          | A    0  | |   nu    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at x, grad_f is the gradient of ft
        evaluated at x, and nu is the Lagrange multiplier associated with the equality
        constraints. We use `solve_kkt_system` to solve this system in O(p^3 + p^2 * M)
        time.

        """
        return solve_kkt_system(
            A=np.hstack((self.A, np.zeros((self.num_eq_constraints, 1)))),
            g=-self.gradient_barrier(x, t),
            hessian_solve=EqualityWithBoundsSolver._solve_arrow_sparsity_pattern,
            eta_inverse=self._hessian_ft_diagonal_inverse(x, t),
            one_over_psi_squared=self._hessian_one_over_psi_squared(x),
        )

    @staticmethod
    def _solve_arrow_sparsity_pattern(
        b: npt.NDArray[np.float64],
        eta_inverse: npt.NDArray[np.float64],
        one_over_psi_squared: float,
    ) -> npt.NDArray[np.float64]:
        """Solve H * x = b.

        Solves a linear system of equations where H has an arrow sparsity pattern:
             _                 _
            |  diag(eta)   eta  |
        H = |                   |
            |_   eta^T   theta _|

        Because of this structure, we can solve the system in linear time. See Notes for
        more details.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right hand side. Can be either a vector or a matrix, in which case we solve the
            system for each column of b.
         eta_inverse : npt.NDArray[np.float64]
            One divided by the diagonal elements of the upper left block of H.
         one_over_psi_squared : float
            1.0 / (theta - np.sum(eta))

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution.

        Notes
        -----
        Like `solve_arrow_sparsity_pattern`, but for the specific instance used to solve:
          minimize    s
          subject to  A * x = b
                      -x <= s.

        In this case, eta[i]^{-1} = (x_i + s)^2, diag_eta_inverse_dot_zeta[i] = 1.0, and
        1 / psi_squared = (s0 + eps - s)^2 / M. Thus, we can solve the system both faster
        and with more numerical stability.

        `solve_arrow_sparsity_pattern` uses 2*M + 1 divides, 3*M multiplies, and 3*M adds,
        or 8*M + 1 flops. `solve_arrow_sparsity_pattern_phase1` uses 0 divides, M + 1
        multiplies, and 2 * M + 1 adds, or 3*M + 2 flops.

        """
        if not np.all(eta_inverse > 0) or one_over_psi_squared <= 0:
            raise NewtonStepError("Hessian is not strictly positive definite.")

        M = eta_inverse.shape[0]
        # Calculate x
        x = np.zeros_like(b)
        if b.ndim == 1:
            if b.shape[0] != M + 1:
                raise ValueError(
                    "Dimension mismatch: b must have M + 1 entries, where M = len(eta)."
                )
            x[M] = (b[M] - np.sum(b[0:M])) * one_over_psi_squared
            x[0:M] = b[0:M] * eta_inverse - x[M]
        elif b.ndim == 2:
            if b.shape[0] != M + 1:
                raise ValueError(
                    "Dimension mismatch: b must have M + 1 rows, where M = len(eta)."
                )
            x[M, :] = (b[M, :] - np.sum(b[0:M, :], axis=0)) * one_over_psi_squared
            x[0:M, :] = b[0:M, :] * eta_inverse[:, np.newaxis] - x[M, :]
        else:
            raise ValueError(
                "Dimension mismatch: b must be either a 1D or 2D NumPy array."
            )

        return x

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + btls_s * delta_x stays strictly feasible.

        In our case, we want a btls_s such that
           -(w + btls_s * delta_w) < s + btls_s * delta_s - lb, and
           s + btls_s * delta_s < s0 + eps.

        The first criterion is equivalent to:
            (w + s) + btls_s * (delta_w + delta_s) > lb,
        which is automatically satisfied whenever delta_w + delta_s >= 0, since w and s
        are feasible (so w + s > lb). So we're only concerned with entries i having
        delta_w[i] + delta_s < 0, in which case we need
            btls_s < (w[i] + s - lb) / -(delta_w[i] + delta_s),
        so we set btls_s as the minimum of these quantities. If all entries satisfy
        delta_w + delta_s >= 0, this constraint is not active.

        Similarly, if delta_s <= 0, there's no problem since s is strictly feasible. But
        if delta_s > 0, we want btls_s < ((s0 + eps) - s) / delta_s

        """
        M = self.dimension
        w = x[0:M]
        s = x[M]
        delta_w = delta_x[0:M]
        delta_s = delta_x[M]

        # Calculate an initial step size btls_s satisfying:
        #   btls_s <= 1
        #   w + btls_s * delta_w > -s - btls_s * delta_s + lb
        #   s + btls_s * delta_s < s0 + eps
        mask = (delta_w + delta_s) < 0
        feasibility = np.full_like(delta_w, np.inf, dtype=np.float64)
        lb = self.lb
        if isinstance(lb, list | tuple | np.ndarray):
            feasibility[mask] = (w[mask] + s - lb[mask]) / -(delta_w[mask] + delta_s)
        else:
            feasibility[mask] = (w[mask] + s - lb) / -(delta_w[mask] + delta_s)
        btls_s = min(
            np.min(feasibility),
            (self.s0_plus_eps - s) / delta_s if delta_s > 0 else np.inf,
        )

        return float(btls_s)

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at w."""
        return x[self.dimension]

    def evaluate_barrier_objective(self, x: npt.NDArray[np.float64], t: float) -> float:
        r"""Calculate ft at x.

        Our barrier objective is:
           ft(x, s) = t * s - \sum_i log(x_i + s - lb) - M * log(s0 + eps - s).

        This differs from the nominal barrier objective in that we give additional
        weight to the last constraint to help it compete with the others. Otherwise,
        it's too easy to decrease the barrier objective by making s larger, rather than
        making x_i larger (which is what we want).

        """
        c = self.constraints(x)
        M = self.dimension
        if np.any(c >= 0):
            return np.inf

        return t * x[M] - np.sum(np.log(-c[0:M])) - M * np.log(-c[M])

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Calculate gradient of f0 at x.

        f0(x, s) = s, so \partial f0 / \partial x_i = 0 and \partial f0 / \partial s =
        1.

        """
        grad = np.zeros_like(x)
        grad[self.dimension] = 1.0
        return grad

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        r"""Calculate gradient of ft at x.

        Our barrier objective is:
           ft(x, s) = t * s - \sum_i log(x_i + s - lb) - M * log(s0 + eps - s),
        so:
           \partial ft / \partial x_i = -1 / (x_i + s - lb), and
           \partial ft / \partial  s  = t - \sum_i (1 / (x_i + s - lb)) + M / (s0 + eps - s)

        """
        M = self.dimension
        g = np.zeros((M + 1,))
        w = x[0:M]
        s = x[M]
        g[0:M] = -1.0 / (w - self.lb + s)
        g[M] = t - np.sum(1.0 / (w - self.lb + s)) + M / (self.s0_plus_eps - s)
        return g

    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Our Hessian has an arrow sparsity pattern,
                 _                 _
                |  diag(eta)  zeta  |
            H = |                   |,
                |_  zeta^T   theta _|
        so the ith entry of H * y is eta[i] * y[i] + zeta[i] * y[-1], for all but the
        last entry of H * y, and the last entry is np.dot(zeta, y[0:-1]) + theta * y[-1].


        """
        M = self.dimension
        eta = self._hessian_ft_diagonal(x, t)
        zeta = self._hessian_ft_edge(x)
        theta = self._hessian_ft_corner(x)
        Hy = np.zeros_like(y)
        Hy[0:M] = eta * y[0:M] + y[M] * zeta
        Hy[M] = np.dot(zeta, y[0:M]) + theta * y[M]
        return Hy

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        r"""Calculate diagonal component of Hessian of ft at x.

        Our barrier objective is:
           ft(x, s) = t * s - \sum_i log(x_i + s - lb) - M * log(s0 + eps - s),
        so:
          \partial ft / \partial x_i = -1 / (x_i + s - lb), and
          \partial ft / \partial  s  = t - \sum_i (1 / (x_i + s - lb)) + M / (s0 + eps - s),
        and:
          eta[i]  := \partial^2 ft / \partial x_i^2
                   = 1 / (x_i + s - lb)^2,
          zeta[i] := \partial^2 ft / \partial s \partial x_i
                   = 1 / (x_i + s - lb)^2,
          theta   := \partial^2 ft / \partial s^2
                   = \sum_i 1 / (x_i + s - lb)^2 + M / (s0 + eps - s)^2.

        """
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return np.square(1.0 / (w - self.lb + s))

    def _hessian_ft_diagonal_inverse(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Numerically stable version of eta^{-1}."""
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return np.square(w - self.lb + s)

    def _hessian_ft_edge(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate last row/column of Hessian of ft at x."""
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return np.square(1.0 / (w - self.lb + s))

    def _hessian_ft_corner(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate bottom right corner of Hessian of ft at x."""
        M = self.dimension
        w = x[0:M]
        s = x[M]
        return (M * ((1.0 / (self.s0_plus_eps - s)) ** 2)) + np.sum(
            np.square(1.0 / (w - self.lb + s))
        )

    def _hessian_one_over_psi_squared(self, x: npt.NDArray[np.float64]) -> float:
        M = self.dimension
        s0_plus_eps_minus_s = self.s0_plus_eps - x[M]
        return (s0_plus_eps_minus_s * s0_plus_eps_minus_s) / M

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the vector of constraints, fi(x) <= 0.

        Our constraints are:
            -x_i <= s - lb,
              s  <= s0 + eps,
        or:
           -x_i - s + lb         <= 0,
                  s - (s0 + eps) <= 0

        """
        M = self.dimension
        c = np.zeros((M + 1,))
        c[0:M] = -(x[0:M] + x[M]) + self.lb
        c[M] = x[M] - self.s0_plus_eps
        return c

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0.

        In this case, the matrix of gradients is:
           _       _
          | -I  -1 |
          |  0   1 |
           -       -

        """
        M = self.dimension
        G = np.zeros((M + 1, M + 1))
        G[0:M, 0:M] = -np.eye(M)
        G[0:M, M] = -1
        G[M, M] = 1
        return G

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x) @ y."""
        M = self.dimension
        b = np.zeros((M + 1,))
        b[0:M] = -y[0:M] - y[M]
        b[M] = y[M]
        return b

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x).T @ y."""
        M = self.dimension
        b = np.zeros((M + 1,))
        b[0:M] = -y[0:M]
        b[M] = -sum(y[0:M]) + y[M]
        return b

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate dual function.

        Parameters
        ----------
         lmbda : vector
            Lagrange multipliers for inequality constraints.
         nu : vector
            Lagrange multipliers for equality constraints.

        Notes
        -----
        The Lagrangian is:
           s - lambda_1^T (x + (s - lb) * 1) + lambda_2 * (s - (s0 + eps)) + nu^T (A*x - b)
          = -b^T nu + lb * 1^T lambda_1 - (s0 + eps) * lambda_2
            +  x^T (A^T nu - lambda_1)
            + s * (1 + lambda_2 - 1^T lambda_1).
        The Lagrangian dual function is:
          g(lambda_1, lambda_2, nu) = -b^T nu + lb * 1^T lambda_1 - (s0 + eps) * lambda_2
                                      + inf_x { x^T (A^T nu - lambda_1) }
                                      + inf_s { s * (1 + lambda_2 - 1^T lambda_1) },
        which is unbounded below unless:
             lambda_1 = A^T nu, and
             lambda_2 = 1^T lambda_1 - 1.
        So the dual function equals -b^T nu + lb * 1^T lambda_1 - (s0 + eps) * lambda_2
        in that case, and is -infinity otherwise.

        """
        M = self.dimension
        if not np.allclose(lmbda[0:M], self.A.T @ nu, atol=1e-3):
            return -np.inf

        if abs(M * lmbda[M] - (np.sum(lmbda[0:M]) - 1)) > 1e-3:
            return -np.inf

        if isinstance(self.lb, list | tuple | np.ndarray):
            lb = np.asarray(self.lb)
        else:
            lb = np.full((M,), self.lb)

        return (
            -np.dot(self.b, nu)
            + np.dot(lb, lmbda[0:M])
            - M * lmbda[M] * self.s0_plus_eps
        )


class EqualityWithBoundsAndImbalanceConstraintSolver(
    EqualityConstrainedInteriorPointMethodSolver, PhaseIInteriorPointSolver
):
    r"""Find x satisfying A * x = b, x > lb, and \| B * x - c \|_\infty < psi.

    We do this by solving:
       minimize    s
       subject to  A * x = b
                   x \succeq lb
                   -s \preceq B * x - c \preceq s

    If the resulting s^\star is > psi, the original problem is infeasible.

    Parameters
    ----------
     B, c : npt.NDArray, optional
        Parameters for imbalance constraints.
     psi : float or list[float], optional
        Parameters for imbalance constraints.
     phase1_solver : EqualityWithBoundsSolver, optional
        This class requires a Phase I solver for the equality and bounds constraints,
        and there are two ways of generating this. The user can either pass an
        EqualityWithBoundsSolver or pass A, b, and lb, and a PhaseISolver will be
        initialized.
     A, b : npt.NDArray
        Equality constraints: A * x = b. Required when phase1_solver is None.
     lb : float or list[float], optional
        Lower bound on elements of x. Defaults to 0.
     settings : OptimizationSettings
        Optimization settings.

    """

    def __init__(
        self,
        B: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        psi: float | list[float] | npt.NDArray[np.float64],
        phase1_solver: EqualityWithBoundsSolver | None = None,
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        lb: float | list[float] | npt.NDArray[np.float64] = 0.0,
        settings: OptimizationSettings | None = None,
    ) -> None:
        if phase1_solver is not None:
            super().__init__(phase1_solver=phase1_solver, settings=settings)
        elif A is not None and b is not None:
            super().__init__(
                phase1_solver=EqualityWithBoundsSolver(
                    A=A,
                    b=b,
                    lb=lb,
                    settings=settings,
                ),
                settings=settings,
            )
        else:
            raise ValueError("Must specify `A` and `b`.")

        self.B = B
        self.c = c
        self.psi = psi

    @property
    def dimension(self) -> int:
        """Problem dimension."""
        return self.A.shape[1]

    @property
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""
        return self.A.shape[0]

    @property
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""
        return self.A.shape[1] + 2 * self.B.shape[0]

    @property
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wrap A."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualityWithBoundsSolver):
            raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

        return self.phase1_solver.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Wrap b."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualityWithBoundsSolver):
            raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

        return self.phase1_solver.b

    def svd_A(  # noqa: N802
        self,
    ) -> tuple[
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
    ]:
        """Calculate and cache SVD of A."""
        if self.phase1_solver is None:
            raise ValueError("phase1_solver is required.")

        if not isinstance(self.phase1_solver, EqualityWithBoundsSolver):
            raise ValueError("phase1_solver must be an EqualityWithBoundsSolver")

        return self.phase1_solver.svd_A()

    @property
    def lb(self) -> float | list[float] | npt.NDArray[np.float64]:
        """Wrap lb."""
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        if not isinstance(self.phase1_solver, EqualityWithBoundsSolver):
            raise ValueError("PhaseISolver must be an EqualityWithBoundsSolver.")

        return self.phase1_solver.lb

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""
        return bool(np.all(np.abs(self.B @ x[0:-1] - self.c) < self.psi))

    def check_for_infeasibility(self, result: NewtonResult) -> None:
        """Check if infeasible."""
        if result.dual_value > 0.0:
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    "Problem certifiably infeasible: dual value was "
                    f"{result.dual_value} > 0.0"
                ),
                result=result,
            )

    def augment_previous_solution(
        self,
        phase1_res: OptimizationResult,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Initialize variable based on Phase I result."""
        x = np.zeros((len(phase1_res.solution) + 1,))
        x[0 : self.dimension] = phase1_res.solution
        eps = 1.0
        x[self.dimension] = (
            max(
                np.max(self.B @ phase1_res.solution - self.c - self.psi),
                np.max(-(self.B @ phase1_res.solution - self.c) - self.psi),
            )
            + eps
        )
        return x

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """De-augment solution."""
        return x[0 : self.dimension]

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        """Initialize barrier parameter.

        Modifies the method from EqualityConstrainedInteriorPointMethodSolver to account
        for the augmented variable.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Initial guess. Must be strictly feasible.

        Returns
        -------
         t : float
            Barrier parameter.

        """
        M = self.dimension
        delta_f0 = self.gradient(x0)
        delta_phi = -self.grad_constraints_transpose_multiply(
            x0, 1.0 / self.constraints(x0)
        )

        A_delta_f0 = self.A @ delta_f0[0:M]
        A_delta_phi = self.A @ delta_phi[0:M]

        U, s, Vh = self.svd_A()
        rank = int(np.sum(s > 1e-5))
        U = U[:, 0:rank]
        s = s[0:rank]
        z_phi = U @ ((U.T @ A_delta_phi) / np.square(s))
        z_f0 = U @ ((U.T @ A_delta_f0) / np.square(s))

        schur_complement = np.dot(delta_f0, delta_f0) - np.dot(A_delta_f0, z_f0)
        t_star_s = -np.dot(delta_f0, delta_phi) + np.dot(A_delta_f0, z_phi)

        return min(
            self.num_ineq_constraints / self.settings.outer_tolerance,
            max(1.0, t_star_s / max(schur_complement, 1e-16)),
        )

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(x) := t * f0(x) - \sum_i log(-fi(x))
          subject to A * x = b.

        Parameters
        ----------
         x : vector
            Current estimate.
         t : float
            Barrier parameter.

        Returns
        -------
         delta_x : vector
            Newton step.
         nu : vector
            Lagrange multiplier associated with equality constraints.

        Notes
        -----
        The Newton step, delta_x, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_x |   | - grad_ft |
          | A    0  | |   nu    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at x, grad_f is the gradient of ft
        evaluated at x, and nu is the Lagrange multiplier associated with the equality
        constraints.

        Our Hessian has a block structure, so we use the `solve_block_plus_one` helper
        function. The upper left block, A11, itself has a special structure: it's
        diagonal plus 2, rank-p updates. So we use two nested calls to
        `solve_rank_p_update` in its solution.

        """
        Bx_minus_c = self.B @ x[: self.dimension] - self.c  # shape (q,)
        eta_inverse: npt.NDArray[np.float64] = self._hessian_ft_diagonal_inverse(x)
        kappa_pos: npt.NDArray[np.float64] = self._hessian_ft_kappa_pos(x, Bx_minus_c)
        kappa_neg: npt.NDArray[np.float64] = self._hessian_ft_kappa_neg(x, Bx_minus_c)

        def A_solve(  # noqa: N802
            b: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return solve_rank_p_update(
                b,
                kappa=kappa_pos,
                A_solve=solve_diagonal_eta_inverse,
                eta_inverse=eta_inverse,
            )

        def A_solve_nested(  # noqa: N802
            b: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return solve_rank_p_update(b, kappa=kappa_neg, A_solve=A_solve)

        return solve_kkt_system(
            A=np.hstack((self.A, np.zeros((self.num_eq_constraints, 1)))),
            g=-self.gradient_barrier(x, t),
            hessian_solve=solve_block_plus_one,
            A12=self._hessian_ft_edge(x, Bx_minus_c),
            A22=self._hessian_ft_corner(x, Bx_minus_c),
            A11_solve=A_solve_nested,
        )

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + theta * delta_x stays strictly feasible."""
        M = self.dimension
        w = x[:M]
        s = x[M]
        delta_w = delta_x[:M]
        delta_s = delta_x[M]

        Bw_minus_c = self.B @ w - self.c
        B_delta_w = self.B @ delta_w

        # Constraint 1: x + theta * delta_x >= lb
        mask = delta_w < 0
        theta1 = np.inf
        lb = self.lb
        if np.any(mask):
            if isinstance(lb, list | tuple | np.ndarray):
                theta1 = np.min((w[mask] - lb[mask]) / -delta_w[mask])
            else:
                theta1 = np.min((w[mask] - lb) / -delta_w[mask])

        # Constraint 2: -psi - s < Bx - c + theta * (B delta_x)
        mask2 = B_delta_w + delta_s < 0
        theta2 = np.inf
        psi = self.psi
        if np.any(mask2):
            if isinstance(psi, list | tuple | np.ndarray):
                theta2 = np.min(
                    (Bw_minus_c[mask2] + psi[mask2] + s)
                    / np.abs(B_delta_w[mask2] + delta_s)
                )
            else:
                theta2 = np.min(
                    (Bw_minus_c[mask2] + psi + s) / np.abs(B_delta_w[mask2] + delta_s)
                )

        # Constraint 3: Bx - c + theta * (B delta_x) < psi + s + theta * delta_s
        mask3 = B_delta_w - delta_s > 0
        theta3 = np.inf
        if np.any(mask3):
            if isinstance(psi, list | tuple | np.ndarray):
                theta3 = np.min(
                    -(Bw_minus_c[mask3] - psi[mask3] - s) / (B_delta_w[mask3] - delta_s)
                )
            else:
                theta3 = np.min(
                    -(Bw_minus_c[mask3] - psi - s) / (B_delta_w[mask3] - delta_s)
                )

        # Take minimum positive theta
        return min(theta1, theta2, theta3)

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at w."""
        return x[self.dimension]

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Calculate gradient of f0 at x.

        f0(x, s) = s, so \partial f0 / \partial x_i = 0 and \partial f0 / \partial s =
        1.

        """
        grad = np.zeros_like(x)
        grad[self.dimension] = 1.0
        return grad

    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y.

        Parameters
        ----------
         x : npt.NDArray
            Current point.
         t : float
            The barrier objective parameter.
         y : npt.NDArray
            Multiplicand.

        Returns
        -------
         Hy : npt.NDArray
            H * y

        Notes
        -----
        Exploits the special structure of the Hessian to compute H * y faster than
        O(M^2).

        """
        M = self.dimension
        y_x = y[:M]  # shape (M,)
        y_s = y[M]  # scalar

        Bx_minus_c = self.B @ x[:M] - self.c  # shape (q,)

        # Diagonal component
        eta = self._hessian_ft_diagonal(x)  # shape (M,)

        # Rank-q factors
        kappa_pos = self._hessian_ft_kappa_pos(x, Bx_minus_c)  # shape (M, q)
        kappa_neg = self._hessian_ft_kappa_neg(x, Bx_minus_c)  # shape (M, q)

        # Mixed derivatives
        hxs = self._hessian_ft_edge(x, Bx_minus_c)  # shape (M,)
        hss = self._hessian_ft_corner(x, Bx_minus_c)  # scalar

        # Compute Hxx * y_x
        Hxx_yx = (  # shape (M,)
            eta * y_x
            + kappa_pos @ (kappa_pos.T @ y_x)
            + kappa_neg @ (kappa_neg.T @ y_x)
        )

        # Add mixed derivative term
        Hy_x = Hxx_yx + hxs * y_s  # shape (M,)

        # Compute lower block: Hxs^T y_x + Hss * y_s
        Hy_s = np.dot(hxs, y_x) + hss * y_s  # scalar

        # Concatenate result
        return np.append(Hy_x, Hy_s)

    def _hessian_ft_diagonal(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate diagonal component of Hessian of ft at x."""
        return np.square(1.0 / (x[0 : self.dimension] - self.lb))

    def _hessian_ft_diagonal_inverse(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate inverse of diagonal component of Hessian of ft at x."""
        return np.square(x[0 : self.dimension] - self.lb)

    def _hessian_ft_kappa_pos(
        self, x: npt.NDArray[np.float64], Bx_minus_c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate positive rank update to Hessian of ft at x."""
        M = self.dimension
        s = x[M]
        denom_pos = s + self.psi + Bx_minus_c  # shape (q,)
        if np.any(denom_pos <= 0):
            raise ValueError(
                "Barrier argument out of domain: denominators must be positive."
            )

        # Each column: B[j, :] / denom_pos[j]
        return self.B.T / denom_pos

    def _hessian_ft_kappa_neg(
        self, x: npt.NDArray[np.float64], Bx_minus_c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate negative rank update to Hessian of ft at x."""
        M = self.dimension
        s = x[M]
        denom_neg = s + self.psi - Bx_minus_c  # shape (q,)
        if np.any(denom_neg <= 0):
            raise ValueError(
                "Barrier argument out of domain: denominators must be positive."
            )
        return self.B.T / denom_neg

    def _hessian_ft_edge(
        self, x: npt.NDArray[np.float64], Bx_minus_c: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate last row/column of Hessian of ft at x."""
        M = self.dimension
        s = x[M]
        denom_pos = s + self.psi + Bx_minus_c
        denom_neg = s + self.psi - Bx_minus_c

        # Domain check for numerical stability
        if np.any(denom_pos <= 0) or np.any(denom_neg <= 0):
            raise ValueError(
                "Barrier argument out of domain: denominators must be positive."
            )

        return self.B.T @ (np.square(1.0 / denom_pos) - np.square(1.0 / denom_neg))

    def _hessian_ft_corner(
        self, x: npt.NDArray[np.float64], Bx_minus_c: npt.NDArray[np.float64]
    ) -> float:
        """Calculate bottom right corner of Hessian of ft at x."""
        M = self.dimension
        s = x[M]
        denom_pos = s + self.psi + Bx_minus_c
        denom_neg = s + self.psi - Bx_minus_c
        # For numerical stability, ensure denominators are not zero or negative
        if np.any(denom_pos <= 0) or np.any(denom_neg <= 0):
            raise ValueError(
                "Barrier argument out of domain: denominators must be positive."
            )

        return np.sum(np.square(1.0 / denom_pos) + np.square(1.0 / denom_neg))

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Calculate the vector of constraints, fi(x) <= 0.

        Our constraints are:
            x_i >= lb, i = 1, ..., M
            -s <= Bj^T * x - cj <= s, j = 1, ..., q,
        where Bj is the jth row of B and cj the jth element of c. We can rewrite this in
        standard form as:
                 -x_i   + lb <= 0, i = 1, ..., M
            -Bj^T x - s + cj <= 0, j = 1, ..., q
             Bj^T x - s - cj <= 0, j = 1, ..., q.

        """
        M = self.dimension
        Bx_minus_c = self.B @ x[0:M] - self.c
        return np.concatenate(
            [
                -x[0:M] + self.lb,
                -Bx_minus_c - np.full_like(self.c, x[M]) - self.psi,
                Bx_minus_c - np.full_like(self.c, x[M]) - self.psi,
            ]
        )

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0.

        In this case, the matrix of gradients is:
           _       _
          | -I   0  |
          | -B  -1  |
          | +B  -1  |
           -       -

        """
        M = self.dimension
        q = self.B.shape[0]

        # Top block: -I (M x M), zeros (M x 1)
        top_left = -np.eye(M)
        top_right = np.zeros((M, 1))

        # Middle block: -B (q x M), -1 (q x 1)
        middle_left = -self.B
        middle_right = -np.ones((q, 1))

        # Lower block: +B (q x M), -1 (q x 1)
        lower_left = self.B
        lower_right = -np.ones((q, 1))

        return np.vstack(
            [
                np.hstack([top_left, top_right]),
                np.hstack([middle_left, middle_right]),
                np.hstack([lower_left, lower_right]),
            ]
        )

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Efficiently compute grad_constraints(x) @ y using block structure.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Current point (not used in this function, but kept for API consistency).
        y : npt.NDArray[np.float64]
            Vector to multiply, shape (M + 1,)

        Returns
        -------
        result : npt.NDArray[np.float64]
            Product, shape (M + 2q,)
        """
        M = self.dimension
        By = self.B @ y[:M]
        return np.concatenate([-y[:M], -By - y[M], By - y[M]])

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Efficiently compute grad_constraints(x).T @ y using block structure.

        Parameters
        ----------
        x : npt.NDArray[np.float64]
            Current point (not used in this function, but kept for API consistency).
        y : npt.NDArray[np.float64]
            Vector to multiply, shape (M + 2q,)

        Returns
        -------
        result : npt.NDArray[np.float64]
            Product, shape (M + 1,)
        """
        M = self.dimension
        q = self.B.shape[0]

        y1 = y[:M]  # Top block
        y2 = y[M : M + q]  # Middle block
        y3 = y[M + q :]  # Lower block

        x_part = -y1 + self.B.T @ (y3 - y2)
        s_part = -np.sum(y2) - np.sum(y3)

        return np.append(x_part, s_part)

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate dual function.

        Parameters
        ----------
         lmbda : vector
            Lagrange multipliers for inequality constraints.
         nu : vector
            Lagrange multipliers for equality constraints.
         x_star : vector
            The solution.

        Returns
        -------
         g : float
            The dual function evaluated at lmbda, nu.

        Notes
        -----
        The Lagrangian is:
          s + nu^T (Ax - b) + lambda_1^T (-x + lb * 1) + lambda_2^T (-B x - s * 1 + c)
            + lambda_3^T (B x - s * 1 - c)
          = (A^T nu - lambda_1 + B^T (lambda_3 - lambda_2))^T x
            + (1 - sum(lambda_2) - sum(lambda_3)) * s
            - b^T nu + lb * sum(lambda_1) + c^T (lambda_2 - lambda_3),
        which is unbounded below unless
            lambda1 = A^T nu + B^T (lambda_3 - lambda_2), and
            sum(lambda_2) + sum(lambda_3) = 1.
        So the dual function equals
            - b^T nu + lb * sum(lambda_1) + c^T (lambda_2 - lambda_3)
        in that case, and is -infinity otherwise.

        """
        M = self.dimension
        q = self.B.shape[0]
        lmbda1 = lmbda[0:M]
        lmbda2 = lmbda[M : (M + q)]
        lmbda3 = lmbda[(M + q) : (M + 2 * q)]

        atol = 1e-3
        if not np.allclose(
            lmbda1, self.A.T @ nu + self.B.T @ (lmbda3 - lmbda2), atol=atol
        ):
            return -np.inf

        if abs(np.sum(lmbda2) + np.sum(lmbda3) - 1) > atol:
            return -np.inf

        if isinstance(self.lb, list | tuple | np.ndarray):
            lb = np.asarray(self.lb)
        else:
            lb = np.full((M,), self.lb)

        return (
            -np.dot(self.b, nu)
            + np.dot(lb, lmbda1)
            + np.dot(self.c - self.psi, lmbda2)
            - np.dot(self.c + self.psi, lmbda3)
        )
