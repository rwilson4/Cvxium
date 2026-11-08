"""Quadratic and sublevel-set constraint layers for Cvxium's interior-point solvers.

Defines three single-constraint layers, each a ``DirectConstraintLayer``:
- ``NormConstraintLayer``: the Euclidean-norm bound ‖B w - c‖ ≤ φ, encoded in squared
  form.
- ``SquaredNormConstraintLayer``: the bound ‖w‖^2 ≤ φ, a cheaper special case with
  B = I and c = 0.
- ``ObjectiveConstraintLayer``: the sublevel set f(w) ≤ φ of any convex ``Objective``.

Because each is a direct layer, it is also an ``Objective``. When one is the active
layer, ``Problem.make_phase1`` builds a direct-form Phase I problem that minimizes the
constraint value itself, subject to the remaining layers, and ``BarrierSolver`` solves
it. Each layer supplies structured Hessians (a low-rank update, a diagonal plus a
rank-one update, or the wrapped objective's own structure) so that Newton steps exploit
them.

"""

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..exceptions import NewtonStepError
from ..linalg.numerical_helpers import (
    aspect_ratio_svd,
    tikhonov_filter,
)
from ..linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    MatrixSystem,
    ZeroSystem,
)
from ..problems.constraint_layers import (
    DirectConstraintLayer,
)
from ..problems.objective import Objective
from ..solvers.settings import (
    LineSearchSettings,
)

if TYPE_CHECKING:
    # Imported for typing only: ``linear_constraints`` imports this module's
    # ``constraint_layers`` dependency chain; guard the equality type used in the
    # ``interior_point`` annotation to avoid any import-order fragility.
    from ..problems.linear_constraints import (
        EqualityConstraintLayer,
    )


class NormConstraintLayer(DirectConstraintLayer):
    """Euclidean-norm constraint ‖B w - c‖ ≤ φ, encoded in squared form.

    The user-visible constraint is ‖B w - c‖ ≤ φ. Internally it is the single constraint
       f(w) = ‖B w - c‖^2 - φ^2 ≤ 0,
    which has the same feasible set (for φ > 0) but is smooth where B w = c. Its
    derivatives are
       ∇f = 2 B^T v,   ∇^2 f = 2 B^T B,   where v = B w - c,
    so the Hessian is constant and of rank at most q. The squared form avoids the
    division by ‖v‖ that the unsquared form needs, and the numerical trouble it causes
    near B w = c.

    As a ``DirectConstraintLayer``, the layer serves as its own Phase I objective: a
    direct-form Phase I minimizes f(w), and its objective Hessian is
    ``hessian_curvature`` at unit weight, the rank-q update 2 B^T B on a zero base,
    which Newton steps exploit.

    The ``shift`` and ``ds`` arguments of the layer protocol must be zero here, and
    ``constraints``, ``hessian_contribution`` and ``btls_max_step`` assert it. A shift
    does not commute with squaring: the bound
       ‖B w - c‖^2 ≤ φ^2 + shift
    is not equivalent to ‖B w - c‖ ≤ φ + shift, so accepting a non-zero shift would
    silently change the feasible region. No caller needs one: ``Problem.make_phase1``
    shifts only the active layer of a slack-form Phase I, and a direct layer always
    takes the direct form.

    Parameters
    ----------
     B : npt.NDArray[np.float64]
        Constraint matrix, of shape (q, M). Must be two-dimensional, or the constructor
        raises ``ValueError``.
     c : npt.NDArray[np.float64]
        Constraint target, of length q. Any other shape makes the constructor raise
        ``ValueError``.
     phi : float
        Norm bound φ. Must be positive, or the constructor raises ``ValueError``.

    """

    def __init__(
        self,
        B: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        phi: float,
    ) -> None:
        if phi <= 0.0:
            raise ValueError(f"phi must be positive (got {phi}).")
        if B.ndim != 2:
            raise ValueError("B must be a 2D array.")
        if c.ndim != 1 or c.shape[0] != B.shape[0]:
            raise ValueError(
                f"c must be a 1D array of length {B.shape[0]} (got shape {c.shape})."
            )
        self.B = B
        self.c = c
        self.phi = float(phi)
        self.phi_squared = float(phi) ** 2

    @property
    def q(self) -> int:
        """Return the number of rows q of B.

        Returns
        -------
         q : int
            The row count of ``B``.

        """
        return self.B.shape[0]

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M, the column count of B.

        Returns
        -------
         M : int
            The column count of ``B``.

        """
        return self.B.shape[1]

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the squared-form constraint value ‖B w - c‖^2 - φ^2.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraint is evaluated, of length M.
         shift : float
            Must be zero; see the class description.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The constraint value, of length 1; negative when w strictly satisfies the
            bound.

        Raises
        ------
         AssertionError
            When ``shift`` is not zero.

        """
        assert shift == 0.0, (
            "NormConstraintLayer does not support non-zero shift in squared-form "
            "encoding; see class docstring."
        )
        v = self.B @ w - self.c
        return np.array([float(np.dot(v, v)) - self.phi_squared])

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint Jacobian G = (2 B^T (B w - c))^T.

        The squared form is smooth everywhere; G is zero where B w = c.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (1, M).

        """
        v = self.B @ w - self.c
        return (2.0 * self.B.T @ v).reshape(1, -1)

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian.

        Computes
           G v = 2 (B w - c)^T (B v)
        without forming G.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length 1.

        """
        Bw_minus_c = self.B @ w - self.c
        Bv = self.B @ v
        return np.array([2.0 * float(np.dot(Bw_minus_c, Bv))])

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian.

        Computes
           G^T u = 2 u_0 B^T (B w - c)
        without forming G. An empty u gives the zero vector.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.
         u : npt.NDArray[np.float64]
            Vector of length 1 (or empty) to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        # An empty multiplier slice contributes zero to the caller's running sum.
        if u.size == 0:
            return np.zeros_like(w)
        Bw_minus_c = self.B @ w - self.c
        return np.asarray(2.0 * float(u[0]) * (self.B.T @ Bw_minus_c))

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the weighted barrier curvature as a rank-(q + 1) update on zero.

        The constraint f = ‖v‖^2 - φ^2, with v = B w - c, has
           ∇f = 2 B^T v,   ∇^2 f = 2 B^T B,
        so its curvature atom is
           C = 2 B^T B - (4 / f) (B^T v) (B^T v)^T.
        Weighted by λ, this is a ``ZeroSystem`` carrying one low-rank update with factor
           κ = [B^T | B^T v],   of shape (M, q + 1),
        and weights
           d = (2 λ, ..., 2 λ, -4 λ / f).
        The barrier method passes λ = -1 / f, giving
           (-2 / f) B^T B + (4 / f^2) (B^T v) (B^T v)^T,
        while the primal-dual method passes its dual variable. Every weight is positive
        on the strict interior, where f < 0 and λ > 0, so the update has no downdate and
        no mixed inertia.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible, of length M.
         lmbda : npt.NDArray[np.float64] | None
            The weight λ, of length 1. Required; the ``None`` default exists only so the
            signature matches ``Objective.hessian_contribution``.
         shift : float
            Must be zero; see the class description.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature λ C.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``, or when w is not strictly feasible (f ≥ 0).
         AssertionError
            When ``shift`` is not zero.

        """
        if lmbda is None:
            raise ValueError(
                "NormConstraintLayer.hessian_contribution requires `lmbda`."
            )
        assert shift == 0.0, (
            "NormConstraintLayer does not support non-zero shift in squared-form "
            "encoding; see class docstring."
        )
        v = self.B @ w - self.c
        f = float(np.dot(v, v)) - self.phi_squared
        if f >= 0.0:
            raise ValueError(
                "Barrier argument out of domain: ||Bw - c||^2 must be strictly less "
                f"than phi^2 (got ||Bw - c||^2 = {float(np.dot(v, v))}, "
                f"phi^2 = {self.phi_squared})."
            )
        lam = float(lmbda[0])
        BTv = self.B.T @ v
        kappa = np.hstack([self.B.T, BTv.reshape(-1, 1)])  # shape (M, q+1)
        d = np.empty(self.q + 1)
        d[: self.q] = 2.0 * lam
        d[-1] = -4.0 * lam / f
        return ZeroSystem(len(w)).low_rank_update(kappa, d)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the weighted constraint curvature 2 λ B^T B as a rank-q update.

        Since ∇^2 f = 2 B^T B, the curvature is a ``ZeroSystem`` carrying one low-rank
        update with factor B^T and every weight 2 λ, positive semidefinite for λ ≥ 0 and
        independent of w. At unit weight it is also the objective Hessian of a
        direct-form Phase I on this layer.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; only its length is used.
         lmbda : npt.NDArray[np.float64]
            The weight λ, of length 1.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature 2 λ B^T B.

        """
        d = np.full(self.q, 2.0 * float(lmbda[0]))
        return ZeroSystem(len(w)).low_rank_update(self.B.T, d)

    def interior_point(
        self, equality: "EqualityConstraintLayer | None" = None
    ) -> npt.NDArray[np.float64] | None:
        """Return the minimum-residual point of the ellipsoid if strictly interior.

        The constraint value f(w) is smallest where ‖B w - c‖ is smallest, so the
        deepest interior point, by constraint value, is the minimum-residual point on
        whatever set the feasibility problem imposes:
        - Without equality constraints, the least-squares solution w_star = B^+ c,
          computed from a thin SVD of B with a Tikhonov-damped pseudo-inverse
          (``tikhonov_filter``) so that a near-degenerate B stays numerically stable. It
          is the ellipsoid's center when B is square and nonsingular.
        - With equality constraints A w = b, the equality-constrained least-squares
          solution, the w block of the KKT system shown below. It is solved through the
          equality layer's ``saddle_system``, so a sparse A stays sparse.

        The KKT system is
           [ 2 B^T B   A^T ] [ w ]   [ 2 B^T c ]
           [    A       0  ] [ ν ] = [    b    ].

        The point is returned only when it is strictly interior (‖B w_star - c‖ < φ)
        and, with equality constraints, satisfies |A w_star - b| < 1e-10 componentwise,
        the same gate ``Solver._solve_feasibility`` applies before accepting it.
        Otherwise, including when the saddle solve raises ``NewtonStepError`` (as it can
        for a rank-deficient B), the method returns ``None`` and the caller falls back
        to the iterative Phase I.

        Parameters
        ----------
         equality : EqualityConstraintLayer | None
            The problem's equality constraints A w = b, or ``None`` for an
            inequality-only feasibility problem.

        Returns
        -------
         w : npt.NDArray[np.float64] | None
            The strictly interior minimum-residual point, satisfying A w = b when
            ``equality`` is given, or ``None`` when that point is not strictly interior
            or not found.

        """
        B = self.B
        c = self.c
        if equality is None:
            # Minimum-norm least-squares center w* = B^+ c, via the shared
            # aspect-ratio SVD and the Tikhonov-damped pseudo-inverse. Damping
            # keeps a near-degenerate ellipsoid (a singular value near the floor)
            # numerically stable rather than amplifying it with a raw ``1 / s``.
            U, s, Vh = aspect_ratio_svd(B)
            filt, _ = tikhonov_filter(s, power=1)
            w_star = Vh.T @ (filt * (U.T @ c))
        else:
            # Equality-constrained least squares min ||B w - c||^2 s.t. A w = b,
            # i.e. the KKT saddle [[2 B^T B, A^T], [A, 0]] [w; nu] = [2 B^T c; b].
            # Route it through the equality layer's injected saddle factory so a
            # sparse A stays sparse (no densification), and so a structured
            # 2 B^T B could later be supplied as a MatrixSystem in place of the
            # DenseSystem without touching this call.
            try:
                w_star, _ = equality.saddle_system(DenseSystem(2.0 * (B.T @ B))).solve(
                    2.0 * (B.T @ c), equality.b
                )
            except NewtonStepError:
                # A rank-deficient B (a degenerate, unbounded ellipsoid) or a
                # singular Schur complement leaves no analytic center; fall
                # through to the normal feasibility path.
                return None
            # The saddle solve lands on the manifold only when b is consistent
            # and the system well-posed; verify before claiming the point lies
            # on it (the same gate the caller applies).
            if not bool(np.all(np.abs(equality.A @ w_star - equality.b) < 1e-10)):
                return None
        v = B @ w_star - c
        r_squared = float(np.dot(v, v))
        if r_squared < self.phi_squared:
            return np.asarray(w_star, dtype=np.float64)
        return None

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps ‖B (w + α dw) - c‖^2 below φ^2.

        With v = B w - c, the constraint at step α is the quadratic inequality
           a α^2 + 2 b α + e < 0,   a = ‖B dw‖^2,   b = v^T B dw,   e = ‖v‖^2 - φ^2.
        Strict feasibility gives e < 0, so when a > 0 the roots have opposite signs and
        the positive root is the step to the boundary. When a = 0 (B dw = 0) the
        constraint value does not change along dw and the step is unbounded. The bound
        is closed form, so ``line_search`` is ignored.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.
         shift : float
            Must be zero; see the class description.
         ds : float
            Must be zero; see the class description.
         line_search : LineSearchSettings | None
            Ignored.

        Returns
        -------
         alpha : float
            The step at which the constraint reaches the boundary; inf when B dw = 0.

        Raises
        ------
         AssertionError
            When ``shift`` or ``ds`` is not zero.
         ValueError
            Propagated from ``math.sqrt`` when w is not strictly feasible and the
            discriminant is negative.

        Notes
        -----
        The positive root is
           (-b + sqrt(b^2 - a e)) / a.
        For b > 0 the numerator cancels catastrophically when |a e| is much smaller than
        b^2, so in that case the code uses the algebraically equal form
           -e / (b + sqrt(b^2 - a e)).

        """
        assert shift == 0.0, (
            "NormConstraintLayer does not support non-zero shift in squared-form "
            "encoding; see class docstring."
        )
        assert ds == 0.0, (
            "NormConstraintLayer does not support non-zero ds in squared-form "
            "encoding; see class docstring."
        )
        v = self.B @ w - self.c
        Bdw = self.B @ dw
        a_q = float(np.dot(Bdw, Bdw))
        b_q = float(np.dot(v, Bdw))
        e_q = float(np.dot(v, v)) - self.phi_squared
        if a_q <= 0.0:
            # ||B(w + alpha dw) - c||^2 is constant in alpha (B dw == 0); the
            # constraint stays satisfied for all alpha.
            return np.inf
        disc = b_q * b_q - a_q * e_q
        # e_q < 0 and a_q > 0 imply disc > 0; the larger root is positive
        # (sum of roots -2 b_q / a_q, product e_q / a_q < 0 -> roots have
        # opposite signs).
        sqrt_disc = math.sqrt(disc)
        # Numerically stable: the textbook (-b_q + sqrt_disc) / a_q cancels
        # catastrophically when b_q > 0 and |a_q e_q| << b_q^2. Rationalize the
        # numerator by multiplying by (b_q + sqrt_disc) / (b_q + sqrt_disc): the
        # positive root equals -e_q / (b_q + sqrt_disc). This is mathematically
        # identical and stable for b_q > 0.
        if b_q > 0.0:
            return float(-e_q / (b_q + sqrt_disc))
        return float((-b_q + sqrt_disc) / a_q)

    def is_satisfied(
        self,
        w: npt.NDArray[np.float64],
    ) -> bool:
        """Test whether ‖B w - c‖ < φ strictly.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         satisfied : bool
            True when ‖B w - c‖^2 < φ^2.

        """
        v = self.B @ w - self.c
        return float(np.dot(v, v)) < self.phi_squared


class SquaredNormConstraintLayer(DirectConstraintLayer):
    """Squared-Euclidean-norm constraint ‖w‖^2 ≤ φ.

    Encoded as the single constraint
       f(w) = w^T w - φ ≤ 0.
    It has the same feasible set as ``NormConstraintLayer(B=I, c=0, phi=sqrt(φ))``. The
    dedicated class exists because with B = I and c = 0 the math collapses to the
    gradient 2 w and the Hessian 2 I, and the general class would spend matrix products
    on what are scalar operations.

    As a ``DirectConstraintLayer``, the layer serves as its own Phase I objective, with
    the diagonal Hessian 2 I, and it offers a ``barrier_parameter_hint`` for that
    problem. Unlike ``NormConstraintLayer``, it accepts a shift, which relaxes the bound
    to w^T w ≤ φ + shift.

    Parameters
    ----------
     phi : float
        The bound φ on the squared norm. Must be positive, or the constructor raises
        ``ValueError``.
     dimension : int
        Decision-variable dimension M. Required because the scalar ``phi`` does not fix
        the length of w.

    """

    def __init__(self, phi: float, dimension: int) -> None:
        if phi <= 0.0:
            raise ValueError(f"phi must be positive (got {phi}).")
        self.phi = float(phi)
        self._dimension: int = dimension

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M given at construction.

        Returns
        -------
         M : int
            The ``dimension`` passed to the constructor.

        """
        return self._dimension

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the shifted constraint value w^T w - φ - shift.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraint is evaluated, of length M.
         shift : float
            Relaxation subtracted from the constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The shifted constraint value, of length 1.

        """
        return np.array([float(np.dot(w, w)) - self.phi - shift])

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint Jacobian G = (2 w)^T.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (1, M).

        """
        return (2.0 * w).reshape(1, -1)

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian.

        Computes G v = 2 w^T v without forming G.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length 1.

        """
        return np.array([2.0 * float(np.dot(w, v))])

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian.

        Computes G^T u = 2 u_0 w without forming G. An empty u gives the zero vector.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.
         u : npt.NDArray[np.float64]
            Vector of length 1 (or empty) to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        # An empty multiplier slice contributes zero to the caller's running sum.
        if u.size == 0:
            return np.zeros_like(w)
        return np.asarray(2.0 * float(u[0]) * w)

    def barrier_parameter_hint(
        self,
        x0: npt.NDArray[np.float64],
        num_ineq: int,
        outer_tolerance: float,
    ) -> float | None:
        """Return a lower bound on t0 for a direct-form Phase I on this layer.

        When this layer is the objective of a direct-form Phase I problem,
        ``BarrierSolver`` raises its initial barrier parameter t0 to max(t0, hint) under
        every adaptive initialization strategy (all but ``UNIT``). Without that floor,
        t0 can be too small for the objective's scale, and the centering
        under-progresses on the objective.

        The hint is a closed-form lower bound on the heuristic
           t0 ≈ m / (f(x0) - p_star)
        (Boyd and Vandenberghe, 2004, section 11.3.1). The Phase I objective satisfies
           f(w) = w^T w - φ ≥ -φ,
        so p_star ≥ -φ and f(x0) - p_star ≤ x0^T x0, giving
           hint = m / max(outer_tolerance, x0^T x0),
        where m is ``num_ineq``. The ``outer_tolerance`` floor avoids dividing by a tiny
        x0^T x0 when x0 is near the origin.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Strictly feasible starting point of the Phase I problem.
         num_ineq : int
            Number of inequality constraints in the Phase I problem (the background
            layers' constraints), the m in the heuristic.
         outer_tolerance : float
            Outer duality-gap tolerance, a floor for the denominator.

        Returns
        -------
         hint : float | None
            The lower bound m / max(outer_tolerance, x0^T x0); never ``None`` for this
            layer.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        return num_ineq / max(outer_tolerance, float(np.dot(x0, x0)))

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the weighted barrier curvature as a diagonal plus a rank-one update.

        With c = w^T w - φ - shift, negative on the strict interior, the constraint has
        ∇f = 2 w and ∇^2 f = 2 I, so its curvature atom is
           C = 2 I - (4 / c) w w^T.
        Weighted by λ, this is a ``DiagonalSystem`` with every entry 2 λ, carrying one
        rank-one update with factor w and weight -4 λ / c. With the barrier method's
        weight
           λ = -1 / c
        this becomes
           -(2 / c) I + (4 / c^2) w w^T,
        while the primal-dual method passes its dual variable as λ. The diagonal's
        reciprocal is passed explicitly as 1 / (2 λ), the same value ``DiagonalSystem``
        computes by default; at the barrier weight it equals -c / 2.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible for the shifted bound, of length M.
         lmbda : npt.NDArray[np.float64] | None
            The weight λ, of length 1, positive. Required; the ``None`` default exists
            only so the signature matches ``Objective.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint value, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature λ C.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``, or when w is not strictly feasible for the
            shifted bound (c ≥ 0).
         ZeroDivisionError
            When λ = 0, from the reciprocal 1 / (2 λ).

        """
        if lmbda is None:
            raise ValueError(
                "SquaredNormConstraintLayer.hessian_contribution requires `lmbda`."
            )
        c_scalar = float(np.dot(w, w)) - self.phi - shift
        if c_scalar >= 0.0:
            raise ValueError(
                "Barrier argument out of domain: ||x||^2 must be strictly less than "
                f"phi + shift (got ||x||^2 = {float(np.dot(w, w))}, "
                f"phi + shift = {self.phi + shift})."
            )
        lam = float(lmbda[0])
        eta = np.full(w.shape[0], 2.0 * lam)
        eta_inverse = np.full(w.shape[0], 1.0 / (2.0 * lam))
        return DiagonalSystem(eta, eta_inverse).low_rank_update(
            w.reshape(-1, 1), np.array([-4.0 * lam / c_scalar])
        )

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the weighted constraint curvature 2 λ I as a diagonal.

        Since ∇^2 f = 2 I, the curvature is a ``DiagonalSystem`` with every entry 2 λ,
        independent of w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; only its length is used.
         lmbda : npt.NDArray[np.float64]
            The weight λ, of length 1.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature 2 λ I.

        """
        return DiagonalSystem(np.full(w.shape[0], 2.0 * float(lmbda[0])))

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps the shifted squared-norm bound strict.

        Finds the largest α with
           ‖w + α dw‖^2 < φ + shift + α ds,
        that is,
           a α^2 + 2 b α + e < 0,
        with
           a = dw^T dw,   b = w^T dw - ds / 2,   e = w^T w - φ - shift.
        Strict feasibility gives e < 0. When a > 0 the roots have opposite signs, and
        the positive root
           (-b + sqrt(b^2 - a e)) / a
        is the bound. When a = 0 (dw = 0) the left side is e - α ds, which reaches zero
        at α = e / ds when ds < 0 and never otherwise. The bound is closed form, so
        ``line_search`` is ignored.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible for the shifted bound, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Ignored.

        Returns
        -------
         alpha : float
            The step at which the constraint reaches the boundary; inf when it never
            does.

        Raises
        ------
         ValueError
            Propagated from ``math.sqrt`` when w is not strictly feasible and the
            discriminant is negative.

        """
        a = float(np.dot(dw, dw))
        b_lin = float(np.dot(w, dw)) - 0.5 * ds  # half of the linear coefficient
        e = float(np.dot(w, w)) - (self.phi + shift)
        if a <= 0.0:
            # ||x + alpha dw||^2 - alpha ds is linear in alpha (dw == 0); the
            # bound is finite only when the linear term grows the LHS.
            if -ds > 0.0:
                return -e / -ds
            return np.inf
        disc = b_lin * b_lin - a * e
        # e < 0 (strict feasibility) and a > 0 imply disc > 0; the larger root
        # (-b_lin + sqrt(disc)) / a is always positive since sqrt(disc) > |b_lin|.
        return (-b_lin + math.sqrt(disc)) / a

    def is_satisfied(
        self,
        w: npt.NDArray[np.float64],
    ) -> bool:
        """Test whether w^T w < φ strictly.

        The test uses the unshifted bound.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         satisfied : bool
            True when w^T w < φ.

        """
        return float(np.dot(w, w)) < self.phi


class ObjectiveConstraintLayer(DirectConstraintLayer):
    """Sublevel-set constraint f(w) ≤ φ built from a convex ``Objective``.

    Wraps any convex ``Objective`` f as the single constraint
       f(w) - φ ≤ 0,
    reusing the objective's value, gradient and structured Hessian. It mirrors the dual
    role of ``DirectConstraintLayer``: a direct layer exposes a constraint value as an
    objective, and this layer exposes an objective as a constraint. A caller can
    therefore promote an objective to a constraint at the ``Problem`` level (as in an
    ε-constraint method) without a dedicated constraint class per objective form.

    The constraint set is convex when f is convex; the caller is responsible for that.
    To encode a superlevel constraint g(w) ≥ φ on a concave g, wrap f = -g, which is
    convex, and pass -φ as the bound.

    As a ``DirectConstraintLayer``, the layer serves as its own Phase I objective, the
    scalar f(w) - φ, whose objective Hessian is ∇^2 f. The Jacobian products,
    ``btls_max_step`` and ``is_satisfied`` are the ``ConstraintLayer`` defaults; in
    particular, the step bound backtracks rather than solving in closed form.

    Parameters
    ----------
     objective : Objective
        The convex objective f whose sublevel set defines the constraint. Its
        ``dimension`` fixes M.
     upper_bound : float
        The bound φ.

    Notes
    -----
    The barrier curvature atom at weight λ is
       λ ∇^2 f - (λ / f_c) ∇f ∇f^T,
    with f_c = f(w) - φ - shift < 0 on the strict interior. Since λ > 0 and f_c < 0, the
    rank-one term is an update, not a downdate.

    """

    def __init__(self, objective: Objective, upper_bound: float) -> None:
        self._f: Objective = objective
        self._phi: float = float(upper_bound)

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M of the wrapped objective.

        Returns
        -------
         M : int
            The wrapped objective's ``dimension``.

        """
        return self._f.dimension

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the shifted constraint value f(w) - φ - shift.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraint is evaluated, of length M.
         shift : float
            Relaxation subtracted from the constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The shifted constraint value, of length 1.

        """
        return np.array([float(self._f.evaluate(w)) - self._phi - shift])

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint Jacobian G = ∇f(w)^T.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The wrapped objective's gradient as a row, of shape (1, M).

        """
        return self._f.gradient(w).reshape(1, -1)

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the weighted barrier curvature of the sublevel-set constraint.

        Computes
           λ ∇^2 f - (λ / f_c) ∇f ∇f^T,   f_c = f(w) - φ - shift,
        as the wrapped objective's ``hessian_contribution`` scaled by λ, carrying one
        rank-one update with factor ∇f(w) and weight -λ / f_c, so the structure of the
        objective's Hessian is preserved. See the class Notes.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible for the shifted bound, of length M.
         lmbda : npt.NDArray[np.float64] | None
            The weight λ, of length 1. Required; the ``None`` default exists only so the
            signature matches ``Objective.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint value, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            The weighted barrier curvature.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``, or when w is not strictly feasible for the
            shifted bound (f_c ≥ 0).

        """
        if lmbda is None:
            raise ValueError(
                "ObjectiveConstraintLayer.hessian_contribution requires `lmbda`."
            )
        f_c = float(self._f.evaluate(w)) - self._phi - shift
        if f_c >= 0.0:
            raise ValueError(
                f"Barrier argument out of domain: f(w) - phi - shift = {f_c} >= 0."
            )
        lam = float(lmbda[0])
        grad = self._f.gradient(w)
        return (
            self._f.hessian_contribution(w)
            .scaled(lam)
            .low_rank_update(grad, np.array([-lam / f_c]))
        )

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the weighted constraint curvature λ ∇^2 f(w).

        This is the wrapped objective's ``hessian_contribution`` at w scaled by λ,
        without the Jacobian outer-product term.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            The weight λ, of length 1.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature λ ∇^2 f(w).

        """
        return self._f.hessian_contribution(w).scaled(float(lmbda[0]))
