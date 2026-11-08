"""Symmetric cones and Nesterov-Todd scalings for Cvxium's conic solvers.

Defines the cone primitives that the homogeneous self-dual solver (``HsdSolver``) and
the second-order cone program helpers need to work over a product cone
   K = K_1 × K_2 × ... × K_q.
``Cone`` is the abstract base: the canonical interior point, membership tests, the step
to the boundary, the Jordan product, the arrow-operator solve, and the Nesterov-Todd
(NT) scaling W, both as an object exposing W-actions and as precomputed factors for the
structured W^2 operations that assemble the Newton system. ``SecondOrderCone`` and
``Nonneg`` are the concrete cones.

``NtScaling`` and its subclasses carry the NT scaling at one primal-dual pair (x, s):
``DenseNtScaling`` for an explicit matrix W, and two private factored forms that apply
W, W^{-1}, and W^2 in time linear in the cone dimension. ``SocNtFactors`` and
``NonnegNtFactors`` hold the precomputed factors of W^2.

A new symmetric cone (for example, the positive semidefinite cone) is added by
subclassing ``Cone``, without changing the solvers.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..exceptions import ConeBoundaryError


@dataclass(frozen=True)
class SocNtFactors:
    """Precomputed Nesterov-Todd scaling factors for one second-order cone block.

    Represents W^2 in factored form,
       W^2 = scale (2 y y^T - J),
    where J = diag(1, -1, ..., -1). Returned by ``SecondOrderCone.prepare_w_squared``.

    Fields
    ------
     scale : float
        Ratio of the Lorentz norms of x and s, where the Lorentz norm of x is sqrt(x^T J
        x).
     y : npt.NDArray[np.float64]
        Unit-Lorentz-norm vector defining the rank-one term.

    """

    scale: float
    y: npt.NDArray[np.float64]


@dataclass(frozen=True)
class NonnegNtFactors:
    """Precomputed Nesterov-Todd scaling factors for one nonnegative orthant block.

    Represents the diagonal
       W^2 = diag(w_sq),   w_sq = x / s (componentwise).
    Returned by ``Nonneg.prepare_w_squared``.

    Fields
    ------
     w_sq : npt.NDArray[np.float64]
        Diagonal of W^2.

    """

    w_sq: npt.NDArray[np.float64]


class Cone(ABC):
    """Abstract symmetric cone supporting a conic interior-point method.

    Concrete subclasses implement the cone-specific arithmetic. A conic solver works
    over a product cone
       K = K_1 × K_2 × ... × K_q,
    handling each block through this interface and otherwise staying cone-agnostic.

    A symmetric cone is proper (closed, convex, pointed, with non-empty interior) and
    self-dual under the standard inner product, so the dual cone is K itself. Each
    symmetric cone carries a Jordan algebra whose identity element e_K is strictly
    interior and serves as the canonical interior point.

    """

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the cone's embedding space.

        Returns
        -------
         n : int
            Length of an element vector.

        """

    @abstractmethod
    def interior_point(self) -> npt.NDArray[np.float64]:
        """Return the canonical strictly interior point e_K.

        The point is the identity element of the cone's Jordan algebra. A conic solver
        uses it to build a strictly feasible starting iterate: both the primal point x
        and the dual slack s start at e_K, since the cone is self-dual.

        Returns
        -------
         e : npt.NDArray[np.float64]
            The identity element, of length n.

        """

    @abstractmethod
    def is_in_cone(
        self,
        x: npt.NDArray[np.float64],
        strict: bool = False,
    ) -> bool:
        """Test whether a vector lies in the cone.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Vector of length n to test.
         strict : bool
            When True, require x to lie in the open interior; otherwise the closed cone
            suffices.

        Returns
        -------
         inside : bool
            True when x is in the cone (or in its interior, when ``strict``).

        Raises
        ------
         ValueError
            When x does not have length n.

        """

    @abstractmethod
    def step_to_boundary(
        self,
        x: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step along a direction that stays in the cone.

        Computes the largest α ≥ 0 such that x + α dx remains in the closed cone. A
        conic solver uses it to choose step lengths.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current point, in the cone.
         dx : npt.NDArray[np.float64]
            Step direction, of length n.

        Returns
        -------
         alpha : float
            The largest feasible step; inf when x + α dx stays in the cone for every α ≥
            0.

        Raises
        ------
         ValueError
            When x or dx does not have length n.

        """

    @abstractmethod
    def jordan_product(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the Jordan product of two vectors in the cone's embedding space.

        Each symmetric cone has an associated Jordan algebra. Its identity element e_K
        (returned by ``interior_point``) satisfies
           e_K ∘ x = x
        for all x. A conic solver measures complementarity through the per-block Jordan
        product x ∘ s.

        Parameters
        ----------
         a : npt.NDArray[np.float64]
            Left operand, of length n.
         b : npt.NDArray[np.float64]
            Right operand, of length n.

        Returns
        -------
         product : npt.NDArray[np.float64]
            The Jordan product a ∘ b, of length n.

        Raises
        ------
         ValueError
            When a or b does not have length n.

        """

    @abstractmethod
    def nt_scaling(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> NtScaling:
        """Compute the Nesterov-Todd scaling at a primal-dual pair (x, s).

        The NT scaling matrix W is symmetric positive definite and satisfies
           W^2 s = x.
        A conic solver uses W to symmetrize the Newton system and keep iterates well
        conditioned near the cone boundary.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the cone.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the cone.

        Returns
        -------
         scaling : NtScaling
            An object that applies W, W^{-1}, and W^2 to vectors.

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When x or s is not strictly inside the cone.

        """

    @abstractmethod
    def prepare_w_squared(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> object:
        """Precompute the Nesterov-Todd factors for the structured W^2 operations.

        Callers compute the factors once per (x, s) pair and pass them to
        ``apply_w_squared`` and ``accumulate_normal_equations``, avoiding a
        recomputation on every call. Each cone uses its own structured form of W^2, so
        the dense matrix is never formed.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the cone.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the cone.

        Returns
        -------
         factors : object
            Cone-specific factors (``SocNtFactors`` or ``NonnegNtFactors``), opaque to
            the caller.

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When x or s is not strictly inside the cone.

        """

    @abstractmethod
    def apply_w_squared(
        self,
        precomputed: object,
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^2 using precomputed Nesterov-Todd factors.

        Uses the cone's structured form of W^2 (rank-one plus diagonal for the
        second-order cone, diagonal for the nonnegative orthant), so the dense matrix is
        never formed. A conic solver uses it in the back-substitution step of the Newton
        system and to assemble per-block W^2 v products.

        Parameters
        ----------
         precomputed : object
            Factors returned by this cone's ``prepare_w_squared``.
         v : npt.NDArray[np.float64]
            Vector of length n to multiply.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v, of length n.

        """

    @abstractmethod
    def accumulate_normal_equations(
        self,
        precomputed: object,
        A_block: npt.NDArray[np.float64],
        c_block: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Compute one cone block's contributions to the reduced Newton system.

        Computes the three contractions
           A_b W^2 A_b^T,   A_b W^2 c_b,   c_b^T W^2 c_b,
        where A_b holds the block's columns of the constraint matrix and c_b the block's
        entries of the cost vector. The homogeneous self-dual solver reduces each Newton
        step to a (p + 1) × (p + 1) system whose coefficients sum these contributions
        over the blocks. Each cone computes them from its structured W^2 without forming
        it.

        Parameters
        ----------
         precomputed : object
            Factors returned by this cone's ``prepare_w_squared``.
         A_block : npt.NDArray[np.float64]
            The block's columns of the constraint matrix, of shape (p, n).
         c_block : npt.NDArray[np.float64]
            The block's entries of the cost vector, of length n.

        Returns
        -------
         M_block : npt.NDArray[np.float64]
            A_b W^2 A_b^T, of shape (p, p).
         AW2c_block : npt.NDArray[np.float64]
            A_b W^2 c_b, of length p.
         cW2c_block : float
            c_b^T W^2 c_b.

        """

    @abstractmethod
    def solve_jordan_arrow(
        self,
        v: npt.NDArray[np.float64],
        rhs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solve the arrow system L(v) z = rhs.

        The arrow operator L(v) is the matrix of left Jordan multiplication by v,
           L(v) y = v ∘ y.
        A conic solver uses it to build the right-hand side of the centering step.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector strictly inside the cone, so that L(v) is invertible.
         rhs : npt.NDArray[np.float64]
            Right-hand side, of length n.

        Returns
        -------
         z : npt.NDArray[np.float64]
            The solution, of length n.

        """


class SecondOrderCone(Cone):
    """Second-order (Lorentz) cone of dimension n.

    The cone is
       K = { x : x_0 ≥ ‖x_{1:}‖ },
    with the interior where the inequality is strict. Its Jordan product is
       x ∘ y = (x^T y, x_0 y_{1:} + y_0 x_{1:}),
    and the spectral decomposition of x has eigenvalues x_0 ± ‖x_{1:}‖. Throughout,
       J = diag(1, -1, ..., -1),
    and the Lorentz norm of x is sqrt(x^T J x).

    Parameters
    ----------
     dim : int
        Dimension n of the cone. Must be at least 2, or the constructor raises
        ``ValueError``; the one-dimensional cone is ``Nonneg(1)``.

    """

    def __init__(self, dim: int) -> None:
        if dim < 2:
            raise ValueError(
                f"SecondOrderCone requires dim >= 2 (got {dim}); "
                "the trivial 1-D cone is the non-negative orthant -- "
                "use Nonneg(1) for that case."
            )
        self._dim = dim

    def dimension(self) -> int:
        """Return the dimension of the cone.

        Returns
        -------
         n : int
            The dimension given at construction.

        """
        return self._dim

    def interior_point(self) -> npt.NDArray[np.float64]:
        """Return the canonical interior point e_K = (1, 0, ..., 0).

        Returns
        -------
         e : npt.NDArray[np.float64]
            The Jordan identity, of length n.

        """
        e = np.zeros(self._dim, dtype=np.float64)
        e[0] = 1.0
        return e

    def is_in_cone(
        self,
        x: npt.NDArray[np.float64],
        strict: bool = False,
    ) -> bool:
        """Test whether x_0 ≥ ‖x_{1:}‖, strictly when requested.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Vector of length n to test.
         strict : bool
            When True, require x_0 > ‖x_{1:}‖ (the open interior).

        Returns
        -------
         inside : bool
            True when x is in the cone (or in its interior, when ``strict``).

        Raises
        ------
         ValueError
            When x does not have length n.

        """
        if x.shape != (self._dim,):
            raise ValueError(
                f"x has shape {x.shape}; expected ({self._dim},) for "
                f"a SOC of dimension {self._dim}."
            )
        norm_rest = float(np.linalg.norm(x[1:]))
        if strict:
            return bool(x[0] > norm_rest)
        return bool(x[0] >= norm_rest)

    def step_to_boundary(
        self,
        x: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step along a direction that stays in the second-order cone.

        Along the ray x + α dx, membership needs two conditions. The first is
        nonnegativity of the Lorentz quadratic
           f(α) = (x_0 + α dx_0)^2 - ‖(x + α dx)_{1:}‖^2 = A α^2 + 2 B α + C,
        where
           A = dx_0^2 - ‖dx_{1:}‖^2,
           B = x_0 dx_0 - x_{1:}^T dx_{1:},
           C = x_0^2 - ‖x_{1:}‖^2 ≥ 0.
        The second is the apex condition
           x_0 + α dx_0 ≥ 0,
        which the squared form alone misses, since f(α) ≥ 0 also holds on the reflected
        cone -K. The step is the smaller of the largest α satisfying each, found in
        closed form.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current point, in the cone (so that C ≥ 0).
         dx : npt.NDArray[np.float64]
            Step direction, of length n.

        Returns
        -------
         alpha : float
            The largest feasible step; inf when x + α dx stays in the cone for every α ≥
            0.

        Raises
        ------
         ValueError
            When x or dx does not have length n.

        """
        if x.shape != (self._dim,) or dx.shape != (self._dim,):
            raise ValueError(
                f"x and dx must have shape ({self._dim},); got "
                f"{x.shape} and {dx.shape}."
            )
        x_0 = float(x[0])
        x_rest = x[1:]
        dx_0 = float(dx[0])
        dx_rest = dx[1:]

        A = dx_0 * dx_0 - float(np.dot(dx_rest, dx_rest))
        B = x_0 * dx_0 - float(np.dot(x_rest, dx_rest))
        C = x_0 * x_0 - float(np.dot(x_rest, x_rest))

        # Magnitude condition: largest alpha >= 0 with A alpha^2 + 2 B alpha + C >= 0.
        # Pick a relative scale for the "A is effectively zero" test.
        scale = abs(A) + abs(B) + abs(C) + 1.0
        if abs(A) <= 1e-15 * scale:
            # Linear: f(alpha) = 2 B alpha + C.
            if B >= 0.0:
                alpha_mag = np.inf
            elif C <= 0.0:
                alpha_mag = 0.0
            else:
                alpha_mag = -C / (2.0 * B)
        else:
            disc = B * B - A * C
            if disc < 0.0:
                # Quadratic stays one sign; with C >= 0 it must be positive.
                # (The A < 0 + disc < 0 case is impossible when C >= 0
                # except in measure-zero boundary numerics.)
                alpha_mag = np.inf if A > 0.0 else 0.0
            else:
                sqrt_disc = float(np.sqrt(disc))
                r_plus = (-B + sqrt_disc) / A
                r_minus = (-B - sqrt_disc) / A
                if A > 0.0:
                    # Opens up; f(0) = C >= 0 places 0 outside (r_minus, r_plus).
                    # Both roots have the same sign (product C/A >= 0); pick
                    # the smaller positive root, or infinity when both <= 0.
                    if r_plus <= 0.0:
                        alpha_mag = np.inf
                    else:
                        alpha_mag = max(r_minus, 0.0)
                else:
                    # A < 0; opens down. f(0) >= 0 puts 0 between the roots
                    # (or at one of them). Largest alpha with f >= 0 is the
                    # larger root.
                    alpha_mag = max(max(r_plus, r_minus), 0.0)

        # Apex condition: x_0 + alpha dx_0 >= 0.
        if dx_0 >= 0.0:
            alpha_apex = np.inf
        else:
            alpha_apex = -x_0 / dx_0

        return min(alpha_mag, alpha_apex)

    def jordan_product(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the second-order cone Jordan product.

        The product is
           a ∘ b = (a^T b, a_0 b_{1:} + b_0 a_{1:}).

        Parameters
        ----------
         a : npt.NDArray[np.float64]
            Left operand, of length n.
         b : npt.NDArray[np.float64]
            Right operand, of length n.

        Returns
        -------
         product : npt.NDArray[np.float64]
            The Jordan product a ∘ b, of length n.

        Raises
        ------
         ValueError
            When a or b does not have length n.

        """
        if a.shape != (self._dim,) or b.shape != (self._dim,):
            raise ValueError(
                f"a and b must have shape ({self._dim},); got {a.shape} and {b.shape}."
            )
        result = np.empty(self._dim, dtype=np.float64)
        result[0] = float(np.dot(a, b))
        result[1:] = a[0] * b[1:] + b[0] * a[1:]
        return result

    def nt_scaling(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> NtScaling:
        """Compute the factored Nesterov-Todd scaling for the second-order cone.

        The returned scaling applies W, W^{-1}, and W^2 in O(n) time, rather than
        forming the dense n × n matrix and inverting and squaring it at O(n^3) cost.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the cone.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the cone.

        Returns
        -------
         scaling : NtScaling
            The factored NT scaling at (x, s).

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When x or s is not strictly inside the cone.

        Notes
        -----
        The construction follows (Andersen, Roos, and Terlaky, 2003) and (Sturm, 1999):
        1. Compute the Lorentz norms
              φ = sqrt(x^T J x),   ψ = sqrt(s^T J s),
           both positive in the strict interior.
        2. Normalize to unit Lorentz norm: x_hat = x / φ and s_hat = s / ψ.
        3. Set γ = sqrt((1 + x_hat^T s_hat) / 2).
        4. Form
              y = (x_hat + J s_hat) / (2 γ),
           the unit-Lorentz-norm Jordan square of the scaling vector w_bar.
        5. Take the Jordan square root w_bar of y (see ``_jordan_sqrt``).
        6. Set
              W = sqrt(φ / ψ) Q(w_bar),
           where Q(w) = 2 w w^T - det(w) J is the quadratic representation, and
           det(w_bar) = 1 by the normalization.

        The identity W^2 s = x holds because Q(w_bar) is an automorphism of the cone, so
        it preserves Lorentz norms: W^2 scales them by φ / ψ, taking the norm of s to
        that of x, while w_bar aligns the directions.

        References
        ----------
        Andersen, Erling D., Roos, Cornelis and Terlaky, Tamás, On Implementing a
        Primal-Dual Interior-Point Method for Conic Quadratic Optimization, Mathematical
        Programming 95, 2003.

        Sturm, Jos F., Using SeDuMi 1.02, a MATLAB Toolbox for Optimization over
        Symmetric Cones, Optimization Methods and Software 11, 1999.

        """
        scale, y = self.factored_w_squared(x, s)
        # Factored rather than dense: skips the O(n^3) inversion and squaring that
        # DenseNtScaling's constructor would do; each apply stays O(n) via the closed
        # form Q(u) v = 2 (u^T v) u - J v.
        w_bar = self._jordan_sqrt(y)
        return _FactoredSocNtScaling(scale=scale, w_bar=w_bar, y=y)

    def factored_w_squared(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """Compute the factored form of W^2 for the second-order cone.

        Returns the factors of
           W^2 = scale (2 y y^T - J),
        where
           scale = φ / ψ,   y = (x_hat + J s_hat) / (2 γ)
        are the Nesterov-Todd intermediates defined in ``nt_scaling``. Because det(y) =
        1 by construction, the quadratic representation simplifies to
           Q(y) = 2 y y^T - J,
        and the Jordan algebra identity Q(w)^2 = Q(w ∘ w) gives
           W^2 = scale Q(w_bar)^2 = scale Q(y).
        So W^2, unlike W itself, needs no Jordan square root.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the cone.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the cone.

        Returns
        -------
         scale : float
            The ratio φ / ψ of the Lorentz norms of x and s.
         y : npt.NDArray[np.float64]
            The unit-Lorentz-norm vector of length n defining the rank-one term.

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When x or s is not strictly inside the cone.

        """
        n = self._dim
        if x.shape != (n,) or s.shape != (n,):
            raise ValueError(
                f"x and s must have shape ({n},); got {x.shape} and {s.shape}."
            )
        phi_sq = float(x[0] * x[0] - np.dot(x[1:], x[1:]))
        psi_sq = float(s[0] * s[0] - np.dot(s[1:], s[1:]))
        if phi_sq <= 0.0 or psi_sq <= 0.0:
            raise ConeBoundaryError(
                "NT scaling requires x and s strictly inside the SOC; got "
                f"x^T J x = {phi_sq}, s^T J s = {psi_sq}."
            )
        phi = np.sqrt(phi_sq)
        psi = np.sqrt(psi_sq)
        x_hat = x / phi
        s_hat = s / psi
        gamma = np.sqrt(0.5 * (1.0 + float(x_hat @ s_hat)))
        y = np.empty(n, dtype=np.float64)
        y[0] = (x_hat[0] + s_hat[0]) / (2.0 * gamma)
        y[1:] = (x_hat[1:] - s_hat[1:]) / (2.0 * gamma)
        scale = float(phi / psi)
        return scale, y

    def prepare_w_squared(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> SocNtFactors:
        """Precompute the second-order cone factors of W^2.

        Packages the result of ``factored_w_squared``.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the cone.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the cone.

        Returns
        -------
         factors : SocNtFactors
            The factors (scale, y), with W^2 = scale (2 y y^T - J).

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When x or s is not strictly inside the cone.

        """
        scale, y = self.factored_w_squared(x, s)
        return SocNtFactors(scale=scale, y=y)

    def apply_w_squared(
        self,
        precomputed: object,
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^2 using the factored second-order cone form.

        Computes
           W^2 v = scale (2 (y^T v) y - J v)
        in O(n) time.

        Parameters
        ----------
         precomputed : object
            The ``SocNtFactors`` returned by ``prepare_w_squared``.
         v : npt.NDArray[np.float64]
            Vector of length n to multiply.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v, of length n.

        """
        assert isinstance(precomputed, SocNtFactors)
        scale, y = precomputed.scale, precomputed.y
        yv = float(y @ v)
        out = 2.0 * yv * y
        out[0] -= v[0]
        out[1:] -= -v[1:]
        return scale * out

    def accumulate_normal_equations(
        self,
        precomputed: object,
        A_block: npt.NDArray[np.float64],
        c_block: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Compute the second-order cone block's contributions to the reduced system.

        With W^2 = scale (2 y y^T - J),
           A_b W^2 A_b^T = scale (2 (A_b y) (A_b y)^T - a_0 a_0^T + A_r A_r^T),
        where a_0 is the first column of A_b and A_r holds the remaining columns; the
        contractions with c_b follow the same pattern. The cost is O(p^2 n) per block,
        dominated by A_r A_r^T, and avoids forming W^2 and the O(p n^2) product A_b W^2.

        Parameters
        ----------
         precomputed : object
            The ``SocNtFactors`` returned by ``prepare_w_squared``.
         A_block : npt.NDArray[np.float64]
            The block's columns of the constraint matrix, of shape (p, n).
         c_block : npt.NDArray[np.float64]
            The block's entries of the cost vector, of length n.

        Returns
        -------
         M_block : npt.NDArray[np.float64]
            A_b W^2 A_b^T, of shape (p, p).
         AW2c_block : npt.NDArray[np.float64]
            A_b W^2 c_b, of length p.
         cW2c_block : float
            c_b^T W^2 c_b.

        """
        assert isinstance(precomputed, SocNtFactors)
        scale, y = precomputed.scale, precomputed.y
        Aby = A_block @ y
        yc = float(y @ c_block)
        A0 = A_block[:, 0]
        A_rest = A_block[:, 1:]
        c0 = float(c_block[0])
        c_rest = c_block[1:]
        M_block = scale * (
            2.0 * np.outer(Aby, Aby) - np.outer(A0, A0) + A_rest @ A_rest.T
        )
        AW2c_block = scale * (2.0 * yc * Aby - c0 * A0 + A_rest @ c_rest)
        cW2c_block = scale * (2.0 * yc * yc - c0 * c0 + float(c_rest @ c_rest))
        return M_block, AW2c_block, cW2c_block

    def solve_jordan_arrow(
        self,
        v: npt.NDArray[np.float64],
        rhs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solve the second-order cone arrow system L(v) z = r in O(n) time.

        The arrow matrix is
                   _               _
           L(v) = |  v_0   v_{1:}^T |
                  |_ v_{1:}  v_0 I _|
        Eliminating the lower-right block v_0 I (a Schur complement) gives a closed form
        that avoids a dense O(n^3) solve. With the cone determinant
           d = v_0^2 - ‖v_{1:}‖^2,
        the solution is
           z_0    = (v_0 r_0 - v_{1:}^T r_{1:}) / d,
           z_{1:} = r_{1:} / v_0 - v_{1:} r_0 / d + v_{1:} (v_{1:}^T r_{1:}) / (v_0 d).

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector strictly inside the cone, so that d > 0 and v_0 > 0.
         rhs : npt.NDArray[np.float64]
            Right-hand side r, of length n.

        Returns
        -------
         z : npt.NDArray[np.float64]
            The solution, of length n.

        """
        v_0 = float(v[0])
        v_rest = v[1:]
        det_L = v_0 * v_0 - float(np.dot(v_rest, v_rest))
        r_0 = float(rhs[0])
        r_rest = rhs[1:]
        z = np.empty_like(v)
        v_dot_r = float(np.dot(v_rest, r_rest))
        z[0] = (v_0 * r_0 - v_dot_r) / det_L
        z[1:] = (
            -v_rest * (r_0 / det_L) + r_rest / v_0 + v_rest * (v_dot_r / (v_0 * det_L))
        )
        return z

    @staticmethod
    def _jordan_sqrt(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the Jordan square root of a vector in the second-order cone.

        For y strictly inside the cone, the spectral decomposition is
           y = λ_+ c_+ + λ_- c_-,
        with eigenvalues λ_± = y_0 ± ‖y_{1:}‖ and orthogonal idempotents
           c_± = (1, ±y_{1:} / ‖y_{1:}‖) / 2.
        The Jordan square root is sqrt(λ_+) c_+ + sqrt(λ_-) c_-. When y_{1:} = 0 (to
        roundoff), y lies on the cone's axis; the decomposition is degenerate, but the
        limit sqrt(y_0) e_K is well defined.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            Vector strictly inside the cone.

        Returns
        -------
         w : npt.NDArray[np.float64]
            The vector w inside the cone with w ∘ w = y.

        """
        n = y.shape[0]
        rest_norm = float(np.linalg.norm(y[1:]))
        if rest_norm <= 1e-14 * max(abs(float(y[0])), 1.0):
            sq = np.zeros(n, dtype=np.float64)
            sq[0] = np.sqrt(float(y[0]))
            return sq
        lam_plus = float(y[0]) + rest_norm
        lam_minus = float(y[0]) - rest_norm
        sqrt_plus = np.sqrt(lam_plus)
        sqrt_minus = np.sqrt(lam_minus)
        sq = np.empty(n, dtype=np.float64)
        sq[0] = 0.5 * (sqrt_plus + sqrt_minus)
        sq[1:] = 0.5 * (sqrt_plus - sqrt_minus) * (y[1:] / rest_norm)
        return sq


class Nonneg(Cone):
    """Nonnegative orthant of dimension n.

    The cone is
       R^n_+ = { x : x_i ≥ 0 for all i },
    the simplest symmetric cone. Its Jordan product is the componentwise product, its
    identity element is e = (1, 1, ..., 1), its arrow operator is L(v) = diag(v), and
    its Nesterov-Todd scaling is
       W = diag(sqrt(x / s)),   so that   W^2 = diag(x / s).
    A product cone can mix nonnegative blocks with second-order cone blocks, for example
    to express sign constraints.

    Parameters
    ----------
     dim : int
        Dimension n of the orthant. Must be at least 1, or the constructor raises
        ``ValueError``.

    """

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"Nonneg requires dim >= 1 (got {dim}).")
        self._dim = dim

    def dimension(self) -> int:
        """Return the dimension of the orthant.

        Returns
        -------
         n : int
            The dimension given at construction.

        """
        return self._dim

    def interior_point(self) -> npt.NDArray[np.float64]:
        """Return the canonical interior point e = (1, 1, ..., 1).

        Returns
        -------
         e : npt.NDArray[np.float64]
            The Jordan identity, of length n.

        """
        return np.ones(self._dim, dtype=np.float64)

    def is_in_cone(
        self,
        x: npt.NDArray[np.float64],
        strict: bool = False,
    ) -> bool:
        """Test whether every component is nonnegative, strictly when requested.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Vector of length n to test.
         strict : bool
            When True, require every component to be strictly positive.

        Returns
        -------
         inside : bool
            True when x is in the orthant (or in its interior, when ``strict``).

        Raises
        ------
         ValueError
            When x does not have length n.

        """
        if x.shape != (self._dim,):
            raise ValueError(
                f"x has shape {x.shape}; expected ({self._dim},) for "
                f"a Nonneg cone of dimension {self._dim}."
            )
        if strict:
            return bool(np.all(x > 0.0))
        return bool(np.all(x >= 0.0))

    def step_to_boundary(
        self,
        x: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step along a direction that stays in the orthant.

        Computes the largest α ≥ 0 with x + α dx ≥ 0 componentwise, which is the minimum
        of -x_i / dx_i over the components with dx_i < 0.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current point, in the orthant.
         dx : npt.NDArray[np.float64]
            Step direction, of length n.

        Returns
        -------
         alpha : float
            The largest feasible step; inf when dx ≥ 0 componentwise.

        Raises
        ------
         ValueError
            When x or dx does not have length n.

        """
        if x.shape != (self._dim,) or dx.shape != (self._dim,):
            raise ValueError(
                f"x and dx must have shape ({self._dim},); got "
                f"{x.shape} and {dx.shape}."
            )
        mask = dx < 0.0
        if not np.any(mask):
            return float("inf")
        return float(np.min(-x[mask] / dx[mask]))

    def jordan_product(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the componentwise product, the orthant's Jordan product.

        Parameters
        ----------
         a : npt.NDArray[np.float64]
            Left operand, of length n.
         b : npt.NDArray[np.float64]
            Right operand, of length n.

        Returns
        -------
         product : npt.NDArray[np.float64]
            The componentwise product a ∘ b, of length n.

        Raises
        ------
         ValueError
            When a or b does not have length n.

        """
        if a.shape != (self._dim,) or b.shape != (self._dim,):
            raise ValueError(
                f"a and b must have shape ({self._dim},); got {a.shape} and {b.shape}."
            )
        return a * b

    def nt_scaling(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> NtScaling:
        """Compute the diagonal Nesterov-Todd scaling W = diag(sqrt(x / s)).

        The returned scaling stores only the diagonal and applies W, W^{-1}, and W^2 in
        O(n) time, rather than forming the dense n × n matrix and inverting and squaring
        it at O(n^3) cost.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the orthant.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the orthant.

        Returns
        -------
         scaling : NtScaling
            The diagonal NT scaling at (x, s).

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When some component of x or s is not strictly positive.

        """
        n = self._dim
        if x.shape != (n,) or s.shape != (n,):
            raise ValueError(
                f"x and s must have shape ({n},); got {x.shape} and {s.shape}."
            )
        if np.any(x <= 0.0) or np.any(s <= 0.0):
            raise ConeBoundaryError(
                "NT scaling requires x and s strictly inside the Nonneg cone "
                "(all components > 0)."
            )
        return _FactoredDiagonalNtScaling(w_diag=np.sqrt(x / s))

    def prepare_w_squared(
        self,
        x: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
    ) -> NonnegNtFactors:
        """Precompute the diagonal of W^2, w_sq = x / s.

        Validates the shapes and the strict interior here, as the second-order cone
        does, so that ``apply_w_squared`` and ``accumulate_normal_equations`` cannot
        silently propagate inf or nan when some s_i = 0.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Primal point, strictly inside the orthant.
         s : npt.NDArray[np.float64]
            Dual slack, strictly inside the orthant.

        Returns
        -------
         factors : NonnegNtFactors
            The diagonal of W^2.

        Raises
        ------
         ValueError
            When x or s does not have length n.
         ConeBoundaryError
            When some component of x or s is not strictly positive.

        """
        n = self._dim
        if x.shape != (n,) or s.shape != (n,):
            raise ValueError(
                f"x and s must have shape ({n},); got {x.shape} and {s.shape}."
            )
        if np.any(x <= 0.0) or np.any(s <= 0.0):
            raise ConeBoundaryError(
                "Nonneg NT scaling requires x and s strictly inside the cone "
                "(all components > 0)."
            )
        return NonnegNtFactors(w_sq=x / s)

    def apply_w_squared(
        self,
        precomputed: object,
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the diagonal W^2.

        Computes W^2 v = w_sq v componentwise, in O(n) time.

        Parameters
        ----------
         precomputed : object
            The ``NonnegNtFactors`` returned by ``prepare_w_squared``.
         v : npt.NDArray[np.float64]
            Vector of length n to multiply.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v, of length n.

        """
        assert isinstance(precomputed, NonnegNtFactors)
        return precomputed.w_sq * v

    def accumulate_normal_equations(
        self,
        precomputed: object,
        A_block: npt.NDArray[np.float64],
        c_block: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Compute the orthant block's contributions to the reduced system.

        W^2 = diag(w_sq) is diagonal, so each contraction scales the columns of A_b by
        w_sq: A_b W^2 A_b^T costs O(p^2 n), and the other two cost O(p n).

        Parameters
        ----------
         precomputed : object
            The ``NonnegNtFactors`` returned by ``prepare_w_squared``.
         A_block : npt.NDArray[np.float64]
            The block's columns of the constraint matrix, of shape (p, n).
         c_block : npt.NDArray[np.float64]
            The block's entries of the cost vector, of length n.

        Returns
        -------
         M_block : npt.NDArray[np.float64]
            A_b W^2 A_b^T, of shape (p, p).
         AW2c_block : npt.NDArray[np.float64]
            A_b W^2 c_b, of length p.
         cW2c_block : float
            c_b^T W^2 c_b.

        """
        assert isinstance(precomputed, NonnegNtFactors)
        w_sq = precomputed.w_sq
        AW2 = A_block * w_sq  # (p, n_b)
        M_block = AW2 @ A_block.T
        AW2c_block = AW2 @ c_block
        cW2c_block = float(np.dot(w_sq, c_block * c_block))
        return M_block, AW2c_block, cW2c_block

    def solve_jordan_arrow(
        self,
        v: npt.NDArray[np.float64],
        rhs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solve the orthant arrow system diag(v) z = rhs by componentwise division.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Strictly positive vector of length n.
         rhs : npt.NDArray[np.float64]
            Right-hand side, of length n.

        Returns
        -------
         z : npt.NDArray[np.float64]
            The solution rhs / v, of length n.

        """
        return rhs / v


class NtScaling(ABC):
    """Abstract Nesterov-Todd scaling at one primal-dual pair (x, s).

    Declares the W-action interface that every concrete scaling shares. Each subclass
    chooses its own representation (an explicit matrix or an O(n) factored form) and
    supplies its own constructor; this base defines none, so the subclass signatures are
    not incompatible overrides. Every implementation satisfies
       W^2 s = x
    for the pair (x, s) at which it was constructed.

    """

    @abstractmethod
    def apply_W(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Wv : npt.NDArray[np.float64]
            The product W v.

        """

    @abstractmethod
    def apply_W_inv(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^{-1}.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Winv_v : npt.NDArray[np.float64]
            The product W^{-1} v.

        """

    @abstractmethod
    def apply_W_squared(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^2.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v.

        """


class DenseNtScaling(NtScaling):
    """Nesterov-Todd scaling stored as an explicit matrix.

    Stores W as a dense matrix and precomputes W^{-1} and W^2 at construction, so each
    apply is a matrix-vector product. Construction costs O(n^3), so the cones in this
    module return factored scalings instead; this class serves a caller that already
    holds an explicit W.

    Parameters
    ----------
     W : npt.NDArray[np.float64]
        Symmetric positive definite scaling matrix with W^2 s = x for the intended pair
        (x, s). The constructor raises ``ValueError`` when W is not square, and NumPy's
        ``LinAlgError`` when it is singular.

    """

    def __init__(self, W: npt.NDArray[np.float64]) -> None:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square; got shape {W.shape}.")
        self._W: npt.NDArray[np.float64] = W
        self._W_inv: npt.NDArray[np.float64] = np.asarray(
            np.linalg.inv(W), dtype=np.float64
        )
        self._W_squared: npt.NDArray[np.float64] = W @ W

    def apply_W(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W.

        A matrix-vector product with the matrix stored at construction.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Wv : npt.NDArray[np.float64]
            The product W v.

        """
        return self._W @ v

    def apply_W_inv(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^{-1}.

        A matrix-vector product with the matrix stored at construction.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Winv_v : npt.NDArray[np.float64]
            The product W^{-1} v.

        """
        return self._W_inv @ v

    def apply_W_squared(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^2.

        A matrix-vector product with the matrix stored at construction.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v.

        """
        return self._W_squared @ v


class _FactoredSocNtScaling(NtScaling):
    """Factored Nesterov-Todd scaling for the second-order cone.

    For the second-order cone,
       W = sqrt(scale) Q(w_bar),   Q(w) = 2 w w^T - det(w) J,
    where w_bar is the unit-Lorentz-norm vector with w_bar ∘ w_bar = y. Storing W
    densely would cost O(n^2) memory and O(n^3) time to invert and square; this class
    keeps only scale, w_bar, J w_bar, and y, and applies each operator in O(n) time:
       W v      = sqrt(scale) (2 (w_bar^T v) w_bar - J v),
       W^{-1} v = (2 ((J w_bar)^T v) (J w_bar) - J v) / sqrt(scale),
       W^2 v    = scale (2 (y^T v) y - J v).
    The inverse uses Q(w)^{-1} = Q(w^{-1}) and the fact that the Jordan inverse of
    w_bar, whose determinant is 1, is J w_bar.

    Parameters
    ----------
     scale : float
        Ratio of the Lorentz norms of x and s.
     w_bar : npt.NDArray[np.float64]
        The scaling vector, the Jordan square root of y.
     y : npt.NDArray[np.float64]
        The unit-Lorentz-norm vector with W^2 = scale (2 y y^T - J).

    """

    def __init__(
        self,
        scale: float,
        w_bar: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        self._scale: float = float(scale)
        self._sqrt_scale: float = float(np.sqrt(scale))
        self._inv_sqrt_scale: float = 1.0 / self._sqrt_scale
        self._w_bar: npt.NDArray[np.float64] = w_bar
        # J w_bar = (w_bar[0], -w_bar[1:]).
        self._j_w_bar: npt.NDArray[np.float64] = w_bar.copy()
        self._j_w_bar[1:] *= -1.0
        # y = w_bar o w_bar (the Jordan square), which defines the factored W^2.
        self._y: npt.NDArray[np.float64] = y

    @staticmethod
    def _q_apply(
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        scale: float,
    ) -> npt.NDArray[np.float64]:
        """Apply a scaled quadratic representation to a vector.

        Computes
           scale Q(u) v = scale (2 (u^T v) u - J v)
        for a unit-determinant u, in O(n) time with a single allocation.

        Parameters
        ----------
         u : npt.NDArray[np.float64]
            Unit-determinant vector defining Q(u).
         v : npt.NDArray[np.float64]
            Vector to multiply.
         scale : float
            Scalar multiplying the result.

        Returns
        -------
         out : npt.NDArray[np.float64]
            The product scale Q(u) v.

        """
        # Single O(n) allocation; scale folded in via final in-place multiply
        # so callers don't pay a second alloc for `scale_factor * out`.
        uv = float(u @ v)
        out = (2.0 * uv) * u
        out[0] -= v[0]
        out[1:] += v[1:]  # subtract J v's tail (which is -v[1:]) -> add v[1:]
        out *= scale
        return out

    def apply_W(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W.

        Computes
           W v = sqrt(scale) (2 (w_bar^T v) w_bar - J v)
        in O(n) time.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Wv : npt.NDArray[np.float64]
            The product W v.

        """
        return self._q_apply(self._w_bar, v, self._sqrt_scale)

    def apply_W_inv(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^{-1}.

        Computes
           W^{-1} v = (2 ((J w_bar)^T v) (J w_bar) - J v) / sqrt(scale)
        in O(n) time.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Winv_v : npt.NDArray[np.float64]
            The product W^{-1} v.

        """
        return self._q_apply(self._j_w_bar, v, self._inv_sqrt_scale)

    def apply_W_squared(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by W^2.

        Computes
           W^2 v = scale (2 (y^T v) y - J v)
        in O(n) time.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v.

        """
        return self._q_apply(self._y, v, self._scale)


class _FactoredDiagonalNtScaling(NtScaling):
    """Factored Nesterov-Todd scaling for the nonnegative orthant.

    Represents W = diag(w_diag), storing only the diagonal, its reciprocal, and its
    square. Each apply is a componentwise product in O(n) time, rather than forming the
    dense n × n matrix and inverting and squaring it at O(n^3) cost.

    Parameters
    ----------
     w_diag : npt.NDArray[np.float64]
        Strictly positive diagonal of W, sqrt(x / s) componentwise.

    """

    def __init__(
        self,
        w_diag: npt.NDArray[np.float64],
    ) -> None:
        self._w_diag: npt.NDArray[np.float64] = w_diag
        self._w_inv_diag: npt.NDArray[np.float64] = 1.0 / w_diag
        # Precompute the elementwise square so ``apply_W_squared`` is a
        # single multiply per call instead of two (the IPM hits this on
        # every Newton step).
        self._w_diag_sq: npt.NDArray[np.float64] = w_diag * w_diag

    def apply_W(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by the diagonal W.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Wv : npt.NDArray[np.float64]
            The componentwise product w_diag v.

        """
        return self._w_diag * v

    def apply_W_inv(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by the diagonal W^{-1}.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         Winv_v : npt.NDArray[np.float64]
            The componentwise quotient v / w_diag.

        """
        return self._w_inv_diag * v

    def apply_W_squared(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by the diagonal W^2.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector to multiply, of the cone's dimension.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The componentwise product w_diag^2 v.

        """
        return self._w_diag_sq * v
