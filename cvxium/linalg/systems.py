"""Composable structured linear systems for Cvxium's Newton steps.

A ``MatrixSystem`` represents a structured symmetric positive-definite matrix H that can
solve H x = b and multiply H y efficiently by exploiting its structure: zero, diagonal,
banded, dense, low-rank updated, block diagonal, permuted, or arrow (a block bordered by
a few dense rows and columns). MatrixSystems compose: a low-rank update can sit on top
of any base, the block-diagonal, permuted and arrow systems take other MatrixSystems as
parts, and ``plus`` sums two systems while keeping their structure, which is how
``Problem.build_hessian_system`` assembles a Hessian from its contributions. The
arithmetic lives in the kernels of ``numerical_helpers``; this module arranges them into
a type-safe tree.

The indefinite KKT matrix
    _        _
   |  H  A^T  |
   |_ A   0  _|
of an equality-constrained Newton step is not a MatrixSystem (it is not positive
definite, and has no ``multiply`` or ``low_rank_update``). ``SaddleSystem`` and its
subclasses solve it, holding a ``MatrixSystem`` for the H block; a ``SaddleFactory``
chooses the subclass for an equality constraint. ``CoupledSaddleSystem`` and
``EqualityCoupledSaddleSystem`` solve the two-block saddle-point systems of a
stationarity solver.

"""

from __future__ import annotations

import enum
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse
from scipy import linalg

from ..exceptions import NewtonStepError
from ..linalg.numerical_helpers import (
    _apply_rank_one_update,
    _apply_rank_p_update,
    _prepare_rank_one_update,
    _prepare_rank_p_update,
    _RankOneUpdatePrep,
    _RankPUpdatePrep,
    _solve_schur_complement,
    factor_banded,
    multiply_banded,
    multiply_block,
    multiply_block_diagonal,
    multiply_block_plus_one,
    multiply_diagonal,
    multiply_rank_one_update,
    multiply_rank_p_update,
    solve_arrow_sparsity_pattern,
    solve_banded,
    solve_banded_factored,
    solve_block_diagonal,
    solve_block_plus_one,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_p_update,
    solve_with_schur,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class IncompatibleStructureError(ValueError):
    """Error raised when two ``MatrixSystem`` shapes cannot be summed structurally.

    Bases of the same shape, or a base and a strictly simpler one, combine through
    ``MatrixSystem.plus``; an incompatible pair, such as a banded matrix and a
    block-diagonal one, raises this rather than silently forming a dense matrix. A
    low-rank update sits on top of any base, so it never raises this by itself.

    """


class Definiteness(enum.Enum):
    """Declared definiteness of a ``MatrixSystem``.

    A ``MatrixSystem`` is positive definite by contract, the shape the barrier and
    primal-dual solvers factor. Maximizing a concave objective by minimizing its
    negation produces a negative-definite Hessian contribution, which
    ``MatrixSystem.negated`` declares by flipping this value.

    This is a declaration (the contract a system advertises), distinct from the
    numerical check ``MatrixSystem.is_positive_definite``.

    """

    POSITIVE_DEFINITE = "positive_definite"
    NEGATIVE_DEFINITE = "negative_definite"
    INDEFINITE = "indefinite"

    @property
    def negated(self) -> Definiteness:
        """Return the definiteness of -H, given that of H.

        Returns
        -------
         definiteness : Definiteness
            Positive and negative definite swap; indefinite stays indefinite.

        """
        if self is Definiteness.POSITIVE_DEFINITE:
            return Definiteness.NEGATIVE_DEFINITE
        if self is Definiteness.NEGATIVE_DEFINITE:
            return Definiteness.POSITIVE_DEFINITE
        return Definiteness.INDEFINITE


def _normalize_low_rank(
    kappa: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64] | float | None,
    dimension: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Normalize the factor and weights of a low-rank update.

    The update is kappa diag(d) kappa^T. Columns of kappa with zero weight contribute
    nothing, so they are dropped. Carried through, they would put a zero on the diagonal
    of the equilibrated Woodbury matrix, which the rank-p preparation rejects. Zero
    weights are reachable in the interior-point loop, where d = λ / (-f) and a
    multiplier on a slack constraint can underflow to exactly zero near the boundary.

    Parameters
    ----------
     kappa : npt.NDArray[np.float64]
        A vector of length M (rank one) or a matrix of shape (M, p) (rank p).
     d : npt.NDArray[np.float64] | float | None
        A scalar or a vector of length p of weights, or ``None`` for unit weights.
     dimension : int
        Order M of the system being updated; ``kappa`` must have this many rows.

    Returns
    -------
     kappa : npt.NDArray[np.float64]
        The factor as a matrix of shape (M, r) with r ≤ p, its zero-weight columns
        dropped (r = 0 when every weight was zero); unchanged when ``d`` is ``None``.
     d : npt.NDArray[np.float64] | None
        The weights as a vector of length r aligned with the returned ``kappa``, or
        ``None`` for unit weights.

    Raises
    ------
     ValueError
        When ``kappa`` is not 1-D or 2-D, its row count is not ``dimension``, or ``d``
        is neither a scalar nor a vector of length p.

    """
    kappa = np.asarray(kappa, dtype=np.float64)
    if kappa.ndim == 1:
        kappa = kappa[:, np.newaxis]
    if kappa.ndim != 2:
        raise ValueError("kappa must be a 1D or 2D array.")
    if kappa.shape[0] != dimension:
        raise ValueError(
            f"kappa has {kappa.shape[0]} rows but the MatrixSystem has dimension "
            f"{dimension}."
        )

    p = kappa.shape[1]
    if d is None:
        return kappa, None

    d_array = np.asarray(d, dtype=np.float64)
    if d_array.ndim == 0:
        d_array = np.full(p, float(d_array))
    if d_array.shape != (p,):
        raise ValueError("d must be a scalar or a 1D array of length kappa.shape[1].")

    # Drop zero-weight columns. They contribute nothing to
    # ``kappa @ diag(d) @ kappa^T``, so removing them leaves the matrix
    # unchanged, but carried through they put a zero on the equilibrated
    # Woodbury matrix's diagonal (``sign(0) == 0``), which the rank-p prep
    # rejects outright. In the interior-point loop ``d = lambda / (-f)``, and a
    # multiplier on a deeply slack constraint routinely underflows to exactly
    # zero near the boundary -- a reachable state, not a caller error.
    nonzero = d_array != 0.0
    if not np.all(nonzero):
        kappa = kappa[:, nonzero]
        d_array = d_array[nonzero]

    return kappa, d_array


class MatrixSystem(ABC):
    """Structured symmetric positive-definite matrix that can solve and multiply itself.

    Each subclass stores its matrix H in a structured form and implements ``solve``,
    ``multiply`` and ``scaled`` in that form. The base supplies ``plus`` (a structured
    sum, through each shape's ``_absorb``), ``low_rank_update``, ``negated``, and a
    dense ``is_positive_definite`` fallback.

    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """

    @abstractmethod
    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows; a matrix is
            solved column by column.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Implementations raise when H is not numerically positive definite.

        """

    @abstractmethod
    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the product H y.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows; a matrix is multiplied column
            by column.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """

    @abstractmethod
    def scaled(self, t: float) -> MatrixSystem:
        """Return t H, the matrix scaled by a nonnegative factor.

        Scaling preserves structure. It weights a Hessian contribution, for example the
        objective's curvature by the barrier parameter, when the Newton system is
        assembled.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            A system of the same structure representing t H.

        """

    def plus(self, other: MatrixSystem) -> MatrixSystem:
        """Return H + other, the structured sum of two systems.

        Assembling a Hessian from its contributions is repeated ``plus``. A
        ``ZeroSystem`` is the additive identity; a low-rank-updated operand re-nests its
        update on top of the summed bases; two pure bases combine through ``_absorb``,
        tried in both orders so the sum does not depend on operand order. Two bases of
        the same shape compose into that shape (diagonal plus diagonal is diagonal).
        This is closure within a shape, not a general cross-shape combine.

        Parameters
        ----------
         other : MatrixSystem
            The system to add, of the same order.

        Returns
        -------
         system : MatrixSystem
            The sum, in the shared structure.

        Raises
        ------
         IncompatibleStructureError
            When the orders differ, or the two bases cannot be combined without
            densifying.

        """
        if self.dimension != other.dimension:
            raise IncompatibleStructureError(
                f"Cannot combine systems of dimension {self.dimension} and "
                f"{other.dimension}."
            )
        # The zero matrix is the additive identity. (A ZeroSystem carrying a
        # low-rank update is a LowRankUpdatedSystem, handled by the branch below;
        # its zero base then drops out here when the wrapper recurses.)
        if isinstance(other, ZeroSystem):
            return self
        if isinstance(self, ZeroSystem):
            return other
        if isinstance(other, LowRankUpdatedSystem):
            # Addition commutes; let the low-rank operand re-nest its update on
            # top of the combined base.
            return other.plus(self)
        # ``_absorb`` lives on the more-structured operand -- a banded system
        # absorbs a diagonal as a width-0 band, but a diagonal does not know
        # what a band is. Addition is commutative, so if ``self`` cannot absorb
        # ``other`` we let ``other`` absorb ``self``; ``plus`` then does not
        # depend on operand order. ``_absorb`` is side-effect-free, so retrying
        # the other direction is safe.
        try:
            return self._absorb(other)
        except IncompatibleStructureError:
            return other._absorb(self)

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Combine two pure (not low-rank-updated) bases of equal order into one.

        The default rejects every combination. Each concrete shape overrides it to
        absorb its own kind and any strictly simpler shape. ``plus`` calls it only for
        pure bases of equal order.

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb.

        Returns
        -------
         system : MatrixSystem
            The sum; implementations return it, the default never does.

        Raises
        ------
         IncompatibleStructureError
            Always, in the default.

        """
        raise IncompatibleStructureError(
            f"Cannot combine {type(self).__name__} with {type(other).__name__}."
        )

    def low_rank_update(
        self,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64] | float | None = None,
        coalesce: bool = True,
    ) -> MatrixSystem:
        """Return the system H + kappa diag(d) kappa^T.

        Defined once here and inherited by every MatrixSystem, so a low-rank update can
        sit on top of any structure. The result solves through the Woodbury identity.

        Parameters
        ----------
         kappa : npt.NDArray[np.float64]
            A vector of length n (rank one) or a matrix of shape (n, p) (rank p).
         d : npt.NDArray[np.float64] | float | None
            A scalar or a vector of length p of weights: positive entries are updates,
            negative entries downdates; ``None`` means unit weights. Zero-weight columns
            are dropped.
         coalesce : bool
            Ignored here, and accepted for a signature uniform with
            ``LowRankUpdatedSystem.low_rank_update``: a first update always nests a
            fresh layer, since there is nothing to merge into.

        Returns
        -------
         system : MatrixSystem
            A ``LowRankUpdatedSystem`` wrapping this system, or this system itself when
            every weight is zero.

        Raises
        ------
         ValueError
            Propagated from ``_normalize_low_rank`` when ``kappa`` or ``d`` is
            malformed.

        """
        kappa, d = _normalize_low_rank(kappa, d, self.dimension)
        if kappa.shape[1] == 0:
            # Every weight was zero, so the update adds nothing: return the base.
            return self

        return LowRankUpdatedSystem(self, kappa, d)

    def is_positive_definite(self) -> bool:
        """Check whether the matrix is numerically positive definite.

        A ``MatrixSystem`` is positive definite by contract, but curvature assembled
        from caller-supplied signs can violate that. This check lets a consumer, such as
        the stationarity solvers' check on each diagonal block, fail cleanly on
        wrong-sign input without densifying the block by hand.

        The default forms the dense matrix by multiplying the identity (n products) and
        attempts a Cholesky factorization, O(n^3). Structured subclasses override it
        with a cheaper test that reuses the factorization they cache.

        Returns
        -------
         is_pd : bool
            True when the Cholesky factorization succeeds.

        """
        dense = self.multiply(np.eye(self.dimension))
        try:
            np.linalg.cholesky(dense)
        except np.linalg.LinAlgError:
            return False
        return True

    @property
    def definiteness(self) -> Definiteness:
        """Return the declared definiteness of the matrix, positive definite by default.

        This is the contract the system advertises, distinct from the numerical check
        ``is_positive_definite``. Every concrete base in this module is positive
        definite; a ``_NegatedSystem`` flips the declaration, so a concave objective's
        Hessian can be carried as negative definite until ``NegatedObjective`` negates
        it back.

        Returns
        -------
         definiteness : Definiteness
            ``Definiteness.POSITIVE_DEFINITE`` in the default.

        """
        return Definiteness.POSITIVE_DEFINITE

    def negated(self) -> MatrixSystem:
        """Return -H as a sign-carrying view, flipping the declared definiteness.

        ``scaled`` takes only a nonnegative factor, so negation is the one sign change
        it does not cover. It exists to maximize a concave objective g by minimizing -g:
        the objective advertises its negative-definite Hessian as ``H.negated()``, and
        ``NegatedObjective`` negates it once more, recovering the positive-definite H
        the solver factors.

        The default wraps this system in a lazy ``_NegatedSystem``, whose own
        ``negated`` collapses the double negation back to this system, so a maximize
        round trip never asks a factorization to handle a negative-definite matrix.

        Returns
        -------
         system : MatrixSystem
            A system representing -H.

        """
        return _NegatedSystem(self)


class _NegatedSystem(MatrixSystem):
    """Lazy negation -H of a wrapped ``MatrixSystem``.

    Built by ``MatrixSystem.negated`` to represent -H without factoring a matrix that is
    not positive definite. ``solve`` and ``multiply`` push the sign through the wrapped
    system,
       (-H)^{-1} b = -(H^{-1} b),   (-H) y = -(H y),
    so the wrapped system does all the arithmetic on its own positive-definite
    structure.

    A second negation collapses: ``negated`` returns the wrapped system rather than
    nesting another wrapper. That makes maximizing a concave objective free: the
    objective advertises ``H.negated()`` (negative definite), ``NegatedObjective``
    negates it again, and the solver receives the original H.

    A ``_NegatedSystem`` is a transient sign carrier, not an assembly base: it is
    expected to be collapsed by a second negation before a Hessian is assembled, and
    ``plus`` with any nonzero system raises ``IncompatibleStructureError``. Its
    ``solve`` does not raise, but its inherited ``is_positive_definite`` is a genuine
    numerical test on -H, so a consumer that checks definiteness rejects a surviving
    negative-definite block.

    Parameters
    ----------
     inner : MatrixSystem
        The system H whose negation this represents.

    """

    def __init__(self, inner: MatrixSystem) -> None:
        self._inner = inner

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._inner.dimension

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve (-H) x = b as x = -(H^{-1} b).

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Propagated from the wrapped system's ``solve``.

        """
        return -self._inner.solve(b)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute (-H) y = -(H y).

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        return -self._inner.multiply(y)

    def scaled(self, t: float) -> MatrixSystem:
        """Return t (-H) = -(t H) for a nonnegative factor t.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            The negation of the wrapped system scaled by t.

        """
        return _NegatedSystem(self._inner.scaled(t))

    def negated(self) -> MatrixSystem:
        """Collapse the double negation, -(-H) = H.

        Returns
        -------
         system : MatrixSystem
            The wrapped system.

        """
        return self._inner

    @property
    def definiteness(self) -> Definiteness:
        """Return the declared definiteness of -H, the wrapped system's flipped.

        Returns
        -------
         definiteness : Definiteness
            The wrapped system's definiteness, negated.

        """
        return self._inner.definiteness.negated


class ZeroSystem(MatrixSystem):
    """Zero matrix of a given order, the additive identity for ``plus``.

    A component with no base curvature contributes a ``ZeroSystem``: a linear objective,
    or an inequality layer whose barrier curvature is purely a low-rank update (the
    update sits on the zero base through ``low_rank_update``). It multiplies to zero and
    drops out of ``plus``. The zero matrix is singular, so ``solve`` raises; a
    ``ZeroSystem`` reaches a solve only after being summed with a positive-definite
    term.

    Parameters
    ----------
     dimension : int
        Order of the matrix.

    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._dimension

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Refuse to solve, since the zero matrix is singular.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side; unused.

        Returns
        -------
         x : npt.NDArray[np.float64]
            Never returned.

        Raises
        ------
         NewtonStepError
            Always.

        """
        raise NewtonStepError("The zero matrix is not invertible.")

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute 0 y, the zero array shaped like y.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         zeros : npt.NDArray[np.float64]
            Zeros of the same shape as ``y``.

        """
        return np.zeros_like(y)

    def scaled(self, t: float) -> MatrixSystem:
        """Return t 0 = 0, this same system.

        Parameters
        ----------
         t : float
            Scale factor; unused.

        Returns
        -------
         system : MatrixSystem
            This system.

        """
        return self

    def negated(self) -> MatrixSystem:
        """Return -0 = 0, this same system.

        Overrides the lazy default so that the negated additive identity stays the
        additive identity and keeps dropping out of ``plus`` (a ``_NegatedSystem``
        wrapping zero would not). A negated linear objective's zero curvature relies on
        this during assembly.

        Returns
        -------
         system : MatrixSystem
            This system.

        """
        return self

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Return 0 + other = other (``plus`` also short-circuits this case).

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb.

        Returns
        -------
         system : MatrixSystem
            ``other``, unchanged.

        """
        return other

    def is_positive_definite(self) -> bool:
        """Check positive definiteness, true only for the empty (order 0) matrix.

        Returns
        -------
         is_pd : bool
            True when the order is 0.

        """
        return self._dimension == 0


class DiagonalSystem(MatrixSystem):
    """Diagonal matrix H = diag(eta).

    Positive definiteness is not checked at construction: a diagonal with zero or
    negative entries may be built and composed through ``plus``, and only ``solve``
    enforces it. That lets a component declare a zero-curvature coordinate, such as a
    slack variable with no objective curvature of its own, that a later barrier term
    makes positive in the assembled sum.

    Parameters
    ----------
     eta : npt.NDArray[np.float64]
        The diagonal entries, a 1-D array of length n, or the constructor raises
        ``ValueError``.
     eta_inverse : npt.NDArray[np.float64] | None
        The reciprocals 1 / eta, for a caller that can compute them more stably than
        dividing. For example, a barrier term with curvature 1 / (x - lb)^2 already
        holds the slack s = x - lb and can pass s^2, avoiding two round trips through
        tiny quantities. It is trusted, not checked against 1 / eta, which would defeat
        its purpose; it must have the shape of ``eta``, or the constructor raises
        ``ValueError``. ``None`` computes 1 / eta (inf for a zero entry).

    """

    def __init__(
        self,
        eta: npt.NDArray[np.float64],
        eta_inverse: npt.NDArray[np.float64] | None = None,
    ) -> None:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim != 1:
            raise ValueError("eta must be a 1D array.")
        if eta_inverse is None:
            # A zero-eta entry yields an inf reciprocal; that is fine -- it
            # cancels when this diagonal is absorbed into a PD sum (its 0/dominant
            # ratio drops out), and a standalone solve against it is rejected by
            # solve(). Suppress the divide warning rather than guard elementwise.
            with np.errstate(divide="ignore"):
                eta_inverse_arr = 1.0 / eta
        else:
            eta_inverse_arr = np.asarray(eta_inverse, dtype=np.float64)
            if eta_inverse_arr.shape != eta.shape:
                raise ValueError("eta_inverse must have the same shape as eta.")
        self._eta = eta
        self._eta_inverse = eta_inverse_arr

    @property
    def eta(self) -> npt.NDArray[np.float64]:
        """Return the diagonal entries of the matrix.

        Returns
        -------
         eta : npt.NDArray[np.float64]
            The diagonal, of length n.

        """
        return self._eta

    @property
    def eta_inverse(self) -> npt.NDArray[np.float64]:
        """Return the reciprocal diagonal, carrying any stable value supplied at build.

        Exposed so a caller that reorders a diagonal (for example, under a coordinate
        permutation) can carry the stable reciprocal forward rather than recompute it
        from ``eta`` and lose precision near a barrier boundary.

        Returns
        -------
         eta_inverse : npt.NDArray[np.float64]
            The reciprocals of the diagonal entries, of length n.

        """
        return self._eta_inverse

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._eta.shape[0]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve diag(eta) x = b.

        Positive definiteness is enforced here, not at construction. The check is on
        ``eta`` rather than ``eta_inverse``, because a zero entry's reciprocal is inf,
        which the kernel's own check (``eta_inverse`` > 0) would let through.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution ``eta_inverse`` ∘ b, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            When some entry of ``eta`` is not positive, or, propagated from
            ``solve_diagonal_eta_inverse``, some caller-supplied reciprocal is not
            positive.
         ValueError
            Propagated from ``solve_diagonal_eta_inverse`` when ``b`` does not have n
            rows.

        """
        if not np.all(self._eta > 0):
            raise NewtonStepError("Hessian is not strictly positive definite.")
        return solve_diagonal_eta_inverse(b, self._eta_inverse)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute diag(eta) y.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        return multiply_diagonal(y, self._eta)

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H, scaling the diagonal and carrying the stable reciprocal.

        Parameters
        ----------
         t : float
            Positive scale factor; t = 0 makes the reciprocals infinite.

        Returns
        -------
         system : MatrixSystem
            A ``DiagonalSystem`` with diagonal t ``eta``, its reciprocals divided by t.

        """
        return DiagonalSystem(t * self._eta, eta_inverse=self._eta_inverse / t)

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Sum with another diagonal, keeping a stable reciprocal for the sum.

        With D = max(eta_a, eta_b) componentwise, the reciprocal of the sum is computed
        as
           1 / (eta_a + eta_b) = (1 / D) / (eta_a / D + eta_b / D),
        where 1 / D is the stored, possibly exact, reciprocal of the dominant entry.
        Factoring out the dominant term keeps a tiny barrier curvature from being lost
        under a large one. Selecting the dominant entry's reciprocal, rather than the
        smaller reciprocal, is correct for any signs: with eta_a < 0 < eta_b and a
        positive sum, the smaller reciprocal is the negative one. It also drops a zero
        entry's inf reciprocal.

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb; must be a ``DiagonalSystem``.

        Returns
        -------
         system : MatrixSystem
            The ``DiagonalSystem`` with diagonal eta_a + eta_b.

        Raises
        ------
         IncompatibleStructureError
            When ``other`` is not a ``DiagonalSystem``.

        """
        if not isinstance(other, DiagonalSystem):
            raise IncompatibleStructureError(
                f"Cannot combine DiagonalSystem with {type(other).__name__}; a "
                "DiagonalSystem absorbs only another diagonal."
            )
        eta = self._eta + other._eta
        # Stable 1/(eta_a + eta_b): factor out the dominant term so a tiny
        # barrier curvature survives the sum instead of being lost under a large
        # one. ``1/E`` is the stored (possibly exact) reciprocal of whichever eta
        # is dominant. Pick it by the dominant eta itself rather than via
        # ``np.minimum`` of the reciprocals: with a documented negative-curvature
        # entry (eta_a < 0 < eta_b, eta_a + eta_b > 0) the smaller reciprocal is
        # the *negative* one, which would yield a negative eta_inverse on a sum
        # that is in fact PD. Selecting the dominant eta's reciprocal is correct
        # for any signs (and still drops a zero entry's inf reciprocal).
        dominant = np.maximum(self._eta, other._eta)
        recip_dominant = np.where(
            self._eta >= other._eta, self._eta_inverse, other._eta_inverse
        )
        eta_inverse = recip_dominant / (self._eta / dominant + other._eta / dominant)
        return DiagonalSystem(eta, eta_inverse=eta_inverse)

    def is_positive_definite(self) -> bool:
        """Check positive definiteness: every diagonal entry strictly positive.

        Returns
        -------
         is_pd : bool
            True when every entry of ``eta`` is positive.

        """
        return bool(np.all(self._eta > 0.0))


class BandedSystem(MatrixSystem):
    """Symmetric banded matrix in SciPy's compact banded storage.

    The Cholesky factor is computed on the first ``solve`` (or ``is_positive_definite``)
    and cached; the matrix is immutable, since every transform returns a new instance,
    so the factor cannot go stale and the several solves of one Newton step share it.

    Parameters
    ----------
     ab : npt.NDArray[np.float64]
        The band storage, of shape (p + 1, n) for bandwidth p, in the convention of
        ``scipy.linalg.cholesky_banded`` (see ``numerical_helpers.solve_banded``). Must
        be 2-D, or the constructor raises ``ValueError``.
     lower : bool
        True when ``ab`` stores the lower triangle, False for the upper.

    """

    def __init__(self, ab: npt.NDArray[np.float64], lower: bool = False) -> None:
        ab = np.asarray(ab, dtype=np.float64)
        if ab.ndim != 2:
            raise ValueError("ab must be a 2D array in compact banded storage.")
        self._ab = ab
        self._lower = lower
        # Lazily-cached banded Cholesky factor. The matrix is immutable (every
        # transform returns a new instance), so the factor cannot go stale; it
        # is computed on the first solve and reused across the remaining solves
        # of the same instance -- the three solves of one KKT/Newton step.
        self._factor: npt.NDArray[np.float64] | None = None

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._ab.shape[1]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b with the cached banded Cholesky factor.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Propagated from ``factor_banded`` when H is not positive definite or has
            non-finite entries.

        """
        if self._factor is None:
            self._factor = factor_banded(self._ab, lower=self._lower)
        return solve_banded_factored(b, self._factor, lower=self._lower)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute H y, exploiting the band structure.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        return multiply_banded(y, self._ab, lower=self._lower)

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H, preserving the band structure.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            A ``BandedSystem`` storing t ``ab``.

        """
        return BandedSystem(t * self._ab, lower=self._lower)

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Sum with another banded matrix, or with a diagonal (a band of width zero).

        Two banded matrices combine into a banded matrix of the wider bandwidth, and a
        diagonal folds into the main diagonal. The two banded operands must use the same
        triangle convention (``lower``).

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb: a ``BandedSystem`` or a ``DiagonalSystem``.

        Returns
        -------
         system : MatrixSystem
            The banded sum.

        Raises
        ------
         IncompatibleStructureError
            When ``other`` is banded with a different ``lower``, or is neither banded
            nor diagonal.

        """
        if isinstance(other, BandedSystem):
            if other._lower != self._lower:
                raise IncompatibleStructureError(
                    "Cannot combine banded systems stored with different "
                    "triangles; both must use the same ``lower`` convention."
                )
            p_self = self._ab.shape[0] - 1
            p_other = other._ab.shape[0] - 1
            p = max(p_self, p_other)
            ab = np.zeros((p + 1, self._ab.shape[1]))
            if self._lower:
                # Row d holds the d-th subdiagonal; the main diagonal is row 0,
                # so the bands align from the top.
                ab[: p_self + 1] += self._ab
                ab[: p_other + 1] += other._ab
            else:
                # Row p - d holds the d-th superdiagonal; the main diagonal is
                # the last row, so the bands align from the bottom.
                ab[p - p_self :] += self._ab
                ab[p - p_other :] += other._ab
            return BandedSystem(ab, lower=self._lower)
        if isinstance(other, DiagonalSystem):
            ab = self._ab.copy()
            main_diagonal = 0 if self._lower else -1
            ab[main_diagonal] = ab[main_diagonal] + other.eta
            return BandedSystem(ab, lower=self._lower)
        raise IncompatibleStructureError(
            f"Cannot combine BandedSystem with {type(other).__name__}; a banded "
            "matrix absorbs only another banded matrix or a diagonal."
        )

    def is_positive_definite(self) -> bool:
        """Check positive definiteness by attempting the banded Cholesky factorization.

        Reuses and populates the cached factor, so a later solve pays nothing.

        Returns
        -------
         is_pd : bool
            True when the factorization succeeds.

        """
        try:
            if self._factor is None:
                self._factor = factor_banded(self._ab, lower=self._lower)
        except NewtonStepError:
            return False
        return True


class DenseSystem(MatrixSystem):
    """Dense symmetric positive-definite matrix.

    A fallback for blocks with no exploitable structure. The Cholesky factor is computed
    on the first ``solve`` (or ``is_positive_definite``) and cached for the remaining
    solves of the same instance, at O(n^3) for the factorization; prefer a structured
    MatrixSystem where one applies.

    Parameters
    ----------
     matrix : npt.NDArray[np.float64]
        The square symmetric matrix, of shape (n, n). The constructor raises
        ``ValueError`` when it is not square, and ``NewtonStepError`` when it has
        non-finite entries.

    """

    def __init__(self, matrix: npt.NDArray[np.float64]) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")
        if not np.all(np.isfinite(matrix)):
            raise NewtonStepError("Hessian contains non-finite entries.")
        self._matrix = matrix
        # Lazily-cached Cholesky factor, reused across the solves of one Newton
        # step. Safe because the matrix is immutable (see BandedSystem).
        self._factor: tuple[npt.NDArray[np.float64], bool] | None = None

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._matrix.shape[0]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b with the cached dense Cholesky factor.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            When the Cholesky factorization fails, so H is not positive definite.

        """
        if self._factor is None:
            try:
                self._factor = linalg.cho_factor(self._matrix, lower=True)
            except np.linalg.LinAlgError:
                raise NewtonStepError(
                    "Hessian is not strictly positive definite."
                ) from None
        result: npt.NDArray[np.float64] = linalg.cho_solve(self._factor, b)
        return result

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute H y without factoring.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        return self._matrix @ y

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            A ``DenseSystem`` storing t H.

        """
        return DenseSystem(t * self._matrix)

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Sum with another dense matrix, or fold a diagonal into the diagonal.

        A dense matrix absorbs any summand that reduces to dense arithmetic: another
        ``DenseSystem`` (added entrywise) or a ``DiagonalSystem`` (added to the main
        diagonal). Other shapes compose only through their own ``_absorb``.

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb: a ``DenseSystem`` or a ``DiagonalSystem``.

        Returns
        -------
         system : MatrixSystem
            The dense sum.

        Raises
        ------
         IncompatibleStructureError
            When ``other`` is neither dense nor diagonal.

        """
        if isinstance(other, DenseSystem):
            return DenseSystem(self._matrix + other._matrix)
        if isinstance(other, DiagonalSystem):
            matrix = self._matrix.copy()
            idx = np.diag_indices_from(matrix)
            matrix[idx] = matrix[idx] + other.eta
            return DenseSystem(matrix)
        raise IncompatibleStructureError(
            f"Cannot combine DenseSystem with {type(other).__name__}; a dense "
            "matrix absorbs only another dense matrix or a diagonal."
        )

    def is_positive_definite(self) -> bool:
        """Check positive definiteness by attempting the dense Cholesky factorization.

        Reuses and populates the cached factor, so a later solve pays nothing.

        Returns
        -------
         is_pd : bool
            True when the factorization succeeds (or a factor is already cached).

        """
        if self._factor is not None:
            return True
        try:
            self._factor = linalg.cho_factor(self._matrix, lower=True)
        except np.linalg.LinAlgError:
            return False
        return True


class LowRankUpdatedSystem(MatrixSystem):
    """Low-rank-updated system H = base + kappa diag(d) kappa^T.

    Solves through the Woodbury identity on top of the base system's own ``solve``: a
    one-time preparation solves the base against the p columns of kappa and factors the
    p × p capacitance matrix, after which each right-hand side costs one base solve plus
    O(n p). A multiply is the base multiply plus O(n p). The preparation is cached, so
    the solves of one Newton step share one capacitance factorization (and, through the
    base's own cache, one base factorization).

    A further ``low_rank_update`` coalesces by default: the new term is merged into this
    one as a single wider update over the same base, rather than a second nested
    Woodbury layer. ``plus`` and ``coalesce=False`` nest instead, to keep updates of
    opposite sign in separate layers.

    Parameters
    ----------
     base : MatrixSystem
        The system being updated.
     kappa : npt.NDArray[np.float64]
        The update factor, already a matrix of shape (n, p).
     d : npt.NDArray[np.float64] | None
        The weights, a vector of length p, or ``None`` for unit weights.
        ``MatrixSystem.low_rank_update`` normalizes callers' arguments into this form.

    """

    def __init__(
        self,
        base: MatrixSystem,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64] | None,
    ) -> None:
        self._base = base
        self._kappa = kappa
        self._d = d
        # Lazily-cached Woodbury data (xi and the factored capacitance matrix),
        # reused across the solves of one Newton step. Safe because the matrix
        # is immutable (see BandedSystem).
        self._prep: _RankOneUpdatePrep | _RankPUpdatePrep | None = None

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._base.dimension

    def _ensure_prep(self) -> _RankOneUpdatePrep | _RankPUpdatePrep:
        """Compute and cache the Woodbury preparation, or return the cached copy.

        Shared by ``solve`` and ``is_positive_definite``, so the rank-one versus rank-p
        choice and the scalar-weight conversion live in one place. A single column uses
        the rank-one (Sherman-Morrison) preparation.

        Returns
        -------
         prep : _RankOneUpdatePrep | _RankPUpdatePrep
            The right-hand-side-independent Woodbury data.

        Raises
        ------
         NewtonStepError
            Propagated from the base solve when the base is not positive definite, or
            from the preparation when the capacitance factorization fails (a downdate
            that breaks definiteness, or an inertia mismatch).

        """
        if self._prep is None:
            if self._kappa.shape[1] == 1:
                d = None if self._d is None else float(self._d[0])
                self._prep = _prepare_rank_one_update(
                    self._kappa[:, 0], self._base.solve, d=d
                )
            else:
                self._prep = _prepare_rank_p_update(
                    self._kappa, self._base.solve, d=self._d
                )
        return self._prep

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b by a Woodbury correction to the base solve.

        The right-hand-side-independent Woodbury data (``xi`` and the factored
        capacitance matrix) is computed on the first solve and cached. A single update
        (p = 1) uses the rank-one Sherman-Morrison path, which avoids forming and
        factoring a 1 × 1 capacitance matrix and its equilibration.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Propagated from ``_ensure_prep`` or the base solve.

        """
        prep = self._ensure_prep()
        if isinstance(prep, _RankOneUpdatePrep):
            return _apply_rank_one_update(b, prep, self._base.solve)
        return _apply_rank_p_update(b, prep, self._base.solve)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute H y as the base product plus the low-rank correction.

        A single update (p = 1) uses the rank-one path.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        if self._kappa.shape[1] == 1:
            d = None if self._d is None else float(self._d[0])
            return multiply_rank_one_update(
                y, self._kappa[:, 0], self._base.multiply, d=d
            )
        return multiply_rank_p_update(y, self._kappa, self._base.multiply, d=self._d)

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H, scaling the base and folding t into the update.

        Scaling by zero annihilates the update, so the scaled base is the whole answer.
        With unit weights, sqrt(t) is folded into kappa so the unweighted fast path
        survives. Otherwise d is scaled by t through ``low_rank_update``, because t d
        can underflow to zeros that were not in d (a small t against an already tiny
        weight λ / (-f)), and those must be dropped too. That call passes
        ``coalesce=False``: when the base is itself low-rank updated, the layering is
        deliberate (it keeps updates of opposite sign in separate capacitance matrices,
        since merging them is catastrophic near a barrier boundary), and scaling an
        assembled Hessian must not flatten it.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            A system representing t H.

        """
        if t == 0.0:
            return self._base.scaled(t)

        if self._d is None:
            return LowRankUpdatedSystem(
                self._base.scaled(t), math.sqrt(t) * self._kappa, None
            )

        return self._base.scaled(t).low_rank_update(
            self._kappa, t * self._d, coalesce=False
        )

    def plus(self, other: MatrixSystem) -> MatrixSystem:
        """Sum the base with another system, then re-nest this update on top.

        Nesting (a fresh Woodbury layer) rather than coalescing keeps each update in its
        own layer, which sign correctness requires when updates of mixed inertia stack:
        a single mixed-sign Woodbury solve is unstable near a barrier boundary.

        Parameters
        ----------
         other : MatrixSystem
            The system to add, of the same order.

        Returns
        -------
         system : MatrixSystem
            A ``LowRankUpdatedSystem`` carrying this update over base + other.

        Raises
        ------
         IncompatibleStructureError
            Propagated from the base's ``plus``.

        """
        return LowRankUpdatedSystem(self._base.plus(other), self._kappa, self._d)

    def low_rank_update(
        self,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64] | float | None = None,
        coalesce: bool = True,
    ) -> MatrixSystem:
        """Return this system with a further low-rank update.

        With ``coalesce=True`` (the default) the new update is merged with this one into
        a single wider update over the same base, so a chain of L updates is one
        Woodbury solve of width ∑ p rather than L nested solves.

        With ``coalesce=False`` the new update nests instead, as a fresh Woodbury layer
        over this whole system. This is the escape hatch for mixed-inertia stacks:
        folding updates and downdates into one capacitance matrix is numerically
        catastrophic near a barrier boundary, so a downdate is applied as a separate
        layer (as in the inner and outer terms of a second-order cone barrier's
        Hessian).

        Parameters
        ----------
         kappa : npt.NDArray[np.float64]
            A vector of length n (rank one) or a matrix of shape (n, q) (rank q).
         d : npt.NDArray[np.float64] | float | None
            A scalar or a vector of length q of weights, or ``None`` for unit weights.
            Zero-weight columns are dropped.
         coalesce : bool
            Merge into this update (True) or nest a new layer (False).

        Returns
        -------
         system : MatrixSystem
            The updated system, or this system itself when every new weight is zero.

        Raises
        ------
         ValueError
            Propagated from ``_normalize_low_rank`` when ``kappa`` or ``d`` is
            malformed.

        """
        kappa, d = _normalize_low_rank(kappa, d, self.dimension)
        if kappa.shape[1] == 0:
            # Every new weight was zero, so this update adds nothing: keep the
            # system as it stands, with whatever update it already carries.
            return self

        if not coalesce:
            return LowRankUpdatedSystem(self, kappa, d)
        combined_kappa = np.hstack([self._kappa, kappa])
        if self._d is None and d is None:
            combined_d: npt.NDArray[np.float64] | None = None
        else:
            d_self = self._d if self._d is not None else np.ones(self._kappa.shape[1])
            d_new = d if d is not None else np.ones(kappa.shape[1])
            combined_d = np.concatenate([d_self, d_new])
        return LowRankUpdatedSystem(self._base, combined_kappa, combined_d)

    def is_positive_definite(self) -> bool:
        """Check whether the assembled matrix H is positive definite.

        Building the cached Woodbury preparation runs the base solve, which raises
        ``NewtonStepError`` when the base is not positive definite, and the guarded
        capacitance factorization, which raises on a downdate that breaks definiteness
        or on an inertia mismatch. A successful preparation therefore certifies H
        positive definite without densifying the base, and a later solve reuses it.

        A ``NewtonStepError`` does not certify that H is indefinite, only that the
        Woodbury path does not apply: a singular (positive-semidefinite) base whose
        update lifts the deficiency gives a positive-definite H, yet its preparation
        raises on the base. So on that error this falls back to the dense default
        (``multiply`` and a Cholesky factorization of H), which settles the sum's true
        definiteness.

        Returns
        -------
         is_pd : bool
            True when H is positive definite.

        """
        try:
            self._ensure_prep()
        except NewtonStepError:
            return super().is_positive_definite()
        return True


class BlockDiagonalSystem(MatrixSystem):
    """Block-diagonal matrix whose diagonal blocks are themselves MatrixSystems.

    Solving and multiplying dispatch block by block, so the blocks may exploit entirely
    different structures.

    Parameters
    ----------
     blocks : Sequence[MatrixSystem]
        The diagonal blocks, in order. Must be non-empty, or the constructor raises
        ``ValueError``.

    """

    def __init__(self, blocks: Sequence[MatrixSystem]) -> None:
        if len(blocks) == 0:
            raise ValueError("blocks must be non-empty.")
        self._blocks = list(blocks)

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return sum(block.dimension for block in self._blocks)

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b one diagonal block at a time.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length n or a matrix with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Propagated from a block's ``solve``.

        """
        return solve_block_diagonal(
            b,
            [block.dimension for block in self._blocks],
            [block.solve for block in self._blocks],
        )

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute H y one diagonal block at a time.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        return multiply_block_diagonal(
            y,
            [block.dimension for block in self._blocks],
            [block.multiply for block in self._blocks],
        )

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H, scaling each diagonal block.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            A ``BlockDiagonalSystem`` of the scaled blocks.

        """
        return BlockDiagonalSystem([block.scaled(t) for block in self._blocks])

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Sum with a block-diagonal system on the same partition, or with a diagonal.

        Two block-diagonal matrices over the same partition combine block by block. A
        diagonal fits every partition: it splits into one diagonal segment per block,
        carrying its stable reciprocal, and each block absorbs its segment through
        ``plus``. Any other shape raises, since a band crossing a block boundary would
        break the partition.

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb: a ``BlockDiagonalSystem`` or a ``DiagonalSystem``.

        Returns
        -------
         system : MatrixSystem
            The block-diagonal sum.

        Raises
        ------
         IncompatibleStructureError
            When ``other`` is block diagonal on a different partition, or neither block
            diagonal nor diagonal; or, propagated from a block's ``plus``, when a block
            cannot absorb its part.

        """
        if isinstance(other, BlockDiagonalSystem):
            blocks = self._blocks
            other_blocks = other._blocks
            if len(blocks) != len(other_blocks) or any(
                a.dimension != b.dimension
                for a, b in zip(blocks, other_blocks, strict=False)
            ):
                raise IncompatibleStructureError(
                    "Cannot combine block-diagonal systems with different partitions."
                )
            return BlockDiagonalSystem(
                [a.plus(b) for a, b in zip(blocks, other_blocks, strict=False)]
            )
        if isinstance(other, DiagonalSystem):
            eta = other.eta
            eta_inverse = other._eta_inverse
            new_blocks: list[MatrixSystem] = []
            start = 0
            for block in self._blocks:
                stop = start + block.dimension
                segment = DiagonalSystem(
                    eta[start:stop], eta_inverse=eta_inverse[start:stop]
                )
                new_blocks.append(block.plus(segment))
                start = stop
            return BlockDiagonalSystem(new_blocks)
        raise IncompatibleStructureError(
            f"Cannot combine BlockDiagonalSystem with {type(other).__name__}; a "
            "block-diagonal matrix absorbs only a matching block-diagonal or a "
            "diagonal."
        )

    def is_positive_definite(self) -> bool:
        """Check positive definiteness: every diagonal block positive definite.

        Returns
        -------
         is_pd : bool
            True when every block is positive definite.

        """
        return all(block.is_positive_definite() for block in self._blocks)


class PermutedSystem(MatrixSystem):
    """Matrix whose exploitable structure appears only in a reordered basis.

    Structure is a property of a matrix together with a variable ordering. A Hessian can
    be block diagonal in one ordering and merely banded, or dense-looking, in another,
    and the decision vector's layout is fixed by the problem rather than chosen for the
    linear algebra. A typical case is a decision matrix whose row-major flattening
    interleaves its columns: a column-separable Hessian is block diagonal only once each
    column's strided indices are gathered (grouped).

    ``inner`` is the system in the grouped basis, and the wrapper presents it in the
    caller's interleaved basis, representing
       H = P^T H_grouped P,
    where P gathers coordinates (P v = v[``to_grouped``]), so that ``inner`` equals
    H[np.ix_(``to_grouped``, ``to_grouped``)]. Because it presents itself in the
    decision-variable basis, it is an ordinary ``MatrixSystem``: a ``Problem`` sums it
    with the constraint layers' contributions correctly, and no permutation-aware solver
    is needed. ``solve`` and ``multiply`` accept a matrix of right-hand sides, which the
    Woodbury path needs, so a low-rank update composes on top of a permuted base.

    Parameters
    ----------
     inner : MatrixSystem
        The system in the grouped (structured) basis.
     to_grouped : npt.NDArray[np.intp]
        A permutation of range(n), n the order of ``inner``, gathering interleaved
        coordinates into the grouped basis. The constructor raises ``ValueError`` when
        it is not such a permutation.
     to_interleaved : npt.NDArray[np.intp] | None
        The inverse permutation, or ``None`` to derive it. The constructor raises
        ``ValueError`` when a supplied one is not the inverse of ``to_grouped``.

    """

    def __init__(
        self,
        inner: MatrixSystem,
        to_grouped: npt.NDArray[np.intp],
        to_interleaved: npt.NDArray[np.intp] | None = None,
    ) -> None:
        to_grouped = np.asarray(to_grouped, dtype=np.intp)
        n = inner.dimension
        if to_grouped.shape != (n,) or not np.array_equal(
            np.sort(to_grouped), np.arange(n)
        ):
            raise ValueError(
                f"to_grouped must be a permutation of range({n}); got shape "
                f"{to_grouped.shape}."
            )
        if to_interleaved is None:
            to_interleaved = np.argsort(to_grouped).astype(np.intp)
        else:
            to_interleaved = np.asarray(to_interleaved, dtype=np.intp)
            if not np.array_equal(to_grouped[to_interleaved], np.arange(n)):
                raise ValueError(
                    "to_interleaved must be the inverse permutation of to_grouped."
                )
        self._inner = inner
        self._to_grouped = to_grouped
        self._to_interleaved = to_interleaved

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._inner.dimension

    def _gather(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reorder the rows of a vector or matrix into the grouped basis.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows, in the interleaved basis.

        Returns
        -------
         grouped : npt.NDArray[np.float64]
            ``v`` with its rows taken in ``to_grouped`` order.

        """
        return v[self._to_grouped] if v.ndim == 1 else v[self._to_grouped, :]

    def _scatter(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reorder the rows of a vector or matrix back into the interleaved basis.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows, in the grouped basis.

        Returns
        -------
         interleaved : npt.NDArray[np.float64]
            ``v`` with its rows taken in ``to_interleaved`` order.

        """
        return v[self._to_interleaved] if v.ndim == 1 else v[self._to_interleaved, :]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b: gather, solve in the grouped basis, scatter back.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side in the interleaved basis, a vector of length n or a matrix
            with n rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution in the interleaved basis, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Propagated from the inner system's ``solve``.

        """
        return self._scatter(self._inner.solve(self._gather(np.asarray(b))))

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute H y: gather, multiply in the grouped basis, scatter back.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length n or a matrix with n rows, in the interleaved basis.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product in the interleaved basis, of the same shape as ``y``.

        """
        return self._scatter(self._inner.multiply(self._gather(np.asarray(y))))

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H; scaling commutes with the permutation.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            A ``PermutedSystem`` over the scaled inner system.

        """
        return PermutedSystem(
            self._inner.scaled(t), self._to_grouped, self._to_interleaved
        )

    @property
    def definiteness(self) -> Definiteness:
        """Return the inner system's declared definiteness; permuting preserves it.

        Returns
        -------
         definiteness : Definiteness
            The inner system's definiteness.

        """
        return self._inner.definiteness

    def is_positive_definite(self) -> bool:
        """Check positive definiteness of the inner system, which reordering preserves.

        Returns
        -------
         is_pd : bool
            The inner system's answer.

        """
        return self._inner.is_positive_definite()

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Fold in an interleaved diagonal, or a sibling under the same permutation.

        The diagonal case is essential, not a convenience: an interior-point solver adds
        a barrier's ``DiagonalSystem`` to the objective's Hessian on every Newton step,
        so a structured Hessian that cannot absorb a diagonal is unusable. A diagonal
        stays diagonal under permutation, so its entries (and stable reciprocals) are
        gathered into the grouped basis and absorbed there.

        A sibling under the same permutation is P^T B P beside this P^T A P, and the two
        sum to P^T (A + B) P. That lets a nonlinear constraint whose barrier curvature
        is a second structured Hessian over the same layout compose with the
        objective's. A different permutation is a different block layout and raises.

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb: a ``DiagonalSystem`` or a ``PermutedSystem`` with the
            same ``to_grouped``.

        Returns
        -------
         system : MatrixSystem
            A ``PermutedSystem`` over the summed inner system.

        Raises
        ------
         IncompatibleStructureError
            When ``other`` is neither, or, propagated from the inner system's ``plus``,
            when the inner systems cannot be summed.

        """
        if isinstance(other, DiagonalSystem):
            permuted = DiagonalSystem(
                other.eta[self._to_grouped],
                eta_inverse=other.eta_inverse[self._to_grouped],
            )
            return PermutedSystem(
                self._inner.plus(permuted), self._to_grouped, self._to_interleaved
            )
        if isinstance(other, PermutedSystem) and np.array_equal(
            self._to_grouped, other._to_grouped
        ):
            return PermutedSystem(
                self._inner.plus(other._inner),
                self._to_grouped,
                self._to_interleaved,
            )
        raise IncompatibleStructureError(
            f"Cannot combine PermutedSystem with {type(other).__name__}; only a "
            "diagonal or a same-permutation PermutedSystem folds into the "
            "permuted block structure."
        )


class ArrowSystem(MatrixSystem):
    """Bordered (arrow) matrix: a structured block with a small dense fringe.

    The matrix is
             _            _
       H =  |  A11    A12  |
            |_ A12^T  A22 _|
    where the upper-left block A11 is any ``MatrixSystem`` of order M, and the border
    A12 (M × p) and corner A22 (p × p) are a small dense fringe of p rows and columns.
    Solving eliminates A11 through its Schur complement. When A11 is a
    ``DiagonalSystem`` and p = 1, the matrix has a literal arrow sparsity pattern, and a
    dedicated, numerically stable kernel is used.

    The whole matrix is assumed positive definite; the indefinite saddle-point case (a
    zero corner) is a ``SaddleSystem``.

    Parameters
    ----------
     upper_left : MatrixSystem
        The block A11, of order M.
     border : npt.NDArray[np.float64]
        The block A12: a vector of length M (for p = 1) or a matrix of shape (M, p). The
        constructor raises ``ValueError`` when it does not have M rows.
     corner : npt.NDArray[np.float64] | float
        The block A22: a scalar (for p = 1) or a matrix of shape (p, p). The constructor
        raises ``ValueError`` when its shape does not match the border.

    """

    def __init__(
        self,
        upper_left: MatrixSystem,
        border: npt.NDArray[np.float64],
        corner: npt.NDArray[np.float64] | float,
    ) -> None:
        M = upper_left.dimension
        border = np.asarray(border, dtype=np.float64)
        if border.ndim == 1:
            border = border[:, np.newaxis]
        if border.ndim != 2 or border.shape[0] != M:
            raise ValueError(
                f"border must have {M} rows to match the upper-left MatrixSystem."
            )
        p = border.shape[1]

        corner_arr = np.asarray(corner, dtype=np.float64)
        if corner_arr.ndim == 0:
            corner_arr = corner_arr.reshape(1, 1)
        if corner_arr.shape != (p, p):
            raise ValueError(
                f"corner must be ({p}, {p}) to match the border's {p} column(s)."
            )

        self._upper_left = upper_left
        self._border = border
        self._corner = corner_arr
        self._p = p

    @property
    def dimension(self) -> int:
        """Return the order of the matrix.

        Returns
        -------
         n : int
            The number of rows (and columns) of the square matrix.

        """
        return self._upper_left.dimension + self._p

    @property
    def upper_left(self) -> MatrixSystem:
        """Return the upper-left block A11.

        Returns
        -------
         upper_left : MatrixSystem
            The block A11, of order M.

        """
        return self._upper_left

    @property
    def border(self) -> npt.NDArray[np.float64]:
        """Return the border block A12.

        Returns
        -------
         border : npt.NDArray[np.float64]
            The block A12, of shape (M, p).

        """
        return self._border

    @property
    def corner(self) -> npt.NDArray[np.float64]:
        """Return the corner block A22.

        Returns
        -------
         corner : npt.NDArray[np.float64]
            The block A22, of shape (p, p).

        """
        return self._corner

    def diagonal_plus_low_rank(
        self,
    ) -> tuple[DiagonalSystem, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Express the arrow as a diagonal plus a rank-two term, when possible.

        When A11 is a ``DiagonalSystem`` and the corner is a single positive finite
        scalar c, the arrow is
           H = D_full + K diag(1, -1) K^T,
        where D_full is the diagonal of A11 with c appended, and the rank-two term
        encodes the border u. With u_ext = [u; 0] of length M + 1 and e the last unit
        vector,
            _        _
           |  0    u  |  =  u_ext e^T + e u_ext^T  =  K diag(1, -1) K^T,
           |_ u^T  0 _|
           K = [u_ext + e, u_ext - e] / sqrt(2).
        Returns ``None`` instead when A11 is not diagonal, p > 1, or the corner is not
        positive and finite.

        Returns
        -------
         D_full : DiagonalSystem
            The diagonal of A11 with c appended, of order M + 1, carrying the stable
            reciprocals.
         K : npt.NDArray[np.float64]
            The rank-two factor, of shape (M + 1, 2).
         d_lr : npt.NDArray[np.float64]
            The weights [1, -1].

        """
        if not isinstance(self._upper_left, DiagonalSystem) or self._p != 1:
            return None
        core = self._upper_left
        c = float(self._corner[0, 0])
        if not (np.isfinite(c) and c > 0):
            return None
        u = self._border[:, 0]
        M = core.dimension
        d_full = np.concatenate([core.eta, [c]])
        with np.errstate(divide="ignore"):
            c_inv = np.float64(1.0) / np.float64(c)
        d_full_inverse = np.concatenate([core.eta_inverse, [c_inv]])
        D_full = DiagonalSystem(d_full, eta_inverse=d_full_inverse)
        u_ext = np.concatenate([u, [0.0]])
        e = np.zeros(M + 1)
        e[M] = 1.0
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        K = np.column_stack([(u_ext + e) * inv_sqrt2, (u_ext - e) * inv_sqrt2])
        d_lr = np.array([1.0, -1.0])
        return D_full, K, d_lr

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H x = b through the Schur complement on the upper-left block.

        Uses the arrow-sparsity kernel for a diagonal A11 with p = 1, the block-plus-one
        kernel for another A11 with p = 1, and the general Schur-complement kernel
        otherwise.

        Parameters
        ----------
         b : npt.NDArray[np.float64]
            Right-hand side, a vector of length M + p or a matrix with M + p rows.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The solution, of the same shape as ``b``.

        Raises
        ------
         NewtonStepError
            Propagated from the kernels or the upper-left solve when H is not positive
            definite.
         ValueError
            Propagated from the kernels when ``b`` has the wrong number of rows.

        """
        if self._p == 1:
            border = self._border[:, 0]
            corner = float(self._corner[0, 0])
            if isinstance(self._upper_left, DiagonalSystem):
                return solve_arrow_sparsity_pattern(
                    b, self._upper_left.eta, border, corner
                )
            return solve_block_plus_one(b, border, corner, self._upper_left.solve)
        return solve_with_schur(b, self._border, self._corner, self._upper_left.solve)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute H y block by block.

        Parameters
        ----------
         y : npt.NDArray[np.float64]
            A vector of length M + p or a matrix with M + p rows.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product, of the same shape as ``y``.

        """
        if self._p == 1:
            return multiply_block_plus_one(
                y,
                self._border[:, 0],
                float(self._corner[0, 0]),
                self._upper_left.multiply,
            )
        return multiply_block(y, self._border, self._corner, self._upper_left.multiply)

    def scaled(self, t: float) -> MatrixSystem:
        """Return t H, scaling the upper-left block, border and corner.

        Parameters
        ----------
         t : float
            Nonnegative scale factor.

        Returns
        -------
         system : MatrixSystem
            An ``ArrowSystem`` of the scaled blocks.

        """
        return ArrowSystem(
            self._upper_left.scaled(t), t * self._border, t * self._corner
        )

    def _absorb(self, other: MatrixSystem) -> MatrixSystem:
        """Sum with a system that has no curvature crossing the border.

        Another ``ArrowSystem`` on the same partition adds block by block. A
        ``DiagonalSystem`` has no cross-block coupling: its first M entries fold into
        A11 and its last p into the corner. A ``BlockDiagonalSystem`` whose block
        boundaries align with the split folds the same way, its interior blocks into A11
        and its corner blocks (formed densely) into the corner. Any shape whose
        curvature crosses the border, such as a dense matrix or a block straddling the
        split, raises, since the sum would lose the arrow pattern.

        Parameters
        ----------
         other : MatrixSystem
            The base to absorb: an ``ArrowSystem``, ``DiagonalSystem`` or
            ``BlockDiagonalSystem``.

        Returns
        -------
         system : MatrixSystem
            The ``ArrowSystem`` sum.

        Raises
        ------
         IncompatibleStructureError
            When ``other`` has a different partition, a block straddles the split, or
            its shape is not absorbable; or, propagated from A11's ``plus``, when the
            interior parts cannot be summed.

        """
        M = self._upper_left.dimension
        p = self._p
        if isinstance(other, ArrowSystem):
            if other._upper_left.dimension != M or other._p != p:
                raise IncompatibleStructureError(
                    "Cannot combine arrow systems with different partitions; the "
                    "upper-left dimension and border width must match."
                )
            return ArrowSystem(
                self._upper_left.plus(other._upper_left),
                self._border + other._border,
                self._corner + other._corner,
            )
        if isinstance(other, DiagonalSystem):
            eta = other.eta
            eta_inverse = other._eta_inverse
            interior = self._upper_left.plus(
                DiagonalSystem(eta[:M], eta_inverse=eta_inverse[:M])
            )
            return ArrowSystem(interior, self._border, self._corner + np.diag(eta[M:]))
        if isinstance(other, BlockDiagonalSystem):
            interior_blocks: list[MatrixSystem] = []
            corner_blocks: list[MatrixSystem] = []
            offset = 0
            for block in other._blocks:
                (interior_blocks if offset < M else corner_blocks).append(block)
                offset += block.dimension
            if sum(block.dimension for block in interior_blocks) != M:
                raise IncompatibleStructureError(
                    "Cannot combine ArrowSystem with a BlockDiagonalSystem whose "
                    "block boundary does not align with the interior/corner split."
                )
            interior_other = (
                interior_blocks[0]
                if len(interior_blocks) == 1
                else BlockDiagonalSystem(interior_blocks)
            )
            interior = self._upper_left.plus(interior_other)
            corner_other = np.zeros((p, p))
            start = 0
            for block in corner_blocks:
                stop = start + block.dimension
                corner_other[start:stop, start:stop] = np.column_stack(
                    [block.multiply(e) for e in np.eye(block.dimension)]
                )
                start = stop
            return ArrowSystem(interior, self._border, self._corner + corner_other)
        raise IncompatibleStructureError(
            f"Cannot combine ArrowSystem with {type(other).__name__}; an arrow "
            "absorbs only a matching arrow, a diagonal, or a block-diagonal aligned "
            "with its interior/corner split."
        )


class SaddleSystem(ABC):
    """Indefinite saddle-point system of an equality-constrained Newton step.

    The matrix is
        _        _
       |  H  A^T  |
       |_ A   0  _|
    with the Hessian H a ``MatrixSystem`` of order M and A the p × M equality-constraint
    matrix. A Newton step from an equality-feasible iterate has a zero lower right-hand
    side; an infeasible-start (primal-dual) step carries the equality residual there.

    Not a ``MatrixSystem``: the matrix is indefinite, so the positive-definite contract
    (``multiply``, ``low_rank_update``, ``plus``) does not apply. It holds a
    ``MatrixSystem`` for the H block, so that block keeps all its structure.

    This is the abstract base. ``SchurComplementSaddleSystem`` eliminates H through
    dense Schur-complement algebra; ``SparseSaddleSystem`` and its subclasses handle a
    sparse A. The Newton, barrier and primal-dual solvers all solve their KKT systems
    through this interface.

    """

    @property
    @abstractmethod
    def hessian(self) -> MatrixSystem:
        """Return the Hessian block H.

        Returns
        -------
         hessian : MatrixSystem
            The H block, of order M.

        """

    @property
    @abstractmethod
    def num_equality_constraints(self) -> int:
        """Return the number p of equality constraints, the row count of A.

        Returns
        -------
         p : int
            The number of equality constraints.

        """

    @abstractmethod
    def solve(
        self,
        b_top: npt.NDArray[np.float64],
        b_bottom: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the saddle-point system.

        Solves
            _        _     _         _     _          _
           |  H  A^T  |   |  delta_x  |   |  b_top     |
           |_ A   0  _|   |_   nu    _| = |_ b_bottom _|
        for the primal step and the equality multipliers.

        Parameters
        ----------
         b_top : npt.NDArray[np.float64]
            Top block of the right-hand side (the negative gradient or dual residual): a
            vector of length M, or a matrix of shape (M, k) solved column by column.
         b_bottom : npt.NDArray[np.float64] | None
            Bottom block (the equality residual), of length p. ``None`` means zero, the
            equality-feasible Newton step; a nonzero value arises in primal-dual
            methods, whose iterates need not satisfy A x = b mid-iteration.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The primal step, the top block of the solution.
         nu : npt.NDArray[np.float64]
            The equality multipliers, the bottom block of the solution.

        Raises
        ------
         NewtonStepError
            Implementations raise when H, or the reduced system, cannot be solved.

        """


class SchurComplementSaddleSystem(SaddleSystem):
    """Saddle-point system solved through the Schur complement on H.

    ``solve`` eliminates H with ``solve_kkt_system``, touching the Hessian only through
    ``H.solve``: p + 1 solves with H, O(p^2 M) to form the p × p Schur complement, and
    O(p^3) to factor it.

    Parameters
    ----------
     hessian : MatrixSystem
        The Hessian block H, of order M.
     A : npt.NDArray[np.float64]
        The equality-constraint matrix, of shape (p, M). The constructor raises
        ``ValueError`` when it is not 2-D or its column count is not M.

    """

    def __init__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64],
    ) -> None:
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A must be a 2D array.")
        if A.shape[1] != hessian.dimension:
            raise ValueError(
                f"A has {A.shape[1]} columns but the Hessian MatrixSystem has "
                f"dimension {hessian.dimension}."
            )
        self._hessian = hessian
        self._A = A

    @property
    def hessian(self) -> MatrixSystem:
        """Return the Hessian block H.

        Returns
        -------
         hessian : MatrixSystem
            The H block, of order M.

        """
        return self._hessian

    @property
    def num_equality_constraints(self) -> int:
        """Return the number p of equality constraints, the row count of A.

        Returns
        -------
         p : int
            The number of equality constraints.

        """
        return self._A.shape[0]

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Return the equality-constraint matrix A.

        Returns
        -------
         A : npt.NDArray[np.float64]
            The matrix A, of shape (p, M).

        """
        return self._A

    def solve(
        self,
        b_top: npt.NDArray[np.float64],
        b_bottom: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the saddle-point system through the Schur complement on H.

        Solves
            _        _     _         _     _          _
           |  H  A^T  |   |  delta_x  |   |  b_top     |
           |_ A   0  _|   |_   nu    _| = |_ b_bottom _|
        for the primal step and the equality multipliers.

        Parameters
        ----------
         b_top : npt.NDArray[np.float64]
            Top block of the right-hand side (the negative gradient or dual residual): a
            vector of length M, or a matrix of shape (M, k) solved column by column.
         b_bottom : npt.NDArray[np.float64] | None
            Bottom block (the equality residual), of length p. ``None`` means zero, the
            equality-feasible Newton step; a nonzero value arises in primal-dual
            methods, whose iterates need not satisfy A x = b mid-iteration.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The primal step, the top block of the solution.
         nu : npt.NDArray[np.float64]
            The equality multipliers, the bottom block of the solution.

        Raises
        ------
         NewtonStepError
            Propagated from ``solve_kkt_system``: H is not positive definite or is
            singular, or the reduced system has no solution.
         ValueError
            Propagated from ``solve_kkt_system`` when ``b_top`` or ``b_bottom`` has the
            wrong length.

        """
        return solve_kkt_system(self._A, b_top, self._hessian.solve, g_bot=b_bottom)


class CoupledSaddleSystem:
    """Two-block saddle-point system of a Newton step for a convex-concave problem.

    Solves the indefinite system
        _              _     _         _     _     _
       |  -M_xx    B    |   |  delta_x  |   |  b_x  |
       |_  B^T    H_yy _|   |_ delta_y _| = |_ b_y _|
    that arises in a Newton step for a saddle-point problem concave in x and convex in
    y. The minus sign on the (1, 1) block is intrinsic: the x-block is negative definite
    and the y-block positive definite, so the whole matrix has n negative and m positive
    eigenvalues. The class stores the negated concave curvature M_xx, which is positive
    definite (for an objective concave in x, -∇^2_xx is positive definite, and an
    interior-point barrier only adds positive-definite curvature), the dense coupling B,
    and the convex-block curvature H_yy, also positive definite.

    Not a ``SaddleSystem``: that class models the bordered equality KKT matrix, with a
    zero (2, 2) block and a ``solve(b_top, b_bottom)`` signature. This system has a
    positive-definite (2, 2) block and its own ``solve(b_x, b_y)``; like a
    ``SaddleSystem``, it holds ``MatrixSystem`` blocks. The stationarity solvers use it.

    Parameters
    ----------
     xx_system : MatrixSystem
        The positive-definite block M_xx, of order n, the negated concave curvature of
        the x-block. Large and typically structured, and touched only through its
        ``solve``.
     coupling : npt.NDArray[np.float64]
        The dense mixed-Hessian coupling B, of shape (n, m), typically with m much
        smaller than n. The constructor raises ``ValueError`` when it is not 2-D or its
        shape does not match the two blocks.
     yy_system : MatrixSystem
        The positive-definite convex-block curvature H_yy, of order m (small).

    Notes
    -----
    Eliminating delta_x gives a positive-definite Schur complement on the small y-block.
    The first block row gives
       M_xx delta_x = B delta_y - b_x,
    and substituting into the second, B^T delta_x + H_yy delta_y = b_y, gives
       S delta_y = b_y + B^T M_xx^{-1} b_x,   S = H_yy + B^T M_xx^{-1} B.
    S is positive definite (H_yy plus a positive-semidefinite term), so it is solved by
    Cholesky with an SVD fallback (``_solve_schur_complement``) (Boyd and Vandenberghe,
    2004, appendix C).

    The m + 1 right-hand sides of the elimination (the m columns of B and the vector
    b_x) are batched into one solve with M_xx on [B, b_x], so M_xx is factored once.
    With c the cost of one solve with M_xx, the total work is O(m c + m^2 n + m^3): the
    batched solve, forming B^T M_xx^{-1} B, and the dense m × m factorization.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    def __init__(
        self,
        xx_system: MatrixSystem,
        coupling: npt.NDArray[np.float64],
        yy_system: MatrixSystem,
    ) -> None:
        coupling = np.asarray(coupling, dtype=np.float64)
        if coupling.ndim != 2:
            raise ValueError("coupling must be a 2D array.")
        n, m = coupling.shape
        if n != xx_system.dimension:
            raise ValueError(
                f"coupling has {n} rows but xx_system has dimension "
                f"{xx_system.dimension}."
            )
        if m != yy_system.dimension:
            raise ValueError(
                f"coupling has {m} columns but yy_system has dimension "
                f"{yy_system.dimension}."
            )
        self._xx_system = xx_system
        self._coupling = coupling
        self._yy_system = yy_system

    @property
    def xx_system(self) -> MatrixSystem:
        """Return the block M_xx, the negated concave curvature of the x-block.

        Returns
        -------
         xx_system : MatrixSystem
            The positive-definite block M_xx, of order n.

        """
        return self._xx_system

    @property
    def coupling(self) -> npt.NDArray[np.float64]:
        """Return the coupling matrix B.

        Returns
        -------
         B : npt.NDArray[np.float64]
            The dense coupling, of shape (n, m).

        """
        return self._coupling

    @property
    def yy_system(self) -> MatrixSystem:
        """Return the convex-block curvature H_yy.

        Returns
        -------
         yy_system : MatrixSystem
            The positive-definite block H_yy, of order m.

        """
        return self._yy_system

    @property
    def dimension_x(self) -> int:
        """Return the dimension n of the x-block.

        Returns
        -------
         n : int
            The order of M_xx.

        """
        return self._xx_system.dimension

    @property
    def dimension_y(self) -> int:
        """Return the dimension m of the y-block.

        Returns
        -------
         m : int
            The order of H_yy.

        """
        return self._yy_system.dimension

    def solve(
        self,
        b_x: npt.NDArray[np.float64],
        b_y: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the coupled saddle through the positive-definite complement on y.

        Eliminates delta_x onto S = H_yy + B^T M_xx^{-1} B (see the class Notes), then
        back-substitutes
           delta_x = M_xx^{-1} (B delta_y - b_x),
        reusing the batched solve rather than solving with M_xx again.

        Parameters
        ----------
         b_x : npt.NDArray[np.float64]
            Top block of the right-hand side, of length n.
         b_y : npt.NDArray[np.float64]
            Bottom block of the right-hand side, of length m.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The x-block step, of length n.
         delta_y : npt.NDArray[np.float64]
            The y-block step, of length m.

        Raises
        ------
         ValueError
            When ``b_x`` does not have length n or ``b_y`` does not have length m.
         NewtonStepError
            Propagated from the solve with M_xx (not positive definite, singular, or
            non-finite) or from ``_solve_schur_complement`` (S non-finite, or no
            solution).

        """
        B = self._coupling
        n, m = B.shape
        b_x = np.asarray(b_x, dtype=np.float64)
        b_y = np.asarray(b_y, dtype=np.float64)
        if b_x.shape != (n,):
            raise ValueError(
                "Dimension mismatch: b_x should have one entry for each row of "
                "the coupling matrix."
            )
        if b_y.shape != (m,):
            raise ValueError(
                "Dimension mismatch: b_y should have one entry for each column of "
                "the coupling matrix."
            )

        # Batch the m + 1 right-hand sides into ONE M_xx solve so the
        # factorization of M_xx is formed once, not m + 1 times. Solving against
        # [B | b_x] yields [M_xx^{-1} B | M_xx^{-1} b_x] in a single call.
        rhs_batch = np.column_stack([B, b_x])  # shape (n, m + 1)
        solved = self._xx_system.solve(rhs_batch)
        Minv_B = solved[:, :m]  # shape (n, m)
        Minv_bx = solved[:, m]  # shape (n,)

        # Form the PD complement S = H_yy + B^T M_xx^{-1} B. H_yy is
        # small (m-by-m); materialize it densely by multiplying against the
        # identity, the MatrixSystem-agnostic way to read out a small block.
        H_yy_dense = self._yy_system.multiply(np.eye(m))  # shape (m, m)
        S = H_yy_dense + B.T @ Minv_B
        # Symmetrize: B^T M_xx^{-1} B is symmetric in exact arithmetic, but
        # rounding can leave S slightly asymmetric; the Cholesky path in
        # _solve_schur_complement assumes symmetry (matches the C in
        # SparseSaddleSystem._solve_diagonal_plus_low_rank).
        S = 0.5 * (S + S.T)

        # Form rhs = b_y + B^T M_xx^{-1} b_x, then solve S delta_y = rhs. S is
        # PD, which is exactly _solve_schur_complement's contract.
        rhs = b_y + B.T @ Minv_bx
        delta_y = _solve_schur_complement(S, rhs)

        # Back-substitute delta_x = M_xx^{-1} (B delta_y - b_x). The batched
        # solve above already produced M_xx^{-1} B and M_xx^{-1} b_x, so reuse
        # them directly rather than solving against M_xx a second time.
        delta_x = Minv_B @ delta_y - Minv_bx

        return delta_x, delta_y


class EqualityCoupledSaddleSystem:
    """Coupled saddle-point system bordered by an equality on the x-block.

    Solves the symmetric, indefinite Newton system
        _                      _     _              _     _      _
       |  -M_xx    B    A_eq^T  |   |  delta_x       |   |  b_x   |
       |   B^T    H_yy    0     |   |  delta_y       | = |  b_y   |
       |_  A_eq    0      0    _|   |_ delta_lambda _|   |_ b_eq _|
    of a convex-concave saddle whose concave x-block also carries a linear equality A_eq
    x = b_eq, with multiplier λ. It is ``CoupledSaddleSystem`` extended by a p-row
    equality border on the x-block only: the equality adds A_eq^T λ to the x-block's
    stationarity and the primal feasibility row A_eq x = b_eq, and leaves the convex
    y-block untouched. It is the coupled analogue of the border a ``SaddleSystem`` puts
    on a single Hessian, and the stationarity solvers use it when the x-block has a true
    equality (for example, 1^T x = c) rather than only inequality barriers.

    Not a ``CoupledSaddleSystem`` subclass: the equality changes ``solve`` to
    ``solve(b_x, b_y, b_eq)``, returning a third block.

    Parameters
    ----------
     xx_system : MatrixSystem
        The positive-definite block M_xx, of order n, the negated concave curvature of
        the x-block; touched only through its ``solve``.
     coupling : npt.NDArray[np.float64]
        The dense mixed-Hessian coupling B, of shape (n, m), typically with m much
        smaller than n.
     yy_system : MatrixSystem
        The positive-definite convex-block curvature H_yy, of order m (small).
     equality : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        The equality matrix A_eq on the x-block, of shape (p, n), with p small
        (typically 1) and full row rank. A sparse matrix is densified at construction.
     The constructor raises ``ValueError`` when ``coupling`` or ``equality`` is not 2-D,
     their shapes do not match the blocks, or ``equality`` has no rows.

    Notes
    -----
    Eliminating delta_x with the first block row,
       M_xx delta_x = B delta_y + A_eq^T delta_lambda - b_x,
    and substituting into the y and λ rows gives a small, dense, symmetric reduced
    system of order m + p,
        _            _     _              _     _                           _
       |  S_yy   G    |   |  delta_y       |   |  b_y + B^T M_xx^{-1} b_x    |
       |_ G^T   S_ll _|   |_ delta_lambda _| = |_ b_eq + A_eq M_xx^{-1} b_x _|
    with
       S_yy = H_yy + B^T M_xx^{-1} B,
       G    = B^T M_xx^{-1} A_eq^T,
       S_ll = A_eq M_xx^{-1} A_eq^T.
    With J = [B, A_eq^T], of shape (n, m + p), the reduced matrix equals
       diag(H_yy, 0) + J^T M_xx^{-1} J.
    It is positive definite when H_yy is and A_eq has full row rank: if v = (v_y, v_λ)
    makes the quadratic form zero, then v_y = 0 from the first term and A_eq^T v_λ = 0
    from the second, so v_λ = 0. It is therefore solved by the same Cholesky with SVD
    fallback (``_solve_schur_complement``) as ``CoupledSaddleSystem``.

    All m + p + 1 right-hand sides (the columns of B and of A_eq^T, and b_x) are batched
    into one solve with M_xx on [B, A_eq^T, b_x], so M_xx is factored once. Compared
    with the equality-free system, the border adds p columns to that solve and grows the
    dense factorization from order m to m + p (Boyd and Vandenberghe, 2004, appendix C).

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    def __init__(
        self,
        xx_system: MatrixSystem,
        coupling: npt.NDArray[np.float64],
        yy_system: MatrixSystem,
        equality: (
            npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        ),
    ) -> None:
        coupling = np.asarray(coupling, dtype=np.float64)
        if coupling.ndim != 2:
            raise ValueError("coupling must be a 2D array.")
        n, m = coupling.shape
        if n != xx_system.dimension:
            raise ValueError(
                f"coupling has {n} rows but xx_system has dimension "
                f"{xx_system.dimension}."
            )
        if m != yy_system.dimension:
            raise ValueError(
                f"coupling has {m} columns but yy_system has dimension "
                f"{yy_system.dimension}."
            )
        equality_any: Any = equality
        if scipy.sparse.issparse(equality_any):
            equality_any = equality_any.toarray()
        equality_dense = np.asarray(equality_any, dtype=np.float64)
        if equality_dense.ndim != 2:
            raise ValueError("equality must be a 2D array.")
        if equality_dense.shape[1] != n:
            raise ValueError(
                f"equality has {equality_dense.shape[1]} columns but the x-block "
                f"has dimension {n}."
            )
        if equality_dense.shape[0] == 0:
            raise ValueError("equality must have at least one row.")
        self._xx_system = xx_system
        self._coupling = coupling
        self._yy_system = yy_system
        self._equality = equality_dense

    @property
    def xx_system(self) -> MatrixSystem:
        """Return the block M_xx, the negated concave curvature of the x-block.

        Returns
        -------
         xx_system : MatrixSystem
            The positive-definite block M_xx, of order n.

        """
        return self._xx_system

    @property
    def coupling(self) -> npt.NDArray[np.float64]:
        """Return the coupling matrix B.

        Returns
        -------
         B : npt.NDArray[np.float64]
            The dense coupling, of shape (n, m).

        """
        return self._coupling

    @property
    def yy_system(self) -> MatrixSystem:
        """Return the convex-block curvature H_yy.

        Returns
        -------
         yy_system : MatrixSystem
            The positive-definite block H_yy, of order m.

        """
        return self._yy_system

    @property
    def equality(self) -> npt.NDArray[np.float64]:
        """Return the x-block equality matrix A_eq.

        Returns
        -------
         A_eq : npt.NDArray[np.float64]
            The dense equality matrix, of shape (p, n).

        """
        return self._equality

    @property
    def dimension_x(self) -> int:
        """Return the dimension n of the x-block.

        Returns
        -------
         n : int
            The order of M_xx.

        """
        return self._xx_system.dimension

    @property
    def dimension_y(self) -> int:
        """Return the dimension m of the y-block.

        Returns
        -------
         m : int
            The order of H_yy.

        """
        return self._yy_system.dimension

    @property
    def num_equalities(self) -> int:
        """Return the number p of equality rows bordering the x-block.

        Returns
        -------
         p : int
            The row count of A_eq.

        """
        return self._equality.shape[0]

    def solve(
        self,
        b_x: npt.NDArray[np.float64],
        b_y: npt.NDArray[np.float64],
        b_eq: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Solve the bordered coupled saddle through the reduced complement on (y, λ).

        Eliminates delta_x onto the positive-definite reduced system of order m + p (see
        the class Notes), then back-substitutes
           delta_x = M_xx^{-1} (B delta_y + A_eq^T delta_lambda - b_x),
        reusing the batched solve.

        Parameters
        ----------
         b_x : npt.NDArray[np.float64]
            Top (x) block of the right-hand side, of length n.
         b_y : npt.NDArray[np.float64]
            Middle (y) block of the right-hand side, of length m.
         b_eq : npt.NDArray[np.float64]
            Bottom (equality) block of the right-hand side, of length p.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The x-block step, of length n.
         delta_y : npt.NDArray[np.float64]
            The y-block step, of length m.
         delta_lambda : npt.NDArray[np.float64]
            The equality-multiplier step, of length p.

        Raises
        ------
         ValueError
            When ``b_x``, ``b_y`` or ``b_eq`` does not have length n, m or p.
         NewtonStepError
            Propagated from the solve with M_xx (not positive definite, singular, or
            non-finite) or from ``_solve_schur_complement`` (the reduced system is
            non-finite or has no solution).

        """
        B = self._coupling
        A_eq = self._equality
        n, m = B.shape
        p = A_eq.shape[0]
        b_x = np.asarray(b_x, dtype=np.float64)
        b_y = np.asarray(b_y, dtype=np.float64)
        b_eq = np.asarray(b_eq, dtype=np.float64)
        if b_x.shape != (n,):
            raise ValueError(
                "Dimension mismatch: b_x should have one entry for each row of "
                "the coupling matrix."
            )
        if b_y.shape != (m,):
            raise ValueError(
                "Dimension mismatch: b_y should have one entry for each column of "
                "the coupling matrix."
            )
        if b_eq.shape != (p,):
            raise ValueError(
                "Dimension mismatch: b_eq should have one entry for each row of "
                "the equality matrix."
            )

        # Batch the m + p + 1 right-hand sides into ONE M_xx solve on
        # [B | A_eq^T | b_x], so the factorization of M_xx is formed once.
        rhs_batch = np.column_stack([B, A_eq.T, b_x])  # shape (n, m + p + 1)
        solved = self._xx_system.solve(rhs_batch)
        Minv_B = solved[:, :m]  # shape (n, m)
        Minv_At = solved[:, m : m + p]  # shape (n, p)
        Minv_bx = solved[:, m + p]  # shape (n,)

        # Form the PD reduced complement on the (y, lambda) block. H_yy is
        # small; materialize it densely against the identity.
        H_yy_dense = self._yy_system.multiply(np.eye(m))  # shape (m, m)
        S_yy = H_yy_dense + B.T @ Minv_B  # shape (m, m)
        G = B.T @ Minv_At  # shape (m, p): B^T M_xx^{-1} A_eq^T
        S_ll = A_eq @ Minv_At  # shape (p, p): A_eq M_xx^{-1} A_eq^T
        reduced = np.block([[S_yy, G], [G.T, S_ll]])  # shape (m + p, m + p)
        # Symmetrize: the cross term is symmetric in exact arithmetic (A_eq
        # M_xx^{-1} B = G^T), but rounding can leave `reduced` slightly
        # asymmetric; the Cholesky path in _solve_schur_complement assumes
        # symmetry.
        reduced = 0.5 * (reduced + reduced.T)

        rhs = np.concatenate([b_y + B.T @ Minv_bx, b_eq + A_eq @ Minv_bx])
        step = _solve_schur_complement(reduced, rhs)  # length m + p
        delta_y = step[:m]
        delta_lambda = step[m:]

        # Back-substitute delta_x = M_xx^{-1} (B delta_y + A_eq^T delta_lambda
        # - b_x), reusing the batched solves rather than refactoring M_xx.
        delta_x = Minv_B @ delta_y + Minv_At @ delta_lambda - Minv_bx

        return delta_x, delta_y, delta_lambda


class SparseSaddleSystem(SaddleSystem):
    """Saddle-point system for a sparse equality-constraint matrix A.

    Like ``SchurComplementSaddleSystem``, this eliminates H through the Schur complement
       S = A H^{-1} A^T,
    but it keeps A in CSR form and never forms the dense p × M constraint matrix. When H
    is a ``DiagonalSystem`` (the Hessian of a barrier-regularized linear program),
    H^{-1} A^T is A^T with its rows rescaled, so S is assembled entirely with sparse
    products and nothing of size p M is allocated. That makes the system usable when A
    has millions of rows and columns but only a handful of nonzeros per column, as in a
    large bipartite network-flow linear program. When H is an ``ArrowSystem`` with a
    diagonal core and a single border column, a Woodbury path keeps the reduced solve on
    the diagonal-Hessian route.

    The reduced p × p system is solved densely, by Cholesky with an SVD fallback for a
    rank-deficient A, so this class scales in M (the number of variables) but not in p
    (the number of constraints). The subclasses ``BandedSchurSaddleSystem`` and
    ``ArrowSaddleSystem`` solve a banded or arrow-shaped S faster.

    Parameters
    ----------
     hessian : MatrixSystem
        The Hessian block H, of order M.
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        The equality-constraint matrix, of shape (p, M), stored as CSR. A dense array is
        accepted and converted, but the point of this class is a sparse A. The
        constructor raises ``ValueError`` when its column count is not M.

    """

    def __init__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
    ) -> None:
        # Store A in CSR form. csr_matrix accepts both a dense array and any
        # scipy.sparse matrix, and never densifies a sparse input. Typed Any so
        # the type checker need not reason about the dense/sparse union on every ``@``
        # and ``.multiply`` below -- the same convention used by
        # LinearInequalityConstraintLayer.
        A_sparse: Any = scipy.sparse.csr_matrix(A)
        if A_sparse.shape[1] != hessian.dimension:
            raise ValueError(
                f"A has {A_sparse.shape[1]} columns but the Hessian MatrixSystem "
                f"has dimension {hessian.dimension}."
            )
        self._hessian = hessian
        self._A = A_sparse

    @property
    def hessian(self) -> MatrixSystem:
        """Return the Hessian block H.

        Returns
        -------
         hessian : MatrixSystem
            The H block, of order M.

        """
        return self._hessian

    @property
    def num_equality_constraints(self) -> int:
        """Return the number p of equality constraints, the row count of A.

        Returns
        -------
         p : int
            The number of equality constraints.

        """
        return self._A.shape[0]

    @property
    def A(self) -> Any:
        """Return the sparse equality-constraint matrix A.

        Returns
        -------
         A : Any
            The matrix A in ``scipy.sparse`` CSR form, of shape (p, M).

        """
        return self._A

    def solve(
        self,
        b_top: npt.NDArray[np.float64],
        b_bottom: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the saddle-point system through the Schur complement on H.

        Takes the same block-elimination steps as ``solve_kkt_system`` (Boyd and
        Vandenberghe, 2004, appendix C), but forms the reduced system with sparse
        products, so the dense p × M matrix is never built. The Hessian is touched only
        through its own ``solve``.

        Solves
            _        _     _         _     _          _
           |  H  A^T  |   |  delta_x  |   |  b_top     |
           |_ A   0  _|   |_   nu    _| = |_ b_bottom _|
        for the primal step and the equality multipliers.

        Parameters
        ----------
         b_top : npt.NDArray[np.float64]
            Top block of the right-hand side (the negative gradient or dual residual): a
            vector of length M, or a matrix of shape (M, k) solved column by column.
         b_bottom : npt.NDArray[np.float64] | None
            Bottom block (the equality residual), of length p. ``None`` means zero, the
            equality-feasible Newton step; a nonzero value arises in primal-dual
            methods, whose iterates need not satisfy A x = b mid-iteration.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The primal step, the top block of the solution.
         nu : npt.NDArray[np.float64]
            The equality multipliers, the bottom block of the solution.

        Raises
        ------
         ValueError
            When ``b_bottom`` does not have length p.
         NewtonStepError
            Propagated from the Hessian's ``solve`` or from ``_reduced_solve``.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        A = self._A
        if b_bottom is not None and len(b_bottom) != A.shape[0]:
            raise ValueError(
                "Dimension mismatch: b_bottom should have one entry for each row of A."
            )

        # neg_c = A H^{-1} b_top - b_bottom. ``self._hessian.solve`` is the
        # Hessian's own fast solve (the arrow solve for an ArrowSystem Hessian),
        # so neg_c never densifies H^{-1}. Names mirror _solve_schur_complement.
        neg_c = A @ self._hessian.solve(b_top)
        if b_bottom is not None:
            neg_c = neg_c - b_bottom

        nu = self._reduced_solve(neg_c)
        delta_x = self._hessian.solve(b_top - (A.T @ nu))
        return delta_x, nu

    def _diagonal_plus_low_rank(
        self,
    ) -> tuple[DiagonalSystem, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Return H as a diagonal plus a rank-two term, when it has that shape.

        Detects an ``ArrowSystem`` Hessian with a ``DiagonalSystem`` core and a scalar
        corner (see ``ArrowSystem.diagonal_plus_low_rank``), the shape of a Hessian with
        one extra coupled variable such as a slack. A plain ``DiagonalSystem`` has no
        border, so it returns ``None`` and stays on the sparse diagonal path; so does
        any other Hessian.

        Returns
        -------
         D_full : DiagonalSystem
            The diagonal part of H = D_full + K diag(d_lr) K^T.
         K : npt.NDArray[np.float64]
            The rank-two factor, of shape (M, 2).
         d_lr : npt.NDArray[np.float64]
            The weights [1, -1].

        """
        hessian = self._hessian
        if isinstance(hessian, ArrowSystem):
            return hessian.diagonal_plus_low_rank()
        return None

    def _sibling_with_diagonal_hessian(
        self, d_full: DiagonalSystem
    ) -> SparseSaddleSystem:
        """Build a saddle of this class with a diagonal Hessian and the same A.

        The sibling's ``_reduced_solve`` is the fast solver for
           P = A D_full^{-1} A^T:
        dense for ``SparseSaddleSystem``, banded for ``BandedSchurSaddleSystem``, and
        banded core plus border for ``ArrowSaddleSystem`` (which overrides this to carry
        its border size).

        Parameters
        ----------
         d_full : DiagonalSystem
            The diagonal Hessian for the sibling.

        Returns
        -------
         sibling : SparseSaddleSystem
            A saddle of the same class over ``d_full`` and A.

        """
        return type(self)(d_full, self._A)

    def _solve_diagonal_plus_low_rank(
        self,
        d_full: DiagonalSystem,
        K: npt.NDArray[np.float64],
        d_lr: npt.NDArray[np.float64],
        neg_c: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solve S nu = neg_c for a diagonal-plus-low-rank Hessian through Woodbury.

        With
           H = D_full + K diag(d_lr) K^T,
        the reduced system is
           S = A H^{-1} A^T = P - Q C^{-1} Q^T,
        where
           P = A D_full^{-1} A^T,   Q = A D_full^{-1} K   (p × 2),
           C = diag(1 / d_lr) + K^T D_full^{-1} K         (2 × 2).
        Writing C^{-1} = W diag(1 / evals) W^T from the eigendecomposition of C gives
           -Q C^{-1} Q^T = kappa diag(d_S) kappa^T,   kappa = Q W,   d_S = -1 / evals,
        so S is a rank-two update of P, and ``solve_rank_p_update`` reuses the sibling's
        fast solve with P. On a reduced system that is not positive definite, it falls
        back to the dense Schur solve with SVD.

        Parameters
        ----------
         d_full : DiagonalSystem
            The diagonal part D_full of H, of order M.
         K : npt.NDArray[np.float64]
            The low-rank factor, of shape (M, 2).
         d_lr : npt.NDArray[np.float64]
            The low-rank weights, of length 2.
         neg_c : npt.NDArray[np.float64]
            The reduced right-hand side, of length p.

        Returns
        -------
         nu : npt.NDArray[np.float64]
            The equality multipliers, of length p.

        Raises
        ------
         NewtonStepError
            Propagated from the dense fallback (``_schur_complement`` or
            ``_solve_schur_complement``) when it also fails.

        """
        sibling = self._sibling_with_diagonal_hessian(d_full)
        d_full_inv = d_full.eta_inverse
        # Q = A D_full^{-1} K = (A scaled by 1/d_full) @ K; all sparse until the
        # small (p, 2) result.
        Q = np.asarray(
            self._A.multiply(d_full_inv.reshape(1, -1)) @ K, dtype=np.float64
        )
        C = np.diag(1.0 / d_lr) + K.T @ (d_full_inv[:, None] * K)
        C = 0.5 * (C + C.T)
        evals, W = np.linalg.eigh(C)
        kappa = Q @ W
        d_S = -1.0 / evals
        try:
            return solve_rank_p_update(
                neg_c, kappa, sibling._reduced_solve, d=d_S, check_inertia=True
            )
        except NewtonStepError:
            # Non-PD reduced system -> dense SVD-capable fallback on the full S.
            return _solve_schur_complement(self._schur_complement(), neg_c)

    def _reduced_solve(self, neg_c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve the reduced system S nu = neg_c, S = A H^{-1} A^T.

        Uses the Woodbury path when H is an ``ArrowSystem`` over a diagonal core;
        otherwise forms the dense p × p complement and solves it by Cholesky with an SVD
        fallback. Subclasses override this to solve a diagonal-Hessian S faster, and the
        Woodbury path reuses that override through the sibling saddle.

        Parameters
        ----------
         neg_c : npt.NDArray[np.float64]
            The reduced right-hand side A H^{-1} b_top - b_bottom, of length p.

        Returns
        -------
         nu : npt.NDArray[np.float64]
            The equality multipliers, of length p.

        Raises
        ------
         NewtonStepError
            Propagated from ``_schur_complement`` or ``_solve_schur_complement``.

        """
        dplr = self._diagonal_plus_low_rank()
        if dplr is not None:
            d_full, K, d_lr = dplr
            return self._solve_diagonal_plus_low_rank(d_full, K, d_lr, neg_c)
        return _solve_schur_complement(self._schur_complement(), neg_c)

    def _sparse_schur_complement(self) -> Any:
        """Form A H^{-1} A^T as a sparse matrix, when H is diagonal.

        For a diagonal H, H^{-1} A^T is A^T with its rows scaled (using the stable
        reciprocals), so the product stays sparse and nothing of size p M is allocated.
        For a general Hessian, H^{-1} does not preserve sparsity.

        Returns
        -------
         S : Any
            The p × p complement in ``scipy.sparse`` form, or ``None`` when H is not a
            ``DiagonalSystem``.

        Raises
        ------
         NewtonStepError
            When H is diagonal with an entry that is not positive.

        """
        A = self._A
        hessian = self._hessian
        if isinstance(hessian, DiagonalSystem):
            if not np.all(hessian.eta > 0):
                raise NewtonStepError("Hessian is not strictly positive definite.")
            # Scale column j of A by 1 / eta_j (carrying the stable reciprocal),
            # then multiply by A^T -- all sparse.
            scaled = A.multiply(hessian.eta_inverse.reshape(1, -1))
            return scaled @ A.T
        return None

    def _schur_complement(self) -> npt.NDArray[np.float64]:
        """Form the dense p × p Schur complement A H^{-1} A^T.

        Uses the sparse assembly for a diagonal H, densified only at the small p × p
        size. For a general Hessian it applies ``H.solve`` to a densified A^T, which is
        correct for any Hessian but reintroduces an M × p dense intermediate, so it
        suits small or dense problems rather than the large-scale case this class
        targets.

        Returns
        -------
         S : npt.NDArray[np.float64]
            The dense complement, of shape (p, p).

        Raises
        ------
         NewtonStepError
            Propagated from ``_sparse_schur_complement`` or the Hessian's ``solve``.

        """
        sparse_schur = self._sparse_schur_complement()
        if sparse_schur is not None:
            return np.asarray(sparse_schur.toarray(), dtype=np.float64)
        hinv_at = self._hessian.solve(np.asarray(self._A.T.toarray(), dtype=np.float64))
        return np.asarray(self._A @ hinv_at, dtype=np.float64)


class BandedSchurSaddleSystem(SparseSaddleSystem):
    """Sparse saddle-point system whose Schur complement is banded.

    Forms the same complement S = A H^{-1} A^T as ``SparseSaddleSystem``, but solves the
    reduced p × p system by banded Cholesky, O(p b^2) for bandwidth b, rather than dense
    O(p^3) Cholesky. This is the fast path when the equality rows are ordered so that S
    is banded: for example, when each row couples only to rows near it in a suitable
    sort order, the bandwidth stays small and the reduced solve is effectively linear in
    the number of constraints.

    The caller asserts the structure by choosing this system (through
    ``BandedSaddleFactory``). The bandwidth is read off the assembled sparse S, and the
    solve falls back to the dense path with SVD when S is too wide for banded storage to
    pay (2 (b + 1) > p), when H is not diagonal, or when S is not positive definite (a
    rank-deficient A), so it is always correct, if not always fast.

    Parameters
    ----------
     hessian : MatrixSystem
        The Hessian block H, of order M.
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        The equality-constraint matrix, of shape (p, M). The constructor raises
        ``ValueError`` when its column count is not M.

    """

    def _reduced_solve(self, neg_c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve the reduced system by banded Cholesky, or fall back to the dense solve.

        A diagonal-plus-low-rank Hessian goes through the shared Woodbury path, whose
        sibling solve with P is this same banded solve with a diagonal Hessian.
        Otherwise the sparse S is packed into upper banded storage and factored, with
        the dense fallbacks described on the class.

        Parameters
        ----------
         neg_c : npt.NDArray[np.float64]
            The reduced right-hand side, of length p.

        Returns
        -------
         nu : npt.NDArray[np.float64]
            The equality multipliers, of length p.

        Raises
        ------
         NewtonStepError
            Propagated from the dense fallback (``_schur_complement`` or
            ``_solve_schur_complement``), or from ``_sparse_schur_complement`` when a
            diagonal H has an entry that is not positive.

        """
        dplr = self._diagonal_plus_low_rank()
        if dplr is not None:
            d_full, K, d_lr = dplr
            return self._solve_diagonal_plus_low_rank(d_full, K, d_lr, neg_c)

        sparse_schur = self._sparse_schur_complement()
        if sparse_schur is None:
            # Non-diagonal Hessian: no sparse form to band; use the dense solve.
            return _solve_schur_complement(self._schur_complement(), neg_c)

        coo = sparse_schur.tocoo()
        p = sparse_schur.shape[0]
        bandwidth = int(np.max(np.abs(coo.row - coo.col))) if coo.nnz > 0 else 0
        # Banded storage is (bandwidth + 1) x p and the banded factorization is
        # O(p * bandwidth^2); both only pay off when the band is much narrower
        # than p. If the Schur complement is not actually banded -- e.g. the
        # equality rows were not ordered to make it so -- fall back to the dense
        # solve rather than silently allocate a near-dense band (a hidden
        # O(p^2) blow-up) and run a banded Cholesky that costs as much as, and
        # is less robust than, the dense SVD-capable path.
        if 2 * (bandwidth + 1) > p:
            return _solve_schur_complement(
                np.asarray(sparse_schur.toarray(), dtype=np.float64), neg_c
            )
        # Upper compact banded storage: ab[b + i - j, j] = S[i, j] for i <= j.
        ab = np.zeros((bandwidth + 1, p))
        upper = coo.row <= coo.col
        ab[bandwidth + coo.row[upper] - coo.col[upper], coo.col[upper]] = coo.data[
            upper
        ]
        try:
            return solve_banded(neg_c, ab, lower=False)
        except NewtonStepError:
            # Rank-deficient (non-PD) Schur complement -> dense SVD fallback.
            return _solve_schur_complement(
                np.asarray(sparse_schur.toarray(), dtype=np.float64), neg_c
            )


class ArrowSaddleSystem(BandedSchurSaddleSystem):
    """Sparse saddle-point system whose reduced Schur complement is an arrow matrix.

    When the equality rows split into a core, ordered so that the core complement A_core
    H^{-1} A_core^T is banded, plus ``n_border`` dense border rows (aggregate side
    constraints that touch many variables), the reduced complement S = A H^{-1} A^T is
    an arrow matrix: a banded core bordered by ``n_border`` dense rows and columns.

    The reduced system is solved by wrapping S in an ``ArrowSystem`` with a
    ``BandedSystem`` core, so the core is factored once and the border is folded in
    through an ``n_border`` × ``n_border`` Schur complement, instead of a dense O(p^3)
    factorization of all of S. The reduced solve is thus exactly an ``ArrowSystem``
    solve.

    The border rows are the last ``n_border`` rows of A. With ``n_border`` = 0 this is
    the ``BandedSchurSaddleSystem`` solve. It falls back to the dense solve with SVD
    when the structure is absent (a non-diagonal Hessian, a core that is not narrowly
    banded, or a reduced system that is not positive definite), so the solution is the
    same either way.

    Parameters
    ----------
     hessian : MatrixSystem
        The Hessian block H, of order M.
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        The equality-constraint matrix, of shape (p, M), whose last ``n_border`` rows
        form the dense border. The constructor raises ``ValueError`` when its column
        count is not M.
     n_border : int
        Number of trailing rows of A that form the border. Must satisfy 0 ≤ ``n_border``
        < p, or the constructor raises ``ValueError``.

    """

    def __init__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
        n_border: int,
    ) -> None:
        super().__init__(hessian, A)
        n_rows = self._A.shape[0]
        if n_border < 0 or n_border >= n_rows:
            raise ValueError(f"n_border must be in [0, {n_rows}); got {n_border}.")
        self._n_border = int(n_border)

    @property
    def num_border(self) -> int:
        """Return the number of trailing rows of A treated as the dense border.

        Returns
        -------
         n_border : int
            The border size given at construction.

        """
        return self._n_border

    def _sibling_with_diagonal_hessian(
        self, d_full: DiagonalSystem
    ) -> SparseSaddleSystem:
        """Build an ``ArrowSaddleSystem`` sibling with a diagonal Hessian.

        The sibling keeps this system's A and border size.

        Parameters
        ----------
         d_full : DiagonalSystem
            The diagonal Hessian for the sibling.

        Returns
        -------
         sibling : SparseSaddleSystem
            An ``ArrowSaddleSystem`` over ``d_full``, A and ``n_border``.

        """
        return ArrowSaddleSystem(d_full, self._A, self._n_border)

    def _reduced_solve(self, neg_c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve the reduced system as a banded-core arrow, or fall back to dense.

        A diagonal-plus-low-rank Hessian goes through the shared Woodbury path first,
        whose sibling solve with P is this same arrow solve with a diagonal Hessian.
        With no border, this is the banded solve. Otherwise the core of the sparse S is
        packed into upper banded storage and S is solved as an ``ArrowSystem``, falling
        back to the dense solve when the core is too wide (2 (b + 1) > p - ``n_border``)
        or the arrow solve fails.

        Parameters
        ----------
         neg_c : npt.NDArray[np.float64]
            The reduced right-hand side, of length p.

        Returns
        -------
         nu : npt.NDArray[np.float64]
            The equality multipliers, of length p.

        Raises
        ------
         NewtonStepError
            Propagated from the dense fallback (``_schur_complement`` or
            ``_solve_schur_complement``), or from ``_sparse_schur_complement`` when a
            diagonal H has an entry that is not positive.

        """
        dplr = self._diagonal_plus_low_rank()
        if dplr is not None:
            d_full, K, d_lr = dplr
            return self._solve_diagonal_plus_low_rank(d_full, K, d_lr, neg_c)

        if self._n_border == 0:
            return super()._reduced_solve(neg_c)

        sparse_schur = self._sparse_schur_complement()
        if sparse_schur is None:
            # Non-diagonal Hessian: no sparse form to band.
            return _solve_schur_complement(self._schur_complement(), neg_c)

        p = sparse_schur.shape[0]
        m = p - self._n_border
        core = sparse_schur[:m, :m].tocoo()
        bandwidth = int(np.max(np.abs(core.row - core.col))) if core.nnz > 0 else 0
        # If the core is not actually narrow-banded the arrow buys nothing; use
        # the dense SVD-capable solve on the whole S instead of allocating a
        # near-dense band.
        if 2 * (bandwidth + 1) > m:
            return _solve_schur_complement(
                np.asarray(sparse_schur.toarray(), dtype=np.float64), neg_c
            )
        # Upper compact banded storage of the core: ab[b + i - j, j] = S[i, j].
        ab = np.zeros((bandwidth + 1, m))
        upper = core.row <= core.col
        ab[bandwidth + core.row[upper] - core.col[upper], core.col[upper]] = core.data[
            upper
        ]
        border = np.asarray(sparse_schur[:m, m:].toarray(), dtype=np.float64)
        corner = np.asarray(sparse_schur[m:, m:].toarray(), dtype=np.float64)
        try:
            arrow = ArrowSystem(BandedSystem(ab), border, corner)
            return arrow.solve(neg_c)
        except NewtonStepError:
            # Non-positive-definite reduced system -> dense SVD fallback.
            return _solve_schur_complement(
                np.asarray(sparse_schur.toarray(), dtype=np.float64), neg_c
            )


class SaddleFactory(Protocol):
    """Protocol for building a ``SaddleSystem`` from a Hessian block and a matrix A.

    A factory is how a caller chooses which structure of A to exploit (a dense Schur
    complement, a sparse one, a banded one, an arrow) without the equality layer
    branching on it: an ``EqualityConstraintLayer`` holds a factory and delegates to it,
    rather than being subclassed. The sparse saddle classes already have this call
    signature, so a factory can be one of the classes below or a configured object that
    carries extra structural metadata, as ``ArrowSaddleFactory`` carries its border
    size.

    """

    def __call__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
    ) -> SaddleSystem:
        """Build the saddle-point system for a Hessian and a constraint matrix.

        Parameters
        ----------
         hessian : MatrixSystem
            The Hessian block H, of order M.
         A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
            The equality-constraint matrix, of shape (p, M).

        Returns
        -------
         saddle : SaddleSystem
            The saddle-point system for H and A.

        """
        ...


class SchurComplementSaddleFactory:
    """Default saddle factory: a dense Schur-complement saddle-point system."""

    def __call__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
    ) -> SaddleSystem:
        """Build a ``SchurComplementSaddleSystem``, densifying a sparse A.

        Parameters
        ----------
         hessian : MatrixSystem
            The Hessian block H, of order M.
         A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
            The equality-constraint matrix, of shape (p, M).

        Returns
        -------
         saddle : SaddleSystem
            A ``SchurComplementSaddleSystem`` over H and the dense A.

        Raises
        ------
         ValueError
            Propagated from the constructor when A is not 2-D or does not have M
            columns.

        """
        A_any: Any = A
        A_dense = A_any.toarray() if scipy.sparse.issparse(A_any) else np.asarray(A_any)
        return SchurComplementSaddleSystem(hessian, A_dense)


class SparseSaddleFactory:
    """Saddle factory that exploits a sparse constraint matrix A."""

    def __call__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
    ) -> SaddleSystem:
        """Build a ``SparseSaddleSystem``, keeping A sparse.

        Parameters
        ----------
         hessian : MatrixSystem
            The Hessian block H, of order M.
         A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
            The equality-constraint matrix, of shape (p, M).

        Returns
        -------
         saddle : SaddleSystem
            A ``SparseSaddleSystem`` over H and A.

        Raises
        ------
         ValueError
            Propagated from the constructor when A does not have M columns.

        """
        return SparseSaddleSystem(hessian, A)


class BandedSaddleFactory:
    """Saddle factory that exploits a banded Schur complement."""

    def __call__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
    ) -> SaddleSystem:
        """Build a ``BandedSchurSaddleSystem``, with a banded reduced solve.

        Parameters
        ----------
         hessian : MatrixSystem
            The Hessian block H, of order M.
         A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
            The equality-constraint matrix, of shape (p, M).

        Returns
        -------
         saddle : SaddleSystem
            A ``BandedSchurSaddleSystem`` over H and A.

        Raises
        ------
         ValueError
            Propagated from the constructor when A does not have M columns.

        """
        return BandedSchurSaddleSystem(hessian, A)


class ArrowSaddleFactory:
    """Saddle factory for a reduced Schur complement shaped as a banded-core arrow.

    Carries the border size, which the plain ``SaddleFactory`` call signature cannot
    convey. See ``ArrowSaddleSystem``.

    Parameters
    ----------
     n_border : int
        Number of trailing rows of the constraint matrix that form the dense border.

    """

    def __init__(self, n_border: int) -> None:
        self.n_border = int(n_border)

    def __call__(
        self,
        hessian: MatrixSystem,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
    ) -> SaddleSystem:
        """Build an ``ArrowSaddleSystem`` with the stored border size.

        Parameters
        ----------
         hessian : MatrixSystem
            The Hessian block H, of order M.
         A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
            The equality-constraint matrix, of shape (p, M).

        Returns
        -------
         saddle : SaddleSystem
            An ``ArrowSaddleSystem`` over H, A and ``n_border``.

        Raises
        ------
         ValueError
            Propagated from the constructor when A does not have M columns, or
            ``n_border`` is not in [0, p).

        """
        return ArrowSaddleSystem(hessian, A, self.n_border)
