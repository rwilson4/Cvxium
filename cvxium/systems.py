"""Composable structured linear systems.

A :class:`System` represents a structured symmetric matrix that knows how to
``solve`` and ``multiply`` itself efficiently, exploiting special structure
(diagonal, banded, low-rank update, block-diagonal, arrow/Schur, KKT saddle).
Systems compose: a low-rank update can be layered on top of *any* base System,
and the block-diagonal, arrow, and KKT systems take other Systems as their
parts. This module is a thin, object-oriented layer over the kernels in
:mod:`cvxium.numerical_helpers` -- the kernels do the arithmetic; the Systems
make the composition an ergonomic, type-safe tree.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import NewtonStepError
from .numerical_helpers import (
    multiply_banded,
    multiply_diagonal,
    multiply_rank_p_update,
    solve_banded,
    solve_diagonal,
    solve_rank_p_update,
)


def _normalize_low_rank(
    kappa: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64] | float | None,
    dimension: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Normalize a low-rank update's ``kappa`` and ``d``.

    Parameters
    ----------
     kappa : npt.NDArray[np.float64]
        A vector (rank-one) or an ``(M, p)`` matrix (rank-p).
     d : npt.NDArray[np.float64] or float, optional
        A scalar or length-p vector of weights, or None for unit weights.
     dimension : int
        Order of the System being updated; ``kappa`` must have this many rows.

    Returns
    -------
     kappa : npt.NDArray[np.float64]
        ``kappa`` as an ``(M, p)`` matrix.
     d : npt.NDArray[np.float64] or None
        ``d`` as a length-p vector, or None for unit weights.

    """
    kappa = np.asarray(kappa, dtype=np.float64)
    if kappa.ndim == 1:
        kappa = kappa[:, np.newaxis]
    if kappa.ndim != 2:
        raise ValueError("kappa must be a 1D or 2D array.")
    if kappa.shape[0] != dimension:
        raise ValueError(
            f"kappa has {kappa.shape[0]} rows but the System has dimension "
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
    return kappa, d_array


class System(ABC):
    """A structured symmetric matrix that can solve and multiply itself."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Order of the (square) matrix."""

    @abstractmethod
    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``H @ x = b`` for ``x``.

        ``b`` may be a vector or a matrix, in which case the system is solved
        for each column.
        """

    @abstractmethod
    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``H @ y``.

        ``y`` may be a vector or a matrix, in which case each column is
        multiplied.
        """

    def low_rank_update(
        self,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64] | float | None = None,
    ) -> "System":
        """Return the System ``H + kappa @ diag(d) @ kappa^T``.

        This capability is defined once here and inherited by every System, so
        a low-rank update can be layered on top of any structure.

        Parameters
        ----------
         kappa : npt.NDArray[np.float64]
            A vector (rank-one) or an ``(M, p)`` matrix (rank-p).
         d : npt.NDArray[np.float64] or float, optional
            A scalar or length-p vector of nonzero weights; positive entries
            are updates, negative entries downdates. When omitted, every weight
            is one.

        Returns
        -------
         system : System
            A new System representing the updated matrix.

        """
        kappa, d = _normalize_low_rank(kappa, d, self.dimension)
        return LowRankUpdatedSystem(self, kappa, d)


class DiagonalSystem(System):
    """The diagonal matrix ``H = diag(eta)``."""

    def __init__(self, eta: npt.NDArray[np.float64]) -> None:
        """Build a diagonal System from its diagonal entries ``eta``."""
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim != 1:
            raise ValueError("eta must be a 1D array.")
        self._eta = eta

    @property
    def eta(self) -> npt.NDArray[np.float64]:
        """Diagonal entries of the matrix."""
        return self._eta

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._eta.shape[0]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``diag(eta) @ x = b``."""
        return solve_diagonal(b, self._eta)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``diag(eta) @ y``."""
        return multiply_diagonal(y, self._eta)


class BandedSystem(System):
    """A symmetric banded matrix in scipy compact banded storage.

    ``ab`` is the ``(p + 1, n)`` compact band storage used by
    ``scipy.linalg.cholesky_banded`` -- see :func:`cvxium.numerical_helpers.
    solve_banded` for the storage convention.
    """

    def __init__(self, ab: npt.NDArray[np.float64], lower: bool = False) -> None:
        """Build a banded System from compact band storage ``ab``."""
        ab = np.asarray(ab, dtype=np.float64)
        if ab.ndim != 2:
            raise ValueError("ab must be a 2D array in compact banded storage.")
        self._ab = ab
        self._lower = lower

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._ab.shape[1]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve the banded system, exploiting the band structure."""
        return solve_banded(b, self._ab, lower=self._lower)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``H @ y``, exploiting the band structure."""
        return multiply_banded(y, self._ab, lower=self._lower)


class DenseSystem(System):
    """A dense symmetric positive-definite matrix.

    A fallback for blocks with no exploitable structure. The Cholesky
    factorization is recomputed on each :meth:`solve`; prefer a structured
    System where one applies.
    """

    def __init__(self, matrix: npt.NDArray[np.float64]) -> None:
        """Build a dense System from a square symmetric matrix."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")
        self._matrix = matrix

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._matrix.shape[0]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``H @ x = b`` via a dense Cholesky factorization."""
        try:
            factor = linalg.cho_factor(self._matrix, lower=True)
        except np.linalg.LinAlgError:
            raise NewtonStepError(
                "Hessian is not strictly positive definite."
            ) from None
        result: npt.NDArray[np.float64] = linalg.cho_solve(factor, b)
        return result

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``H @ y``."""
        return self._matrix @ y


class LowRankUpdatedSystem(System):
    """The System ``H = base + kappa @ diag(d) @ kappa^T``.

    Solving and multiplying are O(p) corrections to the base System's own
    ``solve`` and ``multiply`` (via the Woodbury identity). Layering a further
    low-rank update *coalesces*: the new term is merged into this one as a
    single wider update over the same base System, rather than nesting a second
    Woodbury solve.
    """

    def __init__(
        self,
        base: System,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64] | None,
    ) -> None:
        """Build a low-rank-updated System.

        ``kappa`` must already be an ``(M, p)`` matrix and ``d`` either None or
        a length-p vector; :meth:`System.low_rank_update` normalizes callers'
        arguments before constructing this class.
        """
        self._base = base
        self._kappa = kappa
        self._d = d

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._base.dimension

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``H @ x = b`` via a Woodbury update to the base solve."""
        return solve_rank_p_update(b, self._kappa, self._base.solve, d=self._d)

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``H @ y`` via a low-rank correction to the base multiply."""
        return multiply_rank_p_update(y, self._kappa, self._base.multiply, d=self._d)

    def low_rank_update(
        self,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64] | float | None = None,
    ) -> "System":
        """Return the System with a further low-rank update, coalesced.

        The new update is merged with this one into a single wider update over
        the same base System, so a chain of L updates is one Woodbury solve of
        width ``sum(p)`` rather than L nested solves.
        """
        kappa, d = _normalize_low_rank(kappa, d, self.dimension)
        combined_kappa = np.hstack([self._kappa, kappa])
        if self._d is None and d is None:
            combined_d: npt.NDArray[np.float64] | None = None
        else:
            d_self = self._d if self._d is not None else np.ones(self._kappa.shape[1])
            d_new = d if d is not None else np.ones(kappa.shape[1])
            combined_d = np.concatenate([d_self, d_new])
        return LowRankUpdatedSystem(self._base, combined_kappa, combined_d)
