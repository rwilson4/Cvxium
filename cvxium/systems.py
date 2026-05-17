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
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import NewtonStepError
from .numerical_helpers import (
    multiply_banded,
    multiply_block,
    multiply_block_diagonal,
    multiply_block_plus_one,
    multiply_diagonal,
    multiply_rank_p_update,
    solve_arrow_sparsity_pattern,
    solve_banded,
    solve_block_diagonal,
    solve_block_plus_one,
    solve_diagonal,
    solve_kkt_system,
    solve_rank_p_update,
    solve_with_schur,
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


class BlockDiagonalSystem(System):
    """A block-diagonal matrix whose diagonal blocks are themselves Systems.

    Solving and multiplying dispatch block by block, so the blocks may exploit
    entirely different structures.
    """

    def __init__(self, blocks: Sequence[System]) -> None:
        """Build a block-diagonal System from its ordered diagonal blocks."""
        if len(blocks) == 0:
            raise ValueError("blocks must be non-empty.")
        self._blocks = list(blocks)

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return sum(block.dimension for block in self._blocks)

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``H @ x = b`` one diagonal block at a time."""
        return solve_block_diagonal(
            b,
            [block.dimension for block in self._blocks],
            [block.solve for block in self._blocks],
        )

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``H @ y`` one diagonal block at a time."""
        return multiply_block_diagonal(
            y,
            [block.dimension for block in self._blocks],
            [block.multiply for block in self._blocks],
        )


class ArrowSystem(System):
    """A bordered (arrow / Schur) matrix ``[[A11, A12], [A12^T, A22]]``.

    The upper-left block ``A11`` is supplied as any :class:`System`; the border
    ``A12`` and corner ``A22`` are a small dense fringe of ``p`` rows/columns.
    Solving exploits the Schur complement on ``A11``. When ``A11`` is a
    :class:`DiagonalSystem` with a scalar corner the matrix has a literal arrow
    sparsity pattern, and a dedicated numerically-stable kernel is used.

    The whole matrix is assumed positive definite; use :class:`KKTSystem` for
    the indefinite saddle-point case (a zero corner).
    """

    def __init__(
        self,
        upper_left: System,
        border: npt.NDArray[np.float64],
        corner: npt.NDArray[np.float64] | float,
    ) -> None:
        """Build a bordered System.

        Parameters
        ----------
         upper_left : System
            The ``A11`` block.
         border : npt.NDArray[np.float64]
            The ``A12`` block: a length-M vector (scalar corner) or an
            ``(M, p)`` matrix.
         corner : npt.NDArray[np.float64] or float
            The ``A22`` block: a scalar (vector border) or a ``(p, p)`` matrix.

        """
        M = upper_left.dimension
        border = np.asarray(border, dtype=np.float64)
        if border.ndim == 1:
            border = border[:, np.newaxis]
        if border.ndim != 2 or border.shape[0] != M:
            raise ValueError(
                f"border must have {M} rows to match the upper-left System."
            )
        p = border.shape[1]

        corner = np.asarray(corner, dtype=np.float64)
        if corner.ndim == 0:
            corner = corner.reshape(1, 1)
        if corner.shape != (p, p):
            raise ValueError(
                f"corner must be ({p}, {p}) to match the border's {p} column(s)."
            )

        self._upper_left = upper_left
        self._border = border
        self._corner = corner
        self._p = p

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._upper_left.dimension + self._p

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve ``H @ x = b`` via the Schur complement on the upper-left block."""
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
        """Compute ``H @ y``."""
        if self._p == 1:
            return multiply_block_plus_one(
                y,
                self._border[:, 0],
                float(self._corner[0, 0]),
                self._upper_left.multiply,
            )
        return multiply_block(y, self._border, self._corner, self._upper_left.multiply)


class KKTSystem(System):
    """The saddle-point system ``[[H, A^T], [A, 0]]``.

    The Hessian ``H`` is supplied as a :class:`System` and ``A`` is the
    ``(p, M)`` equality-constraint matrix. This is the bordered system of an
    equality-constrained Newton step; the corner block is zero, so the matrix
    is indefinite and :class:`ArrowSystem` does not apply.

    Because the constraint block of the system is zero, :meth:`solve` requires
    the trailing ``p`` entries of its right-hand side to vanish -- exactly the
    case for a Newton step taken from an equality-feasible iterate. The leading
    ``M`` entries are the (negative) gradient and the solution stacks the
    primal step and the equality multipliers, ``[delta_x; nu]``.
    """

    def __init__(self, hessian: System, A: npt.NDArray[np.float64]) -> None:
        """Build a KKT System from a Hessian System and constraint matrix ``A``."""
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A must be a 2D array.")
        if A.shape[1] != hessian.dimension:
            raise ValueError(
                f"A has {A.shape[1]} columns but the Hessian System has "
                f"dimension {hessian.dimension}."
            )
        self._hessian = hessian
        self._A = A

    @property
    def dimension(self) -> int:
        """Order of the matrix."""
        return self._hessian.dimension + self._A.shape[0]

    def solve(self, b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve the KKT system, exploiting the Schur complement on ``H``.

        The trailing constraint-block entries of ``b`` must be zero.
        """
        M = self._hessian.dimension
        if not np.allclose(b[M:], 0.0):
            raise ValueError(
                "KKTSystem.solve requires the trailing constraint-block entries "
                "of b to be zero (the equality-constraint residual must vanish)."
            )
        delta_x, nu = solve_kkt_system(self._A, b[:M], self._hessian.solve)
        if b.ndim == 1:
            return np.concatenate([delta_x, nu])
        return np.vstack([delta_x, nu])

    def multiply(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute ``[[H, A^T], [A, 0]] @ y`` for an arbitrary ``y``."""
        M = self._hessian.dimension
        y1 = y[:M]
        y2 = y[M:]
        top = self._hessian.multiply(y1) + self._A.T @ y2
        bottom = self._A @ y1
        if y.ndim == 1:
            return np.concatenate([top, bottom])
        return np.vstack([top, bottom])
