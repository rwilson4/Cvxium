"""Structured linear-system solvers and matrix-vector products for Cvxium.

Provides ``solve_*`` and ``multiply_*`` helpers that exploit the special structure of
the Hessians arising in Cvxium's interior-point Newton steps, so that those steps run in
near-linear rather than cubic time. The supported structures are diagonal, rank-one and
rank-p updates and downdates (through the Sherman-Morrison and Woodbury identities),
arrow sparsity, block and block-plus-one Schur complements, symmetric banded, block
diagonal, and the full KKT system (``solve_kkt_system``).

The helpers compose: a rank-p update can wrap an arrow solve, a block-diagonal solve can
dispatch to a different structure per block, and so on. A structured solve takes the
solve of its base matrix as a callable, so any helper here can serve as another's base.
Several solves also have a prepare/apply split (``_prepare_rank_one_update``,
``_prepare_rank_p_update``, ``factor_banded``) so that ``systems`` can factor once and
reuse the factorization across the right-hand sides of one Newton step.

Solves raise ``NewtonStepError`` when the purported Hessian is not positive definite, is
singular, or is non-finite, signaling callers to fall back rather than trust a corrupt
Newton step. ``aspect_ratio_svd`` and ``tikhonov_filter`` support pseudo-inverses of
rank-deficient constraint matrices.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy import linalg
from scipy.linalg import lapack

from ..exceptions import NewtonStepError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def multiply_diagonal(
    y: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Multiply a vector or matrix by a diagonal matrix.

    Computes H y with H = diag(η), in linear time.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column.
     eta : npt.NDArray[np.float64]
        Diagonal η of H, of length M.

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``y`` is not 1D or 2D, or its leading dimension does not match ``eta``.

    """
    if y.ndim == 1:
        if y.shape != eta.shape:
            raise ValueError("y and eta must have the same length.")
        return eta * y
    elif y.ndim == 2:
        if y.shape[0] != eta.shape[0]:
            raise ValueError("Number of rows in y must match length of eta.")
        return eta[:, np.newaxis] * y
    else:
        raise ValueError("y must be either a 1D or 2D NumPy array.")


def solve_diagonal(
    b: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve a diagonal linear system.

    Solves H x = b with H = diag(η), in linear time.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     eta : npt.NDArray[np.float64]
        Diagonal η of H, of length M.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     NewtonStepError
        When some η_i ≤ 0, so H is not positive definite. Checked before the shapes.
     ValueError
        When ``b`` is not 1D or 2D, or its leading dimension does not match ``eta``.

    """
    if not np.all(eta > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if b.ndim == 1:
        if b.shape != eta.shape:
            raise ValueError("b and eta must have the same length.")
        return b / eta
    elif b.ndim == 2:
        if b.shape[0] != eta.shape[0]:
            raise ValueError("Number of rows in beta must match length of eta.")
        return b / eta[:, np.newaxis]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def solve_diagonal_eta_inverse(
    b: npt.NDArray[np.float64],
    eta_inverse: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve a diagonal linear system given the reciprocal of its diagonal.

    Solves H x = b with H = diag(η), given 1 / η, so the solve is a multiplication
    rather than a division. Linear time.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     eta_inverse : npt.NDArray[np.float64]
        The reciprocals 1 / η_i of the diagonal of H, of length M.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     NewtonStepError
        When some 1 / η_i ≤ 0, so H is not positive definite. Checked before the shapes.
     ValueError
        When ``b`` is not 1D or 2D, or its leading dimension does not match
        ``eta_inverse``.

    """
    if not np.all(eta_inverse > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if b.ndim == 1:
        if b.shape != eta_inverse.shape:
            raise ValueError("b and eta_inverse must have the same length.")
        return b * eta_inverse
    elif b.ndim == 2:
        if b.shape[0] != eta_inverse.shape[0]:
            raise ValueError("Number of rows in beta must match length of eta_inverse.")
        return b * eta_inverse[:, np.newaxis]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def multiply_rank_one_update(
    y: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    A_multiply: Callable[..., npt.NDArray[np.float64]],
    d: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Multiply by a rank-one update of a structured matrix.

    Computes H y with
       H = A + d κ κ^T,
    where A has structure that makes A z cheap to compute and κ is a vector. H is a
    rank-one update of A when d > 0 and a downdate when d < 0; when ``d`` is ``None``,
    the weight is 1.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column.
     kappa : npt.NDArray[np.float64]
        The vector κ, of length M.
     A_multiply : Callable[..., npt.NDArray[np.float64]]
        Computes A z, called as ``A_multiply(z, **kwargs)``. Must accept a vector or a
        matrix and preserve its shape.
     d : float | None
        Weight on the rank-one term; ``None`` means 1.
     **kwargs : Any
        Extra arguments passed to ``A_multiply``.

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``y`` is not 1D or 2D, its leading dimension does not match ``kappa``, or
        ``A_multiply`` does not preserve the shape of its input.

    Notes
    -----
    For a vector y, computing
       H y = A y + κ (d κ^T y)
    takes t + 4M flops, where t is the cost of A y:
    1. Compute A y (t flops).
    2. Compute the scalar d κ^T y (2M flops).
    3. Add κ times that scalar to A y (2M flops).

    """
    if y.ndim == 1:
        if y.shape != kappa.shape:
            raise ValueError("y and kappa must have the same length.")
    elif y.ndim == 2:
        if y.shape[0] != kappa.shape[0]:
            raise ValueError("Number of rows in y must match length of kappa.")
    else:
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    scale = d if d is not None else 1.0
    Ay = A_multiply(y, **kwargs)
    if Ay.shape != y.shape:
        raise ValueError(
            f"A_multiply violated shape contract: y shape {y.shape} -> "
            f"A_multiply(y) shape {Ay.shape}. A_multiply must preserve the "
            "shape of its input (handle both 1D vectors and 2D matrices)."
        )
    if y.ndim == 1:
        return Ay + kappa * (scale * np.dot(kappa, y))
    else:
        return Ay + np.outer(kappa, scale * (kappa.T @ y))


def solve_rank_one_update(
    b: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    A_solve: Callable[..., npt.NDArray[np.float64]],
    d: float | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve a rank-one update of a structured system by Sherman-Morrison.

    Solves H x = b with
       H = A + d κ κ^T,
    where A has structure that makes A y = c cheap to solve and κ is a vector. H is a
    rank-one update of A when d > 0 and a downdate when d < 0; when ``d`` is ``None``,
    the weight is 1. Composes ``_prepare_rank_one_update`` and
    ``_apply_rank_one_update``.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     kappa : npt.NDArray[np.float64]
        The vector κ, of length M.
     A_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A y = c, called as ``A_solve(c, **kwargs)``. Must accept a vector or a
        matrix and preserve its shape.
     d : float | None
        Weight on the rank-one term; must be nonzero. ``None`` means 1.
     **kwargs : Any
        Extra arguments passed to ``A_solve``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``d`` is zero, ``b`` is not 1D or 2D, its leading dimension does not match
        ``kappa``, or ``A_solve`` does not preserve the shape of its input.
     NewtonStepError
        When the Sherman-Morrison scalar G is zero to within a relative tolerance (H
        singular), or has the wrong sign for a downdate (H indefinite). Propagated from
        ``_prepare_rank_one_update``.

    Notes
    -----
    By the Sherman-Morrison formula, with the scalar
       G = 1 / d + κ^T A^{-1} κ,
    the solution is
       x = A^{-1} b - ((κ^T A^{-1} b) / G) A^{-1} κ,
    valid whenever G ≠ 0. When A is positive definite, the sign of G reveals the
    definiteness of H:
    - d > 0 (update): G > 0 always, and H is positive definite.
    - d < 0 (downdate): G < 0 means H is positive definite, G = 0 means H is singular,
      and G > 0 means H is indefinite (the downdate is too large).
    The solve raises ``NewtonStepError`` unless H is positive definite.

    For a vector b, the cost is 2t + 6M flops, where t is the cost of one solve with A:
    1. Solve A ξ = κ (t flops).
    2. Form G = 1 / d + κ^T ξ and its reciprocal (2M flops).
    3. Solve A x' = b (t flops).
    4. Compute x = x' - ((κ^T x') / G) ξ (4M flops).

    """
    prep = _prepare_rank_one_update(kappa, A_solve, d=d, **kwargs)
    return _apply_rank_one_update(b, prep, A_solve, **kwargs)


@dataclass(frozen=True)
class _RankOneUpdatePrep:
    """Cached Sherman-Morrison data for repeated rank-one solves with one matrix.

    κ, ξ = A^{-1} κ, and 1 / G do not depend on the right-hand side, so they are
    computed once (one solve with A) and reused across the solves with H = A + d κ κ^T.
    Returned by ``_prepare_rank_one_update`` and consumed by ``_apply_rank_one_update``.

    Fields
    ------
     kappa : npt.NDArray[np.float64]
        The vector κ of the rank-one term.
     xi : npt.NDArray[np.float64]
        The solve ξ = A^{-1} κ.
     den : float
        Reciprocal 1 / G of the Sherman-Morrison scalar G = 1 / d + κ^T ξ.

    """

    kappa: npt.NDArray[np.float64]
    xi: npt.NDArray[np.float64]
    den: float


def _prepare_rank_one_update(
    kappa: npt.NDArray[np.float64],
    A_solve: Callable[..., npt.NDArray[np.float64]],
    d: float | None = None,
    **kwargs: Any,
) -> _RankOneUpdatePrep:
    """Precompute the right-hand-side-independent Sherman-Morrison data.

    Solves A ξ = κ once and forms the Sherman-Morrison scalar
       G = 1 / d + κ^T ξ,
    checking that H = A + d κ κ^T is positive definite (see ``solve_rank_one_update``),
    so that repeated solves with H reuse this work.

    Parameters
    ----------
     kappa : npt.NDArray[np.float64]
        The vector κ, of length M.
     A_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A y = c, called as ``A_solve(c, **kwargs)``; must preserve the shape of
        its input.
     d : float | None
        Weight on the rank-one term; must be nonzero. ``None`` means 1.
     **kwargs : Any
        Extra arguments passed to ``A_solve``.

    Returns
    -------
     prep : _RankOneUpdatePrep
        The cached κ, ξ = A^{-1} κ, and 1 / G.

    Raises
    ------
     ValueError
        When ``d`` is zero, or ``A_solve`` does not preserve the shape of its input.
     NewtonStepError
        When |G| ≤ 10^-12 max(1, |1 / d|, |κ^T ξ|) (H singular or nearly so), or, for a
        downdate, when G is not negative (H indefinite).

    """
    if d is not None and d == 0:
        raise ValueError("d must be nonzero.")

    xi = A_solve(kappa, **kwargs)
    if xi.shape != kappa.shape:
        raise ValueError(
            f"A_solve violated shape contract: kappa shape {kappa.shape} -> "
            f"A_solve(kappa) shape {xi.shape}. A_solve must preserve the "
            "shape of its input."
        )
    schur_diag = 1.0 / d if d is not None else 1.0
    kappa_dot_xi = float(np.dot(kappa, xi))
    G = schur_diag + kappa_dot_xi

    # Relative tolerance: G is singular when the schur_diag and the quadratic form
    # cancel to within the precision of their inputs, not when G falls below an absolute
    # floor. The latter spuriously trips when kappa is small or d is large.
    tol = 1e-12 * max(1.0, abs(schur_diag), abs(kappa_dot_xi))
    if abs(G) <= tol:
        raise NewtonStepError("H is singular or nearly singular")

    # For a downdate (d < 0), H is PD only when G < 0.
    # G > 0 with d < 0 means the downdate is too large; H is indefinite.
    # G = 0 means H is singular.
    if d is not None and d < 0 and G >= 0:
        raise NewtonStepError("H is not positive definite")

    return _RankOneUpdatePrep(kappa=kappa, xi=xi, den=1.0 / G)


def _apply_rank_one_update(
    b: npt.NDArray[np.float64],
    prep: _RankOneUpdatePrep,
    A_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve a rank-one-updated system from cached Sherman-Morrison data.

    Computes
       x = x' - ((κ^T x') / G) ξ,   x' = A^{-1} b,
    using the κ, ξ, and 1 / G cached in ``prep``, so each right-hand side costs one
    solve with A plus O(M) work per column.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     prep : _RankOneUpdatePrep
        Data from ``_prepare_rank_one_update`` for the same A, κ, and d.
     A_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A y = c, called as ``A_solve(c, **kwargs)``; must preserve the shape of
        its input.
     **kwargs : Any
        Extra arguments passed to ``A_solve``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``b`` is not 1D or 2D, its leading dimension does not match κ, or
        ``A_solve`` does not preserve the shape of its input.

    """
    kappa = prep.kappa
    if b.ndim == 1:
        if b.shape != kappa.shape:
            raise ValueError("b and kappa must have the same length.")
    elif b.ndim == 2:
        if b.shape[0] != kappa.shape[0]:
            raise ValueError("Number of rows in b must match length of kappa.")
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    x_prime = A_solve(b, **kwargs)
    if x_prime.shape != b.shape:
        raise ValueError(
            f"A_solve violated shape contract: b shape {b.shape} -> "
            f"A_solve(b) shape {x_prime.shape}. A_solve must preserve the "
            "shape of its input."
        )
    if b.ndim == 1:
        return x_prime - (np.dot(kappa, x_prime) * prep.den) * prep.xi
    return x_prime - prep.den * np.outer(prep.xi, kappa.T @ x_prime)


def multiply_rank_p_update(
    y: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    A_multiply: Callable[..., npt.NDArray[np.float64]],
    d: npt.NDArray[np.float64] | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Multiply by a rank-p update of a structured matrix.

    Computes H y with
       H = A + κ D κ^T,   D = diag(d),
    where A has structure that makes A z cheap to compute and κ is M × p. Positive
    entries of d are updates and negative entries downdates; mixed signs are allowed.
    When ``d`` is ``None``, D = I.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column.
     kappa : npt.NDArray[np.float64]
        The matrix κ, of shape (M, p).
     A_multiply : Callable[..., npt.NDArray[np.float64]]
        Computes A z, called as ``A_multiply(z, **kwargs)``. Must accept a vector or a
        matrix and preserve its shape.
     d : npt.NDArray[np.float64] | None
        Diagonal of D, of length p; ``None`` means all ones.
     **kwargs : Any
        Extra arguments passed to ``A_multiply``.

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``y`` is not 1D or 2D, its leading dimension does not match ``kappa``,
        ``d`` does not have length p, or ``A_multiply`` does not preserve the shape of
        its input.

    Notes
    -----
    For a vector y, computing
       H y = A y + κ (D (κ^T y))
    takes t + 4Mp + p flops, where t is the cost of A y:
    1. Compute A y (t flops).
    2. Compute κ^T y (2Mp flops).
    3. Scale by D (p flops).
    4. Multiply by κ and add to A y (2Mp flops).

    """
    if y.ndim not in (1, 2):
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    if y.shape[0] != kappa.shape[0]:
        raise ValueError("Number of rows in y must match number of rows in kappa.")

    p = kappa.shape[1]

    if d is not None and d.shape != (p,):
        raise ValueError("d must be a 1D array of length p (kappa.shape[1]).")

    kkt = kappa.T @ y  # shape: (p,) or (p, q)
    if d is not None:
        if y.ndim == 1:
            kkt = d * kkt
        else:
            kkt = d[:, np.newaxis] * kkt
    Ay = A_multiply(y, **kwargs)
    if Ay.shape != y.shape:
        raise ValueError(
            f"A_multiply violated shape contract: y shape {y.shape} -> "
            f"A_multiply(y) shape {Ay.shape}. A_multiply must preserve the "
            "shape of its input (handle both 1D vectors and 2D matrices)."
        )
    return Ay + kappa @ kkt


def solve_rank_p_update(
    b: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    A_solve: Callable[..., npt.NDArray[np.float64]],
    d: npt.NDArray[np.float64] | None = None,
    check_inertia: bool = True,
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve a rank-p update of a structured system by the Woodbury identity.

    Solves H x = b with
       H = A + κ D κ^T,   D = diag(d),
    where A has structure that makes A y = c cheap to solve and κ is M × p with p ≪ M.
    Positive entries of d are updates and negative entries downdates; mixed signs are
    allowed. When ``d`` is ``None``, D = I. Composes ``_prepare_rank_p_update`` and
    ``_apply_rank_p_update``.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     kappa : npt.NDArray[np.float64]
        The matrix κ, of shape (M, p).
     A_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A y = c, called as ``A_solve(c, **kwargs)``. Must accept a vector or a
        matrix and preserve its shape.
     d : npt.NDArray[np.float64] | None
        Diagonal of D, of length p, with no zero entries; ``None`` means all ones.
     check_inertia : bool
        When True and ``d`` has mixed signs, verify that the inertia of the Woodbury
        matrix G matches the signs of d, which ensures H is positive definite. Set it to
        False when the caller detects a non-positive-definite Hessian by other means.
     **kwargs : Any
        Extra arguments passed to ``A_solve``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``b`` is not 1D or 2D, ``kappa`` is not 2D, their leading dimensions
        disagree, ``d`` does not have length p or has a zero entry, or ``A_solve`` does
        not preserve the shape of its input.
     NewtonStepError
        When the Woodbury matrix G is singular, non-finite, or has inertia that does not
        match the signs of d, so H is not positive definite. Propagated from
        ``_prepare_rank_p_update`` and ``_apply_rank_p_update``.

    Notes
    -----
    By the Woodbury identity, with the p × p Woodbury matrix
       G = D^{-1} + κ^T A^{-1} κ,
    the solution is
       x = A^{-1} b - A^{-1} κ G^{-1} κ^T A^{-1} b,
    valid whenever G is invertible. The code forms the equilibrated matrix
       |D|^{1/2} G |D|^{1/2}
    instead (see ``_prepare_rank_p_update``), which has the same inertia. When A is
    positive definite, the signs of d determine how the positive definiteness of H is
    checked through G:
    - All d > 0 (pure update): G and H are always positive definite; G is factored by
      Cholesky.
    - All d < 0 (pure downdate): H is positive definite exactly when G is negative
      definite; -G is factored by Cholesky, and a failure means H is indefinite.
    - Mixed signs: H is positive definite exactly when G has as many positive
      eigenvalues as d has positive entries (see ``_verify_woodbury_inertia``); G is
      factored by LU.

    With q right-hand sides and t the cost of one solve with A, the cost is
       (p + q) t + 2Mp (p + 2q)
    plus O(p^3 + p^2 q) for G, which is negligible when p, q ≪ M:
    1. Solve A ξ = κ, with ξ of shape (M, p) (p t).
    2. Form G = D^{-1} + κ^T ξ (2Mp^2).
    3. Factor G (O(p^3)).
    4. Solve A x' = b, with x' of shape (M, q) (q t).
    5. Compute z = κ^T x', of shape (p, q) (2Mpq).
    6. Solve G y = z (O(p^2 q)).
    7. Compute x = x' - ξ y (2Mpq).

    """
    if b.ndim not in (1, 2):
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    prep = _prepare_rank_p_update(
        kappa, A_solve, d=d, check_inertia=check_inertia, **kwargs
    )
    return _apply_rank_p_update(b, prep, A_solve, **kwargs)


@dataclass(frozen=True)
class _RankPUpdatePrep:
    """Cached Woodbury data for repeated rank-p solves with one matrix.

    The equilibrated κ_eq, ξ_eq = A^{-1} κ_eq, and the factored Woodbury matrix do not
    depend on the right-hand side, so they are computed once (p solves with A and one
    small factorization) and reused across the solves with H = A + κ diag(d) κ^T.
    Returned by ``_prepare_rank_p_update`` and consumed by ``_apply_rank_p_update``.

    Fields
    ------
     kappa_eq : npt.NDArray[np.float64]
        Equilibrated factor κ_eq = κ diag(|d|)^{1/2} (κ itself when d is ``None``), of
        shape (M, p).
     xi_eq : npt.NDArray[np.float64]
        The equilibrated solve ξ_eq = A^{-1} κ_eq, of shape (M, p).
     woodbury : _WoodburyFactor
        Factorization of the equilibrated Woodbury matrix.

    """

    kappa_eq: npt.NDArray[np.float64]
    xi_eq: npt.NDArray[np.float64]
    woodbury: _WoodburyFactor


def _prepare_rank_p_update(
    kappa: npt.NDArray[np.float64],
    A_solve: Callable[..., npt.NDArray[np.float64]],
    d: npt.NDArray[np.float64] | None = None,
    check_inertia: bool = True,
    **kwargs: Any,
) -> _RankPUpdatePrep:
    """Precompute the right-hand-side-independent Woodbury data.

    Solves A ξ = κ once, equilibrates by |d|^{1/2}, and forms and factors the p × p
    Woodbury matrix, checking that H = A + κ diag(d) κ^T is positive definite (see
    ``solve_rank_p_update``), so that repeated solves with H reuse this work.

    Parameters
    ----------
     kappa : npt.NDArray[np.float64]
        The matrix κ, of shape (M, p).
     A_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A y = c, called as ``A_solve(c, **kwargs)``; must preserve the shape of
        its input.
     d : npt.NDArray[np.float64] | None
        Diagonal of D, of length p, with no zero entries; ``None`` means all ones.
     check_inertia : bool
        When True and ``d`` has mixed signs, verify that the inertia of the Woodbury
        matrix matches the signs of d.
     **kwargs : Any
        Extra arguments passed to ``A_solve``.

    Returns
    -------
     prep : _RankPUpdatePrep
        The cached κ_eq, ξ_eq = A^{-1} κ_eq, and the factored Woodbury matrix.

    Raises
    ------
     ValueError
        When ``kappa`` is not 2D, ``d`` does not have length p or has a zero entry, or
        ``A_solve`` does not preserve the shape of its input.
     NewtonStepError
        When the Woodbury matrix is singular, non-finite, or has inertia that does not
        match the signs of d, so H is not positive definite. Propagated from
        ``_factor_woodbury``.

    Notes
    -----
    The plain Woodbury matrix
       G = diag(1 / d) + κ^T A^{-1} κ
    is badly conditioned near the boundary of the feasible set, where the barrier
    weights d_i = λ_i / (-f_i) span many orders of magnitude (active constraints drive
    d_i up, vanishing multipliers drive it down). Substituting
       κ_eq = κ diag(|d|)^{1/2},   σ = sign(d),
    gives the equilibrated matrix
       G_eq = diag(σ) + κ_eq^T A^{-1} κ_eq = diag(|d|)^{1/2} G diag(|d|)^{1/2},
    whose conditioning depends on ‖κ_eq^T A^{-1} κ_eq‖ rather than on max 1 / |d_i|.
    Being congruent to G, it has the same inertia. With ξ_eq = A^{-1} κ_eq, the final
    step
       x = x' - ξ_eq G_eq^{-1} κ_eq^T x'
    absorbs the equilibration, so the result is unchanged. G_eq is symmetrized before
    factoring to remove rounding asymmetry.

    """
    if kappa.ndim != 2:
        raise ValueError("kappa must be a 2D array with shape (M, p).")

    p = kappa.shape[1]

    if d is not None:
        if d.shape != (p,):
            raise ValueError("d must be a 1D array of length p (kappa.shape[1]).")
        if not np.all(d != 0):
            raise ValueError("All entries of d must be nonzero.")

    # xi is M-by-p
    xi = A_solve(kappa, **kwargs)
    if xi.shape != kappa.shape:
        # Defensive: a callable that returns the wrong shape (e.g., a closure that
        # broadcasts `(M, 1) / (M,)` into `(M, M)`) silently produces garbage Newton
        # steps downstream. Catch the contract violation here.
        raise ValueError(
            f"A_solve violated shape contract: kappa shape {kappa.shape} "
            f"-> A_solve(kappa) shape {xi.shape}. A_solve must preserve the "
            "shape of its input (handle both 1D vectors and 2D matrices)."
        )

    # Equilibrate by sqrt(|d|) before forming the Woodbury matrix. The naive form
    # G = diag(1/d) + kappa^T A^{-1} kappa is catastrophically ill-conditioned in the
    # IPM boundary regime, where d = lambda / (-f) spans many orders of magnitude
    # (active constraints drive d large, near-zero multipliers drive d small).
    # Substituting kappa_eq = kappa * sqrt(|d|), sign_d = sign(d) yields the equivalent
    # Woodbury matrix
    #     G_eq = diag(sign_d) + kappa_eq^T A^{-1} kappa_eq,
    # whose conditioning depends on ||kappa_eq^T A^{-1} kappa_eq|| rather than max(1/d).
    # The substitution preserves the result via xi_eq = xi * sqrt(|d|) and
    # z_eq = kappa_eq^T x_prime, with the final subtraction x_prime - xi_eq @ y_eq
    # absorbing the equilibration.
    if d is not None:
        sqrt_abs_d = np.sqrt(np.abs(d))
        sign_d = np.sign(d).astype(np.float64)
        kappa_eq = kappa * sqrt_abs_d
        xi_eq = xi * sqrt_abs_d
        G = np.asarray(np.diag(sign_d) + kappa_eq.T @ xi_eq)
    else:
        sign_d = None
        kappa_eq = kappa
        xi_eq = xi
        G = np.asarray(np.eye(p) + kappa.T @ xi)

    # Clean up any asymmetries before factoring
    G = np.asarray(0.5 * (G + G.T))

    woodbury = _factor_woodbury(G, sign_d, check_inertia=check_inertia)
    return _RankPUpdatePrep(kappa_eq=kappa_eq, xi_eq=xi_eq, woodbury=woodbury)


def _apply_rank_p_update(
    b: npt.NDArray[np.float64],
    prep: _RankPUpdatePrep,
    A_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve a rank-p-updated system from cached Woodbury data.

    Computes
       x = x' - ξ_eq G_eq^{-1} κ_eq^T x',   x' = A^{-1} b,
    using the data cached in ``prep``, so each batch of q right-hand sides costs q
    solves with A plus O(M p q + p^2 q) work.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     prep : _RankPUpdatePrep
        Data from ``_prepare_rank_p_update`` for the same A, κ, and d.
     A_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A y = c, called as ``A_solve(c, **kwargs)``; must preserve the shape of
        its input.
     **kwargs : Any
        Extra arguments passed to ``A_solve``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``b`` is not 1D or 2D, its leading dimension does not match κ, or
        ``A_solve`` does not preserve the shape of its input; also propagated from
        ``_apply_woodbury`` for a non-finite right-hand side in the Cholesky case.
     NewtonStepError
        Propagated from ``_apply_woodbury`` when the LU solve fails.

    """
    if b.ndim not in (1, 2):
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    if b.shape[0] != prep.kappa_eq.shape[0]:
        raise ValueError("b.shape[0] must equal kappa.shape[0].")

    # q RHS -> x_prime is M-by-q
    x_prime = A_solve(b, **kwargs)
    if x_prime.shape != b.shape:
        raise ValueError(
            f"A_solve violated shape contract: b shape {b.shape} -> "
            f"A_solve(b) shape {x_prime.shape}. A_solve must preserve the "
            "shape of its input (handle both 1D vectors and 2D matrices)."
        )
    z = prep.kappa_eq.T @ x_prime  # p-by-q

    y = _apply_woodbury(prep.woodbury, z)
    return x_prime - prep.xi_eq @ y


def _diagnose_nonfinite(
    arr: npt.NDArray[np.float64],
    name: str,
) -> str:
    """Build a message describing the NaN and infinite entries of an array.

    Parameters
    ----------
     arr : npt.NDArray[np.float64]
        Array suspected of holding non-finite entries.
     name : str
        Human-readable name used in the message, such as "Woodbury matrix".

    Returns
    -------
     message : str
        A semicolon-delimited message giving the counts of NaN, +Inf, and -Inf entries
        and up to five sample indices, suitable for an exception or a log line.

    """
    nan_mask = np.isnan(arr)
    pos_inf_mask = np.isposinf(arr)
    neg_inf_mask = np.isneginf(arr)
    n_nan = int(nan_mask.sum())
    n_pos_inf = int(pos_inf_mask.sum())
    n_neg_inf = int(neg_inf_mask.sum())
    parts = [f"{name} contains non-finite entries"]
    if n_nan:
        parts.append(f"{n_nan} NaN")
    if n_pos_inf:
        parts.append(f"{n_pos_inf} +Inf")
    if n_neg_inf:
        parts.append(f"{n_neg_inf} -Inf")
    bad = nan_mask | pos_inf_mask | neg_inf_mask
    bad_idx = np.argwhere(bad)
    if bad_idx.size > 0:
        sample = [tuple(int(i) for i in idx) for idx in bad_idx[:5]]
        parts.append(f"first bad indices: {sample}")
    return "; ".join(parts)


def _verify_woodbury_inertia(
    G: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> None:
    """Check that the Woodbury matrix's inertia matches the signs of d.

    Counts the eigenvalues of G that are positive, negative, and zero (within 10^-12
    max(1, max |eigenvalue|)), and compares them with the signs of d.

    Parameters
    ----------
     G : npt.NDArray[np.float64]
        The symmetric Woodbury matrix (plain or equilibrated; congruence preserves
        inertia), of shape (p, p).
     d : npt.NDArray[np.float64]
        The weights d of the rank-p term, or their signs, of length p.

    Raises
    ------
     NewtonStepError
        When G has a zero eigenvalue (H singular or nearly so), or its counts of
        positive and negative eigenvalues differ from those of d (H not positive
        definite).

    Notes
    -----
    With A positive definite, take the inertia of the block matrix
       K = [ A  κ ; κ^T  -D^{-1} ]
    two ways. Eliminating A leaves the Schur complement -G, and eliminating -D^{-1}
    leaves H = A + κ D κ^T, so
       In(A) + In(-G) = In(-D^{-1}) + In(H)
    (Haynsworth inertia additivity). Since In(A) is all positive, H is positive definite
    exactly when In(G) = In(D), that is, when G has as many positive and negative
    eigenvalues as d has positive and negative entries.

    """
    evals = np.linalg.eigvalsh(G)
    tol = 1e-12 * max(1.0, np.max(np.abs(evals)))
    n_pos_G = int(np.sum(evals > tol))
    n_neg_G = int(np.sum(evals < -tol))
    n_zero_G = len(evals) - n_pos_G - n_neg_G
    if n_zero_G > 0:
        raise NewtonStepError("H is singular or nearly singular")
    if n_pos_G != int(np.sum(d > 0)) or n_neg_G != int(np.sum(d < 0)):
        raise NewtonStepError("H is not positive definite")


def _cho_factor_woodbury(
    M: npt.NDArray[np.float64],
    G: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Cholesky-factor the Woodbury matrix, translating failures to ``NewtonStepError``.

    Parameters
    ----------
     M : npt.NDArray[np.float64]
        The matrix to factor: G for a pure update, -G for a pure downdate.
     G : npt.NDArray[np.float64]
        The Woodbury matrix itself, used only for the non-finite diagnostic message.

    Returns
    -------
     cho : tuple[npt.NDArray[np.float64], bool]
        The lower Cholesky factor of ``M`` and the flag True, as returned by
        ``scipy.linalg.cho_factor``.

    Raises
    ------
     NewtonStepError
        When ``M`` is not positive definite, or holds non-finite entries.

    """
    try:
        return linalg.cho_factor(M, lower=True)
    except np.linalg.LinAlgError:
        raise NewtonStepError("H is not positive definite") from None
    except ValueError:
        raise NewtonStepError(_diagnose_nonfinite(G, "Woodbury matrix")) from None


@dataclass(frozen=True)
class _WoodburyFactor:
    """Reusable factorization of the Woodbury matrix G.

    Exactly one of ``cho`` (Cholesky of G, or of -G for a pure downdate) and ``lu`` (LU
    of G, for mixed signs) is set. Separating the factorization from the solve lets a
    caller factor G once and apply it to several right-hand sides, such as the three
    solves of one KKT system.

    Fields
    ------
     cho : tuple[npt.NDArray[np.float64], bool] | None
        Cholesky factor of G, or of -G for a pure downdate, as the (factor, lower) pair
        from ``scipy.linalg.cho_factor``; ``None`` when ``lu`` is set.
     lu : tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]] | None
        LU factor of G, as the (lu, piv) pair from ``scipy.linalg.lu_factor``; ``None``
        when ``cho`` is set.
     negate : bool
        True when ``cho`` factors -G, so that the applied solve negates its result.

    """

    cho: tuple[npt.NDArray[np.float64], bool] | None
    lu: tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]] | None
    negate: bool


def _factor_woodbury(
    G: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64] | None,
    check_inertia: bool = True,
) -> _WoodburyFactor:
    """Factor the Woodbury matrix for repeated solves, according to the signs of d.

    When ``d`` is ``None`` or all positive, factors G by Cholesky; when all negative,
    factors -G by Cholesky; for mixed signs, optionally checks the inertia and then
    factors G by LU. See ``solve_rank_p_update`` for the definiteness reasoning.

    Parameters
    ----------
     G : npt.NDArray[np.float64]
        The symmetric Woodbury matrix, of shape (p, p).
     d : npt.NDArray[np.float64] | None
        The weights d of the rank-p term, or their signs, of length p; ``None`` means
        all positive.
     check_inertia : bool
        When True and ``d`` has mixed signs, verify that the inertia of G matches the
        signs of d before factoring.

    Returns
    -------
     factor : _WoodburyFactor
        The factorization, for use with ``_apply_woodbury``.

    Raises
    ------
     NewtonStepError
        When G (or -G for a pure downdate) is not positive definite, the inertia check
        fails, the LU factorization fails, or G is non-finite.

    """
    if d is None or np.all(d > 0):
        # Pure update: G is always PD when A is PD.
        return _WoodburyFactor(cho=_cho_factor_woodbury(G, G), lu=None, negate=False)

    if np.all(d < 0):
        # Pure downdate: H is PD iff G is ND, i.e. -G is PD.
        return _WoodburyFactor(cho=_cho_factor_woodbury(-G, G), lu=None, negate=True)

    # Mixed signs: optionally verify inertia, then factor via LU.
    if check_inertia:
        _verify_woodbury_inertia(G, d)

    try:
        lu = linalg.lu_factor(G)
    except np.linalg.LinAlgError:
        raise NewtonStepError("Failed to solve Gy = z") from None
    except ValueError:
        raise NewtonStepError(_diagnose_nonfinite(G, "Woodbury matrix")) from None
    return _WoodburyFactor(cho=None, lu=lu, negate=False)


def _apply_woodbury(
    factor: _WoodburyFactor,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve G y = z from a precomputed Woodbury factorization.

    Parameters
    ----------
     factor : _WoodburyFactor
        The factorization from ``_factor_woodbury``.
     z : npt.NDArray[np.float64]
        Right-hand side, of length p or of shape (p, q).

    Returns
    -------
     y : npt.NDArray[np.float64]
        The solution G^{-1} z, of the same shape as ``z``.

    Raises
    ------
     NewtonStepError
        In the LU case, when the solve fails or ``z`` is non-finite.
     ValueError
        In the Cholesky case, propagated from ``scipy.linalg.cho_solve`` when ``z`` is
        non-finite.

    """
    if factor.cho is not None:
        y = linalg.cho_solve(factor.cho, z)
        # G^{-1} z = -((-G)^{-1} z) when the factor is of -G (pure downdate).
        return -y if factor.negate else y
    assert factor.lu is not None  # exactly one of cho/lu is set
    try:
        return linalg.lu_solve(factor.lu, z)
    except np.linalg.LinAlgError:
        raise NewtonStepError("Failed to solve Gy = z") from None
    except ValueError:
        raise NewtonStepError(
            _diagnose_nonfinite(z, "Woodbury right-hand side")
        ) from None


def _solve_woodbury(
    G: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64] | None,
    check_inertia: bool = True,
) -> npt.NDArray[np.float64]:
    """Solve G y = z, choosing the factorization from the signs of d.

    Composes ``_factor_woodbury`` and ``_apply_woodbury`` for a single right-hand side.

    Parameters
    ----------
     G : npt.NDArray[np.float64]
        The symmetric Woodbury matrix, of shape (p, p).
     z : npt.NDArray[np.float64]
        Right-hand side, of length p or of shape (p, q).
     d : npt.NDArray[np.float64] | None
        The weights d of the rank-p term, or their signs; ``None`` means all positive.
     check_inertia : bool
        When True and ``d`` has mixed signs, verify the inertia of G before factoring.

    Returns
    -------
     y : npt.NDArray[np.float64]
        The solution G^{-1} z.

    Raises
    ------
     NewtonStepError
        Propagated from ``_factor_woodbury`` and ``_apply_woodbury``.
     ValueError
        Propagated from ``_apply_woodbury`` for a non-finite ``z`` in the Cholesky case.

    """
    return _apply_woodbury(_factor_woodbury(G, d, check_inertia=check_inertia), z)


def multiply_block_plus_one(
    y: npt.NDArray[np.float64],
    A12: npt.NDArray[np.float64],
    A22: float,
    A11_multiply: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Multiply by a matrix with a block-plus-one structure.

    Computes H y for
             _              _
            |   A11    A12   |
       H =  |                |,
            |_  A12^T  A22  _|
    where A11 is M × M with structure that makes A11 z cheap to compute, A12 is a vector
    of length M, and A22 is a scalar.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column. Must have M + 1 rows.
     A12 : npt.NDArray[np.float64]
        Last column of H above the diagonal, of length M.
     A22 : float
        The bottom-right element of H.
     A11_multiply : Callable[..., npt.NDArray[np.float64]]
        Computes A11 z, called as ``A11_multiply(z, **kwargs)``; must accept a vector or
        a matrix.
     **kwargs : Any
        Extra arguments passed to ``A11_multiply``.

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``A12`` is not 1D, ``y`` does not have M + 1 rows, or ``y`` is not 1D or
        2D.

    """
    if A12.ndim != 1:
        raise ValueError("Dimension mismatch: A12 must be a 1D array.")

    M = A12.shape[0]
    if y.shape[0] != M + 1:
        raise ValueError(
            "Dimension mismatch: y must have M + 1 rows, where M = len(A12)."
        )

    if y.ndim == 1:
        y1 = y[:M]
        y2 = y[M]
    elif y.ndim == 2:
        y1 = y[:M, :]
        y2 = y[M, :]
    else:
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    z = np.zeros_like(y)
    if y.ndim == 1:
        z[:M] = A11_multiply(y1, **kwargs) + A12 * y2
        z[M] = np.dot(A12, y1) + A22 * y2
    else:
        z[:M, :] = A11_multiply(y1, **kwargs) + np.outer(A12, y2)
        z[M, :] = A12.T @ y1 + A22 * y2
    return z


def solve_block_plus_one(
    b: npt.NDArray[np.float64],
    A12: npt.NDArray[np.float64],
    A22: float,
    A11_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve a block-plus-one system by its scalar Schur complement.

    Solves H x = b for
             _              _
            |   A11    A12   |
       H =  |                |,
            |_  A12^T  A22  _|
    where A11 is M × M with structure that makes A11 y = c cheap to solve, A12 is a
    vector of length M, and A22 is a scalar.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column. Must have M + 1 rows.
     A12 : npt.NDArray[np.float64]
        Last column of H above the diagonal, of length M.
     A22 : float
        The bottom-right element of H.
     A11_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A11 y = c, called as ``A11_solve(c, **kwargs)``; must accept a vector or
        a matrix.
     **kwargs : Any
        Extra arguments passed to ``A11_solve``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``A12`` is not 1D, ``b`` does not have M + 1 rows, or ``b`` is not 1D or
        2D.
     NewtonStepError
        When the Schur complement s is not positive, so H is not positive definite.
        Errors from ``A11_solve`` propagate unchanged.

    Notes
    -----
    Let t be the cost of one solve with A11, and q the number of right-hand sides. Split
    b into its first M rows b1 and last row b2, and x likewise into x1 and x2.
    1. Form A12' = A11^{-1} A12 (t flops) and b1' = A11^{-1} b1 (at most q t flops; a
       batched solve may be cheaper).
    2. Form the scalar Schur complement
          s = A22 - A12^T A12'
       (2M flops). When H and A11 are positive definite, s > 0.
    3. Compute x2 = (b2 - A12^T b1') / s (2Mq flops and q divisions); x2 is a scalar
       or a vector of length q.
    4. Compute x1 = b1' - A12' x2 (2Mq flops).
    In total, that is (q + 1) t + 2M (2q + 1) + q flops.

    """
    if A12.ndim != 1:
        raise ValueError("Dimension mismatch")

    M = A12.shape[0]
    if b.shape[0] != M + 1:
        raise ValueError("Dimension mismatch")

    if b.ndim == 1:
        b1 = b[:M]
        b2 = b[M]
    elif b.ndim == 2:
        b1 = b[:M, :]
        b2 = b[M, :]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    A12_prime = A11_solve(A12, **kwargs)
    b1_prime = A11_solve(b1, **kwargs)
    s = A22 - np.dot(A12, A12_prime)
    if s <= 0.0:
        raise NewtonStepError("H is not positive definite")

    # Calculate x
    x = np.zeros_like(b)
    if b.ndim == 1:
        x[M] = (b2 - np.dot(A12, b1_prime)) / s
        x[0:M] = b1_prime - A12_prime * x[M]
        return x

    x[M, :] = (b2 - A12.T @ b1_prime) / s
    x[0:M, :] = b1_prime - np.outer(A12_prime, x[M, :])
    return x


def multiply_block(
    y: npt.NDArray[np.float64],
    A12: npt.NDArray[np.float64],
    A22: npt.NDArray[np.float64],
    A11_multiply: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Multiply by a symmetric 2 × 2 block matrix with a structured leading block.

    Computes H y for
             _              _
            |   A11    A12   |
       H =  |                |,
            |_  A12^T  A22  _|
    where A11 is M × M with structure that makes A11 z cheap to compute, A12 is M × p,
    and A22 is p × p, with p ≥ 2 (use ``multiply_block_plus_one`` when p = 1).

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column. Must have M + p rows.
     A12 : npt.NDArray[np.float64]
        Off-diagonal block of H, of shape (M, p).
     A22 : npt.NDArray[np.float64]
        Lower-right block of H, of shape (p, p).
     A11_multiply : Callable[..., npt.NDArray[np.float64]]
        Computes A11 z, called as ``A11_multiply(z, **kwargs)``; must accept a vector or
        a matrix.
     **kwargs : Any
        Extra arguments passed to ``A11_multiply``.

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``A12`` is not a matrix with at least two columns, ``A22`` is not p × p,
        ``y`` does not have M + p rows, or ``y`` is not 1D or 2D.

    """
    if A12.ndim <= 1 or A12.shape[1] <= 1:
        raise ValueError("Please use `multiply_block_plus_one` for this.")

    if A12.ndim > 2:
        raise ValueError("Dimension mismatch: A12 should be a matrix.")

    M, p = A12.shape
    if A22.ndim != 2 or not all(s == p for s in A22.shape):
        raise ValueError(f"Dimension mismatch: {A22.shape=:}; expected ({p}, {p}).")

    if y.shape[0] != M + p:
        raise ValueError(f"Dimension mismatch: {y.shape[0]=:}; expected {M + p}.")

    if y.ndim == 1:
        y1 = y[:M]
        y2 = y[M:]
    elif y.ndim == 2:
        y1 = y[:M, :]
        y2 = y[M:, :]
    else:
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    z = np.zeros_like(y)
    z[:M] = A11_multiply(y1, **kwargs) + A12 @ y2
    z[M:] = A12.T @ y1 + A22 @ y2
    return z


def multiply_arrow_sparsity_pattern(
    y: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
    theta: float,
) -> npt.NDArray[np.float64]:
    """Multiply by a matrix with an arrow sparsity pattern.

    Computes H y for
             _               _
            |  diag(η)    ζ   |
       H =  |                 |,
            |_   ζ^T      θ  _|
    where η and ζ are vectors of length M and θ is a scalar, in linear time.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column. Must have M + 1 rows.
     eta : npt.NDArray[np.float64]
        Diagonal η of the upper-left block of H, of length M.
     zeta : npt.NDArray[np.float64]
        Last column ζ of H above the diagonal, of length M.
     theta : float
        The bottom-right element θ of H.

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``eta`` and ``zeta`` differ in shape, ``y`` does not have M + 1 rows, or
        ``y`` is not 1D or 2D.

    Notes
    -----
    For a vector y = (y1, y2), with y1 of length M and y2 a scalar,
       z[:M] = η ∘ y1 + ζ y2,
       z[M]  = ζ^T y1 + θ y2,
    which takes 3M + 1 multiplications and 2M additions, or 5M + 1 flops.

    """
    if eta.shape != zeta.shape:
        raise ValueError("Dimension mismatch: eta and zeta had different dimensions.")

    M = eta.shape[0]
    if y.shape[0] != M + 1:
        raise ValueError(
            "Dimension mismatch: y must have M + 1 rows, where M = len(eta)."
        )

    if y.ndim == 1:
        y1 = y[:M]
        y2 = y[M]
    elif y.ndim == 2:
        y1 = y[:M, :]
        y2 = y[M, :]
    else:
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    z = np.zeros_like(y)
    if y.ndim == 1:
        z[:M] = eta * y1 + zeta * y2
        z[M] = np.dot(zeta, y1) + theta * y2
        return z

    z[:M, :] = eta[:, np.newaxis] * y1 + np.outer(zeta, y2)
    z[M, :] = zeta.T @ y1 + theta * y2
    return z


def solve_with_schur(
    b: npt.NDArray[np.float64],
    A12: npt.NDArray[np.float64],
    A22: npt.NDArray[np.float64],
    A11_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve a symmetric 2 × 2 block system by its Schur complement.

    Solves H x = b for
             _              _
            |   A11    A12   |
       H =  |                |,
            |_  A12^T  A22  _|
    where A11 is M × M with structure that makes A11 y = c cheap to solve, A12 is M × p,
    and A22 is p × p, with p ≥ 2 (use ``solve_block_plus_one`` when p = 1).

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column. Must have M + p rows.
     A12 : npt.NDArray[np.float64]
        Off-diagonal block of H, of shape (M, p).
     A22 : npt.NDArray[np.float64]
        Lower-right block of H, of shape (p, p).
     A11_solve : Callable[..., npt.NDArray[np.float64]]
        Solves A11 y = c, called as ``A11_solve(c, **kwargs)``; must accept a vector or
        a matrix.
     **kwargs : Any
        Extra arguments passed to ``A11_solve``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``A12`` is not a matrix with at least two columns, ``A22`` is not p × p,
        ``b`` does not have M + p rows, or ``b`` is not 1D or 2D. Also propagated from
        ``scipy.linalg`` when the Schur complement is finite but the right-hand side is
        not.
     NewtonStepError
        When the Schur complement is not positive definite or is non-finite, so H is not
        positive definite. Errors from ``A11_solve`` propagate unchanged.

    Notes
    -----
    Let t be the cost of one solve with A11, and q the number of right-hand sides. Split
    b into its first M rows b1 and last p rows b2, and x likewise into x1 and x2.
    1. Form A12' = A11^{-1} A12 (at most p t flops; a batched solve may be cheaper) and
       b1' = A11^{-1} b1 (at most q t flops).
    2. Form the p × p Schur complement
          S = A22 - A12^T A12'
       (2Mp^2 flops), symmetrized to remove rounding skew. When H and A11 are positive
       definite, so is S.
    3. Solve S x2 = b2 - A12^T b1': forming the right-hand sides takes 2Mpq flops,
       the Cholesky factorization of S p^3 / 3, and the triangular solves 2p^2 q. x2 is
       p × q.
    4. Compute x1 = b1' - A12' x2 (2Mpq flops).
    In total, that is
       (p + q) t + 2Mp (p + 2q) + p^3 / 3 + 2p^2 q
    flops.

    """
    if A12.ndim <= 1 or A12.shape[1] <= 1:
        raise ValueError("Please use `solve_block_plus_one` for this.")

    if A12.ndim > 2:
        raise ValueError("Dimension mismatch: A12 should be a matrix.")

    M, p = A12.shape
    if A22.ndim != 2 or not all(s == p for s in A22.shape):
        raise ValueError(f"Dimension mismatch: {A22.shape=:}; expected ({p}, {p}).")

    if b.shape[0] != M + p:
        raise ValueError(f"Dimension mismatch: {b.shape[0]=:}; expected {M + p}.")

    if b.ndim == 1:
        b1 = b[:M]
        b2 = b[M:]
    elif b.ndim == 2:
        b1 = b[:M, :]
        b2 = b[M:, :]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    A12_prime = A11_solve(A12, **kwargs)
    b1_prime = A11_solve(b1, **kwargs)
    S = A22 - A12.T @ A12_prime
    # Symmetric cleanup: A12.T @ A12_prime is mathematically symmetric, but
    # floating-point rounding adds a small skew that propagates into Cholesky as a
    # spurious negative-eigenvalue artifact when S is near-singular.
    S = 0.5 * (S + S.T)

    try:
        c, lower = linalg.cho_factor(S, lower=True)
        x2 = linalg.cho_solve((c, lower), b2 - A12.T @ b1_prime)
    except np.linalg.LinAlgError:
        raise NewtonStepError("H is not positive definite") from None
    except ValueError:
        # scipy's check_finite raises ValueError (not LinAlgError) on inf/NaN; a
        # non-finite Schur complement means the Newton step is unusable. Mirrors
        # ``_solve_schur_complement`` and ``solve_banded``.
        if not np.all(np.isfinite(S)):
            raise NewtonStepError(_diagnose_nonfinite(S, "Schur complement")) from None
        raise

    x1 = b1_prime - A12_prime @ x2
    if b.ndim == 1:
        return np.concatenate([x1, x2])

    return np.vstack([x1, x2])


def solve_arrow_sparsity_pattern(
    b: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    zeta: npt.NDArray[np.float64],
    theta: float,
) -> npt.NDArray[np.float64]:
    """Solve a system with an arrow sparsity pattern in linear time.

    Solves H x = b for
             _               _
            |  diag(η)    ζ   |
       H =  |                 |,
            |_   ζ^T      θ  _|
    where η and ζ are vectors of length M and θ is a scalar.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column. Must have M + 1 rows.
     eta : npt.NDArray[np.float64]
        Diagonal η of the upper-left block of H, of length M.
     zeta : npt.NDArray[np.float64]
        Last column ζ of H above the diagonal, of length M.
     theta : float
        The bottom-right element θ of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     NewtonStepError
        When some η_i ≤ 0 (checked first, before the shapes), or when the Schur residual
        ψ^2 (see Notes) is not positive to within a relative tolerance of 10^-12, so H
        is not positive definite.
     ValueError
        When ``eta`` and ``zeta`` differ in shape, ``b`` does not have M + 1 rows, or
        ``b`` is not 1D or 2D.

    Notes
    -----
    The Cholesky factorization H = L L^T is available in closed form:
             _                               _
            |   diag(η)^{1/2}            0    |
       L =  |                                 |
            |_  ζ^T diag(η)^{-1/2}       ψ   _|
    with
       ψ^2 = θ - ζ^T diag(η)^{-1} ζ.
    Write b = (b1, b2) and x = (x1, x2), with b1 and x1 of length M. Forward
    substitution in L y = b gives
       y1 = diag(η)^{-1/2} b1,   y2 = (b2 - ζ^T diag(η)^{-1} b1) / ψ,
    and back substitution in L^T x = y gives
       x2 = y2 / ψ = (b2 - (diag(η)^{-1} ζ)^T b1) / ψ^2,
       x1 = diag(η)^{-1} b1 - x2 diag(η)^{-1} ζ.
    So no square root is needed. Forming diag(η)^{-1} ζ takes M divisions and ψ^2
    another M multiplications and M additions (summed with ``math.fsum``, since θ and
    the inner product nearly cancel as the barrier term shrinks); x2 takes M
    multiplications, M additions, and one division; and x1 takes M divisions, M
    multiplications, and M additions. In total, for a vector b, that is 2M + 1
    divisions, 3M multiplications, and 3M additions, or 8M + 1 flops.

    """
    if not np.all(eta > 0):
        raise NewtonStepError("Hessian is not strictly positive definite.")

    if eta.shape != zeta.shape:
        raise ValueError("Dimension mismatch: eta and zeta had different dimensions.")

    M = eta.shape[0]
    if b.shape[0] != M + 1:
        raise ValueError(
            "Dimension mismatch: b must have M + 1 rows, where M = len(eta)."
        )

    if b.ndim == 1:
        b1 = b[:M]
        b2 = b[M]
        b1_prime = b1 / eta
    elif b.ndim == 2:
        b1 = b[:M, :]
        b2 = b[M, :]
        b1_prime = b1 / eta[:, np.newaxis]
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    # Calculate diag(eta)^{-1} * zeta and psi^2.
    diag_eta_inverse_dot_zeta = zeta / eta
    # Use math.fsum (compensated summation) for the inner product to avoid silent loss
    # of leading digits when theta and the dot product nearly cancel (the natural
    # near-singular regime as the IPM barrier shrinks).
    psi_dot = math.fsum((diag_eta_inverse_dot_zeta * zeta).tolist())
    psi_squared = theta - psi_dot
    # Relative guard: a residual that is positive but smaller than the rounding noise of
    # the cancellation is numerically indistinguishable from a singular Hessian — reject
    # it rather than dividing by it later.
    tol = 1e-12 * (abs(theta) + abs(psi_dot))
    if psi_squared <= tol:
        raise NewtonStepError("Hessian is not strictly positive definite.")

    # Calculate x
    x = np.zeros_like(b)
    if b.ndim == 1:
        x[M] = (b2 - np.dot(zeta, b1_prime)) / psi_squared
        x[0:M] = b1_prime - diag_eta_inverse_dot_zeta * x[M]
        return x

    x[M, :] = (b2 - zeta.T @ b1_prime) / psi_squared
    x[0:M, :] = b1_prime - np.outer(diag_eta_inverse_dot_zeta, x[M, :])
    return x


def multiply_banded(
    y: npt.NDArray[np.float64],
    ab: npt.NDArray[np.float64],
    lower: bool = False,
) -> npt.NDArray[np.float64]:
    """Multiply by a symmetric banded matrix in compact banded storage.

    Computes H y for a symmetric banded H stored as for
    ``scipy.linalg.cholesky_banded``.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column. Must have n rows.
     ab : npt.NDArray[np.float64]
        Banded storage of the symmetric matrix H, of shape (p + 1, n), where n is the
        order of H and p its bandwidth (the number of superdiagonals, or of subdiagonals
        when ``lower``). With ``lower`` False, ab[i, j] = H[j - p + i, j]; with
        ``lower`` True, ab[i, j] = H[j + i, j]. This is the layout of
        ``scipy.linalg.cholesky_banded``.
     lower : bool
        Whether ``ab`` stores the lower band (True) or the upper band (False).

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``y`` is not 1D or 2D, or its leading dimension does not match n.

    Notes
    -----
    Iterates over the p + 1 stored diagonals, using symmetry for the other triangle, so
    the product costs O(n p) per column rather than O(n^2) for a dense matrix.

    """
    if y.ndim not in (1, 2):
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    n = ab.shape[1]
    if y.shape[0] != n:
        raise ValueError(
            f"Dimension mismatch: y has {y.shape[0]} rows but ab implies n={n}."
        )

    p = ab.shape[0] - 1
    z = np.zeros_like(y)

    if lower:
        # ab[d, :n-d] = d-th subdiagonal of H (= d-th superdiagonal by symmetry)
        for d in range(p + 1):
            diag = ab[d, : n - d]
            if d == 0:
                if y.ndim == 1:
                    z += diag * y
                else:
                    z += diag[:, np.newaxis] * y
            else:
                if y.ndim == 1:
                    z[: n - d] += diag * y[d:]
                    z[d:] += diag * y[: n - d]
                else:
                    z[: n - d] += diag[:, np.newaxis] * y[d:]
                    z[d:] += diag[:, np.newaxis] * y[: n - d]
    else:
        # ab[p-d, d:] = d-th superdiagonal of H
        for d in range(p + 1):
            diag = ab[p - d, d:]
            if d == 0:
                if y.ndim == 1:
                    z += diag * y
                else:
                    z += diag[:, np.newaxis] * y
            else:
                if y.ndim == 1:
                    z[: n - d] += diag * y[d:]
                    z[d:] += diag * y[: n - d]
                else:
                    z[: n - d] += diag[:, np.newaxis] * y[d:]
                    z[d:] += diag[:, np.newaxis] * y[: n - d]

    return z


def solve_banded(
    b: npt.NDArray[np.float64],
    ab: npt.NDArray[np.float64],
    lower: bool = False,
) -> npt.NDArray[np.float64]:
    """Solve a symmetric positive definite banded system by banded Cholesky.

    Solves H x = b for a symmetric positive definite banded H stored as for
    ``scipy.linalg.cholesky_banded``. Composes ``factor_banded`` and
    ``solve_banded_factored``.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column. Must have n rows.
     ab : npt.NDArray[np.float64]
        Banded storage of the symmetric matrix H, of shape (p + 1, n), where n is the
        order of H and p its bandwidth (the number of superdiagonals, or of subdiagonals
        when ``lower``). With ``lower`` False, ab[i, j] = H[j - p + i, j]; with
        ``lower`` True, ab[i, j] = H[j + i, j]. This is the layout of
        ``scipy.linalg.cholesky_banded``. To build the upper form from a dense H,
           ab = np.zeros((p + 1, n))
           for d in range(p + 1):
               ab[p - d, d:] = np.diag(H, d)
        and the lower form,
           ab = np.zeros((p + 1, n))
           for d in range(p + 1):
               ab[d, : n - d] = np.diag(H, -d)
     lower : bool
        Whether ``ab`` stores the lower band (True) or the upper band (False).

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``b`` is not 1D or 2D, or its leading dimension does not match n; also
        propagated from ``solve_banded_factored`` for a non-finite ``b``.
     NewtonStepError
        Propagated from ``factor_banded`` when H is not positive definite or is
        non-finite.

    Notes
    -----
    For a matrix of order n and bandwidth p, the factorization costs O(n p^2) and each
    right-hand side O(n p), against O(n^3) and O(n^2) for a dense matrix.

    """
    if b.ndim not in (1, 2):
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    n = ab.shape[1]
    if b.shape[0] != n:
        raise ValueError(
            f"Dimension mismatch: b has {b.shape[0]} rows but ab implies n={n}."
        )

    return solve_banded_factored(b, factor_banded(ab, lower=lower), lower=lower)


@dataclass(frozen=True)
class _TridiagonalFactor:
    """The L D L^T factorization of a symmetric positive definite tridiagonal matrix.

    Holds the output of LAPACK's ``dpttrf``: the diagonal of D and the subdiagonal of
    the unit lower bidiagonal L, for ``dpttrs``.

    Fields
    ------
     d : npt.NDArray[np.float64]
        The diagonal of D, of length n.
     e : npt.NDArray[np.float64]
        The subdiagonal of L, of length n - 1.

    """

    d: npt.NDArray[np.float64]
    e: npt.NDArray[np.float64]


def factor_banded(
    ab: npt.NDArray[np.float64],
    lower: bool = False,
) -> npt.NDArray[np.float64] | _TridiagonalFactor:
    """Cholesky-factor a symmetric banded matrix for repeated solves.

    Returns the factored band from ``scipy.linalg.cholesky_banded``, so a caller can
    spread the O(n p^2) factorization over several right-hand sides (for example, the
    three solves of one KKT system) through ``solve_banded_factored``. A tridiagonal
    matrix (bandwidth 1) instead gets LAPACK's dedicated L D L^T routine ``dpttrf``,
    which avoids the general banded routine's square roots and loop overhead: about
    three times faster to factor, and four times faster to solve, at n = 100,000.

    Parameters
    ----------
     ab : npt.NDArray[np.float64]
        Banded storage of the symmetric positive definite matrix H, of shape (p + 1, n),
        in the layout of ``scipy.linalg.cholesky_banded`` (see ``solve_banded``).
     lower : bool
        Whether ``ab`` stores the lower band (True) or the upper band (False).

    Returns
    -------
     cb : npt.NDArray[np.float64] | _TridiagonalFactor
        The factored band, for ``solve_banded_factored``.

    Raises
    ------
     NewtonStepError
        When H is not positive definite, or ``ab`` holds non-finite entries, so the
        Hessian is unusable.
     ValueError
        Propagated from ``scipy.linalg.cholesky_banded`` when ``ab`` is finite but
        malformed.

    """
    if ab.ndim == 2 and ab.shape[0] == 2 and ab.shape[1] >= 2:
        return _factor_tridiagonal(ab, lower=lower)
    try:
        return linalg.cholesky_banded(ab, lower=lower)
    except np.linalg.LinAlgError:
        raise NewtonStepError("Hessian is not strictly positive definite.") from None
    except ValueError:
        # ``cholesky_banded`` raises ValueError (not LinAlgError) via its
        # ``check_finite`` guard when ``ab`` holds inf/NaN. Treat a non-finite Hessian
        # like a non-PD one so callers' NewtonStepError fallbacks engage (e.g. the
        # dual's mini-Newton in ``_minimize_lagrangian``) rather than crashing. Mirrors
        # ``_solve_schur_complement``.
        if not np.all(np.isfinite(ab)):
            raise NewtonStepError(_diagnose_nonfinite(ab, "banded Hessian")) from None
        raise


def _factor_tridiagonal(
    ab: npt.NDArray[np.float64],
    lower: bool,
) -> _TridiagonalFactor:
    """Factor a symmetric tridiagonal matrix in banded storage with ``dpttrf``.

    Parameters
    ----------
     ab : npt.NDArray[np.float64]
        Banded storage of shape (2, n), as for ``factor_banded``.
     lower : bool
        Whether ``ab`` stores the lower band (True) or the upper band (False).

    Returns
    -------
     factor : _TridiagonalFactor
        The L D L^T factorization.

    Raises
    ------
     NewtonStepError
        When the matrix is not positive definite or holds non-finite entries.

    """
    if not np.all(np.isfinite(ab)):
        # dpttrf does not check, and would return garbage rather than fail.
        raise NewtonStepError(_diagnose_nonfinite(ab, "banded Hessian"))
    if lower:
        d, e = ab[0], ab[1, :-1]
    else:
        d, e = ab[1], ab[0, 1:]
    d_factor, e_factor, info = lapack.dpttrf(d, e)
    if info != 0:
        raise NewtonStepError("Hessian is not strictly positive definite.")
    return _TridiagonalFactor(d=d_factor, e=e_factor)


def solve_banded_factored(
    b: npt.NDArray[np.float64],
    cb: npt.NDArray[np.float64] | _TridiagonalFactor,
    lower: bool = False,
) -> npt.NDArray[np.float64]:
    """Solve a banded system from a precomputed Cholesky factor.

    Each solve costs O(n p) per column, the factorization having been paid once in
    ``factor_banded``.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column. Must have n rows.
     cb : npt.NDArray[np.float64] | _TridiagonalFactor
        The factored band returned by ``factor_banded``.
     lower : bool
        Whether ``cb`` is the lower (True) or upper (False) factor; must match the value
        passed to ``factor_banded``.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        Propagated from ``scipy.linalg.cho_solve_banded`` when ``b`` holds non-finite
        entries or has a shape incompatible with ``cb``. ``factor_banded`` has already
        validated the factorization, so a failure here reflects a malformed right-hand
        side rather than a non-positive-definite Hessian, and is deliberately not
        translated to ``NewtonStepError``.

    """
    if isinstance(cb, _TridiagonalFactor):
        # Match cho_solve_banded, which rejects a non-finite right-hand side.
        b = np.asarray_chkfinite(b, dtype=np.float64)
        if b.shape[0] != cb.d.shape[0]:
            raise ValueError(
                f"Dimension mismatch: b has {b.shape[0]} rows but the factor has "
                f"order {cb.d.shape[0]}."
            )
        x, info = lapack.dpttrs(cb.d, cb.e, b)
        if info != 0:
            raise ValueError(f"dpttrs failed with info = {info}.")
        return np.asarray(x, dtype=np.float64)
    return linalg.cho_solve_banded((cb, lower), b)


def multiply_block_diagonal(
    y: npt.NDArray[np.float64],
    block_sizes: Sequence[int],
    block_multipliers: Sequence[Callable[..., npt.NDArray[np.float64]]],
) -> npt.NDArray[np.float64]:
    """Multiply by a block-diagonal matrix whose blocks have their own structure.

    Computes H y for
             _                        _
            |  H_0                     |
            |      H_1                 |
       H =  |            ...           |,
            |_                H_{k-1} _|
    where each block H_i may have its own structure and ``block_multipliers[i]``
    computes H_i z.

    Parameters
    ----------
     y : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the product is taken column
        by column.
     block_sizes : Sequence[int]
        Order of each diagonal block; must sum to the number of rows of ``y``.
     block_multipliers : Sequence[Callable[..., npt.NDArray[np.float64]]]
        One callable per block; ``block_multipliers[i](z)`` computes H_i z. Each must
        have its block's parameters already bound (for example, with
        ``functools.partial``).

    Returns
    -------
     z : npt.NDArray[np.float64]
        The product H y, of the same shape as ``y``.

    Raises
    ------
     ValueError
        When ``y`` is not 1D or 2D, ``block_sizes`` and ``block_multipliers`` differ in
        length, or ``block_sizes`` does not sum to the number of rows of ``y``. Errors
        from the block callables propagate unchanged.

    Notes
    -----
    Because H is block diagonal,
       H y = (H_0 y_0, H_1 y_1, ..., H_{k-1} y_{k-1}),
    where y_i is the segment of y for block i. The blocks are multiplied independently,
    so the cost is the sum of the per-block costs, and each block may use a different
    structure (diagonal, arrow, rank-p update, and so on). For example, with block 0
    diagonal and block 1 an arrow,
       mult0 = functools.partial(multiply_diagonal, eta=eta0)
       mult1 = functools.partial(
           multiply_arrow_sparsity_pattern, eta=eta1, zeta=zeta1, theta=theta1
       )
       z = multiply_block_diagonal(y, [M0, M1 + 1], [mult0, mult1])

    """
    if y.ndim not in (1, 2):
        raise ValueError("y must be either a 1D or 2D NumPy array.")

    if len(block_sizes) != len(block_multipliers):
        raise ValueError("block_sizes and block_multipliers must have the same length.")

    N = sum(block_sizes)
    if y.shape[0] != N:
        raise ValueError(
            f"Dimension mismatch: y has {y.shape[0]} rows but block_sizes sum to {N}."
        )

    z = np.zeros_like(y)
    start = 0
    for size, multiplier in zip(block_sizes, block_multipliers, strict=True):
        end = start + size
        z[start:end] = multiplier(y[start:end])
        start = end
    return z


def solve_block_diagonal(
    b: npt.NDArray[np.float64],
    block_sizes: Sequence[int],
    block_solvers: Sequence[Callable[..., npt.NDArray[np.float64]]],
) -> npt.NDArray[np.float64]:
    """Solve a block-diagonal system whose blocks have their own structure.

    Solves H x = b for
             _                        _
            |  H_0                     |
            |      H_1                 |
       H =  |            ...           |,
            |_                H_{k-1} _|
    where each block H_i may have its own structure and ``block_solvers[i]`` solves H_i
    x_i = b_i.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right-hand side, a vector or a matrix; for a matrix, the system is solved for
        each column.
     block_sizes : Sequence[int]
        Order of each diagonal block; must sum to the number of rows of ``b``.
     block_solvers : Sequence[Callable[..., npt.NDArray[np.float64]]]
        One callable per block; ``block_solvers[i](c)`` solves H_i x_i = c. Each must
        have its block's parameters already bound (for example, with
        ``functools.partial``).

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution, of the same shape as ``b``.

    Raises
    ------
     ValueError
        When ``b`` is not 1D or 2D, ``block_sizes`` and ``block_solvers`` differ in
        length, or ``block_sizes`` does not sum to the number of rows of ``b``.
     NewtonStepError
        Propagated from a block solver when that block is not positive definite, is
        singular, or is non-finite.

    Notes
    -----
    Because H is block diagonal, H x = b splits into the k independent systems
       H_i x_i = b_i,   i = 0, ..., k - 1,
    where b_i is the segment of b for block i. The cost is the sum of the per-block
    costs, and each block may use a different structure (diagonal, arrow, rank-p update,
    and so on). For example, with block 0 diagonal and block 1 an arrow,
       solver0 = functools.partial(solve_diagonal, eta=eta0)
       solver1 = functools.partial(
           solve_arrow_sparsity_pattern, eta=eta1, zeta=zeta1, theta=theta1
       )
       x = solve_block_diagonal(b, [M0, M1 + 1], [solver0, solver1])

    """
    if b.ndim not in (1, 2):
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    if len(block_sizes) != len(block_solvers):
        raise ValueError("block_sizes and block_solvers must have the same length.")

    N = sum(block_sizes)
    if b.shape[0] != N:
        raise ValueError(
            f"Dimension mismatch: b has {b.shape[0]} rows but block_sizes sum to {N}."
        )

    x = np.zeros_like(b)
    start = 0
    for size, solver in zip(block_sizes, block_solvers, strict=True):
        end = start + size
        x[start:end] = solver(b[start:end])
        start = end
    return x


# Relative floor for the SVD pseudo-inverse of a rank-deficient A (or Schur
# complement). Singular values at or below ``_SVD_RANK_RTOL * s[0]`` are treated
# as numerical noise; ``tikhonov_filter`` places its damping at this floor so the
# filtered reciprocal is bounded rather than diverging as ``s -> 0``.
_SVD_RANK_RTOL: float = 1e-12


def aspect_ratio_svd(
    matrix: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute the thin SVD of a dense matrix, using QR first for very wide inputs.

    For a matrix with far more columns than rows, it is cheaper to take the QR
    factorization
       matrix^T = Q R
    and then the SVD R = U diag(s) V^T of the small triangular factor R, whose order is
    the row count, than to decompose the matrix directly; then
       matrix = R^T Q^T = V diag(s) (Q U)^T.
    The QR route is taken when the matrix has at least 1,000 columns and at most a tenth
    as many rows as columns; otherwise the direct thin SVD is used. Dense matrices only.

    Parameters
    ----------
     matrix : npt.NDArray[np.float64]
        The matrix to decompose, of shape (p, M).

    Returns
    -------
     U : npt.NDArray[np.float64]
        Left singular vectors, of shape (p, k) with k = min(p, M).
     s : npt.NDArray[np.float64]
        Singular values in descending order, of length k.
     Vh : npt.NDArray[np.float64]
        Right singular vectors as rows, of shape (k, M), so that matrix = U diag(s) Vh.

    """
    if matrix.shape[1] < 1_000 or matrix.shape[0] > 0.1 * matrix.shape[1]:
        return linalg.svd(matrix, full_matrices=False)

    # Compute the QR factorization of matrix^T = Q * R.
    Q, R = linalg.qr(matrix.T, mode="economic")

    # Compute the SVD of R = U * diag(s) * Vh.
    U, s, Vh = linalg.svd(R)

    # SVD of matrix is R^T * Q^T = V * diag(s) * U^T * Q^T
    #                            = V * diag(s) * (Q * U)^T.
    QU = Q @ U
    return Vh.T, s, QU.T


def tikhonov_filter(
    s: npt.NDArray[np.float64],
    power: int = 1,
    rtol: float = _SVD_RANK_RTOL,
) -> tuple[npt.NDArray[np.float64], int]:
    """Compute the damped reciprocal of a singular-value spectrum and its rank.

    Replaces the discontinuous truncated pseudo-inverse (a hard rank cut, then a plain
    reciprocal of s or s^2) with a smooth spectral filter that degrades gracefully
    across the rank boundary:
    - power 1, for the pseudo-inverse of A or of a symmetric positive semidefinite S:
         s / (s^2 + λ^2);
    - power 2, for the inverse of A A^T on its range:
         1 / (s^2 + λ^2);
    where λ = ``rtol`` s_0 sits at the rank-cut floor. Singular values at or below the
    floor are treated as noise and mapped to 0, so a redundant constraint row (a zero
    singular value) is dropped, and a range test can use the truncated left singular
    vectors U[:, :rank]. Surviving singular values are damped, so the filter is bounded
    by 1 / (2 λ) (power 1) or 1 / (2 λ^2) (power 2) instead of exploding when a singular
    value lands just above the floor, the conditioning trap that makes a plain
    pseudo-inverse of a rank-deficient A numerically unstable. Modes with s ≫ λ are
    essentially unchanged (about 1 / s or 1 / s^2), so well-conditioned directions pay
    negligible bias.

    Parameters
    ----------
     s : npt.NDArray[np.float64]
        Singular values in descending order, of length k.
     power : int
        1 for a pseudo-inverse of A or of a symmetric S; 2 for the inverse of A A^T.
     rtol : float
        Relative floor: λ = ``rtol`` s_0.

    Returns
    -------
     filt : npt.NDArray[np.float64]
        The damped reciprocal, of length k, zero at and below the floor (all zero when
        ``s`` is empty or s_0 ≤ 0).
     rank : int
        Number of singular values above ``rtol`` s_0.

    Raises
    ------
     ValueError
        When ``power`` is neither 1 nor 2.

    """
    if power not in (1, 2):
        raise ValueError(f"tikhonov_filter supports power 1 or 2, got {power}.")
    filt = np.zeros_like(s)
    if s.size == 0:
        return filt, 0
    s_max = float(s[0])
    if s_max <= 0.0:
        return filt, 0
    lmbda_squared = (rtol * s_max) ** 2
    keep = s > rtol * s_max
    s_kept = s[keep]
    s_kept_squared = np.square(s_kept)
    if power == 1:
        filt[keep] = s_kept / (s_kept_squared + lmbda_squared)
    else:
        filt[keep] = 1.0 / (s_kept_squared + lmbda_squared)
    return filt, int(np.sum(keep))


def _solve_schur_complement(
    neg_S: npt.NDArray[np.float64],
    neg_c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve the KKT Schur-complement system, falling back to a damped SVD.

    Solves (-S) ν = -c, where -S = A H^{-1} A^T is positive semidefinite. Tries Cholesky
    first, which is fast and valid when -S is positive definite. When -S is rank
    deficient, falls back to a Tikhonov-damped SVD pseudo-inverse (``tikhonov_filter``
    with power 1), but only when the right-hand side lies in the range of -S (the range
    of A when H is positive definite); otherwise the KKT system has no solution.

    Parameters
    ----------
     neg_S : npt.NDArray[np.float64]
        The negated Schur complement -S = A H^{-1} A^T, of shape (p, p).
     neg_c : npt.NDArray[np.float64]
        The negated Schur right-hand side -c, of length p.

    Returns
    -------
     nu : npt.NDArray[np.float64]
        The solution ν, of length p; the damped minimum-norm solution in the
        rank-deficient case.

    Raises
    ------
     NewtonStepError
        When ``neg_S`` is non-finite, or it is rank deficient and ``neg_c`` has a
        component outside its range, relative to 10^-10 max(1, ‖``neg_c``‖).

    Notes
    -----
    The range test compares the residual of projecting ``neg_c`` onto the retained left
    singular vectors with the norm of ``neg_c``, rather than using ``np.allclose``,
    whose absolute tolerances hide precision loss when ‖``neg_c``‖ is far from 1.
    Slack-form Phase I is the typical case: its augmented A has a dominant slack column,
    and cancellation in forming the right-hand side can leave the residual near the
    default tolerances.

    """
    try:
        c, lower = linalg.cho_factor(neg_S, lower=True)
        return linalg.cho_solve((c, lower), neg_c)
    except ValueError:
        if not np.all(np.isfinite(neg_S)):
            raise NewtonStepError("Schur complement contains NaN/Inf") from None
    except np.linalg.LinAlgError:
        pass

    # Rank-deficient Schur complement; fall back to a Tikhonov-damped SVD
    # pseudo-inverse (see ``tikhonov_filter``) rather than a hard-cut ``1 / s``.
    # ``-S`` is symmetric positive semidefinite, so its pseudo-inverse is the
    # ``power == 1`` filter. Damping bounds a near-zero singular value -- from a
    # redundant constraint row, or from cancellation in forming ``-S`` -- so it
    # is smoothed instead of amplified into a corrupt Newton step.
    U, s, Vh = linalg.svd(neg_S, full_matrices=False)
    filt, rank = tikhonov_filter(s, power=1)
    U_r = U[:, 0:rank]
    # Range-test: ``neg_c`` must lie in the column span of ``U_r``. The default
    # ``np.allclose`` tolerances (``rtol=1e-5``, ``atol=1e-8``) silently mask precision
    # loss when ``||neg_c||`` is far from unity — the slack-form Phase I configuration
    # in particular forms ``neg_c = A @ b`` against an augmented ``A`` whose slack
    # column dominates the original entries, and cancellation in that product can leave
    # the residual right at the default-tolerance boundary. Use a residual-vs-RHS-norm
    # relative test.
    neg_c_norm = float(np.linalg.norm(neg_c))
    residual_norm = float(np.linalg.norm(U_r @ (U_r.T @ neg_c) - neg_c))
    if residual_norm > 1e-10 * max(1.0, neg_c_norm):
        raise NewtonStepError(
            "KKT system did not have a solution: right-hand side has a "
            f"component outside the range of A (||residual|| = "
            f"{residual_norm:.3e}, ||neg_c|| = {neg_c_norm:.3e}). This "
            "indicates either that A is rank-deficient, or that "
            "floating-point cancellation in forming the Schur complement "
            "has lost precision (common in slack-form Phase I configurations "
            "where A's slack column dominates the original entries)."
        ) from None

    return Vh.T @ (filt * (U.T @ neg_c))


def solve_kkt_system(
    A: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    hessian_solve: Callable[..., npt.NDArray[np.float64]],
    g_bot: npt.NDArray[np.float64] | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve an equality-constrained KKT system by block elimination.

    Solves
        _        _    _   _     _       _
       |  H   A^T |  |  Δx |   |    g    |
       |          |  |     | = |         |
       |_ A    0 _|  |_ ν _|   |_ g_bot _|
    where H is the Hessian, A the p × M equality-constraint matrix, and g_bot defaults
    to 0. When systems in H can be solved in O(M) time, the Schur complement gives Δx in
    O(p^3 + p^2 M) time.

    Parameters
    ----------
     A : npt.NDArray[np.float64]
        The equality-constraint matrix, of shape (p, M).
     g : npt.NDArray[np.float64]
        Upper (gradient) block of the right-hand side, of length M.
     hessian_solve : Callable[..., npt.NDArray[np.float64]]
        Solves H x = y, called as ``hessian_solve(y, **kwargs)``; must accept a vector
        or a matrix.
     g_bot : npt.NDArray[np.float64] | None
        Lower (equality-residual) block of the right-hand side, of length p. ``None``
        means zero, the equality-feasible Newton step. A nonzero ``g_bot`` arises in
        primal-dual methods, where the iterate need not satisfy A x = b mid-iteration.
     **kwargs : Any
        Extra arguments passed to ``hessian_solve``.

    Returns
    -------
     delta_x : npt.NDArray[np.float64]
        The primal step Δx, of length M.
     nu : npt.NDArray[np.float64]
        The dual variable ν, of length p.

    Raises
    ------
     ValueError
        When ``g`` does not have one entry per column of ``A``, or ``g_bot`` is given
        without one entry per row of ``A``.
     NewtonStepError
        Propagated from ``hessian_solve`` (H not positive definite, singular, or
        non-finite) or from ``_solve_schur_complement`` (Schur complement non-finite, or
        the KKT system has no solution).

    Notes
    -----
    Follows the block elimination of (Boyd and Vandenberghe, 2004, Algorithm C.4):
    1. Form B = H^{-1} A^T and b = H^{-1} g: p + 1 solves with H, O((p + 1) M) time when
       each takes O(M).
    2. Form -S = A B, which takes p^2 dot products of length M, O(p^2 M) time, and
       -c = A b - g_bot, O(p M) time.
    3. Solve (-S) ν = -c by Cholesky in O(p^3) time (-S is positive definite when A has
       full row rank). When A is rank deficient, -S is too; the system can usually still
       be solved by SVD, which is slower, so Cholesky is tried first and SVD is the
       fallback (see ``_solve_schur_complement``).
    4. Solve H Δx = g - A^T ν: O(p M) time to form the right-hand side and O(M) for the
       solve.
    In total that is O(p^2 M + p^3), dominated by forming S when p ≤ M.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """
    p, M = A.shape
    if len(g) != M:
        raise ValueError(
            "Dimension mismatch: g should have one entry for each column of A."
        )
    if g_bot is not None and len(g_bot) != p:
        raise ValueError(
            "Dimension mismatch: g_bot should have one entry for each row of A."
        )

    # Step 1: form B = H^{-1} * A^T and b = H^{-1} * g
    B = hessian_solve(A.T, **kwargs)
    b = hessian_solve(g, **kwargs)

    # Step 2: form -S = A * B and -c = A * b - g_bot (g_bot defaults to 0)
    neg_S = A @ B
    neg_c = A @ b
    if g_bot is not None:
        neg_c = neg_c - g_bot

    # Step 3: Solve -S * nu = -c
    nu = _solve_schur_complement(neg_S, neg_c)

    # Step 4: Solve H * delta_x = g - A^T * nu
    delta_x = hessian_solve(g - (A.T @ nu), **kwargs)

    return delta_x, nu
