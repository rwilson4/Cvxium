"""Numerical linear algebra routines."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import NewtonStepError


def solve_diagonal(
    b: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal,
       H = diag(eta).
    Because of this structure, we can solve the system in linear time.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta : npt.NDArray[np.float64]
        Diagonal elements of the upper left block of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

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
    """Solve H * x = b.

    Solves a linear system of equations where H is diagonal,
       H = diag(eta).
    Because of this structure, we can solve the system in linear time.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta_inverse : npt.NDArray[np.float64]
        One divided by diagonal elements of the upper left block of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

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


def solve_rank_one_update(
    b: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    A_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H = A + kappa * kappa^T, where A has some
    special structure that makes it easy to solve A * y = c, and kappa is a vector.
    Thus, H is a rank-one update to A.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     kappa : npt.NDArray[np.float64]
        Rank-one component of H.
     A_solve : Callable
        A function that solves A * y = c. The first argument to A_solve will be c.
        Additional arguments will be passed via **kwargs. A_solve should be able to
        accept multiple right-hand-sides.
     kwargs
        Extra arguments to pass to A_solve.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    Let H = A + kappa * kappa^T, where kappa is a vector of length M. We can solve
    H * x = b in O(2*t + 6*M) time, where t is the time needed to solve A*y = c, as
    follows:
       1. Solve A * x' = b (t time).
       2. Solve A * xi = kappa (t time).
       3. Calculate x as x' - ((kappa^T * x') / (1 + kappa^T * xi)) * xi. This is 2 dot
          products (each involving M multiplies and M-1 adds) plus M multiplies and M
          subtractions, or O(3M) multiplies plus O(3M) adds, plus a single add and a
          division.
    In total that's 2*t, 3M multiplies, and 3M adds.


    """
    if b.ndim == 1:
        if b.shape != kappa.shape:
            raise ValueError("b and kappa must have the same length.")
    elif b.ndim == 2:
        if b.shape[0] != kappa.shape[0]:
            raise ValueError("Number of rows in beta must match length of kappa.")
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    x_prime = A_solve(b, **kwargs)
    xi = A_solve(kappa, **kwargs)
    den = 1.0 / (1.0 + np.dot(kappa, xi))

    if b.ndim == 1:
        return x_prime - (np.dot(kappa, x_prime) * den) * xi
    elif b.ndim == 2:
        return x_prime - den * np.outer(xi, kappa.T @ x_prime)
    else:
        raise ValueError("b must be either a 1D or 2D NumPy array.")


def solve_rank_p_update(
    b: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    A_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H = A + kappa * kappa^T, where A has some
    special structure that makes it easy to solve A * y = c, and kappa is rank p. Thus,
    H is a rank-p update to A.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     kappa : npt.NDArray[np.float64]
        Rank-p component of H.
     A_solve : Callable
        A function that solves A * y = c. The first argument to A_solve will be c.
        Additional arguments will be passed via **kwargs. A_solve should be able to
        accept multiple right-hand-sides.
     kwargs
        Extra arguments to pass to A_solve.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    Let H = A + kappa * kappa^T, where kappa is an M-by-p matrix. We can solve H * x = b
    in O((p + q) * t + 2 * M * p * (p + 2 * q)) time, where t is the number of flops
    needed to solve A*y = c, as follows:
       1. Solve A * xi = kappa. This is p solves with A (at most p * t flops; by passing
          multiple right hand sides it may be less than p * t flops, since we avoid
          duplicating calculations unnecessarily). xi is M-by-p.
       2. Calculate G = I + kappa^T * xi (p^2 * M multiplies and p^2 * (M - 1) + p adds,
          or O(2 * M * p^2) flops). G is p-by-p.
       3. Compute the Cholesky factorization of G (1/3 * p^3 flops).
       4. Solve A * x' = b (When there are q right-hand-sides, this is at most q * t
          flops). x' is M-by-q.
       5. Calculate z = kappa^T * x' (p * q * M multiplies and p * q * (M - 1) adds or
          2 * M * p * q flops). z is p-by-q.
       6. Solve G * y = z, using the Cholesky factorization. It takes 2 * p^2 * q time
          to solve for y. y is p-by-q.
       7. Calculate x as x' - xi @ y. This is M * p * q multiplies and M * p * q adds or
          2 * M * p * q flops.
    In total that's (p + q) * t + 2 * M * p^2 + 4 * M * p * q + (1/3) * p^3 + 2 * p^2 * q
    flops, or (p + q) * t + 2 * M * p * (p + 2 * q) + p^2 * ((1/3) * p + 2 * q) or
    (p + q) * t + 2 * M * p * (p + 2 * q) when p and q << M.

    """
    if b.ndim not in (1, 2):
        raise ValueError("b must be either a 1D or 2D NumPy array.")

    if b.shape[0] != kappa.shape[0]:
        raise ValueError("Number of rows in beta must match length of kappa.")

    # xi is M-by-p
    xi = A_solve(kappa, **kwargs)

    # G is p-by-p
    G = np.eye(kappa.shape[1]) + kappa.T @ xi
    try:
        c, lower = linalg.cho_factor(G, lower=True)
    except np.linalg.LinAlgError:
        raise NewtonStepError("H is not positive definite") from None

    # q RHS -> x_prime is M-by-q
    x_prime = A_solve(b, **kwargs)
    z = kappa.T @ x_prime  # p-by-q

    # y is p-by-q
    y = linalg.cho_solve((c, lower), z)
    return x_prime - xi @ y


def solve_block_plus_one(
    b: npt.NDArray[np.float64],
    A12: npt.NDArray[np.float64],
    A22: float,
    A11_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H has a block structure:
         _              _
        |    A11    A12  |
    H = |                |.
        |_  A12^T   A22 _|

    We assume that A11 has some special structure that allows us to solve A11 * y = c
    efficiencly; that A12 is a vector, and A22 a scalar.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     A12 : npt.NDArray[np.float64]
        Last row/column of H, other than the bottom right element.
     A22 : float
        The bottom right element of H.
     A11_solve : Callable
        A function that solves A11 * y = c. The first argument to A11_solve will be c.
        Additional arguments will be passed via **kwargs. A11_solve should be able to
        accept multiple right-hand-sides.
     kwargs
        Extra arguments to pass to A11_solve.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    Uses the Schur complement to solve the system efficiently. Assume that it takes t
    flops to solve A11 * y = c. Assume A11 is square of dimension M, so that H is square
    of dimension M + 1. Let b1 be the first M rows of b and b2 the last row. Assume
    there are q right-hand-sides, so that b has q columns. Let x1 be the first M rows of
    x, and x2 the last row.

    First form A12' = A11^{-1} A12. This involves 1 solve with A11, or t flops. Next
    form b1' = A11^{-1} b1, which takes q * t flops. (Passing multiple right hand sides
    avoids duplicate calculations, so it may be less than q * t flops.)

    Form the Schur complement, s = A22 - A12^T * A12', which is a scalar. Forming s
    involves a dot products of length M, or 2 * M flops. If H and A11 are both positive
    definite, then s > 0.

    Calculate x2 = (b2 - A12^T * b1') / s. It takes 2 * M * q flops to form the q right
    hand sides and then q divisions by s. x2 is either scalar of a vector of length q.

    Calculate x1 = b1' - A12' * x2 in 2 * M * q flops. Concatenate x1 and x2 as the
    return value. In total, that's (q + 1) * t + 2 * M * (2 * q + 1) flops.

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


def solve_with_schur(
    b: npt.NDArray[np.float64],
    A12: npt.NDArray[np.float64],
    A22: npt.NDArray[np.float64],
    A11_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Solve H * x = b.

    Solves a linear system of equations where H has a block structure:
         _                 _
        |    A11      A12   |
    H = |                   |.
        |_  A12^T     A22  _|

    We assume that A11 has some special structure that allows us to solve A11 * y = c
    efficiencly.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     A12, A22 : npt.NDArray[np.float64]
        Components of H.
     A11_solve : Callable
        A function that solves A11 * y = c. The first argument to A11_solve will be c.
        Additional arguments will be passed via **kwargs. A11_solve should be able to
        accept multiple right-hand-sides.
     kwargs
        Extra arguments to pass to A11_solve.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    Uses the Schur complement to solve the system efficiently. Assume that it takes t
    flops to solve A11 * y = c. Assume A11 is square of dimension M, and that A12 has p
    columns, so that H is square of dimension M + p. Let b1 be the first M rows of b and
    b2 the last p. Assume there are q right-hand-sides, so that b has q columns. Let x1
    be the first M rows of x, and x2 the last row.

    First form A12' = A11^{-1} A12. This involves p solves with A11, or p * t flops.
    (Passing multiple right hand sides avoids duplicate calculations, so it may be less
    than p * t flops.) Next form b1' = A11^{-1} b1, which takes q * t flops.

    Form the Schur complement, S = A22 - A12^T * A12', which is p-by-p. Forming S
    involves p^2 dot products of length M, or 2 * M * p^2 flops. If H and A11 are both
    positive definite, then so is S.

    Determine x2 by solving S * x2 = b2 - A12^T * b1'. It takes 2 * M * p * q flops to
    form the q right hand sides, plus (1/3) * p^3 flops to compute the Cholesky
    decomposition of S, plus 2 * p^2 * q flops to solve the q right hand sides. x2 is
    p-by-q.

    Calculate x1 = b1' - A12' * x2 in 2 * M * p * q flops. Concatenate x1 and x2 as the
    return value. In total, that's (p + q) * t + 2 * M * p * (p + 2 * q) + (1/3) * p^3
    + 2 * p^2 * q


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

    try:
        c, lower = linalg.cho_factor(S, lower=True)
        x2 = linalg.cho_solve((c, lower), b2 - A12.T @ b1_prime)
    except np.linalg.LinAlgError:
        raise NewtonStepError("H is not positive definite") from None

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
    """Solve H * x = b.

    Solves a linear system of equations where H has an arrow sparsity pattern:
         _                 _
        |  diag(eta)  zeta  |
    H = |                   |
        |_  zeta^T   theta _|

    Because of this structure, we can solve the system in linear time. See Notes for
    more details.

    Parameters
    ----------
     b : npt.NDArray[np.float64]
        Right hand side. Can be either a vector or a matrix, in which case we solve the
        system for each column of b.
     eta : npt.NDArray[np.float64]
        Diagonal elements of the upper left block of H.
     zeta : npt.NDArray[np.float64]
        Last row/column of H, other than the bottom right element.
     theta : float
        The bottom right element of H.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The solution.

    Notes
    -----
    In this case, we can calculate the Cholesky factorization of H analytically:
    H = L * L^T, where
         _                                                                          _
        |        diag(eta^{1/2})                         0                           |
    L = |                                                                            |
        |_  zeta^T * diag(eta^{-1/2})  sqrt(theta - zeta^T * diag(eta^{-1}) * zeta) _|.

    Call that bottom right element psi. Let y = L^T * x. THen H * x = b is equivalent to
    L * y = b. Let eta and zeta be of length M (and x, b, and y are therefore of length
    M + 1). Then the first M entries of y are simply b[0:M] / np.sqrt(eta). The last
    element of y satisfies:
        b[M] = zeta^T * diag(eta^{-1/2}) * y[0:M] + psi * y[M]
             = zeta^T * diag(eta^{-1}) * b[0:M] + psi * y[M], or
        y[M] = (1 / psi) * (b[M] - zeta^T * diag(eta^{-1}) * b[0:M]).

    Next we solve L^T * x = y. Starting from the last element, we have:
       psi * x[M] = y[M], or x[M] = y[M] / psi.
    The remaining equations are of the form:
       sqrt(eta[i]) * x[i] + (zeta[i] / sqrt(eta[i])) * x[M] = y[i],
    or x[i] = y[i] / sqrt(eta[i]) - (zeta[i] / eta[i]) * x[M].

    Since
        y[i] / sqrt(eta[i]) = b[i] / eta[i], and
                 y[M] / psi = (b[M] - zeta^T * diag(eta^{-1}) * b[0:M]) / psi_squared,
    we have:
          x[M] = (b[M] - zeta^T * diag(eta^{-1}) * b[0:M]) / psi_squared
               = (b[M] - (diag(eta^{-1}) * zeta)^T * b[0:M]) / psi_squared, and
        x[0:M] = b[0:M] / eta - x[M] * (diag(eta^{-1}) * zeta).

    It takes M divides to calculate diag(eta^{-1}) * zeta, then M multiplies plus M adds
    to calculate psi_squared. Then M multiplies, M adds, and one division to calculate
    x[M]. Then M divides, M multiplies, and M adds to calculate x[0:M]. In total, that's
    2*M + 1 divides, 3*M multiplies, and 3*M adds, or 8*M + 1 flops.

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

    # Calculate diag(eta)^{-1} * zeta and psi^2
    diag_eta_inverse_dot_zeta = zeta / eta
    psi_squared = theta - np.dot(diag_eta_inverse_dot_zeta, zeta)
    if psi_squared <= 0:
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


def solve_kkt_system(
    A: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    hessian_solve: Callable[..., npt.NDArray[np.float64]],
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a KKT system of equations.

    Parameters
    ----------
     A : p-by-M matrix.
        Parameter.
     g : vector of length M
        Right-hand-side.
     hessian_solve : Callable
        A function that solves H * x = y. The first argument to hessian_solve will be y.
        Additional arguments will be passed via **kwargs.
     kwargs
        Extra arguments to pass to hessian_solve.

    Returns
    -------
     delta_x : vector of length M
        Solution to system. See Notes.
     nu : vector of length p
        Solution to system. See Notes.

    Notes
    -----
    Solves:
           _       _   _       _     _   _
          | H   A^T | | delta_x |   |  g  |
          | A    0  | |   nu    | = |  0  |
           -       -   -       -     -   -
    where H is the Hessian.

    When we can solve systems H * x = y in O(M) time, we can exploit the Schur
    complement and the matrix inversion lemma to calculate delta_x in O(p^3 + p^2*M)
    time, were p is the number of rows in A.

    Per the discussion in Boyd and Vandenberghe (2004), Algorithm C.4 (page
    673):
      1. Form B = H^{-1} * A^T and b = H^{-1} * g. This corresponds to p+1 solves. We
         use `hessian_solve` to solve each system in O(M) time, for O((p+1) * M) time
         total.
      2. Form S = -A * B and c = -A * b. Since A is p-by-M and B is M-by-p, forming S
         involves p^2 dot products of length M, which takes (p^2 * M) time. Forming c
         takes O(p * M) time.
      3. Solve S * nu = c via Cholesky decomposition. (S is negative definite, so we
         instead solve -S * nu = -c.) This takes O(p^3) time.
         a. If A is not full rank, S won't be, either. We can typically still solve the
            system using the Singular Value Decomposition (SVD) instead of the Cholesky
            decomposition. The SVD is slower, so we still at least *try* the Cholesky,
            and if that fails, we fall back to SVD.
      4. Solve H * delta_x = g - A^T * nu. This takes O(p*M) time to form the RHS, then
         O(M) time to compute delta_x.
    In total, that's O(M * p^2 + p^3), the time being dominated by forming S.

    """
    p, M = A.shape
    if len(g) != M:
        raise ValueError(
            "Dimension mismatch: g should have one entry for each column of A."
        )

    # Step 1: form B = H^{-1} * A^T and b = H^{-1} * g
    B = hessian_solve(A.T, **kwargs)
    b = hessian_solve(g, **kwargs)

    # Step 2: form -S = A * B and -c = A * b
    neg_S = A @ B
    neg_c = A @ b

    # Step 3: Solve -S * nu = -c
    try:
        c, lower = linalg.cho_factor(neg_S, lower=True)
        nu = linalg.cho_solve((c, lower), neg_c)
    except np.linalg.LinAlgError:
        # This can happen when A is not full rank; fall back to SVD, which is slower but
        # more numerically stable. To be honest though, in my timing experiments, this
        # is really about the same speed as Cholesky, so consider just always doing SVD.
        U, s, Vh = linalg.svd(neg_S, full_matrices=False)
        rank = int(np.sum(s > 1e-10))
        U_r = U[:, 0:rank]
        if not np.allclose(U_r @ (U_r.T @ neg_c), neg_c):
            raise NewtonStepError(
                "KKT system did not have a solution, because A is not full rank."
            ) from None

        s_inv = np.zeros_like(s)
        s_inv[0:rank] = 1.0 / s[0:rank]
        nu = Vh.T @ (s_inv * (U.T @ neg_c))

    # Step 4: Solve H * delta_x = -grad_ft - A^T * nu
    delta_x = hessian_solve(g - (A.T @ nu), **kwargs)

    return delta_x, nu
