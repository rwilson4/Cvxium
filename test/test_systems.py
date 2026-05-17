"""Test the composable System classes."""

import numpy as np
import numpy.typing as npt
import pytest

from cvxium.exceptions import NewtonStepError
from cvxium.systems import (
    ArrowSystem,
    BandedSystem,
    BlockDiagonalSystem,
    DenseSystem,
    DiagonalSystem,
    KKTSystem,
    LowRankUpdatedSystem,
    System,
)


def _random_spd(M: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Return a random M-by-M symmetric positive-definite matrix."""
    A = rng.standard_normal((M, M))
    return A @ A.T + M * np.eye(M)


def _banded_spd(
    M: int, p: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(dense, ab)`` for a random symmetric banded SPD matrix.

    ``dense`` is the full matrix; ``ab`` is its upper compact band storage.
    """
    H = np.zeros((M, M))
    for d in range(1, p + 1):
        vals = rng.standard_normal(M - d)
        idx = np.arange(M - d)
        H[idx, idx + d] = vals
        H[idx + d, idx] = vals
    # Make strictly diagonally dominant -> symmetric positive definite.
    H[np.diag_indices(M)] = np.abs(H).sum(axis=1) + 1.0
    ab = np.zeros((p + 1, M))
    for d in range(p + 1):
        ab[p - d, d:] = np.diag(H, d)
    return H, ab


def _check_solve_and_multiply(
    system: System, dense: npt.NDArray[np.float64], rng: np.random.Generator
) -> None:
    """Assert ``system`` solves and multiplies consistently with ``dense``."""
    M = system.dimension
    assert dense.shape == (M, M)

    # Vector right-hand side.
    b = rng.standard_normal(M)
    x = system.solve(b)
    np.testing.assert_allclose(dense @ x, b, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(system.multiply(b), dense @ b, rtol=1e-8, atol=1e-8)

    # Matrix (multi-column) right-hand side.
    B = rng.standard_normal((M, 4))
    X = system.solve(B)
    np.testing.assert_allclose(dense @ X, B, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(system.multiply(B), dense @ B, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("seed,M", [(1, 5), (2, 50), (3, 200)])
def test_diagonal_system(seed: int, M: int) -> None:
    """A DiagonalSystem solves and multiplies like ``diag(eta)``."""
    rng = np.random.default_rng(seed)
    eta = rng.random(M) + 1.0
    system = DiagonalSystem(eta)
    assert system.dimension == M
    _check_solve_and_multiply(system, np.diag(eta), rng)


def test_diagonal_system_not_positive_definite() -> None:
    """Solving a non-PD diagonal system raises NewtonStepError."""
    system = DiagonalSystem(np.array([1.0, -2.0, 3.0]))
    with pytest.raises(NewtonStepError):
        system.solve(np.ones(3))


@pytest.mark.parametrize("seed,M,p", [(11, 20, 1), (12, 60, 3), (13, 100, 5)])
def test_banded_system(seed: int, M: int, p: int) -> None:
    """A BandedSystem solves and multiplies like its dense matrix."""
    rng = np.random.default_rng(seed)
    dense, ab = _banded_spd(M, p, rng)
    system = BandedSystem(ab)
    assert system.dimension == M
    _check_solve_and_multiply(system, dense, rng)


@pytest.mark.parametrize("seed,M", [(21, 4), (22, 30), (23, 80)])
def test_dense_system(seed: int, M: int) -> None:
    """A DenseSystem solves and multiplies like its matrix."""
    rng = np.random.default_rng(seed)
    dense = _random_spd(M, rng)
    system = DenseSystem(dense)
    assert system.dimension == M
    _check_solve_and_multiply(system, dense, rng)


def test_dense_system_not_positive_definite() -> None:
    """Solving a non-PD dense system raises NewtonStepError."""
    indefinite = np.array([[1.0, 0.0], [0.0, -1.0]])
    system = DenseSystem(indefinite)
    with pytest.raises(NewtonStepError):
        system.solve(np.ones(2))


@pytest.mark.parametrize("seed,M", [(31, 5), (32, 40), (33, 120)])
def test_low_rank_update_rank_one(seed: int, M: int) -> None:
    """A rank-one update over a diagonal base matches the dense result."""
    rng = np.random.default_rng(seed)
    eta = rng.random(M) + 1.0
    kappa = rng.standard_normal(M)
    system = DiagonalSystem(eta).low_rank_update(kappa)
    dense = np.diag(eta) + np.outer(kappa, kappa)
    assert system.dimension == M
    _check_solve_and_multiply(system, dense, rng)


@pytest.mark.parametrize("seed,M,p", [(41, 10, 3), (42, 60, 8), (43, 150, 5)])
def test_low_rank_update_rank_p(seed: int, M: int, p: int) -> None:
    """A rank-p update over a diagonal base matches the dense result."""
    rng = np.random.default_rng(seed)
    eta = rng.random(M) + 1.0
    kappa = rng.standard_normal((M, p))
    d = rng.random(p) + 0.5
    system = DiagonalSystem(eta).low_rank_update(kappa, d)
    dense = np.diag(eta) + kappa @ np.diag(d) @ kappa.T
    _check_solve_and_multiply(system, dense, rng)


@pytest.mark.parametrize("seed,M", [(51, 8), (52, 50)])
def test_low_rank_downdate(seed: int, M: int) -> None:
    """A small downdate keeps the System positive definite and correct."""
    rng = np.random.default_rng(seed)
    eta = rng.random(M) + 5.0
    kappa = rng.standard_normal(M) * 0.1
    system = DiagonalSystem(eta).low_rank_update(kappa, -1.0)
    dense = np.diag(eta) - np.outer(kappa, kappa)
    _check_solve_and_multiply(system, dense, rng)


@pytest.mark.parametrize("seed,M,p,rank", [(61, 40, 3, 4), (62, 90, 5, 2)])
def test_low_rank_update_on_banded(seed: int, M: int, p: int, rank: int) -> None:
    """A low-rank update layers correctly on a non-diagonal base System."""
    rng = np.random.default_rng(seed)
    dense, ab = _banded_spd(M, p, rng)
    kappa = rng.standard_normal((M, rank))
    system = BandedSystem(ab).low_rank_update(kappa)
    _check_solve_and_multiply(system, dense + kappa @ kappa.T, rng)


@pytest.mark.parametrize("seed,M", [(71, 30), (72, 100)])
def test_low_rank_update_coalesces(seed: int, M: int) -> None:
    """Chained low-rank updates coalesce into one wider update."""
    rng = np.random.default_rng(seed)
    eta = rng.random(M) + 1.0
    kappa1 = rng.standard_normal((M, 2))
    d1 = rng.random(2) + 0.5
    kappa2 = rng.standard_normal((M, 3))
    d2 = rng.random(3) + 0.5

    base = DiagonalSystem(eta)
    system = base.low_rank_update(kappa1, d1).low_rank_update(kappa2, d2)

    # Coalesced: a single LowRankUpdatedSystem directly over the diagonal base,
    # not a nested pair.
    assert isinstance(system, LowRankUpdatedSystem)
    assert system._base is base
    assert system._kappa.shape == (M, 5)

    dense = (
        np.diag(eta) + kappa1 @ np.diag(d1) @ kappa1.T + kappa2 @ np.diag(d2) @ kappa2.T
    )
    _check_solve_and_multiply(system, dense, rng)


def test_low_rank_update_coalesces_mixed_weights() -> None:
    """Coalescing materializes unit weights when one update omits ``d``."""
    rng = np.random.default_rng(73)
    M = 25
    eta = rng.random(M) + 1.0
    kappa1 = rng.standard_normal((M, 2))
    kappa2 = rng.standard_normal((M, 2))
    d2 = rng.random(2) + 0.5

    base = DiagonalSystem(eta)
    system = base.low_rank_update(kappa1).low_rank_update(kappa2, d2)

    dense = np.diag(eta) + kappa1 @ kappa1.T + kappa2 @ np.diag(d2) @ kappa2.T
    _check_solve_and_multiply(system, dense, rng)


def _block_diag(
    mats: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Assemble a dense block-diagonal matrix from square blocks."""
    N = sum(m.shape[0] for m in mats)
    H = np.zeros((N, N))
    i = 0
    for m in mats:
        k = m.shape[0]
        H[i : i + k, i : i + k] = m
        i += k
    return H


def _bordered_dense(
    A11: npt.NDArray[np.float64],
    border: npt.NDArray[np.float64],
    corner: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Assemble the dense matrix ``[[A11, border], [border^T, corner]]``."""
    border = np.asarray(border, dtype=float)
    if border.ndim == 1:
        border = border[:, None]
    p = border.shape[1]
    corner = np.asarray(corner, dtype=float).reshape(p, p)
    return np.block([[A11, border], [border.T, corner]])


def _make_pd_corner(
    A11: npt.NDArray[np.float64], border: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return a corner block making ``[[A11, border], [border^T, corner]]`` PD.

    The Schur complement is set to the identity, so the bordered matrix is
    positive definite whenever ``A11`` is.
    """
    border = np.asarray(border, dtype=float)
    if border.ndim == 1:
        border = border[:, None]
    p = border.shape[1]
    return border.T @ np.linalg.solve(A11, border) + np.eye(p)


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_block_diagonal_system(seed: int) -> None:
    """A BlockDiagonalSystem solves and multiplies block by block."""
    rng = np.random.default_rng(seed)
    eta = rng.random(7) + 1.0
    dense_block = _random_spd(5, rng)
    banded_dense, banded_ab = _banded_spd(9, 2, rng)
    system = BlockDiagonalSystem(
        [DiagonalSystem(eta), DenseSystem(dense_block), BandedSystem(banded_ab)]
    )
    assert system.dimension == 21
    dense = _block_diag([np.diag(eta), dense_block, banded_dense])
    _check_solve_and_multiply(system, dense, rng)


@pytest.mark.parametrize("seed,M", [(111, 6), (112, 40)])
def test_arrow_system_diagonal_upper_left(seed: int, M: int) -> None:
    """An ArrowSystem with a diagonal upper-left (literal arrow pattern)."""
    rng = np.random.default_rng(seed)
    eta = rng.random(M) + 1.0
    border = rng.standard_normal(M)
    corner = _make_pd_corner(np.diag(eta), border)
    system = ArrowSystem(DiagonalSystem(eta), border, corner)
    assert system.dimension == M + 1
    _check_solve_and_multiply(
        system, _bordered_dense(np.diag(eta), border, corner), rng
    )


@pytest.mark.parametrize("seed,M", [(121, 5), (122, 30)])
def test_arrow_system_scalar_corner_structured_upper_left(seed: int, M: int) -> None:
    """An ArrowSystem with a scalar corner over a non-diagonal upper-left."""
    rng = np.random.default_rng(seed)
    A11 = _random_spd(M, rng)
    border = rng.standard_normal(M)
    corner = _make_pd_corner(A11, border)
    system = ArrowSystem(DenseSystem(A11), border, corner)
    assert system.dimension == M + 1
    _check_solve_and_multiply(system, _bordered_dense(A11, border, corner), rng)


@pytest.mark.parametrize("seed,M,p", [(131, 8, 3), (132, 35, 4)])
def test_arrow_system_block_corner(seed: int, M: int, p: int) -> None:
    """An ArrowSystem with a multi-column border solves via the Schur complement."""
    rng = np.random.default_rng(seed)
    A11 = _random_spd(M, rng)
    border = rng.standard_normal((M, p))
    corner = _make_pd_corner(A11, border)
    system = ArrowSystem(DenseSystem(A11), border, corner)
    assert system.dimension == M + p
    _check_solve_and_multiply(system, _bordered_dense(A11, border, corner), rng)


@pytest.mark.parametrize("seed,M,p", [(141, 6, 2), (142, 40, 5)])
def test_kkt_system(seed: int, M: int, p: int) -> None:
    """A KKTSystem solves the saddle system and multiplies the saddle matrix."""
    rng = np.random.default_rng(seed)
    hessian = _random_spd(M, rng)
    A = rng.standard_normal((p, M))
    system = KKTSystem(DenseSystem(hessian), A)
    assert system.dimension == M + p

    kkt = np.block([[hessian, A.T], [A, np.zeros((p, p))]])

    # solve: the right-hand side has a zero constraint block.
    g = rng.standard_normal(M)
    b = np.concatenate([g, np.zeros(p)])
    x = system.solve(b)
    np.testing.assert_allclose(kkt @ x, b, rtol=1e-7, atol=1e-7)

    # solve with multiple right-hand sides.
    G = rng.standard_normal((M, 3))
    B = np.vstack([G, np.zeros((p, 3))])
    X = system.solve(B)
    np.testing.assert_allclose(kkt @ X, B, rtol=1e-7, atol=1e-7)

    # multiply: the saddle matrix accepts an arbitrary right-hand side.
    y = rng.standard_normal(M + p)
    np.testing.assert_allclose(system.multiply(y), kkt @ y, rtol=1e-8, atol=1e-8)
    Y = rng.standard_normal((M + p, 3))
    np.testing.assert_allclose(system.multiply(Y), kkt @ Y, rtol=1e-8, atol=1e-8)


def test_kkt_system_rejects_nonzero_constraint_rhs() -> None:
    """KKTSystem.solve requires the trailing constraint-block RHS to vanish."""
    rng = np.random.default_rng(143)
    M, p = 5, 2
    system = KKTSystem(DenseSystem(_random_spd(M, rng)), rng.standard_normal((p, M)))
    with pytest.raises(ValueError, match="constraint-block"):
        system.solve(np.ones(M + p))


def test_nested_composition() -> None:
    """A KKT system over a low-rank update over an arrow over a block-diagonal."""
    rng = np.random.default_rng(151)

    # Block-diagonal upper-left of the arrow.
    eta_a = rng.random(6) + 1.0
    eta_b = rng.random(5) + 1.0
    block_diagonal = BlockDiagonalSystem([DiagonalSystem(eta_a), DiagonalSystem(eta_b)])
    block_diagonal_dense = _block_diag([np.diag(eta_a), np.diag(eta_b)])
    M_bd = 11

    # Arrow system with a two-column border.
    border = rng.standard_normal((M_bd, 2))
    corner = _make_pd_corner(block_diagonal_dense, border)
    arrow = ArrowSystem(block_diagonal, border, corner)
    arrow_dense = _bordered_dense(block_diagonal_dense, border, corner)
    M_arrow = M_bd + 2

    # Low-rank update -> the Hessian System.
    kappa = rng.standard_normal((M_arrow, 3))
    hessian = arrow.low_rank_update(kappa)
    hessian_dense = arrow_dense + kappa @ kappa.T

    # KKT border with equality constraints.
    p = 4
    A = rng.standard_normal((p, M_arrow))
    system = KKTSystem(hessian, A)
    assert system.dimension == M_arrow + p

    kkt = np.block([[hessian_dense, A.T], [A, np.zeros((p, p))]])

    g = rng.standard_normal(M_arrow)
    b = np.concatenate([g, np.zeros(p)])
    x = system.solve(b)
    np.testing.assert_allclose(kkt @ x, b, rtol=1e-7, atol=1e-7)

    y = rng.standard_normal(M_arrow + p)
    np.testing.assert_allclose(system.multiply(y), kkt @ y, rtol=1e-8, atol=1e-8)
