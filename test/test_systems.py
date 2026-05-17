"""Test the composable System classes."""

import numpy as np
import numpy.typing as npt
import pytest

from cvxium.exceptions import NewtonStepError
from cvxium.systems import (
    BandedSystem,
    DenseSystem,
    DiagonalSystem,
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
