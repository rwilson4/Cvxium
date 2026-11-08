"""Test numerical helpers."""

import functools
import time
import unittest
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from parameterized import parameterized
from scipy import linalg

from cvxium.exceptions import NewtonStepError
from cvxium.linalg.numerical_helpers import (
    _solve_schur_complement,
    multiply_arrow_sparsity_pattern,
    multiply_block_diagonal,
    multiply_diagonal,
    multiply_rank_one_update,
    multiply_rank_p_update,
    solve_arrow_sparsity_pattern,
    solve_banded,
    solve_block_diagonal,
    solve_block_plus_one,
    solve_diagonal,
    solve_kkt_system,
    solve_rank_one_update,
    solve_rank_p_update,
    solve_with_schur,
    tikhonov_filter,
)


def _make_spd_banded(
    n: int, p: int, rng: np.random.RandomState
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (H, ab_upper, ab_lower) for a random SPD banded matrix.

    H is the dense n-by-n matrix. ab_upper and ab_lower are the upper and lower banded
    storage forms accepted by scipy.linalg.cholesky_banded.

    """
    # Build a banded lower-triangular factor L, then H = L @ L^T
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = rng.rand() + 1.0
        for k in range(1, min(p + 1, i + 1)):
            L[i, i - k] = rng.randn() * 0.3
    H = L @ L.T

    # Upper banded form: ab_upper[p - d, d:] = diag(H, d)
    ab_upper = np.zeros((p + 1, n))
    for d in range(p + 1):
        ab_upper[p - d, d:] = np.diag(H, d)

    # Lower banded form: ab_lower[d, :n - d] = diag(H, -d)
    ab_lower = np.zeros((p + 1, n))
    for d in range(p + 1):
        ab_lower[d, : n - d] = np.diag(H, -d)

    return H, ab_upper, ab_lower


def _make_block_diagonal_system(seed: int, M0: int, M1: int, M2: int) -> tuple[
    npt.NDArray[np.float64],  # H (dense block diagonal)
    list[int],  # block_sizes
    list[Callable[..., npt.NDArray[np.float64]]],  # block_multipliers
    list[Callable[..., npt.NDArray[np.float64]]],  # block_solvers
]:
    """Build a block diagonal test system with three different block structures.

    Block 0 (size M0): diagonal. Block 1 (size M1 + 1): arrow sparsity pattern. Block 2
    (size M2): diagonal + rank-1 update.

    """
    np.random.seed(seed)

    # Block 0: diagonal
    eta0 = np.random.rand(M0) + 1.0
    H0 = np.diag(eta0)

    # Block 1: arrow sparsity pattern
    eta1 = np.random.rand(M1) + 1.0
    zeta1 = np.random.randn(M1)
    theta1 = np.dot(zeta1 / eta1, zeta1) + 1.0
    H1 = np.zeros((M1 + 1, M1 + 1))
    H1[:M1, :M1] = np.diag(eta1)
    H1[M1, :M1] = zeta1
    H1[:M1, M1] = zeta1
    H1[M1, M1] = theta1

    # Block 2: diagonal + rank-1 update
    eta2 = np.random.rand(M2) + 1.0
    kappa2 = np.random.randn(M2)
    H2 = np.diag(eta2) + np.outer(kappa2, kappa2)

    # Assemble full block diagonal matrix
    s0, s1, s2 = M0, M1 + 1, M2
    N = s0 + s1 + s2
    H = np.zeros((N, N))
    H[:s0, :s0] = H0
    H[s0 : s0 + s1, s0 : s0 + s1] = H1
    H[s0 + s1 :, s0 + s1 :] = H2

    block_sizes = [s0, s1, s2]

    block_multipliers: list[Callable[..., npt.NDArray[np.float64]]] = [
        functools.partial(multiply_diagonal, eta=eta0),
        functools.partial(
            multiply_arrow_sparsity_pattern, eta=eta1, zeta=zeta1, theta=theta1
        ),
        functools.partial(
            multiply_rank_one_update,
            kappa=kappa2,
            A_multiply=multiply_diagonal,
            eta=eta2,
        ),
    ]

    block_solvers: list[Callable[..., npt.NDArray[np.float64]]] = [
        functools.partial(solve_diagonal, eta=eta0),
        functools.partial(
            solve_arrow_sparsity_pattern, eta=eta1, zeta=zeta1, theta=theta1
        ),
        functools.partial(
            solve_rank_one_update, kappa=kappa2, A_solve=solve_diagonal, eta=eta2
        ),
    ]

    return H, block_sizes, block_multipliers, block_solvers


class NumericalHelpersTest(unittest.TestCase):
    @parameterized.expand(
        [
            (101, 100),
            (201, 200),
            (301, 50),
            (401, 500),
            (501, 13),
        ]
    )
    def test_solve_diagonal(self, seed: int, M: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        b = np.random.randn(M)

        st = time.time()
        x_expected = np.linalg.solve(np.diag(eta), b)
        mt = time.time()
        x = solve_diagonal(b, eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(np.diag(eta) @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (102, 100, 20),
            (202, 200, 30),
            (302, 50, 5),
            (402, 500, 100),
            (502, 13, 3),
        ]
    )
    def test_solve_diagonal_multiple_rhs(self, seed: int, M: int, p: int) -> None:
        """Test solving H*x = B by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        b = np.random.randn(M, p)

        st = time.time()
        x_expected = np.linalg.solve(np.diag(eta), b)
        mt = time.time()
        x = solve_diagonal(b, eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(np.diag(eta) @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (103, 100),
            (203, 200),
            (303, 50),
            (403, 500),
            (503, 13),
        ]
    )
    def test_solve_arrow_sparsity_pattern(self, seed: int, M: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0
        b = np.random.randn(M + 1)

        H = np.zeros((M + 1, M + 1))
        H[0:M, 0:M] = np.diag(eta)
        H[M, 0:M] = zeta
        H[0:M, M] = zeta
        H[M, M] = theta

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_arrow_sparsity_pattern(b, eta, zeta, theta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (104, 100, 20),
            (204, 200, 30),
            (304, 50, 5),
            (404, 500, 100),
            (504, 13, 3),
        ]
    )
    def test_solve_arrow_sparsity_pattern_multiple_rhs(
        self, seed: int, M: int, p: int
    ) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0
        b = np.random.randn(M + 1, p)

        H = np.zeros((M + 1, M + 1))
        H[0:M, 0:M] = np.diag(eta)
        H[M, 0:M] = zeta
        H[0:M, M] = zeta
        H[M, M] = theta

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_arrow_sparsity_pattern(b, eta, zeta, theta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (105, 100),
            (205, 200),
            (305, 50),
            (405, 500),
            (505, 13),
        ]
    )
    def test_solve_diagonal_plus_rank_one(self, seed: int, M: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        H = np.diag(eta) + np.outer(kappa, kappa)
        b = np.random.randn(M)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, eta=eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (106, 100, 20),
            (206, 200, 30),
            (306, 50, 5),
            (406, 500, 100),
            (506, 13, 3),
        ]
    )
    def test_solve_diagonal_plus_rank_one_multiple_rhs(
        self, seed: int, M: int, p: int
    ) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        H = np.diag(eta) + np.outer(kappa, kappa)
        b = np.random.randn(M, p)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, eta=eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (107, 100, 3),
            (207, 200, 10),
            (307, 50, 2),
            (407, 500, 10),
            (507, 13, 2),
        ]
    )
    def test_solve_arrow_plus_rank_p(self, seed: int, M: int, p: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        kappa = np.random.randn(M + 1, p)
        H = A + kappa @ kappa.T
        b = np.random.randn(M + 1)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_arrow_sparsity_pattern,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (108, 100, 3, 20),
            (208, 200, 10, 30),
            (308, 50, 2, 5),
            (408, 500, 10, 100),
            (508, 13, 2, 3),
        ]
    )
    def test_solve_arrow_plus_rank_p_multiple_rhs(
        self, seed: int, M: int, p: int, q: int
    ) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        kappa = np.random.randn(M + 1, p)
        H = A + kappa @ kappa.T
        b = np.random.randn(M + 1, q)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_arrow_sparsity_pattern,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )
        et = time.time()

        print(f"Slow way completed in {1e3 * (mt - st):.03f} ms")
        print(f"Fast way completed in {1e3 * (et - mt):.03f} ms")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (120, 100, 2),
            (220, 200, 3),
            (320, 50, 2),
            (420, 500, 4),
            (520, 13, 2),
        ]
    )
    def test_solve_rank_p_update_mixed_signs(self, seed: int, M: int, p: int) -> None:
        """Test solving (D + K C K^T) x = b with mixed signs in d."""
        rng = np.random.default_rng(seed)
        eta = rng.random(M) + 1.0  # diagonal
        kappa = rng.standard_normal((M, p))
        signs = np.ones(p)
        signs[0] = -1.0  # first column subtracted

        D = np.diag(eta)
        C = np.diag(signs)
        H = D + kappa @ C @ kappa.T
        b = rng.standard_normal(M)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_diagonal,
            d=signs,
            check_inertia=False,
            eta=eta,
        )

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (121, 100, 2, 10),
            (221, 200, 3, 15),
            (321, 50, 2, 5),
            (421, 500, 4, 20),
            (521, 13, 2, 3),
        ]
    )
    def test_solve_rank_p_update_mixed_signs_multiple_rhs(
        self, seed: int, M: int, p: int, q: int
    ) -> None:
        """Test solving (D + K C K^T) X = B with mixed signs and multiple RHS."""
        rng = np.random.default_rng(seed)
        eta = rng.random(M) + 1.0
        kappa = rng.standard_normal((M, p))
        signs = np.ones(p)
        signs[0] = -1.0

        D = np.diag(eta)
        C = np.diag(signs)
        H = D + kappa @ C @ kappa.T
        b = rng.standard_normal((M, q))

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_diagonal,
            d=signs,
            check_inertia=False,
            eta=eta,
        )

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (601, 40, 8),
            (602, 100, 16),
            (603, 30, 6),
        ]
    )
    def test_solve_rank_p_update_extreme_d_spread(
        self, seed: int, M: int, p: int
    ) -> None:
        """Verify equilibration handles IPM-boundary d spreads (1e-12 to 1e+6).

        Without equilibration, ``G = diag(1/d) + kappa^T A^{-1} kappa`` becomes
        catastrophically ill-conditioned when d entries span many orders of
        magnitude, and the Cholesky-based Woodbury solve loses most of its
        precision. With equilibration, ``G_eq = diag(sign_d) + kappa_eq^T A^{-1}
        kappa_eq`` (kappa_eq = kappa * sqrt(|d|)) is bounded by the rank-q
        contribution's own conditioning, recovering full precision.
        """
        rng = np.random.default_rng(seed)
        eta = rng.uniform(0.5, 2.0, M)  # PD diagonal A
        kappa = rng.standard_normal((M, p))
        d_big = rng.uniform(1e4, 1e6, p // 2)
        d_tiny = rng.uniform(1e-12, 1e-10, p - p // 2)
        d = np.concatenate([d_big, d_tiny])
        rng.shuffle(d)
        b = rng.standard_normal(M)

        H = np.diag(eta) + kappa @ np.diag(d) @ kappa.T

        x = solve_rank_p_update(b, kappa, A_solve=solve_diagonal, d=d, eta=eta)

        # Tolerance scaled to ``cond(H)``: with ``max(d) * ||kappa||^2`` up to
        # ~1e9, the dense reference itself only retains ~7-8 digits, so we
        # check the relative residual rather than componentwise equality.
        rel_residual = float(np.linalg.norm(H @ x - b) / np.linalg.norm(b))
        assert rel_residual < 1e-6, f"relative residual {rel_residual:.3e} too large"

    @parameterized.expand(
        [
            (109, 100),
            (209, 200),
            (309, 50),
            (409, 500),
            (509, 13),
        ]
    )
    def test_solve_block_plus_one(self, seed: int, M: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        # Construct block matrix H
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        H = np.zeros((M + 1, M + 1))
        H[0:M, 0:M] = np.diag(eta)
        H[M, 0:M] = zeta
        H[0:M, M] = zeta
        H[M, M] = theta

        b = np.random.randn(M + 1)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_block_plus_one(b, zeta, theta, A11_solve=solve_diagonal, eta=eta)
        et = time.time()
        xalt = solve_arrow_sparsity_pattern(b, eta, zeta, theta)
        ft = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")
        print(f"Faster way completed in {1e6 * (ft - et):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x, xalt, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (110, 100, 20),
            (210, 200, 30),
            (310, 50, 5),
            (410, 500, 100),
            (510, 13, 3),
        ]
    )
    def test_solve_block_plus_one_multiple_rhs(self, seed: int, M: int, p: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        # Construct block matrix H
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        H = np.zeros((M + 1, M + 1))
        H[0:M, 0:M] = np.diag(eta)
        H[M, 0:M] = zeta
        H[0:M, M] = zeta
        H[M, M] = theta

        b = np.random.randn(M + 1, p)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_block_plus_one(b, zeta, theta, A11_solve=solve_diagonal, eta=eta)
        et = time.time()
        xalt = solve_arrow_sparsity_pattern(b, eta, zeta, theta)
        ft = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")
        print(f"Faster way completed in {1e6 * (ft - et):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x, xalt, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (111, 100, 3),
            (211, 200, 5),
            (311, 50, 2),
            (411, 500, 10),
            (511, 13, 2),
        ]
    )
    def test_solve_with_schur(self, seed: int, M: int, p: int) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        # Construct block matrix H
        eta = np.random.rand(M) + 1.0
        A12 = np.random.randn(M, p)

        # Construct A22 so that H is positive definite
        Ssqrt = np.random.randn(p, p)
        S = Ssqrt.T @ Ssqrt
        A12_prime = A12 / eta[:, np.newaxis]
        A22 = S + A12.T @ A12_prime

        H = np.zeros((M + p, M + p))
        H[0:M, 0:M] = np.diag(eta)
        H[M:, 0:M] = A12.T
        H[0:M, M:] = A12
        H[M:, M:] = A22

        b = np.random.randn(M + p)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_with_schur(b, A12, A22, A11_solve=solve_diagonal, eta=eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand([(np.nan,), (np.inf,), (-np.inf,)])
    def test_solve_with_schur_nonfinite_raises_newton_step_error(
        self,
        bad_value: float,
    ) -> None:
        """A non-finite Schur complement degrades to ``NewtonStepError``.

        ``scipy.linalg.cho_factor`` raises ``ValueError`` (not ``LinAlgError``)
        via its ``check_finite`` guard when the Schur complement ``S = A22 -
        A12^T A11^{-1} A12`` holds inf/NaN. ``solve_with_schur`` must translate
        that to ``NewtonStepError`` so callers' graceful fallbacks engage rather
        than letting the raw ``ValueError`` crash the solve.
        """
        np.random.seed(733)
        M, p = 13, 2
        eta = np.random.rand(M) + 1.0
        A12 = np.random.randn(M, p)
        A12_prime = A12 / eta[:, np.newaxis]
        Ssqrt = np.random.randn(p, p)
        A22 = Ssqrt.T @ Ssqrt + A12.T @ A12_prime
        A22[0, 0] = bad_value  # poison the Schur complement
        b = np.random.randn(M + p)

        with self.assertRaises(NewtonStepError):
            solve_with_schur(b, A12, A22, A11_solve=solve_diagonal, eta=eta)

    @parameterized.expand(
        [
            (112, 100, 3, 20),
            (212, 200, 5, 30),
            (312, 50, 2, 5),
            (412, 500, 10, 100),
            (512, 13, 2, 3),
        ]
    )
    def test_solve_with_schur_multiple_rhs(
        self, seed: int, M: int, p: int, q: int
    ) -> None:
        """Test solving H*x = b by doing it the slow way."""
        np.random.seed(seed)
        # Construct block matrix H
        eta = np.random.rand(M) + 1.0
        A12 = np.random.randn(M, p)

        # Construct A22 so that H is positive definite
        Ssqrt = np.random.randn(p, p)
        S = Ssqrt.T @ Ssqrt
        A12_prime = A12 / eta[:, np.newaxis]
        A22 = S + A12.T @ A12_prime

        H = np.zeros((M + p, M + p))
        H[0:M, 0:M] = np.diag(eta)
        H[M:, 0:M] = A12.T
        H[0:M, M:] = A12
        H[M:, M:] = A22

        b = np.random.randn(M + p, q)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_with_schur(b, A12, A22, A11_solve=solve_diagonal, eta=eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (113, 100, 20),
            (213, 200, 30),
            (313, 50, 5),
            (413, 500, 100),
            (513, 13, 3),
        ]
    )
    def test_solve_kkt_system_hessian_diagonal(self, seed: int, M: int, p: int) -> None:
        """Test solving KKT system and checking answer."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        H = np.diag(eta)
        A = np.random.randn(p, M)
        g = np.random.randn(M)

        KKT = np.zeros((M + p, M + p))
        KKT[0:M, 0:M] = H
        KKT[0:M, M:] = A.T
        KKT[M:, 0:M] = A
        rhs = np.zeros((M + p,))
        rhs[0:M] = g

        st = time.time()
        x_expected = np.linalg.solve(KKT, rhs)
        mt = time.time()
        delta_w_expected = x_expected[0:M]
        nu_expected = x_expected[M:]

        delta_w, nu = solve_kkt_system(A, g, hessian_solve=solve_diagonal, eta=eta)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(delta_w, delta_w_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(nu, nu_expected, rtol=1e-8, atol=1e-8)

        lhs_top = H @ delta_w + A.T @ nu
        rhs_top = g
        lhs_bot = A @ delta_w
        rhs_bot = np.zeros(p)

        # Verify H * delta_w + A^T * xi = g
        np.testing.assert_allclose(lhs_top, rhs_top, rtol=1e-8, atol=1e-8)

        # Verify A * delta_w = 0
        np.testing.assert_allclose(lhs_bot, rhs_bot, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (114, 100, 20),
            (214, 200, 30),
            (314, 50, 5),
            (414, 500, 100),
            (514, 13, 3),
        ]
    )
    def test_solve_kkt_system_hessian_diagonal_plus_rank_one(
        self, seed: int, M: int, p: int
    ) -> None:
        """Test solving KKT system and checking answer."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        H = np.diag(eta) + np.outer(kappa, kappa)
        A = np.random.randn(p, M)
        g = np.random.randn(M)

        KKT = np.zeros((M + p, M + p))
        KKT[0:M, 0:M] = H
        KKT[0:M, M:] = A.T
        KKT[M:, 0:M] = A
        rhs = np.zeros((M + p,))
        rhs[0:M] = g

        st = time.time()
        x_expected = np.linalg.solve(KKT, rhs)
        mt = time.time()
        delta_w_expected = x_expected[0:M]
        nu_expected = x_expected[M:]

        delta_w, nu = solve_kkt_system(
            A,
            g,
            hessian_solve=solve_rank_one_update,
            kappa=kappa,
            A_solve=solve_diagonal,
            eta=eta,
        )
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(delta_w, delta_w_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(nu, nu_expected, rtol=1e-8, atol=1e-8)

        lhs_top = H @ delta_w + A.T @ nu
        rhs_top = g
        lhs_bot = A @ delta_w
        rhs_bot = np.zeros(p)

        # Verify H * delta_w + A^T * xi = g
        np.testing.assert_allclose(lhs_top, rhs_top, rtol=1e-8, atol=1e-8)

        # Verify A * delta_w = 0
        np.testing.assert_allclose(lhs_bot, rhs_bot, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (115, 100, 20),
            (215, 200, 30),
            (315, 50, 5),
            (415, 500, 100),
            (515, 13, 3),
        ]
    )
    def test_solve_kkt_system_hessian_arrow_sparsity_pattern(
        self, seed: int, M: int, p: int
    ) -> None:
        """Test solving KKT system and checking answer."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        H = np.zeros((M + 1, M + 1))
        H[0:M, 0:M] = np.diag(eta)
        H[M, 0:M] = zeta
        H[0:M, M] = zeta
        H[M, M] = theta

        A = np.zeros((p, M + 1))
        A[:, 0:M] = np.random.randn(p, M)
        g = np.random.randn(M + 1)

        KKT = np.zeros((M + 1 + p, M + 1 + p))
        KKT[0 : M + 1, 0 : M + 1] = H
        KKT[0 : M + 1, M + 1 :] = A.T
        KKT[M + 1 :, 0 : M + 1] = A
        rhs = np.zeros((M + 1 + p,))
        rhs[0 : M + 1] = g

        st = time.time()
        x_expected = np.linalg.solve(KKT, rhs)
        mt = time.time()
        delta_x_expected = x_expected[0 : M + 1]
        nu_expected = x_expected[M + 1 :]

        delta_x, nu = solve_kkt_system(
            A,
            g,
            hessian_solve=solve_arrow_sparsity_pattern,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(delta_x, delta_x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(nu, nu_expected, rtol=1e-8, atol=1e-8)

        lhs_top = H @ delta_x + A.T @ nu
        rhs_top = g
        lhs_bot = A @ delta_x
        rhs_bot = np.zeros(p)

        # Verify H * delta_w + A^T * xi = g
        np.testing.assert_allclose(lhs_top, rhs_top, rtol=1e-8, atol=1e-8)

        # Verify A * delta_w = 0
        np.testing.assert_allclose(lhs_bot, rhs_bot, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (116, 100, 20),
            (216, 200, 30),
            (317, 50, 5),
            (416, 500, 100),
            (516, 13, 3),
        ]
    )
    def test_solve_kkt_system_hessian_diagonal_plus_rank_one_rank_deficient(
        self, seed: int, M: int, p: int
    ) -> None:
        """Test solving KKT system and checking answer."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        H = np.diag(eta) + np.outer(kappa, kappa)
        g = np.random.randn(M)

        A = np.random.randn(p, M)
        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        A = U @ np.diag(s) @ Vh

        KKT = np.zeros((M + p, M + p))
        KKT[0:M, 0:M] = H
        KKT[0:M, M:] = A.T
        KKT[M:, 0:M] = A
        rhs = np.zeros((M + p,))
        rhs[0:M] = g

        st = time.time()
        x_expected = np.linalg.solve(KKT, rhs)
        mt = time.time()
        delta_w_expected = x_expected[0:M]
        # nu_expected = x_expected[M:]

        delta_w, nu = solve_kkt_system(
            A,
            g,
            hessian_solve=solve_rank_one_update,
            kappa=kappa,
            A_solve=solve_diagonal,
            eta=eta,
        )
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(delta_w, delta_w_expected, rtol=1e-8, atol=1e-8)
        # Since A is rank deficient, the solution is not unique. We return the minimum norm
        # solution, but who knows what Scipy does.
        # assert np.sqrt(np.dot(nu, nu)) <= np.sqrt(np.dot(nu_expected, nu_expected))

        lhs_top = H @ delta_w + A.T @ nu
        rhs_top = g
        lhs_bot = A @ delta_w
        rhs_bot = np.zeros(p)

        # Verify H * delta_w + A^T * xi = g
        np.testing.assert_allclose(lhs_top, rhs_top, rtol=1e-8, atol=1e-8)

        # Verify A * delta_w = 0
        np.testing.assert_allclose(lhs_bot, rhs_bot, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (117, 100, 3),
            (217, 200, 5),
            (317, 50, 2),
            (417, 500, 10),
            (517, 13, 2),
        ]
    )
    def test_solve_banded(self, seed: int, M: int, p: int) -> None:
        """Test solving H*x = b for a banded SPD matrix (single RHS)."""
        rng = np.random.RandomState(seed)
        H, ab_upper, ab_lower = _make_spd_banded(M, p, rng)
        b = rng.randn(M)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x_upper = solve_banded(b, ab_upper, lower=False)
        et = time.time()
        x_lower = solve_banded(b, ab_lower, lower=True)
        ft = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way (upper) completed in {1e6 * (et - mt):.03f} us")
        print(f"Fast way (lower) completed in {1e6 * (ft - et):.03f} us")

        np.testing.assert_allclose(x_upper, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x_lower, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x_upper, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (118, 100, 3, 20),
            (218, 200, 5, 30),
            (318, 50, 2, 5),
            (418, 500, 10, 100),
            (518, 13, 2, 3),
        ]
    )
    def test_solve_banded_multiple_rhs(self, seed: int, M: int, p: int, q: int) -> None:
        """Test solving H*x = B for a banded SPD matrix (multiple RHS)."""
        rng = np.random.RandomState(seed)
        H, ab_upper, ab_lower = _make_spd_banded(M, p, rng)
        b = rng.randn(M, q)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x_upper = solve_banded(b, ab_upper, lower=False)
        et = time.time()
        x_lower = solve_banded(b, ab_lower, lower=True)
        ft = time.time()

        print(f"Slow way completed in {1e3 * (mt - st):.03f} ms")
        print(f"Fast way (upper) completed in {1e3 * (et - mt):.03f} ms")
        print(f"Fast way (lower) completed in {1e3 * (ft - et):.03f} ms")

        np.testing.assert_allclose(x_upper, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x_lower, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x_upper, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand([(np.nan,), (np.inf,), (-np.inf,)])
    def test_solve_banded_nonfinite_hessian_raises_newton_step_error(
        self,
        bad_value: float,
    ) -> None:
        """A non-finite banded Hessian degrades to ``NewtonStepError``.

        ``scipy.linalg.cholesky_banded`` raises ``ValueError`` (not
        ``LinAlgError``) via its ``check_finite`` guard when ``ab`` holds
        inf/NaN. ``solve_banded`` must translate that to ``NewtonStepError`` so
        callers' graceful fallbacks engage (e.g. the dual's mini-Newton in
        ``_minimize_lagrangian``) rather than letting the raw ``ValueError``
        crash the solve.
        """
        rng = np.random.RandomState(631)
        _, _, ab_lower = _make_spd_banded(20, 3, rng)
        b = rng.randn(20)

        ab_lower[0, 5] = bad_value  # corrupt a main-diagonal entry

        with self.assertRaises(NewtonStepError):
            solve_banded(b, ab_lower, lower=True)

    @parameterized.expand(
        [
            (119, 20, 13, 10),
            (219, 50, 19, 30),
            (319, 10, 9, 5),
            (419, 100, 49, 50),
            (519, 7, 4, 3),
        ]
    )
    def test_multiply_block_diagonal(
        self, seed: int, M0: int, M1: int, M2: int
    ) -> None:
        """Test H @ y for a block diagonal matrix with mixed block structures."""
        H, block_sizes, block_multipliers, _ = _make_block_diagonal_system(
            seed, M0, M1, M2
        )
        N = sum(block_sizes)
        np.random.seed(seed + 1000)
        y = np.random.randn(N)

        st = time.time()
        z_expected = H @ y
        mt = time.time()
        z = multiply_block_diagonal(y, block_sizes, block_multipliers)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(z, z_expected, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (120, 20, 13, 10, 5),
            (220, 50, 19, 30, 10),
            (320, 10, 9, 5, 3),
            (420, 100, 49, 50, 20),
            (520, 7, 4, 3, 4),
        ]
    )
    def test_multiply_block_diagonal_multiple_rhs(
        self, seed: int, M0: int, M1: int, M2: int, q: int
    ) -> None:
        """Test H @ Y for a block diagonal matrix with multiple RHS."""
        H, block_sizes, block_multipliers, _ = _make_block_diagonal_system(
            seed, M0, M1, M2
        )
        N = sum(block_sizes)
        np.random.seed(seed + 1000)
        y = np.random.randn(N, q)

        st = time.time()
        z_expected = H @ y
        mt = time.time()
        z = multiply_block_diagonal(y, block_sizes, block_multipliers)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(z, z_expected, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (121, 20, 13, 10),
            (221, 50, 19, 30),
            (321, 10, 9, 5),
            (421, 100, 49, 50),
            (521, 7, 4, 3),
        ]
    )
    def test_solve_block_diagonal(self, seed: int, M0: int, M1: int, M2: int) -> None:
        """Test H * x = b for a block diagonal matrix with mixed block structures."""
        H, block_sizes, _, block_solvers = _make_block_diagonal_system(seed, M0, M1, M2)
        N = sum(block_sizes)
        np.random.seed(seed + 1000)
        b = np.random.randn(N)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_block_diagonal(b, block_sizes, block_solvers)
        et = time.time()

        print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
        print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H * x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (122, 20, 13, 10, 5),
            (222, 50, 19, 30, 10),
            (322, 10, 9, 5, 3),
            (422, 100, 49, 50, 20),
            (522, 7, 4, 3, 4),
        ]
    )
    def test_solve_block_diagonal_multiple_rhs(
        self, seed: int, M0: int, M1: int, M2: int, q: int
    ) -> None:
        """Test H * x = B for a block diagonal matrix with multiple RHS."""
        H, block_sizes, _, block_solvers = _make_block_diagonal_system(seed, M0, M1, M2)
        N = sum(block_sizes)
        np.random.seed(seed + 1000)
        b = np.random.randn(N, q)

        st = time.time()
        x_expected = np.linalg.solve(H, b)
        mt = time.time()
        x = solve_block_diagonal(b, block_sizes, block_solvers)
        et = time.time()

        print(f"Slow way completed in {1e3 * (mt - st):.03f} ms")
        print(f"Fast way completed in {1e3 * (et - mt):.03f} ms")

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H * x = B
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (123, 100),
            (223, 200),
            (323, 50),
            (423, 500),
            (523, 13),
        ]
    )
    def test_solve_diagonal_plus_rank_one_with_d(self, seed: int, M: int) -> None:
        """Test solving H*x = b for H = diag(eta) + d*kappa*kappa^T with scalar d."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        d = float(np.random.rand() + 0.5)
        H = np.diag(eta) + d * np.outer(kappa, kappa)
        b = np.random.randn(M)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, d=d, eta=eta)

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (124, 100, 20),
            (224, 200, 30),
            (324, 50, 5),
            (424, 500, 100),
            (524, 13, 3),
        ]
    )
    def test_solve_diagonal_plus_rank_one_with_d_multiple_rhs(
        self, seed: int, M: int, p: int
    ) -> None:
        """Test solving H*x = b for H = diag(eta) + d*kappa*kappa^T, multiple RHS."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        d = float(np.random.rand() + 0.5)
        H = np.diag(eta) + d * np.outer(kappa, kappa)
        b = np.random.randn(M, p)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, d=d, eta=eta)

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (125, 100, 3),
            (225, 200, 10),
            (325, 50, 2),
            (425, 500, 10),
            (525, 13, 2),
        ]
    )
    def test_solve_arrow_plus_rank_p_with_d(self, seed: int, M: int, p: int) -> None:
        """Test solving H*x = b for H = arrow + kappa @ D @ kappa^T with diagonal D."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        kappa = np.random.randn(M + 1, p)
        d = np.random.rand(p) + 0.5
        H = A + kappa @ np.diag(d) @ kappa.T
        b = np.random.randn(M + 1)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_arrow_sparsity_pattern,
            d=d,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (126, 100, 3, 20),
            (226, 200, 10, 30),
            (326, 50, 2, 5),
            (426, 500, 10, 100),
            (526, 13, 2, 3),
        ]
    )
    def test_solve_arrow_plus_rank_p_with_d_multiple_rhs(
        self, seed: int, M: int, p: int, q: int
    ) -> None:
        """Test solving H*x = b for H = arrow + kappa @ D @ kappa^T, multiple RHS."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        kappa = np.random.randn(M + 1, p)
        d = np.random.rand(p) + 0.5
        H = A + kappa @ np.diag(d) @ kappa.T
        b = np.random.randn(M + 1, q)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_arrow_sparsity_pattern,
            d=d,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

        # Verify H*x = b
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    # -----------------------------------------------------------------------
    # Downdate solve tests
    # -----------------------------------------------------------------------

    @parameterized.expand(
        [
            (127, 100),
            (227, 200),
            (327, 50),
            (427, 500),
            (527, 13),
        ]
    )
    def test_solve_rank_one_downdate_pd(self, seed: int, M: int) -> None:
        """Solve H*x = b for H = diag(eta) + d*kappa*kappa^T, d < 0, H still PD."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        # Choose |d| < 1 / (kappa^T A^{-1} kappa) so the downdate keeps H PD.
        max_d = 1.0 / np.dot(kappa, kappa / eta)
        d = -float(np.random.rand() * 0.5 * max_d)
        H = np.diag(eta) + d * np.outer(kappa, kappa)
        b = np.random.randn(M)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, d=d, eta=eta)

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (128, 100),
            (228, 200),
            (328, 50),
            (428, 500),
            (528, 13),
        ]
    )
    def test_solve_rank_one_downdate_indefinite_raises(self, seed: int, M: int) -> None:
        """NewtonStepError raised when rank-one downdate makes H indefinite."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        kappa = np.random.randn(M)
        # Choose |d| > 1 / (kappa^T A^{-1} kappa) so the downdate makes H indefinite.
        max_d = 1.0 / np.dot(kappa, kappa / eta)
        d = -float((1.0 + np.random.rand()) * max_d)
        b = np.random.randn(M)

        with self.assertRaises(NewtonStepError):
            solve_rank_one_update(b, kappa, A_solve=solve_diagonal, d=d, eta=eta)

    @parameterized.expand(
        [
            (129, 100, 3),
            (229, 200, 10),
            (329, 50, 2),
            (429, 500, 10),
            (529, 13, 2),
        ]
    )
    def test_solve_rank_p_downdate_pd(self, seed: int, M: int, p: int) -> None:
        """Solve H*x = b for H = arrow + kappa @ D @ kappa^T, all d < 0, H still PD."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        kappa = np.random.randn(M + 1, p)
        # Scale kappa down so the downdate is small enough to keep H PD.
        kappa = kappa / (np.linalg.norm(kappa) * 4)
        d = -(np.random.rand(p) + 0.5)
        H = A + kappa @ np.diag(d) @ kappa.T

        # Verify H is actually PD before testing.
        assert np.all(np.linalg.eigvalsh(H) > 0), "Test setup error: H is not PD"

        b = np.random.randn(M + 1)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_arrow_sparsity_pattern,
            d=d,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    @parameterized.expand(
        [
            (130, 100, 3),
            (230, 200, 10),
            (330, 50, 2),
            (430, 500, 10),
            (530, 13, 2),
        ]
    )
    def test_solve_rank_p_downdate_indefinite_raises(
        self, seed: int, M: int, p: int
    ) -> None:
        """NewtonStepError raised when rank-p downdate makes H indefinite."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        # Large kappa so the downdate dominates and H becomes indefinite.
        kappa = np.random.randn(M + 1, p) * 5.0
        d = -(np.random.rand(p) + 1.0)
        H = A + kappa @ np.diag(d) @ kappa.T

        # Verify H is actually indefinite before testing.
        assert np.any(
            np.linalg.eigvalsh(H) < 0
        ), "Test setup error: H is not indefinite"

        b = np.random.randn(M + 1)

        with self.assertRaises(NewtonStepError):
            solve_rank_p_update(
                b,
                kappa,
                A_solve=solve_arrow_sparsity_pattern,
                d=d,
                eta=eta,
                zeta=zeta,
                theta=theta,
            )

    @parameterized.expand(
        [
            (131, 100, 4),
            (231, 200, 6),
            (331, 50, 3),
            (431, 500, 8),
            (531, 13, 3),
        ]
    )
    def test_solve_rank_p_mixed_d(self, seed: int, M: int, p: int) -> None:
        """Solve H*x = b for H = arrow + kappa @ D @ kappa^T with mixed-sign d, H PD."""
        np.random.seed(seed)
        eta = np.random.rand(M) + 1.0
        zeta = np.random.randn(M)
        theta = np.dot(zeta / eta, zeta) + 1.0

        A = np.zeros((M + 1, M + 1))
        A[0:M, 0:M] = np.diag(eta)
        A[M, 0:M] = zeta
        A[0:M, M] = zeta
        A[M, M] = theta

        # Scale kappa down so mixed d still leaves H PD.
        kappa = np.random.randn(M + 1, p)
        kappa = kappa / (np.linalg.norm(kappa) * 4)
        d = np.where(
            np.arange(p) % 2 == 0,
            np.random.rand(p) + 0.5,
            -(np.random.rand(p) * 0.1),
        )
        H = A + kappa @ np.diag(d) @ kappa.T

        # Verify H is actually PD before testing.
        assert np.all(np.linalg.eigvalsh(H) > 0), "Test setup error: H is not PD"

        b = np.random.randn(M + 1)

        x_expected = np.linalg.solve(H, b)
        x = solve_rank_p_update(
            b,
            kappa,
            A_solve=solve_arrow_sparsity_pattern,
            d=d,
            eta=eta,
            zeta=zeta,
            theta=theta,
        )

        np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)

    # ----- Defensive shape-contract checks on A_solve / A_multiply callables -----
    #
    # `solve_rank_p_update` and friends invoke the user-supplied `A_solve` /
    # `A_multiply` callables with both 1D vectors and 2D matrices. A naive closure
    # (e.g. `lambda rhs: rhs / diag_vec` where diag_vec is 1D) silently broadcasts
    # `(M, 1) / (M,)` into `(M, M)` and returns garbage downstream. The defensive
    # checks below catch this class of bug at the contract violation point.

    def test_solve_rank_p_update_rejects_broadcasting_a_solve(self) -> None:
        """A_solve that broadcasts (M, 1) / (M,) into (M, M) is rejected."""
        M, p = 5, 1
        diag_vec = np.arange(1.0, M + 1)
        # Buggy: returns shape (M, M) when called with kappa shape (M, 1).
        buggy_solve = lambda rhs: rhs / diag_vec  # noqa: E731
        kappa = np.ones((M, p))
        b = np.ones(M)
        with self.assertRaisesRegex(ValueError, "A_solve violated shape contract"):
            solve_rank_p_update(b, kappa, A_solve=buggy_solve)

    def test_solve_rank_one_update_rejects_shape_mismatch_a_solve(self) -> None:
        """A_solve that returns the wrong shape is rejected."""
        M = 5
        # Buggy: returns shape (M, 1) when called with kappa shape (M,).
        buggy_solve = lambda rhs: rhs.reshape(-1, 1)  # noqa: E731
        kappa = np.ones(M)
        b = np.ones(M)
        with self.assertRaisesRegex(ValueError, "A_solve violated shape contract"):
            solve_rank_one_update(b, kappa, A_solve=buggy_solve)

    def test_multiply_rank_p_update_rejects_broadcasting_a_multiply(self) -> None:
        """A_multiply that returns the wrong shape is rejected."""
        M, p = 5, 1
        diag_vec = np.arange(1.0, M + 1)
        buggy_multiply = lambda v: v.reshape(-1, 1) * diag_vec  # noqa: E731
        kappa = np.ones((M, p))
        y = np.ones(M)
        with self.assertRaisesRegex(ValueError, "A_multiply violated shape contract"):
            multiply_rank_p_update(y, kappa, A_multiply=buggy_multiply)

    def test_multiply_rank_one_update_rejects_shape_mismatch_a_multiply(self) -> None:
        """A_multiply that returns the wrong shape is rejected."""
        M = 5
        buggy_multiply = lambda v: v.reshape(-1, 1)  # noqa: E731
        kappa = np.ones(M)
        y = np.ones(M)
        with self.assertRaisesRegex(ValueError, "A_multiply violated shape contract"):
            multiply_rank_one_update(y, kappa, A_multiply=buggy_multiply)

    @parameterized.expand([(1e-6,), (1e-3,), (1.0,), (1e3,), (1e6,)])
    def test_solve_rank_one_update_singularity_guard_is_scale_invariant(
        self,
        scale: float,
    ) -> None:
        """The singularity guard fires on scale-invariant cancellation, not on absolute G.

        Construct a rank-one update where the rescaled inputs would cause the
        absolute 1e-12 guard to misfire — spuriously raising on tiny kappa where
        G naturally lives below 1e-12, or silently accepting a near-singular H on
        huge kappa where the absolute floor is meaningless. The relative guard is
        invariant under uniform rescaling of kappa.
        """
        np.random.seed(7777)
        M = 30
        eta = np.ones(M)
        kappa = scale * np.ones(M)
        b = scale * np.random.randn(M)
        # H = I + kappa kappa^T is non-singular at every scale (G = 1 + ||kappa||^2 > 0
        # always). The fix's effect is in the singularity *guard*, not numerical
        # accuracy of the solve — residual error scales with cond(H) and is
        # governed elsewhere.
        x = solve_rank_one_update(b, kappa, A_solve=solve_diagonal, eta=eta)
        assert np.all(np.isfinite(x))

    def test_solve_arrow_sparsity_pattern_rejects_cancellation_noise(self) -> None:
        """Reject psi^2 values that are positive but indistinguishable from cancellation noise.

        Construct ``theta = zeta^T diag(eta)^{-1} zeta * (1 + 1e-15)`` so that the
        ``psi_squared = theta - dot`` cancellation produces a tiny positive value
        (~1e-14) that lies well within the rounding noise of either operand. Old
        code (``psi_squared <= 0``) silently accepts this and divides by it,
        producing a wildly wrong solution; the relative guard correctly raises.
        """
        np.random.seed(8888)
        M = 30
        zeta = np.random.randn(M)
        eta = np.ones(M)
        zeta_dot = float(np.dot(zeta, zeta))
        # Theta sits a few rounding ulps above the true dot — psi^2 is float noise.
        theta = zeta_dot * (1.0 + 1e-15)
        b = np.zeros(M + 1)
        b[M] = 1.0

        with self.assertRaisesRegex(
            NewtonStepError, "Hessian is not strictly positive definite"
        ):
            solve_arrow_sparsity_pattern(b, eta, zeta, theta)

    def test_solve_schur_complement_range_test_uses_relative_tolerance(self) -> None:
        """Range-test on neg_c uses ||residual|| / ||neg_c||, not numpy default tols.

        Construct a rank-1 ``neg_S`` (cho_factor fails → SVD path) with leading
        direction ``v1`` and a ``neg_c`` whose out-of-range component along ``v2``
        has magnitude ``0.1`` while ``||neg_c|| ~ 1e6``. The default
        ``np.allclose(rtol=1e-5, atol=1e-8)`` tolerance is then ``~10`` and
        silently accepts the bad RHS; the relative ``1e-10 * max(1, ||neg_c||)``
        tolerance is ``~1e-4`` and correctly raises with a diagnostic.

        This is the slack-form Phase I failure pattern where ``neg_c = A @ b`` has
        a slack column dominating its norm.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        neg_S = 1e6 * np.outer(v1, v1)  # rank-1; cho_factor fails -> SVD path.
        neg_c = 1e6 * v1 + 0.1 * v2  # out-of-range component of magnitude 0.1.

        with self.assertRaisesRegex(
            NewtonStepError, "KKT system did not have a solution"
        ):
            _solve_schur_complement(neg_S, neg_c)

    def test_solve_schur_complement_svd_rank_uses_relative_threshold(self) -> None:
        """SVD rank cut treats trailing singular values < 1e-12 * s_max as zero.

        Constructs a 2x2 indefinite matrix with eigenvalues ``[1e8, -1e-7]`` so
        Cholesky fails and the SVD fallback fires. The trailing singular value
        ``1e-7`` survives an absolute ``1e-10`` threshold and is treated as a real
        direction; with a relative threshold scaled by ``s_max=1e8``, it falls
        below ``1e-4`` and is correctly classified as numerical noise.

        Float-precision noise on ``neg_c`` in the trailing direction then exposes
        the bug: with the absolute threshold, the spurious ``1e7``-magnitude
        reciprocal amplifies the noise into a sizeable error in ``nu``; with the
        relative threshold, the trailing component is zeroed out and ``nu`` is
        correct.
        """
        v1 = np.array([0.6, 0.8])
        v2 = np.array([-0.8, 0.6])  # orthogonal to v1.
        neg_S = 1e8 * np.outer(v1, v1) - 1e-7 * np.outer(v2, v2)
        # Noise representative of float precision (~ machine_epsilon * 2.5).
        neg_c = 2.5 * v1 + 1e-15 * v2

        nu = _solve_schur_complement(neg_S, neg_c)

        expected = v1 * (2.5 / 1e8)
        np.testing.assert_allclose(nu, expected, rtol=1e-6, atol=1e-12)


class TikhonovFilterTest(unittest.TestCase):
    """Tikhonov-damped SVD pseudo-inverse filter.

    The filter replaces the hard-cut ``1 / s`` (or ``1 / s^2``) pseudo-inverse
    that made a rank-deficient ``A`` unstable: a singular value landing just
    above an absolute cutoff produced an unbounded reciprocal that amplified
    float noise into a corrupt step. These tests pin the two properties that fix
    it -- reciprocal accuracy far above the floor, and boundedness at/below it.
    """

    def test_power_one_matches_reciprocal_far_above_floor(self) -> None:
        """For s >> lmbda, the power-1 filter is the plain reciprocal 1 / s."""
        s = np.array([10.0, 5.0, 1.0])
        filt, rank = tikhonov_filter(s, power=1)
        np.testing.assert_allclose(filt, 1.0 / s, rtol=1e-10)
        self.assertEqual(rank, 3)

    def test_power_two_matches_inverse_square_far_above_floor(self) -> None:
        """For s >> lmbda, the power-2 filter is 1 / s^2."""
        s = np.array([10.0, 5.0, 1.0])
        filt, rank = tikhonov_filter(s, power=2)
        np.testing.assert_allclose(filt, 1.0 / np.square(s), rtol=1e-10)
        self.assertEqual(rank, 3)

    def test_drops_singular_values_at_or_below_relative_floor(self) -> None:
        """Singular values <= rtol * s[0] are treated as noise and zeroed."""
        s = np.array([1.0, 1e-13, 1e-16])  # floor = 1e-12 * 1.0 = 1e-12
        filt, rank = tikhonov_filter(s, power=1)
        self.assertEqual(rank, 1)
        self.assertEqual(filt[1], 0.0)
        self.assertEqual(filt[2], 0.0)
        np.testing.assert_allclose(filt[0], 1.0, rtol=1e-10)

    def test_power_one_is_bounded_where_naive_reciprocal_blows_up(self) -> None:
        """The repro: a near-floor singular value is damped, not amplified.

        With ``s[1] = 1.2e-12`` sitting between the floor ``lmbda = 1e-12`` and
        ``2 lmbda``, the naive pseudo-inverse ``1 / s`` (8.3e11) exceeds the
        Tikhonov bound ``1 / (2 lmbda) = 5e11``, whereas the filtered value stays
        under it. This is exactly the amplification that corrupted the Newton
        step for a rank-deficient ``A``.
        """
        s = np.array([1.0, 1.2e-12])
        lmbda = 1e-12 * s[0]
        filt, rank = tikhonov_filter(s, power=1)
        self.assertEqual(rank, 2)
        naive = 1.0 / s[1]
        self.assertGreater(naive, 1.0 / (2.0 * lmbda))  # naive overshoots
        self.assertLess(filt[1], naive)  # filter damps it
        self.assertLessEqual(float(np.max(filt)), 1.0 / (2.0 * lmbda))

    def test_power_two_is_bounded_near_floor(self) -> None:
        """The power-2 filter is bounded by 1 / (2 lmbda^2) at the floor."""
        s = np.array([1.0, 1.2e-12])
        lmbda_squared = (1e-12 * s[0]) ** 2
        filt, _ = tikhonov_filter(s, power=2)
        self.assertLessEqual(float(np.max(filt)), 1.0 / (2.0 * lmbda_squared))

    def test_rtol_parameter_controls_the_floor(self) -> None:
        """A looser rtol drops more of the spectrum."""
        s = np.array([1.0, 1e-6])
        _, rank_tight = tikhonov_filter(s, power=1, rtol=1e-8)
        _, rank_loose = tikhonov_filter(s, power=1, rtol=1e-4)
        self.assertEqual(rank_tight, 2)  # 1e-6 > 1e-8 floor -> kept
        self.assertEqual(rank_loose, 1)  # 1e-6 <= 1e-4 floor -> dropped

    def test_empty_spectrum_returns_empty_and_zero_rank(self) -> None:
        """An empty singular-value vector (zero-row A) has rank 0."""
        filt, rank = tikhonov_filter(np.array([]), power=1)
        self.assertEqual(rank, 0)
        self.assertEqual(filt.size, 0)

    def test_all_zero_spectrum_returns_zeros(self) -> None:
        """A spectrum with s[0] == 0 yields an all-zero filter and rank 0."""
        filt, rank = tikhonov_filter(np.zeros(3), power=2)
        self.assertEqual(rank, 0)
        np.testing.assert_array_equal(filt, np.zeros(3))

    def test_unsupported_power_raises(self) -> None:
        """Only power 1 and 2 are defined; anything else must fail loudly
        rather than silently fall through to the power-2 formula."""
        s = np.array([1.0, 0.5])
        for bad in (0, 3, -1):
            with self.assertRaises(ValueError):
                tikhonov_filter(s, power=bad)
