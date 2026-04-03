"""Test numerical helpers."""

import time

import numpy as np
import numpy.typing as npt
import pytest
from scipy import linalg

from cvxium.numerical_helpers import (
    solve_arrow_sparsity_pattern,
    solve_banded,
    solve_block_plus_one,
    solve_diagonal,
    solve_kkt_system,
    solve_rank_one_update,
    solve_rank_p_update,
    solve_with_schur,
)


@pytest.mark.parametrize(
    "seed,M",
    [
        (101, 100),
        (201, 200),
        (301, 50),
        (401, 500),
        (501, 13),
    ],
)
def test_solve_diagonal(seed: int, M: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (102, 100, 20),
        (202, 200, 30),
        (302, 50, 5),
        (402, 500, 100),
        (502, 13, 3),
    ],
)
def test_solve_diagonal_multiple_rhs(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M",
    [
        (103, 100),
        (203, 200),
        (303, 50),
        (403, 500),
        (503, 13),
    ],
)
def test_solve_arrow_sparsity_pattern(seed: int, M: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (104, 100, 20),
        (204, 200, 30),
        (304, 50, 5),
        (404, 500, 100),
        (504, 13, 3),
    ],
)
def test_solve_arrow_sparsity_pattern_multiple_rhs(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M",
    [
        (105, 100),
        (205, 200),
        (305, 50),
        (405, 500),
        (505, 13),
    ],
)
def test_solve_diagonal_plus_rank_one(seed: int, M: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (106, 100, 20),
        (206, 200, 30),
        (306, 50, 5),
        (406, 500, 100),
        (506, 13, 3),
    ],
)
def test_solve_diagonal_plus_rank_one_multiple_rhs(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (107, 100, 3),
        (207, 200, 10),
        (307, 50, 2),
        (407, 500, 10),
        (507, 13, 2),
    ],
)
def test_solve_arrow_plus_rank_p(seed: int, M: int, p: int) -> None:
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
        b, kappa, A_solve=solve_arrow_sparsity_pattern, eta=eta, zeta=zeta, theta=theta
    )
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

    # Verify H*x = b
    np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "seed,M,p,q",
    [
        (108, 100, 3, 20),
        (208, 200, 10, 30),
        (308, 50, 2, 5),
        (408, 500, 10, 100),
        (508, 13, 2, 3),
    ],
)
def test_solve_arrow_plus_rank_p_multiple_rhs(
    seed: int, M: int, p: int, q: int
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
        b, kappa, A_solve=solve_arrow_sparsity_pattern, eta=eta, zeta=zeta, theta=theta
    )
    et = time.time()

    print(f"Slow way completed in {1e3 * (mt - st):.03f} ms")
    print(f"Fast way completed in {1e3 * (et - mt):.03f} ms")

    np.testing.assert_allclose(x, x_expected, rtol=1e-8, atol=1e-8)

    # Verify H*x = b
    np.testing.assert_allclose(H @ x, b, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "seed,M",
    [
        (109, 100),
        (209, 200),
        (309, 50),
        (409, 500),
        (509, 13),
    ],
)
def test_solve_block_plus_one(seed: int, M: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (110, 100, 20),
        (210, 200, 30),
        (310, 50, 5),
        (410, 500, 100),
        (510, 13, 3),
    ],
)
def test_solve_block_plus_one_multiple_rhs(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (111, 100, 3),
        (211, 200, 5),
        (311, 50, 2),
        (411, 500, 10),
        (511, 13, 2),
    ],
)
def test_solve_with_schur(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p,q",
    [
        (112, 100, 3, 20),
        (212, 200, 5, 30),
        (312, 50, 2, 5),
        (412, 500, 10, 100),
        (512, 13, 2, 3),
    ],
)
def test_solve_with_schur_multiple_rhs(seed: int, M: int, p: int, q: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (113, 100, 20),
        (213, 200, 30),
        (313, 50, 5),
        (413, 500, 100),
        (513, 13, 3),
    ],
)
def test_solve_kkt_system_hessian_diagonal(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (114, 100, 20),
        (214, 200, 30),
        (314, 50, 5),
        (414, 500, 100),
        (514, 13, 3),
    ],
)
def test_solve_kkt_system_hessian_diagonal_plus_rank_one(
    seed: int, M: int, p: int
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (115, 100, 20),
        (215, 200, 30),
        (315, 50, 5),
        (415, 500, 100),
        (515, 13, 3),
    ],
)
def test_solve_kkt_system_hessian_arrow_sparsity_pattern(
    seed: int, M: int, p: int
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (116, 100, 20),
        (216, 200, 30),
        (317, 50, 5),
        (416, 500, 100),
        (516, 13, 3),
    ],
)
def test_solve_kkt_system_hessian_diagonal_plus_rank_one_rank_deficient(
    seed: int, M: int, p: int
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


def _make_spd_banded(
    n: int, p: int, rng: np.random.RandomState
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (H, ab_upper, ab_lower) for a random SPD banded matrix.

    H is the dense n-by-n matrix.  ab_upper and ab_lower are the upper and lower
    banded storage forms accepted by scipy.linalg.cholesky_banded.
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


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (117, 100, 3),
        (217, 200, 5),
        (317, 50, 2),
        (417, 500, 10),
        (517, 13, 2),
    ],
)
def test_solve_banded(seed: int, M: int, p: int) -> None:
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


@pytest.mark.parametrize(
    "seed,M,p,q",
    [
        (118, 100, 3, 20),
        (218, 200, 5, 30),
        (318, 50, 2, 5),
        (418, 500, 10, 100),
        (518, 13, 2, 3),
    ],
)
def test_solve_banded_multiple_rhs(seed: int, M: int, p: int, q: int) -> None:
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
