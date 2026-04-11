"""Test matrix multiply helpers."""

import time

import numpy as np
import pytest

from cvxium.numerical_helpers import (
    multiply_arrow_sparsity_pattern,
    multiply_banded,
    multiply_block,
    multiply_block_plus_one,
    multiply_diagonal,
    multiply_rank_one_update,
    multiply_rank_p_update,
)


@pytest.mark.parametrize(
    "seed,M",
    [
        (1101, 100),
        (1201, 200),
        (1301, 50),
        (1401, 500),
        (1501, 13),
    ],
)
def test_multiply_diagonal(seed: int, M: int) -> None:
    """Test computing H@y by doing it the slow way."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    y = np.random.randn(M)

    H = np.diag(eta)

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_diagonal(y, eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (1102, 100, 20),
        (1202, 200, 30),
        (1302, 50, 5),
        (1402, 500, 100),
        (1502, 13, 3),
    ],
)
def test_multiply_diagonal_multiple_rhs(seed: int, M: int, p: int) -> None:
    """Test computing H@Y for a matrix Y by doing it the slow way."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    Y = np.random.randn(M, p)

    H = np.diag(eta)

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_diagonal(Y, eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M",
    [
        (1103, 100),
        (1203, 200),
        (1303, 50),
        (1403, 500),
        (1503, 13),
    ],
)
def test_multiply_arrow_sparsity_pattern(seed: int, M: int) -> None:
    """Test computing H@y by doing it the slow way."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    zeta = np.random.randn(M)
    theta = np.dot(zeta / eta, zeta) + 1.0
    y = np.random.randn(M + 1)

    H = np.zeros((M + 1, M + 1))
    H[0:M, 0:M] = np.diag(eta)
    H[M, 0:M] = zeta
    H[0:M, M] = zeta
    H[M, M] = theta

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_arrow_sparsity_pattern(y, eta, zeta, theta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (1104, 100, 20),
        (1204, 200, 30),
        (1304, 50, 5),
        (1404, 500, 100),
        (1504, 13, 3),
    ],
)
def test_multiply_arrow_sparsity_pattern_multiple_rhs(
    seed: int, M: int, p: int
) -> None:
    """Test computing H@Y for a matrix Y by doing it the slow way."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    zeta = np.random.randn(M)
    theta = np.dot(zeta / eta, zeta) + 1.0
    Y = np.random.randn(M + 1, p)

    H = np.zeros((M + 1, M + 1))
    H[0:M, 0:M] = np.diag(eta)
    H[M, 0:M] = zeta
    H[0:M, M] = zeta
    H[M, M] = theta

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_arrow_sparsity_pattern(Y, eta, zeta, theta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M",
    [
        (1105, 100),
        (1205, 200),
        (1305, 50),
        (1405, 500),
        (1505, 13),
    ],
)
def test_multiply_diagonal_plus_rank_one(seed: int, M: int) -> None:
    """Test computing H@y for H = diag(eta) + kappa*kappa^T."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    kappa = np.random.randn(M)
    H = np.diag(eta) + np.outer(kappa, kappa)
    y = np.random.randn(M)

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_rank_one_update(y, kappa, A_multiply=multiply_diagonal, eta=eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (1106, 100, 20),
        (1206, 200, 30),
        (1306, 50, 5),
        (1406, 500, 100),
        (1506, 13, 3),
    ],
)
def test_multiply_diagonal_plus_rank_one_multiple_rhs(
    seed: int, M: int, p: int
) -> None:
    """Test computing H@Y for H = diag(eta) + kappa*kappa^T, matrix Y."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    kappa = np.random.randn(M)
    H = np.diag(eta) + np.outer(kappa, kappa)
    Y = np.random.randn(M, p)

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_rank_one_update(Y, kappa, A_multiply=multiply_diagonal, eta=eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,r",
    [
        (1107, 100, 3),
        (1207, 200, 10),
        (1307, 50, 2),
        (1407, 500, 10),
        (1507, 13, 2),
    ],
)
def test_multiply_arrow_plus_rank_p(seed: int, M: int, r: int) -> None:
    """Test computing H@y for H = arrow + kappa*kappa^T."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    zeta = np.random.randn(M)
    theta = np.dot(zeta / eta, zeta) + 1.0
    kappa = np.random.randn(M + 1, r)

    A = np.zeros((M + 1, M + 1))
    A[0:M, 0:M] = np.diag(eta)
    A[M, 0:M] = zeta
    A[0:M, M] = zeta
    A[M, M] = theta

    H = A + kappa @ kappa.T
    y = np.random.randn(M + 1)

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_rank_p_update(
        y,
        kappa,
        A_multiply=multiply_arrow_sparsity_pattern,
        eta=eta,
        zeta=zeta,
        theta=theta,
    )
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,r,p",
    [
        (1108, 100, 3, 20),
        (1208, 200, 10, 30),
        (1308, 50, 2, 5),
        (1408, 500, 10, 50),
        (1508, 13, 2, 4),
    ],
)
def test_multiply_arrow_plus_rank_p_multiple_rhs(
    seed: int, M: int, r: int, p: int
) -> None:
    """Test computing H@Y for H = arrow + kappa*kappa^T, matrix Y."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    zeta = np.random.randn(M)
    theta = np.dot(zeta / eta, zeta) + 1.0
    kappa = np.random.randn(M + 1, r)

    A = np.zeros((M + 1, M + 1))
    A[0:M, 0:M] = np.diag(eta)
    A[M, 0:M] = zeta
    A[0:M, M] = zeta
    A[M, M] = theta

    H = A + kappa @ kappa.T
    Y = np.random.randn(M + 1, p)

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_rank_p_update(
        Y,
        kappa,
        A_multiply=multiply_arrow_sparsity_pattern,
        eta=eta,
        zeta=zeta,
        theta=theta,
    )
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M",
    [
        (1109, 100),
        (1209, 200),
        (1309, 50),
        (1409, 500),
        (1509, 13),
    ],
)
def test_multiply_block_plus_one(seed: int, M: int) -> None:
    """Test computing H@y for a block-plus-one structured H."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    A12 = np.random.randn(M)
    A22 = float(np.random.rand() + 1.0)

    H = np.zeros((M + 1, M + 1))
    H[0:M, 0:M] = np.diag(eta)
    H[0:M, M] = A12
    H[M, 0:M] = A12
    H[M, M] = A22

    y = np.random.randn(M + 1)

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_block_plus_one(y, A12, A22, A11_multiply=multiply_diagonal, eta=eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,p",
    [
        (1110, 100, 20),
        (1210, 200, 30),
        (1310, 50, 5),
        (1410, 500, 100),
        (1510, 13, 3),
    ],
)
def test_multiply_block_plus_one_multiple_rhs(seed: int, M: int, p: int) -> None:
    """Test computing H@Y for block-plus-one H, matrix Y."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    A12 = np.random.randn(M)
    A22 = float(np.random.rand() + 1.0)

    H = np.zeros((M + 1, M + 1))
    H[0:M, 0:M] = np.diag(eta)
    H[0:M, M] = A12
    H[M, 0:M] = A12
    H[M, M] = A22

    Y = np.random.randn(M + 1, p)

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_block_plus_one(Y, A12, A22, A11_multiply=multiply_diagonal, eta=eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,r",
    [
        (1111, 100, 5),
        (1211, 200, 8),
        (1311, 50, 3),
        (1411, 300, 10),
        (1511, 13, 2),
    ],
)
def test_multiply_block(seed: int, M: int, r: int) -> None:
    """Test computing H@y for a general block-structured H."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    A12 = np.random.randn(M, r)
    # Make A22 symmetric positive definite
    tmp = np.random.randn(r, r)
    A22 = tmp @ tmp.T + np.eye(r)

    H = np.zeros((M + r, M + r))
    H[0:M, 0:M] = np.diag(eta)
    H[0:M, M:] = A12
    H[M:, 0:M] = A12.T
    H[M:, M:] = A22

    y = np.random.randn(M + r)

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_block(y, A12, A22, A11_multiply=multiply_diagonal, eta=eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,r,p",
    [
        (1112, 100, 5, 20),
        (1212, 200, 8, 30),
        (1312, 50, 3, 5),
        (1412, 300, 10, 50),
        (1512, 13, 2, 4),
    ],
)
def test_multiply_block_multiple_rhs(seed: int, M: int, r: int, p: int) -> None:
    """Test computing H@Y for a general block-structured H, matrix Y."""
    np.random.seed(seed)
    eta = np.random.rand(M) + 1.0
    A12 = np.random.randn(M, r)
    tmp = np.random.randn(r, r)
    A22 = tmp @ tmp.T + np.eye(r)

    H = np.zeros((M + r, M + r))
    H[0:M, 0:M] = np.diag(eta)
    H[0:M, M:] = A12
    H[M:, 0:M] = A12.T
    H[M:, M:] = A22

    Y = np.random.randn(M + r, p)

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_block(Y, A12, A22, A11_multiply=multiply_diagonal, eta=eta)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


def _make_banded(M: int, bw: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a random symmetric positive-definite banded matrix and its ab storage."""
    np.random.seed(seed)
    # Build dense symmetric positive-definite banded matrix
    H = np.zeros((M, M))
    for d in range(bw + 1):
        vals = np.random.randn(M - d)
        H += np.diag(vals, d)
        if d > 0:
            H += np.diag(vals, -d)
    # Ensure positive definiteness
    H += (bw + 1) * np.eye(M)

    # Convert to upper banded storage
    ab = np.zeros((bw + 1, M))
    for d in range(bw + 1):
        ab[bw - d, d:] = np.diag(H, d)

    return H, ab


@pytest.mark.parametrize(
    "seed,M,bw",
    [
        (1113, 100, 3),
        (1213, 200, 5),
        (1313, 50, 2),
        (1413, 500, 8),
        (1513, 13, 4),
    ],
)
def test_multiply_banded(seed: int, M: int, bw: int) -> None:
    """Test computing H@y for a banded symmetric matrix."""
    H, ab = _make_banded(M, bw, seed)
    np.random.seed(seed + 1)
    y = np.random.randn(M)

    st = time.time()
    z_expected = H @ y
    mt = time.time()
    z = multiply_banded(y, ab)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,bw,p",
    [
        (1114, 100, 3, 20),
        (1214, 200, 5, 30),
        (1314, 50, 2, 5),
        (1414, 500, 8, 10),
        (1514, 13, 4, 3),
    ],
)
def test_multiply_banded_multiple_rhs(seed: int, M: int, bw: int, p: int) -> None:
    """Test computing H@Y for a banded symmetric matrix, matrix Y."""
    H, ab = _make_banded(M, bw, seed)
    np.random.seed(seed + 1)
    Y = np.random.randn(M, p)

    st = time.time()
    Z_expected = H @ Y
    mt = time.time()
    Z = multiply_banded(Y, ab)
    et = time.time()

    print(f"Slow way completed in {1e6 * (mt - st):.03f} us")
    print(f"Fast way completed in {1e6 * (et - mt):.03f} us")

    np.testing.assert_allclose(Z, Z_expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "seed,M,bw",
    [
        (1115, 100, 3),
        (1215, 200, 5),
        (1315, 50, 2),
        (1415, 500, 8),
        (1515, 13, 4),
    ],
)
def test_multiply_banded_lower(seed: int, M: int, bw: int) -> None:
    """Test computing H@y for a banded matrix stored in lower form."""
    H, ab_upper = _make_banded(M, bw, seed)
    np.random.seed(seed + 1)
    y = np.random.randn(M)

    # Convert upper banded to lower banded storage
    ab_lower = np.zeros((bw + 1, M))
    for d in range(bw + 1):
        ab_lower[d, : M - d] = np.diag(H, -d)

    z_expected = H @ y
    z = multiply_banded(y, ab_lower, lower=True)

    np.testing.assert_allclose(z, z_expected, rtol=1e-12, atol=1e-12)
