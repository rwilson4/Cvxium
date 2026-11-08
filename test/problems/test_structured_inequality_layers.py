"""Tests for the banded and arrow one-sided linear-inequality layers.

Both assemble the barrier Hessian `A^T diag(-lmbda / f) A` as a structured
system (banded, or banded core plus rank-K arrow border). Each test checks the
structured assembly against the dense reference and the guard on a missing
`lmbda`.
"""

import unittest

import numpy as np
import numpy.typing as npt
import scipy.sparse

from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.linear_constraints import (
    ArrowLinearInequalityLayer,
    BandedLinearInequalityLayer,
)


def _difference_matrix(pairs: list[tuple[int, int]], n: int) -> npt.NDArray[np.float64]:
    """Dense difference matrix: row r is +1 at pairs[r][0], -1 at pairs[r][1]."""
    A = np.zeros((len(pairs), n), dtype=np.float64)
    for r, (a, b) in enumerate(pairs):
        A[r, a] = 1.0
        A[r, b] = -1.0
    return A


class BandedLinearInequalityLayerTest(unittest.TestCase):
    """The banded layer reproduces the dense barrier Hessian."""

    def test_hessian_contribution_matches_dense(self) -> None:
        rng = np.random.default_rng(2026)
        n = 8
        pairs = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (0, 2),
            (2, 4),
            (4, 6),
        ]
        A = _difference_matrix(pairs, n)
        w = rng.standard_normal(n)
        slack = rng.uniform(0.5, 1.5, len(pairs))
        cu = A @ w + slack  # so f = A w - cu = -slack < 0 (interior)
        # ridge 0 so the assembly equals the dense Hessian exactly.
        layer = BandedLinearInequalityLayer(
            scipy.sparse.csr_matrix(A), cu=cu, ridge_scale=0.0, ridge_floor=0.0
        )

        lam = rng.uniform(0.5, 1.5, len(pairs))
        wgt = -lam / (A @ w - cu)  # = lam / slack > 0
        eta = rng.uniform(0.5, 2.0, n)
        system = layer.hessian_contribution(w, lam).plus(DiagonalSystem(eta))
        H = np.diag(eta) + A.T @ np.diag(wgt) @ A

        rhs = rng.standard_normal(n)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    def test_hessian_contribution_requires_lmbda(self) -> None:
        A = _difference_matrix([(0, 1), (1, 2)], 3)
        layer = BandedLinearInequalityLayer(A, cu=np.ones(2))
        with self.assertRaisesRegex(ValueError, "requires `lmbda`"):
            layer.hessian_contribution(np.zeros(3))

    def test_ridge_keeps_singular_hessian_solvable(self) -> None:
        """The default ridge makes the singular Laplacian Hessian solvable.

        A difference-constraint Hessian A^T W A is a graph Laplacian, singular on
        the constant vector. With no diagonal base and the default ridge, the
        banded Cholesky must still succeed (this is the DualLPMatcher-at-scale
        failure mode: a generous box leaves negligible diagonal regularization).
        """
        rng = np.random.default_rng(11)
        n = 6
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]  # path Laplacian
        A = _difference_matrix(pairs, n)
        w = rng.standard_normal(n)
        cu = A @ w + rng.uniform(0.5, 1.5, len(pairs))
        layer = BandedLinearInequalityLayer(A, cu=cu)  # default ridge
        lam = rng.uniform(0.5, 1.5, len(pairs))
        system = layer.hessian_contribution(w, lam)  # no diagonal base
        rhs = rng.standard_normal(n)
        x = system.solve(rhs)
        self.assertTrue(np.all(np.isfinite(x)))
        np.testing.assert_allclose(system.multiply(x), rhs, rtol=1e-4, atol=1e-6)


class ArrowLinearInequalityLayerTest(unittest.TestCase):
    """The arrow layer reproduces the dense banded-core-plus-border Hessian."""

    def test_hessian_contribution_matches_dense(self) -> None:
        rng = np.random.default_rng(7)
        M, K = 6, 2
        n = M + K
        # Each row: a core difference (two core columns) plus one border column.
        core_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)]
        border_of = [0, 1, 0, 1, 0, 1, 0]
        A = np.zeros((len(core_pairs), n), dtype=np.float64)
        for r, (a, b) in enumerate(core_pairs):
            A[r, a] = 1.0
            A[r, b] = -1.0
            A[r, M + border_of[r]] = 1.0
        w = rng.standard_normal(n)
        slack = rng.uniform(0.5, 1.5, len(core_pairs))
        cu = A @ w + slack
        # ridge 0 so the assembly equals the dense Hessian exactly.
        layer = ArrowLinearInequalityLayer(
            scipy.sparse.csr_matrix(A),
            cu=cu,
            core_dimension=M,
            ridge_scale=0.0,
            ridge_floor=0.0,
        )
        lam = rng.uniform(0.5, 1.5, len(core_pairs))
        wgt = -lam / (A @ w - cu)
        eta = rng.uniform(0.5, 2.0, n)
        system = layer.hessian_contribution(w, lam).plus(DiagonalSystem(eta))
        H = np.diag(eta) + A.T @ np.diag(wgt) @ A

        rhs = rng.standard_normal(n)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    def test_ridge_keeps_singular_core_solvable(self) -> None:
        """With a singular (Laplacian) core, the default ridge still solves."""
        rng = np.random.default_rng(11)
        M, K = 4, 1
        n = M + K
        core_pairs = [(0, 1), (1, 2), (2, 3)]  # path Laplacian core: singular
        A = np.zeros((len(core_pairs), n), dtype=np.float64)
        for r, (a, b) in enumerate(core_pairs):
            A[r, a] = 1.0
            A[r, b] = -1.0
            A[r, M] = 1.0
        w = rng.standard_normal(n)
        cu = A @ w + rng.uniform(0.5, 1.5, len(core_pairs))
        layer = ArrowLinearInequalityLayer(A, cu=cu, core_dimension=M)
        lam = rng.uniform(0.5, 1.5, len(core_pairs))
        system = layer.hessian_contribution(w, lam)
        rhs = rng.standard_normal(n)
        x = system.solve(rhs)
        self.assertTrue(np.all(np.isfinite(x)))
        np.testing.assert_allclose(system.multiply(x), rhs, rtol=1e-4, atol=1e-6)

    def test_rejects_bad_core_dimension(self) -> None:
        A = _difference_matrix([(0, 1), (1, 2)], 3)
        with self.assertRaisesRegex(ValueError, "core_dimension"):
            ArrowLinearInequalityLayer(A, cu=np.ones(2), core_dimension=3)

    def test_hessian_contribution_requires_lmbda(self) -> None:
        A = np.zeros((2, 3), dtype=np.float64)
        A[0, 0], A[0, 1], A[0, 2] = 1.0, -1.0, 1.0
        A[1, 1], A[1, 0], A[1, 2] = 1.0, -1.0, 1.0
        layer = ArrowLinearInequalityLayer(A, cu=np.ones(2), core_dimension=2)
        with self.assertRaisesRegex(ValueError, "requires `lmbda`"):
            layer.hessian_contribution(np.zeros(3))
