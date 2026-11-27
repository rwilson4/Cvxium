"""Tests for the banded, arrow and sparse-arrow linear-inequality layers.

Each assembles the barrier Hessian `A^T diag(-lmbda / f) A` as a structured
system (banded, or banded core plus rank-K arrow border). Each test checks the
structured assembly against the dense reference and the guard on a missing
`lmbda`.
"""

import unittest

import numpy as np
import numpy.typing as npt
import scipy.sparse

from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    LowRankUpdatedSystem,
    SparseArrowSystem,
    ZeroSystem,
)
from cvxium.problems.linear_constraints import (
    ArrowLinearInequalityLayer,
    BandedLinearInequalityLayer,
    BoundsConstraintLayer,
    LinearInequalityConstraintLayer,
    SparseArrowLinearInequalityLayer,
)
from cvxium.problems.objective import QuadraticObjective
from cvxium.problems.problem import Problem
from cvxium.solvers.barrier_solver import BarrierSolver
from cvxium.solvers.primal_dual_solver import PrimalDualSolver


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


def _epigraph_matrix(
    rng: np.random.Generator, n_slack: int, n_x: int
) -> scipy.sparse.csr_array:
    """Rows [slack part | x part], with slacks first (the layer's column order).

    - One row per slack t_j: g_j^T x - t_j, with g_j dense over x.
    - Rows over x alone (dose limits).
    - One budget row summing every slack (two core nonzeros or more: a heavy row).
    - One two-sided row touching one slack and x.
    """
    slack_rows = np.hstack([-np.eye(n_slack), rng.uniform(0.1, 1.0, (n_slack, n_x))])
    x_rows = np.hstack([np.zeros((3, n_slack)), rng.uniform(0.1, 1.0, (3, n_x))])
    budget = np.hstack([np.ones((1, n_slack)), np.zeros((1, n_x))])
    return scipy.sparse.csr_array(np.vstack([slack_rows, x_rows, budget]))


class SparseArrowLinearInequalityLayerTest(unittest.TestCase):
    """The sparse-arrow layer reproduces the dense barrier Hessian, and solves agree."""

    def test_hessian_contribution_matches_dense(self) -> None:
        rng = np.random.default_rng(11)
        n_slack, n_x = 7, 4
        A = _epigraph_matrix(rng, n_slack, n_x)
        m, n = A.shape
        w = rng.uniform(0.5, 1.0, n)
        Aw = A @ w
        # Two-sided on row 0, one-sided upper elsewhere, all strictly interior.
        cu = Aw + rng.uniform(0.5, 1.5, m)
        cl = np.full(m, -np.inf)
        cl[0] = Aw[0] - 0.7
        layer = SparseArrowLinearInequalityLayer(
            A, cl=cl, cu=cu, core_dimension=n_slack
        )
        self.assertEqual(layer.num_heavy_rows, 1)
        lam = rng.uniform(0.5, 1.5, layer.num_constraints(n))
        H = layer.hessian_contribution(w, lam)
        self.assertIsInstance(H, SparseArrowSystem)
        assert isinstance(H, SparseArrowSystem)
        self.assertIsInstance(H.upper_left, LowRankUpdatedSystem)

        f = layer.constraints(w)
        G = layer.grad_constraints(w)
        expected = G.T @ np.diag(-lam / f) @ G
        actual = H.multiply(np.eye(n))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        # Solvable once a bounds-like diagonal is added, as in a real problem.
        total = H.plus(DiagonalSystem(np.full(n, 0.1)))
        b = rng.standard_normal(n)
        np.testing.assert_allclose(
            total.solve(b), np.linalg.solve(expected + 0.1 * np.eye(n), b), rtol=1e-9
        )

    def test_no_heavy_rows_gives_diagonal_core(self) -> None:
        rng = np.random.default_rng(12)
        A = _epigraph_matrix(rng, 5, 3)[:-1]  # drop the budget row
        n = A.shape[1]
        w = rng.uniform(0.5, 1.0, n)
        layer = SparseArrowLinearInequalityLayer(A, cu=A @ w + 1.0, core_dimension=5)
        self.assertEqual(layer.num_heavy_rows, 0)
        H = layer.hessian_contribution(w, np.ones(A.shape[0]))
        assert isinstance(H, SparseArrowSystem)
        self.assertIsInstance(H.upper_left, DiagonalSystem)

    def test_rejects_bad_core_dimension(self) -> None:
        A = scipy.sparse.csr_array(np.ones((2, 3)))
        for M in (0, 3):
            with self.assertRaises(ValueError):
                SparseArrowLinearInequalityLayer(A, cu=np.ones(2), core_dimension=M)

    def test_hessian_contribution_requires_lmbda(self) -> None:
        A = scipy.sparse.csr_array(np.ones((2, 3)))
        layer = SparseArrowLinearInequalityLayer(A, cu=np.ones(2), core_dimension=1)
        with self.assertRaises(ValueError):
            layer.hessian_contribution(np.zeros(3))

    def test_solve_matches_unstructured_layer(self) -> None:
        # minimize 1/2 x^T Q x + q^T x + c^T t over slacks t and decisions x, with
        # t_j >= g_j^T x - h_j, a dose cap, a slack budget, and bounds: the sparse
        # arrow formulation must reach the dense formulation's optimum. With these
        # data every hinge row and the budget (the heavy row) bind, and the slacks
        # are strictly positive.
        rng = np.random.default_rng(13)
        n_slack, n_x = 8, 5
        A = _epigraph_matrix(rng, n_slack, n_x)
        m, n = A.shape
        cu = np.r_[np.full(n_slack, -0.05), np.full(3, 4.0), 2.0]
        F = rng.standard_normal((n_x, n_x))
        Q_x = F @ F.T + 0.1 * np.eye(n_x)
        c = np.r_[0.2 * np.ones(n_slack), -rng.uniform(0.5, 1.5, n_x)]
        Q_full = np.zeros((n, n))
        Q_full[n_slack:, n_slack:] = Q_x
        structured_q = SparseArrowSystem(
            ZeroSystem(n_slack), scipy.sparse.csr_array((n_slack, n_x)), Q_x
        )
        bounds = BoundsConstraintLayer(lb=0.0, dimension=n)
        structured = Problem.minimize(
            QuadraticObjective(c=c, hessian_contribution=structured_q)
        ).subject_to(
            ineq=[
                bounds,
                SparseArrowLinearInequalityLayer(A, cu=cu, core_dimension=n_slack),
            ]
        )
        reference = Problem.minimize(
            QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q_full))
        ).subject_to(ineq=[bounds, LinearInequalityConstraintLayer(A, cu=cu)])
        expected = reference.solve(solver=BarrierSolver()).solution
        for solver in (BarrierSolver(), PrimalDualSolver()):
            result = structured.solve(solver=solver)
            np.testing.assert_allclose(result.solution, expected, atol=1e-5)
