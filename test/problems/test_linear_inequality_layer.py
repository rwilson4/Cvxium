"""Tests for LinearInequalityConstraintLayer."""

import unittest

import numpy as np
import scipy.sparse

from cvxium.problems.linear_constraints import (
    LinearInequalityConstraintLayer,
)


class LinearInequalityConstraintLayerCuOnlyTest(unittest.TestCase):
    """All rows finite-cu, no cl side."""

    def setUp(self) -> None:
        self.A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.cu = np.array([2.0, 3.0, 4.0])
        self.layer = LinearInequalityConstraintLayer(A=self.A, cu=self.cu)

    def test_num_constraints(self) -> None:
        self.assertEqual(self.layer.num_constraints(2), 3)
        self.assertEqual(self.layer.num_cu, 3)
        self.assertEqual(self.layer.num_cl, 0)

    def test_constraints_strictly_feasible(self) -> None:
        w = np.array([0.5, 0.5])
        cons = self.layer.constraints(w)
        self.assertTrue(np.all(cons < 0))
        np.testing.assert_allclose(cons, self.A @ w - self.cu)

    def test_constraints_with_shift(self) -> None:
        w = np.array([1.0, 1.0])
        cons = self.layer.constraints(w, shift=0.1)
        np.testing.assert_allclose(cons, (self.A @ w - self.cu) - 0.1)

    def test_grad_constraints_matches_A(self) -> None:
        w = np.array([0.0, 0.0])
        np.testing.assert_allclose(self.layer.grad_constraints(w), self.A)

    def test_grad_constraints_is_cached_and_read_only(self) -> None:
        """The constant Jacobian is formed once and cannot be mutated in place."""
        G = self.layer.grad_constraints(np.array([0.5, 0.5]))
        self.assertIs(self.layer.grad_constraints(np.array([1.0, -1.0])), G)
        with self.assertRaises(ValueError):
            G[0, 0] = 7.0
        np.testing.assert_array_equal(G, self.A)

    def test_grad_multiplies_match_default(self) -> None:
        w = np.array([0.5, 0.7])
        v = np.array([0.3, -0.2])
        u = np.array([0.4, -0.5, 0.6])
        np.testing.assert_allclose(
            self.layer.grad_constraints_multiply(w, v),
            self.layer.grad_constraints(w) @ v,
        )
        np.testing.assert_allclose(
            self.layer.grad_constraints_transpose_multiply(w, u),
            self.layer.grad_constraints(w).T @ u,
        )

    def test_btls_max_step_no_violation_when_dw_inactive(self) -> None:
        w = np.array([0.0, 0.0])
        dw = np.array([-1.0, -1.0])
        self.assertEqual(self.layer.btls_max_step(w, dw), np.inf)

    def test_btls_max_step_finite_when_dw_pushes_boundary(self) -> None:
        w = np.array([0.0, 0.0])
        dw = np.array([1.0, 1.0])
        np.testing.assert_allclose(self.layer.btls_max_step(w, dw), 2.0)

    def test_is_satisfied_strict(self) -> None:
        self.assertTrue(self.layer.is_satisfied(np.array([0.0, 0.0])))
        self.assertFalse(self.layer.is_satisfied(np.array([2.0, 0.0])))


class LinearInequalityConstraintLayerClOnlyTest(unittest.TestCase):
    """All rows finite-cl, no cu side."""

    def setUp(self) -> None:
        self.A = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.cl = np.array([0.0, 0.0])
        self.layer = LinearInequalityConstraintLayer(A=self.A, cl=self.cl)

    def test_num_constraints(self) -> None:
        self.assertEqual(self.layer.num_constraints(2), 2)
        self.assertEqual(self.layer.num_cu, 0)
        self.assertEqual(self.layer.num_cl, 2)

    def test_constraints_layout(self) -> None:
        w = np.array([0.5, 0.7])
        cons = self.layer.constraints(w)
        np.testing.assert_allclose(cons, self.cl - self.A @ w)

    def test_grad_constraints_negated(self) -> None:
        w = np.array([0.0, 0.0])
        np.testing.assert_allclose(self.layer.grad_constraints(w), -self.A)

    def test_btls_max_step_against_lower_bound(self) -> None:
        w = np.array([1.0, 1.0])
        dw = np.array([-1.0, 0.0])
        np.testing.assert_allclose(self.layer.btls_max_step(w, dw), 1.0)


class LinearInequalityConstraintLayerTwoSidedTest(unittest.TestCase):
    """Mix of both-finite, cu-only, and cl-only rows."""

    def setUp(self) -> None:
        self.A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.cl = np.array([-1.0, -np.inf, 0.5])
        self.cu = np.array([1.0, 2.0, np.inf])
        self.layer = LinearInequalityConstraintLayer(A=self.A, cl=self.cl, cu=self.cu)

    def test_only_finite_bound_rows_participate(self) -> None:
        self.assertEqual(self.layer.num_cu, 2)
        self.assertEqual(self.layer.num_cl, 2)
        self.assertEqual(self.layer.num_constraints(2), 4)
        cons = self.layer.constraints(np.array([0.4, 0.4]))
        self.assertEqual(cons.shape, (4,))
        self.assertTrue(np.all(cons < 0))

    def test_constraints_layout_cu_then_cl(self) -> None:
        w = np.array([0.0, 0.5])
        cons = self.layer.constraints(w)
        Ax = self.A @ w
        expected_cu = np.array([Ax[0] - 1.0, Ax[1] - 2.0])
        expected_cl = np.array([-1.0 - Ax[0], 0.5 - Ax[2]])
        np.testing.assert_allclose(cons, np.concatenate([expected_cu, expected_cl]))

    def test_grad_consistent_with_multiplies(self) -> None:
        w = np.array([0.2, 0.3])
        v = np.array([0.1, -0.4])
        u = np.array([0.5, -0.2, 0.3, -0.1])
        G = self.layer.grad_constraints(w)
        np.testing.assert_allclose(self.layer.grad_constraints_multiply(w, v), G @ v)
        np.testing.assert_allclose(
            self.layer.grad_constraints_transpose_multiply(w, u), G.T @ u
        )

    def test_btls_two_sided_takes_min(self) -> None:
        w = np.array([0.0, 0.0])
        dw = np.array([1.0, 1.0])
        self.assertAlmostEqual(self.layer.btls_max_step(w, dw), 1.0)


class LinearInequalityConstraintLayerSparseDenseTest(unittest.TestCase):
    """Sparse and dense `A` produce identical outputs at every entry point."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        m, n = 5, 8
        rows, cols = [], []
        for i in range(m):
            picks = rng.choice(n, size=3, replace=False)
            for j in picks:
                rows.append(i)
                cols.append(j)
        data = np.ones(len(rows), dtype=np.float64)
        self.A_sparse = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(m, n), dtype=np.float64
        )
        self.A_dense = np.asarray(self.A_sparse.toarray(), dtype=np.float64)
        self.cl = np.array([-1.0, -np.inf, -0.5, -2.0, -np.inf])
        self.cu = np.array([3.0, 2.0, np.inf, 4.0, 1.5])
        self.dense_layer = LinearInequalityConstraintLayer(
            A=self.A_dense, cl=self.cl, cu=self.cu
        )
        self.sparse_layer = LinearInequalityConstraintLayer(
            A=self.A_sparse, cl=self.cl, cu=self.cu
        )

    def test_constraints_match(self) -> None:
        w = np.full(8, 0.1)
        np.testing.assert_allclose(
            self.dense_layer.constraints(w), self.sparse_layer.constraints(w)
        )

    def test_grad_constraints_match(self) -> None:
        w = np.zeros(8)
        np.testing.assert_allclose(
            self.dense_layer.grad_constraints(w),
            self.sparse_layer.grad_constraints(w),
        )

    def test_grad_multiplies_match(self) -> None:
        rng = np.random.default_rng(1)
        w = rng.standard_normal(8)
        v = rng.standard_normal(8)
        u = rng.standard_normal(self.dense_layer.num_constraints(8))
        np.testing.assert_allclose(
            self.dense_layer.grad_constraints_multiply(w, v),
            self.sparse_layer.grad_constraints_multiply(w, v),
        )
        np.testing.assert_allclose(
            self.dense_layer.grad_constraints_transpose_multiply(w, u),
            self.sparse_layer.grad_constraints_transpose_multiply(w, u),
        )


class LinearInequalityConstraintLayerValidationTest(unittest.TestCase):
    """Constructor-time validation."""

    def test_rejects_free_row(self) -> None:
        A = np.eye(2)
        with self.assertRaisesRegex(ValueError, "no constraint"):
            LinearInequalityConstraintLayer(
                A=A,
                cl=np.array([-np.inf, 0.0]),
                cu=np.array([np.inf, 1.0]),
            )

    def test_rejects_all_infinite(self) -> None:
        A = np.eye(2)
        with self.assertRaisesRegex(ValueError, "no constraint|at least one finite"):
            LinearInequalityConstraintLayer(
                A=A,
                cl=np.array([-np.inf, -np.inf]),
                cu=np.array([np.inf, np.inf]),
            )

    def test_rejects_degenerate_box(self) -> None:
        A = np.eye(2)
        with self.assertRaisesRegex(ValueError, "interior is empty"):
            LinearInequalityConstraintLayer(
                A=A,
                cl=np.array([1.0, 0.0]),
                cu=np.array([1.0, 1.0]),
            )

    def test_rejects_shape_mismatch(self) -> None:
        A = np.eye(3)
        with self.assertRaisesRegex(ValueError, "does not match `A` rows"):
            LinearInequalityConstraintLayer(A=A, cu=np.array([1.0, 2.0]))
