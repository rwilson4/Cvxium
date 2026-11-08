"""Test Objective subclasses."""

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    ZeroSystem,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import (
    LiftedObjective,
    LinearObjective,
    NegatedObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.solvers.barrier_solver import (
    BarrierSolver,
)


def _make_psd_matrix(rng: np.random.Generator, M: int) -> npt.NDArray[np.float64]:
    """Sample a random symmetric PD matrix with bounded condition number."""
    A = rng.standard_normal((M, M))
    return np.asarray(A @ A.T + M * np.eye(M))


class LinearObjectiveTest(unittest.TestCase):
    """Tests for `LinearObjective`."""

    @parameterized.expand(
        [
            (101, 10),
            (202, 50),
            (303, 200),
        ]
    )
    def test_evaluate(self, seed: int, M: int) -> None:
        """`evaluate(w)` returns ``c @ w``."""
        rng = np.random.default_rng(seed)
        c = rng.standard_normal(M)
        w = rng.standard_normal(M)
        objective = LinearObjective(c=c)
        np.testing.assert_allclose(
            objective.evaluate(w), float(c @ w), rtol=1e-6, atol=1e-12
        )

    @parameterized.expand([(104, 10), (205, 50)])
    def test_gradient(self, seed: int, M: int) -> None:
        """`gradient(w)` returns ``c`` regardless of ``w``."""
        rng = np.random.default_rng(seed)
        c = rng.standard_normal(M)
        w = rng.standard_normal(M)
        objective = LinearObjective(c=c)
        np.testing.assert_array_equal(objective.gradient(w), c)


class QuadraticObjectiveTest(unittest.TestCase):
    """Tests for `QuadraticObjective` (basic methods + dense baseline)."""

    @parameterized.expand([(601, 5), (602, 25), (603, 100)])
    def test_evaluate(self, seed: int, M: int) -> None:
        """`evaluate(w)` returns `(1/2) w^T Q w + c^T w`."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        w = rng.standard_normal(M)
        objective = QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q))
        expected = 0.5 * float(w @ Q @ w) + float(c @ w)
        np.testing.assert_allclose(objective.evaluate(w), expected, rtol=1e-6)

    @parameterized.expand([(611, 5), (612, 50)])
    def test_gradient(self, seed: int, M: int) -> None:
        """`gradient(w)` returns `Q w + c`."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        w = rng.standard_normal(M)
        objective = QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q))
        np.testing.assert_allclose(objective.gradient(w), Q @ w + c, rtol=1e-6)


class FeasibilityThresholdTest(unittest.TestCase):
    """`Objective.feasibility_threshold` and `with_feasibility_threshold`."""

    def test_default_is_none(self) -> None:
        """An ordinary objective has no feasibility threshold."""
        assert LinearObjective(c=np.zeros(3)).feasibility_threshold is None

    def test_with_feasibility_threshold_returns_a_copy(self) -> None:
        """``with_feasibility_threshold`` sets the threshold on a copy, type preserved."""
        base = LinearObjective(c=np.array([1.0, 2.0]))
        feas = base.with_feasibility_threshold(0.0)
        assert feas.feasibility_threshold == 0.0
        assert base.feasibility_threshold is None  # original untouched
        assert isinstance(feas, LinearObjective)  # concrete type preserved

    def test_copy_preserves_behavior_and_linearity(self) -> None:
        """The copy evaluates identically and keeps the native ``is_linear``."""
        rng = np.random.default_rng(7)
        c = rng.standard_normal(4)
        base = LinearObjective(c=c)
        feas = base.with_feasibility_threshold(-1.0)
        w = rng.standard_normal(4)
        np.testing.assert_allclose(
            feas.evaluate(w), base.evaluate(w), rtol=1e-6, atol=1e-12
        )
        assert feas.is_linear is True  # the fast LP dual path is preserved


class NegatedObjectiveTest(unittest.TestCase):
    """Tests for `NegatedObjective`, the ``-g`` wrapper behind maximize."""

    @staticmethod
    def _quadratic(seed: int, M: int) -> tuple[QuadraticObjective, DenseSystem]:
        """Build a PD-Hessian quadratic ``g`` and return it with its ``Q`` system."""
        rng = np.random.default_rng(seed)
        Q = DenseSystem(_make_psd_matrix(rng, M))
        c = rng.standard_normal(M)
        return QuadraticObjective(c=c, hessian_contribution=Q), Q

    def test_value_and_gradient_are_negated(self) -> None:
        """``-g`` reports the negated value and gradient of ``g``."""
        g, _ = self._quadratic(seed=1, M=6)
        neg = NegatedObjective(g)
        w = np.random.default_rng(2).standard_normal(6)
        self.assertAlmostEqual(neg.evaluate(w), -g.evaluate(w))
        np.testing.assert_allclose(neg.gradient(w), -g.gradient(w), rtol=1e-12)

    def test_dimension_delegates_to_wrapped(self) -> None:
        """``-g`` pins the wrapped objective's decision dimension."""
        g, _ = self._quadratic(seed=3, M=9)
        self.assertEqual(NegatedObjective(g).dimension, 9)

    def test_hessian_is_negative_definite(self) -> None:
        """Negating a PD (convex) Hessian yields an ND (concave) contribution."""
        g, _ = self._quadratic(seed=4, M=5)
        w = np.zeros(5)
        H_neg = NegatedObjective(g).hessian_contribution(w)
        self.assertFalse(H_neg.is_positive_definite())
        b = np.random.default_rng(5).standard_normal(5)
        np.testing.assert_allclose(
            H_neg.multiply(b), -g.hessian_contribution(w).multiply(b), rtol=1e-12
        )

    def test_double_negation_collapses_to_original_hessian(self) -> None:
        """Maximizing ``-g`` restores the exact PD system the solver factors.

        ``NegatedObjective(NegatedObjective(g))`` is the maximize round-trip; its
        Hessian contribution must be the original ``Q`` object, not a nested
        sign-carrier -- so the solver only ever factors the PD system.
        """
        g, Q = self._quadratic(seed=6, M=4)
        w = np.zeros(4)
        round_trip = NegatedObjective(NegatedObjective(g))
        self.assertIs(round_trip.hessian_contribution(w), Q)

    def test_negated_linear_hessian_stays_zero_system(self) -> None:
        """``-g`` of a linear objective keeps a ``ZeroSystem`` curvature.

        The ``-0 = 0`` special case keeps a negated linear objective composable
        in assembly rather than wrapping the zero in a sign-carrier.
        """
        g = LinearObjective(c=np.array([1.0, -2.0, 3.0]))
        H = NegatedObjective(g).hessian_contribution(np.zeros(3))
        self.assertIsInstance(H, ZeroSystem)

    def test_is_linear_forwards_wrapped_flag(self) -> None:
        """``-g`` is affine exactly when ``g`` is, keeping the fast LP dual path.

        The wrapper must forward ``is_linear`` -- a negated ``LinearObjective``
        stays linear, and only then do solvers dispatch the analytical LP dual
        path instead of the mini-Newton fallback.
        """
        linear = NegatedObjective(LinearObjective(c=np.array([1.0, -2.0, 3.0])))
        self.assertTrue(linear.is_linear)
        curved, _ = self._quadratic(seed=10, M=4)
        self.assertFalse(NegatedObjective(curved).is_linear)

    def test_at_negates_and_preserves_caching(self) -> None:
        """The per-point context negates value, gradient, and Hessian."""
        g, _ = self._quadratic(seed=7, M=5)
        w = np.random.default_rng(8).standard_normal(5)
        point = NegatedObjective(g).at(w)
        self.assertAlmostEqual(point.evaluate(), -g.evaluate(w))
        np.testing.assert_allclose(point.gradient(), -g.gradient(w), rtol=1e-12)
        self.assertFalse(point.hessian_contribution().is_positive_definite())

    def test_domain_bound_delegates(self) -> None:
        """Negation does not change the domain; the step bound is the wrapped one."""
        g, _ = self._quadratic(seed=9, M=4)
        w = np.zeros(4)
        dw = np.ones(4)
        self.assertEqual(
            NegatedObjective(g).btls_keep_in_domain(w, dw),
            g.btls_keep_in_domain(w, dw),
        )


class LiftedObjectiveTest(unittest.TestCase):
    """`LiftedObjective` sees the leading components and ignores the rest."""

    def test_value_gradient_and_diagonal_hessian(self) -> None:
        base = QuadraticObjective(
            c=np.array([1.0, -2.0]),
            hessian_contribution=DiagonalSystem(np.array([2.0, 4.0])),
        )
        lifted = LiftedObjective(base, extra_dimension=3)
        x = np.array([0.5, 1.5, 9.0, -9.0, 7.0])

        self.assertEqual(lifted.dimension, 5)
        self.assertAlmostEqual(lifted.evaluate(x), base.evaluate(x[:2]))
        np.testing.assert_allclose(
            lifted.gradient(x), np.concatenate([base.gradient(x[:2]), np.zeros(3)])
        )
        hessian = lifted.hessian_contribution(x)
        # A diagonal Hessian stays diagonal, so the structured assembly keeps
        # its fast path.
        self.assertIsInstance(hessian, DiagonalSystem)
        assert isinstance(hessian, DiagonalSystem)  # Satisfy type-checker
        np.testing.assert_allclose(hessian.eta, [2.0, 4.0, 0.0, 0.0, 0.0])

    def test_dense_hessian_is_embedded_block_diagonally(self) -> None:
        Q = np.array([[2.0, 1.0], [1.0, 2.0]])
        base = QuadraticObjective(c=np.zeros(2), hessian_contribution=DenseSystem(Q))
        lifted = LiftedObjective(base, extra_dimension=2)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            lifted.hessian_contribution(y).multiply(y), [4.0, 5.0, 0.0, 0.0]
        )

    def test_linearity_is_forwarded(self) -> None:
        self.assertTrue(LiftedObjective(LinearObjective(np.ones(2)), 1).is_linear)
        base = QuadraticObjective(
            c=np.zeros(2), hessian_contribution=DiagonalSystem(np.ones(2))
        )
        self.assertFalse(LiftedObjective(base, 1).is_linear)

    def test_extra_dimension_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            LiftedObjective(LinearObjective(np.ones(2)), 0)

    def test_epigraph_solve(self) -> None:
        r"""Minimize `\|w\|^2` over `[w; t]` with `w \ge 1 - t`, `t \ge 0`, `\sum t \le 1/2`.

        The budget is shared evenly, so `t = (1/4, 1/4)` and `w = (3/4, 3/4)`.
        """
        M, q = 2, 2
        base = QuadraticObjective(
            c=np.zeros(M), hessian_contribution=DiagonalSystem(np.full(M, 2.0))
        )
        problem = Problem.minimize(LiftedObjective(base, extra_dimension=q)).subject_to(
            ineq=[
                BoundsConstraintLayer(
                    lb=np.zeros(M + q), ub=np.full(M + q, 10.0), dimension=M + q
                ),
                LinearInequalityConstraintLayer(
                    A=np.array(
                        [[-1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, -1.0], [0, 0, 1, 1]]
                    ),
                    cu=np.array([-1.0, -1.0, 0.5]),
                ),
            ]
        )
        result = problem.solve(BarrierSolver(), x0=np.array([2.0, 2.0, 0.1, 0.1]))
        np.testing.assert_allclose(result.solution, [0.75, 0.75, 0.25, 0.25], atol=1e-5)
