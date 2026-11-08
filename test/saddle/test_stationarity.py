"""Tests for two-block stationarity: residual, barrier signs, and Newton step."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium.linalg.systems import (
    CoupledSaddleSystem,
    EqualityCoupledSaddleSystem,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
)
from cvxium.saddle.stationarity import (
    QuadraticSaddleObjective,
    StationarityProblem,
)

if TYPE_CHECKING:
    from cvxium.problems.constraint_layers import (
        ConstraintLayer,
    )


def _random_spd(dimension: int, seed: int) -> npt.NDArray[np.float64]:
    """Return a random ``(dimension, dimension)`` SPD matrix ``A^T A + n I``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dimension, dimension))
    return A.T @ A + dimension * np.eye(dimension)


def _random_quadratic(
    n: int,
    m: int,
    seed: int,
) -> QuadraticSaddleObjective:
    """Build a random ``QuadraticSaddleObjective`` of block sizes ``(n, m)``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(m)
    P = _random_spd(n, seed=seed + 1)
    R = _random_spd(m, seed=seed + 2)
    C = rng.standard_normal((n, m))
    return QuadraticSaddleObjective(a, b, P, C, R)


def _box(
    center: npt.NDArray[np.float64],
    half_width: float,
) -> BoundsConstraintLayer:
    """Return a box layer strictly containing ``center`` by ``half_width``."""
    return BoundsConstraintLayer(
        lb=center - half_width,
        ub=center + half_width,
    )


def _barrier_log_sum(
    domain: list[ConstraintLayer],
    w: npt.NDArray[np.float64],
) -> float:
    r"""Return `\sum_i \log(-f_i(w))` over the layers in ``domain``."""
    total = 0.0
    for layer in domain:
        total += float(np.sum(np.log(-layer.constraints(w))))
    return total


class StationarityTest(unittest.TestCase):
    """Tests for ``QuadraticSaddleObjective`` and ``StationarityProblem``."""

    @parameterized.expand(
        [
            ("skinny", 30, 4),
            ("small", 5, 3),
            ("medium", 12, 2),
        ]
    )
    def test_gradients_match_finite_difference(
        self, _name: str, n: int, m: int
    ) -> None:
        r"""Central-difference of ``evaluate`` matches ``gradient_x``/``gradient_y``."""
        obj = _random_quadratic(n, m, seed=n * 100 + m)
        rng = np.random.default_rng(hash((n, m)) % (2**32))
        x = rng.standard_normal(n)
        y = rng.standard_normal(m)
        eps = 1e-6

        fd_x = np.zeros(n)
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps
            fd_x[i] = (obj.evaluate(x + e, y) - obj.evaluate(x - e, y)) / (2 * eps)
        fd_y = np.zeros(m)
        for j in range(m):
            e = np.zeros(m)
            e[j] = eps
            fd_y[j] = (obj.evaluate(x, y + e) - obj.evaluate(x, y - e)) / (2 * eps)

        np.testing.assert_allclose(obj.gradient_x(x, y), fd_x, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(obj.gradient_y(x, y), fd_y, rtol=1e-6, atol=1e-7)

    def test_residual_unconstrained_is_bare_gradient(self) -> None:
        r"""With empty domains, the residual is the concatenated raw gradient."""
        n, m = 8, 3
        obj = _random_quadratic(n, m, seed=17)
        problem = StationarityProblem(objective=obj, x_domain=[], y_domain=[])
        rng = np.random.default_rng(555)
        x = rng.standard_normal(n)
        y = rng.standard_normal(m)
        expected = np.concatenate([obj.gradient_x(x, y), obj.gradient_y(x, y)])
        for t in (0.5, 1.0, 37.0):
            np.testing.assert_allclose(
                problem.residual(x, y, t), expected, rtol=1e-12, atol=1e-12
            )

    def test_residual_matches_fd_of_barrier_smoothed_objective(self) -> None:
        r"""The residual is the gradient of the barrier-smoothed potential.

        The potential consistent with the SUBTRACT-for-x / ADD-for-y barrier
        signs is
           Phi_t = zeta + (1/t) sum log(-f^X) - (1/t) sum log(-f^Y),
        so d Phi_t / dx == r_x and d Phi_t / dy == r_y. (This is the opposite
        sign on each log-sum from a naive reading -- the concave x-block
        subtracts its barrier gradient while the convex y-block adds it.)
        """
        n, m = 6, 3
        obj = _random_quadratic(n, m, seed=71)
        x0 = np.full(n, 0.25)
        y0 = np.full(m, -0.15)
        x_domain: list[ConstraintLayer] = [_box(x0, 1.0)]
        y_domain: list[ConstraintLayer] = [_box(y0, 0.8)]
        problem = StationarityProblem(
            objective=obj, x_domain=x_domain, y_domain=y_domain
        )
        t = 3.0

        def phi(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
            barrier = _barrier_log_sum(x_domain, x) - _barrier_log_sum(y_domain, y)
            return obj.evaluate(x, y) + (1.0 / t) * barrier

        eps = 1e-6
        fd_x = np.zeros(n)
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps
            fd_x[i] = (phi(x0 + e, y0) - phi(x0 - e, y0)) / (2 * eps)
        fd_y = np.zeros(m)
        for j in range(m):
            e = np.zeros(m)
            e[j] = eps
            fd_y[j] = (phi(x0, y0 + e) - phi(x0, y0 - e)) / (2 * eps)

        r = problem.residual(x0, y0, t)
        np.testing.assert_allclose(r[:n], fd_x, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(r[n:], fd_y, rtol=1e-5, atol=1e-6)

    def test_build_coupled_system_matches_dense_jacobian(self) -> None:
        r"""The assembled saddle equals the numerical Jacobian of ``residual``."""
        n, m = 7, 3
        obj = _random_quadratic(n, m, seed=29)
        x0 = np.full(n, 0.1)
        y0 = np.full(m, 0.05)
        x_domain: list[ConstraintLayer] = [_box(x0, 1.2)]
        y_domain: list[ConstraintLayer] = [_box(y0, 0.9)]
        problem = StationarityProblem(
            objective=obj, x_domain=x_domain, y_domain=y_domain
        )
        t = 2.5

        system = problem.build_coupled_system(x0, y0, t)
        # No x_equality, so the plain (non-bordered) system: narrow for the type
        # checker before calling its two-argument ``solve`` below.
        assert isinstance(system, CoupledSaddleSystem)
        m_xx_dense = system.xx_system.multiply(np.eye(n))
        h_yy_dense = system.yy_system.multiply(np.eye(m))
        B = system.coupling
        jacobian = np.block([[-m_xx_dense, B], [B.T, h_yy_dense]])

        eps = 1e-6
        z0 = np.concatenate([x0, y0])
        numerical = np.zeros((n + m, n + m))
        for k in range(n + m):
            e = np.zeros(n + m)
            e[k] = eps
            zp = z0 + e
            zm = z0 - e
            r_plus = problem.residual(zp[:n], zp[n:], t)
            r_minus = problem.residual(zm[:n], zm[n:], t)
            numerical[:, k] = (r_plus - r_minus) / (2 * eps)

        np.testing.assert_allclose(jacobian, numerical, rtol=1e-5, atol=1e-5)

        rng = np.random.default_rng(1234)
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)
        delta_x, delta_y = system.solve(b_x, b_y)
        expected = np.linalg.solve(jacobian, np.concatenate([b_x, b_y]))
        np.testing.assert_allclose(
            np.concatenate([delta_x, delta_y]), expected, rtol=1e-8, atol=1e-9
        )

    def test_newton_step_solves_quadratic_saddle_in_one_step(self) -> None:
        r"""With empty domains, one Newton step lands on the exact saddle point.

        The pure quadratic saddle solves
           | -P   C | | x |   | -a |
           |  C^T R | | y | = | -b |,
        so its gradient residual is affine and exact Newton converges in one
        step: the post-step residual is (numerically) zero.
        """
        n, m = 9, 4
        obj = _random_quadratic(n, m, seed=88)
        problem = StationarityProblem(objective=obj, x_domain=[], y_domain=[])
        rng = np.random.default_rng(4321)
        x0 = rng.standard_normal(n)
        y0 = rng.standard_normal(m)
        t = 1.0

        delta_x, delta_y = problem.newton_step(x0, y0, t)
        x1 = x0 + delta_x
        y1 = y0 + delta_y

        residual = problem.residual(x1, y1, t)
        np.testing.assert_allclose(residual, np.zeros(n + m), atol=1e-8)

    def test_quadratic_saddle_objective_validation(self) -> None:
        r"""Bad shapes raise ``ValueError`` with a clear message."""
        n, m = 4, 2
        a = np.zeros(n)
        b = np.zeros(m)
        P = np.eye(n)
        C = np.zeros((n, m))
        R = np.eye(m)

        with self.assertRaisesRegex(ValueError, "P"):
            QuadraticSaddleObjective(a, b, np.eye(n + 1), C, R)
        with self.assertRaisesRegex(ValueError, "R"):
            QuadraticSaddleObjective(a, b, P, C, np.eye(m + 1))
        with self.assertRaisesRegex(ValueError, "C"):
            QuadraticSaddleObjective(a, b, P, np.zeros((n, m + 1)), R)
        with self.assertRaisesRegex(ValueError, "a"):
            QuadraticSaddleObjective(np.zeros((n, 1)), b, P, C, R)
        with self.assertRaisesRegex(ValueError, "b"):
            QuadraticSaddleObjective(a, np.zeros((m, 1)), P, C, R)

    def test_empty_and_nonempty_domains_shapes(self) -> None:
        r"""Dimension accessors and residual/system shapes are correct."""
        n, m = 10, 3
        obj = _random_quadratic(n, m, seed=99)
        self.assertEqual(obj.dimension_x, n)
        self.assertEqual(obj.dimension_y, m)

        x0 = np.full(n, 0.2)
        y0 = np.full(m, -0.1)
        empty = StationarityProblem(objective=obj)
        self.assertEqual(empty.dimension_x, n)
        self.assertEqual(empty.dimension_y, m)
        self.assertEqual(empty.residual(x0, y0, 1.0).shape, (n + m,))
        empty_system = empty.build_coupled_system(x0, y0, 1.0)
        self.assertEqual(empty_system.dimension_x, n)
        self.assertEqual(empty_system.dimension_y, m)

        nonempty = StationarityProblem(
            objective=obj,
            x_domain=[_box(x0, 1.0)],
            y_domain=[_box(y0, 1.0)],
        )
        self.assertEqual(nonempty.residual(x0, y0, 2.0).shape, (n + m,))
        system = nonempty.build_coupled_system(x0, y0, 2.0)
        self.assertEqual(system.dimension_x, n)
        self.assertEqual(system.dimension_y, m)
        delta_x, delta_y = nonempty.newton_step(x0, y0, 2.0)
        self.assertEqual(delta_x.shape, (n,))
        self.assertEqual(delta_y.shape, (m,))

    def test_non_positive_t_raises(self) -> None:
        r"""``residual``/``build_coupled_system`` reject ``t <= 0``."""
        n, m = 4, 2
        obj = _random_quadratic(n, m, seed=101)
        problem = StationarityProblem(objective=obj)
        x0 = np.zeros(n)
        y0 = np.zeros(m)
        for t in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                problem.residual(x0, y0, t)
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                problem.build_coupled_system(x0, y0, t)

    def test_newton_step_rejects_x_equality(self) -> None:
        r"""``newton_step`` refuses a problem with a stateful equality multiplier."""
        n, m, p = 4, 2, 1
        obj = _random_quadratic(n, m, seed=103)
        rng = np.random.default_rng(104)
        layer = EqualityConstraintLayer(
            rng.standard_normal((p, n)), rng.standard_normal(p)
        )
        problem = StationarityProblem(objective=obj, x_equality=layer)
        with self.assertRaisesRegex(ValueError, "does not support x_equality"):
            problem.newton_step(np.zeros(n), np.zeros(m), 1.0)

    def test_post_init_rejects_dimension_mismatches(self) -> None:
        r"""``__post_init__`` rejects a domain/equality layer of the wrong size."""
        n, m = 4, 2
        obj = _random_quadratic(n, m, seed=105)
        with self.assertRaisesRegex(ValueError, "x_domain"):
            StationarityProblem(objective=obj, x_domain=[_box(np.zeros(n + 1), 1.0)])
        with self.assertRaisesRegex(ValueError, "y_domain"):
            StationarityProblem(objective=obj, y_domain=[_box(np.zeros(m + 1), 1.0)])
        rng = np.random.default_rng(106)
        bad_equality = EqualityConstraintLayer(
            rng.standard_normal((1, n + 1)), rng.standard_normal(1)
        )
        with self.assertRaisesRegex(ValueError, "x_equality"):
            StationarityProblem(objective=obj, x_equality=bad_equality)


class StationarityProblemEqualityTest(unittest.TestCase):
    """Tests for the optional x-block equality channel of ``StationarityProblem``."""

    @staticmethod
    def _equality(n: int, p: int, seed: int) -> EqualityConstraintLayer:
        """A random full-row-rank ``EqualityConstraintLayer`` on an ``n``-vector."""
        rng = np.random.default_rng(seed)
        return EqualityConstraintLayer(
            rng.standard_normal((p, n)), rng.standard_normal(p)
        )

    def test_num_equalities(self) -> None:
        r"""``num_equalities`` is zero without an equality and ``p`` with one."""
        obj = _random_quadratic(6, 3, seed=1)
        self.assertEqual(StationarityProblem(obj).num_equalities, 0)
        problem = StationarityProblem(obj, x_equality=self._equality(6, 2, seed=2))
        self.assertEqual(problem.num_equalities, 2)

    def test_residual_appends_equality_row_and_multiplier(self) -> None:
        r"""With an equality, ``residual`` is ``[grad_x + A^T lam; grad_y; A x - b]``.

        No domains, so the barrier term vanishes and the residual is exactly the
        objective gradients plus the multiplier term and the primal-feasibility row.
        """
        n, m, p = 7, 3, 2
        obj = _random_quadratic(n, m, seed=10)
        layer = self._equality(n, p, seed=11)
        problem = StationarityProblem(obj, x_equality=layer)
        rng = np.random.default_rng(12)
        x = rng.standard_normal(n)
        y = rng.standard_normal(m)
        lam = rng.standard_normal(p)

        r = problem.residual(x, y, t=3.0, lam=lam)
        self.assertEqual(r.shape, (n + m + p,))
        np.testing.assert_allclose(
            r[:n], obj.gradient_x(x, y) + layer.A.T @ lam, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(r[n : n + m], obj.gradient_y(x, y), rtol=1e-12)
        np.testing.assert_allclose(r[n + m :], layer.A @ x - layer.b, rtol=1e-12)

    def test_residual_requires_lam_when_equality_present(self) -> None:
        problem = StationarityProblem(
            _random_quadratic(5, 2, seed=20), x_equality=self._equality(5, 1, seed=21)
        )
        with self.assertRaisesRegex(ValueError, "lam is required"):
            problem.residual(np.zeros(5), np.zeros(2), t=1.0)

    def test_residual_rejects_lam_without_equality(self) -> None:
        problem = StationarityProblem(_random_quadratic(5, 2, seed=22))
        with self.assertRaisesRegex(ValueError, "no x_equality"):
            problem.residual(np.zeros(5), np.zeros(2), t=1.0, lam=np.zeros(1))

    def test_residual_rejects_wrong_lam_shape(self) -> None:
        problem = StationarityProblem(
            _random_quadratic(5, 2, seed=23), x_equality=self._equality(5, 2, seed=24)
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            problem.residual(np.zeros(5), np.zeros(2), t=1.0, lam=np.zeros(3))

    def test_build_coupled_system_is_bordered_with_equality(self) -> None:
        r"""``build_coupled_system`` returns a bordered system carrying ``A_eq``."""
        n, m, p = 8, 3, 2
        layer = self._equality(n, p, seed=30)
        problem = StationarityProblem(
            _random_quadratic(n, m, seed=31), x_equality=layer
        )
        system = problem.build_coupled_system(np.zeros(n), np.zeros(m), t=2.0)
        self.assertIsInstance(system, EqualityCoupledSaddleSystem)
        assert isinstance(system, EqualityCoupledSaddleSystem)
        self.assertEqual(system.num_equalities, p)
        np.testing.assert_array_equal(system.equality, layer.A)

    def test_build_coupled_system_is_plain_without_equality(self) -> None:
        problem = StationarityProblem(_random_quadratic(6, 2, seed=32))
        system = problem.build_coupled_system(np.zeros(6), np.zeros(2), t=2.0)
        self.assertIsInstance(system, CoupledSaddleSystem)
        self.assertNotIsInstance(system, EqualityCoupledSaddleSystem)

    # `newton_step` rejecting an equality problem and `__post_init__` rejecting an
    # x_equality dimension mismatch are already covered by
    # `test_newton_step_rejects_x_equality` and
    # `test_post_init_rejects_dimension_mismatches` above.
