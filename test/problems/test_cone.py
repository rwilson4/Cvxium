"""Tests for the Cone API skeleton (CVX-HSD-2)."""

import unittest

import numpy as np
from parameterized import parameterized

from cvxium.problems.cone import (
    Cone,
    DenseNtScaling,
    Nonneg,
    NtScaling,
    SecondOrderCone,
)


def _random_soc_interior_point(
    cone: SecondOrderCone, rng: np.random.Generator, slack: float = 1.0
) -> np.ndarray:
    """Random strictly-interior SOC point with ``x_0 - ||x_{1:}|| >= slack``."""
    n = cone.dimension()
    rest = rng.standard_normal(n - 1)
    x_0 = float(np.linalg.norm(rest)) + slack
    x = np.empty(n)
    x[0] = x_0
    x[1:] = rest
    return x


def _dense_soc_jordan_solve(v: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Dense ``L(v) z = rhs`` solve for the SOC arrow matrix."""
    n = v.shape[0]
    L = np.empty((n, n))
    L[0, 0] = v[0]
    L[0, 1:] = v[1:]
    L[1:, 0] = v[1:]
    L[1:, 1:] = v[0] * np.eye(n - 1)
    return np.linalg.solve(L, rhs)


class ConeTest(unittest.TestCase):
    """Tests for the Cone API skeleton (SecondOrderCone, Nonneg, NtScaling)."""

    def setUp(self) -> None:
        self.soc3 = SecondOrderCone(dim=3)

    # --- SecondOrderCone: construction ---

    def test_soc_dimension_stored(self) -> None:
        cone = SecondOrderCone(dim=5)
        assert cone.dimension() == 5

    def test_soc_minimum_dimension_two(self) -> None:
        """SOC requires dim >= 2; the trivial 1-D case is the non-negative orthant."""
        with self.assertRaisesRegex(ValueError, "dim >= 2"):
            SecondOrderCone(dim=1)

    def test_soc_zero_dim_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dim >= 2"):
            SecondOrderCone(dim=0)

    # --- SecondOrderCone: interior_point ---

    @parameterized.expand([(2,), (3,), (7,), (100,)])
    def test_soc_interior_point_shape_and_values(self, dim: int) -> None:
        e = SecondOrderCone(dim=dim).interior_point()
        expected = np.zeros(dim)
        expected[0] = 1.0
        np.testing.assert_array_equal(e, expected)

    def test_soc_interior_point_is_strictly_in_cone(self) -> None:
        """``e_K`` is by definition strictly interior to the SOC."""
        cone = SecondOrderCone(dim=5)
        assert cone.is_in_cone(cone.interior_point(), strict=True)

    # --- SecondOrderCone: is_in_cone (membership) ---

    def test_soc_strict_interior(self) -> None:
        assert self.soc3.is_in_cone(np.array([2.0, 1.0, 0.0]))
        assert self.soc3.is_in_cone(np.array([2.0, 1.0, 0.0]), strict=True)

    def test_soc_boundary_non_strict_passes_strict_fails(self) -> None:
        # ||(1, 0)|| = 1 = x_0 -> on the cone boundary.
        x = np.array([1.0, 1.0, 0.0])
        assert self.soc3.is_in_cone(x)
        assert not self.soc3.is_in_cone(x, strict=True)

    def test_soc_apex_passes_non_strict_fails_strict(self) -> None:
        # The cone apex (origin) is in the closed cone but not the interior.
        zero = np.zeros(3)
        assert self.soc3.is_in_cone(zero)
        assert not self.soc3.is_in_cone(zero, strict=True)

    def test_soc_membership_outside(self) -> None:
        # x_0 < ||x_{1:}||: outside the cone.
        assert not self.soc3.is_in_cone(np.array([0.5, 1.0, 0.0]))
        # Negative apex: outside even when ||x_{1:}|| = 0.
        assert not self.soc3.is_in_cone(np.array([-1.0, 0.0, 0.0]))

    def test_soc_membership_wrong_shape_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected"):
            self.soc3.is_in_cone(np.array([1.0, 0.0]))  # dim 2, not 3

    # --- SecondOrderCone: step_to_boundary ---

    def test_soc_step_zero_direction_returns_infinity(self) -> None:
        x = np.array([2.0, 0.0, 0.0])
        dx = np.zeros(3)
        assert self.soc3.step_to_boundary(x, dx) == np.inf

    def test_soc_step_strictly_interior_direction_returns_infinity(
        self,
    ) -> None:
        # x interior (apex axis), dx along same direction -> stays interior.
        x = np.array([1.0, 0.0, 0.0])
        dx = np.array([1.0, 0.0, 0.0])
        assert self.soc3.step_to_boundary(x, dx) == np.inf

    def test_soc_step_apex_blocking_direction(self) -> None:
        # x = (2, 0, 0), dx = (-1, 0, 0): x_0 hits 0 at alpha = 2.
        x = np.array([2.0, 0.0, 0.0])
        dx = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.soc3.step_to_boundary(x, dx), 2.0, delta=1e-10)

    def test_soc_step_pure_radial_widening(self) -> None:
        # x = (2, 0, 0), dx = (0, 1, 0): boundary when ||(alpha, 0)|| = 2 -> alpha = 2.
        x = np.array([2.0, 0.0, 0.0])
        dx = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(self.soc3.step_to_boundary(x, dx), 2.0, delta=1e-10)

    def test_soc_step_offset_radial_widening(self) -> None:
        # x = (2, 1, 0), dx = (0, 1, 0): boundary when ||(1+alpha, 0)|| = 2 -> alpha = 1.
        x = np.array([2.0, 1.0, 0.0])
        dx = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(self.soc3.step_to_boundary(x, dx), 1.0, delta=1e-10)

    def test_soc_step_combined_apex_and_radial(self) -> None:
        # x = (1, 0, 0), dx = (-1, -1, 0): need (1-alpha)^2 >= alpha^2
        # -> 1 - 2 alpha >= 0 -> alpha <= 0.5.
        x = np.array([1.0, 0.0, 0.0])
        dx = np.array([-1.0, -1.0, 0.0])
        self.assertAlmostEqual(self.soc3.step_to_boundary(x, dx), 0.5, delta=1e-10)

    def test_soc_step_iterate_lands_strictly_inside_just_before_boundary(
        self,
    ) -> None:
        """Empirical check: ``x + (1 - eps) alpha dx`` is strictly inside; ``+eps`` is outside."""
        x = np.array([2.0, 1.0, 0.0])
        dx = np.array([-0.5, 0.7, 0.3])
        alpha = self.soc3.step_to_boundary(x, dx)
        # Strictly inside just before the step.
        assert self.soc3.is_in_cone(x + (1.0 - 1e-6) * alpha * dx, strict=True)
        # Strictly outside just past the step.
        assert not self.soc3.is_in_cone(x + (1.0 + 1e-6) * alpha * dx)

    def test_soc_step_apex_condition_dominates_over_squared_form(
        self,
    ) -> None:
        """The apex constraint ``x_0 + alpha dx_0 >= 0`` bites independently.

        Without it, ``(-2, 1, 0)`` would satisfy ``(-2)^2 >= 1`` and look "in
        the cone" while actually sitting in the dual reflection. The
        guard returns the apex-limited step instead.
        """
        x = np.array([1.0, 0.0, 0.0])
        dx = np.array([-1.0, 0.0, 0.0])
        # Apex hits 0 at alpha=1; squared form alone would allow alpha=2
        # (with x = -1, then squared check (-1)^2 >= 0 = ||x_{1:}||^2 holds).
        # Correct answer is 1.
        self.assertAlmostEqual(self.soc3.step_to_boundary(x, dx), 1.0, delta=1e-10)

    def test_soc_step_wrong_shape_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have shape"):
            self.soc3.step_to_boundary(np.zeros(3), np.zeros(2))

    # --- SecondOrderCone: jordan_product ---

    def test_soc_jordan_product_explicit_formula(self) -> None:
        cone = SecondOrderCone(dim=4)
        a = np.array([2.0, 1.0, 1.0, 1.0])
        b = np.array([1.0, 0.5, -0.5, 0.0])
        # a^T b = 2 + 0.5 - 0.5 + 0 = 2.0; a_0 b_{1:} = 2*[0.5,-0.5,0] = [1,-1,0];
        # b_0 a_{1:} = 1*[1,1,1] = [1,1,1]; sum = [2,0,1].
        expected = np.array([2.0, 2.0, 0.0, 1.0])
        np.testing.assert_array_equal(cone.jordan_product(a, b), expected)

    def test_soc_jordan_identity(self) -> None:
        """``e_K`` is the Jordan-algebra identity: ``e_K . x = x``."""
        cone = SecondOrderCone(dim=5)
        e = cone.interior_point()
        rng = np.random.default_rng(9001)
        for _trial in range(5):
            x = rng.standard_normal(5)
            np.testing.assert_allclose(cone.jordan_product(e, x), x, atol=1e-12)
            np.testing.assert_allclose(cone.jordan_product(x, e), x, atol=1e-12)

    def test_soc_jordan_product_is_commutative(self) -> None:
        cone = SecondOrderCone(dim=4)
        rng = np.random.default_rng(9002)
        for _trial in range(5):
            a = rng.standard_normal(4)
            b = rng.standard_normal(4)
            np.testing.assert_allclose(
                cone.jordan_product(a, b), cone.jordan_product(b, a), atol=1e-12
            )

    def test_soc_jordan_product_is_bilinear(self) -> None:
        cone = SecondOrderCone(dim=3)
        rng = np.random.default_rng(9003)
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        c = rng.standard_normal(3)
        alpha, beta = 1.7, -0.4
        np.testing.assert_allclose(
            cone.jordan_product(alpha * a + beta * b, c),
            alpha * cone.jordan_product(a, c) + beta * cone.jordan_product(b, c),
            atol=1e-12,
        )

    def test_soc_jordan_product_wrong_shape_raises(self) -> None:
        cone = SecondOrderCone(dim=3)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            cone.jordan_product(np.zeros(3), np.zeros(2))

    # --- SecondOrderCone: nt_scaling ---

    def test_soc_W_squared_s_equals_x_random_pairs(self) -> None:
        """``W^2 s = x`` to machine precision across random interior pairs."""
        rng = np.random.default_rng(8801)
        for dim in (2, 3, 7):
            for _trial in range(5):
                cone = SecondOrderCone(dim=dim)
                x = _random_soc_interior_point(cone, rng)
                s = _random_soc_interior_point(cone, rng)
                scaling = cone.nt_scaling(x, s)
                np.testing.assert_allclose(
                    scaling.apply_W_squared(s), x, rtol=1e-10, atol=1e-12
                )

    def test_soc_W_is_symmetric(self) -> None:
        """``W`` is symmetric in the Euclidean sense (``Q(w_bar)`` is symmetric)."""
        rng = np.random.default_rng(8802)
        cone = SecondOrderCone(dim=4)
        x = _random_soc_interior_point(cone, rng)
        s = _random_soc_interior_point(cone, rng)
        scaling = cone.nt_scaling(x, s)
        # Probe via apply_W on basis vectors.
        n = cone.dimension()
        W = np.column_stack([scaling.apply_W(np.eye(n)[i]) for i in range(n)])
        np.testing.assert_allclose(W, W.T, rtol=1e-12, atol=1e-12)

    def test_soc_W_inv_is_inverse(self) -> None:
        """``apply_W_inv(apply_W(v)) == v`` round-trips."""
        rng = np.random.default_rng(8803)
        cone = SecondOrderCone(dim=5)
        x = _random_soc_interior_point(cone, rng)
        s = _random_soc_interior_point(cone, rng)
        scaling = cone.nt_scaling(x, s)
        v = rng.standard_normal(5)
        np.testing.assert_allclose(
            scaling.apply_W_inv(scaling.apply_W(v)), v, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            scaling.apply_W(scaling.apply_W_inv(v)), v, rtol=1e-10, atol=1e-12
        )

    def test_soc_W_squared_consistent_with_apply_W_twice(self) -> None:
        """``apply_W_squared(v) == apply_W(apply_W(v))`` (consistency of cached W^2)."""
        rng = np.random.default_rng(8804)
        cone = SecondOrderCone(dim=4)
        x = _random_soc_interior_point(cone, rng)
        s = _random_soc_interior_point(cone, rng)
        scaling = cone.nt_scaling(x, s)
        v = rng.standard_normal(4)
        np.testing.assert_allclose(
            scaling.apply_W_squared(v),
            scaling.apply_W(scaling.apply_W(v)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_soc_self_scaling_is_identity_up_to_scalar(self) -> None:
        """When ``s = x``, ``W^2 = I`` (the central path's symmetric point)."""
        cone = SecondOrderCone(dim=3)
        x = np.array([2.0, 0.5, -0.5])
        scaling = cone.nt_scaling(x, x)
        # W^2 s = x with s = x gives W^2 x = x; for general v, W^2 v should
        # be v (since x is interior, NT scaling collapses to identity).
        n = cone.dimension()
        for i in range(n):
            e_i = np.eye(n)[i]
            np.testing.assert_allclose(
                scaling.apply_W_squared(e_i), e_i, rtol=1e-10, atol=1e-12
            )

    def test_soc_nt_scaling_outside_cone_raises(self) -> None:
        cone = SecondOrderCone(dim=3)
        x_inside = np.array([2.0, 0.5, 0.5])
        s_outside = np.array([0.1, 1.0, 0.0])  # ||(1,0)|| = 1 > 0.1
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            cone.nt_scaling(x_inside, s_outside)

    def test_soc_nt_scaling_wrong_shape_raises(self) -> None:
        cone = SecondOrderCone(dim=3)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            cone.nt_scaling(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    # --- SecondOrderCone: factored_w_squared ---

    def test_soc_factored_form_matches_dense_w_squared(self) -> None:
        """``scale * (2 y y^T - J)`` agrees with ``apply_W_squared`` per block."""
        rng = np.random.default_rng(9001)
        for dim in (2, 3, 7, 12):
            for _trial in range(5):
                cone = SecondOrderCone(dim=dim)
                x = _random_soc_interior_point(cone, rng)
                s = _random_soc_interior_point(cone, rng)
                scale, y = cone.factored_w_squared(x, s)
                J = np.eye(dim)
                J[1:, 1:] *= -1.0
                W_sq_factored = scale * (2.0 * np.outer(y, y) - J)
                scaling = cone.nt_scaling(x, s)
                W_sq_dense = np.column_stack(
                    [scaling.apply_W_squared(np.eye(dim)[:, i]) for i in range(dim)]
                )
                np.testing.assert_allclose(
                    W_sq_factored, W_sq_dense, rtol=1e-10, atol=1e-12
                )

    def test_soc_factored_form_satisfies_w_squared_s_equals_x(self) -> None:
        """``scale * (2 (y^T s) y - J s) = x`` (the NT defining identity)."""
        rng = np.random.default_rng(9002)
        for dim in (2, 4, 8):
            cone = SecondOrderCone(dim=dim)
            x = _random_soc_interior_point(cone, rng)
            s = _random_soc_interior_point(cone, rng)
            scale, y = cone.factored_w_squared(x, s)
            Js = np.empty(dim)
            Js[0] = s[0]
            Js[1:] = -s[1:]
            W_sq_s = scale * (2.0 * float(y @ s) * y - Js)
            np.testing.assert_allclose(W_sq_s, x, rtol=1e-10, atol=1e-12)

    def test_soc_factored_y_has_unit_lorentz_norm(self) -> None:
        """``det(y) = y_0^2 - ||y_{1:}||^2 = 1`` by NT normalization."""
        rng = np.random.default_rng(9003)
        cone = SecondOrderCone(dim=5)
        for _ in range(5):
            x = _random_soc_interior_point(cone, rng)
            s = _random_soc_interior_point(cone, rng)
            _, y = cone.factored_w_squared(x, s)
            det_y = float(y[0] * y[0] - np.dot(y[1:], y[1:]))
            self.assertAlmostEqual(det_y, 1.0, delta=1e-10)

    def test_soc_factored_outside_cone_raises(self) -> None:
        cone = SecondOrderCone(dim=3)
        x_inside = np.array([2.0, 0.5, 0.5])
        s_outside = np.array([0.1, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            cone.factored_w_squared(x_inside, s_outside)

    # --- SecondOrderCone: solve_jordan_arrow ---

    def test_soc_solve_jordan_arrow_matches_dense_solve(self) -> None:
        rng = np.random.default_rng(9201)
        for dim in (2, 3, 5, 12, 50):
            cone = SecondOrderCone(dim=dim)
            for _trial in range(5):
                v = _random_soc_interior_point(cone, rng)
                rhs = rng.standard_normal(dim)
                np.testing.assert_allclose(
                    cone.solve_jordan_arrow(v, rhs),
                    _dense_soc_jordan_solve(v, rhs),
                    rtol=1e-10,
                    atol=1e-12,
                )

    def test_soc_solve_jordan_arrow_solves_l_v_z_equals_rhs(self) -> None:
        """Defining identity: ``L(v) @ solve(v, rhs) == rhs``."""
        rng = np.random.default_rng(9202)
        cone = SecondOrderCone(dim=7)
        for _ in range(5):
            v = _random_soc_interior_point(cone, rng)
            rhs = rng.standard_normal(7)
            z = cone.solve_jordan_arrow(v, rhs)
            # L(v) @ z = (v_0 z_0 + v_{1:}^T z_{1:},  v_{1:} z_0 + v_0 z_{1:})
            recovered = np.empty(7)
            recovered[0] = v[0] * z[0] + float(v[1:] @ z[1:])
            recovered[1:] = v[1:] * z[0] + v[0] * z[1:]
            np.testing.assert_allclose(recovered, rhs, rtol=1e-10, atol=1e-12)

    # --- Nonneg cone ---

    def test_nonneg_dimension_minimum(self) -> None:
        """Nonneg accepts dim >= 1 (unlike SOC which requires dim >= 2)."""
        assert Nonneg(dim=1).dimension() == 1
        assert Nonneg(dim=5).dimension() == 5

    def test_nonneg_zero_dim_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dim >= 1"):
            Nonneg(dim=0)

    def test_nonneg_interior_point_is_ones(self) -> None:
        cone = Nonneg(dim=4)
        np.testing.assert_array_equal(cone.interior_point(), np.ones(4))
        assert cone.is_in_cone(cone.interior_point(), strict=True)

    def test_nonneg_membership_strict_and_boundary(self) -> None:
        cone = Nonneg(dim=3)
        assert cone.is_in_cone(np.array([1.0, 2.0, 3.0]))
        assert cone.is_in_cone(np.array([1.0, 2.0, 3.0]), strict=True)
        # Boundary: zero entry passes non-strict, fails strict.
        boundary = np.array([1.0, 0.0, 3.0])
        assert cone.is_in_cone(boundary)
        assert not cone.is_in_cone(boundary, strict=True)
        # Negative entry fails both.
        assert not cone.is_in_cone(np.array([1.0, -0.1, 3.0]))

    def test_nonneg_membership_wrong_shape_raises(self) -> None:
        cone = Nonneg(dim=3)
        with self.assertRaisesRegex(ValueError, "expected"):
            cone.is_in_cone(np.array([1.0, 2.0]))

    def test_nonneg_step_to_boundary_no_descent_returns_inf(self) -> None:
        cone = Nonneg(dim=3)
        assert cone.step_to_boundary(np.ones(3), np.ones(3)) == np.inf
        assert cone.step_to_boundary(np.ones(3), np.zeros(3)) == np.inf

    def test_nonneg_step_to_boundary_picks_smallest_component(self) -> None:
        cone = Nonneg(dim=3)
        # x = (3, 1, 5), dx = (-1, -2, 1) -> components 0,1 drive: 3, 0.5 -> min 0.5.
        x = np.array([3.0, 1.0, 5.0])
        dx = np.array([-1.0, -2.0, 1.0])
        self.assertAlmostEqual(cone.step_to_boundary(x, dx), 0.5, delta=1e-10)

    def test_nonneg_step_to_boundary_lands_at_boundary(self) -> None:
        cone = Nonneg(dim=4)
        x = np.array([2.0, 1.0, 3.0, 0.5])
        dx = np.array([-1.0, -0.5, 0.1, -2.0])
        alpha = cone.step_to_boundary(x, dx)
        assert cone.is_in_cone(x + (1.0 - 1e-9) * alpha * dx, strict=True)
        assert not cone.is_in_cone(x + (1.0 + 1e-9) * alpha * dx)

    def test_nonneg_jordan_product_is_componentwise(self) -> None:
        cone = Nonneg(dim=3)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(
            cone.jordan_product(a, b), np.array([4.0, 10.0, 18.0])
        )
        # Identity property: e . x = x.
        e = cone.interior_point()
        x = np.array([7.0, -1.0, 0.5])
        np.testing.assert_array_equal(cone.jordan_product(e, x), x)

    def test_nonneg_nt_scaling_satisfies_w_squared_s_equals_x(self) -> None:
        """``W^2 s = x`` componentwise; the defining NT identity."""
        rng = np.random.default_rng(9101)
        for dim in (1, 3, 7, 12):
            cone = Nonneg(dim=dim)
            for _trial in range(5):
                x = rng.uniform(0.1, 5.0, size=dim)
                s = rng.uniform(0.1, 5.0, size=dim)
                scaling = cone.nt_scaling(x, s)
                np.testing.assert_allclose(
                    scaling.apply_W_squared(s), x, rtol=1e-12, atol=1e-12
                )

    def test_nonneg_nt_scaling_outside_cone_raises(self) -> None:
        cone = Nonneg(dim=3)
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            cone.nt_scaling(np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            cone.nt_scaling(np.array([1.0, 1.0, 1.0]), np.array([1.0, -0.1, 1.0]))

    def test_nonneg_apply_w_squared_matches_dense(self) -> None:
        """``apply_w_squared(precomputed, v) = (x/s) * v`` matches dense ``W^2 v``."""
        rng = np.random.default_rng(9102)
        for dim in (1, 4, 10):
            cone = Nonneg(dim=dim)
            x = rng.uniform(0.1, 5.0, size=dim)
            s = rng.uniform(0.1, 5.0, size=dim)
            v = rng.standard_normal(dim)
            dense = cone.nt_scaling(x, s).apply_W_squared(v)
            np.testing.assert_allclose(
                cone.apply_w_squared(cone.prepare_w_squared(x, s), v),
                dense,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_nonneg_accumulate_normal_equations_matches_dense(self) -> None:
        """Per-block ``(A W^2 A^T, A W^2 c, c^T W^2 c)`` matches dense oracle."""
        rng = np.random.default_rng(9103)
        for dim, p in [(3, 2), (8, 5)]:
            cone = Nonneg(dim=dim)
            x = rng.uniform(0.1, 5.0, size=dim)
            s = rng.uniform(0.1, 5.0, size=dim)
            A = rng.standard_normal((p, dim))
            c = rng.standard_normal(dim)
            W_sq = np.diag(x / s)
            M_oracle = A @ W_sq @ A.T
            AW2c_oracle = A @ W_sq @ c
            cW2c_oracle = float(c @ W_sq @ c)
            M, AW2c, cW2c = cone.accumulate_normal_equations(
                cone.prepare_w_squared(x, s), A, c
            )
            np.testing.assert_allclose(M, M_oracle, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(AW2c, AW2c_oracle, rtol=1e-12, atol=1e-12)
            self.assertAlmostEqual(cW2c, cW2c_oracle, delta=1e-10)

    def test_nonneg_solve_jordan_arrow_is_componentwise_division(self) -> None:
        """``L(v) = diag(v)``; solve is ``rhs / v`` elementwise."""
        cone = Nonneg(dim=4)
        v = np.array([2.0, 0.5, 1.0, 4.0])
        rhs = np.array([8.0, 1.0, -3.0, 12.0])
        np.testing.assert_array_equal(
            cone.solve_jordan_arrow(v, rhs), np.array([4.0, 2.0, -3.0, 3.0])
        )

    # --- NtScaling carrier ---

    def test_nt_scaling_apply_W_matches_matmul(self) -> None:
        rng = np.random.default_rng(8901)
        n = 4
        # Build a random invertible W (no cone semantics required for this test).
        W = rng.standard_normal((n, n))
        W = W @ W.T + n * np.eye(n)  # symmetric PD for cleanliness
        scaling = DenseNtScaling(W=W)
        v = rng.standard_normal(n)
        np.testing.assert_allclose(scaling.apply_W(v), W @ v)
        np.testing.assert_allclose(scaling.apply_W_squared(v), W @ (W @ v))
        np.testing.assert_allclose(
            scaling.apply_W_inv(W @ v), v, rtol=1e-10, atol=1e-12
        )

    def test_nt_scaling_non_square_W_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            DenseNtScaling(W=np.zeros((2, 3)))

    def test_nt_scaling_is_abstract(self) -> None:
        """``NtScaling`` cannot be instantiated directly; it's an ABC."""
        abstract_cls: type[NtScaling] = NtScaling  # type: ignore[type-abstract]
        with self.assertRaises(TypeError):
            abstract_cls()

    # --- Cone ABC ---

    def test_cone_is_abstract(self) -> None:
        """``Cone`` cannot be instantiated directly; it's an ABC."""
        abstract_cls: type[Cone] = Cone  # type: ignore[type-abstract]
        with self.assertRaises(TypeError):
            abstract_cls()
