"""Smoke tests for the ConstraintLayer ABC."""

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    MatrixSystem,
    ZeroSystem,
)
from cvxium.problems.constraint_layers import (
    ConstraintLayer,
    DirectConstraintLayer,
    SlackConstraintLayer,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    ImbalanceConstraintLayer,
)
from cvxium.problems.objective import (
    QuadraticObjective,
)
from cvxium.problems.quadratic_constraints import (
    NormConstraintLayer,
    ObjectiveConstraintLayer,
    SquaredNormConstraintLayer,
)
from cvxium.solvers.optimization import (
    TrivialFeasibilitySolver,
)
from cvxium.solvers.settings import (
    LineSearchSettings,
)


class _CompleteLayer(SlackConstraintLayer):
    """Minimal complete subclass: a single bound `w[0] <= 0`."""

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def num_constraints(self, M: int) -> int:
        return 1

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        return np.array([w[0] - shift])

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        g = np.zeros((1, w.shape[0]))
        g[0, 0] = 1.0
        return g

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        # A single linear bound: genuinely rank-one, so the low-rank helper.
        return self._low_rank_barrier_contribution(w, lmbda, shift)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        # The bound `w[0] <= 0` is affine, so zero curvature.
        return ZeroSystem(len(w))


class _IncompleteLayer(ConstraintLayer):
    """Subclass missing required overrides; must remain abstract."""

    def num_constraints(self, M: int) -> int:
        return 1


class ConstraintLayerABCTest(unittest.TestCase):
    """ABC-instantiation smoke tests for the ConstraintLayer hierarchy."""

    def test_abc_cannot_be_instantiated(self) -> None:
        abstract_cls: type[ConstraintLayer] = ConstraintLayer  # type: ignore[type-abstract]
        with self.assertRaises(TypeError):
            abstract_cls()

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        incomplete_cls: type[ConstraintLayer] = (
            _IncompleteLayer  # type: ignore[type-abstract]
        )
        with self.assertRaises(TypeError):
            incomplete_cls()

    def test_complete_subclass_instantiates(self) -> None:
        layer = _CompleteLayer()
        assert layer.num_constraints(3) == 1
        assert layer.is_satisfied(np.array([-1.0, 0.0, 0.0]))

    def test_direct_layer_interior_point_default_is_none(self) -> None:
        """A `DirectConstraintLayer` with no override returns ``None`` for its interior.

        ``SquaredNormConstraintLayer`` inherits the base ABC default; only
        ``NormConstraintLayer`` overrides it with a closed-form ellipsoid center.
        """
        layer = SquaredNormConstraintLayer(phi=1.0, dimension=3)
        assert layer.interior_point() is None


class AffineCurvatureDeclarationTest(unittest.TestCase):
    """Affine layers declare a ``ZeroSystem`` ``hessian_curvature`` explicitly.

    ``hessian_curvature`` is abstract -- there is no default -- so affine layers
    (`BoundsConstraintLayer`, `ImbalanceConstraintLayer`,
    `LinearInequalityConstraintLayer`) declare a ``ZeroSystem`` and curved layers
    (`NormConstraintLayer`, `SquaredNormConstraintLayer`) declare real curvature.

    """

    def test_affine_layer_curvature_is_zero(self) -> None:
        """An affine layer declares ``ZeroSystem`` curvature (`Q_i = 0`)."""
        layer = _CompleteLayer()
        system = layer.hessian_curvature(np.zeros(3), np.array([0.5]))
        np.testing.assert_array_equal(system.multiply(np.ones(3)), np.zeros(3))


class BaseMaxStepReachesTheUnitStepTest(unittest.TestCase):
    """The default `btls_max_step` has to allow a full Newton step.

    Callers use it as ``s = min(1, beta * btls_max_step(...))``. An iterative
    search that starts at 1 therefore tops out at ``beta`` -- roughly a half
    step, forever, including in the quadratic phase where Boyd and
    Vandenberghe 9.5.3 guarantees the unit step is accepted. Starting the
    search at ``1 / beta`` is what lets that expression reach 1.

    This is not hypothetical: the slack Phase-I layers in `phase1.py` inherit
    this default, so every Phase I Newton step was halved.

    """

    def test_a_far_boundary_permits_the_full_unit_step(self) -> None:
        """beta * alpha must be able to reach 1, so alpha must exceed 1."""
        layer = _CompleteLayer()
        settings = LineSearchSettings()
        # w[0] = -10 against dw[0] = +1: the constraint w[0] <= 0 does not
        # bind until step 10, far past anything the line search would want.
        alpha = layer.btls_max_step(
            np.array([-10.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            line_search=settings,
        )

        self.assertGreater(alpha, 1.0)
        self.assertAlmostEqual(settings.beta * alpha, 1.0)

    def test_a_near_boundary_still_shrinks(self) -> None:
        """Raising the starting point must not stop it backing off."""
        layer = _CompleteLayer()
        settings = LineSearchSettings()
        # Binds at step 0.3, so the search has to come down from 1 / beta.
        alpha = layer.btls_max_step(
            np.array([-0.3, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            line_search=settings,
        )

        self.assertLess(alpha, 0.3)
        self.assertGreater(alpha, 0.0)


class BoundsConstraintLayerMathTest(unittest.TestCase):
    """Math-only tests for BoundsConstraintLayer.

    End-to-end Phase I behaviour is exercised in `test_linear_programs.py` and
    `test_quadratic_programs.py`; these tests pin down the per-method math.

    """

    def test_num_constraints_returns_M(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=5)
        assert layer.num_constraints(5) == 5
        assert layer.num_constraints(1) == 1
        assert layer.num_constraints(100) == 100

    def test_constraints_scalar_lb_no_shift(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        c = layer.constraints(w)
        assert c.shape == (3,)
        np.testing.assert_allclose(c, np.array([-1.0, -2.0, -3.0]))

    def test_constraints_scalar_lb_with_shift(self) -> None:
        layer = BoundsConstraintLayer(lb=0.5, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        shift = 0.25
        c = layer.constraints(w, shift=shift)
        # f_i = -w_i + lb - shift
        np.testing.assert_allclose(c, np.array([-0.75, -1.75, -2.75]))

    def test_constraints_array_lb_no_shift(self) -> None:
        lb = np.array([0.1, 0.2, 0.3])
        layer = BoundsConstraintLayer(lb=lb)
        w = np.array([1.0, 2.0, 3.0])
        c = layer.constraints(w)
        assert c.shape == (3,)
        np.testing.assert_allclose(c, np.array([-0.9, -1.8, -2.7]))

    def test_constraints_array_lb_with_shift(self) -> None:
        lb = np.array([0.1, 0.2, 0.3])
        layer = BoundsConstraintLayer(lb=lb)
        w = np.array([1.0, 2.0, 3.0])
        shift = 0.05
        c = layer.constraints(w, shift=shift)
        np.testing.assert_allclose(c, np.array([-0.95, -1.85, -2.75]))

    def test_grad_constraints_returns_negative_identity(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=4)
        w = np.array([1.0, 2.0, 3.0, 4.0])
        G = layer.grad_constraints(w)
        assert G.shape == (4, 4)
        np.testing.assert_allclose(G, -np.eye(4))

    def test_grad_constraints_multiply_matches_materialized(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -1.5, 2.0])
        G = layer.grad_constraints(w)
        expected = G @ v
        result = layer.grad_constraints_multiply(w, v)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, -v)

    def test_grad_constraints_transpose_multiply_matches_materialized(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        u = np.array([0.5, -1.5, 2.0])
        G = layer.grad_constraints(w)
        expected = G.T @ u
        result = layer.grad_constraints_transpose_multiply(w, u)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, -u)

    def test_btls_max_step_all_nonneg_dw_returns_inf(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([0.1, 0.0, 0.5])
        assert layer.btls_max_step(w, dw) == np.inf

    def test_btls_max_step_some_negative_dw_scalar_lb(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([-0.5, 1.0, -1.5])
        # Closed form: min_{i: dw_i<0} (w_i - lb) / -dw_i
        # i=0: 1.0 / 0.5 = 2.0; i=2: 3.0 / 1.5 = 2.0
        self.assertAlmostEqual(layer.btls_max_step(w, dw), 2.0)

    def test_btls_max_step_some_negative_dw_array_lb(self) -> None:
        lb = np.array([0.1, 0.2, 0.3])
        layer = BoundsConstraintLayer(lb=lb)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([-0.5, 1.0, -1.5])
        # i=0: (1.0 - 0.1) / 0.5 = 1.8; i=2: (3.0 - 0.3) / 1.5 = 1.8
        self.assertAlmostEqual(layer.btls_max_step(w, dw), 1.8)

    def test_btls_max_step_with_shift(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([-0.5, 1.0, -1.5])
        shift = 0.5
        # (w - lb + shift) / -dw -> i=0: 1.5/0.5=3.0; i=2: 3.5/1.5≈2.333
        self.assertAlmostEqual(layer.btls_max_step(w, dw, shift=shift), 3.5 / 1.5)

    def test_btls_max_step_with_ds_matches_iterative(self) -> None:
        """Joint-motion ds: closed-form override matches the default iterative impl."""
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([-0.5, 0.2, -1.5])
        shift = 0.4
        ds = 0.3
        # Closed form via the override.
        alpha_override = layer.btls_max_step(w, dw, shift=shift, ds=ds)
        # The boundary at alpha is reached when w_i + alpha dw_i = lb_i - (shift +
        # alpha ds), i.e. alpha (dw_i + ds) = lb_i - w_i - shift. Only rows with
        # dw_i + ds < 0 produce a finite cap. With ds = 0.3:
        # row 0: dw + ds = -0.2 -> (1.0 - 0 + 0.4)/0.2 = 7.0
        # row 1: dw + ds = 0.5 -> no cap
        # row 2: dw + ds = -1.2 -> (3.0 - 0 + 0.4)/1.2 = 2.8333...
        self.assertAlmostEqual(alpha_override, 3.4 / 1.2)
        # At alpha_override the active constraint (row 2, lb = 0) lands on the
        # joint boundary: f_2(w + alpha dw) = shift + alpha ds, where the
        # bounds layer encodes f_i(w) = lb_i - w_i.
        f_at_alpha = 0.0 - (w[2] + alpha_override * dw[2])
        rhs = shift + alpha_override * ds
        self.assertAlmostEqual(f_at_alpha, rhs, delta=1e-12)

    def test_btls_max_step_with_ds_negative_matches_no_cap(self) -> None:
        """With ds < 0 (shift shrinking), some rows that were inactive activate."""
        layer = BoundsConstraintLayer(lb=0.0, dimension=2)
        w = np.array([1.0, 2.0])
        dw = np.array([0.1, 0.2])  # all positive: no cap with ds = 0
        # ds = 0 -> infinity.
        assert layer.btls_max_step(w, dw) == np.inf
        # ds = -0.3 -> denom = dw - 0.3 = [-0.2, -0.1]; both active.
        # bound = (w - lb + shift)/(-(dw + ds)) -> shift=0:
        # row 0: 1.0 / 0.2 = 5.0; row 1: 2.0 / 0.1 = 20.0; min = 5.0.
        self.assertAlmostEqual(layer.btls_max_step(w, dw, ds=-0.3), 5.0)

    def test_btls_max_step_ignores_line_search(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([-0.5, 1.0, -1.5])
        without = layer.btls_max_step(w, dw)
        with_settings = layer.btls_max_step(
            w, dw, line_search=LineSearchSettings(beta=0.1, min_step=1e-12)
        )
        self.assertAlmostEqual(without, with_settings)

    def test_is_satisfied_strictly_above_lb(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        assert layer.is_satisfied(np.array([0.1, 1.0, 2.0])) is True

    def test_is_satisfied_at_lb_is_false(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=3)
        # Equality is not strict; should be False.
        assert layer.is_satisfied(np.array([0.0, 1.0, 2.0])) is False

    def test_is_satisfied_below_lb_is_false(self) -> None:
        layer = BoundsConstraintLayer(lb=0.5, dimension=3)
        assert layer.is_satisfied(np.array([0.6, 0.4, 1.0])) is False

    def test_is_satisfied_array_lb(self) -> None:
        lb = np.array([0.1, 0.2, 0.3])
        layer = BoundsConstraintLayer(lb=lb)
        assert layer.is_satisfied(np.array([0.2, 0.3, 0.4])) is True
        assert layer.is_satisfied(np.array([0.2, 0.2, 0.4])) is False


class BoundsConstraintLayerUpperOnlyMathTest(unittest.TestCase):
    """Math-only tests for the upper-bound-only configuration."""

    def test_num_constraints_returns_M(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=5)
        assert layer.num_constraints(5) == 5
        assert layer.num_constraints(100) == 100

    def test_constraints_scalar_ub_no_shift(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        c = layer.constraints(w)
        assert c.shape == (3,)
        # f_i = w_i - ub
        np.testing.assert_allclose(c, np.array([-9.0, -8.0, -7.0]))

    def test_constraints_array_ub_with_shift(self) -> None:
        ub = np.array([5.0, 6.0, 7.0])
        layer = BoundsConstraintLayer(ub=ub)
        w = np.array([1.0, 2.0, 3.0])
        shift = 0.5
        c = layer.constraints(w, shift=shift)
        # f_i = w_i - ub - shift
        np.testing.assert_allclose(c, np.array([-4.5, -4.5, -4.5]))

    def test_grad_constraints_returns_positive_identity(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=4)
        G = layer.grad_constraints(np.array([1.0, 2.0, 3.0, 4.0]))
        assert G.shape == (4, 4)
        np.testing.assert_allclose(G, np.eye(4))

    def test_grad_constraints_multiply_matches_materialized(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -1.5, 2.0])
        expected = layer.grad_constraints(w) @ v
        result = layer.grad_constraints_multiply(w, v)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, v)

    def test_grad_constraints_transpose_multiply_matches_materialized(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        u = np.array([0.5, -1.5, 2.0])
        expected = layer.grad_constraints(w).T @ u
        result = layer.grad_constraints_transpose_multiply(w, u)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, u)

    def test_btls_max_step_all_nonpos_dw_returns_inf(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([-0.1, 0.0, -0.5])
        assert layer.btls_max_step(w, dw) == np.inf

    def test_btls_max_step_some_positive_dw_array_ub(self) -> None:
        ub = np.array([5.0, 6.0, 7.0])
        layer = BoundsConstraintLayer(ub=ub)
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([0.5, -1.0, 1.5])
        # i=0: (5 - 1)/0.5 = 8.0; i=2: (7 - 3)/1.5 ≈ 2.667; min ≈ 2.667
        self.assertAlmostEqual(layer.btls_max_step(w, dw), 4.0 / 1.5)

    def test_btls_max_step_with_ds_uses_dw_minus_ds(self) -> None:
        """For the upper bound, the joint motion uses ``dw - ds`` (not ``+ ds``)."""
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        # All dw nonpositive: with ds=0 returns inf.
        dw = np.array([-0.1, 0.0, -0.5])
        assert layer.btls_max_step(w, dw) == np.inf
        # ds = -0.4 -> denom = dw - ds = dw + 0.4 = [0.3, 0.4, -0.1]; rows 0,1 active.
        # bound = (ub - w + shift)/denom -> shift=0:
        # row 0: 9/0.3 = 30; row 1: 8/0.4 = 20; min = 20.
        self.assertAlmostEqual(layer.btls_max_step(w, dw, ds=-0.4), 20.0)

    def test_is_satisfied_strictly_below_ub(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        assert layer.is_satisfied(np.array([0.0, 5.0, 9.9])) is True

    def test_is_satisfied_at_ub_is_false(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        assert layer.is_satisfied(np.array([0.0, 10.0, 5.0])) is False

    def test_is_satisfied_above_ub_is_false(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=3)
        assert layer.is_satisfied(np.array([0.0, 11.0, 5.0])) is False


class BoundsConstraintLayerTwoSidedMathTest(unittest.TestCase):
    """Math-only tests for the two-sided box configuration."""

    def test_num_constraints_returns_2M(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=5)
        assert layer.num_constraints(5) == 10
        assert layer.num_constraints(1) == 2
        assert layer.num_constraints(100) == 200

    def test_constraints_stack_layout(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        c = layer.constraints(w)
        assert c.shape == (6,)
        # First M: -w + lb = -w; second M: w - ub = w - 10.
        np.testing.assert_allclose(c[:3], np.array([-1.0, -2.0, -3.0]))
        np.testing.assert_allclose(c[3:], np.array([-9.0, -8.0, -7.0]))

    def test_constraints_with_shift_loosens_both_sides(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=2)
        w = np.array([1.0, 5.0])
        shift = 0.25
        c = layer.constraints(w, shift=shift)
        # Lower: -w + 0 - 0.25; upper: w - 10 - 0.25.
        np.testing.assert_allclose(c, np.array([-1.25, -5.25, -9.25, -5.25]))

    def test_grad_constraints_vstacks_neg_then_pos_identity(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        G = layer.grad_constraints(w)
        assert G.shape == (6, 3)
        np.testing.assert_allclose(G[:3], -np.eye(3))
        np.testing.assert_allclose(G[3:], np.eye(3))

    def test_grad_constraints_multiply_matches_materialized(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -1.5, 2.0])
        expected = layer.grad_constraints(w) @ v
        result = layer.grad_constraints_multiply(w, v)
        np.testing.assert_allclose(result, expected)
        # First M = -v, second M = v.
        np.testing.assert_allclose(result[:3], -v)
        np.testing.assert_allclose(result[3:], v)

    def test_grad_constraints_transpose_multiply_matches_materialized(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        w = np.array([1.0, 2.0, 3.0])
        u = np.array([0.5, -1.5, 2.0, 0.1, 0.2, 0.3])
        expected = layer.grad_constraints(w).T @ u
        result = layer.grad_constraints_transpose_multiply(w, u)
        np.testing.assert_allclose(result, expected)
        # Closed form: -u[:3] + u[3:].
        np.testing.assert_allclose(result, np.array([-0.4, 1.7, -1.7]))

    def test_btls_max_step_lower_binds_when_dw_negative(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        w = np.array([1.0, 5.0, 5.0])
        dw = np.array([-0.5, 0.0, 0.0])
        # Only lower side bites at row 0: alpha = (1 - 0)/0.5 = 2.0.
        self.assertAlmostEqual(layer.btls_max_step(w, dw), 2.0)

    def test_btls_max_step_upper_binds_when_dw_positive(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        w = np.array([5.0, 5.0, 1.0])
        dw = np.array([0.0, 0.0, 1.5])
        # Only upper side bites at row 2: alpha = (10 - 1)/1.5 = 6.0.
        self.assertAlmostEqual(layer.btls_max_step(w, dw), 6.0)

    def test_btls_max_step_min_over_both_sides(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=2)
        w = np.array([1.0, 5.0])
        dw = np.array([-0.5, 2.0])
        # Lower at row 0: 1.0/0.5 = 2.0. Upper at row 1: (10-5)/2 = 2.5. min = 2.0.
        self.assertAlmostEqual(layer.btls_max_step(w, dw), 2.0)

    def test_btls_max_step_zero_direction_returns_inf(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=2)
        w = np.array([1.0, 5.0])
        dw = np.zeros(2)
        assert layer.btls_max_step(w, dw) == np.inf

    def test_is_satisfied_inside_box(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)
        assert layer.is_satisfied(np.array([0.5, 5.0, 9.5])) is True

    def test_is_satisfied_at_lb_or_ub_is_false(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=2)
        assert layer.is_satisfied(np.array([0.0, 5.0])) is False
        assert layer.is_satisfied(np.array([5.0, 10.0])) is False


class BoundsConstraintLayerInfiniteEntriesTest(unittest.TestCase):
    """An infinite bound means no bound: it contributes no constraint."""

    @staticmethod
    def _layer() -> BoundsConstraintLayer:
        # Component 1 is uncapped, component 3 has no lower bound, component 2
        # has neither.
        return BoundsConstraintLayer(
            lb=np.array([0.0, 0.0, -np.inf, -np.inf]),
            ub=np.array([10.0, np.inf, np.inf, 5.0]),
        )

    def test_infinite_entries_add_no_constraints(self) -> None:
        self.assertEqual(self._layer().num_constraints(4), 4)

    def test_constraints_hold_only_the_finite_bounds(self) -> None:
        w = np.array([1.0, 2.0, 3.0, 4.0])
        # Lower block: components 0, 1. Upper block: components 0, 3.
        np.testing.assert_allclose(
            self._layer().constraints(w, shift=0.5),
            np.array([-1.5, -2.5, -9.5, -1.5]),
        )

    def test_barrier_objective_is_finite(self) -> None:
        w = np.array([1.0, 2.0, 3.0, 4.0])
        expected = -np.sum(np.log([1.0, 2.0, 9.0, 1.0]))
        self.assertAlmostEqual(self._layer().barrier_objective(w), expected)

    def test_jacobian_products_match_the_materialized_jacobian(self) -> None:
        layer = self._layer()
        w = np.array([1.0, 2.0, 3.0, 4.0])
        G = layer.grad_constraints(w)
        np.testing.assert_allclose(
            G,
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        v = np.array([0.5, -1.5, 2.0, 0.25])
        np.testing.assert_allclose(layer.grad_constraints_multiply(w, v), G @ v)
        u = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(w, u), G.T @ u
        )

    def test_curvature_comes_only_from_the_finite_bounds(self) -> None:
        w = np.array([1.0, 2.0, 3.0, 4.0])
        lmbda = np.array([1.0, 0.5, 1.0 / 9.0, 1.0])  # 1 / slack, as the barrier sets
        system = self._layer().hessian_contribution(w, lmbda)
        assert isinstance(system, DiagonalSystem)
        np.testing.assert_allclose(
            system.eta, np.array([1.0 + 1.0 / 81.0, 0.25, 0.0, 1.0])
        )
        np.testing.assert_allclose(
            system.eta_inverse, np.array([81.0 / 82.0, 4.0, np.inf, 1.0])
        )

    def test_finite_bounds_keep_the_full_layout(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, ub=np.array([10.0, 20.0]))
        self.assertEqual(layer.num_constraints(2), 4)


class BoundsConstraintLayerValidationTest(unittest.TestCase):
    """Construction-time invariants."""

    def test_both_none_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            BoundsConstraintLayer()

    def test_lb_equals_ub_scalar_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "lb < ub"):
            BoundsConstraintLayer(lb=1.0, ub=1.0)

    def test_lb_greater_than_ub_scalar_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "lb < ub"):
            BoundsConstraintLayer(lb=2.0, ub=1.0)

    def test_lb_array_overlaps_ub_scalar_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "lb < ub"):
            BoundsConstraintLayer(lb=np.array([0.0, 0.0, 5.0]), ub=4.0)

    def test_incompatible_shapes_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "broadcast-compatible"):
            BoundsConstraintLayer(
                lb=np.array([0.0, 1.0, 2.0]),
                ub=np.array([5.0, 6.0]),
            )

    def test_scalar_pair_strict_ok(self) -> None:
        # No exception.
        BoundsConstraintLayer(lb=-1.0, ub=1.0, dimension=2)

    def test_array_pair_strict_ok(self) -> None:
        BoundsConstraintLayer(
            lb=np.array([0.0, 0.5]),
            ub=np.array([1.0, 1.5]),
        )

    def test_lb_only_ok(self) -> None:
        layer = BoundsConstraintLayer(lb=0.0, dimension=2)
        assert layer.lb == 0.0
        assert layer.ub is None

    def test_ub_only_ok(self) -> None:
        layer = BoundsConstraintLayer(ub=10.0, dimension=2)
        assert layer.lb is None
        assert layer.ub == 10.0

    def test_scalar_bounds_require_dimension(self) -> None:
        """Scalar bounds pin no size, so an explicit ``dimension`` is required."""
        with self.assertRaisesRegex(ValueError, "must specify `dimension`"):
            BoundsConstraintLayer(lb=0.0, ub=1.0)
        with self.assertRaisesRegex(ValueError, "must specify `dimension`"):
            BoundsConstraintLayer(lb=-1.0)
        with self.assertRaisesRegex(ValueError, "must specify `dimension`"):
            BoundsConstraintLayer(ub=1.0)

    def test_vector_bounds_infer_dimension(self) -> None:
        """An array bound pins ``M`` via its length; no explicit dimension needed."""
        assert BoundsConstraintLayer(lb=np.zeros(4)).dimension == 4
        assert BoundsConstraintLayer(ub=np.ones(3)).dimension == 3
        assert BoundsConstraintLayer(lb=[-1.0, -1.0], ub=[1.0, 1.0]).dimension == 2

    def test_explicit_dimension_matches_array_ok(self) -> None:
        """An explicit ``dimension`` agreeing with the array length is accepted."""
        assert BoundsConstraintLayer(lb=np.zeros(5), dimension=5).dimension == 5

    def test_explicit_dimension_disagrees_with_array_raises(self) -> None:
        """An explicit ``dimension`` that contradicts the array length raises."""
        with self.assertRaisesRegex(ValueError, "disagrees"):
            BoundsConstraintLayer(lb=np.zeros(4), dimension=3)

    def test_array_bounds_disagree_on_length_raises(self) -> None:
        """Two array bounds of different lengths cannot pin a single ``M``."""
        with self.assertRaisesRegex(ValueError, "broadcast-compatible|agree"):
            BoundsConstraintLayer(lb=np.zeros(4), ub=np.ones(3))

    def test_zero_dim_ndarray_treated_as_scalar(self) -> None:
        """A 0-d ndarray bound is treated as a scalar, not a length-1 array."""
        with self.assertRaisesRegex(ValueError, "must specify `dimension`"):
            BoundsConstraintLayer(lb=np.array(0.0), ub=np.array(1.0))
        assert BoundsConstraintLayer(lb=np.array(0.0), dimension=3).dimension == 3


class ImbalanceConstraintLayerMathTest(unittest.TestCase):
    """Math-only tests for ImbalanceConstraintLayer.

    End-to-end Phase I behaviour is exercised by the PyRake test suite (which drives
    this layer through `JointCalibrator`); these tests pin down the per-method math.

    """

    def _make_layer(
        self,
        scalar_psi: bool = False,
    ) -> tuple[
        ImbalanceConstraintLayer,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | float,
    ]:
        B = np.array([[1.0, 0.5, -0.25], [0.0, 2.0, 1.0]])
        c = np.array([0.1, -0.2])
        psi: npt.NDArray[np.float64] | float
        if scalar_psi:
            psi = 0.5
        else:
            psi = np.array([0.4, 0.6])
        layer = ImbalanceConstraintLayer(B=B, c=c, psi=psi)
        return layer, B, c, psi

    def test_num_constraints_returns_two_q(self) -> None:
        layer, B, _, _ = self._make_layer()
        assert layer.num_constraints(B.shape[1]) == 2 * B.shape[0]
        assert layer.num_constraints(B.shape[1]) == 4

    def test_q_property(self) -> None:
        layer, B, _, _ = self._make_layer()
        assert layer.q == B.shape[0]

    def test_psi_property_array(self) -> None:
        layer, _, _, psi = self._make_layer()
        assert isinstance(psi, np.ndarray)
        np.testing.assert_allclose(layer.psi, psi)

    def test_psi_property_scalar_broadcasts(self) -> None:
        layer, B, _, psi = self._make_layer(scalar_psi=True)
        expected = np.full(B.shape[0], psi, dtype=np.float64)
        assert layer.psi.shape == (B.shape[0],)
        np.testing.assert_allclose(layer.psi, expected)

    def test_constraints_scalar_psi_no_shift(self) -> None:
        layer, B, c, psi = self._make_layer(scalar_psi=True)
        w = np.array([1.0, -0.5, 2.0])
        Bw_minus_c = B @ w - c
        expected = np.concatenate([-Bw_minus_c - psi, Bw_minus_c - psi])
        out = layer.constraints(w)
        assert out.shape == (2 * B.shape[0],)
        np.testing.assert_allclose(out, expected)

    def test_constraints_array_psi_no_shift(self) -> None:
        layer, B, c, psi = self._make_layer()
        w = np.array([1.0, -0.5, 2.0])
        Bw_minus_c = B @ w - c
        expected = np.concatenate(
            [-Bw_minus_c - np.asarray(psi), Bw_minus_c - np.asarray(psi)]
        )
        np.testing.assert_allclose(layer.constraints(w), expected)

    def test_constraints_array_psi_with_shift(self) -> None:
        layer, B, c, psi = self._make_layer()
        w = np.array([1.0, -0.5, 2.0])
        shift = 0.05
        Bw_minus_c = B @ w - c
        expected = np.concatenate(
            [
                -Bw_minus_c - np.asarray(psi) - shift,
                Bw_minus_c - np.asarray(psi) - shift,
            ]
        )
        np.testing.assert_allclose(layer.constraints(w, shift=shift), expected)

    def test_grad_constraints_returns_stacked_neg_pos_B(self) -> None:
        layer, B, _, _ = self._make_layer()
        w = np.array([1.0, -0.5, 2.0])
        G = layer.grad_constraints(w)
        assert G.shape == (2 * B.shape[0], B.shape[1])
        np.testing.assert_allclose(G, np.vstack([-B, B]))

    def test_grad_constraints_multiply_matches_materialized(self) -> None:
        layer, B, _, _ = self._make_layer()
        w = np.array([1.0, -0.5, 2.0])
        rng = np.random.default_rng(0)
        v = rng.standard_normal(int(B.shape[1]))
        G = layer.grad_constraints(w)
        np.testing.assert_allclose(layer.grad_constraints_multiply(w, v), G @ v)

    def test_grad_constraints_transpose_multiply_matches_materialized(self) -> None:
        layer, B, _, _ = self._make_layer()
        w = np.array([1.0, -0.5, 2.0])
        rng = np.random.default_rng(1)
        u = rng.standard_normal(2 * B.shape[0])
        G = layer.grad_constraints(w)
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(w, u), G.T @ u
        )

    def test_btls_max_step_zero_B_dw_returns_inf(self) -> None:
        layer, B, _, _ = self._make_layer()
        w = np.array([1.0, -0.5, 2.0])
        # The cross product of B's rows spans its null space. Its entries and every
        # product with B are exact in binary floating point, so B @ dw is exactly
        # zero on any BLAS (an SVD null vector leaves ~1e-16, a finite step).
        dw = np.array([1.0, -1.0, 2.0])
        assert np.all(B @ dw == 0.0)
        assert layer.btls_max_step(w, dw) == np.inf

    def test_btls_max_step_matches_closed_form(self) -> None:
        layer, B, c, psi = self._make_layer()
        psi_arr = np.asarray(psi)
        # Pick a strictly feasible w.
        w = np.zeros(B.shape[1])
        Bw_minus_c = B @ w - c
        assert np.all(np.abs(Bw_minus_c) < psi_arr)

        rng = np.random.default_rng(11)
        dw = rng.standard_normal(int(B.shape[1]))
        B_dw = B @ dw

        # Closed-form min over both sides.
        candidates = []
        for i in range(B.shape[0]):
            if B_dw[i] > 0:
                candidates.append((psi_arr[i] - Bw_minus_c[i]) / B_dw[i])
            if B_dw[i] < 0:
                candidates.append((psi_arr[i] + Bw_minus_c[i]) / -B_dw[i])
        expected = min(candidates) if candidates else np.inf

        self.assertAlmostEqual(layer.btls_max_step(w, dw), expected)

    def test_btls_max_step_ignores_line_search(self) -> None:
        layer, B, c, _ = self._make_layer()
        w = np.zeros(B.shape[1])
        rng = np.random.default_rng(3)
        dw = rng.standard_normal(int(B.shape[1]))
        without = layer.btls_max_step(w, dw)
        with_settings = layer.btls_max_step(
            w, dw, line_search=LineSearchSettings(beta=0.1, min_step=1e-12)
        )
        self.assertAlmostEqual(without, with_settings)

    def test_btls_max_step_with_ds_matches_brute_force(self) -> None:
        """Joint-motion ds: override agrees with a brute-force closed-form computation."""
        layer, B, c, psi = self._make_layer()
        psi_arr = np.asarray(psi)
        w = np.zeros(B.shape[1])
        Bw_minus_c = B @ w - c
        assert np.all(np.abs(Bw_minus_c) < psi_arr)

        rng = np.random.default_rng(1717)
        dw = rng.standard_normal(int(B.shape[1]))
        B_dw = B @ dw
        shift = 0.05
        ds = 0.07

        # Brute-force candidate set: for each side, compute alpha bound when
        # the per-row denom > 0.
        candidates = []
        for i in range(B.shape[0]):
            d_pos = B_dw[i] - ds
            if d_pos > 0:
                candidates.append((psi_arr[i] + shift - Bw_minus_c[i]) / d_pos)
            d_neg = -B_dw[i] - ds
            if d_neg > 0:
                candidates.append((psi_arr[i] + shift + Bw_minus_c[i]) / d_neg)
        expected = min(candidates) if candidates else np.inf

        self.assertAlmostEqual(layer.btls_max_step(w, dw, shift=shift, ds=ds), expected)

    def test_is_satisfied_strictly_inside(self) -> None:
        layer, B, c, psi = self._make_layer()
        w = np.zeros(B.shape[1])
        # |Bw - c| = |c| = [0.1, 0.2]; psi = [0.4, 0.6].
        assert layer.is_satisfied(w) is True

    def test_is_satisfied_at_boundary_is_false(self) -> None:
        # Construct w where |Bw - c|_inf == psi exactly (not strictly less).
        B = np.array([[1.0, 0.0]])
        c = np.array([0.0])
        psi = np.array([0.5])
        layer = ImbalanceConstraintLayer(B=B, c=c, psi=psi)
        # Bw - c = w[0]; pick w[0] = 0.5 -> equal to psi.
        w = np.array([0.5, 0.0])
        assert layer.is_satisfied(w) is False

    def test_is_satisfied_outside_is_false(self) -> None:
        layer, B, _, _ = self._make_layer()
        # Pick w making the first row blow past psi.
        w = np.array([10.0, 0.0, 0.0])
        assert layer.is_satisfied(w) is False


class NormConstraintLayerMathTest(unittest.TestCase):
    """Math-only tests for NormConstraintLayer.

    The structure-aware Phase I solver (`MinimizeConstraintValueSolver`-based) does not
    yet exist, so end-to-end solve tests are intentionally omitted; these tests pin down
    the per-method math.

    """

    def test_invalid_phi(self) -> None:
        """phi must be strictly positive."""
        B = np.eye(3)
        c = np.zeros(3)
        with self.assertRaises(ValueError):
            NormConstraintLayer(B=B, c=c, phi=0.0)
        with self.assertRaises(ValueError):
            NormConstraintLayer(B=B, c=c, phi=-1.0)

    def test_invalid_shapes(self) -> None:
        """B and c must have compatible shapes."""
        B = np.zeros((3, 5))
        with self.assertRaises(ValueError):
            NormConstraintLayer(B=B, c=np.zeros(4), phi=1.0)
        with self.assertRaises(ValueError):
            NormConstraintLayer(B=np.zeros(5), c=np.zeros(5), phi=1.0)

    def test_constraint_value(self) -> None:
        """f(w) = ||Bw - c||^2 - phi^2 (squared-form encoding)."""
        rng = np.random.default_rng(9001)
        M, q = 7, 4
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        phi = 2.5
        layer = NormConstraintLayer(B=B, c=c, phi=phi)
        w = rng.standard_normal(M)
        v = B @ w - c
        expected = float(np.dot(v, v)) - phi * phi
        np.testing.assert_allclose(layer.constraints(w), [expected], atol=1e-12)
        # Non-zero shift is rejected by the squared-form encoding.
        with self.assertRaises(AssertionError):
            layer.constraints(w, shift=0.3)

    @parameterized.expand([(9101,), (9102,), (9103,)])
    def test_grad_consistency(self, seed: int) -> None:
        """Analytical gradient matches central-difference numerical gradient."""
        rng = np.random.default_rng(seed)
        M, q = 6, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        phi = 5.0
        layer = NormConstraintLayer(B=B, c=c, phi=phi)
        w = rng.standard_normal(M)

        analytical = layer.grad_constraints(w)[0]
        h = 1e-6
        numerical = np.zeros(M)
        for j in range(M):
            e_j = np.zeros(M)
            e_j[j] = 1.0
            numerical[j] = (
                layer.constraints(w + h * e_j)[0] - layer.constraints(w - h * e_j)[0]
            ) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-7)

    @parameterized.expand([(9111,), (9112,)])
    def test_grad_constraints_multiply_consistency(self, seed: int) -> None:
        """grad_constraints_multiply(w, v) matches grad_constraints(w) @ v."""
        rng = np.random.default_rng(seed)
        M, q = 6, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        layer = NormConstraintLayer(B=B, c=c, phi=4.0)
        w = rng.standard_normal(M)
        v = rng.standard_normal(M)
        expected = layer.grad_constraints(w) @ v
        np.testing.assert_allclose(
            layer.grad_constraints_multiply(w, v), expected, atol=1e-12
        )

    @parameterized.expand([(9121,), (9122,)])
    def test_grad_constraints_transpose_multiply_consistency(self, seed: int) -> None:
        """grad_constraints_transpose_multiply(w, u) matches grad_constraints(w).T @ u."""
        rng = np.random.default_rng(seed)
        M, q = 6, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        layer = NormConstraintLayer(B=B, c=c, phi=4.0)
        w = rng.standard_normal(M)
        u = rng.standard_normal(1)
        expected = layer.grad_constraints(w).T @ u
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(w, u), expected, atol=1e-12
        )

    def test_grad_constraints_transpose_multiply_empty_u(self) -> None:
        """Empty multiplier slice returns the zero primal vector.

        Arises when the layered IPM driver iterates layers and slices a length-zero
        multiplier (e.g., during Phase I dual evaluation, where the active layer's
        contribution to the running G^T u sum is zero).

        """
        rng = np.random.default_rng(9131)
        M, q = 6, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        layer = NormConstraintLayer(B=B, c=c, phi=4.0)
        w = rng.standard_normal(M)
        result = layer.grad_constraints_transpose_multiply(
            w, np.array([], dtype=np.float64)
        )
        np.testing.assert_array_equal(result, np.zeros(M))

    def test_grad_returns_zero_at_origin(self) -> None:
        """When B w == c, grad_constraints returns zeros (smooth squared form)."""
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = np.array([0.5, -0.5])
        layer = NormConstraintLayer(B=B, c=c, phi=1.0)
        # At w = c, Bw - c = 0; the squared-form gradient 2 B^T (Bw - c) is
        # naturally zero (no subgradient branch needed).
        w = c.copy()
        G = layer.grad_constraints(w)
        assert G.shape == (1, 2)
        np.testing.assert_allclose(G, np.zeros((1, 2)), atol=1e-12)
        # Multiplies should also yield zeros.
        v = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            layer.grad_constraints_multiply(w, v), np.zeros(1), atol=1e-12
        )
        u = np.array([0.7])
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(w, u), np.zeros(2), atol=1e-12
        )

    @parameterized.expand([(9301,), (9302,)])
    def test_btls_max_step(self, seed: int) -> None:
        """btls_max_step lands exactly on the boundary."""
        rng = np.random.default_rng(seed)
        M, q = 5, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        phi = 2.0
        layer = NormConstraintLayer(B=B, c=c, phi=phi)

        # Strictly feasible w with ||Bw - c|| < phi.
        w = np.asarray(np.linalg.lstsq(B, c, rcond=None)[0], dtype=np.float64)
        assert np.linalg.norm(B @ w - c) < phi
        dw = rng.standard_normal(M)
        alpha = layer.btls_max_step(w, dw)
        boundary = np.linalg.norm(B @ (w + alpha * dw) - c)
        np.testing.assert_allclose(boundary, phi, atol=1e-9)

    def test_btls_max_step_rejects_nonzero_shift_and_ds(self) -> None:
        """Squared-form encoding asserts shift == 0 and ds == 0 in btls_max_step."""
        rng = np.random.default_rng(11_201)
        M, q = 5, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        layer = NormConstraintLayer(B=B, c=c, phi=2.0)
        w = np.asarray(np.linalg.lstsq(B, c, rcond=None)[0], dtype=np.float64)
        dw = rng.standard_normal(M)
        with self.assertRaises(AssertionError):
            layer.btls_max_step(w, dw, shift=0.3)
        with self.assertRaises(AssertionError):
            layer.btls_max_step(w, dw, ds=0.05)

    def test_is_satisfied(self) -> None:
        """is_satisfied checks strict feasibility."""
        B = np.eye(3)
        c = np.zeros(3)
        phi = 1.0
        layer = NormConstraintLayer(B=B, c=c, phi=phi)
        assert layer.is_satisfied(np.array([0.5, 0.0, 0.0]))
        assert not layer.is_satisfied(np.array([1.0, 0.0, 0.0]))
        assert not layer.is_satisfied(np.array([2.0, 0.0, 0.0]))

    def test_hessian_curvature_matches_lambda_hessian(self) -> None:
        """`hessian_curvature` equals `lambda * nabla^2 f = 2 lambda B^T B`."""
        rng = np.random.default_rng(2027)
        M, q = 5, 3
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        layer = NormConstraintLayer(B=B, c=c, phi=4.0)
        lam = 0.85
        system = layer.hessian_curvature(rng.standard_normal(M), np.array([lam]))
        expected = 2.0 * lam * (B.T @ B)
        v = rng.standard_normal(M)
        np.testing.assert_allclose(
            system.multiply(v), expected @ v, rtol=1e-10, atol=1e-12
        )

    def test_hessian_contribution_matches_dense(self) -> None:
        """Structured `hessian_contribution` assembles the same barrier Hessian.

        With the barrier weight `lambda = -1/f` the layer's contribution, summed
        onto a diagonal base `D = diag(eta)`, must reproduce the dense barrier
        Hessian

            H = D + (-2/f) B^T B + (4/f^2) (B^T v)(B^T v)^T.

        Verifies both ``solve`` and ``multiply``.
        """
        rng = np.random.default_rng(2026)
        M, q = 4, 3
        B = rng.standard_normal((q, M))
        c_vec = rng.standard_normal(q)
        phi = 5.0
        layer = NormConstraintLayer(B=B, c=c_vec, phi=phi)
        w = rng.standard_normal(M)
        v = B @ w - c_vec
        assert float(np.linalg.norm(v)) < phi

        eta = np.array([0.7, 1.3, 0.5, 2.1])
        f = float(np.dot(v, v)) - phi * phi  # shift = 0; f < 0 on the interior
        lam = -1.0 / f  # barrier weight

        system = DiagonalSystem(eta).plus(
            layer.hessian_contribution(w, np.array([lam]))
        )

        BTv = B.T @ v
        H = np.diag(eta) + (-2.0 / f) * (B.T @ B) + (4.0 / (f * f)) * np.outer(BTv, BTv)

        rhs = rng.standard_normal(M)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    def test_hessian_contribution_requires_lmbda(self) -> None:
        """``hessian_contribution`` raises when called without per-constraint weights."""
        layer = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=2.0)
        with self.assertRaisesRegex(ValueError, "requires `lmbda`"):
            layer.hessian_contribution(np.zeros(2))

    def test_hessian_contribution_out_of_domain(self) -> None:
        """``hessian_contribution`` raises when the iterate is infeasible (f >= 0)."""
        layer = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=1.0)
        w = np.array([2.0, 0.0])  # ||Bw - c||^2 = 4 > phi^2 = 1
        with self.assertRaisesRegex(ValueError, "out of domain"):
            layer.hessian_contribution(w, np.array([1.0]))

    def test_hessian_contribution_rejects_shift(self) -> None:
        """``hessian_contribution`` rejects a non-zero ``shift`` (squared-form)."""
        layer = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=2.0)
        w = np.array([0.5, 0.0])  # ||Bw - c||^2 = 0.25 < phi^2 = 4 (feasible)
        with self.assertRaisesRegex(AssertionError, "non-zero shift"):
            layer.hessian_contribution(w, np.array([1.0]), shift=0.3)

    def test_interior_point_returns_center(self) -> None:
        """``interior_point`` returns the ellipsoid center `B^{-1} c`, strictly interior."""
        B = np.array([[2.0, 0.0], [0.0, 3.0]])
        c = np.array([2.0, 3.0])  # center = B^{-1} c = [1, 1]
        layer = NormConstraintLayer(B=B, c=c, phi=1.0)
        w = layer.interior_point()
        assert w is not None
        np.testing.assert_allclose(w, np.array([1.0, 1.0]), atol=1e-12)
        # The center has zero residual, so f(w) = -phi^2 < 0: strictly interior.
        assert float(layer.constraints(w)[0]) < 0.0

    def test_interior_point_none_when_empty_interior(self) -> None:
        """``interior_point`` returns ``None`` when no strictly interior point exists.

        With an overdetermined ``B`` whose least-squares residual to ``c`` already
        exceeds ``phi``, the ellipsoid has empty (strict) interior.
        """
        B = np.array([[1.0], [0.0]])  # shape (2, 1)
        c = np.array([0.0, 5.0])  # min ||B w - c|| = 5 (at w = 0), attained residual
        layer = NormConstraintLayer(B=B, c=c, phi=1.0)  # 5 >= phi = 1: no interior
        assert layer.interior_point() is None

    def test_interior_point_with_equality_returns_constrained_center(self) -> None:
        """``interior_point(eq)`` returns the equality-constrained least-squares point.

        ``min ||w||^2 s.t. w1 + w2 = 2`` has analytic solution ``w = [1, 1]``. It
        lies on the manifold and is strictly inside the radius-5 ball, so the hook
        returns it (rather than the unconstrained center, the origin).
        """
        layer = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=5.0)
        eq = EqualityConstraintLayer(A=np.array([[1.0, 1.0]]), b=np.array([2.0]))
        w = layer.interior_point(eq)
        assert w is not None
        np.testing.assert_allclose(w, np.array([1.0, 1.0]), atol=1e-10)
        np.testing.assert_allclose(eq.A @ w, eq.b, atol=1e-10)  # on the manifold
        assert float(layer.constraints(w)[0]) < 0.0  # strictly interior

    def test_interior_point_with_equality_none_when_manifold_misses_ball(self) -> None:
        """``interior_point(eq)`` is ``None`` when the manifold misses the ball's interior.

        The closest manifold point to the ball center is ``[1, 1]`` with residual
        ``sqrt(2) ~= 1.414``; with ``phi = 1`` that exceeds the radius, so the ball
        and manifold do not intersect in the strict interior and ``None`` results.
        """
        layer = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=1.0)
        eq = EqualityConstraintLayer(A=np.array([[1.0, 1.0]]), b=np.array([2.0]))
        assert layer.interior_point(eq) is None


class SquaredNormConstraintLayerMathTest(unittest.TestCase):
    """Math-only tests for SquaredNormConstraintLayer."""

    def test_invalid_phi(self) -> None:
        """phi must be strictly positive."""
        with self.assertRaises(ValueError):
            SquaredNormConstraintLayer(phi=0.0, dimension=3)
        with self.assertRaises(ValueError):
            SquaredNormConstraintLayer(phi=-1.0, dimension=3)

    def test_requires_dimension(self) -> None:
        """``dimension`` is a required arg and is stored as the pinned size.

        The scalar ``phi`` pins no size on its own, so the constructor takes an
        explicit ``dimension``; omitting it is a type error caught statically (and a
        ``TypeError`` at runtime), so the runtime check here pins the stored value.
        """
        M = 4
        layer = SquaredNormConstraintLayer(phi=1.0, dimension=M)
        assert layer.dimension == M

    def test_num_constraints_is_one(self) -> None:
        """Layer always emits exactly one scalar constraint."""
        layer = SquaredNormConstraintLayer(phi=1.0, dimension=1)
        assert layer.num_constraints(1) == 1
        assert layer.num_constraints(7) == 1
        assert layer.num_constraints(100) == 1

    def test_constraint_value(self) -> None:
        """f(x) = ||x||^2 - phi, with and without shift."""
        rng = np.random.default_rng(11001)
        M = 7
        phi = 2.5
        layer = SquaredNormConstraintLayer(phi=phi, dimension=M)
        x = rng.standard_normal(M)
        expected = float(np.dot(x, x)) - phi
        np.testing.assert_allclose(layer.constraints(x), [expected], atol=1e-12)
        np.testing.assert_allclose(
            layer.constraints(x, shift=0.3), [expected - 0.3], atol=1e-12
        )

    @parameterized.expand([(11101,), (11102,), (11103,)])
    def test_grad_consistency(self, seed: int) -> None:
        """Analytical gradient matches central-difference numerical gradient."""
        rng = np.random.default_rng(seed)
        M = 6
        layer = SquaredNormConstraintLayer(phi=5.0, dimension=M)
        x = rng.standard_normal(M)

        analytical = layer.grad_constraints(x)[0]
        np.testing.assert_allclose(analytical, 2.0 * x, atol=1e-12)

        h = 1e-6
        numerical = np.zeros(M)
        for j in range(M):
            e_j = np.zeros(M)
            e_j[j] = 1.0
            numerical[j] = (
                layer.constraints(x + h * e_j)[0] - layer.constraints(x - h * e_j)[0]
            ) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-7)

    @parameterized.expand([(11111,), (11112,)])
    def test_grad_constraints_multiply_consistency(self, seed: int) -> None:
        """grad_constraints_multiply(x, v) matches grad_constraints(x) @ v."""
        rng = np.random.default_rng(seed)
        M = 6
        layer = SquaredNormConstraintLayer(phi=4.0, dimension=M)
        x = rng.standard_normal(M)
        v = rng.standard_normal(M)
        expected = layer.grad_constraints(x) @ v
        np.testing.assert_allclose(
            layer.grad_constraints_multiply(x, v), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            layer.grad_constraints_multiply(x, v),
            [2.0 * float(np.dot(x, v))],
            atol=1e-12,
        )

    @parameterized.expand([(11121,), (11122,)])
    def test_grad_constraints_transpose_multiply_consistency(self, seed: int) -> None:
        """grad_constraints_transpose_multiply(x, u) matches grad_constraints(x).T @ u."""
        rng = np.random.default_rng(seed)
        M = 6
        layer = SquaredNormConstraintLayer(phi=4.0, dimension=M)
        x = rng.standard_normal(M)
        u = rng.standard_normal(1)
        expected = layer.grad_constraints(x).T @ u
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(x, u), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            layer.grad_constraints_transpose_multiply(x, u),
            2.0 * float(u[0]) * x,
            atol=1e-12,
        )

    def test_grad_constraints_transpose_multiply_empty_u(self) -> None:
        """Empty multiplier slice returns the zero primal vector.

        Arises when the layered IPM driver iterates layers and slices a length-zero
        multiplier (e.g., during Phase I dual evaluation, where the active layer's
        contribution to the running G^T u sum is zero).

        """
        rng = np.random.default_rng(11231)
        M = 5
        layer = SquaredNormConstraintLayer(phi=4.0, dimension=M)
        x = rng.standard_normal(M)
        result = layer.grad_constraints_transpose_multiply(
            x, np.array([], dtype=np.float64)
        )
        np.testing.assert_array_equal(result, np.zeros(M))

    @parameterized.expand([(0.0,), (0.3,)])
    def test_btls_max_step_lands_on_boundary(self, shift: float) -> None:
        """btls_max_step lands exactly on the squared-norm boundary phi + shift."""
        rng = np.random.default_rng(11301)
        M = 5
        phi = 2.0
        layer = SquaredNormConstraintLayer(phi=phi, dimension=M)
        # Strictly feasible x with ||x||^2 < phi + shift.
        x: npt.NDArray[np.float64] = np.asarray(0.3 * rng.standard_normal(M))
        assert float(np.dot(x, x)) < phi + shift
        # Random direction; with positive probability ||x + a dx||^2 grows.
        dx = rng.standard_normal(M)
        alpha = layer.btls_max_step(x, dx, shift=shift)
        assert np.isfinite(alpha)
        assert alpha > 0
        boundary = float(np.dot(x + alpha * dx, x + alpha * dx))
        np.testing.assert_allclose(boundary, phi + shift, atol=1e-9)

    @parameterized.expand([(0.2,), (-0.2,)])
    def test_btls_max_step_with_ds_lands_on_joint_boundary(self, ds: float) -> None:
        """Joint motion: ||x + alpha dx||^2 = phi + shift + alpha ds at the cap."""
        rng = np.random.default_rng(11_311)
        M = 5
        phi = 2.0
        layer = SquaredNormConstraintLayer(phi=phi, dimension=M)
        x: npt.NDArray[np.float64] = np.asarray(0.3 * rng.standard_normal(M))
        shift = 0.3
        assert float(np.dot(x, x)) < phi + shift
        dx = rng.standard_normal(M)
        alpha = layer.btls_max_step(x, dx, shift=shift, ds=ds)
        assert np.isfinite(alpha) and alpha > 0
        boundary = float(np.dot(x + alpha * dx, x + alpha * dx))
        np.testing.assert_allclose(boundary, phi + shift + alpha * ds, atol=1e-9)

    def test_btls_max_step_inf_when_dw_zero(self) -> None:
        """When dw == 0, btls_max_step is infinite (constraint is constant)."""
        layer = SquaredNormConstraintLayer(phi=1.0, dimension=3)
        x = np.array([0.1, -0.2, 0.3])
        dw = np.zeros(3)
        assert layer.btls_max_step(x, dw) == np.inf

    def test_btls_max_step_when_norm_decreases_returns_finite_root(self) -> None:
        """Returns a finite root for an inward initial direction.

        ||x + a dx||^2 is an upward parabola in alpha, so even a direction that
        initially decreases the norm still has a finite outer root where the trajectory
        re-enters the boundary.

        """
        layer = SquaredNormConstraintLayer(phi=1.0, dimension=2)
        # x near the boundary, dx pointing inward.
        x = np.array([0.9, 0.0])
        assert float(np.dot(x, x)) < 1.0
        dx = -x  # ||x + a dx||^2 = (1 - a)^2 ||x||^2; minimized at a=1.
        alpha = layer.btls_max_step(x, dx)
        # At alpha = 2, x + a dx = -x; ||.||^2 = ||x||^2 < phi. The parabola
        # ||x + a dx||^2 = (1 - a)^2 * 0.81 reaches phi=1.0 when |1-a| = 1/0.9,
        # i.e. a = 1 + 1/0.9 ~= 2.111 (the larger positive root).
        boundary = float(np.dot(x + alpha * dx, x + alpha * dx))
        np.testing.assert_allclose(boundary, 1.0, atol=1e-9)
        np.testing.assert_allclose(alpha, 1.0 + 1.0 / 0.9, atol=1e-9)

    def test_is_satisfied_strict(self) -> None:
        """is_satisfied checks strict feasibility (boundary returns False)."""
        layer = SquaredNormConstraintLayer(phi=1.0, dimension=3)
        assert layer.is_satisfied(np.array([0.5, 0.0, 0.0]))
        # ||x||^2 == phi -> not strictly satisfied.
        assert not layer.is_satisfied(np.array([1.0, 0.0, 0.0]))
        assert not layer.is_satisfied(np.array([2.0, 0.0, 0.0]))

    def test_hessian_curvature_matches_lambda_hessian(self) -> None:
        """`hessian_curvature` equals `lambda * nabla^2 f = 2 lambda I_M`."""
        M = 6
        layer = SquaredNormConstraintLayer(phi=1.5, dimension=M)
        lam = 0.4
        system = layer.hessian_curvature(np.zeros(M), np.array([lam]))
        expected = 2.0 * lam * np.eye(M)
        v = np.random.default_rng(2029).standard_normal(M)
        np.testing.assert_allclose(
            system.multiply(v), expected @ v, rtol=1e-10, atol=1e-12
        )


class BarrierParameterHintTest(unittest.TestCase):
    """`DirectConstraintLayer.barrier_parameter_hint` and MCVS wiring."""

    @parameterized.expand(
        [
            (NormConstraintLayer(B=np.eye(3), c=np.zeros(3), phi=2.0),),
        ]
    )
    def test_default_returns_none(self, layer: DirectConstraintLayer) -> None:
        """Layers without a bespoke override fall back to the ``None`` default."""
        x0 = np.array([0.1, 0.2, 0.3])
        assert (
            layer.barrier_parameter_hint(x0, num_ineq=4, outer_tolerance=1e-9) is None
        )

    def test_squared_norm_hint_uses_norm_squared(self) -> None:
        r"""SquaredNorm hint matches ``num_ineq / max(tol, ||x0||^2)``."""
        layer = SquaredNormConstraintLayer(phi=10.0, dimension=3)
        x0 = np.array([1.0, 2.0, 2.0])  # ||x0||^2 = 9
        hint = layer.barrier_parameter_hint(x0, num_ineq=6, outer_tolerance=1e-9)
        assert hint is not None
        np.testing.assert_allclose(hint, 6.0 / 9.0, atol=1e-12)

    def test_squared_norm_hint_floors_with_outer_tolerance(self) -> None:
        """Tiny ``||x0||^2`` is clipped to ``outer_tolerance`` to avoid blow-up."""
        layer = SquaredNormConstraintLayer(phi=10.0, dimension=5)
        x0 = np.zeros(5)  # ||x0||^2 = 0
        outer_tolerance = 1e-6
        num_ineq = 4
        hint = layer.barrier_parameter_hint(
            x0, num_ineq=num_ineq, outer_tolerance=outer_tolerance
        )
        assert hint is not None
        np.testing.assert_allclose(hint, num_ineq / outer_tolerance, atol=1e-12)


class TrivialFeasibilitySolverTest(unittest.TestCase):
    """Tests for the recursion base case ``TrivialFeasibilitySolver``."""

    def test_returns_x0_unchanged(self) -> None:
        solver = TrivialFeasibilitySolver()
        x0 = np.array([1.0, -2.0, 3.5])
        result = solver.solve(x0=x0)
        np.testing.assert_array_equal(result.solution, x0)

    def test_raises_without_x0(self) -> None:
        solver = TrivialFeasibilitySolver()
        with self.assertRaises(ValueError):
            solver.solve()

    def test_num_constraints(self) -> None:
        solver = TrivialFeasibilitySolver()
        assert solver.num_eq_constraints == 0
        assert solver.num_ineq_constraints == 0


class ObjectiveConstraintLayerMathTest(unittest.TestCase):
    """ObjectiveConstraintLayer wraps a convex Objective as ``f(w) <= phi``."""

    @staticmethod
    def _dense_case(M: int = 4, seed: int = 2026) -> tuple[
        ObjectiveConstraintLayer,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((M, M))
        Q = A @ A.T + M * np.eye(M)  # symmetric positive definite (dense) Hessian
        c = rng.standard_normal(M)
        f = QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q))
        w = 0.1 * rng.standard_normal(M)
        phi = float(f.evaluate(w)) + 5.0  # f(w) < phi: strictly interior
        return ObjectiveConstraintLayer(f, phi), w, Q, c, phi

    def test_dimension_and_num_constraints(self) -> None:
        layer, w, _, _, _ = self._dense_case()
        self.assertEqual(layer.dimension, len(w))
        self.assertEqual(layer.num_constraints(len(w)), 1)

    def test_constraints_and_gradient(self) -> None:
        layer, w, Q, c, phi = self._dense_case()
        fval = 0.5 * float(w @ Q @ w) + float(c @ w)
        np.testing.assert_allclose(layer.constraints(w), [fval - phi], rtol=1e-10)
        np.testing.assert_allclose(
            layer.grad_constraints(w), (Q @ w + c).reshape(1, -1), rtol=1e-10
        )
        # Inherited Objective face (used by MCVS Phase I).
        self.assertAlmostEqual(layer.evaluate(w), fval - phi)
        np.testing.assert_allclose(layer.gradient(w), Q @ w + c, rtol=1e-10)

    def test_constraints_shift(self) -> None:
        layer, w, _, _, _ = self._dense_case()
        base = float(layer.constraints(w)[0])
        np.testing.assert_allclose(layer.constraints(w, shift=0.5), [base - 0.5])

    def test_hessian_contribution_matches_dense(self) -> None:
        layer, w, Q, c, _ = self._dense_case()
        f_c = float(layer.constraints(w)[0])
        self.assertLess(f_c, 0.0)
        lam = -1.0 / f_c  # barrier weight
        system = layer.hessian_contribution(w, np.array([lam]))
        grad = Q @ w + c
        H = lam * Q - (lam / f_c) * np.outer(grad, grad)
        rhs = np.array([1.0, -2.0, 0.5, 3.0])
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    def test_hessian_contribution_composes_with_background(self) -> None:
        # Diagonal objective so the background-plus-layer sum stays clean; checks
        # the layer composes into a larger barrier Hessian via MatrixSystem.plus.
        q = np.array([2.0, 3.0, 1.5, 4.0])
        c = np.array([0.3, -0.2, 0.1, 0.4])
        f = QuadraticObjective(c=c, hessian_contribution=DiagonalSystem(q))
        w = np.array([0.05, -0.1, 0.2, 0.0])
        phi = float(f.evaluate(w)) + 3.0
        layer = ObjectiveConstraintLayer(f, phi)
        f_c = float(layer.constraints(w)[0])
        lam = -1.0 / f_c
        eta = np.array([0.7, 1.3, 0.5, 2.1])
        system = DiagonalSystem(eta).plus(
            layer.hessian_contribution(w, np.array([lam]))
        )
        grad = q * w + c
        H = np.diag(eta) + lam * np.diag(q) - (lam / f_c) * np.outer(grad, grad)
        rhs = np.array([1.0, -1.0, 2.0, 0.5])
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )

    def test_hessian_curvature_matches_lambda_hessian(self) -> None:
        layer, w, Q, _, _ = self._dense_case()
        lam = 1.7
        system = layer.hessian_curvature(w, np.array([lam]))
        rhs = np.array([1.0, -1.0, 2.0, 0.5])
        np.testing.assert_allclose(system.multiply(rhs), lam * (Q @ rhs), rtol=1e-10)

    def test_hessian_contribution_requires_lmbda(self) -> None:
        layer, w, _, _, _ = self._dense_case()
        with self.assertRaises(ValueError):
            layer.hessian_contribution(w)

    def test_hessian_contribution_out_of_domain(self) -> None:
        layer, w, _, _, _ = self._dense_case()
        w_far = w + 10.0  # SPD Hessian -> f grows well past phi
        self.assertGreaterEqual(float(layer.constraints(w_far)[0]), 0.0)
        with self.assertRaises(ValueError):
            layer.hessian_contribution(w_far, np.array([1.0]))
