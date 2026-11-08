"""Tests for the LineSearch / MeritFunction primitives."""

import unittest
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from cvxium.exceptions import (
    ConstraintBoundaryError,
    InvalidDescentDirectionError,
    SevereCurvatureError,
)
from cvxium.solvers.line_search import (
    ArmijoBarrierMerit,
    LineSearch,
    MeritFunction,
    ResidualNormMerit,
    mehrotra_step_cap,
)


def _constant_finder(
    alpha: float,
) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]:
    """Max-step finder returning a fixed value; ignores `x` and `delta_x`."""

    def finder(
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
    ) -> float:
        del x, delta_x
        return alpha

    return finder


class _ScalarQuadraticMerit(MeritFunction):
    r"""Merit `m(x) = 0.5 (x - x*)^T (x - x*)` reduced along `delta_x`.

    Acceptance threshold mirrors `Armijo` with `alpha = 0.01`. The base
    point's merit value is captured at construction so the threshold is
    deterministic.

    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
        alpha: float = 0.01,
    ) -> None:
        self._x = x
        self._delta_x = delta_x
        self._x_star = x_star
        self._alpha = alpha
        self._m0 = 0.5 * float(np.dot(x - x_star, x - x_star))
        self._grad_dot = float(np.dot(x - x_star, delta_x))
        self._neg_alpha_grad = -alpha * self._grad_dot

    def evaluate(self, x_trial: npt.NDArray[np.float64], s: float) -> float:
        del s
        diff = x_trial - self._x_star
        return 0.5 * float(np.dot(diff, diff))

    def accept_threshold(self, s: float) -> float:
        return self._m0 - s * self._neg_alpha_grad

    def directional_derivative_sign_check(self) -> None:
        if self._grad_dot > 0:
            raise InvalidDescentDirectionError(
                message="Not a descent direction.",
                grad_ft_dot_delta_x=self._grad_dot,
            )

    def required_improvement(self, s: float) -> float:
        return s * self._neg_alpha_grad

    def actual_improvement(self, trial_value: float) -> float:
        return self._m0 - trial_value


class LineSearchHappyPathTest(unittest.TestCase):
    """First-trial-accepted and standard contraction behaviour."""

    def test_first_trial_accepted_returns_step_cap(self) -> None:
        x = np.asarray([2.0, 2.0], dtype=np.float64)
        x_star = np.zeros(2, dtype=np.float64)
        delta_x = x_star - x  # pure descent to the minimum
        merit = _ScalarQuadraticMerit(x=x, delta_x=delta_x, x_star=x_star)
        search = LineSearch(
            merit=merit,
            max_step_finder=_constant_finder(np.inf),
            beta=0.5,
            min_step=1e-3,
        )
        s = search.find_step(x=x, delta_x=delta_x, step_cap=1.0)
        # The first trial s=1 lands exactly at the minimum, so the merit
        # drops from 4.0 to 0.0 -- well below the Armijo threshold.
        self.assertEqual(s, 1.0)

    def test_step_cap_caps_first_trial(self) -> None:
        x = np.asarray([2.0, 2.0], dtype=np.float64)
        x_star = np.zeros(2, dtype=np.float64)
        delta_x = x_star - x
        merit = _ScalarQuadraticMerit(x=x, delta_x=delta_x, x_star=x_star)
        search = LineSearch(
            merit=merit,
            max_step_finder=_constant_finder(np.inf),
            beta=0.5,
            min_step=1e-3,
        )
        # step_cap below 1.0 should pin the returned step even when a
        # larger step is accepted by the merit.
        s = search.find_step(x=x, delta_x=delta_x, step_cap=0.25)
        self.assertEqual(s, 0.25)

    def test_feasibility_cap_dominates_step_cap(self) -> None:
        x = np.asarray([2.0, 2.0], dtype=np.float64)
        x_star = np.zeros(2, dtype=np.float64)
        delta_x = x_star - x
        merit = _ScalarQuadraticMerit(x=x, delta_x=delta_x, x_star=x_star)
        # max-step finder returns alpha=0.4 so beta*alpha = 0.2 caps the
        # nominal step_cap=1.0.
        search = LineSearch(
            merit=merit,
            max_step_finder=_constant_finder(0.4),
            beta=0.5,
            min_step=1e-3,
        )
        s = search.find_step(x=x, delta_x=delta_x, step_cap=1.0)
        self.assertAlmostEqual(s, 0.2)


class LineSearchSignCheckTest(unittest.TestCase):
    """The descent-sign guard runs before any merit evaluations."""

    def test_ascent_direction_raises(self) -> None:
        x = np.asarray([2.0, 2.0], dtype=np.float64)
        x_star = np.zeros(2, dtype=np.float64)
        # `delta_x = +grad` points uphill from `x`.
        delta_x = x - x_star
        merit = _ScalarQuadraticMerit(x=x, delta_x=delta_x, x_star=x_star)
        search = LineSearch(
            merit=merit,
            max_step_finder=_constant_finder(np.inf),
            beta=0.5,
            min_step=1e-3,
        )
        with self.assertRaises(InvalidDescentDirectionError):
            search.find_step(x=x, delta_x=delta_x, step_cap=1.0)


class LineSearchBoundaryErrorTest(unittest.TestCase):
    """ConstraintBoundaryError fires when feasibility cap is below min_step."""

    def test_max_step_below_min_step_raises(self) -> None:
        x = np.asarray([2.0, 2.0], dtype=np.float64)
        x_star = np.zeros(2, dtype=np.float64)
        delta_x = x_star - x
        merit = _ScalarQuadraticMerit(x=x, delta_x=delta_x, x_star=x_star)
        # `beta * max_step = 0.5 * 1e-5 = 5e-6 < min_step = 1e-3`.
        search = LineSearch(
            merit=merit,
            max_step_finder=_constant_finder(1e-5),
            beta=0.5,
            min_step=1e-3,
        )
        with self.assertRaises(ConstraintBoundaryError):
            search.find_step(x=x, delta_x=delta_x, step_cap=1.0)


class LineSearchSevereCurvatureTest(unittest.TestCase):
    """SevereCurvatureError fires when Armijo rejects all the way down to min_step."""

    def test_armijo_failure_below_min_step_raises(self) -> None:
        class _AlwaysReject(MeritFunction):
            """Merit that never accepts any positive step."""

            def evaluate(self, x_trial: npt.NDArray[np.float64], s: float) -> float:
                del x_trial, s
                return 1.0  # always above threshold

            def accept_threshold(self, s: float) -> float:
                return -1.0  # never accept

            def directional_derivative_sign_check(self) -> None:
                return None

            def required_improvement(self, s: float) -> float:
                return 1.0

            def actual_improvement(self, trial_value: float) -> float:
                return -2.0

        x = np.zeros(2, dtype=np.float64)
        delta_x = np.ones(2, dtype=np.float64)
        search = LineSearch(
            merit=_AlwaysReject(),
            max_step_finder=_constant_finder(np.inf),
            beta=0.5,
            min_step=1e-3,
        )
        with self.assertRaises(SevereCurvatureError) as ctx:
            search.find_step(x=x, delta_x=delta_x, step_cap=1.0)
        # The diagnostic carries the merit's `required_improvement` /
        # `actual_improvement` for the trial value at the rejection step.
        self.assertEqual(ctx.exception.required_improvement, 1.0)
        self.assertEqual(ctx.exception.actual_improvement, -2.0)


class ArmijoBarrierMeritTest(unittest.TestCase):
    """Acceptance threshold and diagnostics match the pre-refactor formula."""

    def test_accept_threshold_formula(self) -> None:
        # ft0 = 10.0, alpha * lambda^2 = 0.5; the legacy Armijo check was
        # `ft_new + s * neg_alpha_grad > ft0`, equivalent to
        # `ft_new > ft0 - s * neg_alpha_grad`.
        merit = ArmijoBarrierMerit(
            evaluate_barrier_objective=lambda x: 0.0,
            ft0=10.0,
            neg_alpha_grad=0.5,
            grad_ft_dot_delta_x=-1.0,
        )
        self.assertEqual(merit.accept_threshold(s=0.0), 10.0)
        self.assertEqual(merit.accept_threshold(s=1.0), 9.5)
        self.assertEqual(merit.accept_threshold(s=0.25), 9.875)

    def test_diagnostics(self) -> None:
        merit = ArmijoBarrierMerit(
            evaluate_barrier_objective=lambda x: 0.0,
            ft0=10.0,
            neg_alpha_grad=0.5,
            grad_ft_dot_delta_x=-1.0,
        )
        self.assertEqual(merit.required_improvement(s=0.5), 0.25)
        self.assertEqual(merit.actual_improvement(trial_value=7.0), 3.0)

    def test_descent_check_raises_on_positive_inner_product(self) -> None:
        merit = ArmijoBarrierMerit(
            evaluate_barrier_objective=lambda x: 0.0,
            ft0=10.0,
            neg_alpha_grad=0.5,
            grad_ft_dot_delta_x=0.3,
        )
        with self.assertRaises(InvalidDescentDirectionError) as ctx:
            merit.directional_derivative_sign_check()
        self.assertEqual(ctx.exception.grad_ft_dot_delta_x, 0.3)


class TestMehrotraStepCap(unittest.TestCase):
    """Mehrotra fraction-to-boundary cap: closed-form dual-positivity step."""

    def test_no_negative_delta_returns_gamma_times_one(self) -> None:
        # All positive delta_lmbda -> s_max_lambda = 1.0. mu = 10/6 ~ 1.67
        # gives gamma = max(0.99, 1 - 1.67) = 0.99.
        lmbda = np.ones(6, dtype=np.float64)
        delta_lmbda = np.ones(6, dtype=np.float64)
        s = mehrotra_step_cap(lmbda, delta_lmbda, eta=10.0, tau_min=0.99)
        self.assertAlmostEqual(s, 0.99)

    def test_floor_outside_tail(self) -> None:
        # Outside the tail (mu >= 1 - tau_min) the cap = tau_min * s_max_lambda.
        lmbda = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        delta_lmbda = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        s = mehrotra_step_cap(lmbda, delta_lmbda, eta=6.0, tau_min=0.99)
        self.assertAlmostEqual(s, 0.99)

    def test_tail_relaxes_toward_s_max(self) -> None:
        # In the tail (mu -> 0), gamma -> 1.0 so s -> s_max_lambda.
        lmbda = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        delta_lmbda = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        s = mehrotra_step_cap(lmbda, delta_lmbda, eta=1e-9, tau_min=0.99)
        self.assertGreater(s, 0.999)

    def test_s_max_lambda_dominates_when_smaller_than_gamma(self) -> None:
        # delta_lmbda = -2 for the smallest lmbda = 1 -> s_max_lambda = 0.5.
        lmbda = np.array([1.0, 10.0], dtype=np.float64)
        delta_lmbda = np.array([-2.0, -1.0], dtype=np.float64)
        s = mehrotra_step_cap(lmbda, delta_lmbda, eta=1e-9, tau_min=0.99)
        self.assertAlmostEqual(s, 0.5, places=6)

    def test_tau_min_knob_controls_floor(self) -> None:
        # mu large -> gamma = max(0.5, 1 - large) = 0.5.
        lmbda = np.ones(3, dtype=np.float64)
        delta_lmbda = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        s = mehrotra_step_cap(lmbda, delta_lmbda, eta=100.0, tau_min=0.5)
        self.assertAlmostEqual(s, 0.5)


class TestResidualNormMerit(unittest.TestCase):
    """Acceptance threshold and diagnostics for the PD residual-norm merit."""

    def test_evaluate_delegates_to_closure(self) -> None:
        captured: list[float] = []

        def compute(s: float) -> float:
            captured.append(s)
            return 0.42

        merit = ResidualNormMerit(
            compute_trial_residual_norm=compute, r_norm0=1.0, alpha=0.01
        )
        x = np.zeros(2, dtype=np.float64)
        self.assertEqual(merit.evaluate(x, s=0.5), 0.42)
        self.assertEqual(captured, [0.5])

    def test_accept_threshold_formula(self) -> None:
        merit = ResidualNormMerit(
            compute_trial_residual_norm=lambda s: 0.0, r_norm0=10.0, alpha=0.01
        )
        self.assertEqual(merit.accept_threshold(s=0.0), 10.0)
        self.assertAlmostEqual(merit.accept_threshold(s=1.0), 9.9)
        self.assertAlmostEqual(merit.accept_threshold(s=0.5), 9.95)

    def test_diagnostics(self) -> None:
        merit = ResidualNormMerit(
            compute_trial_residual_norm=lambda s: 0.0, r_norm0=10.0, alpha=0.01
        )
        self.assertAlmostEqual(merit.required_improvement(s=0.5), 0.05)
        self.assertAlmostEqual(merit.actual_improvement(trial_value=7.0), 3.0)

    def test_descent_check_is_noop(self) -> None:
        # PD's descent for the residual norm is structurally guaranteed by the
        # Newton system; the merit's sign check must not raise.
        merit = ResidualNormMerit(
            compute_trial_residual_norm=lambda s: 0.0, r_norm0=1.0, alpha=0.01
        )
        merit.directional_derivative_sign_check()
