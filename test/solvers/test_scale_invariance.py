"""Scale invariance of the barrier and primal-dual solvers (issue #48)."""

import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
from parameterized import parameterized
from scipy.optimize import linprog

from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import LinearObjective
from cvxium.problems.problem import Problem
from cvxium.solvers.barrier_solver import BarrierSolver
from cvxium.solvers.optimization import ProblemMarginallyFeasibleError
from cvxium.solvers.primal_dual_solver import PrimalDualSolver
from cvxium.solvers.settings import OptimizationSettings, OuterSettings
from cvxium.solvers.solver import EqualitySolver, Solver


def _scaled_lp(
    seed: int, c_scale: float = 1.0, data_scale: float = 1.0
) -> tuple[Problem, npt.NDArray[np.float64], float]:
    """An LP with three free variables, scaled in its objective or its constraints.

    Minimize (c_scale c)^T x subject to (s G) x <= s h, (s A) x = s b and x_j >= 0
    off the first three coordinates, with s = ``data_scale``. Returns the problem, a
    strictly feasible x0 and the HiGHS optimum.
    """
    rng = np.random.default_rng(seed)
    n, m, n_free = 10, 12, 3
    G = rng.standard_normal((m, n))
    x0 = rng.uniform(0.5, 1.5, n)
    h = G @ x0 + rng.uniform(0.5, 1.5, m)
    c = -G.T @ rng.uniform(0.1, 1.0, m) + np.r_[np.zeros(n_free), np.ones(n - n_free)]
    lb = np.r_[np.full(n_free, -np.inf), np.zeros(n - n_free)]
    A = rng.standard_normal((2, n))
    problem = Problem(
        obj=LinearObjective(c=c_scale * c),
        ineq=[
            BoundsConstraintLayer(lb=lb),
            LinearInequalityConstraintLayer(A=data_scale * G, cu=data_scale * h),
        ],
        eq=EqualityConstraintLayer(A=data_scale * A, b=data_scale * (A @ x0)),
    )
    bounds = [(None, None)] * n_free + [(0.0, None)] * (n - n_free)
    reference = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=A @ x0, bounds=bounds)
    return problem, x0, c_scale * float(reference.fun)


_SOLVERS = (("barrier", BarrierSolver), ("primal_dual", PrimalDualSolver))


def _solver(name: str, objective_scale: Any = None) -> Solver:
    settings = OptimizationSettings(
        outer=OuterSettings(objective_scale=objective_scale)
    )
    return BarrierSolver(settings) if name == "barrier" else PrimalDualSolver(settings)


class LargeEqualityDataTest(unittest.TestCase):
    """Equality data of size 1e6 no longer sends Phase I into endless recursion.

    The Phase I cascade accepts a start only on the equality manifold, which it
    tested as |A x - b| < 1e-10 absolutely. Near |b| = 1e6 adjacent doubles are
    1e-10 apart, so no start passed, and every level transformed its Phase I problem
    again, adding a slack each time, until ``RecursionError``. The test is now
    relative to each row's scale (``EqualityConstraintLayer.is_satisfied``).
    """

    @parameterized.expand(
        [
            (f"{name}_seed_{seed}_{'warm' if warm else 'cold'}", name, seed, warm)
            for name, _ in _SOLVERS
            for seed in (1, 4, 7)
            for warm in (True, False)
        ]
    )
    def test_solves(self, _name: str, solver: str, seed: int, warm: bool) -> None:
        problem, x0, optimum = _scaled_lp(seed, data_scale=1e6)

        result = _solver(solver).solve(problem, x0 if warm else None)

        self.assertTrue(result.succeeded, result.message)
        self.assertAlmostEqual(
            result.objective_value / optimum, 1.0, delta=1e-4, msg=result.message
        )

    def test_a_projected_point_is_on_the_manifold_at_any_scale(self) -> None:
        rng = np.random.default_rng(0)
        for scale in (1.0, 1e6, 1e12):
            with self.subTest(scale=scale):
                A = scale * rng.standard_normal((2, 8))
                eq = EqualityConstraintLayer(A=A, b=A @ rng.standard_normal(8))
                w = np.asarray(
                    EqualitySolver(constraint=eq).solve(x0=np.zeros(8)).solution
                )
                self.assertTrue(eq.is_satisfied(w))
                self.assertFalse(eq.is_satisfied(w + 1e-6 * np.abs(w).max()))

    def test_a_rejected_slack_form_start_stops_instead_of_recursing(self) -> None:
        """Should rounding still defeat the start's test, Phase I stops and says so."""
        problem, _, _ = _scaled_lp(1)
        with (
            patch.object(Solver, "_phase1_start_is_accepted", return_value=False),
            self.assertRaises(ProblemMarginallyFeasibleError),
        ):
            BarrierSolver().solve(problem)


class ObjectiveScaleTest(unittest.TestCase):
    """``OuterSettings.objective_scale`` makes the gap test scale-invariant.

    The default stays absolute: gap ≤ ``tolerance``. Scaling the objective by 1e6
    then asks for twelve significant digits, beyond double precision, and the
    barrier failed outright; scaling it by 1e-6 made the test pass at once. With
    ``"relative"`` the test is gap ≤ tolerance max(1, |f0|); with ``"auto"`` it is
    relative to the size of the objective's terms, invariant in both directions.

    The primal-dual cases at c x 1e6 sit at the edge of double precision: a relative
    tolerance of 1e-6 on gradients of size 1e6, reached by stalling toward the soft
    tolerance over a hundred or more iterations. Their iteration counts are
    deterministic on a machine but not across machines (102 here, 189 on CI for one
    seed), so they are listed under the perf baseline's ``__ignore__``; their
    assertions still run.
    """

    @parameterized.expand(
        [
            (f"{name}_{c_scale:g}_seed_{seed}", name, c_scale, seed)
            for name, _ in _SOLVERS
            for c_scale in (1e3, 1e6)
            for seed in (0, 1, 2)
        ]
    )
    def test_relative_solves_a_large_objective(
        self, _name: str, solver: str, c_scale: float, seed: int
    ) -> None:
        problem, x0, optimum = _scaled_lp(seed, c_scale=c_scale)

        result = _solver(solver, "relative").solve(problem, x0)

        self.assertTrue(result.succeeded, result.message)
        self.assertAlmostEqual(result.objective_value / optimum, 1.0, delta=1e-5)

    @parameterized.expand(
        [
            (f"{name}_seed_{seed}", name, seed)
            for name, _ in _SOLVERS
            for seed in (0, 1, 2)
        ]
    )
    def test_auto_solves_a_small_objective(
        self, _name: str, solver: str, seed: int
    ) -> None:
        """At c x 1e-6 the absolute default stops at once, up to 1100% off."""
        problem, x0, optimum = _scaled_lp(seed, c_scale=1e-6)

        result = _solver(solver, "auto").solve(problem, x0)

        self.assertTrue(result.succeeded, result.message)
        self.assertLess(abs(result.objective_value - optimum), 1e-3 * abs(optimum))

    @parameterized.expand([("barrier", "barrier"), ("primal_dual", "primal_dual")])
    def test_a_fixed_scale_scales_the_gap(self, _name: str, solver: str) -> None:
        """A fixed scale S makes the acceptance gap tolerance S."""
        problem, x0, optimum = _scaled_lp(0, c_scale=1e3)

        result = _solver(solver, abs(optimum)).solve(problem, x0)

        self.assertTrue(result.succeeded, result.message)
        self.assertAlmostEqual(result.objective_value / optimum, 1.0, delta=1e-5)

    @parameterized.expand(
        [
            ("zero", 0.0),
            ("negative", -1.0),
            ("inf", np.inf),
            ("nan", np.nan),
            ("word", "absolute"),
            ("bool", True),
        ]
    )
    def test_an_invalid_scale_is_rejected(self, _name: str, scale: Any) -> None:
        with self.assertRaisesRegex(ValueError, "objective_scale"):
            OuterSettings(objective_scale=scale)

    @parameterized.expand(
        [("default", None), ("relative", "relative"), ("auto", "auto")]
    )
    def test_a_valid_scale_is_accepted(self, _name: str, scale: Any) -> None:
        self.assertEqual(OuterSettings(objective_scale=scale).objective_scale, scale)
