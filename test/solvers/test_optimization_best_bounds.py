"""``InteriorPointMethodSolver`` reports the best (tightest valid) duality-gap
certificate, not the last transient per-step estimate.

A feasible point's objective is a valid upper bound on p* and every barrier
multiplier is >= 0 (its dual is a valid lower bound), so the tightest certificate is
``min(objective) - max(dual)`` across centering steps. Barrier iterates satisfy
A x = b only approximately, so the primal bound comes from a repaired, verified
point, and a dual that exceeds it is discarded (``EqualityDriftTest``,
``WeakDualityTest``).
The per-step gap is non-monotone: at high ``t`` the conditioning degrades and the
dual sags even while the primal holds, so the *last* raw gap is not a trustworthy
convergence signal. ``solve`` must therefore record the running-best certificate in
``duality_gaps`` -- monotone non-increasing, ending at the reported
``suboptimality`` -- and never let a sagged final dual become the reported bound.
"""

import unittest
from typing import Literal
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.problems.result import SolveResult, SolveStatus
from cvxium.solvers.barrier_solver import (
    _BarrierAdapter,
)
from cvxium.solvers.optimization import (
    _CenteringOutcome,
)
from cvxium.solvers.settings import (
    OptimizationSettings,
    OuterSettings,
)


def _result(
    obj: float,
    dual: float,
    marker: float,
    solution: npt.NDArray[np.float64] | None = None,
    status: Literal[0, 1, 2] = 0,
) -> _CenteringOutcome:
    """A centering result with a marker so the test can tell steps apart."""
    return _CenteringOutcome(
        solution=np.full(3, marker) if solution is None else solution,
        objective_value=obj,
        barrier_objective_value=obj,
        dual_value=dual,
        equality_multipliers=np.zeros(0),
        inequality_multipliers=np.full(3, marker),
        suboptimalities=[obj - dual],
        nits=2,
        status=status,
        message="completed",
    )


def _pinned_t0(x0: npt.NDArray[np.float64]) -> float:
    """Pin the barrier init so num_steps is comfortably > 2."""
    return 1.0


class OptimizationBestBoundsTest(unittest.TestCase):
    def test_duality_gaps_track_running_best_not_raw_per_step(self) -> None:
        r"""The gap trace is the running-best certificate, not the raw per-step gap.

        A scripted centering sequence makes the dual sag hard on the final (high-``t``)
        step while the primal is unchanged. A raw trace would *spike* there; the
        running-best trace must stay monotone non-increasing, end at the reported
        ``suboptimality``, and keep the earlier tight dual as ``dual_value``.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        # tolerance below the achievable gap so the loop runs every step (no early
        # stop) and reaches the high-t step where the dual sags.
        settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-9))
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if last_step:
                # Dual sags hard at high t while the primal is unchanged.
                return _result(5.0, -100.0, 9.0)
            if outer_nit == 1:
                return _result(10.0, 2.0, 1.0)
            # Best primal (obj 5.0) certified tightly (gap 1e-4 < tolerance_soft).
            return _result(5.0, 4.9999, 2.0)

        # Pin the barrier init so num_steps is comfortably > 2.
        with (
            patch.object(adapter, "initialize_barrier_parameter", new=_pinned_t0),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        gaps = result.barrier_diagnostics.duality_gaps
        assert len(gaps) >= 3  # ran past the first two scripted steps to the sag
        assert all(gaps[i + 1] <= gaps[i] + 1e-9 for i in range(len(gaps) - 1)), gaps
        np.testing.assert_allclose(
            gaps[-1], result.suboptimality, rtol=1e-6, atol=1e-12
        )
        # The sagged final dual must not become the reported certificate.
        np.testing.assert_allclose(result.dual_value, 4.9999, rtol=1e-6, atol=1e-12)

    def test_a_nan_dual_is_no_certificate(self) -> None:
        """A NaN dual must not read as a zero duality gap.

        Comparisons against NaN are all False, so a NaN first dual used to become
        the incumbent for good, and the gap `max(0, min(nan, ...))` read as 0.0:
        the solve stopped after one centering step, reporting a NaN dual.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-3))
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if outer_nit == 1:
                return _result(10.0, float("nan"), 1.0)
            return _result(5.0, 4.9999, 2.0)

        with (
            patch.object(adapter, "initialize_barrier_parameter", new=_pinned_t0),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        gaps = result.barrier_diagnostics.duality_gaps
        self.assertGreater(gaps[0], 0.0)
        self.assertGreaterEqual(len(gaps), 2)
        np.testing.assert_allclose(result.dual_value, 4.9999, rtol=1e-6, atol=1e-12)


class EqualityDriftTest(unittest.TestCase):
    r"""A step that drifted off A x = b must not certify a wrong answer.

    The barrier keeps iterates strictly inside the inequalities, but A x = b holds
    only to the accuracy of the Newton solves. A centering step that stalls can drift
    far enough that its objective falls below p*. Taken at face value it becomes the
    best primal, ``max(0, best_primal - best_dual)`` clamps to zero, and the solve
    reports convergence with an equality-infeasible solution (Cvxium issue #21).

    Each test scripts two honest steps -- the second certified to a gap of 1e-4 --
    then a stalled step off the affine set whose raw objective is below p*, then
    honest steps again.
    """

    @staticmethod
    def _solve(
        problem: Problem,
        honest: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        p_star: float,
        drifted: npt.NDArray[np.float64],
    ) -> SolveResult:
        # tolerance below the scripted gap, so the loop runs past the drifted step
        # rather than stopping early on the honest certificate.
        settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-9))
        adapter = _BarrierAdapter(problem=problem, settings=settings)
        f = adapter.evaluate_objective
        first, best = honest
        assert f(drifted) < p_star  # the trap: below every valid dual

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if outer_nit == 1:
                return _result(f(first), p_star - 0.1, 1.0, solution=first)
            if outer_nit == 3:
                return _result(
                    f(drifted), float("-inf"), 3.0, solution=drifted, status=1
                )
            return _result(f(best), p_star - 1e-4, 2.0, solution=best)

        with (
            patch.object(adapter, "initialize_barrier_parameter", new=_pinned_t0),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            return adapter.solve(x0=best)

    def test_a_drifted_step_is_repaired_rather_than_trusted(self) -> None:
        r"""Projected back onto A x = b, the drifted step bounds p* honestly.

        minimize (1/2) ‖x‖^2 - 1^T x subject to 1^T x = 1.5, x >= 0, with
        x* = (0.5, 0.5, 0.5) and p* = -1.125.
        """
        A, b = np.ones((1, 3)), np.array([1.5])
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        honest = (np.array([0.6, 0.45, 0.45]), np.full(3, 0.5))
        result = self._solve(problem, honest, -1.125, np.full(3, 0.7))

        np.testing.assert_allclose(A @ result.solution, b, atol=1e-12)
        self.assertGreaterEqual(result.objective_value, -1.125 - 1e-12)
        # The certificate is the honest 1e-4, not a clamped zero.
        np.testing.assert_allclose(result.suboptimality, 1e-4, rtol=1e-6)
        self.assertIs(result.status, SolveStatus.ACCEPTABLE)

    def test_a_step_that_cannot_be_repaired_offers_no_bound(self) -> None:
        """When projection would leave the interior, the step is not a bound.

        minimize x_1 subject to x_1 + x_2 = 1, x >= 0, so p* = 0. Projecting the
        drifted (0.0001, 1.3, 1) removes 0.15005 from x_1 and x_2, driving x_1
        negative.
        """
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            eq=EqualityConstraintLayer(A=np.array([[1.0, 1.0, 0.0]]), b=np.ones(1)),
        )
        best = np.array([1e-3 + 1e-4, 1.0 - 1e-3 - 1e-4, 1.0])
        honest = (np.array([0.1, 0.9, 1.0]), best)
        # p* is 0, but the scripted honest duals sit just below the best step.
        result = self._solve(problem, honest, 1e-3, np.array([1e-4, 1.3, 1.0]))

        np.testing.assert_allclose(result.solution, best, atol=1e-12)
        np.testing.assert_allclose(result.suboptimality, 2e-4, rtol=1e-6)


class WeakDualityTest(unittest.TestCase):
    def test_a_dual_above_a_feasible_objective_is_discarded(self) -> None:
        """A dual that a later primal exposes as invalid never certifies the gap.

        Weak duality puts every valid dual at or below every feasible objective. A
        dual of 6.0 is ruled out once a feasible point reaches 5.0; kept, it would
        clamp the gap to zero and stop the solve as converged.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-9))
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if outer_nit == 1:
                return _result(10.0, 6.0, 1.0)
            return _result(5.0, 4.9999, 2.0)

        with (
            patch.object(adapter, "initialize_barrier_parameter", new=_pinned_t0),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        gaps = result.barrier_diagnostics.duality_gaps
        self.assertTrue(all(g > 0.0 for g in gaps), gaps)
        np.testing.assert_allclose(result.dual_value, 4.9999, rtol=1e-9)
        np.testing.assert_allclose(result.suboptimality, 1e-4, rtol=1e-6)
