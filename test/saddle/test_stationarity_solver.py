"""Tests for the infeasible-start Newton saddle solver ``StationaritySolver``."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from cvxium.diagnostics.trace import ListTrace
from cvxium.exceptions import (
    ConstraintBoundaryError,
    NewtonStepError,
    ProblemInfeasibleError,
)
from cvxium.linalg.systems import (
    CoupledSaddleSystem,
    DenseSystem,
    EqualityCoupledSaddleSystem,
    MatrixSystem,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.quadratic_constraints import (
    NormConstraintLayer,
    SquaredNormConstraintLayer,
)
from cvxium.problems.result import (
    PrimalDualDiagnostics,
    SaddleBarrierDiagnostics,
    SolveStatus,
    TwoBlockSolveResult,
)
from cvxium.saddle.primal_dual_stationarity_solver import (
    PrimalDualStationaritySolver,
    _made_no_progress,
    _resolve_block,
)
from cvxium.saddle.stationarity import (
    QuadraticSaddleObjective,
    StationarityConvergenceError,
    StationarityProblem,
    TwoBlockObjective,
    TwoBlockStationaritySolver,
    two_block_result_message,
)
from cvxium.saddle.stationarity_solver import (
    _DUAL_REGION_DECREMENT_SQ_MAX,
    _INITIAL_BARRIER_PARAMETER,
    StationaritySolver,
    _BlockDual,
)
from cvxium.solvers.settings import (
    OptimizationSettings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cvxium.problems.constraint_layers import (
        ConstraintLayer,
    )


def _random_spd(dimension: int, seed: int) -> npt.NDArray[np.float64]:
    """Return a random ``(dimension, dimension)`` SPD matrix ``A^T A + n I``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dimension, dimension))
    return A.T @ A + dimension * np.eye(dimension)


def _quadratic(
    n: int,
    m: int,
    seed: int,
    coupling_scale: float = 1.0,
    curvature_scale: float = 1.0,
) -> QuadraticSaddleObjective:
    """Build a random ``QuadraticSaddleObjective`` of block sizes ``(n, m)``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(m)
    P = curvature_scale * _random_spd(n, seed=seed + 1)
    R = curvature_scale * _random_spd(m, seed=seed + 2)
    C = coupling_scale * rng.standard_normal((n, m))
    return QuadraticSaddleObjective(a, b, P, C, R)


def _dense_saddle(
    obj: QuadraticSaddleObjective,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Closed-form saddle solving ``[[-P, C], [C^T, R]] [x; y] = [-a; -b]``."""
    n = obj.dimension_x
    m = obj.dimension_y
    zero = np.zeros(n + m)
    # Recover P, R, C, a, b via the objective's own gradients/curvatures.
    P = obj.neg_hessian_xx(zero[:n], zero[n:]).multiply(np.eye(n))
    R = obj.hessian_yy(zero[:n], zero[n:]).multiply(np.eye(m))
    C = obj.mixed_hessian(zero[:n], zero[n:])
    a = obj.gradient_x(np.zeros(n), np.zeros(m))
    b = obj.gradient_y(np.zeros(n), np.zeros(m))
    kkt = np.block([[-P, C], [C.T, R]])
    rhs = np.concatenate([-a, -b])
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n], sol[n:]


def _dense_equality_kkt(
    obj: QuadraticSaddleObjective,
    A_eq: npt.NDArray[np.float64],
    b_eq: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Closed-form equality-bordered stationary point ``(x, y)``.

    Solves the barrier-free KKT of ``min_y max_x zeta`` under ``A_eq x = b_eq``::

       | -P    C    A_eq^T | | x |   | -a   |
       | C^T   R    0      | | y | = | -b   |
       | A_eq  0    0      | | l |   | b_eq |
    """
    n = obj.dimension_x
    m = obj.dimension_y
    p = A_eq.shape[0]
    zx, zy = np.zeros(n), np.zeros(m)
    P = obj.neg_hessian_xx(zx, zy).multiply(np.eye(n))
    R = obj.hessian_yy(zx, zy).multiply(np.eye(m))
    C = obj.mixed_hessian(zx, zy)
    a = obj.gradient_x(zx, zy)
    b = obj.gradient_y(zx, zy)
    kkt = np.block(
        [
            [-P, C, A_eq.T],
            [C.T, R, np.zeros((m, p))],
            [A_eq, np.zeros((p, m)), np.zeros((p, p))],
        ]
    )
    rhs = np.concatenate([-a, -b, b_eq])
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n], sol[n : n + m]


def _central_path_duals(
    domain: Sequence[ConstraintLayer],
    w: npt.NDArray[np.float64],
    t: float,
) -> npt.NDArray[np.float64]:
    r"""Central-path inequality duals ``lambda_i = -1 / (t f_i(w))`` for ``domain``.

    Stacked in layer order to match the dual convention of
    ``primal_dual_residual``/``build_primal_dual_coupled_system``. On this locus
    the primal-dual helpers must reproduce the barrier ``residual``/
    ``build_coupled_system`` at the same ``t``. Empty for an empty domain.
    """
    if not domain:
        return np.zeros(0)
    slices = [-1.0 / (t * layer.constraints(w)) for layer in domain]
    return np.concatenate(slices)


def _final_barrier_parameter(
    result: TwoBlockSolveResult, settings: OptimizationSettings
) -> float:
    """Reconstruct the barrier parameter ``t`` at the final centering.

    The outer loop starts at ``t_0 = max(1.0, min_initial)`` and multiplies by
    ``mu`` after each non-final outer step, breaking before the multiply on the
    converging step. Hence ``t_final = t_0 * mu^(outer_steps - 1)``, where the
    number of outer barrier steps is the length of the barrier diagnostics'
    per-outer-step ``duality_gaps`` trajectory. Takes the actual ``settings`` the
    solver was run with (rather than a fresh default) so a test passing
    non-default ``min_initial_barrier_parameter`` or ``barrier_multiplier`` still
    reconstructs the right ``t_final``.
    """
    outer = settings.outer
    t0 = max(_INITIAL_BARRIER_PARAMETER, outer.min_initial_barrier_parameter)
    outer_steps = len(result.barrier_diagnostics.duality_gaps)
    return t0 * outer.barrier_multiplier ** (outer_steps - 1)


class _NegativeCurvatureObjective(TwoBlockObjective):
    """A two-block objective whose ``neg_hessian_xx`` is (wrongly) NOT PD.

    Returns a negative-definite x-block so the coupled saddle lacks the expected
    inertia; used to exercise ``StationaritySolver._check_inertia``.
    """

    def __init__(self, n: int, m: int) -> None:
        self._n = n
        self._m = m

    @property
    def dimension_x(self) -> int:
        return self._n

    @property
    def dimension_y(self) -> int:
        return self._m

    def evaluate(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        return 0.5 * float(x @ x + y @ y)

    def gradient_x(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.asarray(np.ones(self._n) + x)

    def gradient_y(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.asarray(np.ones(self._m) + y)

    def neg_hessian_xx(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> MatrixSystem:
        # WRONG sign: negative definite, so the inertia check must reject it.
        return DenseSystem(-np.eye(self._n))

    def hessian_yy(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> MatrixSystem:
        return DenseSystem(np.eye(self._m))

    def mixed_hessian(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.zeros((self._n, self._m))


class TwoBlockResultMessageTest(unittest.TestCase):
    """Unit tests for the shared ``two_block_result_message`` formatter."""

    def test_omits_gap_when_none(self) -> None:
        # Backward-compatible default: no gap argument -> no gap clause, and the
        # sentence still ends in a single period.
        msg = two_block_result_message(SolveStatus.OPTIMAL, "engine")
        self.assertEqual(
            msg,
            "Two-block stationarity solver 'engine' converged to the requested "
            "tolerance.",
        )

    def test_appends_gap_on_success(self) -> None:
        # The gap rides inside the terminal period, mirroring the naive barrier's
        # "(duality gap ...)." style, formatted with %.3g.
        msg = two_block_result_message(SolveStatus.OPTIMAL, "engine", 0.14567)
        self.assertEqual(
            msg,
            "Two-block stationarity solver 'engine' converged to the requested "
            "tolerance (saddle gap 0.146).",
        )

    def test_appends_gap_on_iteration_exhaustion(self) -> None:
        # A wide gap on a non-converged (MAX_ITERATIONS) result is exactly the
        # signal we want surfaced, so the clause is not gated on success.
        msg = two_block_result_message(SolveStatus.MAX_ITERATIONS, "engine", 3.2)
        self.assertEqual(
            msg,
            "Two-block stationarity solver 'engine' exhausted its iteration "
            "budget before reaching tolerance (saddle gap 3.2).",
        )

    def test_cause_replaces_the_status_derived_description(self) -> None:
        # MAX_ITERATIONS is the status every short exit shares, so the wording it
        # implies -- a spent iteration budget -- is wrong for the breaks that stop
        # early. When the engine names the real cause that is what gets reported,
        # and the gap clause still rides inside the terminal period.
        cause = "a constraint slack reached its boundary"
        msg = two_block_result_message(
            SolveStatus.MAX_ITERATIONS, "engine", 3.2, cause=cause
        )
        self.assertEqual(
            msg,
            "Two-block stationarity solver 'engine' stopped: a constraint slack "
            "reached its boundary (saddle gap 3.2).",
        )
        self.assertNotIn("budget", msg)


class StationaritySolverTest(unittest.TestCase):
    """End-to-end tests for ``StationaritySolver``."""

    def test_conforms_to_two_block_protocol(self) -> None:
        """``StationaritySolver`` satisfies the ``TwoBlockStationaritySolver`` shape."""
        solver = StationaritySolver()
        # Runtime method-presence check.
        self.assertIsInstance(solver, TwoBlockStationaritySolver)
        # Static conformance: the type checker verifies this assignment, so a
        # signature drift on either side fails the type-checking target.
        engine: TwoBlockStationaritySolver = solver
        self.assertIs(engine, solver)

    def test_unconstrained_quadratic_saddle(self) -> None:
        """Empty domains: the solver hits the exact dense saddle, residual ~ 0."""
        obj = _quadratic(n=5, m=3, seed=7)
        problem = StationarityProblem(objective=obj)
        result = StationaritySolver().solve(problem)
        x_star, y_star = _dense_saddle(obj)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-8)
        np.testing.assert_allclose(result.y, y_star, atol=1e-8)
        # No barrier: residual at any t is the raw joint gradient.
        residual = problem.residual(result.x, result.y, t=1.0)
        self.assertLess(float(np.linalg.norm(residual)), 1e-8)

    def test_box_constrained_saddle_interior(self) -> None:
        """A wide box that does not bind recovers the unconstrained saddle."""
        obj = _quadratic(n=3, m=2, seed=11)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        result = StationaritySolver().solve(problem, x0=x_star, y0=y_star)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)

    def test_box_constrained_saddle_binding(self) -> None:
        """A box that binds: verify barrier stationarity and gap at the final t."""
        obj = _quadratic(n=3, m=2, seed=13)
        x_star, _ = _dense_saddle(obj)
        # Push an upper bound below the unconstrained optimum so it binds.
        ub = x_star - 0.5
        lb = x_star - 5.0
        x0 = x_star - 2.0
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=lb, ub=ub)],
        )
        settings = OptimizationSettings()
        n_barriers = 2 * obj.dimension_x
        result = StationaritySolver(settings=settings).solve(problem, x0=x0)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        # Cross-check: assemble the barrier residual by hand at the returned
        # point and confirm it is small at the reconstructed final t.
        t_final = _final_barrier_parameter(result, settings)
        residual = problem.residual(result.x, result.y, t_final)
        self.assertLess(float(np.linalg.norm(residual)), 1e-5)
        self.assertLessEqual(n_barriers / t_final, settings.outer.tolerance)
        # The optimum should ride against (just inside) the binding bound.
        self.assertTrue(np.all(result.x < ub))

    def test_matrix_game_bilinear(self) -> None:
        """Weak curvature with dominant coupling: the canonical saddle converges."""
        obj = _quadratic(n=4, m=4, seed=17, coupling_scale=3.0, curvature_scale=0.1)
        problem = StationarityProblem(objective=obj)
        result = StationaritySolver().solve(problem)
        x_star, y_star = _dense_saddle(obj)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-8)
        np.testing.assert_allclose(result.y, y_star, atol=1e-8)

    def test_relative_tolerance_accepts_large_objective_sooner(self) -> None:
        """A relative gap tolerance accepts a large-unit objective in fewer steps."""
        obj = _quadratic(n=3, m=2, seed=31, curvature_scale=100.0)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        absolute = StationaritySolver().solve(problem, x0=x_star, y0=y_star)
        rel_settings = OptimizationSettings(
            outer=replace(OptimizationSettings().outer, tolerance_rel=1e-3)
        )
        relative = StationaritySolver(settings=rel_settings).solve(
            problem, x0=x_star, y0=y_star
        )
        self.assertEqual(absolute.status, SolveStatus.OPTIMAL)
        self.assertEqual(relative.status, SolveStatus.OPTIMAL)
        # A large |value| makes the relative threshold looser than the absolute
        # floor, so the relative run stops no later than the absolute one.
        self.assertLessEqual(
            len(relative.barrier_diagnostics.duality_gaps),
            len(absolute.barrier_diagnostics.duality_gaps),
        )
        n_barriers = 2 * (obj.dimension_x + obj.dimension_y)
        t_final = _final_barrier_parameter(relative, rel_settings)
        threshold = 1e-3 * abs(obj.evaluate(relative.x, relative.y))
        self.assertLessEqual(
            n_barriers / t_final, max(rel_settings.outer.tolerance, threshold)
        )

    def test_absolute_floor_terminates_zero_value_game(self) -> None:
        """With optimal value ~0 and a relative tolerance, the floor still stops."""
        rng = np.random.default_rng(37)
        n, m = 3, 3
        # a = b = 0 puts the saddle at the origin with value exactly zero.
        P = 0.1 * _random_spd(n, seed=41)
        R = 0.1 * _random_spd(m, seed=43)
        C = 2.0 * rng.standard_normal((n, m))
        obj = QuadraticSaddleObjective(np.zeros(n), np.zeros(m), P, C, R)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-5.0 * np.ones(n), ub=5.0 * np.ones(n))],
            y_domain=[BoundsConstraintLayer(lb=-5.0 * np.ones(m), ub=5.0 * np.ones(m))],
        )
        rel_settings = OptimizationSettings(
            outer=replace(OptimizationSettings().outer, tolerance_rel=1e-4)
        )
        result = StationaritySolver(settings=rel_settings).solve(
            problem, x0=np.zeros(n), y0=np.zeros(m)
        )
        # A pure relative test (threshold -> 0 as value -> 0) would never
        # terminate; the mandatory absolute floor does.
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertLess(abs(obj.evaluate(result.x, result.y)), 1e-4)
        np.testing.assert_allclose(result.x, np.zeros(n), atol=1e-5)
        np.testing.assert_allclose(result.y, np.zeros(m), atol=1e-5)

    def test_inertia_check_rejects_wrong_sign(self) -> None:
        """A non-PD ``neg_hessian_xx`` must trip the inertia check."""
        problem = StationarityProblem(objective=_NegativeCurvatureObjective(n=2, m=2))
        with self.assertRaises(ValueError) as ctx:
            StationaritySolver().solve(problem)
        self.assertIn("inertia", str(ctx.exception))

    def test_inertia_check_skippable(self) -> None:
        """With ``check_inertia=False`` the guard is skipped; the failure raises.

        The wrong-sign block still cannot be factored, so the coupled Newton solve
        fails. With the inertia ``ValueError`` skipped, the failure surfaces as a
        non-convergence ``StationarityConvergenceError`` -- not the inertia error,
        and not an ``INFEASIBLE`` status (reserved for a genuine infeasibility
        certificate). This proves the once-only inertia ``ValueError`` was skipped
        while the failure stays contained. The best-effort iterate rides on the
        exception with a ``MAX_ITERATIONS`` status; a result is *returned* only
        when the solve reaches the user's tolerance.
        """
        problem = StationarityProblem(objective=_NegativeCurvatureObjective(n=2, m=2))
        with self.assertRaises(StationarityConvergenceError) as ctx:
            StationaritySolver(check_inertia=False).solve(problem)
        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)

    def test_outer_budget_exhaustion_raises(self) -> None:
        """Exhausting the outer barrier budget raises with the best-effort iterate.

        Capping the outer loop at a single barrier step leaves the gap
        ``n_barriers / t`` far above tolerance, so the solve cannot reach the
        user's tolerance and raises rather than returning a status a caller might
        overlook. The attached best-effort result carries ``MAX_ITERATIONS``.
        """
        obj = _quadratic(n=2, m=2, seed=5)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        settings = OptimizationSettings(
            outer=replace(OptimizationSettings().outer, max_outer_iterations=1)
        )
        with self.assertRaises(StationarityConvergenceError) as ctx:
            StationaritySolver(settings=settings).solve(problem, x0=x_star, y0=y_star)
        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)
        self.assertEqual(len(ctx.exception.result.barrier_diagnostics.duality_gaps), 1)
        self.assertGreater(ctx.exception.suboptimality, 0.0)

    def test_phase1_feasibilizes_none_start(self) -> None:
        """A ``None`` start over a non-empty domain is feasibilized by Phase I.

        With a box wide enough to contain the unconstrained saddle, the per-block
        Phase I supplies a strictly-interior start and the solve reaches that
        saddle.
        """
        obj = _quadratic(n=2, m=2, seed=19)
        x_star, y_star = _dense_saddle(obj)
        radius = float(np.max(np.abs(x_star))) + 5.0
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-radius * np.ones(2), ub=radius * np.ones(2))
            ],
        )
        result = StationaritySolver().solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)

    def test_phase1_feasibilizes_infeasible_start(self) -> None:
        """An explicit out-of-domain start is feasibilized by Phase I.

        The start sits outside the box; Phase I seeds from it, recovers a strictly-
        interior point, and the solve converges to a box-respecting iterate.
        """
        obj = _quadratic(n=2, m=2, seed=23)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        result = StationaritySolver().solve(problem, x0=np.array([5.0, 0.0]))
        self.assertIn(result.status, (SolveStatus.OPTIMAL, SolveStatus.ACCEPTABLE))
        np.testing.assert_array_less(-1.0 - 1e-6, result.x)
        np.testing.assert_array_less(result.x, 1.0 + 1e-6)

    def test_phase1_reports_infeasible_domain(self) -> None:
        """An empty domain (box + contradictory equality) raises ProblemInfeasibleError.

        ``sum(x) = 5`` cannot hold inside ``[0, 1]^2``; the x-block Phase I
        certifies infeasibility, re-raised as a ``ProblemInfeasibleError`` naming
        the block.
        """
        obj = _quadratic(n=2, m=2, seed=31)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=np.zeros(2), ub=np.ones(2))],
            x_equality=EqualityConstraintLayer(A=np.ones((1, 2)), b=np.array([5.0])),
        )
        with self.assertRaises(ProblemInfeasibleError) as ctx:
            StationaritySolver().solve(problem)
        self.assertIn("x-block", str(ctx.exception))

    def test_requires_start_matching_dimension(self) -> None:
        """A wrong-shape start raises a clear ``ValueError``, not a numpy error."""
        obj = _quadratic(n=2, m=2, seed=31)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        with self.assertRaises(ValueError) as ctx:
            StationaritySolver().solve(problem, x0=np.zeros(3))
        self.assertIn("shape", str(ctx.exception))

    def test_result_fields(self) -> None:
        """Status, counts, duals, diagnostics, and iterate are populated."""
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
        )
        result = StationaritySolver().solve(problem, x0=np.zeros(3))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertEqual(result.x.shape, (3,))
        self.assertEqual(result.y.shape, (2,))
        self.assertGreater(result.n_iterations, 0)
        diagnostics = result.barrier_diagnostics
        self.assertGreater(len(diagnostics.duality_gaps), 0)
        # inner_nits and inner_suboptimalities agree per outer step (the shape
        # BarrierDiagnostics.plot_newton_convergence indexes).
        self.assertEqual(
            len(diagnostics.inner_nits), len(diagnostics.inner_suboptimalities)
        )
        # The x-block has a BoundsConstraintLayer (six inequalities); its barrier
        # duals are recovered on the result.
        self.assertIsNotNone(result.duals)
        self.assertEqual(result.inequality_multipliers.shape, (6,))

    def test_inequality_duals_mask_boundary_constraint_as_nan(self) -> None:
        """A boundary constraint (``f_i == 0``) reports NaN; interior ones stay positive.

        A best-effort ``MAX_ITERATIONS`` iterate can sit on a constraint
        boundary, where the log-barrier estimate ``-1 / (t f_i)`` blows up to
        ``+-inf``. Those entries must be masked to NaN so callers do not trust a
        spurious multiplier, while every strictly-interior constraint keeps a
        finite, nonnegative dual.
        """
        problem = StationarityProblem(
            objective=_quadratic(n=2, m=1, seed=7),
            x_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        # x[1] sits exactly on its upper bound, so the stacked constraints hold a
        # single ``f == 0`` entry; every other bound is strictly interior.
        duals = StationaritySolver._inequality_duals(
            problem, x=np.array([0.0, 1.0]), y=np.zeros(1), t=1.0
        )
        assert duals is not None
        self.assertEqual(int(np.sum(np.isnan(duals))), 1)
        self.assertTrue(bool(np.all(duals[np.isfinite(duals)] > 0.0)))

    def test_saddle_value_and_bracket_are_surfaced(self) -> None:
        """The result carries the saddle value and a matching value bracket."""
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
        )
        settings = OptimizationSettings()
        result = StationaritySolver(settings=settings).solve(problem, x0=np.zeros(3))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        # The surfaced value is the achieved saddle value zeta(x, y).
        self.assertAlmostEqual(
            result.objective_value, float(obj.evaluate(result.x, result.y)), places=10
        )
        diag = result.saddle_barrier_diagnostics
        self.assertIsInstance(diag, SaddleBarrierDiagnostics)
        # One bracket per outer step, aligned with the gap trajectory.
        self.assertEqual(len(diag.upper_bounds), len(diag.duality_gaps))
        self.assertEqual(len(diag.lower_bounds), len(diag.duality_gaps))
        # The bracket width equals the surrogate gap at every step ...
        for upper, lower, gap in zip(
            diag.upper_bounds, diag.lower_bounds, diag.duality_gaps, strict=False
        ):
            self.assertAlmostEqual(upper - lower, gap, places=12)
            self.assertLess(lower, upper)
        # ... and the reported suboptimality is that final bracket width, with
        # dual_value the final lower bound.
        self.assertAlmostEqual(
            result.suboptimality, diag.upper_bounds[-1] - diag.lower_bounds[-1]
        )
        self.assertAlmostEqual(result.dual_value, diag.lower_bounds[-1], places=12)

    def test_message_reports_the_certified_saddle_gap(self) -> None:
        """The barrier result message quotes its reported saddle-value gap."""
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
        )
        result = StationaritySolver().solve(problem, x0=np.zeros(3))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        # The gap in the message is exactly the surfaced suboptimality (the
        # certified/belt bracket width), formatted the same way the message does.
        self.assertIn(f"(saddle gap {result.suboptimality:.3g})", result.message)

    def test_value_bracket_encloses_the_true_saddle_value(self) -> None:
        """With a wide (loose) bracket the closed form encloses the known value."""
        obj = _quadratic(n=3, m=2, seed=11)
        x_star, y_star = _dense_saddle(obj)
        v_star = float(obj.evaluate(x_star, y_star))
        # A wide, non-binding box: the constrained saddle equals the
        # unconstrained one, so v_star is the true value V.
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        # A loose outer tolerance stops early, leaving a genuinely wide bracket
        # that must still straddle V -- a real test of the bound directions.
        loose = OptimizationSettings(
            outer=replace(OptimizationSettings().outer, tolerance=1e-1)
        )
        result = StationaritySolver(settings=loose).solve(problem, x0=x_star, y0=y_star)
        diag = result.saddle_barrier_diagnostics
        lower, upper = diag.lower_bounds[-1], diag.upper_bounds[-1]
        # The bracket is wide (the loose stop) yet still brackets V.
        self.assertGreater(upper - lower, 1e-3)
        self.assertLessEqual(lower, v_star)
        self.assertGreaterEqual(upper, v_star)

    def test_bracket_offsets_use_the_correct_block_counts(self) -> None:
        """Lower carries the y-block (min) count, upper the x-block (max) count.

        Uses asymmetric block counts (x-box -> 6 inequalities ``m_x``, y-box -> 4
        ``m_y``) so transposing the two offsets changes the numbers: the objective
        solves ``max_x min_y zeta``, so the inner minimization sets
        ``lower = value - m_y / t`` and the outer maximization
        ``upper = value + m_x / t``. A swap would put ``m_x`` on the lower side and
        ``m_y`` on the upper, which this test rejects.
        """
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
            y_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(2), ub=10.0 * np.ones(2))
            ],
        )
        result = StationaritySolver().solve(problem, x0=np.zeros(3), y0=np.zeros(2))
        m_x, m_y = 6, 4
        self.assertNotEqual(m_x, m_y)
        diag = result.saddle_barrier_diagnostics
        value = float(obj.evaluate(result.x, result.y))
        # The reported width is the (swap-invariant) surrogate gap (m_x + m_y) / t,
        # so it recovers the final barrier parameter t without assuming a side.
        t = (m_x + m_y) / result.suboptimality
        self.assertAlmostEqual(diag.upper_bounds[-1], value + m_x / t, places=9)
        self.assertAlmostEqual(diag.lower_bounds[-1], value - m_y / t, places=9)

    def test_unconstrained_saddle_bracket_collapses_to_the_value(self) -> None:
        """No barriers: the bracket is a point at the exact saddle value."""
        obj = _quadratic(n=5, m=3, seed=7)
        problem = StationarityProblem(objective=obj)
        result = StationaritySolver().solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertEqual(result.suboptimality, 0.0)
        diag = result.saddle_barrier_diagnostics
        self.assertEqual(diag.upper_bounds, diag.lower_bounds)
        self.assertAlmostEqual(result.dual_value, result.objective_value, places=12)


class StationarityCertifiedBoundsTest(unittest.TestCase):
    """Tests for the rigorous per-block dual "suspenders" (``_BlockDual``)."""

    def test_block_dual_recovers_exact_quadratic_infimum(self) -> None:
        """An unconstrained quadratic block: ``evaluate`` returns the exact inf.

        ``f0(w) = 1/2 w.w - b.w`` is minimized at ``w* = b`` with value
        ``-1/2 b.b``. The mini-Newton lands there in one step and reports
        ``delta^2 = 0``, so the corrected bound is exact.
        """
        b = np.array([0.3, -0.4, 0.1])
        dual = _BlockDual(
            objective_value=lambda w: 0.5 * float(w @ w) - float(b @ w),
            objective_gradient=lambda w: np.asarray(w - b, dtype=np.float64),
            objective_curvature=lambda w: DenseSystem(np.eye(3)),
            domain=[],
            duals=np.zeros(0),
            settings=OptimizationSettings(),
        )
        self.assertAlmostEqual(
            dual.evaluate(np.zeros(3)), -0.5 * float(b @ b), places=12
        )

    def test_block_dual_defers_when_it_cannot_certify(self) -> None:
        """A singular block Hessian yields no certificate: ``evaluate`` -> -inf."""
        dual = _BlockDual(
            objective_value=lambda w: float(w @ w),
            objective_gradient=lambda w: np.asarray(2.0 * w + 1.0, dtype=np.float64),
            objective_curvature=lambda w: DenseSystem(np.zeros((2, 2))),
            domain=[],
            duals=np.zeros(0),
            settings=OptimizationSettings(),
        )
        self.assertEqual(dual.evaluate(np.zeros(2)), float("-inf"))

    def test_block_dual_defers_on_non_finite_value(self) -> None:
        """A non-finite block value defers rather than certifying a NaN bound.

        The gradient is zero, so the mini-Newton reports ``delta^2 = 0`` (well
        inside the region guard), yet the Lagrangian value is NaN. ``evaluate``
        must still return ``-inf`` -- honoring "never falsely certifies" -- rather
        than pass the NaN through as a certificate.
        """
        dual = _BlockDual(
            objective_value=lambda w: float("nan"),
            objective_gradient=lambda w: np.zeros_like(w),
            objective_curvature=lambda w: DenseSystem(np.eye(2)),
            domain=[],
            duals=np.zeros(0),
            settings=OptimizationSettings(),
        )
        self.assertEqual(dual.evaluate(np.zeros(2)), float("-inf"))

    def test_block_inequality_duals_stay_finite_at_the_boundary(self) -> None:
        """Non-interior constraint values never yield inf/nan or negative duals.

        ``lambda_i = -1/(t f_i)`` is a valid (nonnegative) multiplier only for a
        strictly-interior ``f_i < 0``. On a non-interior iterate the recovery must
        still return finite, nonnegative multipliers so the certificate is never
        poisoned by an inf/nan (or a weak-duality-breaking negative) dual.
        """
        layer = BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))
        # First coordinate sits exactly on the upper bound (f_i == 0); the second
        # lies outside it (f_i > 0) -- both non-interior.
        duals = StationaritySolver._block_inequality_duals(
            [layer], np.array([1.0, 2.0]), 1.0
        )
        self.assertGreater(duals.size, 0)
        self.assertTrue(np.all(np.isfinite(duals)))
        self.assertTrue(np.all(duals >= 0.0))

    def test_certified_bounds_present_and_aligned(self) -> None:
        """Certified brackets are recorded per outer step and are well-formed."""
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
            y_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(2), ub=10.0 * np.ones(2))
            ],
        )
        result = StationaritySolver().solve(problem, x0=np.zeros(3), y0=np.zeros(2))
        diag = result.saddle_barrier_diagnostics
        n = len(diag.duality_gaps)
        self.assertEqual(len(diag.certified_upper_bounds), n)
        self.assertEqual(len(diag.certified_lower_bounds), n)
        # Each inner-centered step certifies (finite) with lower <= upper.
        for lower, upper in zip(
            diag.certified_lower_bounds, diag.certified_upper_bounds, strict=False
        ):
            self.assertTrue(np.isfinite(lower))
            self.assertTrue(np.isfinite(upper))
            self.assertLessEqual(lower, upper)

    def test_certified_bracket_straddles_true_value(self) -> None:
        """Every certified bracket rigorously encloses the known saddle value V."""
        obj = _quadratic(n=3, m=2, seed=11)
        x_star, y_star = _dense_saddle(obj)
        v_star = float(obj.evaluate(x_star, y_star))
        # Wide, non-binding box: the constrained V equals the unconstrained value.
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        result = StationaritySolver().solve(problem, x0=x_star, y0=y_star)
        diag = result.saddle_barrier_diagnostics
        for lower, upper in zip(
            diag.certified_lower_bounds, diag.certified_upper_bounds, strict=False
        ):
            if np.isfinite(lower):
                self.assertLessEqual(lower, v_star + 1e-6)
            if np.isfinite(upper):
                self.assertGreaterEqual(upper, v_star - 1e-6)

    def test_certified_bracket_is_rigorous_off_central_path(self) -> None:
        """A loose stop leaves the belt approximate but the certified bracket exact.

        The suspenders' whole point: even when the outer loop stops early (off the
        central path), the final certified bracket is a *guaranteed* enclosure of
        V, because each block Lagrangian is re-minimized from the centered iterate.
        """
        obj = _quadratic(n=3, m=2, seed=11)
        x_star, y_star = _dense_saddle(obj)
        v_star = float(obj.evaluate(x_star, y_star))
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        loose = OptimizationSettings(
            outer=replace(OptimizationSettings().outer, tolerance=1e-1)
        )
        result = StationaritySolver(settings=loose).solve(problem, x0=x_star, y0=y_star)
        diag = result.saddle_barrier_diagnostics
        lower, upper = diag.certified_lower_bounds[-1], diag.certified_upper_bounds[-1]
        self.assertTrue(np.isfinite(lower))
        self.assertTrue(np.isfinite(upper))
        self.assertLessEqual(lower, v_star + 1e-6)
        self.assertGreaterEqual(upper, v_star - 1e-6)

    def test_result_prefers_certified_certificate(self) -> None:
        """``dual_value``/``suboptimality`` come from the certified bracket (1A)."""
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
        )
        result = StationaritySolver().solve(problem, x0=np.zeros(3))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        diag = result.saddle_barrier_diagnostics
        cert_lower = diag.certified_lower_bounds[-1]
        cert_upper = diag.certified_upper_bounds[-1]
        self.assertTrue(np.isfinite(cert_lower))
        self.assertTrue(np.isfinite(cert_upper))
        # With a finite certificate, the result reports it in place of the belt.
        self.assertAlmostEqual(result.dual_value, cert_lower, places=12)
        self.assertAlmostEqual(result.suboptimality, cert_upper - cert_lower, places=12)

    def test_certified_bounds_with_equality(self) -> None:
        """Equality-only solve: the certified bracket pins the exact KKT value.

        Exercises the no-barrier certified path and the ``nu = -lam`` equality
        term of the x-block upper bound.
        """
        obj = _quadratic(n=5, m=3, seed=101)
        rng = np.random.default_rng(7)
        a_eq = rng.standard_normal((2, 5))
        b_eq = rng.standard_normal(2)
        problem = StationarityProblem(
            objective=obj,
            x_equality=EqualityConstraintLayer(A=a_eq, b=b_eq),
        )
        result = StationaritySolver().solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        x_star, y_star = _dense_equality_kkt(obj, a_eq, b_eq)
        v_star = float(obj.evaluate(x_star, y_star))
        diag = result.saddle_barrier_diagnostics
        cert_lower = diag.certified_lower_bounds[-1]
        cert_upper = diag.certified_upper_bounds[-1]
        self.assertTrue(np.isfinite(cert_lower))
        self.assertTrue(np.isfinite(cert_upper))
        self.assertLessEqual(cert_lower, v_star + 1e-6)
        self.assertGreaterEqual(cert_upper, v_star - 1e-6)
        self.assertAlmostEqual(result.dual_value, cert_lower, places=12)

    def test_block_dual_defers_on_finite_out_of_region_decrement(self) -> None:
        """A finite decrement beyond the self-concordant region defers to -inf.

        The other deferral tests trip the ``delta^2 is None`` (singular solve) and
        non-finite-value guards; this exercises the region check itself -- a
        *finite* ``delta^2 > _DUAL_REGION_DECREMENT_SQ_MAX``. A deliberately
        understated (near-singular but invertible) curvature makes the Newton step
        overshoot, so the mini-Newton cannot pull the decrement back inside its
        budget. The self-concordant ``-delta^2`` correction is only valid in
        region, so ``evaluate`` must defer (-inf) rather than certify a bound from
        outside it.
        """
        eps = 1e-3
        dual = _BlockDual(
            objective_value=lambda w: 0.5 * float(w @ w),
            objective_gradient=lambda w: np.asarray(w, dtype=np.float64),
            objective_curvature=lambda w: DenseSystem(eps * np.eye(1)),
            domain=[],
            duals=np.zeros(0),
            settings=OptimizationSettings(),
        )
        w0 = np.array([1.0])
        # The raw decrement at w0 is finite (not a singular None) but far outside
        # the region: delta^2 = g^T H^{-1} g = 1 / eps.
        delta2 = dual._newton_decrement_sq(w0)
        assert delta2 is not None
        self.assertTrue(math.isfinite(delta2))
        self.assertGreater(delta2, _DUAL_REGION_DECREMENT_SQ_MAX)
        self.assertEqual(dual.evaluate(w0), float("-inf"))

    def test_reported_bracket_never_crosses_on_mixed_fallback(self) -> None:
        """A certified side paired with a deferred side reports the belt width (1A).

        Off the central path a certified lower bound can sit *above* the belt's
        upper bound; subtracting a certified side from a belt side per bound would
        report a negative ``suboptimality``. When either side defers, the width
        must fall back to the always-nonnegative belt ``upper - lower`` while
        ``dual_value`` still reports the best (certified) lower bound available.
        """
        # cert_lower finite and above the belt upper (the crossing case), cert_upper
        # deferred (+inf): a per-side subtraction would give 9.0 - 8.0's cross.
        dual_value, suboptimality = StationaritySolver._reported_bracket(
            9.0, float("inf"), 7.0, 8.0
        )
        self.assertEqual(dual_value, 9.0)
        self.assertAlmostEqual(suboptimality, 1.0)  # belt width 8 - 7, never negative
        self.assertGreaterEqual(suboptimality, 0.0)
        # Deferred lower, certified upper: dual_value falls back to the belt lower.
        dual_value, suboptimality = StationaritySolver._reported_bracket(
            float("-inf"), 8.5, 7.0, 9.0
        )
        self.assertEqual(dual_value, 7.0)
        self.assertAlmostEqual(suboptimality, 2.0)  # belt width 9 - 7
        # Both certified: the tighter certified width is reported, not the belt's.
        dual_value, suboptimality = StationaritySolver._reported_bracket(
            7.5, 8.5, 7.0, 9.0
        )
        self.assertEqual(dual_value, 7.5)
        self.assertAlmostEqual(suboptimality, 1.0)  # 8.5 - 7.5, not the belt's 2.0


class StationaritySolverEqualityTest(unittest.TestCase):
    """End-to-end checks of the x-equality (bordered-elimination) solve path."""

    @staticmethod
    def _equality(
        n: int, p: int, seed: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Random full-row-rank ``A_eq`` of shape ``(p, n)`` and target ``b_eq``."""
        rng = np.random.default_rng(seed)
        A_eq = rng.standard_normal((p, n))
        b_eq = rng.standard_normal(p)
        return A_eq, b_eq

    def test_equality_only_matches_dense_kkt(self) -> None:
        """No inequality domain: the solve must hit the closed-form KKT point."""
        obj = _quadratic(n=5, m=3, seed=101)
        A_eq, b_eq = self._equality(n=5, p=2, seed=7)
        problem = StationarityProblem(
            objective=obj,
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        result = StationaritySolver().solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        x_star, y_star = _dense_equality_kkt(obj, A_eq, b_eq)
        np.testing.assert_allclose(result.x, x_star, atol=1e-7)
        np.testing.assert_allclose(result.y, y_star, atol=1e-7)
        np.testing.assert_allclose(A_eq @ result.x, b_eq, atol=1e-8)

    def test_infeasible_start_reaches_equality(self) -> None:
        """An x0 that violates ``A_eq x = b_eq`` is driven feasible by Newton."""
        obj = _quadratic(n=4, m=2, seed=202)
        A_eq, b_eq = self._equality(n=4, p=1, seed=11)
        problem = StationarityProblem(
            objective=obj,
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        # Deliberately start far from the equality manifold.
        result = StationaritySolver().solve(problem, x0=10.0 * np.ones(4))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        x_star, y_star = _dense_equality_kkt(obj, A_eq, b_eq)
        np.testing.assert_allclose(result.x, x_star, atol=1e-7)
        np.testing.assert_allclose(A_eq @ result.x, b_eq, atol=1e-8)

    def test_equality_with_nonbinding_box_matches_kkt(self) -> None:
        """A slack box leaves the equality-constrained KKT point unchanged."""
        obj = _quadratic(n=4, m=3, seed=303)
        A_eq, b_eq = self._equality(n=4, p=1, seed=13)
        x_star, y_star = _dense_equality_kkt(obj, A_eq, b_eq)
        # A box wide enough that the KKT optimum sits strictly interior.
        radius = np.max(np.abs(x_star)) + 5.0
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-radius * np.ones(4), ub=radius * np.ones(4))
            ],
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        result = StationaritySolver().solve(problem, x0=np.zeros(4))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)
        np.testing.assert_allclose(A_eq @ result.x, b_eq, atol=1e-6)

    def test_budget_equality_holds_with_binding_box(self) -> None:
        """A sum-budget equality is honored even when box bounds are active."""
        obj = _quadratic(n=3, m=2, seed=404)
        # Simplex-style budget: entries sum to 1.
        A_eq = np.ones((1, 3))
        b_eq = np.array([1.0])
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=np.zeros(3), ub=np.ones(3))],
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        result = StationaritySolver().solve(problem, x0=np.full(3, 1.0 / 3.0))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertAlmostEqual(float(np.sum(result.x)), 1.0, places=6)
        # Box bounds respected (strictly interior up to the barrier tolerance).
        self.assertTrue(np.all(result.x >= -1e-6))
        self.assertTrue(np.all(result.x <= 1.0 + 1e-6))


class StationarityPrimalDualHelpersTest(unittest.TestCase):
    """Cross-checks that the primal-dual helpers reproduce the barrier engine.

    ``StationarityProblem.primal_dual_residual`` and
    ``build_primal_dual_coupled_system`` take explicit per-block duals where the
    barrier ``residual``/``build_coupled_system`` take an inverse weight ``t``. On
    the central path ``lambda_i = -1/(t f_i)`` the two families must coincide --
    the property the primal-dual keystone relies on -- and these tests pin it.
    """

    def test_pd_residual_matches_barrier_on_central_path(self) -> None:
        """No equality: the PD residual equals the barrier residual at lambda=-1/(t f)."""
        obj = _quadratic(n=4, m=3, seed=61)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 3.0, ub=x_star + 3.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 3.0, ub=y_star + 3.0)],
        )
        # An interior, non-optimal point so both forms are non-trivially exercised.
        x = x_star + 0.7
        y = y_star - 0.4
        t = 5.0
        lam_x = _central_path_duals(problem.x_domain, x, t)
        lam_y = _central_path_duals(problem.y_domain, y, t)
        pd = problem.primal_dual_residual(x, y, lam_x, lam_y)
        barrier = problem.residual(x, y, t)
        np.testing.assert_allclose(pd, barrier, atol=1e-10)

    def test_pd_system_matches_barrier_on_central_path(self) -> None:
        """No equality: the PD coupled solve equals the barrier solve at lambda=-1/(t f)."""
        obj = _quadratic(n=6, m=2, seed=67)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 3.0, ub=x_star + 3.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 3.0, ub=y_star + 3.0)],
        )
        x = x_star + 0.5
        y = y_star + 0.3
        t = 4.0
        lam_x = _central_path_duals(problem.x_domain, x, t)
        lam_y = _central_path_duals(problem.y_domain, y, t)
        pd_system = problem.build_primal_dual_coupled_system(x, y, lam_x, lam_y)
        barrier_system = problem.build_coupled_system(x, y, t)
        # Both are the plain (non-bordered) saddle; narrow the union to solve(b_x, b_y).
        assert isinstance(pd_system, CoupledSaddleSystem)
        assert isinstance(barrier_system, CoupledSaddleSystem)
        rng = np.random.default_rng(71)
        b_x = rng.standard_normal(obj.dimension_x)
        b_y = rng.standard_normal(obj.dimension_y)
        dx_pd, dy_pd = pd_system.solve(b_x, b_y)
        dx_bar, dy_bar = barrier_system.solve(b_x, b_y)
        np.testing.assert_allclose(dx_pd, dx_bar, atol=1e-9)
        np.testing.assert_allclose(dy_pd, dy_bar, atol=1e-9)

    def test_pd_residual_matches_barrier_with_equality(self) -> None:
        """With an x-equality: the PD residual matches the barrier residual.

        The shared equality multiplier enters both residuals identically (the
        ``A_eq^T lambda`` x-row term and the ``A_eq x - b_eq`` feasibility row), so
        passing it as ``lam_eq`` to one and ``lam`` to the other must agree.
        """
        obj = _quadratic(n=5, m=3, seed=73)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-5.0 * np.ones(5), ub=5.0 * np.ones(5))],
            x_equality=EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([2.0])),
        )
        x = np.full(5, 0.3)
        y = np.array([0.1, -0.2, 0.4])
        t = 6.0
        lam_eq = np.array([0.9])
        lam_x = _central_path_duals(problem.x_domain, x, t)
        lam_y = _central_path_duals(problem.y_domain, y, t)
        pd = problem.primal_dual_residual(x, y, lam_x, lam_y, lam_eq=lam_eq)
        barrier = problem.residual(x, y, t, lam=lam_eq)
        np.testing.assert_allclose(pd, barrier, atol=1e-10)

    def test_pd_system_matches_barrier_with_equality(self) -> None:
        """With an x-equality: the bordered PD solve matches the barrier solve."""
        obj = _quadratic(n=5, m=2, seed=79)
        A_eq = np.array([[1.0, -1.0, 0.5, 0.0, 2.0]])
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-5.0 * np.ones(5), ub=5.0 * np.ones(5))],
            x_equality=EqualityConstraintLayer(A=A_eq, b=np.array([1.0])),
        )
        x = np.full(5, 0.2)
        y = np.array([0.3, -0.1])
        t = 4.0
        lam_x = _central_path_duals(problem.x_domain, x, t)
        lam_y = _central_path_duals(problem.y_domain, y, t)
        pd_system = problem.build_primal_dual_coupled_system(x, y, lam_x, lam_y)
        barrier_system = problem.build_coupled_system(x, y, t)
        # Both are equality-bordered; narrow to solve(b_x, b_y, b_eq).
        assert isinstance(pd_system, EqualityCoupledSaddleSystem)
        assert isinstance(barrier_system, EqualityCoupledSaddleSystem)
        rng = np.random.default_rng(83)
        b_x = rng.standard_normal(obj.dimension_x)
        b_y = rng.standard_normal(obj.dimension_y)
        b_e = rng.standard_normal(1)
        dx_pd, dy_pd, dl_pd = pd_system.solve(b_x, b_y, b_e)
        dx_bar, dy_bar, dl_bar = barrier_system.solve(b_x, b_y, b_e)
        np.testing.assert_allclose(dx_pd, dx_bar, atol=1e-9)
        np.testing.assert_allclose(dy_pd, dy_bar, atol=1e-9)
        np.testing.assert_allclose(dl_pd, dl_bar, atol=1e-9)

    def test_pd_residual_rejects_wrong_dual_lengths(self) -> None:
        """A mismatched inequality-dual length raises a clear ``ValueError``."""
        obj = _quadratic(n=3, m=2, seed=89)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-np.ones(3), ub=np.ones(3))],
        )
        # The x-domain box carries 2*3 = 6 constraints; pass one too few.
        with self.assertRaises(ValueError) as ctx:
            problem.primal_dual_residual(
                np.zeros(3), np.zeros(2), np.ones(5), np.zeros(0)
            )
        self.assertIn("lam_x", str(ctx.exception))

    def test_pd_residual_requires_lam_eq_with_equality(self) -> None:
        """A problem with an x-equality demands ``lam_eq``; omitting it raises."""
        obj = _quadratic(n=3, m=2, seed=91)
        problem = StationarityProblem(
            objective=obj,
            x_equality=EqualityConstraintLayer(A=np.ones((1, 3)), b=np.array([0.0])),
        )
        with self.assertRaises(ValueError) as ctx:
            problem.primal_dual_residual(
                np.zeros(3), np.zeros(2), np.zeros(0), np.zeros(0)
            )
        self.assertIn("lam_eq", str(ctx.exception))

    def test_pd_residual_rejects_lam_eq_without_equality(self) -> None:
        """Supplying ``lam_eq`` when there is no x_equality raises."""
        obj = _quadratic(n=3, m=2, seed=93)
        problem = StationarityProblem(objective=obj)
        with self.assertRaises(ValueError) as ctx:
            problem.primal_dual_residual(
                np.zeros(3),
                np.zeros(2),
                np.zeros(0),
                np.zeros(0),
                lam_eq=np.array([1.0]),
            )
        self.assertIn("lam_eq", str(ctx.exception))

    def test_pd_system_requires_strict_interior(self) -> None:
        """A boundary point trips the strict-interior guard in the PD Hessian."""
        obj = _quadratic(n=3, m=2, seed=97)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-np.ones(3), ub=np.ones(3))],
        )
        # x rides the upper bound, so f_i = 0 there and the 1/f_i atom diverges.
        with self.assertRaises(ValueError) as ctx:
            problem.build_primal_dual_coupled_system(
                np.ones(3), np.zeros(2), np.ones(6), np.zeros(0)
            )
        self.assertIn("interior", str(ctx.exception))


class PrimalDualStationaritySolverTest(unittest.TestCase):
    """End-to-end tests for the Mehrotra ``PrimalDualStationaritySolver`` keystone.

    The primal-dual engine follows the central path directly (no outer barrier
    loop), so where useful its optimum is cross-checked against both the closed-form
    dense saddle and the barrier ``StationaritySolver`` -- the two engines must land
    on the same joint stationary point.
    """

    def test_conforms_to_two_block_protocol(self) -> None:
        """``PrimalDualStationaritySolver`` satisfies the Protocol shape."""
        solver = PrimalDualStationaritySolver()
        # Runtime method-presence check.
        self.assertIsInstance(solver, TwoBlockStationaritySolver)
        # Static conformance: the type checker verifies this assignment.
        engine: TwoBlockStationaritySolver = solver
        self.assertIs(engine, solver)

    def test_unconstrained_quadratic_saddle(self) -> None:
        """Empty domains reduce to a single exact Newton step onto the dense saddle."""
        obj = _quadratic(n=5, m=3, seed=7)
        problem = StationarityProblem(objective=obj)
        result = PrimalDualStationaritySolver().solve(problem)
        x_star, y_star = _dense_saddle(obj)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-8)
        np.testing.assert_allclose(result.y, y_star, atol=1e-8)
        residual = problem.residual(result.x, result.y, t=1.0)
        self.assertLess(float(np.linalg.norm(residual)), 1e-8)

    def test_result_surfaces_saddle_value_and_gap(self) -> None:
        """The PD engine reports the saddle value and a nonnegative surrogate gap.

        Parity with the barrier engine: ``result.objective`` carries the value
        evaluated at the returned iterate and a suboptimality equal to the
        surrogate duality gap ``s^T lambda``, which is nonnegative and collapses
        toward zero on a clean solve. The empty-domain case has no slacks, so its
        gap is exactly zero.
        """
        obj = _quadratic(n=3, m=2, seed=17)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        result = PrimalDualStationaritySolver().solve(problem, x0=x_star, y0=y_star)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        objective = result.objective
        assert objective is not None
        self.assertAlmostEqual(objective.value, float(obj.evaluate(result.x, result.y)))
        self.assertGreaterEqual(objective.suboptimality, 0.0)
        self.assertLess(objective.suboptimality, 1e-3)
        # No inequality domain -> no slacks -> the surrogate gap is exactly zero.
        unconstrained = StationarityProblem(objective=obj)
        bare = PrimalDualStationaritySolver().solve(unconstrained)
        bare_objective = bare.objective
        assert bare_objective is not None
        self.assertEqual(bare_objective.suboptimality, 0.0)

    def test_message_reports_the_surrogate_gap(self) -> None:
        """The PD result message quotes its surrogate gap (its saddle-gap analog)."""
        obj = _quadratic(n=3, m=2, seed=17)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        result = PrimalDualStationaritySolver().solve(problem, x0=x_star, y0=y_star)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        # The gap in the message is exactly the surfaced suboptimality (the
        # surrogate gap s^T lambda), formatted the same way the message does.
        self.assertIn(f"(saddle gap {result.suboptimality:.3g})", result.message)

    def test_box_constrained_saddle_interior(self) -> None:
        """A wide box that does not bind recovers the unconstrained saddle."""
        obj = _quadratic(n=3, m=2, seed=11)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        result = PrimalDualStationaritySolver().solve(problem, x0=x_star, y0=y_star)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)

    def test_binding_box_matches_barrier_engine(self) -> None:
        """A binding box: the PD optimum matches the barrier engine's, inside the bound."""
        obj = _quadratic(n=3, m=2, seed=13)
        x_star, _ = _dense_saddle(obj)
        # Push an upper bound below the unconstrained optimum so it binds.
        ub = x_star - 0.5
        lb = x_star - 5.0
        x0 = x_star - 2.0
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=lb, ub=ub)],
        )
        pd = PrimalDualStationaritySolver().solve(problem, x0=x0)
        barrier = StationaritySolver().solve(problem, x0=x0)
        self.assertEqual(pd.status, SolveStatus.OPTIMAL)
        # Both engines converge to the same constrained joint stationary point.
        np.testing.assert_allclose(pd.x, barrier.x, atol=1e-3)
        np.testing.assert_allclose(pd.y, barrier.y, atol=1e-3)
        # The optimum rides just inside the binding bound.
        self.assertTrue(np.all(pd.x < ub))

    def test_curved_y_ball_non_binding_recovers_saddle(self) -> None:
        """A slack quadratic y-ball over an unconstrained x-block recovers the saddle.

        The y-block carries a single nonlinear (ellipsoidal) constraint rather than
        a box, and the x-block is unconstrained -- so there is exactly one
        complementarity pair (``m_tot == 1``). Sized generously (radius above the
        unconstrained optimum), the ball does not bind and the PD engine must still
        land on the dense saddle. This is the first curved two-block coverage, and
        it guards the y-central-path floor's divisor: dividing ``score`` by the pair
        count alone would inflate the floor here, inject a spurious y-dual, and
        stall; flooring the divisor at the primal DOF keeps it well-behaved.
        """
        obj = _quadratic(n=3, m=2, seed=23)
        x_star, y_star = _dense_saddle(obj)
        phi = float(y_star @ y_star) + 5.0
        problem = StationarityProblem(
            objective=obj,
            y_domain=[SquaredNormConstraintLayer(phi=phi, dimension=obj.dimension_y)],
        )
        # Start strictly interior (zeros is inside the ball, phi > 0) so no Phase I
        # is needed -- the lone-ellipsoid Phase I interior_point hook is a separate
        # unlanded fix and is not what this test exercises.
        result = PrimalDualStationaritySolver().solve(
            problem, x0=x_star, y0=np.zeros(obj.dimension_y)
        )
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)

    def test_curved_y_ball_binding_matches_barrier_engine(self) -> None:
        """A binding quadratic y-ball: PD matches the barrier engine on the manifold.

        The y-ball is tightened until the worst-case y rides its boundary while a
        binding x-box holds the x-block linear -- the exact shape (curved y, linear
        x) that starved the shared-step engine on real robust-MMM data. Both engines
        must converge to the same joint stationary point, with y inside the ball and
        x inside its bound.
        """
        obj = _quadratic(n=3, m=2, seed=29)
        x_star, y_star = _dense_saddle(obj)
        phi = max(0.25 * float(y_star @ y_star), 1e-2)
        # Push an upper bound below the unconstrained optimum so the x-box binds.
        ub = x_star - 0.5
        lb = x_star - 5.0
        x0 = x_star - 2.0
        y0 = np.zeros(obj.dimension_y)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=lb, ub=ub)],
            y_domain=[SquaredNormConstraintLayer(phi=phi, dimension=obj.dimension_y)],
        )
        pd = PrimalDualStationaritySolver().solve(problem, x0=x0, y0=y0)
        barrier = StationaritySolver().solve(problem, x0=x0, y0=y0)
        self.assertEqual(pd.status, SolveStatus.OPTIMAL)
        # Both engines converge to the same constrained joint stationary point.
        np.testing.assert_allclose(pd.x, barrier.x, atol=1e-3)
        np.testing.assert_allclose(pd.y, barrier.y, atol=1e-3)
        # x rides just inside the binding box; y stays within the ball.
        self.assertTrue(np.all(pd.x < ub))
        self.assertLess(float(pd.y @ pd.y), phi + 1e-6)

    def test_matrix_game_bilinear(self) -> None:
        """Weak curvature with dominant coupling: the canonical saddle converges."""
        obj = _quadratic(n=4, m=4, seed=17, coupling_scale=3.0, curvature_scale=0.1)
        problem = StationarityProblem(objective=obj)
        result = PrimalDualStationaritySolver().solve(problem)
        x_star, y_star = _dense_saddle(obj)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-8)
        np.testing.assert_allclose(result.y, y_star, atol=1e-8)

    def test_phase1_feasibilizes_none_start(self) -> None:
        """A ``None`` start over a non-empty domain is feasibilized by Phase I."""
        obj = _quadratic(n=2, m=2, seed=19)
        x_star, y_star = _dense_saddle(obj)
        radius = float(np.max(np.abs(x_star))) + 5.0
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-radius * np.ones(2), ub=radius * np.ones(2))
            ],
        )
        result = PrimalDualStationaritySolver().solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)

    def test_phase1_feasibilizes_infeasible_start(self) -> None:
        """An explicit out-of-domain start is feasibilized by Phase I."""
        obj = _quadratic(n=2, m=2, seed=23)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        result = PrimalDualStationaritySolver().solve(problem, x0=np.array([5.0, 0.0]))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_array_less(-1.0 - 1e-6, result.x)
        np.testing.assert_array_less(result.x, 1.0 + 1e-6)

    def test_inertia_check_rejects_wrong_sign(self) -> None:
        """A non-PD ``neg_hessian_xx`` must trip the once-only inertia check."""
        problem = StationarityProblem(objective=_NegativeCurvatureObjective(n=2, m=2))
        with self.assertRaises(ValueError) as ctx:
            PrimalDualStationaritySolver().solve(problem)
        self.assertIn("inertia", str(ctx.exception))

    def test_iteration_budget_exhaustion_raises(self) -> None:
        """Exhausting the iteration budget raises with the best-effort iterate.

        Capping the inner iterations at one Mehrotra step on a binding-box saddle
        (which needs several) leaves the KKT score above the soft tolerance, so
        neither the strict test nor the soft-tolerance fallback fires. Rather than
        returning a ``MAX_ITERATIONS`` status a caller might overlook, the engine
        raises ``StationarityConvergenceError`` carrying the best-effort iterate.
        """
        obj = _quadratic(n=3, m=2, seed=13)
        x_star, _ = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star - 0.5)],
        )
        trace = ListTrace()
        settings = OptimizationSettings(
            inner=replace(OptimizationSettings().inner, max_iterations=1),
            trace=trace,
        )
        with self.assertRaises(StationarityConvergenceError) as ctx:
            PrimalDualStationaritySolver(settings=settings).solve(
                problem, x0=x_star - 2.0
            )
        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)
        # This exit really is budget exhaustion, so the trace says so -- and names
        # the cap it exhausted, rather than the bare status wording every short
        # exit would otherwise share.
        # First record, not the only one: giving up inside a single iteration also
        # arms the start-point retry, which traces a run of its own.
        done = next(r for r in trace.records() if r["event"] == "pd_done")
        self.assertIn("1-iteration budget", done["reason"])

    def test_seed_hugging_a_face_is_recentered_and_retried(self) -> None:
        """A strictly-interior seed that leaves no room to step is re-derived.

        ``_resolve_block`` accepts any start with every ``f < 0``, and that test is
        too weak to guarantee a usable start: this seed clears the floor hyperplane
        by 1e-12 against box faces 20 wide, so the run gives up without ever
        improving on the seed. Where it gives up is rounding noise: the first dual
        step on the hugging row is a near-cancellation divided by the 1e-12 slack,
        so on some CPUs the first step is capped to nothing (iteration 0) and on
        others it overshoots and stalls one iteration later. Either way ``solve``
        throws the seed away for the block's Phase I point and runs again. The
        trace shows both runs -- the first one's honest diagnosis, then the one
        that converged.
        """
        obj = _quadratic(n=6, m=2, seed=5)
        x_star, _ = _dense_saddle(obj)
        weights = np.full(6, 1.0 / 6.0)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=x_star - 20.0, ub=x_star + 20.0),
                LinearInequalityConstraintLayer(
                    A=weights.reshape(1, -1),
                    cl=np.array([float(weights @ x_star) + 1.0]),
                    cu=None,
                ),
            ],
            y_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        trace = ListTrace()
        result = PrimalDualStationaritySolver(
            settings=OptimizationSettings(trace=trace)
        ).solve(problem, x0=x_star + (1.0 + 1e-12) * np.ones(6))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        starts = [r for r in trace.records() if r["event"] == "pd_start"]
        first, second = [r for r in trace.records() if r["event"] == "pd_done"]
        self.assertEqual(len(starts), 2)
        self.assertIn("strictly-feasible interior", first["reason"])
        self.assertGreater(second["nits"], 1)

    def test_made_no_progress_compares_the_best_score_with_the_start(self) -> None:
        """The retry gate reads the KKT score, not the iteration count.

        Each case is a failed run's exception built by hand, so the rule is pinned
        independently of the floating-point path any real solve takes. A run that
        stalled at iteration 1 after blowing the gap up is as empty as one that
        stalled at iteration 0, and the equality residual counts toward the start
        score just as it counts toward ``best``.
        """

        def failure(
            gaps: tuple[float, ...],
            primal: tuple[float, ...],
            dual: tuple[float, ...],
            best: float,
        ) -> StationarityConvergenceError:
            result = TwoBlockSolveResult.from_blocks(
                x=np.zeros(3),
                y=np.zeros(2),
                status=SolveStatus.MAX_ITERATIONS,
                message="failed",
                solver="pd",
                n_iterations=len(gaps),
                diagnostics=PrimalDualDiagnostics(
                    surrogate_gaps=gaps,
                    primal_residual_norms=primal,
                    dual_residual_norms=dual,
                ),
            )
            return StationarityConvergenceError("failed", best, result)

        # Stalled at iteration 1 with the gap 4 orders of magnitude worse.
        self.assertTrue(
            _made_no_progress(failure((244.0, 3.5e6), (0.0, 0.0), (0.0, 1e6), 244.0))
        )
        # Stalled at iteration 0.
        self.assertTrue(_made_no_progress(failure((244.0,), (0.0,), (0.0,), 244.0)))
        # The loop never ran, so no iterate was recorded at all.
        self.assertTrue(_made_no_progress(failure((), (), (), math.inf)))
        # Reduced the gap before failing.
        self.assertFalse(
            _made_no_progress(failure((244.0, 90.0), (0.0, 0.0), (0.0, 1.0), 90.0))
        )
        # The gap rose but the equality residual, which dominated the start,
        # fell: the score improved from 5 to 2.
        self.assertFalse(
            _made_no_progress(failure((1.0, 2.0), (5.0, 0.0), (0.2, 0.1), 2.0))
        )

    def test_a_seeded_run_that_made_progress_is_not_retried(self) -> None:
        """A failure after real progress is handed back, not recentered.

        A well-centered seed with a 3-iteration budget reduces the KKT score and
        then runs out of iterations. Its best iterate is better than anything a
        fresh start would have after the same work, so ``solve`` raises the
        caller's diagnosis without ever deriving a Phase I point.
        """
        obj = _quadratic(n=6, m=2, seed=5)
        x_star, _ = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 20.0, ub=x_star + 20.0)],
            y_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        settings = OptimizationSettings(
            inner=replace(OptimizationSettings().inner, max_iterations=3),
        )
        solver = PrimalDualStationaritySolver(settings=settings)
        with (
            patch.object(solver, "_centered_start") as centered_start,
            self.assertRaises(StationarityConvergenceError) as caught,
        ):
            solver.solve(problem, x0=x_star.copy(), y0=np.zeros(2))
        centered_start.assert_not_called()
        diagnostics = caught.exception.result.primal_dual_diagnostics
        self.assertEqual(len(diagnostics.surrogate_gaps), 3)
        self.assertLess(caught.exception.suboptimality, diagnostics.surrogate_gaps[0])

    def test_resolve_block_flags_only_a_seed_it_took_verbatim(self) -> None:
        """``_resolve_block`` reports whether the start is the caller's or its own.

        This flag is what arms the recentering retry, and only a seed returned
        untouched can be improved on: a Phase I point is centered by construction,
        so re-deriving it would repeat the same failure at twice the cost. The
        three cases the flag has to separate are a usable seed, an infeasible one,
        and no seed at all -- the last two both route through Phase I.

        Deliberately on the private helper rather than through ``solve``: the
        public tests above cover the usable-seed case (retry fires) and the
        no-seed case (it does not), but an *infeasible* seed and no seed at all
        are indistinguishable from outside -- both silently become the same
        Phase I point, so no public observation separates them.
        """
        obj = _quadratic(n=3, m=2, seed=13)
        x_star, _ = _dense_saddle(obj)
        domain = [BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star - 0.5)]
        settings = OptimizationSettings()
        interior, verbatim = _resolve_block(
            x_star - 2.0, 3, domain, None, "x", settings
        )
        self.assertTrue(verbatim)
        np.testing.assert_allclose(interior, x_star - 2.0)
        # Outside the box, so the seed is only a hint to Phase I.
        _, verbatim = _resolve_block(x_star + 99.0, 3, domain, None, "x", settings)
        self.assertFalse(verbatim)
        _, verbatim = _resolve_block(None, 3, domain, None, "x", settings)
        self.assertFalse(verbatim)

    def test_phase_i_start_is_not_retried(self) -> None:
        """The geometry that needs recentering runs once when it is not seeded.

        Handed no start, ``solve`` resolves one through Phase I -- the same point
        the retry would recenter to. On the domain of the recentering test above
        that point converges directly, so the retry is never armed and the trace
        shows a single run. The ill-centered seed is the defect, not the domain.
        """
        obj = _quadratic(n=6, m=2, seed=5)
        x_star, _ = _dense_saddle(obj)
        weights = np.full(6, 1.0 / 6.0)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=x_star - 20.0, ub=x_star + 20.0),
                LinearInequalityConstraintLayer(
                    A=weights.reshape(1, -1),
                    cl=np.array([float(weights @ x_star) + 1.0]),
                    cu=None,
                ),
            ],
            y_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        trace = ListTrace()
        result = PrimalDualStationaritySolver(
            settings=OptimizationSettings(trace=trace)
        ).solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertEqual(
            len([r for r in trace.records() if r["event"] == "pd_start"]), 1
        )

    def test_failed_recentering_reraises_the_original_diagnosis(self) -> None:
        """A recentering pass that fails leaves the caller's own diagnosis intact.

        The retry is strictly best-effort: it may turn a failure into a success,
        never into a *different* failure. Phase I can decline a domain the main
        loop accepted -- a lone ``SquaredNormConstraintLayer`` is exactly such a
        case today, since ``BarrierSolver`` requires at least one inequality layer
        and this layer offers no ``interior_point`` hook to short-circuit that --
        and a caller who then read a ``ValueError`` about barrier internals would
        be looking at the recovery's problem instead of their own. The original
        ``StationarityConvergenceError`` propagates, with the recovery's failure
        chained as context. The test below covers the sibling layer that does
        supply the hook, and so is rescued rather than re-raised.
        """
        obj = _quadratic(n=6, m=2, seed=5)
        x_star, _ = _dense_saddle(obj)
        weights = np.full(6, 1.0 / 6.0)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=x_star - 20.0, ub=x_star + 20.0),
                LinearInequalityConstraintLayer(
                    A=weights.reshape(1, -1),
                    cl=np.array([float(weights @ x_star) + 1.0]),
                    cu=None,
                ),
            ],
            y_domain=[SquaredNormConstraintLayer(dimension=2, phi=1.0)],
        )
        with self.assertRaises(StationarityConvergenceError) as caught:
            PrimalDualStationaritySolver().solve(
                problem,
                x0=x_star + (1.0 + 1e-12) * np.ones(6),
                y0=np.zeros(2),
            )
        self.assertIn("strictly-feasible interior", caught.exception.result.message)
        self.assertIsInstance(caught.exception.__context__, ValueError)

    def test_recentering_failure_outside_the_optimization_family_is_absorbed(
        self,
    ) -> None:
        """Any retry failure is absorbed, not just the ``OptimizationError`` ones.

        ``ProblemInfeasibleError``, ``NewtonStepError`` and the line-search
        errors all derive straight from ``Exception``, so a catch tuple written
        as ``(OptimizationError, ValueError)`` would let them escape and hand
        the caller the recovery's diagnosis in place of their own -- exactly
        what this branch exists to prevent.
        """
        obj = _quadratic(n=6, m=2, seed=5)
        x_star, _ = _dense_saddle(obj)
        weights = np.full(6, 1.0 / 6.0)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=x_star - 20.0, ub=x_star + 20.0),
                LinearInequalityConstraintLayer(
                    A=weights.reshape(1, -1),
                    cl=np.array([float(weights @ x_star) + 1.0]),
                    cu=None,
                ),
            ],
            y_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        # This seed arms the retry (see the recentering test above); the patch
        # then makes the retry fail on a non-OptimizationError.
        x0 = x_star + (1.0 + 1e-12) * np.ones(6)
        for error in (
            ProblemInfeasibleError("no interior point"),
            NewtonStepError("singular KKT system"),
            ConstraintBoundaryError("step hit the boundary"),
            np.linalg.LinAlgError("singular matrix"),
        ):
            with self.subTest(error=type(error).__name__):
                solver = PrimalDualStationaritySolver()
                with (
                    patch.object(solver, "_centered_start", side_effect=error),
                    self.assertRaises(StationarityConvergenceError) as caught,
                ):
                    solver.solve(problem, x0=x0)
                self.assertIn(
                    "strictly-feasible interior", caught.exception.result.message
                )
                self.assertIs(caught.exception.__context__, error)

    def test_recentering_rescues_a_lone_ellipsoid_y_block(self) -> None:
        """The confidence-ellipsoid y-domain recenters like any other block.

        A y-block whose only inequality is one ellipsoid is the shape the robust
        confidence region builds, and it used to defeat Phase I outright: the
        direct-form path hoists the sole layer into the objective slot and leaves
        ``BarrierSolver`` with no inequality at all. ``NormConstraintLayer`` now
        answers ``interior_point`` with its analytic center, so the recovery has a
        centered point to fall back on and the rescue reaches the block that
        motivated it -- not just the polyhedral x side.
        """
        obj = _quadratic(n=6, m=2, seed=5)
        x_star, _ = _dense_saddle(obj)
        weights = np.full(6, 1.0 / 6.0)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=x_star - 20.0, ub=x_star + 20.0),
                LinearInequalityConstraintLayer(
                    A=weights.reshape(1, -1),
                    cl=np.array([float(weights @ x_star) + 1.0]),
                    cu=None,
                ),
            ],
            y_domain=[NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=1.0)],
        )
        trace = ListTrace()
        result = PrimalDualStationaritySolver(
            settings=OptimizationSettings(trace=trace)
        ).solve(
            problem,
            x0=x_star + (1.0 + 1e-12) * np.ones(6),
            y0=np.zeros(2),
        )
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertEqual(
            len([r for r in trace.records() if r["event"] == "pd_start"]), 2
        )

    def test_requires_start_matching_dimension(self) -> None:
        """A wrong-shape start raises a clear ``ValueError``, not a numpy error."""
        obj = _quadratic(n=2, m=2, seed=31)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=-np.ones(2), ub=np.ones(2))],
        )
        with self.assertRaises(ValueError) as ctx:
            PrimalDualStationaritySolver().solve(problem, x0=np.zeros(3))
        self.assertIn("shape", str(ctx.exception))

    def test_result_fields(self) -> None:
        """Status, counts, duals, and per-iteration diagnostics are populated."""
        obj = _quadratic(n=3, m=2, seed=29)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
        )
        result = PrimalDualStationaritySolver().solve(problem, x0=np.zeros(3))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        self.assertEqual(result.x.shape, (3,))
        self.assertEqual(result.y.shape, (2,))
        self.assertGreater(result.n_iterations, 0)
        # The primal-dual engine records one diagnostics entry per iteration.
        diagnostics = result.primal_dual_diagnostics
        self.assertEqual(len(diagnostics.surrogate_gaps), result.n_iterations)
        self.assertEqual(len(diagnostics.primal_residual_norms), result.n_iterations)
        self.assertEqual(len(diagnostics.dual_residual_norms), result.n_iterations)
        # The x-block BoundsConstraintLayer contributes six inequality duals.
        self.assertIsNotNone(result.duals)
        self.assertEqual(result.inequality_multipliers.shape, (6,))


class _FixedConstraintLayer:
    """A ``ConstraintLayer`` stand-in that ignores ``w`` and always returns the
    same constraint values.

    Duck-typed (not a real ``ConstraintLayer`` subclass): the interior-guard
    regression below only needs ``num_constraints``/``constraints`` to force a
    specific ``s = -f`` deterministically, independent of what a real Newton
    step could ever produce.
    """

    def __init__(self, values: npt.NDArray[np.float64]) -> None:
        self._values = values

    def num_constraints(self, dimension: int) -> int:
        return self._values.shape[0]

    def constraints(
        self, w: npt.NDArray[np.float64], shift: float = 0.0
    ) -> npt.NDArray[np.float64]:
        return self._values - shift


class _FixedResidualProblem:
    """A minimal ``StationarityProblem`` stand-in for the interior-guard test.

    Implements only what ``PrimalDualStationaritySolver._run``/``_run_equality``
    touch before the interior guard fires: dimensions, domains,
    ``num_equalities``, ``primal_dual_residual``, and ``objective.evaluate``.
    Deliberately omits ``build_primal_dual_coupled_system`` -- the point of the
    test is that a non-interior first iterate must break out before that would
    ever be called.
    """

    class _ZeroObjective:
        def evaluate(
            self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
        ) -> float:
            return 0.0

    def __init__(
        self,
        n: int,
        x_constraint_values: npt.NDArray[np.float64],
        residual: npt.NDArray[np.float64],
    ) -> None:
        self.dimension_x = n
        self.dimension_y = 0
        self.x_domain = [_FixedConstraintLayer(x_constraint_values)]
        self.y_domain: list[object] = []
        self.num_equalities = 0
        self._residual = residual
        self.objective = _FixedResidualProblem._ZeroObjective()

    def primal_dual_residual(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
        lam_eq: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        return self._residual


class PrimalDualStationaritySolverInteriorGuardTest(unittest.TestCase):
    """Regression: a boundary iterate must never be certified ``ACCEPTABLE``.

    ``_run``/``_run_equality`` recorded the best-seen KKT-score iterate and
    checked convergence BEFORE checking whether that iterate was strictly
    interior. A constraint whose slack sits exactly on the boundary (``s <=
    0``, e.g. from machine-epsilon underflow in the convergence tail) could
    therefore still be latched as ``best`` -- and later handed back by the
    soft-tolerance fallback as an ``ACCEPTABLE`` "solution" that violates the
    strict-interior invariant ``build_primal_dual_coupled_system`` requires.
    The fix computes the interior guard first and gates both the best-iterate
    update and the ``OPTIMAL`` check on it.
    """

    def test_run_does_not_accept_a_boundary_first_iterate(self) -> None:
        # s_x = [1e-4, 0.0]: one comfortably interior constraint, one sitting
        # exactly on the boundary (not strictly interior) -> interior=False.
        x_constraint_values = np.array([-1e-4, 0.0])
        # A residual small enough that the KKT score lands inside
        # `tolerance_soft` (default 1e-3) but outside the strict `tolerance`
        # (default 1e-6) -- "good enough" to trip the soft-tolerance fallback
        # if this boundary point were (wrongly) recorded as `best`.
        residual = np.array([1e-4, 1e-4])
        problem = cast(
            "StationarityProblem",
            _FixedResidualProblem(
                n=2, x_constraint_values=x_constraint_values, residual=residual
            ),
        )
        solver = PrimalDualStationaritySolver()

        # As of this diff, exhausting the budget without OPTIMAL/ACCEPTABLE
        # raises rather than returning a MAX_ITERATIONS status (see
        # `_raise_nonconvergence`). The boundary point must never be accepted,
        # so the correct outcome here is that raise, not a returned
        # `ACCEPTABLE` result.
        with self.assertRaises(StationarityConvergenceError) as ctx:
            solver._run(problem, np.array([1.0, 2.0]), np.zeros(0))
        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)

    def test_run_equality_does_not_accept_a_boundary_first_iterate(self) -> None:
        """Same defect, mirrored in the ``x_equality`` (bordered ``nu``) loop."""
        x_constraint_values = np.array([-1e-4, 0.0])
        residual = np.array([1e-4, 1e-4])
        problem = cast(
            "StationarityProblem",
            _FixedResidualProblem(
                n=2, x_constraint_values=x_constraint_values, residual=residual
            ),
        )
        solver = PrimalDualStationaritySolver()

        with self.assertRaises(StationarityConvergenceError) as ctx:
            solver._run_equality(problem, np.array([1.0, 2.0]), np.zeros(0))
        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)

    def test_run_wraps_a_newton_step_failure_with_the_best_iterate(self) -> None:
        """A ``NewtonStepError`` inside the loop must not escape unwrapped.

        The barrier engine catches a failed Newton solve and lets its centering
        loop stop cleanly (``StationaritySolver._center``), but the primal-dual
        loop had no such guard: a singular, indefinite or non-finite coupled
        system raised a bare ``NewtonStepError`` that matched neither
        ``SaddlePointSolver``'s handler nor the CDM delegate's -- both key off the
        wrapped types -- and reached the caller carrying no iterate at all. A
        Newton step that cannot be computed is a non-convergence, so it must
        arrive as a ``StationarityConvergenceError`` with the best interior
        iterate attached and the original error chained as the cause.
        """
        obj = _quadratic(n=3, m=2, seed=13)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
        )
        solver = PrimalDualStationaritySolver()
        failure = NewtonStepError("H is not positive definite")
        with (
            patch.object(solver, "_solve_direction", side_effect=failure),
            self.assertRaises(StationarityConvergenceError) as ctx,
        ):
            solver._run(problem, x_star - 1.0, y_star)

        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)
        # Chained, so the traceback still reaches the step that gave up.
        self.assertIs(ctx.exception.__cause__, failure)
        self.assertIn("could not be computed", str(ctx.exception))
        # The attached iterate is the strictly-interior point recorded at the top
        # of the iteration whose Newton solve then failed -- not an empty result.
        self.assertEqual(ctx.exception.result.x.shape, x_star.shape)
        self.assertTrue(np.all(np.isfinite(ctx.exception.result.x)))

    def test_run_equality_wraps_a_newton_step_failure(self) -> None:
        """Same guard, mirrored in the ``x_equality`` (bordered ``nu``) loop."""
        obj = _quadratic(n=3, m=2, seed=404)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=np.zeros(3), ub=np.ones(3))],
            x_equality=EqualityConstraintLayer(A=np.ones((1, 3)), b=np.array([1.0])),
        )
        solver = PrimalDualStationaritySolver()
        failure = NewtonStepError("Schur complement contains NaN/Inf")
        with (
            patch.object(solver, "_solve_direction", side_effect=failure),
            self.assertRaises(StationarityConvergenceError) as ctx,
        ):
            solver._run_equality(problem, np.full(3, 1.0 / 3.0), np.zeros(2))

        self.assertEqual(ctx.exception.result.status, SolveStatus.MAX_ITERATIONS)
        self.assertIs(ctx.exception.__cause__, failure)
        self.assertIn("could not be computed", str(ctx.exception))
        self.assertTrue(np.all(np.isfinite(ctx.exception.result.x)))


class PrimalDualStationaritySolverEqualityTest(unittest.TestCase):
    """End-to-end tests of the primal-dual ``x_equality`` (bordered/``nu``) path.

    Mirrors ``StationaritySolverEqualityTest``: the primal-dual optimum is
    cross-checked against the closed-form bordered KKT point and, on a binding box,
    against the barrier ``StationaritySolver`` -- the two engines must agree.
    """

    @staticmethod
    def _equality(
        n: int, p: int, seed: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Random full-row-rank ``A_eq`` of shape ``(p, n)`` and target ``b_eq``."""
        rng = np.random.default_rng(seed)
        A_eq = rng.standard_normal((p, n))
        b_eq = rng.standard_normal(p)
        return A_eq, b_eq

    def test_equality_only_matches_dense_kkt(self) -> None:
        """No inequality domain: the bordered solve hits the closed-form KKT point."""
        obj = _quadratic(n=5, m=3, seed=101)
        A_eq, b_eq = self._equality(n=5, p=2, seed=7)
        problem = StationarityProblem(
            objective=obj,
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        result = PrimalDualStationaritySolver().solve(problem)
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        x_star, y_star = _dense_equality_kkt(obj, A_eq, b_eq)
        np.testing.assert_allclose(result.x, x_star, atol=1e-8)
        np.testing.assert_allclose(result.y, y_star, atol=1e-8)
        np.testing.assert_allclose(A_eq @ result.x, b_eq, atol=1e-8)

    def test_infeasible_start_reaches_equality(self) -> None:
        """An x0 off ``A_eq x = b_eq`` is driven onto the manifold by Newton."""
        obj = _quadratic(n=4, m=2, seed=202)
        A_eq, b_eq = self._equality(n=4, p=1, seed=11)
        problem = StationarityProblem(
            objective=obj,
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        # Deliberately start far from the equality manifold.
        result = PrimalDualStationaritySolver().solve(problem, x0=10.0 * np.ones(4))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        x_star, _ = _dense_equality_kkt(obj, A_eq, b_eq)
        np.testing.assert_allclose(result.x, x_star, atol=1e-8)
        np.testing.assert_allclose(A_eq @ result.x, b_eq, atol=1e-8)

    def test_equality_with_nonbinding_box_matches_kkt(self) -> None:
        """A slack box leaves the equality-constrained KKT point unchanged."""
        obj = _quadratic(n=4, m=3, seed=303)
        A_eq, b_eq = self._equality(n=4, p=1, seed=13)
        x_star, y_star = _dense_equality_kkt(obj, A_eq, b_eq)
        # A box wide enough that the KKT optimum sits strictly interior.
        radius = float(np.max(np.abs(x_star))) + 5.0
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-radius * np.ones(4), ub=radius * np.ones(4))
            ],
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        result = PrimalDualStationaritySolver().solve(problem, x0=np.zeros(4))
        self.assertEqual(result.status, SolveStatus.OPTIMAL)
        np.testing.assert_allclose(result.x, x_star, atol=1e-4)
        np.testing.assert_allclose(result.y, y_star, atol=1e-4)
        np.testing.assert_allclose(A_eq @ result.x, b_eq, atol=1e-6)

    def test_budget_equality_holds_with_binding_box(self) -> None:
        """A sum-budget equality is honored even when box bounds are active."""
        obj = _quadratic(n=3, m=2, seed=404)
        # Simplex-style budget: entries sum to 1.
        A_eq = np.ones((1, 3))
        b_eq = np.array([1.0])
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=np.zeros(3), ub=np.ones(3))],
            x_equality=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        x0 = np.full(3, 1.0 / 3.0)
        pd = PrimalDualStationaritySolver().solve(problem, x0=x0)
        self.assertEqual(pd.status, SolveStatus.OPTIMAL)
        self.assertAlmostEqual(float(np.sum(pd.x)), 1.0, places=6)
        # Box bounds respected (strictly interior up to the barrier tolerance).
        self.assertTrue(np.all(pd.x >= -1e-6))
        self.assertTrue(np.all(pd.x <= 1.0 + 1e-6))
        # The barrier engine lands on the same constrained joint stationary point.
        barrier = StationaritySolver().solve(problem, x0=x0)
        np.testing.assert_allclose(pd.x, barrier.x, atol=1e-3)
        np.testing.assert_allclose(pd.y, barrier.y, atol=1e-3)


class _RecordingStationaritySolver:
    """Spy ``TwoBlockStationaritySolver`` that records its forwarded arguments.

    Returns a fixed ``sentinel`` result so ``StationarityProblem.solve`` -- a
    pure delegation -- can be checked on exactly what it forwards and what it
    returns, without depending on any real solve converging.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.problem: StationarityProblem | None = None
        self.x0: npt.NDArray[np.float64] | None = None
        self.y0: npt.NDArray[np.float64] | None = None
        self.sentinel = TwoBlockSolveResult.from_blocks(
            x=np.zeros(3),
            y=np.zeros(2),
            status=SolveStatus.OPTIMAL,
            message="sentinel",
            solver="recording",
            n_iterations=0,
        )

    def solve(
        self,
        problem: StationarityProblem,
        x0: npt.NDArray[np.float64] | None = None,
        y0: npt.NDArray[np.float64] | None = None,
    ) -> TwoBlockSolveResult:
        self.calls += 1
        self.problem = problem
        self.x0 = x0
        self.y0 = y0
        return self.sentinel


class StationarityProblemSolveSugarTest(unittest.TestCase):
    """Tests for ``StationarityProblem.solve``, the ``Problem.solve`` analog."""

    def test_barrier_sugar_matches_direct_engine_call(self) -> None:
        """``problem.solve(engine, ...)`` equals ``engine.solve(problem, ...)``."""
        obj = _quadratic(n=5, m=3, seed=7)
        problem = StationarityProblem(objective=obj)
        via_sugar = problem.solve(StationaritySolver())
        direct = StationaritySolver().solve(problem)
        np.testing.assert_array_equal(via_sugar.x, direct.x)
        np.testing.assert_array_equal(via_sugar.y, direct.y)
        self.assertEqual(via_sugar.status, direct.status)

    def test_primal_dual_sugar_matches_direct_engine_call(self) -> None:
        """The sugar dispatches to whichever engine it is handed (Mehrotra)."""
        obj = _quadratic(n=4, m=2, seed=11)
        x_star, y_star = _dense_saddle(obj)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )
        via_sugar = problem.solve(PrimalDualStationaritySolver(), x0=x_star, y0=y_star)
        direct = PrimalDualStationaritySolver().solve(problem, x0=x_star, y0=y_star)
        np.testing.assert_array_equal(via_sugar.x, direct.x)
        np.testing.assert_array_equal(via_sugar.y, direct.y)
        self.assertEqual(via_sugar.status, direct.status)

    def test_sugar_forwards_warm_starts(self) -> None:
        """The sugar forwards ``self`` and the exact ``x0``/``y0`` to the engine.

        Convergence alone cannot guard this: the barrier engine reaches the
        saddle whether or not the warm starts arrive, so a solve-and-check test
        would still pass if the sugar dropped them. A recording engine instead
        captures precisely what it received, so the assertion fails the moment
        forwarding regresses.
        """
        problem = StationarityProblem(objective=_quadratic(n=3, m=2, seed=13))
        x0 = np.array([0.1, 0.2, 0.3])
        y0 = np.array([-0.4, 0.5])

        engine = _RecordingStationaritySolver()
        # The spy conforms to the Protocol the sugar dispatches on.
        self.assertIsInstance(engine, TwoBlockStationaritySolver)
        result = problem.solve(engine, x0=x0, y0=y0)

        # Pure delegation: the same problem and the same warm-start arrays reach
        # the engine, and its own result is handed straight back.
        self.assertEqual(engine.calls, 1)
        self.assertIs(engine.problem, problem)
        self.assertIs(engine.x0, x0)
        self.assertIs(engine.y0, y0)
        self.assertIs(result, engine.sentinel)

    def test_sugar_forwards_absent_warm_starts_as_none(self) -> None:
        """Omitting the warm starts forwards ``None``, not a substituted default."""
        problem = StationarityProblem(objective=_quadratic(n=3, m=2, seed=17))

        engine = _RecordingStationaritySolver()
        problem.solve(engine)

        self.assertEqual(engine.calls, 1)
        self.assertIs(engine.problem, problem)
        self.assertIsNone(engine.x0)
        self.assertIsNone(engine.y0)


class TwoBlockStepRecordingTest(unittest.TestCase):
    """Both two-block engines record the line-search step they accepted.

    Without this the step trajectory is an empty tuple, and every consumer
    guard built on it -- `steps_past_quadratic_convergence()` for the barrier,
    `late_step_sizes()` for primal-dual -- passes vacuously forever. That was
    the state of `robust_optimization`'s guards until this recording existed,
    so these tests protect the *precondition* for those guards, not just the
    data.
    """

    @staticmethod
    def _bounded_problem(seed: int) -> StationarityProblem:
        """A saddle with both blocks boxed, so the line search actually engages."""
        obj = _quadratic(n=3, m=2, seed=seed)
        x_star, y_star = _dense_saddle(obj)
        return StationarityProblem(
            objective=obj,
            x_domain=[BoundsConstraintLayer(lb=x_star - 5.0, ub=x_star + 5.0)],
            y_domain=[BoundsConstraintLayer(lb=y_star - 5.0, ub=y_star + 5.0)],
        )

    def test_the_barrier_engine_records_one_step_per_inner_iteration(self) -> None:
        problem = self._bounded_problem(seed=29)

        result = StationaritySolver().solve(problem, x0=np.zeros(3))
        diagnostics = result.barrier_diagnostics

        self.assertIsInstance(diagnostics, SaddleBarrierDiagnostics)
        self.assertGreater(len(diagnostics.inner_btls_steps), 0)
        # One outer entry per centering, and within each, one step per accepted
        # inner Newton iteration -- the alignment
        # `steps_past_quadratic_convergence()` relies on when it slices by
        # `inner_quadratic_from`.
        self.assertEqual(
            list(diagnostics.inner_nits),
            [len(steps) for steps in diagnostics.inner_btls_steps],
        )
        every_step = [s for steps in diagnostics.inner_btls_steps for s in steps]
        self.assertTrue(
            all(0.0 < s <= 1.0 for s in every_step),
            f"steps outside (0, 1]: {sorted(every_step)[:8]}",
        )

    def test_the_barrier_engine_leaves_the_quadratic_index_unset(self) -> None:
        """Deliberate: a residual norm carries no self-concordance threshold.

        Pinned so the emptiness stays a documented choice rather than drifting
        into a silent bug. Consumers assert on the step trajectory directly.
        """
        problem = self._bounded_problem(seed=29)

        diagnostics = (
            StationaritySolver().solve(problem, x0=np.zeros(3)).barrier_diagnostics
        )

        self.assertEqual((), diagnostics.inner_quadratic_from)
        self.assertEqual((), diagnostics.steps_past_quadratic_convergence())

    def test_the_primal_dual_engine_records_its_accepted_steps(self) -> None:
        problem = self._bounded_problem(seed=17)

        result = PrimalDualStationaritySolver().solve(problem)
        diagnostics = result.diagnostics

        self.assertIsInstance(diagnostics, PrimalDualDiagnostics)
        assert isinstance(diagnostics, PrimalDualDiagnostics)  # narrow for mypy
        self.assertGreater(len(diagnostics.step_sizes), 0)
        self.assertGreater(len(diagnostics.late_step_sizes()), 0)
        self.assertTrue(
            all(0.0 < s <= 1.0 for s in diagnostics.step_sizes),
            f"steps outside (0, 1]: {sorted(diagnostics.step_sizes)[:8]}",
        )
        # Recorded at the bottom of the loop body, so never longer than the
        # top-of-body trajectories and at most one shorter.
        gap = len(diagnostics.surrogate_gaps) - len(diagnostics.step_sizes)
        self.assertIn(gap, (0, 1), f"step_sizes is {gap} entries behind the gaps")

    def test_the_primal_dual_engine_records_steps_with_an_equality(self) -> None:
        """The equality loop is a separate code path and records too."""
        obj = _quadratic(n=3, m=2, seed=17)
        problem = StationarityProblem(
            objective=obj,
            x_domain=[
                BoundsConstraintLayer(lb=-10.0 * np.ones(3), ub=10.0 * np.ones(3))
            ],
            x_equality=EqualityConstraintLayer(A=np.ones((1, 3)), b=np.zeros(1)),
        )

        result = PrimalDualStationaritySolver().solve(problem)
        diagnostics = result.diagnostics

        assert isinstance(diagnostics, PrimalDualDiagnostics)  # narrow for mypy
        self.assertGreater(len(diagnostics.step_sizes), 0)
        self.assertTrue(
            all(0.0 < s <= 1.0 for s in diagnostics.step_sizes),
            f"steps outside (0, 1]: {sorted(diagnostics.step_sizes)[:8]}",
        )
