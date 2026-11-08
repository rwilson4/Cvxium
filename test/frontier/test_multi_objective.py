"""Tests for MultiObjectiveOptimizer and FrontierResults."""

import contextlib
import os
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import numpy.typing as npt

from cvxium import (
    Duals,
    ObjectiveOutcome,
    SolveResult,
    SolveStatus,
)
from cvxium._typing import none_throws
from cvxium.exceptions import OptimizationError
from cvxium.frontier.multi_objective import (
    _BLAS_THREAD_ENV_VARS,
    FrontierPoint,
    FrontierResults,
    MultiObjectiveOptimizer,
    _knee_space_transform,
    _single_thread_blas,
)

# ---------------------------------------------------------------------------
# Analytical test solver: minimize ||x||^2 s.t. ||x - d||^2 <= phi  (primary=0)
#                    or: minimize ||x-d||^2 s.t. ||x||^2 <= phi      (primary=1)
#
# Two objectives:
#   f0(x) = ||x||^2
#   f1(x) = ||x - d||^2
#
# Analytical solutions:
#   primary=0, bound phi on f1:
#     phi >= ||d||^2 → x* = 0,            f0* = 0,              f1* = ||d||^2
#     phi <  ||d||^2 → x* = d(1-√φ/‖d‖), f0* = (‖d‖-√φ)^2,   f1* = φ
#
#   primary=1, bound phi on f0:
#     phi >= ||d||^2 → x* = d,            f0* = ||d||^2,        f1* = 0
#     phi <  ||d||^2 → x* = d·√φ/‖d‖,    f0* = φ,              f1* = (‖d‖-√φ)^2
#
# In both cases the Pareto frontier is the curve f0=(‖d‖-√f1)^2, f1 in [0,‖d‖^2].
# ---------------------------------------------------------------------------


def _make_ipm_result(x: npt.NDArray[np.float64], obj: float) -> SolveResult:
    """Construct a minimal optimization result for testing."""
    return SolveResult(
        solution=x.copy(),
        status=SolveStatus.OPTIMAL,
        message="analytical",
        solver="barrier",
        n_iterations=1,
        objective=ObjectiveOutcome(value=obj, suboptimality=0.0, dual_value=obj),
        duals=Duals(inequality_multipliers=np.array([0.0])),
    )


class BallConstrainedNorm(MultiObjectiveOptimizer):
    r"""Analytical two-objective solver respecting primary_objective.

    Objectives:
        f0(x) = ||x||^2
        f1(x) = ||x - d||^2

    When primary_objective=0: solve_with_bounds([phi]) minimises f0 s.t. f1 <= phi. When
    primary_objective=1: solve_with_bounds([phi]) minimises f1 s.t. f0 <= phi.

    """

    def __init__(self, d: npt.NDArray[np.float64], primary_objective: int = 0) -> None:
        super().__init__(primary_objective=primary_objective)
        self.d = d
        self._d_norm = float(np.linalg.norm(d))

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        phi = float(bounds[0])
        d_norm_sq = self._d_norm**2
        if self.primary_objective == 0:
            # minimize ||x||^2 s.t. ||x-d||^2 <= phi
            x = (
                np.zeros_like(self.d)
                if phi >= d_norm_sq
                else self.d * (1.0 - np.sqrt(phi) / self._d_norm)
            )
        else:
            # minimize ||x-d||^2 s.t. ||x||^2 <= phi
            x = (
                self.d.copy()
                if phi >= d_norm_sq
                else self.d * (np.sqrt(phi) / self._d_norm)
            )
        return _make_ipm_result(x, float(np.dot(x, x)))

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        # minimize f0 -> x=0; minimize f1 -> x=d
        x = np.zeros_like(self.d) if objective_index == 0 else self.d.copy()
        return _make_ipm_result(x, float(np.dot(x, x)))

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array([float(np.dot(x, x)), float(np.dot(x - self.d, x - self.d))])


class SeparableBallConstraints(MultiObjectiveOptimizer):
    r"""Analytical three-objective solver (separable by dimension).

    Decision variable x in R^2. Objectives:
        f0(x) = x[0]^2 + x[1]^2
        f1(x) = (x[0] - 1)^2
        f2(x) = (x[1] - 1)^2

    primary_objective=0: solve_with_bounds([phi1, phi2]) imposes f1<=phi1, f2<=phi2.
    Separable solution: x[k]* = 0 if phi_{k+1} >= 1 else 1 - sqrt(phi_{k+1}).

    """

    def __init__(self, primary_objective: int = 0) -> None:
        super().__init__(primary_objective=primary_objective)

    def _solve_1d(self, phi: float) -> float:
        """Minimize x^2 s.t. (x-1)^2 <= phi."""
        if phi >= 1.0:
            return 0.0
        return 1.0 - np.sqrt(phi)

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        v0 = self._solve_1d(float(bounds[0]))
        v1 = self._solve_1d(float(bounds[1]))
        x = np.array([v0, v1])
        return _make_ipm_result(x, float(np.dot(x, x)))

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        if objective_index == 0:
            x = np.zeros(2)
        elif objective_index == 1:
            x = np.array([1.0, 0.0])
        else:
            x = np.array([0.0, 1.0])
        return _make_ipm_result(x, float(np.dot(x, x)))

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [
                float(x[0] ** 2 + x[1] ** 2),
                float((x[0] - 1.0) ** 2),
                float((x[1] - 1.0) ** 2),
            ]
        )


# ---------------------------------------------------------------------------
# Helper: a solver whose auxiliary corner solves always fail (to test knee()
# behaviour with incomplete corners).
# ---------------------------------------------------------------------------


class PartialCornerSolver(BallConstrainedNorm):
    """Like BallConstrainedNorm but minimize_objective fails for objective 1."""

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        if objective_index == 1:
            raise OptimizationError("synthetic failure", 1.0, np.zeros(1))
        return super().minimize_objective(objective_index)


# ---------------------------------------------------------------------------
# Tests: two objectives, primary_objective=0
# ---------------------------------------------------------------------------


class TwoObjectiveFrontierTest(unittest.TestCase):
    """Tests for a two-objective optimizer."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        d = rng.standard_normal(5)
        self.solver = BallConstrainedNorm(d=d)

    def test_trace_returns_points(self) -> None:
        """trace() returns a non-empty FrontierResults."""
        results = self.solver.trace(num_points=10)
        assert isinstance(results, FrontierResults)
        assert len(results.points) > 0

    def test_corners_computed(self) -> None:
        """Two corner points are computed, one per objective."""
        results = self.solver.trace(num_points=5)
        assert len(results.corners) == 2

    def test_primary_corner_minimizes_f0(self) -> None:
        """Corner 0 (primary) achieves f0 near zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[0].objectives[0], 0.0, delta=1e-10)

    def test_auxiliary_corner_minimizes_f1(self) -> None:
        """Corner 1 (auxiliary) achieves f1 near zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[1].objectives[1], 0.0, delta=1e-10)

    def test_frontier_monotone(self) -> None:
        """As phi (auxiliary bound) increases, f0 is non-increasing."""
        results = self.solver.trace(num_points=20)
        interior = [p for p in results.points if not np.any(np.isinf(p.bounds))]
        interior.sort(key=lambda p: float(p.bounds[0]))
        f0_values = [p.objectives[0] for p in interior]
        for i in range(len(f0_values) - 1):
            assert f0_values[i] >= f0_values[i + 1] - 1e-10

    def test_frontier_matches_analytical(self) -> None:
        """Frontier points match the analytical formula f0 = (||d|| - sqrt(f1))^2."""
        d_norm = self.solver._d_norm
        results = self.solver.trace(num_points=20)
        for p in results.points:
            f0, f1 = float(p.objectives[0]), float(p.objectives[1])
            expected_f0 = max(0.0, (d_norm - np.sqrt(max(f1, 0.0))) ** 2)
            self.assertAlmostEqual(f0, expected_f0, delta=1e-8)

    def test_knee_is_a_frontier_point(self) -> None:
        """knee() returns one of the points in results.points (by identity)."""
        results = self.solver.trace(num_points=10)
        knee = results.knee()
        assert any(knee is p for p in results.points)

    def test_knee_not_at_extreme(self) -> None:
        """Knee is not the same as either corner (it should be in the interior)."""
        results = self.solver.trace(num_points=20)
        knee = results.knee()
        assert knee.objectives[0] > 1e-6
        assert knee.objectives[1] > 1e-6

    def test_primary_objective_stored(self) -> None:
        """FrontierResults records primary_objective correctly."""
        results = self.solver.trace(num_points=5)
        assert results.primary_objective == 0

    def test_diagnostics(self) -> None:
        """n_attempted and n_skipped are tracked correctly."""
        num_points = 7
        results = self.solver.trace(num_points=num_points)
        # For N=2 there is 1 auxiliary, so grid has num_points points.
        assert results.n_attempted == num_points
        # This solver never raises, so nothing is skipped.
        assert results.n_skipped == 0

    def test_num_points_validation(self) -> None:
        """trace() raises ValueError for num_points < 1."""
        with self.assertRaisesRegex(ValueError, "num_points must be >= 1"):
            self.solver.trace(num_points=0)
        with self.assertRaisesRegex(ValueError, "num_points must be >= 1"):
            self.solver.trace(num_points=-5)

    def test_grid_excludes_corner_bounds(self) -> None:
        """Grid points use bounds strictly between the corner objective values.

        The corners are already solved in step 1 of trace(); re-solving at the corner
        bounds is redundant and can leave the inner solver's feasible set with empty
        interior. Regression test for that fix.

        """
        num_points = 8
        results = self.solver.trace(num_points=num_points)
        bounds_min = float(results.corners[1].objectives[1])
        bounds_max = float(results.corners[0].objectives[1])
        if bounds_min > bounds_max:
            bounds_min, bounds_max = bounds_max, bounds_min
        interior = [p for p in results.points if not np.any(np.isinf(p.bounds))]
        # All interior grid points have bounds strictly inside the range.
        for p in interior:
            phi = float(p.bounds[0])
            assert phi > bounds_min + 1e-12
            assert phi < bounds_max - 1e-12

    def test_deduplication(self) -> None:
        """No two points in results.points share an objective vector."""
        results = self.solver.trace(num_points=20)
        for i, p in enumerate(results.points):
            for q in results.points[:i]:
                assert not np.allclose(p.objectives, q.objectives, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: two objectives, primary_objective=1
# ---------------------------------------------------------------------------


class TwoObjectiveFrontierAlternatePrimaryTest(unittest.TestCase):
    """Correctness tests when the non-default objective is primary."""

    def setUp(self) -> None:
        rng = np.random.default_rng(99)
        d = rng.standard_normal(4)
        self.solver = BallConstrainedNorm(d=d, primary_objective=1)

    def test_primary_corner_minimizes_f1(self) -> None:
        """With primary=1, corner 0 minimizes f1 (||x-d||^2) to zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[0].objectives[1], 0.0, delta=1e-10)

    def test_auxiliary_corner_minimizes_f0(self) -> None:
        """With primary=1, corner 1 (the auxiliary corner) minimizes f0 to zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[1].objectives[0], 0.0, delta=1e-10)

    def test_frontier_monotone(self) -> None:
        """As f0 bound increases, f1 (primary) is non-increasing."""
        results = self.solver.trace(num_points=20)
        interior = [p for p in results.points if not np.any(np.isinf(p.bounds))]
        interior.sort(key=lambda p: float(p.bounds[0]))
        f1_values = [float(p.objectives[1]) for p in interior]
        for i in range(len(f1_values) - 1):
            assert f1_values[i] >= f1_values[i + 1] - 1e-10

    def test_frontier_matches_analytical(self) -> None:
        """With primary=1, frontier still satisfies f1 = (||d|| - sqrt(f0))^2."""
        d_norm = self.solver._d_norm
        results = self.solver.trace(num_points=20)
        for p in results.points:
            f0, f1 = float(p.objectives[0]), float(p.objectives[1])
            expected_f1 = max(0.0, (d_norm - np.sqrt(max(f0, 0.0))) ** 2)
            self.assertAlmostEqual(f1, expected_f1, delta=1e-8)

    def test_knee_available(self) -> None:
        """knee() works for primary_objective=1."""
        results = self.solver.trace(num_points=10)
        knee = results.knee()
        assert any(knee is p for p in results.points)

    def test_primary_objective_in_results(self) -> None:
        """FrontierResults records primary_objective=1."""
        results = self.solver.trace(num_points=5)
        assert results.primary_objective == 1


# ---------------------------------------------------------------------------
# Tests: FrontierResults.knee geometry
# ---------------------------------------------------------------------------


class KneeGeometryTest(unittest.TestCase):
    """Unit tests for the knee calculation."""

    def _make_point(
        self, objectives: list[float], solution: list[float] | None = None
    ) -> FrontierPoint:
        if solution is None:
            solution = [0.0] * len(objectives)
        ipm = _make_ipm_result(np.array(solution), 0.0)
        return FrontierPoint(
            objectives=np.array(objectives),
            solution=np.array(solution),
            bounds=np.array([np.inf]),
            ipm_result=ipm,
        )

    def test_knee_2d_midpoint(self) -> None:
        """For a curved 2-D frontier, knee is interior (not at the corners)."""
        corners = [
            self._make_point([0.0, 1.0]),
            self._make_point([1.0, 0.0]),
        ]
        f1_vals = np.linspace(0.0, 1.0, 51)
        points = list(corners) + [
            self._make_point([(1.0 - np.sqrt(f1)) ** 2, f1]) for f1 in f1_vals[1:-1]
        ]
        results = FrontierResults(
            points=points,
            corners=corners,
            primary_objective=0,
            n_attempted=len(points),
            n_skipped=0,
        )
        knee = results.knee()
        assert knee.objectives[0] > 0.01
        assert knee.objectives[1] > 0.01

    def test_knee_raises_on_empty_points(self) -> None:
        """knee() raises ValueError when points list is empty."""
        corners = [self._make_point([0.0, 1.0]), self._make_point([1.0, 0.0])]
        results = FrontierResults(
            points=[], corners=corners, primary_objective=0, n_attempted=0, n_skipped=0
        )
        with self.assertRaisesRegex(ValueError, "No frontier points"):
            results.knee()

    def test_knee_raises_with_single_corner(self) -> None:
        """knee() raises ValueError when fewer than 2 corners are available."""
        corner = self._make_point([0.0, 1.0])
        point = self._make_point([0.5, 0.5])
        results = FrontierResults(
            points=[corner, point],
            corners=[corner],
            primary_objective=0,
            n_attempted=1,
            n_skipped=0,
        )
        with self.assertRaisesRegex(ValueError, "at least 2 corner points"):
            results.knee()

    def test_knee_raises_with_incomplete_corners_3d(self) -> None:
        """knee() raises when n_corners < n_objectives (partial corner failure)."""
        # 3-D objectives but only 2 corners (one auxiliary solve failed).
        corners = [
            self._make_point([0.0, 1.0, 1.0]),
            self._make_point([1.0, 0.0, 1.0]),
        ]
        points = [*corners, self._make_point([0.5, 0.5, 0.5])]
        results = FrontierResults(
            points=points,
            corners=corners,
            primary_objective=0,
            n_attempted=1,
            n_skipped=0,
        )
        with self.assertRaisesRegex(ValueError, "exactly one corner per objective"):
            results.knee()


# ---------------------------------------------------------------------------
# Tests: three objectives
# ---------------------------------------------------------------------------


class ThreeObjectiveFrontierTest(unittest.TestCase):
    """Tests for the three-objective separable solver."""

    def setUp(self) -> None:
        self.solver = SeparableBallConstraints(primary_objective=0)

    def test_trace_returns_points(self) -> None:
        """trace() returns a non-empty FrontierResults for N=3."""
        results = self.solver.trace(num_points=5)
        assert len(results.points) > 0

    def test_three_corners_computed(self) -> None:
        """Three corner points are computed for three objectives."""
        results = self.solver.trace(num_points=5)
        assert len(results.corners) == 3

    def test_primary_corner_f0_zero(self) -> None:
        """Primary corner minimizes f0 to zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[0].objectives[0], 0.0, delta=1e-10)

    def test_aux1_corner_f1_zero(self) -> None:
        """First auxiliary corner minimizes f1 to zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[1].objectives[1], 0.0, delta=1e-10)

    def test_aux2_corner_f2_zero(self) -> None:
        """Second auxiliary corner minimizes f2 to zero."""
        results = self.solver.trace(num_points=5)
        self.assertAlmostEqual(results.corners[2].objectives[2], 0.0, delta=1e-10)

    def test_grid_size(self) -> None:
        """For N=3 and num_points=k, the grid contributes at most k^2 points."""
        num_points = 4
        results = self.solver.trace(num_points=num_points)
        assert results.n_attempted == num_points**2

    def test_knee_available(self) -> None:
        """knee() returns a FrontierPoint for N=3."""
        results = self.solver.trace(num_points=5)
        knee = results.knee()
        assert isinstance(knee, FrontierPoint)
        assert any(knee is p for p in results.points)


class PlotTest(unittest.TestCase):
    """Tests for FrontierResults.plot()."""

    def test_plot_returns_axes(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        matplotlib.use("Agg")
        plt.close("all")
        d = np.array([1.0, 2.0])
        solver = BallConstrainedNorm(d)
        fr = solver.trace(num_points=10)
        ax = fr.plot()
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_no_annotation(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        matplotlib.use("Agg")
        plt.close("all")
        d = np.array([1.0, 2.0])
        solver = BallConstrainedNorm(d)
        fr = solver.trace(num_points=10)
        ax = fr.plot(annotate_knee=False)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_a_crossed_band_is_not_shaded(self) -> None:
        """A negative band width is finite, and meaningless.

        When the samples contradict convexity the chord and tangent bounds
        cross, so the band width goes negative while staying finite. Shading
        it would draw an inverted lens that reads as an error bar rather than
        as the warning it is, so the guard is on the sign and not just on
        finiteness.

        """
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        plt.close("all")
        solver = BallConstrainedNorm(np.array([1.0, 2.0]))
        fr = solver.trace(num_points=10)
        env = fr.envelope(num=20)
        crossed = replace(env, max_band_width=-0.5)

        _, ax = plt.subplots()
        with (
            mock.patch.object(type(fr), "envelope", return_value=crossed),
            mock.patch.object(ax, "fill_betweenx") as shaded,
        ):
            fr.plot(ax=ax)

        shaded.assert_not_called()
        plt.close("all")

    def test_plot_raises_for_3_objectives(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        solver = SeparableBallConstraints()
        fr = solver.trace(num_points=3)
        with self.assertRaises(NotImplementedError):
            fr.plot()

    def test_plot_raises_for_empty_frontier(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        fr = FrontierResults(
            points=[],
            corners=[],
            primary_objective=0,
            n_attempted=0,
            n_skipped=0,
        )
        with self.assertRaises(ValueError):
            fr.plot()


# ---------------------------------------------------------------------------
# Tests: warm-start threading
# ---------------------------------------------------------------------------


class WarmStartTracker(BallConstrainedNorm):
    """BallConstrainedNorm that records x0 inputs and solutions returned."""

    def __init__(self, d: npt.NDArray[np.float64], primary_objective: int = 0) -> None:
        super().__init__(d=d, primary_objective=primary_objective)
        self.warm_starts_swb: list[npt.NDArray[np.float64] | None] = []
        self.solutions_swb: list[npt.NDArray[np.float64]] = []
        self.warm_starts_mo: list[npt.NDArray[np.float64] | None] = []
        self.solutions_mo: list[npt.NDArray[np.float64]] = []

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        self.warm_starts_swb.append(x0)
        result = super().solve_with_bounds(bounds, x0=x0)
        self.solutions_swb.append(result.solution)
        return result

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        self.warm_starts_mo.append(x0)
        result = super().minimize_objective(objective_index, x0=x0)
        self.solutions_mo.append(result.solution)
        return result


class WarmStartThreadingTest(unittest.TestCase):
    """Tests that x0 is threaded from one solve to the next."""

    def test_grid_receives_warm_start(self) -> None:
        """After the first grid solve, subsequent calls receive a non-None warm_start."""
        d = np.array([1.0, 2.0, 3.0])
        solver = WarmStartTracker(d)
        solver.trace(num_points=5)

        # The first grid call should receive the last corner result
        # (non-None), and subsequent calls should also be non-None.
        assert len(solver.warm_starts_swb) == 5
        for ws in solver.warm_starts_swb:
            assert ws is not None

    def test_auxiliary_corners_receive_warm_start(self) -> None:
        """Auxiliary minimize_objective calls receive the primary result."""
        d = np.array([1.0, 2.0])
        solver = WarmStartTracker(d)
        solver.trace(num_points=3)

        # minimize_objective is called twice: once for primary (no x0),
        # once for auxiliary (with primary solution as x0).
        assert len(solver.warm_starts_mo) == 2
        assert solver.warm_starts_mo[0] is None  # primary corner
        np.testing.assert_array_equal(
            solver.warm_starts_mo[1],
            solver.solutions_mo[0],
        )

    def test_warm_start_chains_across_grid(self) -> None:
        """Each grid solve receives the solution of the previous one."""
        d = np.array([1.0, 2.0])
        solver = WarmStartTracker(d)
        solver.trace(num_points=4)

        # Verify the chain: warm_starts_swb[k] should be the solution
        # returned by the (k-1)-th call, not a stale or repeated value.
        for k in range(1, len(solver.warm_starts_swb)):
            np.testing.assert_array_equal(
                solver.warm_starts_swb[k],
                solver.solutions_swb[k - 1],
            )


# ---------------------------------------------------------------------------
# Tests: callback support
# ---------------------------------------------------------------------------


class CallbackTest(unittest.TestCase):
    """Tests for the callback parameter of trace()."""

    def test_callback_invocation_count_two_objectives(self) -> None:
        """Callback is called once per successful solve (corners + grid)."""
        rng = np.random.default_rng(42)
        d = rng.standard_normal(5)
        solver = BallConstrainedNorm(d=d)
        num_points = 10
        calls: list[tuple[int, int]] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            calls.append((k, total))
            return False

        solver.trace(num_points=num_points, callback=cb)

        # N=2 objectives: 2 corners + num_points grid = 12 total solves.
        # BallConstrainedNorm never fails, so all 12 trigger the callback.
        expected_total = 2 + num_points
        assert len(calls) == expected_total
        assert all(t == expected_total for _, t in calls)

    def test_callback_k_increments_continuously(self) -> None:
        """k increments from 1 to total across corners and grid."""
        rng = np.random.default_rng(99)
        d = rng.standard_normal(3)
        solver = BallConstrainedNorm(d=d)
        num_points = 7
        k_values: list[int] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            k_values.append(k)
            return False

        solver.trace(num_points=num_points, callback=cb)

        expected_total = 2 + num_points
        assert k_values == list(range(1, expected_total + 1))

    def test_callback_early_termination(self) -> None:
        """Returning True from callback stops the sweep early."""
        rng = np.random.default_rng(42)
        d = rng.standard_normal(5)
        solver = BallConstrainedNorm(d=d)
        calls: list[int] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            calls.append(k)
            return k >= 3  # stop after 3rd successful solve

        results = solver.trace(num_points=10, callback=cb)

        assert len(calls) == 3
        assert calls == [1, 2, 3]
        # k=3 stops during grid phase (after 2 corners + 1 grid point),
        # so we get 2 corners + 1 grid point = 3 unique points max
        # (minus deduplication, but corners and first grid point are distinct).
        assert len(results.points) <= 3

    def test_callback_early_termination_during_corners(self) -> None:
        """Stopping during corner phase skips the grid entirely."""
        solver = SeparableBallConstraints(primary_objective=0)
        calls: list[int] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            calls.append(k)
            return k >= 2  # stop after second corner

        results = solver.trace(num_points=5, callback=cb)

        assert len(calls) == 2
        # Grid was skipped, so n_attempted should be 0.
        assert results.n_attempted == 0

    def test_callback_three_objectives(self) -> None:
        """Callback count is correct for 3-objective problems."""
        solver = SeparableBallConstraints(primary_objective=0)
        num_points = 4
        calls: list[tuple[int, int]] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            calls.append((k, total))
            return False

        solver.trace(num_points=num_points, callback=cb)

        # N=3: 3 corners + 4^2=16 grid = 19 total.
        expected_total = 3 + num_points**2
        assert len(calls) == expected_total
        assert all(t == expected_total for _, t in calls)
        # k should run 1..19
        k_values = [k for k, _ in calls]
        assert k_values == list(range(1, expected_total + 1))

    def test_callback_not_called_for_skipped_solves(self) -> None:
        """Failed solves do not trigger the callback."""
        # Use PartialCornerSolver where objective 1 always fails.
        rng = np.random.default_rng(42)
        d = rng.standard_normal(5)
        solver = PartialCornerSolver(d=d)
        num_points = 5
        calls: list[int] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            calls.append(k)
            return False

        results = solver.trace(num_points=num_points, callback=cb)

        # N=2: total expected = 2 + 5 = 7.
        # But corner 1 fails, so only 1 corner + 5 grid = 6 callbacks.
        # (The counter still counts only successful solves.)
        assert results.n_skipped == 0  # grid points don't fail
        assert len(calls) == 1 + num_points
        # k should run 1..6 (skipping the failed corner doesn't increment k).
        assert calls == list(range(1, 1 + num_points + 1))

    def test_no_callback_matches_original_behavior(self) -> None:
        """trace() without callback produces the same results."""
        rng = np.random.default_rng(42)
        d = rng.standard_normal(5)
        solver = BallConstrainedNorm(d=d)

        results_no_cb = solver.trace(num_points=10)
        results_with_cb = solver.trace(num_points=10, callback=lambda p, k, t: False)

        assert len(results_no_cb.points) == len(results_with_cb.points)
        for p1, p2 in zip(results_no_cb.points, results_with_cb.points, strict=False):
            np.testing.assert_allclose(p1.objectives, p2.objectives, atol=1e-12)


# ---------------------------------------------------------------------------
# Helper: a degenerate solver where all objectives share the same minimiser.
# ---------------------------------------------------------------------------


class DegenerateSolver(MultiObjectiveOptimizer):
    r"""Two-objective solver where both objectives have the same minimiser.

      f0(x) = ||x - d||^2
      f1(x) = ||x - d||^2

    Every corner and every grid point returns x* = d, so the frontier is a single point.

    """

    def __init__(self, d: npt.NDArray[np.float64]) -> None:
        super().__init__(primary_objective=0)
        self.d = d
        self.n_solves = 0

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        self.n_solves += 1
        x = self.d.copy()
        return _make_ipm_result(x, 0.0)

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        x = self.d.copy()
        return _make_ipm_result(x, 0.0)

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        diff = x - self.d
        val = float(np.dot(diff, diff))
        return np.array([val, val])


# ---------------------------------------------------------------------------
# Tests: degenerate frontier detection
# ---------------------------------------------------------------------------


class DegenerateFrontierTest(unittest.TestCase):
    """Tests for degenerate Pareto frontier detection."""

    def test_degenerate_skips_grid(self) -> None:
        """When all corners coincide, the grid sweep is skipped entirely."""
        d = np.array([1.0, 2.0, 3.0])
        solver = DegenerateSolver(d=d)
        results = solver.trace(num_points=50)

        # Grid sweep was skipped, so solve_with_bounds was never called.
        assert solver.n_solves == 0
        assert results.n_attempted == 0
        assert results.n_skipped == 0

    def test_degenerate_single_unique_point(self) -> None:
        """Degenerate frontier collapses to a single unique point."""
        d = np.array([1.0, 2.0])
        solver = DegenerateSolver(d=d)
        results = solver.trace(num_points=20)

        # After deduplication, only one point remains.
        assert len(results.points) == 1
        np.testing.assert_allclose(
            results.points[0].objectives, np.array([0.0, 0.0]), atol=1e-10
        )

    def test_degenerate_corners_still_present(self) -> None:
        """Corners are still computed even when degenerate."""
        d = np.array([1.0, 2.0, 3.0])
        solver = DegenerateSolver(d=d)
        results = solver.trace(num_points=10)

        assert len(results.corners) == 2


# ---------------------------------------------------------------------------
# Helper: a solver that raises on some grid points (to exercise n_skipped in
# the parallel path).
# ---------------------------------------------------------------------------


class SometimesInfeasibleBall(BallConstrainedNorm):
    """BallConstrainedNorm whose grid solve is infeasible for small bounds.

    ``solve_with_bounds`` raises for ``phi`` below half the objective range, so
    roughly the lower half of the auxiliary grid is skipped. Corner solves
    (``minimize_objective``) always succeed, so the frontier still has both
    corners.

    """

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        phi = float(bounds[0])
        if phi < 0.5 * self._d_norm**2:
            raise OptimizationError("synthetic infeasibility", 1.0, np.zeros(1))
        return super().solve_with_bounds(bounds, x0=x0)


# ---------------------------------------------------------------------------
# Tests: BLAS single-thread context manager
# ---------------------------------------------------------------------------


class SingleThreadBlasTest(unittest.TestCase):
    """Unit tests for the _single_thread_blas context manager."""

    def test_pins_all_vars_inside(self) -> None:
        """Inside the context, every BLAS var is set to "1"."""
        with _single_thread_blas():
            for var in _BLAS_THREAD_ENV_VARS:
                assert os.environ.get(var) == "1"

    def test_restores_prior_value(self) -> None:
        """A pre-existing value is restored on exit."""
        var = _BLAS_THREAD_ENV_VARS[0]
        # patch.dict restores whatever the process really had, so a developer's
        # or CI's own OMP_NUM_THREADS survives the test.
        with mock.patch.dict(os.environ, {var: "7"}, clear=False):
            with _single_thread_blas():
                assert os.environ[var] == "1"
            assert os.environ[var] == "7"

    def test_removes_var_that_was_unset(self) -> None:
        """A var that was unset before entry is removed again on exit."""
        var = _BLAS_THREAD_ENV_VARS[0]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(var, None)
            with _single_thread_blas():
                assert os.environ[var] == "1"
            assert var not in os.environ


# ---------------------------------------------------------------------------
# Tests: parallel grid sweep (trace(n_jobs != 1))
# ---------------------------------------------------------------------------


class ParallelTraceTest(unittest.TestCase):
    """Tests that trace(n_jobs != 1) matches the sequential sweep."""

    @staticmethod
    def _objset(results: FrontierResults) -> list[tuple[float, float]]:
        return sorted(
            (float(p.objectives[0]), float(p.objectives[1])) for p in results.points
        )

    def _assert_same_frontier(self, seq: FrontierResults, par: FrontierResults) -> None:
        """Assert the two sweeps found the same objective vectors, to tolerance.

        The sequential sweep warm-starts each solve from the previous IPM result
        while the parallel one solves cold, so the two converge to fixed points
        that agree only to solver tolerance. Exact float equality would be flaky.
        """
        seq_objs, par_objs = self._objset(seq), self._objset(par)
        self.assertEqual(len(seq_objs), len(par_objs))
        np.testing.assert_allclose(
            np.array(par_objs), np.array(seq_objs), rtol=0.0, atol=1e-9
        )

    def test_parallel_matches_sequential_two_objectives(self) -> None:
        """Parallel and sequential traces produce the same frontier."""
        rng = np.random.default_rng(7)
        d = rng.standard_normal(5)
        seq = BallConstrainedNorm(d=d).trace(num_points=12, n_jobs=1)
        par = BallConstrainedNorm(d=d).trace(num_points=12, n_jobs=4)
        self._assert_same_frontier(seq, par)

    def test_parallel_matches_sequential_three_objectives(self) -> None:
        """Parallel matches sequential for a 3-objective (2-D grid) problem."""
        seq = SeparableBallConstraints().trace(num_points=5, n_jobs=1)
        par = SeparableBallConstraints().trace(num_points=5, n_jobs=4)
        self._assert_same_frontier(seq, par)

    def test_parallel_diagnostics(self) -> None:
        """n_attempted counts the whole grid; nothing is skipped."""
        rng = np.random.default_rng(11)
        d = rng.standard_normal(4)
        num_points = 10
        par = BallConstrainedNorm(d=d).trace(num_points=num_points, n_jobs=4)
        assert par.n_attempted == num_points
        assert par.n_skipped == 0

    def test_parallel_corners_present(self) -> None:
        """Corners are still computed (sequentially) in the parallel path."""
        rng = np.random.default_rng(3)
        d = rng.standard_normal(4)
        par = BallConstrainedNorm(d=d).trace(num_points=6, n_jobs=4)
        assert len(par.corners) == 2
        self.assertAlmostEqual(par.corners[0].objectives[0], 0.0, delta=1e-10)
        self.assertAlmostEqual(par.corners[1].objectives[1], 0.0, delta=1e-10)

    def test_parallel_counts_skipped(self) -> None:
        """Grid points that raise are counted in n_skipped, not points."""
        rng = np.random.default_rng(5)
        d = rng.standard_normal(5)
        num_points = 12
        seq = SometimesInfeasibleBall(d=d).trace(num_points=num_points, n_jobs=1)
        par = SometimesInfeasibleBall(d=d).trace(num_points=num_points, n_jobs=4)
        # Same set of points survive, and the same count is skipped, regardless
        # of sequential vs parallel execution.
        assert seq.n_skipped == par.n_skipped
        assert seq.n_skipped > 0
        assert seq.n_attempted == par.n_attempted == num_points
        self._assert_same_frontier(seq, par)

    def test_parallel_callback_fires_per_success(self) -> None:
        """Callback fires once per successful solve (order not guaranteed)."""
        rng = np.random.default_rng(9)
        d = rng.standard_normal(4)
        num_points = 8
        calls: list[int] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            calls.append(k)
            return False

        BallConstrainedNorm(d=d).trace(num_points=num_points, n_jobs=4, callback=cb)
        # 2 corners (sequential) + num_points grid solves, all successful.
        assert len(calls) == 2 + num_points
        # k is contiguous 1..total even though grid completion order varies.
        assert sorted(calls) == list(range(1, 2 + num_points + 1))

    def test_n_jobs_negative_uses_default_pool(self) -> None:
        """n_jobs < 1 (one worker per core) still produces the frontier."""
        rng = np.random.default_rng(13)
        d = rng.standard_normal(4)
        seq = BallConstrainedNorm(d=d).trace(num_points=6, n_jobs=1)
        par = BallConstrainedNorm(d=d).trace(num_points=6, n_jobs=-1)
        self._assert_same_frontier(seq, par)


# ---------------------------------------------------------------------------
# Tests: the overridable _build_grids hook
# ---------------------------------------------------------------------------


def _swept_bounds(results: FrontierResults) -> list[float]:
    """The sorted bounds the sweep actually solved at, corners excluded.

    Corner points carry an infinite bound (they were solved by
    ``minimize_objective``, not ``solve_with_bounds``), which is how they are
    told apart from grid points here.
    """
    return sorted(
        float(p.bounds[0]) for p in results.points if not np.any(np.isinf(p.bounds))
    )


class _CustomGridBall(BallConstrainedNorm):
    """A BallConstrainedNorm whose sweep grid is a fixed custom array.

    Overrides only ``_build_grids`` -- ignoring the corner-derived
    bounds_min/bounds_max -- to prove that ``trace`` routes grid construction
    through the hook and feeds the returned values verbatim to
    ``solve_with_bounds``.

    """

    def __init__(
        self,
        d: npt.NDArray[np.float64],
        custom_grid: npt.NDArray[np.float64],
    ) -> None:
        super().__init__(d=d, primary_objective=0)
        self._custom_grid = np.asarray(custom_grid, dtype=float)

    def _build_grids(
        self,
        bounds_min: npt.NDArray[np.float64],
        bounds_max: npt.NDArray[np.float64],
        num_points: int,
        n_aux: int,
    ) -> list[npt.NDArray[np.float64]]:
        return [self._custom_grid.copy()]


class _BoundRecordingBall(BallConstrainedNorm):
    """Records each point's auxiliary coordinate as the bound it was solved at."""

    def point_objectives(
        self, result: SolveResult, bounds: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        objectives = self.evaluate_objectives(result.solution)
        if np.all(np.isfinite(bounds)):
            objectives[1] = 2.0 * float(bounds[0])  # unmistakably not attained
        return objectives


class PointObjectivesHookTest(unittest.TestCase):
    """Frontier points take their objectives from ``point_objectives``."""

    def test_default_is_the_attained_objectives(self) -> None:
        solver = BallConstrainedNorm(np.array([3.0, 0.0]))
        for point in solver.trace(num_points=4).points:
            np.testing.assert_allclose(
                point.objectives, solver.evaluate_objectives(point.solution)
            )

    def test_an_override_reaches_the_sweep_but_not_the_corners(self) -> None:
        solver = _BoundRecordingBall(np.array([3.0, 0.0]))
        results = solver.trace(num_points=4)
        swept = [p for p in results.points if np.all(np.isfinite(p.bounds))]
        self.assertEqual(len(swept), 4)
        for point in swept:
            self.assertAlmostEqual(float(point.objectives[1]), 2.0 * point.bounds[0])
        for corner in results.corners:
            np.testing.assert_allclose(
                corner.objectives, solver.evaluate_objectives(corner.solution)
            )

    def test_an_override_reaches_the_parallel_sweep(self) -> None:
        solver = _BoundRecordingBall(np.array([3.0, 0.0]))
        results = solver.trace(num_points=4, n_jobs=2)
        swept = [p for p in results.points if np.all(np.isfinite(p.bounds))]
        self.assertEqual(len(swept), 4)
        for point in swept:
            self.assertAlmostEqual(float(point.objectives[1]), 2.0 * point.bounds[0])


class BuildGridsHookTest(unittest.TestCase):
    """trace() builds its sweep grid via the overridable _build_grids hook."""

    def test_default_reproduces_interior_linspace(self) -> None:
        """The default hook yields the interior linspace grid used before."""
        solver = BallConstrainedNorm(np.array([3.0, 0.0]), primary_objective=0)
        num_points = 6
        results = solver.trace(num_points=num_points)
        # Bracket the auxiliary objective by its two corner values rather than
        # relying on where each corner lands in the list.
        aux_at_corners = [float(c.objectives[1]) for c in results.corners]
        expected = np.linspace(
            min(aux_at_corners), max(aux_at_corners), num_points + 2
        )[1:-1]
        interior = _swept_bounds(results)
        self.assertEqual(len(interior), num_points)
        np.testing.assert_allclose(interior, np.sort(expected), atol=1e-9)

    def test_override_drives_the_sweep(self) -> None:
        """An overridden _build_grids feeds its own values verbatim as bounds."""
        custom = np.array([0.5, 1.5, 2.5, 4.0])
        solver = _CustomGridBall(np.array([3.0, 0.0]), custom_grid=custom)
        # num_points is ignored by the override; a distinct value means the
        # default grid could not coincide with ``custom`` by accident.
        results = solver.trace(num_points=3)
        interior = _swept_bounds(results)
        self.assertEqual(len(interior), len(custom))
        np.testing.assert_allclose(interior, np.sort(custom), atol=1e-9)

    def test_callback_total_matches_the_overridden_grid(self) -> None:
        """``total`` counts the grid the hook actually returned, not num_points.

        The pre-corner estimate of ``total`` assumes the default grid; an
        override that returns a different number of points must not leave the
        progress counter running past its own denominator.
        """
        custom = np.array([0.5, 1.5, 2.5, 4.0])
        solver = _CustomGridBall(np.array([3.0, 0.0]), custom_grid=custom)
        seen: list[tuple[int, int]] = []

        def cb(point: FrontierPoint, k: int, total: int) -> bool:
            seen.append((k, total))
            return False

        solver.trace(num_points=3, callback=cb)
        # 2 corners + len(custom) grid solves, all successful.
        self.assertEqual([k for k, _ in seen], list(range(1, 2 + len(custom) + 1)))
        # The sweep callbacks see the corrected denominator.
        self.assertEqual(seen[-1][1], 2 + len(custom))

    def test_empty_axis_sweeps_nothing_and_keeps_the_corners(self) -> None:
        """A zero-length axis is a well-formed request for an empty sweep."""
        solver = _CustomGridBall(np.array([3.0, 0.0]), custom_grid=np.array([]))
        results = solver.trace(num_points=3)
        self.assertEqual(_swept_bounds(results), [])
        self.assertEqual(results.n_attempted, 0)
        self.assertEqual(len(results.corners), 2)

    def test_no_axes_at_all_is_rejected(self) -> None:
        """Returning zero axes raises rather than solving at ``bounds=[]``.

        ``itertools.product()`` over no axes yields one *empty* combination, so
        without the guard this degenerates into a single meaningless solve at an
        empty bounds vector instead of the empty sweep the override meant.
        """
        solver = _CustomGridBall(np.array([3.0, 0.0]), custom_grid=np.array([]))
        with (
            mock.patch.object(_CustomGridBall, "_build_grids", return_value=[]),
            self.assertRaisesRegex(ValueError, "returned no axes"),
        ):
            solver.trace(num_points=3)


class FrontierEnvelopeTest(unittest.TestCase):
    """FrontierResults exposes the certified interpolant, not just a polyline."""

    def setUp(self) -> None:
        rng = np.random.default_rng(90210)
        self.solver = BallConstrainedNorm(d=rng.standard_normal(5))

    def test_envelope_bounds_the_frontier_on_both_sides_when_slopes_are_known(
        self,
    ) -> None:
        """Chord above, tangent below, and a finite band between them."""
        results = self.solver.trace(num_points=8)
        # A convex frontier's chord is an upper bound; supply the exact slopes
        # so the tangent bound is available too.
        x = np.array([float(p.objectives[1]) for p in results.points])
        y = np.array([float(p.objectives[0]) for p in results.points])
        order = np.argsort(x)
        results.tangent_slopes = np.gradient(y[order], x[order])[np.argsort(order)]

        env = results.envelope(num=50)

        self.assertTrue(np.all(env.tangent <= env.chord + 1e-9))
        self.assertTrue(np.isfinite(env.max_band_width))

    def test_envelope_degrades_without_slopes(self) -> None:
        """No slopes means no lower bound, but still a smooth curve."""
        results = self.solver.trace(num_points=8)
        self.assertIsNone(results.tangent_slopes)

        env = results.envelope(num=50)

        self.assertTrue(np.all(np.isneginf(env.tangent)))
        self.assertEqual(len(env.curve), 50)

    def test_trace_asks_the_optimizer_for_slopes(self) -> None:
        """The hook is consulted, and the base answer is honestly None."""
        with mock.patch.object(
            BallConstrainedNorm, "tangent_slopes", return_value=None
        ) as hook:
            results = self.solver.trace(num_points=5)

        hook.assert_called_once()
        self.assertIsNone(results.tangent_slopes)

    def test_a_subclass_can_supply_slopes(self) -> None:
        """An override reaches the results, and thence the envelope."""

        class Sloped(BallConstrainedNorm):
            def tangent_slopes(
                self, results: FrontierResults
            ) -> npt.NDArray[np.float64]:
                return np.full(len(results.points), -1.0)

        solver = Sloped(d=np.array([1.0, 2.0, 3.0]))
        results = solver.trace(num_points=5)

        self.assertIsNotNone(results.tangent_slopes)
        self.assertEqual(len(none_throws(results.tangent_slopes)), len(results.points))

    def test_envelope_rejects_three_objectives(self) -> None:
        """The convexity argument is a two-objective one."""
        results = SeparableBallConstraints().trace(num_points=3)

        with self.assertRaisesRegex(ValueError, "two-objective"):
            results.envelope()

    def test_slopes_survive_a_rebuild_through_the_constructor(self) -> None:
        """The pattern that has bitten twice: rebuilding and losing the slopes.

        ``tangent_slopes`` is the only state a results object carries that is
        not implied by its points, so a rebuild that forgets it downgrades every
        envelope downstream in silence. Passing it through the constructor is
        what makes the omission visible.

        """
        original = self.solver.trace(num_points=5)
        original.tangent_slopes = np.full(len(original.points), -1.0)

        rebuilt = FrontierResults(
            points=original.points,
            corners=original.corners,
            primary_objective=original.primary_objective,
            n_attempted=original.n_attempted,
            n_skipped=original.n_skipped,
            tangent_slopes=original.tangent_slopes,
        )

        np.testing.assert_allclose(
            none_throws(rebuilt.tangent_slopes), none_throws(original.tangent_slopes)
        )
        self.assertTrue(np.isfinite(rebuilt.envelope(num=20).max_band_width))

    @staticmethod
    def _asymmetric_results() -> FrontierResults:
        """A frontier whose objectives occupy clearly different ranges.

        A symmetric fixture cannot detect a transpose: when both objectives
        span the same interval, swapping the axes leaves the extents unchanged
        and every assertion on them still passes.

        """
        t = np.linspace(0.0, 1.0, 21)
        primary = 3.0 + 7.0 * (1.0 - np.sqrt(t)) ** 2  # spans [3, 10]
        auxiliary = 100.0 + 250.0 * t  # spans [100, 350]
        pts = [
            FrontierPoint(
                objectives=np.array([pr, au]),
                solution=np.zeros(1),
                bounds=np.array([np.inf]),
                ipm_result=_make_ipm_result(np.zeros(1), 0.0),
            )
            for pr, au in zip(primary, auxiliary, strict=False)
        ]
        return FrontierResults(
            points=pts,
            corners=[pts[0], pts[-1]],
            primary_objective=0,
            n_attempted=len(pts),
            n_skipped=0,
        )

    def test_plot_draws_the_interpolant_in_the_plots_orientation(self) -> None:
        """The interpolant must not be drawn transposed.

        ``plot`` puts the PRIMARY objective on the x-axis; ``envelope`` follows
        the interpolation module's convention of the swept AUXILIARY objective
        on its ``x``. Drawing ``env.x`` as the abscissa flips the curve
        relative to the points, the fill and the axis labels.

        """
        results = self._asymmetric_results()

        ax = results.plot(annotate_knee=False)
        drawn_x = np.concatenate([line.get_xdata() for line in ax.lines])
        drawn_y = np.concatenate([line.get_ydata() for line in ax.lines])

        self.assertAlmostEqual(float(drawn_x.min()), 3.0, delta=0.5)
        self.assertAlmostEqual(float(drawn_x.max()), 10.0, delta=0.5)
        self.assertAlmostEqual(float(drawn_y.min()), 100.0, delta=1.0)
        self.assertAlmostEqual(float(drawn_y.max()), 350.0, delta=1.0)

    def test_plot_falls_back_when_the_envelope_cannot_be_built(self) -> None:
        """A frontier that will not interpolate is still worth rendering."""
        results = self.solver.trace(num_points=6)

        with mock.patch.object(
            FrontierResults, "envelope", side_effect=ValueError("nope")
        ):
            ax = results.plot(annotate_knee=False)

        self.assertGreater(len(ax.lines), 0)

    def test_envelope_rejects_a_single_point(self) -> None:
        """The two-point contract is enforced here, not left to the kernel."""
        results = self.solver.trace(num_points=6)
        results.points = results.points[:1]

        with self.assertRaisesRegex(ValueError, "at least two points"):
            results.envelope()


class RefineKneeTest(unittest.TestCase):
    """The knee is solved for, not picked from the sweep grid."""

    def setUp(self) -> None:
        self.d: npt.NDArray[np.float64] = np.array([3.0, 4.0])

    @staticmethod
    def _sloped_solver(
        d: npt.NDArray[np.float64] | None = None,
    ) -> BallConstrainedNorm:
        """``BallConstrainedNorm`` with its analytic frontier derivative.

        The frontier is ``f0 = (||d|| - sqrt(f1))^2``, so
        ``df0/df1 = -(||d|| - sqrt(f1)) / sqrt(f1)``.

        A static method taking ``d`` rather than reading ``self.d``, so other
        test classes can borrow the fixture without borrowing the test methods
        that come with inheriting it.

        """
        d = np.array([3.0, 4.0]) if d is None else d
        norm = float(np.linalg.norm(d))

        class Sloped(BallConstrainedNorm):
            def tangent_slopes(
                self, results: FrontierResults
            ) -> npt.NDArray[np.float64]:
                out = []
                for p in results.points:
                    f1 = float(p.objectives[1])
                    out.append(
                        np.nan if f1 <= 0.0 else -(norm - np.sqrt(f1)) / np.sqrt(f1)
                    )
                return np.array(out)

        return Sloped(d=d)

    def test_it_stops_once_the_tangent_parallels_the_chord(self) -> None:
        """Every extra step is a real solve, so convergence must end the loop.

        The cap is 24. Running it out when the tangent already parallels the
        chord to tolerance is roughly twenty wasted solves, and this runs once
        per cell on a calibration path.

        """
        solver = self._sloped_solver()
        results = solver.trace(num_points=10)

        with mock.patch.object(type(solver), "_probe", wraps=solver._probe) as probed:
            knee = solver.refine_knee(results)

        self.assertIsNotNone(knee)
        # 2 on this fixture. The bound is loose enough not to pin the
        # arithmetic and tight enough to fail if the loop runs out its cap.
        self.assertLessEqual(probed.call_count, 8)

    def test_a_first_probe_failure_reports_no_knee(self) -> None:
        """Before any narrowing the iterate is just the middle of the span.

        Handing that back as "the knee" is worse than saying nothing: the
        caller cannot tell it from a converged answer and will calibrate to
        it, whereas None makes it fall back to the conventional band.

        """
        solver = self._sloped_solver()
        results = solver.trace(num_points=10)

        with mock.patch.object(type(solver), "_probe", return_value=None):
            self.assertIsNone(solver.refine_knee(results))

    def test_a_later_probe_failure_keeps_the_bracketed_iterate(self) -> None:
        """Once narrowed, the last iterate brackets the knee and is worth keeping.

        The failure has to fire on the SECOND probe. This fixture converges on
        slope shortly after that, so failing any later may never fire at all
        and the test would pass on the ordinary converged path -- green for the
        wrong reason. The assertion on the call count catches that, and
        comparing against the unperturbed knee proves the abort branch ran
        rather than convergence.

        The returned point is pinned exactly, because "somewhere strictly
        inside the corner span" does not discriminate: the first-probe-failure
        path refuses to hand back a point in that same region. What separates
        the two is not where the point sits but what is known about it -- the
        first probe's slope was compared against the chord, so the knee is on a
        known side of it. Pinning it to that probe's own bound keeps the
        distinction deliberate.

        Pinned to the bound the probe was *called with*, rather than to a
        formula for where the step lands. Where it lands is ``_next_probe``'s
        business and has changed once already, when the first step went from a
        bisection to an interpolation; a test that restates the step rule
        breaks on every such change while testing nothing about this branch.

        Two independent traces, not one refined twice. Refinement now folds its
        probes back into the point set, so a second pass over the same results
        object is seeded by the first pass's answer and converges immediately
        -- which is the feature working, but it leaves no second probe to fail
        on and the call-count assertion above would fire.

        """
        solver = self._sloped_solver()
        converged = none_throws(solver.refine_knee(solver.trace(num_points=10)))

        results = solver.trace(num_points=10)
        real = solver._probe
        calls: list[float] = []

        def fail_on_the_second(bound: float) -> object:
            calls.append(bound)
            return None if len(calls) > 1 else real(bound)

        with mock.patch.object(solver, "_probe", fail_on_the_second):
            aborted = solver.refine_knee(results)

        self.assertEqual(len(calls), 2)  # one good probe, then the failure
        self.assertIsNotNone(aborted)
        aux = [float(c.objectives[1]) for c in results.corners]
        self.assertAlmostEqual(float(none_throws(aborted).bounds[0]), calls[0])
        self.assertGreater(calls[0], min(aux))
        self.assertLess(calls[0], max(aux))
        # And not the converged answer, so the abort branch really ran.
        self.assertNotAlmostEqual(
            float(none_throws(aborted).objectives[1]),
            float(converged.objectives[1]),
        )

    def test_refined_knee_sits_where_the_tangent_parallels_the_chord(self) -> None:
        """That equality is what maximizing distance to the chord means."""
        solver = self._sloped_solver()
        results = solver.trace(num_points=8)
        pri, aux = results.primary_objective, 1 - results.primary_objective
        c0, c1 = results.corners
        m_chord = (float(c1.objectives[pri]) - float(c0.objectives[pri])) / (
            float(c1.objectives[aux]) - float(c0.objectives[aux])
        )

        knee = none_throws(solver.refine_knee(results))

        norm = float(np.linalg.norm(self.d))
        f1 = float(knee.objectives[aux])
        slope = -(norm - np.sqrt(f1)) / np.sqrt(f1)
        self.assertAlmostEqual(slope, m_chord, places=3)

    def test_refined_knee_does_not_move_with_the_grid(self) -> None:
        """The whole point: the answer stops depending on where we sampled."""
        solver = self._sloped_solver()
        coarse = none_throws(solver.refine_knee(solver.trace(num_points=5)))
        fine = none_throws(solver.refine_knee(solver.trace(num_points=25)))

        np.testing.assert_allclose(
            coarse.objectives, fine.objectives, rtol=1e-3, atol=1e-6
        )

    def test_a_straight_frontier_reports_no_knee(self) -> None:
        """Saying "undetermined" beats returning noise that looks precise."""
        solver = self._sloped_solver()
        results = solver.trace(num_points=8)
        results.tangent_slopes = np.full(len(results.points), -1.0)

        self.assertIsNone(solver.refine_knee(results))

    def test_without_slopes_there_is_nothing_to_root_find_on(self) -> None:
        """The base optimizer supplies none, so it declines rather than guesses."""
        results = BallConstrainedNorm(d=self.d).trace(num_points=6)

        self.assertIsNone(results.tangent_slopes)
        self.assertIsNone(BallConstrainedNorm(d=self.d).refine_knee(results))


class ReciprocalFrontier(MultiObjectiveOptimizer):
    r"""An analytic frontier that is convex in both knee spaces and asymmetric.

    The decision variable is the auxiliary objective itself, and

        f0(phi) = 1 / phi^2,   f1(phi) = phi,   phi in [phi_min, phi_max].

    Chosen deliberately over ``BallConstrainedNorm``, whose frontier
    ``f0 = (||d|| - sqrt(f1))^2`` is symmetric under exchanging the axes and
    therefore puts the knee at the same point in every knee space -- a fixture
    that cannot tell a working transform from an ignored one. (The same trap
    as the transposed-envelope bug: equal-range axes hide orientation errors.)

    Both value functions are convex, which is the precondition ``"root"``
    needs: ``sqrt(f0) = 1 / phi`` is convex decreasing, and the primary
    objective is its square.

    """

    def __init__(self, phi_min: float = 1.0, phi_max: float = 10.0) -> None:
        super().__init__(primary_objective=0)
        self.phi_min = phi_min
        self.phi_max = phi_max

    def _at(self, phi: float) -> SolveResult:
        phi = min(max(phi, self.phi_min), self.phi_max)
        x = np.array([phi])
        return _make_ipm_result(x, 1.0 / phi**2)

    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        return self._at(float(bounds[0]))

    def minimize_objective(
        self,
        objective_index: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SolveResult:
        # f0 = 1/phi^2 is smallest at the largest phi; f1 = phi at the smallest.
        return self._at(self.phi_max if objective_index == 0 else self.phi_min)

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        phi = float(x[0])
        return np.array([1.0 / phi**2, phi])

    def tangent_slopes(self, results: FrontierResults) -> npt.NDArray[np.float64]:
        """d f0 / d phi = -2 / phi^3, known in closed form."""
        return np.array([-2.0 / float(p.objectives[1]) ** 3 for p in results.points])


class KneeSpaceTest(unittest.TestCase):
    r"""The knee can be measured in the primary objective or its square root.

    Closed-form answers on :class:`ReciprocalFrontier`, ``phi`` in [1, 10]:

      - primary space: chord slope ``(1 - 0.01) / (1 - 10) = -0.11``, and
        ``f0'(phi) = -2/phi^3``, so the tangent parallels it at
        ``phi = (2/0.11)^(1/3) = 2.632``.
      - root space: ``sqrt(f0) = 1/phi``, chord slope
        ``(1 - 0.1) / (1 - 10) = -0.1``, derivative ``-1/phi^2``, so the
        tangent parallels it at ``phi = sqrt(10) = 3.162``.

    The two differ by 20% of the bracket, which is what makes this fixture
    able to fail.

    """

    PRIMARY_KNEE: float = (2.0 / 0.11) ** (1.0 / 3.0)
    ROOT_KNEE: float = float(np.sqrt(10.0))

    def test_the_default_is_the_primary_objective(self) -> None:
        """Existing callers must keep the knee they had."""
        results = ReciprocalFrontier().trace(num_points=40)
        self.assertEqual(
            results.knee().objectives[1], results.knee("primary").objectives[1]
        )

    def test_the_root_space_moves_the_sampled_knee(self) -> None:
        """A transform that is ignored would return the same point."""
        results = ReciprocalFrontier().trace(num_points=40)
        primary = float(results.knee("primary").objectives[1])
        root = float(results.knee("root").objectives[1])

        self.assertNotAlmostEqual(primary, root)
        # Looser in the auxiliary objective, which is the direction the
        # stretch at the low-primary end of the axis implies.
        self.assertGreater(root, primary)
        self.assertAlmostEqual(primary, self.PRIMARY_KNEE, delta=0.25)
        self.assertAlmostEqual(root, self.ROOT_KNEE, delta=0.25)

    def test_the_root_knee_maximizes_distance_to_the_root_chord(self) -> None:
        """Checked against a hand-rolled max-chord in transformed coordinates."""
        solver = ReciprocalFrontier()
        results = solver.trace(num_points=40)
        c0, c1 = results.corners[0], results.corners[1]
        p0 = np.array([np.sqrt(c0.objectives[0]), c0.objectives[1]])
        p1 = np.array([np.sqrt(c1.objectives[0]), c1.objectives[1]])
        v = p1 - p0
        normal = np.array([-v[1], v[0]]) / float(np.linalg.norm(v))
        distances = [
            abs(
                float(
                    np.dot(
                        np.array([np.sqrt(p.objectives[0]), p.objectives[1]]) - p0,
                        normal,
                    )
                )
            )
            for p in results.points
        ]
        expected = results.points[int(np.argmax(distances))]

        self.assertEqual(results.knee("root").objectives[1], expected.objectives[1])

    def test_refinement_solves_for_the_root_space_knee(self) -> None:
        """Bisection in root space converges on the closed-form answer."""
        solver = ReciprocalFrontier()
        results = solver.trace(num_points=10)
        knee = none_throws(solver.refine_knee(results, knee_space="root"))

        # Much tighter than the sampled grid can manage (its spacing is 0.9)
        # -- that is the point of refining rather than taking the argmax. Not
        # tighter still: the search converges on the SLOPE paralleling the
        # chord to 1e-3 relative, so the position it stops at inherits
        # whatever that implies and is not itself the quantity being driven
        # to tolerance.
        self.assertAlmostEqual(float(knee.objectives[1]), self.ROOT_KNEE, delta=5e-3)

    def test_refinement_in_each_space_lands_on_its_own_chord(self) -> None:
        """The two searches disagree, and each is right in its own coordinates."""
        solver = ReciprocalFrontier()
        results = solver.trace(num_points=10)
        primary = none_throws(solver.refine_knee(results))
        root = none_throws(solver.refine_knee(results, knee_space="root"))

        # Same slope-not-position tolerance as above. The separation the
        # second assertion checks is 0.53, a hundred times the slack here.
        self.assertAlmostEqual(
            float(primary.objectives[1]), self.PRIMARY_KNEE, delta=5e-3
        )
        self.assertGreater(float(root.objectives[1]), float(primary.objectives[1]))

    def test_the_root_slope_is_the_chain_rule_on_the_primary_slope(self) -> None:
        """``_slope_at`` must divide by ``2 sqrt(f0)``, not report the raw dual."""
        solver = ReciprocalFrontier()
        results = solver.trace(num_points=10)
        point = results.points[len(results.points) // 2]
        raw = none_throws(solver._slope_at(results, point))
        rooted = none_throws(solver._slope_at(results, point, "root"))

        self.assertAlmostEqual(
            rooted, raw / (2.0 * float(np.sqrt(point.objectives[0]))), places=12
        )

    def test_an_unknown_knee_space_is_rejected(self) -> None:
        """A typo must not fall through to the identity transform."""
        results = ReciprocalFrontier().trace(num_points=5)
        with self.assertRaisesRegex(ValueError, "knee_space"):
            # A caller without a type checker can pass any string; that is the test.
            results.knee("sqrt")  # type: ignore[arg-type]

    def test_the_root_space_rejects_a_negative_primary_objective(self) -> None:
        """Square-rooting a negative objective would produce a silent nan."""
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _knee_space_transform(-1.0, "root")

    def test_a_zero_primary_objective_reports_a_vertical_tangent(self) -> None:
        """Infinity is filtered by the finite-slope checks; nan would not be."""
        value, derivative = _knee_space_transform(0.0, "root")
        self.assertEqual(value, 0.0)
        self.assertEqual(derivative, np.inf)


class RefinedKneeIsRecordedTest(unittest.TestCase):
    r"""``refine_knee`` leaves its answer on the frontier it ran on.

    The refined knee is off-grid by construction, so a consumer holding only
    the results object cannot recover it and will reach for the sampled
    :meth:`FrontierResults.knee` instead. That substitutes the nearest grid
    point for the answer -- which is exactly the quantization ``refine_knee``
    exists to remove -- and nothing about the result looks wrong.

    ``knee_refinement_ran`` carries the other half: "no refinement has been
    run" and "a refinement ran and found no bend" are both ``refined_knee is
    None`` but call for opposite behaviour downstream.

    """

    def _solver(self) -> BallConstrainedNorm:
        return RefineKneeTest._sloped_solver()

    def test_a_fresh_frontier_has_not_been_refined(self) -> None:
        """The flag must start false, or every consumer reads a phantom."""
        results = self._solver().trace(num_points=6)

        self.assertFalse(results.knee_refinement_ran)
        self.assertIsNone(results.refined_knee)

    def test_the_answer_is_recorded_as_well_as_returned(self) -> None:
        solver = self._solver()
        results = solver.trace(num_points=6)

        knee = solver.refine_knee(results)

        self.assertIsNotNone(knee)
        self.assertTrue(results.knee_refinement_ran)
        self.assertIs(results.refined_knee, knee)

    def test_the_recorded_knee_is_off_the_sweep_grid(self) -> None:
        """Otherwise recording it would be a pointless indirection.

        The refined knee lands strictly between two sweep samples -- if it
        always coincided with one, a consumer could keep using the sampled
        argmax and never notice the difference.

        It does then *become* a sampled point, because refinement folds its
        probes back into the frontier, and after that ``knee()`` returns it.
        That is not the premise collapsing; it is the two answers agreeing
        because the better one is now in the set the worse one is chosen from.
        Both halves are asserted, against the grid as it stood before the
        search ran.

        """
        solver = self._solver()
        results = solver.trace(num_points=6)
        grid = list(results.points)

        solver.refine_knee(results)
        refined = results.refined_knee
        assert refined is not None

        self.assertFalse(
            any(np.allclose(refined.objectives, p.objectives, atol=1e-8) for p in grid),
            "the refined knee fell on a sweep sample, so this fixture cannot "
            "tell refinement from argmax",
        )
        # By identity: FrontierPoint is a plain dataclass, so ``==`` on two of
        # them compares ndarrays elementwise and ``in`` raises on the result.
        self.assertTrue(any(p is refined for p in results.points))
        self.assertTrue(
            np.allclose(results.knee().objectives, refined.objectives, atol=1e-8)
        )

    def test_an_early_exit_records_that_it_looked(self) -> None:
        """A refusal is a finding, and the flag is how it travels.

        Without slopes the search cannot start. ``refined_knee`` stays None,
        but ``knee_refinement_ran`` must still flip, so a consumer can tell
        this from a frontier nobody has asked about.

        """
        solver = self._solver()
        results = solver.trace(num_points=6)
        results.tangent_slopes = None

        self.assertIsNone(solver.refine_knee(results))
        self.assertTrue(results.knee_refinement_ran)
        self.assertIsNone(results.refined_knee)

    def test_a_declined_knee_is_recorded_too(self) -> None:
        """A frontier too straight to have a knee still counts as looked at."""
        solver = self._solver()
        results = solver.trace(num_points=6)

        # A spread of exactly zero is the straight-frontier branch.
        results.tangent_slopes = np.full(len(results.points), -1.0)

        self.assertIsNone(solver.refine_knee(results))
        self.assertTrue(results.knee_refinement_ran)
        self.assertIsNone(results.refined_knee)


class SeededSecantRefinementTest(unittest.TestCase):
    r"""The root-find uses the slope magnitudes, not just their signs.

    Bisection reads one bit per solve. Every solve on this frontier reports
    its own tangent, and the sweep has already paid for one at every sample,
    so both the starting interval and the step inside it can be had from
    information already in hand.

    What must NOT change is the answer. These tests pin the probe count
    falling and the located knee staying put, because a faster search that
    lands somewhere else is not a faster search.

    """

    @staticmethod
    def _solver() -> BallConstrainedNorm:
        return RefineKneeTest._sloped_solver()

    def _probe_count(
        self, seeded: bool, solver: MultiObjectiveOptimizer | None = None
    ) -> tuple[int, float]:
        """Probes used and knee found, with the seeding either on or off."""
        solver = self._solver() if solver is None else solver
        results = solver.trace(num_points=10)
        stack = contextlib.ExitStack()
        with stack:
            if not seeded:
                stack.enter_context(
                    mock.patch.object(
                        type(solver),
                        "_seed_bracket",
                        staticmethod(lambda s, m, lo, hi: (lo, None, hi, None)),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        type(solver),
                        "_next_probe",
                        staticmethod(lambda lo, flo, hi, fhi: 0.5 * (lo + hi)),
                    )
                )
            probed = stack.enter_context(
                mock.patch.object(type(solver), "_probe", wraps=solver._probe)
            )
            knee = solver.refine_knee(results)
        return probed.call_count, float(none_throws(knee).objectives[1])

    def test_it_costs_fewer_solves_than_bisection(self) -> None:
        """Measured on ``ReciprocalFrontier``, and NOT on the ball fixture.

        The ball fixture cannot show this, and the reason is worth recording
        rather than rediscovering. Its knee sits at 6.25 on a corner span of
        [0, 25] -- exactly a quarter of the way -- so bisection's second
        midpoint lands on the answer to machine precision. Two probes, by
        coincidence of the arithmetic. No search can beat that, and a fixture
        whose knee falls on a dyadic fraction of its own span is not evidence
        about frontiers in general.

        ``ReciprocalFrontier``'s knee is at ``(2/0.11)**(1/3)``, which is not
        a convenient fraction of anything.

        """
        seeded, _ = self._probe_count(seeded=True, solver=ReciprocalFrontier())
        plain, _ = self._probe_count(seeded=False, solver=ReciprocalFrontier())

        self.assertLess(seeded, plain)

    def test_the_seeded_search_finds_the_closed_form_knee(self) -> None:
        """Speed is only interesting if the destination is unchanged."""
        _, seeded = self._probe_count(seeded=True, solver=ReciprocalFrontier())

        self.assertAlmostEqual(seeded, KneeSpaceTest.PRIMARY_KNEE, delta=5e-3)

    def test_it_finds_the_same_knee_as_bisection(self) -> None:
        """The whole point. A different answer would make the speed worthless.

        Compared at the tolerance the search itself converges to rather than
        exactly: the two stop at different iterates of the same root, so they
        agree to the stopping rule and no further.

        """
        _, seeded = self._probe_count(seeded=True)
        _, plain = self._probe_count(seeded=False)

        self.assertAlmostEqual(seeded, plain, delta=5e-3 * abs(plain))

    def test_the_first_probe_lands_inside_the_seeded_interval(self) -> None:
        """The seed has to reach the search, not merely exist beside it.

        Every other test here passes if ``_refine_knee_search`` ignores
        ``_seed_bracket`` and starts from the corner span, because
        interpolation alone still converges and still converges to the same
        root. This is the one that notices, and it notices the way a reader
        would: the first solve happens somewhere the old search could not have
        put it.

        """
        solver = ReciprocalFrontier()
        results = solver.trace(num_points=10)
        aux = [float(c.objectives[1]) for c in results.corners]
        m_chord = self._chord_slope(results)
        lo, _, hi, _ = solver._seed_bracket(
            solver._knee_samples(results, "primary"), m_chord, min(aux), max(aux)
        )

        bounds: list[float] = []
        real = solver._probe

        def record(bound: float) -> object:
            bounds.append(bound)
            return real(bound)

        with mock.patch.object(solver, "_probe", record):
            solver.refine_knee(results)

        self.assertTrue(bounds, "the search never probed")
        self.assertGreaterEqual(bounds[0], lo)
        self.assertLessEqual(bounds[0], hi)
        # And not where the unseeded search would have gone first.
        self.assertNotAlmostEqual(bounds[0], 0.5 * (min(aux) + max(aux)))

    def test_the_seed_brackets_the_root(self) -> None:
        """Narrower than the corner span, and still containing the answer."""
        solver = self._solver()
        results = solver.trace(num_points=10)
        knee = none_throws(solver.refine_knee(results))
        target = float(knee.objectives[1])

        aux = [float(c.objectives[1]) for c in results.corners]
        m_chord = self._chord_slope(results)
        lo, _, hi, _ = solver._seed_bracket(
            solver._knee_samples(results, "primary"), m_chord, min(aux), max(aux)
        )

        self.assertLess(hi - lo, max(aux) - min(aux))
        self.assertLessEqual(lo, target)
        self.assertLessEqual(target, hi)

    @staticmethod
    def _chord_slope(results: FrontierResults) -> float:
        c0, c1 = results.corners[0], results.corners[1]
        return (float(c1.objectives[0]) - float(c0.objectives[0])) / (
            float(c1.objectives[1]) - float(c0.objectives[1])
        )

    def test_a_bracket_with_an_unknown_end_bisects(self) -> None:
        """Corners report no slope, so a root outside the samples has one end.

        Interpolating through a missing residual is not possible, and guessing
        one is worse than halving. The midpoint is the honest step until both
        ends are known.

        """
        self.assertAlmostEqual(
            MultiObjectiveOptimizer._next_probe(2.0, None, 10.0, 3.0), 6.0
        )
        self.assertAlmostEqual(
            MultiObjectiveOptimizer._next_probe(2.0, -1.0, 10.0, None), 6.0
        )

    def test_an_interpolated_step_lands_at_the_linear_root(self) -> None:
        """Residuals -1 and +3 over [0, 4] put the root a quarter of the way."""
        self.assertAlmostEqual(
            MultiObjectiveOptimizer._next_probe(0.0, -1.0, 4.0, 3.0), 1.0
        )

    def test_a_step_that_would_leave_the_bracket_falls_back(self) -> None:
        """Same-sign residuals are not a bracket; the interpolant extrapolates.

        Duals off an aborted barrier solve can produce exactly this. Bisection
        is wrong-ish here too -- there may be no root inside -- but it cannot
        run away, which is the property worth keeping.

        """
        self.assertAlmostEqual(
            MultiObjectiveOptimizer._next_probe(0.0, 1.0, 4.0, 3.0), 2.0
        )

    def test_a_flat_residual_bisects(self) -> None:
        """Equal endpoint residuals give the interpolant no slope to use."""
        self.assertAlmostEqual(
            MultiObjectiveOptimizer._next_probe(0.0, 2.0, 4.0, 2.0), 2.0
        )

    def test_a_straight_frontier_still_reports_no_knee(self) -> None:
        """Seeding must not talk the search into answering an empty question."""
        solver = self._solver()
        results = solver.trace(num_points=10)
        results.tangent_slopes = np.full(len(results.points), -1.0)

        self.assertIsNone(solver.refine_knee(results))


class RefinementProbesJoinTheFrontierTest(unittest.TestCase):
    r"""A probe is a frontier solve, so it belongs in the point set.

    The refinement probes are solved at bounds the grid never visited, with
    the same optimizer and the same objectives, and they cluster where the
    curve bends hardest -- which is precisely where a uniform grid samples
    worst. Reading their slopes and discarding the points left the frontier
    least informative exactly where it had just been proved to matter, and
    left ``n_attempted`` reporting a cost lower than the one paid.

    Folding them in is unconditional, so these tests are about the invariants
    that folding must not break: sort order, the documented index alignment
    between ``points`` and ``tangent_slopes``, the units those slopes are in,
    and the two counters.

    """

    def _solver(self) -> BallConstrainedNorm:
        return RefineKneeTest._sloped_solver()

    @staticmethod
    def _off_grid_probe(
        solver: BallConstrainedNorm, results: FrontierResults
    ) -> FrontierPoint:
        """A fresh solve at a bound strictly between two swept ones.

        The corners carry infinite bounds, so halving one of those lands back
        on the unconstrained minimum -- an existing point, which the dedup
        rejects before any of the behaviour under test is reached.

        """
        bounds = sorted(
            float(pt.bounds[0]) for pt in results.points if np.isfinite(pt.bounds[0])
        )
        return none_throws(solver._probe(0.5 * (bounds[0] + bounds[1])))

    def test_the_probes_are_in_the_point_set(self) -> None:
        solver = self._solver()
        results = solver.trace(num_points=6)
        before = list(results.points)

        solver.refine_knee(results)

        self.assertGreater(len(results.points), len(before))
        added = [p for p in results.points if all(p is not q for q in before)]
        self.assertTrue(added, "refinement added no points")
        for probe in added:
            self.assertFalse(
                any(
                    np.allclose(probe.objectives, q.objectives, atol=1e-8)
                    for q in before
                ),
                "a folded probe duplicates a grid point",
            )

    def test_the_point_set_stays_sorted_by_the_primary_objective(self) -> None:
        """``envelope`` interpolates in this order; unsorted, it draws nonsense."""
        solver = self._solver()
        results = solver.trace(num_points=6)

        solver.refine_knee(results)

        pri = results.primary_objective
        values = [float(p.objectives[pri]) for p in results.points]
        self.assertEqual(values, sorted(values))

    def test_the_slopes_stay_aligned_with_the_points(self) -> None:
        solver = self._solver()
        results = solver.trace(num_points=6)

        solver.refine_knee(results)

        slopes = none_throws(results.tangent_slopes)
        self.assertEqual(len(slopes), len(results.points))
        for point, slope in zip(results.points, slopes, strict=False):
            expected = solver._raw_slope_at(results, point)
            if expected is None:
                self.assertFalse(np.isfinite(slope))
            else:
                self.assertAlmostEqual(float(slope), expected, places=9)

    def test_a_folded_slope_is_in_minimization_coordinates(self) -> None:
        """Not in the knee space the search happened to be working in.

        The two differ by the chain-rule factor of the ``root`` transform, so
        a probe folded in carrying the scaled number would sit in the array
        one entry out of units from its neighbours -- visible as a slightly
        wrong error band on the envelope, and as nothing at all anywhere else.

        """
        solver = self._solver()
        results = solver.trace(num_points=6)

        knee = none_throws(solver.refine_knee(results, knee_space="root"))

        index = next(i for i, p in enumerate(results.points) if p is knee)
        folded = float(none_throws(results.tangent_slopes)[index])
        raw = none_throws(solver._raw_slope_at(results, knee))
        scaled = none_throws(solver._rescale_slope(raw, knee, "root"))

        self.assertAlmostEqual(folded, raw, places=9)
        self.assertNotAlmostEqual(raw, scaled, places=6)

    def test_every_probe_is_counted_as_attempted(self) -> None:
        solver = self._solver()
        results = solver.trace(num_points=6)
        attempted = results.n_attempted
        probes = 0

        real = solver._probe

        def count(bound: float) -> object:
            nonlocal probes
            probes += 1
            return real(bound)

        with mock.patch.object(solver, "_probe", count):
            solver.refine_knee(results)

        self.assertGreater(probes, 0)
        self.assertEqual(results.n_attempted, attempted + probes)

    def test_a_failed_probe_is_counted_as_skipped_and_adds_no_point(self) -> None:
        solver = self._solver()
        results = solver.trace(num_points=6)
        attempted, skipped = results.n_attempted, results.n_skipped
        points = len(results.points)

        with mock.patch.object(solver, "_probe", lambda bound: None):
            solver.refine_knee(results)

        self.assertEqual(results.n_attempted, attempted + 1)
        self.assertEqual(results.n_skipped, skipped + 1)
        self.assertEqual(len(results.points), points)

    def test_a_probe_on_top_of_an_existing_point_is_not_duplicated(self) -> None:
        """Near convergence a probe landing on a grid point is expected."""
        solver = self._solver()
        results = solver.trace(num_points=6)
        points = len(results.points)

        results.absorb_probe(results.points[2], 0.0)

        self.assertEqual(len(results.points), points)
        self.assertEqual(len(none_throws(results.tangent_slopes)), points)

    def test_absorbing_into_a_frontier_without_slopes_keeps_it_without_them(
        self,
    ) -> None:
        """The base optimizer reports no slopes; folding must not invent any."""
        solver = self._solver()
        results = solver.trace(num_points=6)
        probe = self._off_grid_probe(solver, results)
        results.tangent_slopes = None
        points = len(results.points)

        results.absorb_probe(probe, None)

        self.assertEqual(len(results.points), points + 1)
        self.assertIsNone(results.tangent_slopes)

    def test_a_misaligned_slope_array_refuses_the_point(self) -> None:
        """A frontier one sample coarser beats one whose slopes are off by one.

        There is no insertion into an already-misaligned array that restores
        the alignment, so the only choices are to decline the point or to
        corrupt every consumer of the slopes. Declining is loud in the log and
        silent everywhere else.

        """
        solver = self._solver()
        results = solver.trace(num_points=6)
        probe = self._off_grid_probe(solver, results)
        results.tangent_slopes = none_throws(results.tangent_slopes)[:-1]
        points = len(results.points)

        with self.assertLogs("cvxium.multi_objective", level="WARNING"):
            results.absorb_probe(probe, -1.0)

        self.assertEqual(len(results.points), points)
