"""Tests for MultiObjectiveOptimizer and FrontierResults."""

import numpy as np
import numpy.typing as npt
import pytest

from cvxium import InteriorPointMethodResult
from cvxium.exceptions import OptimizationError
from cvxium.multi_objective import (
    FrontierPoint,
    FrontierResults,
    MultiObjectiveOptimizer,
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


def _make_ipm_result(
    x: npt.NDArray[np.float64], obj: float
) -> InteriorPointMethodResult:
    """Construct a minimal InteriorPointMethodResult for testing."""
    return InteriorPointMethodResult(
        solution=x.copy(),
        objective_value=obj,
        dual_value=obj,
        equality_multipliers=np.array([]),
        inequality_multipliers=np.array([0.0]),
        suboptimality=0.0,
        duality_gaps=[0.0],
        nits=1,
        inner_nits=[1],
        inner_suboptimalities=[[0.0]],
        status=0,
        message="analytical",
    )


class BallConstrainedNorm(MultiObjectiveOptimizer):
    r"""Analytical two-objective solver respecting primary_objective.

    Objectives:
        f0(x) = ||x||^2
        f1(x) = ||x - d||^2

    When primary_objective=0: solve_with_bounds([phi]) minimises f0 s.t. f1 <= phi.
    When primary_objective=1: solve_with_bounds([phi]) minimises f1 s.t. f0 <= phi.
    """

    def __init__(self, d: npt.NDArray[np.float64], primary_objective: int = 0) -> None:
        super().__init__(primary_objective=primary_objective)
        self.d = d
        self._d_norm = float(np.linalg.norm(d))

    def solve_with_bounds(
        self, bounds: npt.NDArray[np.float64]
    ) -> InteriorPointMethodResult:
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

    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
        # minimize f0 → x=0; minimize f1 → x=d
        x = np.zeros_like(self.d) if objective_index == 0 else self.d.copy()
        return _make_ipm_result(x, float(np.dot(x, x)))

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array([float(np.dot(x, x)), float(np.dot(x - self.d, x - self.d))])


class SeparableBallConstraints(MultiObjectiveOptimizer):
    r"""Analytical three-objective solver (separable by dimension).

    Decision variable x in R^2.
    Objectives:
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
        self, bounds: npt.NDArray[np.float64]
    ) -> InteriorPointMethodResult:
        x0 = self._solve_1d(float(bounds[0]))
        x1 = self._solve_1d(float(bounds[1]))
        x = np.array([x0, x1])
        return _make_ipm_result(x, float(np.dot(x, x)))

    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
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

    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
        if objective_index == 1:
            raise OptimizationError("synthetic failure", 1.0, np.zeros(1))
        return super().minimize_objective(objective_index)


# ---------------------------------------------------------------------------
# Tests: two objectives, primary_objective=0
# ---------------------------------------------------------------------------


class TestTwoObjectiveFrontier:
    """Tests for a two-objective optimizer."""

    @pytest.fixture
    def solver(self) -> BallConstrainedNorm:
        rng = np.random.default_rng(42)
        d = rng.standard_normal(5)
        return BallConstrainedNorm(d=d)

    def test_trace_returns_points(self, solver: BallConstrainedNorm) -> None:
        """trace() returns a non-empty FrontierResults."""
        results = solver.trace(num_points=10)
        assert isinstance(results, FrontierResults)
        assert len(results.points) > 0

    def test_corners_computed(self, solver: BallConstrainedNorm) -> None:
        """Two corner points are computed, one per objective."""
        results = solver.trace(num_points=5)
        assert len(results.corners) == 2

    def test_primary_corner_minimizes_f0(self, solver: BallConstrainedNorm) -> None:
        """Corner 0 (primary) achieves f0 near zero."""
        results = solver.trace(num_points=5)
        assert results.corners[0].objectives[0] == pytest.approx(0.0, abs=1e-10)

    def test_auxiliary_corner_minimizes_f1(self, solver: BallConstrainedNorm) -> None:
        """Corner 1 (auxiliary) achieves f1 near zero."""
        results = solver.trace(num_points=5)
        assert results.corners[1].objectives[1] == pytest.approx(0.0, abs=1e-10)

    def test_frontier_monotone(self, solver: BallConstrainedNorm) -> None:
        """As phi (auxiliary bound) increases, f0 is non-increasing."""
        results = solver.trace(num_points=20)
        interior = [p for p in results.points if not np.any(np.isinf(p.bounds))]
        interior.sort(key=lambda p: float(p.bounds[0]))
        f0_values = [p.objectives[0] for p in interior]
        for i in range(len(f0_values) - 1):
            assert f0_values[i] >= f0_values[i + 1] - 1e-10

    def test_frontier_matches_analytical(self, solver: BallConstrainedNorm) -> None:
        """Frontier points match the analytical formula f0 = (||d|| - sqrt(f1))^2."""
        d_norm = solver._d_norm
        results = solver.trace(num_points=20)
        for p in results.points:
            f0, f1 = float(p.objectives[0]), float(p.objectives[1])
            expected_f0 = max(0.0, (d_norm - np.sqrt(max(f1, 0.0))) ** 2)
            assert f0 == pytest.approx(expected_f0, abs=1e-8)

    def test_knee_is_a_frontier_point(self, solver: BallConstrainedNorm) -> None:
        """knee() returns one of the points in results.points (by identity)."""
        results = solver.trace(num_points=10)
        knee = results.knee()
        assert any(knee is p for p in results.points)

    def test_knee_not_at_extreme(self, solver: BallConstrainedNorm) -> None:
        """Knee is not the same as either corner (it should be in the interior)."""
        results = solver.trace(num_points=20)
        knee = results.knee()
        assert knee.objectives[0] > 1e-6
        assert knee.objectives[1] > 1e-6

    def test_primary_objective_stored(self, solver: BallConstrainedNorm) -> None:
        """FrontierResults records primary_objective correctly."""
        results = solver.trace(num_points=5)
        assert results.primary_objective == 0

    def test_diagnostics(self, solver: BallConstrainedNorm) -> None:
        """n_attempted and n_skipped are tracked correctly."""
        num_points = 7
        results = solver.trace(num_points=num_points)
        # For N=2 there is 1 auxiliary, so grid has num_points points.
        assert results.n_attempted == num_points
        # This solver never raises, so nothing is skipped.
        assert results.n_skipped == 0

    def test_num_points_validation(self, solver: BallConstrainedNorm) -> None:
        """trace() raises ValueError for num_points < 1."""
        with pytest.raises(ValueError, match="num_points must be >= 1"):
            solver.trace(num_points=0)
        with pytest.raises(ValueError, match="num_points must be >= 1"):
            solver.trace(num_points=-5)

    def test_deduplication(self, solver: BallConstrainedNorm) -> None:
        """No two points in results.points share an objective vector."""
        results = solver.trace(num_points=20)
        for i, p in enumerate(results.points):
            for q in results.points[:i]:
                assert not np.allclose(p.objectives, q.objectives, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: two objectives, primary_objective=1
# ---------------------------------------------------------------------------


class TestTwoObjectiveFrontierAlternatePrimary:
    """Correctness tests when the non-default objective is primary."""

    @pytest.fixture
    def solver(self) -> BallConstrainedNorm:
        rng = np.random.default_rng(99)
        d = rng.standard_normal(4)
        return BallConstrainedNorm(d=d, primary_objective=1)

    def test_primary_corner_minimizes_f1(self, solver: BallConstrainedNorm) -> None:
        """With primary=1, corner 0 minimizes f1 (||x-d||^2) to zero."""
        results = solver.trace(num_points=5)
        assert results.corners[0].objectives[1] == pytest.approx(0.0, abs=1e-10)

    def test_auxiliary_corner_minimizes_f0(self, solver: BallConstrainedNorm) -> None:
        """With primary=1, corner 1 (the auxiliary corner) minimizes f0 to zero."""
        results = solver.trace(num_points=5)
        assert results.corners[1].objectives[0] == pytest.approx(0.0, abs=1e-10)

    def test_frontier_monotone(self, solver: BallConstrainedNorm) -> None:
        """As f0 bound increases, f1 (primary) is non-increasing."""
        results = solver.trace(num_points=20)
        interior = [p for p in results.points if not np.any(np.isinf(p.bounds))]
        interior.sort(key=lambda p: float(p.bounds[0]))
        f1_values = [float(p.objectives[1]) for p in interior]
        for i in range(len(f1_values) - 1):
            assert f1_values[i] >= f1_values[i + 1] - 1e-10

    def test_frontier_matches_analytical(self, solver: BallConstrainedNorm) -> None:
        """With primary=1, frontier still satisfies f1 = (||d|| - sqrt(f0))^2."""
        d_norm = solver._d_norm
        results = solver.trace(num_points=20)
        for p in results.points:
            f0, f1 = float(p.objectives[0]), float(p.objectives[1])
            expected_f1 = max(0.0, (d_norm - np.sqrt(max(f0, 0.0))) ** 2)
            assert f1 == pytest.approx(expected_f1, abs=1e-8)

    def test_knee_available(self, solver: BallConstrainedNorm) -> None:
        """knee() works for primary_objective=1."""
        results = solver.trace(num_points=10)
        knee = results.knee()
        assert any(knee is p for p in results.points)

    def test_primary_objective_in_results(self, solver: BallConstrainedNorm) -> None:
        """FrontierResults records primary_objective=1."""
        results = solver.trace(num_points=5)
        assert results.primary_objective == 1


# ---------------------------------------------------------------------------
# Tests: FrontierResults.knee geometry
# ---------------------------------------------------------------------------


class TestKneeGeometry:
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
        with pytest.raises(ValueError, match="No frontier points"):
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
        with pytest.raises(ValueError, match="at least 2 corner points"):
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
        with pytest.raises(ValueError, match="exactly one corner per objective"):
            results.knee()


# ---------------------------------------------------------------------------
# Tests: three objectives
# ---------------------------------------------------------------------------


class TestThreeObjectiveFrontier:
    """Tests for the three-objective separable solver."""

    @pytest.fixture
    def solver(self) -> SeparableBallConstraints:
        return SeparableBallConstraints(primary_objective=0)

    def test_trace_returns_points(self, solver: SeparableBallConstraints) -> None:
        """trace() returns a non-empty FrontierResults for N=3."""
        results = solver.trace(num_points=5)
        assert len(results.points) > 0

    def test_three_corners_computed(self, solver: SeparableBallConstraints) -> None:
        """Three corner points are computed for three objectives."""
        results = solver.trace(num_points=5)
        assert len(results.corners) == 3

    def test_primary_corner_f0_zero(self, solver: SeparableBallConstraints) -> None:
        """Primary corner minimizes f0 to zero."""
        results = solver.trace(num_points=5)
        assert results.corners[0].objectives[0] == pytest.approx(0.0, abs=1e-10)

    def test_aux1_corner_f1_zero(self, solver: SeparableBallConstraints) -> None:
        """First auxiliary corner minimizes f1 to zero."""
        results = solver.trace(num_points=5)
        assert results.corners[1].objectives[1] == pytest.approx(0.0, abs=1e-10)

    def test_aux2_corner_f2_zero(self, solver: SeparableBallConstraints) -> None:
        """Second auxiliary corner minimizes f2 to zero."""
        results = solver.trace(num_points=5)
        assert results.corners[2].objectives[2] == pytest.approx(0.0, abs=1e-10)

    def test_grid_size(self, solver: SeparableBallConstraints) -> None:
        """For N=3 and num_points=k, the grid contributes at most k^2 points."""
        num_points = 4
        results = solver.trace(num_points=num_points)
        assert results.n_attempted == num_points**2

    def test_knee_available(self, solver: SeparableBallConstraints) -> None:
        """knee() returns a FrontierPoint for N=3."""
        results = solver.trace(num_points=5)
        knee = results.knee()
        assert isinstance(knee, FrontierPoint)
        assert any(knee is p for p in results.points)


class TestPlot:
    """Tests for FrontierResults.plot()."""

    @staticmethod
    def test_plot_returns_axes() -> None:
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

    @staticmethod
    def test_plot_no_annotation() -> None:
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

    @staticmethod
    def test_plot_raises_for_3_objectives() -> None:
        import matplotlib

        matplotlib.use("Agg")
        solver = SeparableBallConstraints()
        fr = solver.trace(num_points=3)
        with pytest.raises(NotImplementedError):
            fr.plot()

    @staticmethod
    def test_plot_raises_for_empty_frontier() -> None:
        import matplotlib

        matplotlib.use("Agg")
        fr = FrontierResults(
            points=[],
            corners=[],
            primary_objective=0,
            n_attempted=0,
            n_skipped=0,
        )
        with pytest.raises(ValueError):
            fr.plot()
