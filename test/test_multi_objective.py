"""Tests for MultiObjectiveOptimizer and FrontierResults."""


import numpy as np
import numpy.typing as npt
import pytest

from cvxium import InteriorPointMethodResult
from cvxium.multi_objective import (
    FrontierPoint,
    FrontierResults,
    MultiObjectiveOptimizer,
)

# ---------------------------------------------------------------------------
# Analytical test solver: minimize ||x||^2 s.t. ||x - d||^2 <= phi
#
# Two objectives:
#   f0(x) = ||x||^2          (primary)
#   f1(x) = ||x - d||^2     (auxiliary, bounded by phi)
#
# Analytical solution for given phi:
#   If phi >= ||d||^2: x* = 0         (unconstrained min is feasible)
#   Else:              x* = d * (1 - sqrt(phi) / ||d||)
#
# The frontier is the curve f0 = (||d|| - sqrt(f1))^2 for f1 in [0, ||d||^2].
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
    r"""Analytical two-objective solver.

    Objectives:
        f0(x) = ||x||^2
        f1(x) = ||x - d||^2

    solve_with_bounds([phi]) solves:
        minimize f0(x) s.t. f1(x) <= phi
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
        if phi >= d_norm_sq:
            x = np.zeros_like(self.d)
        else:
            x = self.d * (1.0 - np.sqrt(phi) / self._d_norm)
        return _make_ipm_result(x, float(np.dot(x, x)))

    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
        if objective_index == 0:
            x = np.zeros_like(self.d)
        else:
            x = self.d.copy()
        return _make_ipm_result(x, float(np.dot(x - self.d * objective_index, x - self.d * objective_index)))

    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        f0 = float(np.dot(x, x))
        f1 = float(np.dot(x - self.d, x - self.d))
        return np.array([f0, f1])


class SeparableBallConstraints(MultiObjectiveOptimizer):
    r"""Analytical three-objective solver (separable by dimension).

    Decision variable x in R^2.
    Objectives:
        f0(x) = x[0]^2 + x[1]^2
        f1(x) = (x[0] - 1)^2
        f2(x) = (x[1] - 1)^2

    solve_with_bounds([phi1, phi2]) has separable solution:
        x[k]* = 0 if phi_{k+1} >= 1 else 1 - sqrt(phi_{k+1})
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
        return np.array([
            float(x[0] ** 2 + x[1] ** 2),
            float((x[0] - 1.0) ** 2),
            float((x[1] - 1.0) ** 2),
        ])


# ---------------------------------------------------------------------------
# Tests: two objectives
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
        primary_corner = results.corners[0]
        assert primary_corner.objectives[0] == pytest.approx(0.0, abs=1e-10)

    def test_auxiliary_corner_minimizes_f1(self, solver: BallConstrainedNorm) -> None:
        """Corner 1 (auxiliary) achieves f1 near zero."""
        results = solver.trace(num_points=5)
        aux_corner = results.corners[1]
        assert aux_corner.objectives[1] == pytest.approx(0.0, abs=1e-10)

    def test_frontier_monotone(self, solver: BallConstrainedNorm) -> None:
        """As phi (auxiliary bound) increases, f0 is non-increasing."""
        results = solver.trace(num_points=20)
        # Sort points by bounds (phi value).
        interior_points = [p for p in results.points if not np.any(np.isinf(p.bounds))]
        interior_points.sort(key=lambda p: float(p.bounds[0]))
        f0_values = [p.objectives[0] for p in interior_points]
        # f0 should be non-increasing as the constraint relaxes.
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
        # The knee should not have f0=0 (primary corner) nor f1=0 (aux corner).
        assert knee.objectives[0] > 1e-6
        assert knee.objectives[1] > 1e-6

    def test_primary_objective_stored(self, solver: BallConstrainedNorm) -> None:
        """FrontierResults records primary_objective correctly."""
        results = solver.trace(num_points=5)
        assert results.primary_objective == 0

    @pytest.mark.parametrize("primary", [0, 1])
    def test_primary_objective_constructor(self, primary: int) -> None:
        """primary_objective constructor parameter is respected."""
        d = np.array([1.0, 2.0, 3.0])
        solver = BallConstrainedNorm(d=d, primary_objective=primary)
        assert solver.primary_objective == primary
        results = solver.trace(num_points=5)
        assert results.primary_objective == primary


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
        """For a symmetric curve, knee is near the midpoint of the chord."""
        # Corners at (0, 1) and (1, 0); midpoint of chord at (0.5, 0.5).
        corners = [
            self._make_point([0.0, 1.0]),
            self._make_point([1.0, 0.0]),
        ]
        # Frontier: f0 + f1 = 1 (linear — all points equidistant from chord).
        # Use a curved frontier: f0 = (1 - sqrt(f1))^2 with f1 in [0,1].
        f1_vals = np.linspace(0.0, 1.0, 51)
        points = list(corners) + [
            self._make_point([(1.0 - np.sqrt(f1)) ** 2, f1]) for f1 in f1_vals[1:-1]
        ]
        results = FrontierResults(points=points, corners=corners, primary_objective=0)
        knee = results.knee()
        # The knee should be interior, not at the corners.
        assert knee.objectives[0] > 0.01
        assert knee.objectives[1] > 0.01

    def test_knee_raises_on_empty_points(self) -> None:
        """knee() raises ValueError when points list is empty."""
        corners = [self._make_point([0.0, 1.0]), self._make_point([1.0, 0.0])]
        results = FrontierResults(points=[], corners=corners, primary_objective=0)
        with pytest.raises(ValueError, match="No frontier points"):
            results.knee()

    def test_knee_raises_with_single_corner(self) -> None:
        """knee() raises ValueError when fewer than 2 corners are available."""
        corner = self._make_point([0.0, 1.0])
        point = self._make_point([0.5, 0.5])
        results = FrontierResults(
            points=[corner, point], corners=[corner], primary_objective=0
        )
        with pytest.raises(ValueError, match="at least 2 corner points"):
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
        # corners (3) + up to num_points^2 grid points
        assert len(results.points) <= 3 + num_points**2

    def test_knee_available(self, solver: SeparableBallConstraints) -> None:
        """knee() returns a FrontierPoint for N=3."""
        results = solver.trace(num_points=5)
        knee = results.knee()
        assert isinstance(knee, FrontierPoint)
        assert any(knee is p for p in results.points)
