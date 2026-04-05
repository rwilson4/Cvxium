"""Multi-objective optimization and Pareto frontier tracing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product as itertools_product

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .exceptions import OptimizationError, ProblemInfeasibleError
from .optimization import InteriorPointMethodResult


@dataclass
class FrontierPoint:
    """A single evaluated point on the Pareto frontier.

    Parameters
    ----------
    objectives : array of shape (N,)
        The value of each objective at this point.
    solution : array
        The decision variable at this point.
    bounds : array of shape (N-1,)
        The upper bounds on auxiliary objectives that were imposed when solving.
        Set to ``np.full(N-1, np.inf)`` for corner points where no bounds were
        imposed.
    ipm_result : InteriorPointMethodResult
        The raw result from the underlying solver.

    """

    objectives: npt.NDArray[np.float64]
    solution: npt.NDArray[np.float64]
    bounds: npt.NDArray[np.float64]
    ipm_result: InteriorPointMethodResult


class FrontierResults:
    """Results from tracing a Pareto frontier.

    Parameters
    ----------
    points : list[FrontierPoint]
        All successfully evaluated points, including the corner points.
    corners : list[FrontierPoint]
        The N corner points, one per objective. ``corners[0]`` minimizes the
        primary objective; ``corners[k]`` minimizes the k-th auxiliary objective
        (in the order they appear in ``evaluate_objectives``, skipping the
        primary).
    primary_objective : int
        Index of the primary objective.

    """

    def __init__(
        self,
        points: list[FrontierPoint],
        corners: list[FrontierPoint],
        primary_objective: int,
    ) -> None:
        self.points = points
        self.corners = corners
        self.primary_objective = primary_objective

    def knee(self) -> FrontierPoint:
        """Return the frontier point maximally distant from the corner hyperplane.

        Identifies the "knee in the curve" by finding the point on the frontier
        farthest from the hyperplane that passes through the N corner points
        (where N is the number of objectives). For two objectives this reduces
        to the standard max-chord-distance method.

        Returns
        -------
        FrontierPoint

        Raises
        ------
        ValueError
            If fewer than 2 corner points are available or the frontier is empty.

        """
        if not self.points:
            raise ValueError("No frontier points to evaluate.")

        n_corners = len(self.corners)
        if n_corners < 2:
            raise ValueError(
                f"Need at least 2 corner points for knee calculation; got {n_corners}."
            )

        corner_objs = np.array([c.objectives for c in self.corners])  # (N, N)

        # Normal to the hyperplane through the N corners.
        # Translate to origin using the first corner, then find the null space
        # of the matrix whose rows are the translated corner vectors.
        p0 = corner_objs[0]
        vecs = corner_objs[1:] - p0  # (N-1, N)

        if n_corners == 2:
            # 2D: rotate the single vector 90 degrees.
            v = vecs[0]
            normal = np.array([-v[1], v[0]], dtype=float)
        else:
            # General: last right singular vector spans the null space.
            _, _, vt = linalg.svd(vecs)
            normal = vt[-1].astype(float)

        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("Corner points are degenerate (all identical).")
        normal /= norm

        distances = np.array(
            [abs(float(np.dot(p.objectives - p0, normal))) for p in self.points]
        )
        return self.points[int(distances.argmax())]


class MultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective Pareto frontier tracing.

    To use this class, subclass it and implement:

    - :meth:`solve_with_bounds` — minimize the primary objective subject to
      upper bounds on each auxiliary objective.
    - :meth:`minimize_objective` — minimize a single objective with no bounds
      on the others.
    - :meth:`evaluate_objectives` — return all N objective values at a point.

    Then call :meth:`trace` to sweep over a uniform grid of auxiliary bounds
    and collect the resulting Pareto-optimal points.

    Parameters
    ----------
    primary_objective : int, default=0
        Index into the vector returned by :meth:`evaluate_objectives` that
        identifies the objective to minimize during the frontier sweep. All
        other objectives become auxiliary and receive parametric upper bounds.

    """

    def __init__(self, primary_objective: int = 0) -> None:
        self.primary_objective = primary_objective

    @abstractmethod
    def solve_with_bounds(
        self, bounds: npt.NDArray[np.float64]
    ) -> InteriorPointMethodResult:
        """Minimize the primary objective subject to auxiliary upper bounds.

        Parameters
        ----------
        bounds : array of shape (N-1,)
            ``bounds[j]`` is the upper bound imposed on auxiliary objective
            ``j`` (in the order objectives appear in :meth:`evaluate_objectives`,
            skipping the primary).

        Returns
        -------
        InteriorPointMethodResult

        """
        ...

    @abstractmethod
    def minimize_objective(
        self, objective_index: int
    ) -> InteriorPointMethodResult:
        """Minimize a single objective with no bounds on the others.

        Used by :meth:`trace` to compute the N corner points of the frontier
        and to auto-detect the sweep range for each auxiliary objective.

        Parameters
        ----------
        objective_index : int
            Index into the vector returned by :meth:`evaluate_objectives`.

        Returns
        -------
        InteriorPointMethodResult

        """
        ...

    @abstractmethod
    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return all N objective values at decision variable ``x``.

        Parameters
        ----------
        x : array
            Decision variable.

        Returns
        -------
        array of shape (N,)
            Objective values. The element at index ``self.primary_objective``
            is the primary objective; all others are auxiliary.

        """
        ...

    def trace(self, num_points: int = 50) -> FrontierResults:
        """Trace the Pareto frontier over a uniform grid.

        Algorithm
        ---------
        1. Call :meth:`minimize_objective` for every objective to get N corner
           points and auto-detect the sweep range for each auxiliary.
        2. Build a uniform grid of ``num_points`` values per auxiliary dimension
           spanning ``[bounds_min, bounds_max]``.
        3. For each grid point, call :meth:`solve_with_bounds`; silently skip
           any point that raises :class:`~cvxium.OptimizationError` or
           :class:`~cvxium.ProblemInfeasibleError`.
        4. Return all successfully evaluated points together with the corners.

        Parameters
        ----------
        num_points : int, default=50
            Number of grid points along each auxiliary-objective dimension.
            Total solves is at most ``N * num_points^(N-1)`` where ``N`` is
            the number of objectives.

        Returns
        -------
        FrontierResults

        """
        # --- Step 1: compute corner points ---
        primary_ipm = self.minimize_objective(self.primary_objective)
        primary_objs = self.evaluate_objectives(primary_ipm.solution)
        n_objectives = len(primary_objs)

        if n_objectives < 2:
            raise ValueError(
                f"Need at least 2 objectives; evaluate_objectives returned "
                f"{n_objectives} value(s)."
            )

        n_aux = n_objectives - 1
        aux_indices = [i for i in range(n_objectives) if i != self.primary_objective]

        primary_corner = FrontierPoint(
            objectives=primary_objs,
            solution=primary_ipm.solution.copy(),
            bounds=np.full(n_aux, np.inf),
            ipm_result=primary_ipm,
        )

        aux_corners: list[FrontierPoint] = []
        for aux_idx in aux_indices:
            try:
                ipm = self.minimize_objective(aux_idx)
                objs = self.evaluate_objectives(ipm.solution)
                aux_corners.append(
                    FrontierPoint(
                        objectives=objs,
                        solution=ipm.solution.copy(),
                        bounds=np.full(n_aux, np.inf),
                        ipm_result=ipm,
                    )
                )
            except (OptimizationError, ProblemInfeasibleError):
                pass

        corners: list[FrontierPoint] = [primary_corner, *aux_corners]

        # --- Step 2: auto-detect sweep range ---
        # bounds_max[j] = value of auxiliary j at the primary minimum
        bounds_max = np.array([float(primary_objs[i]) for i in aux_indices])

        # bounds_min[j] = value of auxiliary j at its own minimum (corner j+1)
        # Fall back to bounds_max if that corner failed (will yield a degenerate
        # one-point sweep, which trace() handles gracefully by returning corners).
        bounds_min = bounds_max.copy()
        for j, aux_corner in enumerate(aux_corners):
            bounds_min[j] = float(aux_corner.objectives[aux_indices[j]])

        # Ensure min <= max (swap if objectives are oriented the other way).
        for j in range(n_aux):
            if bounds_min[j] > bounds_max[j]:
                bounds_min[j], bounds_max[j] = bounds_max[j], bounds_min[j]

        # --- Step 3: grid sweep ---
        grids = [
            np.linspace(bounds_min[j], bounds_max[j], num_points)
            for j in range(n_aux)
        ]

        points: list[FrontierPoint] = list(corners)

        for bounds_combo in itertools_product(*grids):
            bounds = np.array(bounds_combo)
            try:
                ipm = self.solve_with_bounds(bounds)
                objs = self.evaluate_objectives(ipm.solution)
                points.append(
                    FrontierPoint(
                        objectives=objs,
                        solution=ipm.solution.copy(),
                        bounds=bounds.copy(),
                        ipm_result=ipm,
                    )
                )
            except (OptimizationError, ProblemInfeasibleError):
                continue

        return FrontierResults(
            points=points,
            corners=corners,
            primary_objective=self.primary_objective,
        )


__all__: list[str] = [
    "FrontierPoint",
    "FrontierResults",
    "MultiObjectiveOptimizer",
]
