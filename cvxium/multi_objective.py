"""Multi-objective optimization and Pareto frontier tracing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product as itertools_product

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
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
        The upper bounds on auxiliary objectives that were imposed when solving. Set to
        ``np.full(N-1, np.inf)`` for corner points where no bounds were imposed.
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
        All successfully evaluated points, including the corner points, deduplicated by
        objective vector.
    corners : list[FrontierPoint]
        The N corner points, one per objective. ``corners[0]`` minimizes the primary
        objective; ``corners[k]`` minimizes the k-th auxiliary objective (in the order
        they appear in ``evaluate_objectives``, skipping the primary). If a corner solve
        failed, fewer than N corners are stored and `knee` will raise.
    primary_objective : int
        Index of the primary objective.
    n_attempted : int
        Number of grid points where `MultiObjectiveOptimizer.solve_with_bounds` was
        called.
    n_skipped : int
        Number of grid points that were skipped because the solver raised
        `cvxium.OptimizationError` or `cvxium.ProblemInfeasibleError`.

    """

    def __init__(
        self,
        points: list[FrontierPoint],
        corners: list[FrontierPoint],
        primary_objective: int,
        n_attempted: int,
        n_skipped: int,
    ) -> None:
        self.points = points
        self.corners = corners
        self.primary_objective = primary_objective
        self.n_attempted = n_attempted
        self.n_skipped = n_skipped

    def knee(self) -> FrontierPoint:
        """Return the frontier point maximally distant from the corner hyperplane.

        Identifies the "knee in the curve" by finding the point on the frontier farthest
        from the hyperplane that passes through the N corner points (where N is the
        number of objectives). For two objectives this reduces to the standard
        max-chord-distance method.

        The reference hyperplane is constructed by translating the corner points so that
        ``corners[0]`` is at the origin, then finding the null-space of the matrix whose
        rows are the remaining translated corners via SVD. This requires the N corner
        points to be **affinely independent**: if they are (nearly) coplanar in
        objective space the normal will be numerically unreliable. The check ``norm ==
        0`` catches exact degeneracy; near- degeneracy is not currently detected.

        Returns
        -------
        FrontierPoint

        Raises
        ------
        ValueError
            If the frontier is empty, if fewer than N corners are available (where N =
            number of objectives inferred from the first point), or if the corner points
            are degenerate.

        """
        if not self.points:
            raise ValueError("No frontier points to evaluate.")

        n_objectives = len(self.points[0].objectives)
        n_corners = len(self.corners)

        if n_corners < 2:
            raise ValueError(
                f"Need at least 2 corner points for knee calculation; got {n_corners}."
            )
        if n_corners != n_objectives:
            raise ValueError(
                f"knee() requires exactly one corner per objective "
                f"(N={n_objectives}), but {n_corners} corner(s) are available. "
                f"This typically means one or more minimize_objective() calls "
                f"failed during trace()."
            )

        corner_objs = np.array([c.objectives for c in self.corners])  # (N, N)

        # Normal to the hyperplane through the N corners.
        # Translate to origin using the first corner, then find the null space
        # of the (N-1) x N matrix whose rows are the translated corner vectors.
        # For N objectives, that matrix has exactly one null-space direction.
        p0 = corner_objs[0]
        vecs = corner_objs[1:] - p0  # (N-1, N)

        _, _, vt = linalg.svd(vecs, full_matrices=True)
        normal = vt[-1].astype(float)  # last right singular vector = null space

        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError(
                "Corner points are degenerate (all identical or collinear)."
            )
        normal /= norm

        distances = np.array(
            [abs(float(np.dot(p.objectives - p0, normal))) for p in self.points]
        )
        return self.points[int(distances.argmax())]

    def plot(
        self,
        annotate_knee: bool = True,
        ax: Axes | None = None,
        x_label: str = "Primary Objective",
        y_label: str = "Auxiliary Objective",
    ) -> Axes:
        r"""Plot the 2-objective Pareto frontier.

        Parameters
        ----------
        annotate_knee : bool, default=True
            If True, mark the knee point and draw a tangent line estimated numerically
            from adjacent frontier points.
        ax : matplotlib Axes, optional
            If provided, plot onto this Axes; otherwise create a new figure.
        x_label : str, default="Primary Objective"
        y_label : str, default="Auxiliary Objective"

        Returns
        -------
        Axes

        Raises
        ------
        NotImplementedError
            If there are more than 2 objectives.
        ValueError
            If the frontier has no points.

        Notes
        -----
        The tangent slope is approximated by the secant between the two nearest frontier
        points rather than from Lagrange multipliers, so it is correct regardless of how
        ``solve_with_bounds`` normalizes the auxiliary constraint internally.

        """
        if not self.points:
            raise ValueError("No frontier points to plot.")
        n_obj = len(self.points[0].objectives)
        if n_obj != 2:
            raise NotImplementedError(
                f"plot() supports only 2-objective frontiers; got {n_obj} objectives."
            )

        if ax is None:
            _, ax = plt.subplots()

        # Sort by auxiliary objective (f1) ascending so the curve is coherent.
        pts = sorted(self.points, key=lambda p: float(p.objectives[1]))
        xs = [float(p.objectives[0]) for p in pts]  # primary (f0), x-axis
        ys = [float(p.objectives[1]) for p in pts]  # auxiliary (f1), y-axis

        ax.plot(xs, ys)
        ax.fill_between(xs, ys, max(ys), color="lightblue", alpha=0.5)

        if annotate_knee and len(pts) >= 2:
            try:
                knee_pt = self.knee()
                kx = float(knee_pt.objectives[0])
                ky = float(knee_pt.objectives[1])

                # Tangent slope from neighboring points (avoids dependence on
                # Lagrange multiplier normalization inside solve_with_bounds).
                knee_idx = next(
                    (
                        i
                        for i, p in enumerate(pts)
                        if np.allclose(p.objectives, knee_pt.objectives, atol=1e-8)
                    ),
                    None,
                )
                slope: float | None = None
                if knee_idx is not None and 0 < knee_idx < len(pts) - 1:
                    dx = xs[knee_idx + 1] - xs[knee_idx - 1]
                    dy = ys[knee_idx + 1] - ys[knee_idx - 1]
                    if abs(dx) > 0:
                        slope = dy / dx

                if slope is not None:
                    width = max(min(0.5 * (kx - min(xs)), 0.5 * (max(xs) - kx)), 0.0)
                    x_vals = np.linspace(kx - width, kx + width)
                    ax.plot(x_vals, ky + slope * (x_vals - kx), color="orange")

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.hlines(
                    y=ky, xmin=xlim[0], xmax=kx, colors="orange", linestyles="dashed"
                )
                ax.vlines(
                    x=kx, ymin=ylim[0], ymax=ky, colors="orange", linestyles="dashed"
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except ValueError:
                pass  # knee() raised; skip annotation

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax


class MultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective Pareto frontier tracing.

    To use this class, subclass it and implement:

    - `solve_with_bounds` — minimize the primary objective subject to upper bounds on
      each auxiliary objective.
    - `minimize_objective` — minimize a single objective with no bounds on the others.
    - `evaluate_objectives` — return all N objective values at a point.

    Then call `trace` to sweep over a uniform grid of auxiliary bounds and collect the
    resulting Pareto-optimal points.

    Parameters
    ----------
    primary_objective : int, default=0
        Index into the vector returned by `evaluate_objectives` that identifies the
        objective to minimize during the frontier sweep. All other objectives become
        auxiliary and receive parametric upper bounds.

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
            ``bounds[j]`` is the upper bound imposed on auxiliary objective ``j`` (in
            the order objectives appear in `evaluate_objectives`, skipping the primary).

        Returns
        -------
        InteriorPointMethodResult

        """
        ...

    @abstractmethod
    def minimize_objective(self, objective_index: int) -> InteriorPointMethodResult:
        """Minimize a single objective with no bounds on the others.

        Used by `trace` to compute the N corner points of the frontier and to
        auto-detect the sweep range for each auxiliary objective.

        Parameters
        ----------
        objective_index : int
            Index into the vector returned by `evaluate_objectives`.

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
            Objective values. The element at index ``self.primary_objective`` is the
            primary objective; all others are auxiliary.

        """
        ...

    def trace(self, num_points: int = 50) -> FrontierResults:
        """Trace the Pareto frontier over a uniform grid.

        Algorithm
        ---------
        1. Call `minimize_objective` for every objective to get N corner points and
           auto-detect the sweep range for each auxiliary.
        2. Build a uniform grid of ``num_points`` values per auxiliary dimension
           spanning ``[bounds_min, bounds_max]``.
        3. For each grid point, call `solve_with_bounds`; silently skip any point that
           raises `cvxium.OptimizationError` or `cvxium.ProblemInfeasibleError`.
        4. Deduplicate by objective vector (within ``1e-8`` absolute tolerance), then
           return all unique points together with the corners.

        Parameters
        ----------
        num_points : int, default=50
            Number of grid points along each auxiliary-objective dimension. Total solves
            is at most ``N + num_points^(N-1)`` where ``N`` is the number of objectives.

        Returns
        -------
        FrontierResults

        Raises
        ------
        ValueError
            If ``num_points < 1`` or fewer than 2 objectives are detected.

        """
        if num_points < 1:
            raise ValueError(f"num_points must be >= 1; got {num_points}.")

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

        # bounds_min[j] = value of auxiliary j at its own minimum (corner j+1).
        # Fall back to bounds_max if that corner failed (yields a degenerate
        # one-point sweep; trace() still returns the corners gracefully).
        bounds_min = bounds_max.copy()
        for j, aux_corner in enumerate(aux_corners):
            bounds_min[j] = float(aux_corner.objectives[aux_indices[j]])

        # Ensure min <= max (swap if objectives are oriented the other way).
        for j in range(n_aux):
            if bounds_min[j] > bounds_max[j]:
                bounds_min[j], bounds_max[j] = bounds_max[j], bounds_min[j]

        # --- Step 3: grid sweep ---
        grids = [
            np.linspace(bounds_min[j], bounds_max[j], num_points) for j in range(n_aux)
        ]

        # Corners are prepended so they survive deduplication (first-seen wins).
        points: list[FrontierPoint] = list(corners)
        n_attempted = 0
        n_skipped = 0

        for bounds_combo in itertools_product(*grids):
            bounds = np.array(bounds_combo)
            n_attempted += 1
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
                n_skipped += 1

        # --- Step 4: deduplicate by objective vector ---
        unique: list[FrontierPoint] = []
        for p in points:
            if not any(
                np.allclose(p.objectives, q.objectives, atol=1e-8) for q in unique
            ):
                unique.append(p)

        unique.sort(key=lambda p: float(p.objectives[self.primary_objective]))

        return FrontierResults(
            points=unique,
            corners=corners,
            primary_objective=self.primary_objective,
            n_attempted=n_attempted,
            n_skipped=n_skipped,
        )


__all__: list[str] = [
    "FrontierPoint",
    "FrontierResults",
    "MultiObjectiveOptimizer",
]
