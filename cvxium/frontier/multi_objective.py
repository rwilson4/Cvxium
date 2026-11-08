"""Pareto frontier tracing for multi-objective problems solved with Cvxium.

``MultiObjectiveOptimizer`` is the abstract base. A subclass supplies three solves:
minimize the primary objective subject to upper bounds on the auxiliary objectives
(``solve_with_bounds``), minimize a single objective (``minimize_objective``), and
evaluate every objective at a point (``evaluate_objectives``). ``trace`` then solves the
N corners of the frontier, sweeps a grid of auxiliary bounds between them (the
epsilon-constraint method), and returns the deduplicated points as ``FrontierResults``.
The sweep can warm-start each solve from the previous one, or fan the grid out over
worker processes.

``FrontierResults`` knows how to analyze and display itself: ``knee`` picks the sampled
point farthest from the hyperplane through the corners, ``envelope`` turns a
two-objective frontier into a smooth curve with certified upper and lower bounds (via
``interpolate_frontier``), and ``plot`` draws it. For two objectives,
``MultiObjectiveOptimizer.refine_knee`` solves for the knee rather than reading it off
the grid, using the frontier tangent slopes a subclass reports through
``tangent_slopes``; each of its probes is folded back into the results.

Everything here works in Cvxium's native minimization convention.

"""

import bisect
import itertools
import logging
import math
import multiprocessing
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product as itertools_product
from typing import Generic, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy import linalg
from typing_extensions import TypeVar

from ..exceptions import (
    BacktrackingLineSearchError,
    NewtonStepError,
    OptimizationError,
    ProblemInfeasibleError,
)
from ..frontier.frontier_interpolation import (
    FrontierEnvelope,
    interpolate_frontier,
)
from ..problems.result import SolveResult

logger: logging.Logger = logging.getLogger("cvxium.multi_objective")

# Relative bracket width at which knee refinement stops. A backstop for a
# frontier whose slopes are too flat for the slope criterion to trigger; the
# root-find normally converges on slope well before this.
_KNEE_BRACKET_TOL: float = 1e-4

# Fraction of the current bracket that an interpolated step must leave clear at
# each end to be trusted. A step nearer an endpoint than this is either chasing
# noise in the duals or about to stall against that endpoint, and the midpoint
# is the better use of the solve. Loose on purpose: the point is to exclude
# degenerate steps, not to second-guess a working interpolant.
_SECANT_MARGIN: float = 1e-3

# Warm-start payload threaded between successive solves during a frontier sweep.
# Defaults to the decision vector; a subclass whose
# inner solver carries extra state (dual iterates, a barrier parameter, ...) can
# parametrize this with a richer type and override ``_extract_warm_start``.
WarmStartT = TypeVar("WarmStartT", default=npt.NDArray[np.float64])


def _knee_space_transform(
    value: float, knee_space: Literal["primary", "root"]
) -> tuple[float, float]:
    """Transform the primary objective into the knee space and differentiate.

    The knee is "where the tangent parallels the chord", which is not invariant to a
    nonlinear reparametrization of either axis. Rescaling an axis affinely leaves it
    alone (the normal rotates exactly enough to compensate), but a square root does not,
    so a frontier can honestly be said to bend in one place measured in the primary
    objective and somewhere else measured in its square root. Which one is wanted is a
    modelling question, and this is where the caller answers it.

    Returns both the transformed value and its derivative from one if/elif, so the two
    cannot be updated out of step. The derivative turns a tangent slope in the primary
    objective into the tangent slope in the knee space, by the chain rule.

    Parameters
    ----------
     value : float
        The primary objective at a frontier point.
     knee_space : Literal["primary", "root"]
        ``"primary"`` is the identity. ``"root"`` measures the knee in sqrt(primary).

    Returns
    -------
     t : float
        The transformed value T(value).
     dt : float
        The derivative T'(value). For ``"root"`` at a zero primary objective this is inf
        (a vertical tangent), so that the finite-slope filters upstream drop the point
        rather than propagate a nan.

    Raises
    ------
     ValueError
        When ``knee_space`` is neither ``"primary"`` nor ``"root"``, or when ``"root"``
        is asked for at a negative primary objective.

    Notes
    -----
    When ``"root"`` is legitimate. Both transforms must be increasing, so that a point
    farther from the chord stays farther from it, and the frontier must stay convex in
    the transformed coordinate, or the monotone root-find in
    ``MultiObjectiveOptimizer.refine_knee`` loses its precondition. Increasing is free.
    Convexity is not, and it does not follow from convexity of the untransformed
    frontier: the square root of a convex function need not be convex, as sqrt(a)
    against the convex f(a) = a shows.

    What does hold: a frontier is the value function
       p(u) = inf { g(x) : h(x) ≤ u },
    which is convex in u whenever g and h are convex. So when the primary objective is
    the square of a nonnegative convex function (a squared norm, say), the square-root
    frontier is the value function of the same feasible set under that convex function
    itself, and is convex by the same argument. ``"root"`` is for that case. Under a
    primary objective that is not such a square, the transform is still increasing and
    ``FrontierResults.knee`` still returns an argmax, but ``refine_knee`` is no longer
    solving a monotone problem and may converge to the wrong root.

    """
    if knee_space == "primary":
        return value, 1.0
    if knee_space != "root":
        raise ValueError(f"knee_space must be 'primary' or 'root', got {knee_space!r}.")
    if value < 0.0:
        raise ValueError(
            f"knee_space='root' needs a nonnegative primary objective, got {value:.6g}."
        )

    root = math.sqrt(value)
    # A zero primary objective is a vertical tangent in root coordinates.
    # Reporting inf lets the finite-slope filters upstream drop the point
    # rather than propagating a nan into the chord comparison.
    return root, math.inf if root == 0.0 else 0.5 / root


@dataclass
class FrontierPoint:
    """A single evaluated point on the Pareto frontier.

    Fields
    ------
     objectives : npt.NDArray[np.float64]
        The value of each objective at this point, of length N, as recorded by
        ``MultiObjectiveOptimizer.point_objectives`` (the primary corner is recorded
        with ``evaluate_objectives`` directly).
     solution : npt.NDArray[np.float64]
        The decision variable at this point.
     bounds : npt.NDArray[np.float64]
        The bounds passed to ``solve_with_bounds``, of length N - 1 for the default
        grid; ``np.full(N - 1, np.inf)`` for a corner, where no bounds were imposed.
        When ``_build_grids`` is overridden, this holds whatever control values that
        grid supplied (for example, a scalarization weight).
     ipm_result : SolveResult
        The raw result from the inner solver.

    """

    objectives: npt.NDArray[np.float64]
    solution: npt.NDArray[np.float64]
    bounds: npt.NDArray[np.float64]
    ipm_result: SolveResult


class FrontierResults:
    """Points, corners, and tangent slopes of a traced Pareto frontier.

    Returned by ``MultiObjectiveOptimizer.trace``. Locates the sampled knee (``knee``),
    interpolates a two-objective frontier with certified bounds (``envelope``), and
    plots it (``plot``). ``MultiObjectiveOptimizer.refine_knee`` adds its probes to the
    points and records its conclusion here.

    Parameters
    ----------
     points : list[FrontierPoint]
        All successfully evaluated points, corners included, deduplicated by objective
        vector (absolute tolerance 1e-8) and sorted by the primary objective.
        Knee-refinement probes are folded in as they are solved (see ``absorb_probe``),
        so this is every distinct point the frontier cost, not just the ones laid out on
        the grid.
     corners : list[FrontierPoint]
        The corner points, one per objective whose minimization succeeded.
        ``corners[0]`` minimizes the primary objective; the rest minimize the auxiliary
        objectives in index order, skipping the primary. A failed auxiliary corner is
        dropped rather than stored as a placeholder, so later entries shift down, and
        ``knee`` raises ``ValueError`` when fewer than N corners remain.
     primary_objective : int
        Index of the primary objective.
     n_attempted : int
        Number of calls to ``MultiObjectiveOptimizer.solve_with_bounds``: the grid
        points (for a parallel sweep stopped early, only those whose results were
        observed), plus any knee-refinement probes absorbed afterwards. Corner solves
        are not counted.
     n_skipped : int
        How many of those failed. A grid solve fails when it raises
        ``OptimizationError`` or ``ProblemInfeasibleError``; a knee probe also when it
        raises ``NewtonStepError`` or ``BacktrackingLineSearchError``.
     tangent_slopes : npt.NDArray[np.float64] | None
        Frontier slope d(primary) / d(auxiliary) at each point, in minimization
        coordinates and index-aligned with ``points``, from
        ``MultiObjectiveOptimizer.tangent_slopes``. Without it ``envelope`` has no lower
        bound (it still smooths and still bounds above, but the result is uncertified),
        and ``refine_knee`` reports no knee.

        Pass this through when rebuilding a results object. It is the only piece of
        state not implied by the points, so a rebuild that forgets it downgrades every
        envelope downstream and nothing complains: a plot simply stops shading its error
        band. It is a constructor argument rather than an attribute set afterwards so
        that the signature asks the question.

    Notes
    -----
    Two attributes are not constructor arguments, because refinement needs a traced
    frontier to run on and so cannot happen before the object exists:
    - ``refined_knee`` (``FrontierPoint | None``) is what
      ``MultiObjectiveOptimizer.refine_knee`` last concluded about this frontier, or
      ``None`` if it has not run or reported no knee.
    - ``knee_refinement_ran`` (``bool``) says whether ``refine_knee`` has run at all.
      This separates "no knee has been solved for" from "a knee was solved for and the
      frontier turned out to be straight", which a bare ``refined_knee is None`` cannot
      express. Falling back to the sampled ``knee`` is right in the first case and wrong
      in the second, where the sampled argmax is exactly the grid-dependent number
      ``refine_knee`` declined to report.

    """

    def __init__(
        self,
        points: list[FrontierPoint],
        corners: list[FrontierPoint],
        primary_objective: int,
        n_attempted: int,
        n_skipped: int,
        tangent_slopes: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.points = points
        self.corners = corners
        self.primary_objective = primary_objective
        self.n_attempted = n_attempted
        self.n_skipped = n_skipped
        self.tangent_slopes: npt.NDArray[np.float64] | None = tangent_slopes
        self.refined_knee: FrontierPoint | None = None
        self.knee_refinement_ran: bool = False

    def record_refined_knee(self, knee: FrontierPoint | None) -> None:
        """Record the conclusion of ``MultiObjectiveOptimizer.refine_knee``.

        Called by ``refine_knee`` on every return path, including the ones that report
        no knee: "we looked and there is no bend" is a finding, and a consumer that
        cannot see it will quietly substitute the sampled argmax for it. Sets
        ``refined_knee`` and ``knee_refinement_ran``.

        Parameters
        ----------
         knee : FrontierPoint | None
            The refined knee, or ``None`` when refinement reported no knee.

        """
        self.refined_knee = knee
        self.knee_refinement_ran = True

    def absorb_probe(self, point: FrontierPoint | None, slope: float | None) -> None:
        """Fold one knee-refinement solve into the traced point set.

        A refinement probe is a frontier solve like any other: same optimizer, same kind
        of bound, an objective vector on the same curve. Reading its slope and then
        discarding it would leave the frontier sampled least densely exactly where the
        search has just shown the shape matters most, and leave ``n_attempted``
        understating what the trace cost. Absorbing it means every property, envelope,
        and plot that reads ``points`` picks the refinement up with no further wiring.
        It is unconditional: a solve already paid for is information.

        Every call increments ``n_attempted``. A failed probe also increments
        ``n_skipped``. A probe within 1e-8 (absolute) of an existing objective vector,
        the same rule ``trace`` deduplicates by, is counted but not inserted; near
        convergence that is the expected case. When ``tangent_slopes`` is already
        misaligned with ``points`` the probe is declined with a logged warning, since no
        insertion can restore the alignment. Otherwise the probe is inserted where it
        keeps ``points`` sorted by the primary objective, and its slope is inserted at
        the same index of ``tangent_slopes``.

        Parameters
        ----------
         point : FrontierPoint | None
            The probe, or ``None`` if its solve failed.
         slope : float | None
            The probe's tangent slope in minimization coordinates, not in whatever space
            the knee search is working in: ``tangent_slopes`` holds the former, and one
            entry in different units from its neighbours would surface as a slightly
            wrong error band rather than as an exception. ``None`` records "no slope
            known here" as nan.

        """
        self.n_attempted += 1
        if point is None:
            self.n_skipped += 1
            return

        # Same rule and same tolerance the trace itself dedupes by. A probe
        # landing on a grid point is the expected case near convergence, not an
        # anomaly.
        if any(
            np.allclose(point.objectives, q.objectives, atol=1e-8, rtol=0)
            for q in self.points
        ):
            return

        slopes = self.tangent_slopes
        if slopes is not None and len(slopes) != len(self.points):
            # ``tangent_slopes`` is documented as index-aligned with ``points``,
            # and no insertion into an already-misaligned array restores that.
            # Declining the point is the lesser harm: a frontier one sample
            # coarser than it might have been, rather than one whose slopes
            # belong to the wrong points.
            logger.warning(
                "Not folding a knee probe into the frontier: %d slopes for %d points.",
                len(slopes),
                len(self.points),
            )
            return

        pri = self.primary_objective
        index = bisect.bisect_left(
            [float(p.objectives[pri]) for p in self.points],
            float(point.objectives[pri]),
        )
        self.points.insert(index, point)
        if slopes is not None:
            self.tangent_slopes = np.insert(
                slopes, index, math.nan if slope is None else slope
            )

    def _in_knee_space(
        self, point: FrontierPoint, knee_space: Literal["primary", "root"]
    ) -> npt.NDArray[np.float64]:
        """Return a point's objective vector with the primary coordinate transformed.

        Parameters
        ----------
         point : FrontierPoint
            The point to transform.
         knee_space : Literal["primary", "root"]
            The transform to apply to the primary objective (see
            ``_knee_space_transform``).

        Returns
        -------
         objectives : npt.NDArray[np.float64]
            A copy of the objective vector, of length N, with the primary entry
            transformed.

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform``: an unrecognized ``knee_space``,
            or ``"root"`` at a negative primary objective.

        """
        objectives = np.asarray(point.objectives, dtype=np.float64).copy()
        pri = self.primary_objective
        objectives[pri] = _knee_space_transform(float(objectives[pri]), knee_space)[0]
        return objectives

    def knee(self, knee_space: Literal["primary", "root"] = "primary") -> FrontierPoint:
        """Return the frontier point farthest from the hyperplane through the corners.

        Identifies the "knee in the curve" as the point farthest from the hyperplane
        through the N corner points, where N is the number of objectives. For two
        objectives this is the standard max-chord-distance method. The argmax is
        invariant to rescaling any objective, since the hyperplane's normal rotates to
        compensate, but not to a nonlinear transform of one, which is what
        ``knee_space`` selects.

        The hyperplane's normal is the last right singular vector of the (N - 1) × N
        matrix whose rows are the corners translated so that ``corners[0]`` is at the
        origin. This requires the N corners to be affinely independent; if they are
        (nearly) coplanar in objective space the normal is numerically unreliable. That
        is not detected: the code's check for a zero normal cannot fire, since singular
        vectors have unit norm, so identical or collinear corners yield an arbitrary
        normal rather than a ``ValueError``.

        Parameters
        ----------
         knee_space : Literal["primary", "root"]
            Which monotone transform of the primary objective to measure the bend in.
            See ``_knee_space_transform`` for when ``"root"`` is well posed.

        Returns
        -------
         knee : FrontierPoint
            The point itself, in untransformed coordinates: ``knee_space`` selects the
            point, it does not change how the point is reported.

        Raises
        ------
         ValueError
            When the frontier has no points, when fewer than two corners are available,
            or when the number of corners differs from the number of objectives
            (typically because a corner solve failed during ``trace``). Also propagated
            from ``_knee_space_transform`` for an unrecognized ``knee_space``, or for
            ``"root"`` at a negative primary objective.

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

        corner_objs = np.array(
            [self._in_knee_space(c, knee_space) for c in self.corners]
        )  # (N, N)

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
            [
                abs(float(np.dot(self._in_knee_space(p, knee_space) - p0, normal)))
                for p in self.points
            ]
        )
        return self.points[int(distances.argmax())]

    def envelope(self, num: int = 200) -> FrontierEnvelope:
        """Smooth the frontier and bound how wrong the smoothing can be.

        Joining solved points with straight segments asserts something false, that the
        frontier is piecewise linear, and the error is one-sided, so a coarse stretch
        reads as a hard trade-off that is not there. A two-objective Pareto boundary is
        convex in minimization coordinates, which turns interpolation into a bounding
        problem with an exact answer on both sides: every chord lies on or above the
        boundary, every tangent on or below it. Between consecutive points the truth is
        trapped in a lens whose width is a measured error bar.

        Passes the auxiliary objective as x, the primary objective as y, and
        ``tangent_slopes`` as the slopes to ``interpolate_frontier``, which verifies
        each slope against the other points before trusting it. Without
        ``tangent_slopes`` (or when no slope survives verification) there is no tangent
        bound: ``tangent`` is all -inf, ``max_band_width`` is inf, and the curve falls
        back to shape-preserving smoothing. Still a better plot than the polyline, just
        no longer certified. See ``MultiObjectiveOptimizer.tangent_slopes`` for
        supplying them.

        Parameters
        ----------
         num : int
            Number of points on the returned dense grid.

        Returns
        -------
         envelope : FrontierEnvelope
            The dense grid, bounds, and curve, in minimization coordinates, with the
            auxiliary objective on ``x``.

        Raises
        ------
         ValueError
            When the frontier does not have exactly two objectives (including when it
            has no points) or has fewer than two points. Also propagated from
            ``interpolate_frontier``: ``num`` below 2, fewer than two distinct auxiliary
            values, non-finite objective values, or ``tangent_slopes`` of a different
            length from ``points``.

        Notes
        -----
        Convexity is assumed, not checked. Because the verified tangents lie below every
        point, ``max_band_width`` stays nonnegative to within the verification tolerance
        even when the points contradict convexity; such a contradiction shows instead as
        ``tangent`` rising above ``chord`` between two points, so compare those two
        arrays to detect it.

        """
        if not self.points or len(self.points[0].objectives) != 2:
            raise ValueError(
                "envelope() is defined for two-objective frontiers only; got "
                f"{len(self.points[0].objectives) if self.points else 0}."
            )

        if len(self.points) < 2:
            raise ValueError(
                "envelope() needs at least two points at distinct auxiliary "
                f"values; got {len(self.points)}."
            )

        aux = 1 - self.primary_objective
        x = np.array([float(p.objectives[aux]) for p in self.points])
        y = np.array([float(p.objectives[self.primary_objective]) for p in self.points])
        return interpolate_frontier(x, y, slopes=self.tangent_slopes, num=num)

    def _draw_frontier(
        self, ax: Axes, xs: list[float], ys: list[float]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Draw the certified interpolant, or the polyline if it is unavailable.

        The chord through consecutive points is an upper bound on a convex frontier, so
        straight segments systematically overstate the primary objective between samples
        and a coarse stretch reads as a hard trade-off that is not there. This draws the
        curve from ``envelope`` instead and, when a tangent bound exists
        (``max_band_width`` finite), shades the lens between ``tangent`` and ``chord``
        in grey, so the plot shows the resolution it has. If ``envelope`` raises
        ``ValueError``, ``ArithmeticError``, or ``LinAlgError``, it draws the polyline
        through the points instead: a frontier that cannot be interpolated is still
        worth plotting.

        The shading also requires ``max_band_width`` ≥ 0, but verified tangents keep the
        band nonnegative to within tolerance, so that test essentially never fails and
        does not catch points that contradict convexity. Where the bounds cross between
        two points the lens is shaded inverted; see the Notes of ``envelope``.

        Note the transpose. ``plot`` puts the primary objective on the x-axis, while
        ``envelope`` follows the interpolation module's convention of the swept
        auxiliary objective on its ``x``. So ``env.curve`` is the abscissa here and
        ``env.x`` the ordinate, and the band is shaded with ``fill_betweenx``.

        Parameters
        ----------
         ax : Axes
            The axes to draw on.
         xs : list[float]
            Primary objective at each point, ordered by the auxiliary objective; used
            only for the polyline fallback.
         ys : list[float]
            Auxiliary objective at each point, in the same order.

        Returns
        -------
         fx : npt.NDArray[np.float64]
            Abscissa (primary objective) of the curve actually drawn, so that the caller
            shades the region under the same curve rather than under the raw polyline.
         fy : npt.NDArray[np.float64]
            Ordinate (auxiliary objective) of the curve actually drawn.

        """
        try:
            env = self.envelope()
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
            # A frontier that cannot be interpolated is still worth plotting;
            # falling back is strictly better than failing to render.
            logger.debug("Falling back to the polyline: envelope() raised %s", e)
            ax.plot(xs, ys)
            return np.asarray(xs), np.asarray(ys)

        ax.plot(env.curve, env.x)
        # Shade only when a tangent bound exists. The ``>= 0.0`` test skips
        # shading for a negative band, but the tangents are verified against
        # the samples, so the band cannot meaningfully go negative and this
        # does not catch crossed bounds: a crossing shows as ``tangent >
        # chord`` between samples, and is shaded inverted here.
        if np.isfinite(env.max_band_width) and env.max_band_width >= 0.0:
            ax.fill_betweenx(env.x, env.tangent, env.chord, color="grey", alpha=0.25)

        return env.curve, env.x

    def plot(
        self,
        annotate_knee: bool = True,
        ax: Axes | None = None,
        x_label: str = "Primary Objective",
        y_label: str = "Auxiliary Objective",
    ) -> Axes:
        """Plot the two-objective Pareto frontier.

        Puts the primary objective on the x-axis and the auxiliary objective on the
        y-axis. The frontier is drawn by ``_draw_frontier`` (the certified interpolant,
        with its error band shaded when slopes are known, or else the polyline), and the
        region between that curve and the largest auxiliary value is shaded light blue.

        With ``annotate_knee``, the sampled ``knee`` (in the primary knee space) is
        marked with dashed guide lines to the axes and, when it is not an end point, a
        short tangent line. This is the sampled knee, not ``refined_knee``. The
        annotation is skipped when ``knee`` raises ``ValueError``.

        Parameters
        ----------
         annotate_knee : bool
            When True, mark the knee point and draw a tangent line estimated from the
            adjacent frontier points.
         ax : Axes | None
            Axes to plot onto; ``None`` creates a new figure.
         x_label : str
            Label for the x-axis (the primary objective).
         y_label : str
            Label for the y-axis (the auxiliary objective).

        Returns
        -------
         ax : Axes
            The axes plotted onto.

        Raises
        ------
         ValueError
            When the frontier has no points.
         NotImplementedError
            When the frontier does not have exactly two objectives.

        Notes
        -----
        The knee's tangent slope is the secant between the knee's two neighbours (in
        order of the auxiliary objective) rather than a value derived from Lagrange
        multipliers, so it does not depend on how ``solve_with_bounds`` normalizes the
        auxiliary constraint internally.

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

        # Sort by auxiliary objective ascending so the curve is coherent.
        aux_idx = 1 - self.primary_objective
        pts = sorted(self.points, key=lambda p: float(p.objectives[aux_idx]))
        pri = self.primary_objective
        xs = [float(p.objectives[pri]) for p in pts]  # primary, x-axis
        ys = [float(p.objectives[aux_idx]) for p in pts]  # auxiliary, y-axis

        fx, fy = self._draw_frontier(ax, xs, ys)
        # Shade from the same curve that was drawn. Filling from the raw points
        # instead leaves the region's boundary a polyline, which reintroduces
        # exactly the vertex the interpolant was there to remove -- and the two
        # coincide closely enough that the artifact reads as the curve itself.
        ax.fill_between(fx, fy, max(fy), color="lightblue", alpha=0.5)

        if annotate_knee and len(pts) >= 2:
            try:
                knee_pt = self.knee()
                kx = float(knee_pt.objectives[pri])
                ky = float(knee_pt.objectives[aux_idx])

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


class MultiObjectiveOptimizer(ABC, Generic[WarmStartT]):
    """Abstract base for tracing the Pareto frontier of a multi-objective problem.

    A subclass implements three methods:
    - ``solve_with_bounds``: minimize the primary objective subject to upper bounds on
      each auxiliary objective.
    - ``minimize_objective``: minimize a single objective with no bounds on the others.
    - ``evaluate_objectives``: return all N objective values at a point.
    Then ``trace`` solves the corners, sweeps a grid of auxiliary bounds between them,
    and collects the resulting Pareto-optimal points.

    Optional overrides extend it: ``tangent_slopes`` reports the frontier's slope at
    each point, which certifies ``FrontierResults.envelope`` and enables
    ``refine_knee``; ``point_objectives`` records objectives that are not a function of
    the solution alone; ``_build_grids`` sweeps a different control variable; and
    ``_extract_warm_start`` threads a richer warm-start payload, of type ``WarmStartT``,
    between solves.

    Parameters
    ----------
     primary_objective : int
        Index into the vector returned by ``evaluate_objectives`` of the objective to
        minimize during the sweep. All other objectives are auxiliary and receive
        parametric upper bounds. The two-objective features (``envelope``,
        ``refine_knee``) assume it is 0 or 1.

    Notes
    -----
    Warm starts. ``trace`` threads each solve's warm-start payload (by default its
    solution) into the ``x0`` of the next ``solve_with_bounds`` or
    ``minimize_objective``. Whether that helps depends on the inner solver. The previous
    optimum lies on the boundary of the constraints that bind there, since a Pareto
    frontier is exactly the locus where they bind, so it is at best marginally feasible
    for the next subproblem. Both ``PrimalDualSolver`` and ``BarrierSolver`` need a
    strictly feasible start and run Phase I (``Solver._solve_feasibility``) when ``x0``
    is not one; they differ in what happens next.
    - Primal-dual solvers (``PrimalDualSolver``) handle warm starts well. They chase KKT
      residuals directly and have no central path to return to, so a near-optimal ``x0``
      looks almost done and progress starts from the first step. Measured speedups on
      frontier sweeps range from about 4.6x (a distance objective with a swept variance
      ceiling and an active ∞-norm constraint) down to parity (a variance objective
      whose active ∞-norm constraint forces extra Newton iterations for active-set
      discovery). The primal-dual solver was never slower than the barrier method in
      these measurements, so preferring it is robust even where the speedup is small.
    - Barrier solvers (``BarrierSolver``, ``InteriorPointMethodSolver``) struggle. A
      start next to the boundary is far from the central path for the initial barrier
      parameter t, so the first Newton steps are cut short by the backtracking line
      search before real progress begins. And a marginal start that numerical noise tips
      into infeasibility sends the solve through Phase I; a Phase I failure raises
      ``ProblemInfeasibleError``, which ``_sweep_grid`` counts as a skipped point, so it
      shows up as a missing grid point rather than an exception. In practice a naive
      barrier warm start makes a sweep slower than cold starts.
    So an implementation should either use a primal-dual inner solver and forward
    ``x0``, or use a barrier solver and ignore ``x0``, letting each grid solve start
    cold. The cause is geometric, not an implementation detail: a point near the
    previous optimum is what a primal-dual method wants and exactly what a barrier
    method does not.

    Threading richer warm-start state. By default the payload threaded between solves is
    the decision vector. A subclass whose solver carries additional warm-start state can
    parametrize ``MultiObjectiveOptimizer`` with a payload type and override
    ``_extract_warm_start`` to return it; the base threads that opaque payload through
    ``x0`` unchanged. This also lets a subclass that sweeps a constraint in the
    loosening direction, so that the previous optimum stays strictly interior to the
    next subproblem, warm-start a barrier solver safely: the boundary adjacency
    described above does not arise there.

    """

    def __init__(self, primary_objective: int = 0) -> None:
        self.primary_objective = primary_objective

    @abstractmethod
    def solve_with_bounds(
        self,
        bounds: npt.NDArray[np.float64],
        x0: WarmStartT | None = None,
    ) -> SolveResult:
        """Minimize the primary objective subject to upper bounds on the auxiliaries.

        ``trace`` calls this at every grid point and ``refine_knee`` at every probe. An
        implementation should raise ``OptimizationError`` or ``ProblemInfeasibleError``
        when a subproblem cannot be solved: ``trace`` then skips the point
        (``refine_knee`` also tolerates ``NewtonStepError`` and
        ``BacktrackingLineSearchError``), while any other exception propagates out of
        the sweep.

        Parameters
        ----------
         bounds : npt.NDArray[np.float64]
            ``bounds[j]`` is the upper bound on auxiliary objective j (in the order the
            objectives appear in ``evaluate_objectives``, skipping the primary), of
            length N - 1. When ``_build_grids`` is overridden, this holds whatever
            control values that grid supplies instead.
         x0 : WarmStartT | None
            Warm-start payload from the previous solve (see ``_extract_warm_start``), or
            ``None`` for a cold start. Adjacent frontier points typically have similar
            solutions, but whether to use it depends on the inner solver; see the class
            Notes.

        Returns
        -------
         result : SolveResult
            The inner solver's result; its ``solution`` is the decision variable.

        """
        ...

    @abstractmethod
    def minimize_objective(
        self,
        objective_index: int,
        x0: WarmStartT | None = None,
    ) -> SolveResult:
        """Minimize a single objective with no bounds on the others.

        Used by ``trace`` to compute the N corner points of the frontier, which also fix
        the sweep range for each auxiliary objective. An exception from the primary
        objective's solve propagates out of ``trace``; an auxiliary solve that raises
        ``OptimizationError`` or ``ProblemInfeasibleError`` just drops that corner.

        Parameters
        ----------
         objective_index : int
            Index into the vector returned by ``evaluate_objectives``.
         x0 : WarmStartT | None
            Warm-start payload from the previous solve, or ``None`` for a cold start.

        Returns
        -------
         result : SolveResult
            The inner solver's result; its ``solution`` is the decision variable.

        """
        ...

    def refine_knee(
        self,
        results: FrontierResults,
        max_iter: int = 24,
        slope_spread_tol: float = 1e-3,
        knee_space: Literal["primary", "root"] = "primary",
    ) -> FrontierPoint | None:
        """Solve for the knee instead of picking the nearest sampled point.

        On a convex two-objective frontier f(a), with a the auxiliary objective, the
        knee has a closed form. Distance to the corner chord is proportional to the
        vertical gap chord(a) - f(a); differentiating, the gap is largest where
           f'(a) = m_chord,
        the tangent parallel to the chord. Convexity makes f' monotone, so this is a
        one-dimensional monotone root-find in which every probe is a real solve that
        reports its own slope. The knee is therefore solved for rather than read off an
        interpolant or quantized to the sweep grid, and the final iterate is itself a
        frontier point, with no confirming solve.

        The search starts from the sampled interval whose slopes bracket the chord
        slope, and steps by safeguarded regula falsi (see ``_next_probe``). It stops as
        soon as the tangent parallels the chord to within ``slope_spread_tol``, which is
        the quantity being solved for, so it usually costs a handful of solves rather
        than the full ``max_iter``. A relative-bracket backstop catches a frontier too
        flat for that criterion to bite.

        Returns ``None`` when the knee is not well determined, which is a real condition
        rather than a failure: when ``results`` has no ``tangent_slopes``, does not have
        exactly two objectives and two corners, has corners at the same auxiliary value,
        or has fewer than two finite slopes; when the frontier is straight to tolerance
        (see Notes); and when the first probe fails or reports no slope. Every probe is
        folded into ``results`` with ``FrontierResults.absorb_probe``, so this updates
        its ``points``, counters, and ``tangent_slopes``.

        Parameters
        ----------
         results : FrontierResults
            A traced two-objective frontier, used for its corners (hence the chord) and
            its slopes. Must carry ``tangent_slopes``.
         max_iter : int
            Cap on the number of probes. A ceiling rather than a target: the search
            normally converges on slope long before reaching it, and each probe is a
            real solve.
         slope_spread_tol : float
            Does double duty. Below this spread of slopes across the traced points,
            relative to the largest slope magnitude, the frontier counts as straight and
            no knee is reported at all. Within this of the chord slope, relative to the
            chord slope's own magnitude (which the corners fix and the grid does not), a
            probe's tangent counts as parallel and the search stops.
         knee_space : Literal["primary", "root"]
            Which monotone transform of the primary objective to solve in. Both the
            chord and every probe's tangent are measured in it, so the search is
            internally consistent; what changes is the answer, since a nonlinear
            transform genuinely moves the bend. ``"root"`` requires a primary objective
            that is the square of a nonnegative convex function, or the monotone
            precondition is lost; see ``_knee_space_transform``.

        Returns
        -------
         knee : FrontierPoint | None
            The refined knee, or ``None`` when the knee is not well determined.

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform`` for an unrecognized
            ``knee_space``, or for ``"root"`` at a negative primary objective, unless
            one of the early ``None`` returns above comes first.
         Exception
            Any exception from ``solve_with_bounds`` other than the four that ``_probe``
            treats as a failed probe, and any exception from ``tangent_slopes``,
            propagates.

        Notes
        -----
        A straight frontier has no knee, and saying so is the point. When the slope is
        nearly constant it nearly equals the chord slope everywhere, so the root-find is
        ill-conditioned, but so is the question. Max chord distance would still return
        an argmax, determined by differences near the noise floor and liable to move a
        long way if the grid changes. This returns ``None`` instead, leaving the caller
        to fall back to a rule that does not depend on curvature that is not there. That
        is also the regime where the choice matters least: along a straight stretch
        every point trades at the same rate.

        The answer is recorded on ``results`` via
        ``FrontierResults.record_refined_knee``, as well as returned. A refined knee is
        off-grid by construction, so a consumer that only has the results object cannot
        recover it and will reach for the sampled ``knee`` instead, marking a point the
        caller did not use and looking entirely correct doing it.

        Unlike ``FrontierResults.envelope``, the search uses the slopes as reported,
        without verifying them against the other points.

        """
        knee = self._refine_knee_search(results, max_iter, slope_spread_tol, knee_space)
        results.record_refined_knee(knee)
        return knee

    def _refine_knee_search(
        self,
        results: FrontierResults,
        max_iter: int,
        slope_spread_tol: float,
        knee_space: Literal["primary", "root"],
    ) -> FrontierPoint | None:
        """Run the knee root-find whose answer ``refine_knee`` records and returns.

        Split out so that recording the answer happens once, at a single exit, rather
        than at each of the many places this search can return from. Every one of those
        is a conclusion worth recording, and a new one added later would otherwise have
        to remember to record for itself.

        Parameters
        ----------
         results : FrontierResults
            The traced frontier; probes are absorbed into it.
         max_iter : int
            Cap on the number of probes.
         slope_spread_tol : float
            Relative tolerance for the straightness test and for convergence.
         knee_space : Literal["primary", "root"]
            The transform of the primary objective to solve in.

        Returns
        -------
         knee : FrontierPoint | None
            The refined knee, or ``None`` when the knee is not well determined.

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform``.

        Notes
        -----
        1. Return ``None`` unless ``results`` has slopes, exactly two corners, exactly
           two objectives, and a nonzero corner span in the auxiliary objective. Two
           corners is not the same as two objectives: ``trace`` drops a corner whose
           solve failed.
        2. Compute the chord slope m_chord between the corners, in the knee space.
        3. Return ``None`` if the finite traced slopes span less than
           ``slope_spread_tol`` times their largest magnitude (the straightness test).
        4. Seed the bracket from the sampled slopes (``_seed_bracket``). The bracket
           backstop is measured against the corner span, a property of the problem,
           rather than the seeded interval, so a coarser sweep does not stop sooner.
        5. Probe at ``_next_probe``, absorb the probe into ``results``, and stop with it
           when
              |slope - m_chord| ≤ ``slope_spread_tol`` × |m_chord|,
           using the largest traced slope magnitude in place of |m_chord| when the chord
           is exactly flat. Otherwise replace the endpoint on the probe's side of the
           root, halving the retained endpoint's residual when the same endpoint is
           retained twice in a row (the Illinois modification of regula falsi), and stop
           when the bracket is narrower than ``_KNEE_BRACKET_TOL`` times the corner
           span.
        A probe that fails or reports no slope ends the search through ``_aborted``.

        """
        if results.tangent_slopes is None or len(results.corners) != 2:
            return None

        # Two corners is not the same as two objectives: ``trace`` drops a
        # corner whose solve failed, so a three-objective frontier that lost
        # one arrives here with exactly two. ``aux = 1 - primary`` is then not
        # a valid auxiliary index, and ``_probe`` would call
        # ``solve_with_bounds`` with one bound where N-1 are expected.
        if not results.points or len(results.points[0].objectives) != 2:
            return None

        pri, aux = self.primary_objective, 1 - self.primary_objective
        c0, c1 = results.corners[0], results.corners[1]
        span = float(c1.objectives[aux]) - float(c0.objectives[aux])
        if span == 0.0:
            return None

        y0, _ = _knee_space_transform(float(c0.objectives[pri]), knee_space)
        y1, _ = _knee_space_transform(float(c1.objectives[pri]), knee_space)
        m_chord = (y1 - y0) / span

        known = self._slopes_in_knee_space(results, knee_space)
        if known is None or known.size < 2:
            return None

        scale = float(np.max(np.abs(known)))
        if scale == 0.0 or float(known.max() - known.min()) / scale < slope_spread_tol:
            logger.info(
                "No knee reported: frontier slopes span %.3g relative to %.3g, so "
                "the curve is straight to within tolerance and the knee is not "
                "determined.",
                float(known.max() - known.min()),
                scale,
            )
            return None

        corner_lo, corner_hi = sorted(
            (float(c0.objectives[aux]), float(c1.objectives[aux]))
        )
        lo, f_lo, hi, f_hi = self._seed_bracket(
            self._knee_samples(results, knee_space), m_chord, corner_lo, corner_hi
        )
        # Anchored to the CORNER span rather than to the seeded interval. The
        # seed narrows where the search starts; measuring the stopping point
        # against it too would silently tighten this backstop by whatever
        # factor the grid happened to supply, so a coarser sweep would stop
        # sooner. The corner span is a property of the problem.
        bracket = corner_hi - corner_lo
        best: FrontierPoint | None = None
        replaced: str | None = None
        for narrowed in range(max_iter):
            mid = self._next_probe(lo, f_lo, hi, f_hi)
            probe = self._probe(mid)
            raw = None if probe is None else self._raw_slope_at(results, probe)

            # Before either early return below, so the frontier keeps the
            # solves it paid for even on the paths where the search gives up
            # part-way. A probe that failed is absorbed too -- as a skip.
            results.absorb_probe(probe, raw)

            if probe is None:
                return self._aborted(best, narrowed, "a probe failed to solve")

            best = probe
            slope = None if raw is None else self._rescale_slope(raw, probe, knee_space)
            if slope is None:
                return self._aborted(best, narrowed, "a probe reported no slope")

            # Converged in the quantity being solved for. Every further
            # iteration is a real solve, so running the full ``max_iter`` once
            # the tangent already parallels the chord to tolerance buys
            # nothing; for a caller that refines many knees that is the
            # difference between a knee costing a few solves and one costing
            # two dozen.
            #
            # Measured against the CHORD slope, not against the spread of the
            # traced slopes. The chord comes from the two corners, so the
            # stopping point does not move with ``num_points``; scaling by the
            # traced spread would make it do so, which is the defect this whole
            # routine exists to remove. Note that the *bracket* is seeded
            # from the grid and therefore does depend on it -- but a bracket
            # only decides where the search starts, not which root it finds.
            #
            # The one exception is a chord that is exactly flat, where there is
            # no chord magnitude to measure against and ``scale`` -- read off
            # the traced slopes -- is the only scale on hand. That case is
            # grid-dependent, and knowingly so: the alternative is an absolute
            # tolerance on a quantity whose units the caller chooses.
            residual = slope - m_chord
            if abs(residual) <= slope_spread_tol * (abs(m_chord) or scale):
                return best

            # ``f'`` increases with the auxiliary objective on a convex
            # frontier, so the root is to the right while the tangent is still
            # steeper than the chord.
            #
            # The halving is Illinois' fix for regula falsi's one failure mode:
            # a convex residual can retain the same endpoint every iteration,
            # so that endpoint goes stale and the interpolant creeps toward the
            # root from one side, slower than bisection would. Damping the
            # retained residual pulls the next interpolant back across.
            if residual < 0.0:
                if replaced == "lo" and f_hi is not None:
                    f_hi *= 0.5
                lo, f_lo, replaced = mid, residual, "lo"
            else:
                if replaced == "hi" and f_lo is not None:
                    f_lo *= 0.5
                hi, f_hi, replaced = mid, residual, "hi"

            # Backstop for a frontier whose slope is too flat for the criterion
            # above to bite: the bracket cannot usefully shrink further.
            if (hi - lo) <= _KNEE_BRACKET_TOL * bracket:
                return best

        return best

    def _knee_samples(
        self, results: FrontierResults, knee_space: Literal["primary", "root"]
    ) -> list[tuple[float, float]]:
        """Pair each finite knee-space slope with its auxiliary coordinate.

        The sweep already paid for these: each solve reports its own tangent, so the
        frontier's derivative is known at every sample before the root-find begins.
        Pairing each slope with the coordinate it belongs to, rather than filtering down
        to a bare array (which is all the straightness test needs), is what lets the
        search start from an interval rather than from the whole corner span. Points and
        slopes are zipped, so a slope array shorter than ``points`` shortens the result
        rather than raising.

        Parameters
        ----------
         results : FrontierResults
            The traced frontier.
         knee_space : Literal["primary", "root"]
            The transform of the primary objective the slopes are restated in.

        Returns
        -------
         samples : list[tuple[float, float]]
            (auxiliary coordinate, knee-space slope) at every point whose rescaled slope
            is finite, sorted by the auxiliary coordinate (``points`` is ordered by the
            primary objective, and the caller walks this looking for a sign change).
            Empty when ``results`` has no slopes.

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform``.

        """
        slopes = results.tangent_slopes
        if slopes is None:
            return []

        pri, aux = self.primary_objective, 1 - self.primary_objective
        samples: list[tuple[float, float]] = []
        # Zipped rather than indexed, for the reason given on
        # ``_slopes_in_knee_space``: a short slope array should shorten the
        # result, not raise on a broadcast.
        for point, slope in zip(results.points, slopes, strict=False):
            _, factor = _knee_space_transform(float(point.objectives[pri]), knee_space)
            scaled = float(slope) * factor
            if math.isfinite(scaled):
                samples.append((float(point.objectives[aux]), scaled))

        samples.sort()
        return samples

    @staticmethod
    def _seed_bracket(
        samples: list[tuple[float, float]],
        m_chord: float,
        corner_lo: float,
        corner_hi: float,
    ) -> tuple[float, float | None, float, float | None]:
        """Narrow the corner span to the sampled interval that holds the root.

        The residual
           r(x) = slope(x) - m_chord
        increases with the auxiliary coordinate x on a convex frontier, so the root lies
        in the first consecutive pair of samples whose residuals change sign. That
        interval costs nothing, since the slopes are already in hand, and it arrives
        with its endpoint residuals, which let the first probe interpolate instead of
        bisect.

        When no consecutive pair changes sign inside the corner span, the root lies
        beyond the outermost sample, and the interval runs from that sample to the
        corner on the far side; the corner end has no residual in hand. With no usable
        sample at all, the interval is the whole corner span. ``_next_probe`` bisects
        until both ends have a residual.

        Scanning consecutive pairs rather than taking the last negative and first
        positive sample is deliberate. Convexity makes the residual monotone in exact
        arithmetic, but multipliers read off a barrier solve that may have aborted are
        not guaranteed to respect that, and the pairwise scan returns a genuine bracket
        from a sequence that is only mostly sorted, where the extremal rule could return
        an inverted one.

        Parameters
        ----------
         samples : list[tuple[float, float]]
            (auxiliary coordinate, knee-space slope) pairs sorted by coordinate, from
            ``_knee_samples``.
         m_chord : float
            Slope of the chord between the corners, in the knee space.
         corner_lo : float
            The smaller of the corners' auxiliary coordinates.
         corner_hi : float
            The larger of the corners' auxiliary coordinates.

        Returns
        -------
         lo : float
            Left end of the bracket.
         f_lo : float | None
            Residual at ``lo``, or ``None`` when ``lo`` is a corner.
         hi : float
            Right end of the bracket.
         f_hi : float | None
            Residual at ``hi``, or ``None`` when ``hi`` is a corner.

        """
        for (x_a, slope_a), (x_b, slope_b) in itertools.pairwise(samples):
            f_a, f_b = slope_a - m_chord, slope_b - m_chord
            if f_a <= 0.0 <= f_b and corner_lo <= x_a < x_b <= corner_hi:
                return x_a, f_a, x_b, f_b

        if samples:
            f_first = samples[0][1] - m_chord
            f_last = samples[-1][1] - m_chord
            if f_last < 0.0 and corner_lo <= samples[-1][0] < corner_hi:
                return samples[-1][0], f_last, corner_hi, None
            if f_first > 0.0 and corner_lo < samples[0][0] <= corner_hi:
                return corner_lo, None, samples[0][0], f_first

        return corner_lo, None, corner_hi, None

    @staticmethod
    def _next_probe(
        lo: float, f_lo: float | None, hi: float, f_hi: float | None
    ) -> float:
        """Choose the next probe: the secant step where safe, else the midpoint.

        Bisection reads only the sign of the residual and throws the magnitude away,
        which is most of what each solve reported. The secant (linear interpolant)
        through the two endpoint residuals,
           step = hi - f_hi (hi - lo) / (f_hi - f_lo),
        uses both, and over a bracket narrow enough for the residual to be nearly
        straight it lands close enough to converge on the next probe rather than in
        several more.

        It is only safe inside the bracket, though. An interpolant fitted to multipliers
        off a barrier solve can step outside it or land on an endpoint, so a step that
        does not leave a margin of ``_SECANT_MARGIN`` times the bracket width at both
        ends is replaced by the midpoint. The midpoint is also used when either residual
        is unknown or the two are equal.

        Parameters
        ----------
         lo : float
            Left end of the bracket.
         f_lo : float | None
            Residual at ``lo``, or ``None`` when unknown.
         hi : float
            Right end of the bracket.
         f_hi : float | None
            Residual at ``hi``, or ``None`` when unknown.

        Returns
        -------
         x : float
            The auxiliary coordinate at which to probe next, strictly inside the
            bracket.

        Notes
        -----
        The margin excludes degenerate steps but does not guarantee bisection's halving:
        an accepted secant step can shrink the bracket by as little as the margin
        fraction. The remedy for regula falsi's one-sided stagnation is the Illinois
        damping applied by the caller (``_refine_knee_search``), and ``max_iter`` bounds
        the cost regardless.

        """
        midpoint = 0.5 * (lo + hi)
        if f_lo is None or f_hi is None or f_hi == f_lo:
            return midpoint

        step = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        margin = _SECANT_MARGIN * (hi - lo)
        return step if lo + margin <= step <= hi - margin else midpoint

    @staticmethod
    def _aborted(
        best: FrontierPoint | None, narrowed: int, why: str
    ) -> FrontierPoint | None:
        """Decide what to return when the root-find stops early.

        The test is whether the iterate carries a bracketing guarantee. After at least
        one narrowing step, a probe's slope has been compared against the chord, so the
        knee lies in the current bracket, and ``best`` is either an endpoint of that
        bracket (when the next probe failed) or inside it (when the next probe reported
        no slope). Either way it is within the bracket width of the knee, and returning
        it, with a warning, beats discarding the work.

        With no narrowing, the abort came on the first probe. When the bracket was not
        seeded from the samples, that probe is just the midpoint of the corner span and
        makes no claim at all about where the knee is. Callers cannot tell a converged
        answer from an aborted one, so they get ``None`` and fall back to a conventional
        rule, which is the honest outcome for a search that never started.

        Parameters
        ----------
         best : FrontierPoint | None
            The last probe that solved, or ``None`` if none has.
         narrowed : int
            Number of narrowing steps taken so far.
         why : str
            Reason for stopping, for the log message.

        Returns
        -------
         knee : FrontierPoint | None
            ``best`` when at least one narrowing step was taken, else ``None``.

        """
        if narrowed:
            logger.warning(
                "Knee refinement stopped after %d narrowing step(s): %s. "
                "Returning the last iterate, which brackets the knee but is "
                "not converged.",
                narrowed,
                why,
            )
            return best

        logger.warning(
            "No knee reported: %s on the first probe, so the bracket was never "
            "narrowed and the last iterate is just the middle of the corner "
            "span.",
            why,
        )
        return None

    def _probe(self, bound: float) -> FrontierPoint | None:
        """Solve at one auxiliary bound and wrap the result as a frontier point.

        Used by the knee search, so it passes a single bound: the frontier has two
        objectives. A solve that raises ``OptimizationError``,
        ``ProblemInfeasibleError``, ``NewtonStepError``, or
        ``BacktrackingLineSearchError`` is logged and reported as ``None``. The last two
        do not derive from ``OptimizationError``, so they are named separately; left
        out, a probe failing either way would escape ``refine_knee`` and the caller
        would lose the whole traced frontier instead of degrading to the last bracketed
        iterate.

        Parameters
        ----------
         bound : float
            Upper bound on the auxiliary objective.

        Returns
        -------
         point : FrontierPoint | None
            The solved point, with objectives from ``point_objectives``, or ``None``
            when the solve failed in one of the ways above.

        Raises
        ------
         Exception
            Any other exception from ``solve_with_bounds``, and any exception from
            ``point_objectives``, propagates.

        """
        bounds = np.array([bound])
        try:
            res = self.solve_with_bounds(bounds)
        except (
            OptimizationError,
            ProblemInfeasibleError,
            # Neither of these derives from OptimizationError -- both come
            # straight off Exception -- so naming the base is not enough. Left
            # out, a probe that fails either way escapes ``refine_knee``
            # entirely and the caller discards the whole traced frontier
            # instead of degrading to the last bracketed iterate.
            NewtonStepError,
            BacktrackingLineSearchError,
        ) as e:
            logger.debug("Knee probe at %.6g failed: %s", bound, e)
            return None

        solution = np.asarray(res.solution)
        return FrontierPoint(
            objectives=self.point_objectives(res, bounds),
            solution=solution,
            bounds=bounds,
            ipm_result=res,
        )

    def _slopes_in_knee_space(
        self, results: FrontierResults, knee_space: Literal["primary", "root"]
    ) -> npt.NDArray[np.float64] | None:
        """Return the traced tangent slopes rescaled into the knee space, finite only.

        The straightness test that decides whether a frontier has a knee at all has to
        ask the question in the same coordinates the search will use: the square root
        compresses one end of the primary axis and stretches the other, so a curve can
        be straight to tolerance in one space and not the other. For ``"root"``, points
        and slopes are zipped, so a slope array shorter than ``points`` shortens the
        result rather than raising.

        Parameters
        ----------
         results : FrontierResults
            The traced frontier.
         knee_space : Literal["primary", "root"]
            The transform of the primary objective to restate the slopes in.

        Returns
        -------
         slopes : npt.NDArray[np.float64] | None
            The finite rescaled slopes, in point order, or ``None`` when ``results`` has
            no slopes.

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform`` when ``knee_space`` is not
            ``"primary"``: an unrecognized value, or ``"root"`` at a negative primary
            objective.

        """
        slopes = results.tangent_slopes
        if slopes is None:
            return None
        if knee_space == "primary":
            return slopes[np.isfinite(slopes)]

        pri = self.primary_objective
        # Zipped rather than indexed: ``tangent_slopes`` is documented as
        # aligned with ``points``, and pairing them makes a short array shorten
        # the result instead of raising on a broadcast.
        scaled = np.array(
            [
                slope * _knee_space_transform(float(p.objectives[pri]), knee_space)[1]
                for p, slope in zip(results.points, slopes, strict=False)
            ]
        )
        return scaled[np.isfinite(scaled)]

    def _raw_slope_at(
        self, results: FrontierResults, point: FrontierPoint
    ) -> float | None:
        """Return the frontier slope at a single probe, in minimization coordinates.

        Calls ``tangent_slopes`` on a one-point ``FrontierResults`` holding the probe
        and sharing the corners of ``results``. One probe feeds two consumers wanting
        different coordinates: the knee search compares against a chord measured in the
        knee space (see ``_rescale_slope``), while a probe folded back into the point
        set must carry the unscaled number, because that is what ``tangent_slopes``
        holds.

        Parameters
        ----------
         results : FrontierResults
            The traced frontier, for its corners and primary objective.
         point : FrontierPoint
            The probe.

        Returns
        -------
         slope : float | None
            The slope d(primary) / d(auxiliary) at the probe, or ``None`` when
            ``tangent_slopes`` returns ``None`` or a non-finite value.

        Raises
        ------
         Exception
            Whatever ``tangent_slopes`` raises propagates.

        """
        slopes = self.tangent_slopes(
            FrontierResults(
                points=[point],
                corners=results.corners,
                primary_objective=results.primary_objective,
                n_attempted=1,
                n_skipped=0,
            )
        )
        if slopes is None or not np.isfinite(slopes[0]):
            return None

        return float(slopes[0])

    def _rescale_slope(
        self,
        slope: float,
        point: FrontierPoint,
        knee_space: Literal["primary", "root"],
    ) -> float | None:
        """Restate a minimization-coordinate slope in the knee space.

        Multiplies by the derivative of the knee-space transform at the point's primary
        objective (the chain rule).

        Parameters
        ----------
         slope : float
            The slope d(primary) / d(auxiliary).
         point : FrontierPoint
            The point the slope belongs to.
         knee_space : Literal["primary", "root"]
            The transform of the primary objective.

        Returns
        -------
         scaled : float | None
            The slope in the knee space, or ``None`` when it is not finite (for example,
            ``"root"`` at a zero primary objective).

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform``.

        """
        _, factor = _knee_space_transform(
            float(point.objectives[self.primary_objective]), knee_space
        )
        scaled = slope * factor
        return scaled if math.isfinite(scaled) else None

    def _slope_at(
        self,
        results: FrontierResults,
        point: FrontierPoint,
        knee_space: Literal["primary", "root"] = "primary",
    ) -> float | None:
        """Return the frontier slope at a single probe, rescaled into the knee space.

        Composes ``_raw_slope_at`` and ``_rescale_slope``. The knee search calls those
        two separately, since it also needs the raw slope to absorb the probe.

        Parameters
        ----------
         results : FrontierResults
            The traced frontier, for its corners and primary objective.
         point : FrontierPoint
            The probe.
         knee_space : Literal["primary", "root"]
            The transform of the primary objective.

        Returns
        -------
         slope : float | None
            The slope in the knee space, or ``None`` when it is unknown or not finite.

        Raises
        ------
         ValueError
            Propagated from ``_knee_space_transform``.

        """
        raw = self._raw_slope_at(results, point)
        return None if raw is None else self._rescale_slope(raw, point, knee_space)

    def tangent_slopes(
        self, results: FrontierResults
    ) -> npt.NDArray[np.float64] | None:
        """Return the frontier tangent slope d(primary) / d(auxiliary) at each point.

        A scalarized sweep knows its own frontier derivative for free. A subproblem
        minimizing
           f_0 + λ f_1
        lands where the tangent has slope -λ, by the envelope theorem; an
        epsilon-constraint subproblem gets the same number from the KKT multiplier on
        its bound. Supplying it upgrades ``FrontierResults.envelope`` from a smoothing
        heuristic to an interpolation of value and slope, unlocks the tangent bound that
        certifies it, and enables ``refine_knee``.

        The base returns ``None``: correct but uninformative. It cannot be derived
        generically, because ``solve_with_bounds`` chooses how a bound spreads into
        constraint rows, so only the subclass knows which multipliers belong to it and
        how they combine. A subclass that bounds one objective with one row can return
        that row's multiplier directly; one that spreads a scalar bound across several
        scaled rows has to aggregate them. The multipliers for a point are on
        ``point.ipm_result.duals``.

        ``trace`` calls this on the freshly traced frontier, before returning it. The
        knee search also calls it once per probe, on a one-point ``FrontierResults``
        that shares the traced frontier's corners (see ``_raw_slope_at``), so an
        implementation must work from ``results.points`` and not assume the full grid.

        Parameters
        ----------
         results : FrontierResults
            The frontier whose slopes are wanted.

        Returns
        -------
         slopes : npt.NDArray[np.float64] | None
            Slopes in minimization coordinates, of length ``len(results.points)`` and
            aligned with ``results.points``, or ``None`` when unknown. Non-finite
            entries mean "no slope known here" and are skipped; a corner with a
            genuinely vertical tangent should report nan. ``envelope`` verifies the
            slopes against the other points before trusting them, so an inaccurate one
            loosens its bound rather than invalidating it; ``refine_knee`` uses them as
            reported.

        """
        return None

    @abstractmethod
    def evaluate_objectives(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return all N objective values at a decision variable.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Decision variable.

        Returns
        -------
         objectives : npt.NDArray[np.float64]
            Objective values, of length N. The element at index
            ``self.primary_objective`` is the primary objective; all others are
            auxiliary.

        """
        ...

    def point_objectives(
        self, result: SolveResult, bounds: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return the objective vector to record for a point solved at ``bounds``.

        The grid sweep, the auxiliary corners, and the knee probes all build their
        ``FrontierPoint`` objectives through this; the primary corner uses
        ``evaluate_objectives`` directly. The default is
        ``evaluate_objectives(result.solution)``: the objectives are a function of the
        decision vector alone. Override it when they are not, for example when an
        auxiliary coordinate is the bound a point was solved at rather than anything its
        solution attains.

        Parameters
        ----------
         result : SolveResult
            The result of ``solve_with_bounds`` or ``minimize_objective``.
         bounds : npt.NDArray[np.float64]
            The auxiliary bounds it was solved at, of length N - 1; all inf for a
            corner.

        Returns
        -------
         objectives : npt.NDArray[np.float64]
            The objective vector to record, of length N.

        """
        del bounds
        return self.evaluate_objectives(result.solution)

    def _extract_warm_start(self, result: SolveResult) -> WarmStartT:
        """Return the warm-start payload to thread from one solve to the next.

        The sequential sweep calls this on each completed solve (corners included) and
        passes the returned value as the ``x0`` of the following ``solve_with_bounds``
        or ``minimize_objective``; the parallel sweep never warm-starts. The default
        threads the decision vector ``result.solution``, matching the default
        ``WarmStartT`` of ``npt.NDArray[np.float64]``. Override it in a subclass whose
        inner solver benefits from additional state (dual iterates, a barrier parameter)
        by returning that richer payload and parametrizing the class with its type.

        Parameters
        ----------
         result : SolveResult
            The result just produced by ``solve_with_bounds`` or ``minimize_objective``.
            A subclass that returns a ``SolveResult`` subclass can read its extra fields
            here.

        Returns
        -------
         payload : WarmStartT
            The value to pass as the next solve's ``x0``.

        """
        return cast("WarmStartT", result.solution)

    def _compute_corners(
        self,
        aux_indices: list[int],
        n_aux: int,
        primary_ipm: SolveResult,
        primary_objs: npt.NDArray[np.float64],
        callback: Callable[[FrontierPoint, int, int], bool] | None = None,
        counter: list[int] | None = None,
        total: int = 0,
    ) -> tuple[FrontierPoint, list[FrontierPoint | None], bool]:
        """Compute the primary and auxiliary corner points.

        Wraps the already-solved primary corner, then minimizes each auxiliary objective
        in turn, warm-starting each solve from the last successful one. An auxiliary
        solve that raises ``OptimizationError`` or ``ProblemInfeasibleError`` leaves
        that corner as ``None``.

        Parameters
        ----------
         aux_indices : list[int]
            Indices of the auxiliary objectives.
         n_aux : int
            Number of auxiliary objectives.
         primary_ipm : SolveResult
            Result from minimizing the primary objective.
         primary_objs : npt.NDArray[np.float64]
            Objective values at the primary corner, of length N.
         callback : Callable[[FrontierPoint, int, int], bool] | None
            Called as ``callback(point, k, total)`` after each successful solve; returns
            True to stop early. Ignored unless ``counter`` is also given.
         counter : list[int] | None
            Mutable one-element list holding the running 1-based solve index ``k``,
            shared across the corner and grid phases.
         total : int
            Total expected number of solves, passed through to ``callback``.

        Returns
        -------
         primary_corner : FrontierPoint
            The primary corner, with all-inf bounds.
         aux_corners : list[FrontierPoint | None]
            One entry per auxiliary objective, ``None`` where the solve failed or was
            not attempted after an early stop.
         stopped : bool
            True if the callback requested early termination.

        Raises
        ------
         Exception
            Any other exception from ``minimize_objective``, ``point_objectives``,
            ``_extract_warm_start``, or the callback propagates, including
            ``NewtonStepError`` and ``BacktrackingLineSearchError``.

        """
        primary_corner = FrontierPoint(
            objectives=primary_objs,
            solution=primary_ipm.solution.copy(),
            bounds=np.full(n_aux, np.inf),
            ipm_result=primary_ipm,
        )

        stopped = False
        if callback is not None and counter is not None:
            counter[0] += 1
            stopped = callback(primary_corner, counter[0], total)

        aux_corners: list[FrontierPoint | None] = [None for _ in range(n_aux)]
        if stopped:
            return primary_corner, aux_corners, stopped

        prev_x0: WarmStartT | None = self._extract_warm_start(primary_ipm)
        for j, aux_idx in enumerate(aux_indices):
            try:
                ipm = self.minimize_objective(aux_idx, x0=prev_x0)
                objs = self.point_objectives(ipm, np.full(n_aux, np.inf))
                point = FrontierPoint(
                    objectives=objs,
                    solution=ipm.solution.copy(),
                    bounds=np.full(n_aux, np.inf),
                    ipm_result=ipm,
                )
                aux_corners[j] = point
                prev_x0 = self._extract_warm_start(ipm)
                if callback is not None and counter is not None:
                    counter[0] += 1
                    stopped = callback(point, counter[0], total)
                    if stopped:
                        break
            except (OptimizationError, ProblemInfeasibleError):
                pass

        return primary_corner, aux_corners, stopped

    def _build_grids(
        self,
        bounds_min: npt.NDArray[np.float64],
        bounds_max: npt.NDArray[np.float64],
        num_points: int,
        n_aux: int,
    ) -> list[npt.NDArray[np.float64]]:
        """Build the per-auxiliary sweep grids fed to ``solve_with_bounds``.

        Returns one array per auxiliary dimension; ``trace`` sweeps their Cartesian
        product, passing each combination as the ``bounds`` argument to
        ``solve_with_bounds``.

        The default is a uniform grid of ``num_points`` values strictly between the
        corner objective values ``bounds_min`` and ``bounds_max``. The endpoints are
        excluded because the corners already sit there, and solving at the exact bounds
        can leave the inner feasible set with empty interior, which breaks
        interior-point methods.

        Override this to sweep a different control variable, for example a scalar weight
        λ in a weighted-sum scalarization, instead of an objective-space bound. An
        override may ignore ``bounds_min`` and ``bounds_max`` entirely; whatever arrays
        it returns are handed verbatim to ``solve_with_bounds`` as ``bounds``, so a
        matching ``solve_with_bounds`` override must interpret them accordingly. An
        override must return at least one array: the Cartesian product of an empty list
        is a single empty combination, one solve at ``bounds = array([])``, which no
        ``solve_with_bounds`` can interpret, so ``trace`` rejects it. A returned array
        may be empty, though: that is a well-formed request for an empty sweep, and
        ``trace`` reports only the corners.

        Parameters
        ----------
         bounds_min : npt.NDArray[np.float64]
            Lower end of the default sweep for each auxiliary objective, of length
            ``n_aux``.
         bounds_max : npt.NDArray[np.float64]
            Upper end of the default sweep for each auxiliary objective, of length
            ``n_aux``.
         num_points : int
            Number of interior grid values per auxiliary dimension.
         n_aux : int
            Number of auxiliary objectives, N - 1.

        Returns
        -------
         grids : list[npt.NDArray[np.float64]]
            One array of sweep values per auxiliary dimension.

        """
        return [
            np.linspace(bounds_min[j], bounds_max[j], num_points + 2)[1:-1]
            for j in range(n_aux)
        ]

    def _sweep_grid(
        self,
        grids: list[npt.NDArray[np.float64]],
        x0: WarmStartT | None = None,
        callback: Callable[[FrontierPoint, int, int], bool] | None = None,
        counter: list[int] | None = None,
        total: int = 0,
    ) -> tuple[list[FrontierPoint], int, int, bool]:
        """Solve at each grid point in order, warm-starting each from the last.

        Sweeps the Cartesian product of ``grids``, passing the payload of the last
        successful solve as ``x0``. A solve that raises ``OptimizationError`` or
        ``ProblemInfeasibleError`` is counted as skipped and leaves the warm start
        unchanged.

        Parameters
        ----------
         grids : list[npt.NDArray[np.float64]]
            Per-auxiliary-dimension grids; their Cartesian product is swept.
         x0 : WarmStartT | None
            Warm-start payload for the first solve.
         callback : Callable[[FrontierPoint, int, int], bool] | None
            Called as ``callback(point, k, total)`` after each successful solve; returns
            True to stop early. Ignored unless ``counter`` is also given.
         counter : list[int] | None
            Mutable one-element list holding the running 1-based solve index ``k``,
            shared across the corner and grid phases.
         total : int
            Total expected number of solves, passed through to ``callback``.

        Returns
        -------
         points : list[FrontierPoint]
            The successfully solved points, in grid order.
         n_attempted : int
            Number of grid points at which ``solve_with_bounds`` was called.
         n_skipped : int
            Number of those that failed.
         stopped : bool
            True if the callback requested early termination.

        Raises
        ------
         Exception
            Any other exception from ``solve_with_bounds``, ``point_objectives``,
            ``_extract_warm_start``, or the callback propagates, including
            ``NewtonStepError`` and ``BacktrackingLineSearchError``.

        """
        points: list[FrontierPoint] = []
        n_attempted = 0
        n_skipped = 0
        stopped = False
        prev_x0: WarmStartT | None = x0
        for bounds_combo in itertools_product(*grids):
            bounds = np.array(bounds_combo)
            n_attempted += 1
            try:
                ipm = self.solve_with_bounds(bounds, x0=prev_x0)
                objs = self.point_objectives(ipm, bounds)
                point = FrontierPoint(
                    objectives=objs,
                    solution=ipm.solution.copy(),
                    bounds=bounds.copy(),
                    ipm_result=ipm,
                )
                points.append(point)
                prev_x0 = self._extract_warm_start(ipm)
                if callback is not None and counter is not None:
                    counter[0] += 1
                    stopped = callback(point, counter[0], total)
                    if stopped:
                        break
            except (OptimizationError, ProblemInfeasibleError):
                n_skipped += 1
        return points, n_attempted, n_skipped, stopped

    def _sweep_grid_parallel(
        self,
        grids: list[npt.NDArray[np.float64]],
        n_jobs: int,
        callback: Callable[[FrontierPoint, int, int], bool] | None = None,
        counter: list[int] | None = None,
        total: int = 0,
    ) -> tuple[list[FrontierPoint], int, int, bool]:
        """Solve the grid points concurrently across worker processes.

        Fans the grid solves out over a ``ProcessPoolExecutor`` using the ``"spawn"``
        start method (fresh interpreters, so each worker imports numpy under the
        single-threaded BLAS environment set by ``_single_thread_blas``). The grid
        points are mathematically independent, so this is an embarrassingly parallel
        drop-in for ``_sweep_grid``, with two deliberate differences:
        - Cold starts. No warm start is threaded between points; each solves from
          scratch (see ``_solve_frontier_point``).
        - Unordered callback. The callback fires in completion order, not grid order, so
          ``k`` still runs contiguously but a given ``k`` need not correspond to the
          k-th grid point. Early termination (the callback returning True) cancels the
          points the executor has not yet dispatched, but solves already dispatched
          (running in a worker, or already moved to the executor's internal call queue)
          cannot be withdrawn: the sweep waits for them during shutdown and discards
          their results.

        Parameters
        ----------
         grids : list[npt.NDArray[np.float64]]
            Per-auxiliary-dimension grids; their Cartesian product is swept.
         n_jobs : int
            Number of worker processes. A value below 1 uses the executor default (one
            worker per core).
         callback : Callable[[FrontierPoint, int, int], bool] | None
            Called as ``callback(point, k, total)`` after each successful solve; returns
            True to stop early. Ignored unless ``counter`` is also given.
         counter : list[int] | None
            Mutable one-element list holding the running 1-based solve index ``k``,
            shared across the corner and grid phases.
         total : int
            Total expected number of solves, passed through to ``callback``.

        Returns
        -------
         points : list[FrontierPoint]
            The successfully solved points, in completion order.
         n_attempted : int
            Number of grid points whose result was observed, successful or skipped;
            points cancelled or discarded after an early stop are not counted.
         n_skipped : int
            Number of observed points whose solve failed.
         stopped : bool
            True if the callback requested early termination.

        Raises
        ------
         Exception
            An exception raised in a worker other than the two that
            ``_solve_frontier_point`` treats as a skipped point is re-raised here, as is
            a failure to pickle the optimizer; either ends the sweep.

        """
        bounds_list = [np.array(combo) for combo in itertools_product(*grids)]
        points: list[FrontierPoint] = []
        n_skipped = 0
        stopped = False

        max_workers = n_jobs if n_jobs > 0 else None
        ctx = multiprocessing.get_context("spawn")
        with _single_thread_blas():
            executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
            try:
                futures = [
                    executor.submit(_solve_frontier_point, self, bounds)
                    for bounds in bounds_list
                ]
                for future in as_completed(futures):
                    point = future.result()
                    if point is None:
                        n_skipped += 1
                        continue
                    points.append(point)
                    if callback is not None and counter is not None:
                        counter[0] += 1
                        if callback(point, counter[0], total):
                            stopped = True
                            break
            finally:
                # ``cancel_futures`` drops every grid point still pending in the
                # executor. Points already moved into its internal call queue are
                # marked running, so neither it nor ``Future.cancel`` reaches them;
                # with the solves already running in workers, an early stop still
                # waits on up to about ``2 * max_workers + 1`` solves (CPython).
                executor.shutdown(wait=True, cancel_futures=True)

        # A grid point was "attempted" if its solve completed AND its result was
        # observed here -- successfully (a point) or with a handled
        # infeasibility/error (skipped). After an early stop, queued points are
        # cancelled and never run; the in-flight solves do finish during shutdown,
        # but their results are discarded and are not counted either.
        n_attempted = len(points) + n_skipped
        return points, n_attempted, n_skipped, stopped

    def trace(
        self,
        num_points: int = 50,
        callback: Callable[[FrontierPoint, int, int], bool] | None = None,
        n_jobs: int = 1,
    ) -> FrontierResults:
        """Trace the Pareto frontier over a grid of auxiliary bounds.

        Parameters
        ----------
         num_points : int
            Number of grid points along each auxiliary-objective dimension. With the
            default ``_build_grids``, the total number of solves is at most
               N + num_points^(N - 1),
            where N is the number of objectives; an override that returns a differently
            shaped grid changes that count.
         callback : Callable[[FrontierPoint, int, int], bool] | None
            If given, called after each successful solve as ``callback(point, k,
            total)``, where ``point`` is the ``FrontierPoint`` just computed, ``k`` is
            the 1-based solve index (corners and grid points numbered continuously), and
            ``total`` is the total expected number of solves. The corner callbacks
            necessarily fire against the estimate N + num_points^(N - 1), since the grid
            cannot be built until the corners have fixed its bounds; after that
            ``total`` is re-derived from the grid ``_build_grids`` actually returned.
            The two agree unless ``_build_grids`` is overridden. Return True to stop the
            sweep early, False to continue. Failed solves do not trigger the callback.
            When ``n_jobs`` is not 1 the callback fires in completion order rather than
            grid order.
         n_jobs : int
            Number of worker processes for the grid sweep. 1 runs the sweep sequentially
            in-process and warm-starts each solve from the previous one. Any other value
            fans the grid solves out over a ``ProcessPoolExecutor`` (any value below 1
            means one worker per core), which is a large win when a single solve is
            expensive: the grid points are independent, so wall-clock time can drop
            nearly linearly in the worker count. The parallel path solves each point
            cold, so it is best paired with an inner solver whose cold-start cost is
            what dominates. Corners are always computed sequentially before the sweep.
            Requires the optimizer (and everything it references) to be picklable.

        Returns
        -------
         results : FrontierResults
            The deduplicated points sorted by the primary objective, the corners, the
            solve counts, and the slopes from ``tangent_slopes``.

        Raises
        ------
         ValueError
            When ``num_points`` is below 1, when ``evaluate_objectives`` returns fewer
            than two objectives, or when ``_build_grids`` returns no axes at all.
         Exception
            Any exception from the primary objective's ``minimize_objective`` or
            ``evaluate_objectives`` (``OptimizationError`` and
            ``ProblemInfeasibleError`` included) propagates, as does one from an
            auxiliary corner or grid solve other than those two (for example
            ``NewtonStepError`` or ``BacktrackingLineSearchError``), and any exception
            from ``tangent_slopes``.

        Notes
        -----
        1. Call ``minimize_objective`` for every objective to get the N corner points
           (see ``_compute_corners``). A failed auxiliary corner is dropped.
        2. Set the sweep range for each auxiliary objective: from its value at the
           primary corner to its own minimum at its corner, swapped if needed so the
           range is ascending. When an auxiliary corner failed, its range collapses to
           the single value at the primary corner.
        3. If all corners share one objective vector (within 1e-8), the frontier is a
           single point: skip the sweep and log it. The sweep is also skipped when the
           callback stopped during the corners.
        4. Call ``_build_grids`` for the sweep grids. By default that is a uniform grid
           of ``num_points`` values per auxiliary dimension strictly between the ends of
           the range; the endpoints are excluded because the corners already sit there,
           and re-solving there can leave the inner feasible set with empty interior,
           which breaks interior-point methods.
        5. Call ``solve_with_bounds`` at each grid point, sequentially with warm starts
           or in parallel. A point whose solve raises ``OptimizationError`` or
           ``ProblemInfeasibleError`` is skipped silently.
        6. Deduplicate by objective vector (within 1e-8 absolute tolerance, corners
           first so they survive), sort by the primary objective, and attach
           ``tangent_slopes``.

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

        # Provisional callback denominator. The real grid cannot be built until
        # the corners have fixed its bounds, so the corner callbacks necessarily
        # fire against this estimate; it is exact for the default
        # ``_build_grids`` and corrected below once the real grid exists.
        total = n_objectives + num_points ** (n_objectives - 1)
        counter: list[int] = [0]

        primary_corner, aux_corners, stopped = self._compute_corners(
            aux_indices,
            n_aux,
            primary_ipm,
            primary_objs,
            callback=callback,
            counter=counter,
            total=total,
        )
        corners: list[FrontierPoint] = [primary_corner] + [
            c for c in aux_corners if c is not None
        ]

        # --- Step 2: auto-detect sweep range ---
        bounds_max = np.array([float(primary_objs[i]) for i in aux_indices])
        bounds_min = bounds_max.copy()
        for j in range(n_aux):
            aux_corner = aux_corners[j]
            if aux_corner is not None:
                bounds_min[j] = float(aux_corner.objectives[aux_indices[j]])

        # Ensure min <= max (swap if objectives are oriented the other way).
        for j in range(n_aux):
            if bounds_min[j] > bounds_max[j]:
                bounds_min[j], bounds_max[j] = bounds_max[j], bounds_min[j]

        # --- Degenerate detection ---
        # If all corners yield essentially the same objective vector,
        # the frontier is a single point and the grid sweep is wasted.
        corner_objs = np.array([c.objectives for c in corners])
        degenerate = len(corners) >= 2 and np.allclose(
            corner_objs, corner_objs[0], atol=1e-8, rtol=0
        )
        if degenerate:
            logger.info(
                "Degenerate Pareto frontier detected: all %d corner "
                "points yield the same objective vector. Skipping "
                "grid sweep.",
                len(corners),
            )

        # --- Step 3: grid sweep ---
        if stopped or degenerate:
            grid_points: list[FrontierPoint] = []
            n_attempted = 0
            n_skipped = 0
        else:
            grids = self._build_grids(bounds_min, bounds_max, num_points, n_aux)
            if not grids:
                # The Cartesian product of no axes is one *empty* combination, so
                # this would silently become a single solve at ``bounds =
                # array([])`` -- not the empty sweep the override presumably
                # meant. An override wanting no sweep should return an empty
                # axis, which the product handles correctly.
                raise ValueError(
                    f"{type(self).__name__}._build_grids returned no axes; it "
                    "must return at least one array (an empty array is how you "
                    "request an empty sweep)."
                )
            # An override is free to return a different number of axes, or a
            # different number of values per axis, than the default. Re-derive
            # the denominator from the grid that actually exists so the sweep
            # callbacks see a ``total`` that ``k`` can reach.
            n_grid = 1
            for grid in grids:
                n_grid *= len(grid)
            total = n_objectives + n_grid
            if n_jobs == 1:
                # Use the last successful corner result as x0 for the first
                # grid solve, threading continuity from corners to sweep.
                last_corner_x0 = self._extract_warm_start(corners[-1].ipm_result)
                grid_points, n_attempted, n_skipped, _ = self._sweep_grid(
                    grids,
                    x0=last_corner_x0,
                    callback=callback,
                    counter=counter,
                    total=total,
                )
            else:
                grid_points, n_attempted, n_skipped, _ = self._sweep_grid_parallel(
                    grids,
                    n_jobs=n_jobs,
                    callback=callback,
                    counter=counter,
                    total=total,
                )

        # Corners are prepended so they survive deduplication (first-seen wins).
        points: list[FrontierPoint] = list(corners) + grid_points

        # --- Step 4: deduplicate by objective vector ---
        unique: list[FrontierPoint] = []
        for pt in points:
            if not any(
                np.allclose(pt.objectives, q.objectives, atol=1e-8, rtol=0)
                for q in unique
            ):
                unique.append(pt)

        unique.sort(key=lambda pt: float(pt.objectives[self.primary_objective]))

        results = FrontierResults(
            points=unique,
            corners=corners,
            primary_objective=self.primary_objective,
            n_attempted=n_attempted,
            n_skipped=n_skipped,
        )
        results.tangent_slopes = self.tangent_slopes(results)
        return results


# Environment variables that cap the per-process thread pool of the common
# BLAS / OpenMP backends. Parallel frontier fan-out (``trace(n_jobs != 1)``)
# spawns one worker process per grid point; if each worker's BLAS in turn
# spawns its own pool across every core, the machine is oversubscribed by
# roughly ``n_jobs x n_cores`` threads and every individual solve slows by an
# order of magnitude -- wiping out (and then some) the parallel speedup. Pinning
# these to ``"1"`` in the *parent* before the workers are spawned makes each
# freshly-imported worker single-threaded, because a spawned child inherits the
# parent's environment at creation time and re-imports numpy under it. Measured
# on a downstream robust frontier: pinning turned a 0.74x slowdown into a ~7x
# speedup, and it is free on a single solve (BLAS threading buys nothing on
# these small dense systems).
_BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextmanager
def _single_thread_blas() -> Iterator[None]:
    """Pin BLAS and OpenMP backends to a single thread for the duration.

    Sets each variable in ``_BLAS_THREAD_ENV_VARS`` to ``"1"`` on entry and restores its
    prior value, or removes it if it was previously unset, on exit. Wrap the spawning of
    a frontier worker pool with this so the workers inherit a single-threaded BLAS
    environment; see the comment on ``_BLAS_THREAD_ENV_VARS`` for why oversubscription
    otherwise dominates.

    Not concurrency-safe: ``os.environ`` is process-global, so two overlapping uses in
    the same process (or an overlap with unrelated code toggling the same variables)
    race on the save and restore: the inner scope reads the outer scope's ``"1"`` as the
    prior value and restores that instead of the original. Intended to wrap one
    worker-pool launch at a time.

    """
    saved: dict[str, str | None] = {
        var: os.environ.get(var) for var in _BLAS_THREAD_ENV_VARS
    }
    try:
        for var in _BLAS_THREAD_ENV_VARS:
            os.environ[var] = "1"
        yield
    finally:
        for var, prior in saved.items():
            if prior is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prior


def _solve_frontier_point(
    optimizer: "MultiObjectiveOptimizer[WarmStartT]",
    bounds: npt.NDArray[np.float64],
) -> FrontierPoint | None:
    """Solve one frontier grid point in a worker process.

    Module-level, hence picklable, so it can be dispatched to a ``ProcessPoolExecutor``.
    Solves cold (``x0=None``): the parallel sweep does not thread warm starts, since the
    points are dispatched concurrently and have no ordering to chain through. Returns
    ``None`` rather than raising when the solve raises ``OptimizationError`` or
    ``ProblemInfeasibleError``, so the parent can count it as a skipped point without
    tearing down the pool.

    Parameters
    ----------
     optimizer : MultiObjectiveOptimizer[WarmStartT]
        The optimizer, pickled into the worker. Any state it mutates during the solve (a
        cached last result, say) is confined to the worker's copy.
     bounds : npt.NDArray[np.float64]
        Upper bounds on the auxiliary objectives for this grid point, of length N - 1.

    Returns
    -------
     point : FrontierPoint | None
        The solved point, or ``None`` when the solve failed in one of the two ways
        above.

    Raises
    ------
     Exception
        Any other exception from ``solve_with_bounds`` or ``point_objectives``
        propagates and is re-raised in the parent by ``_sweep_grid_parallel``.

    """
    try:
        ipm = optimizer.solve_with_bounds(bounds, x0=None)
        objs = optimizer.point_objectives(ipm, bounds)
        return FrontierPoint(
            objectives=objs,
            solution=ipm.solution.copy(),
            bounds=bounds.copy(),
            ipm_result=ipm,
        )
    except (OptimizationError, ProblemInfeasibleError):
        return None


__all__: list[str] = [
    "FrontierPoint",
    "FrontierResults",
    "MultiObjectiveOptimizer",
]
