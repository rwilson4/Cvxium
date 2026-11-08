"""Smooth interpolation and rigorous bounds for a two-objective Pareto frontier.

A traced frontier is a handful of solved points. Plotting it means joining them, and the
usual straight segments quietly assert something false: that the frontier is piecewise
linear. It is not, and the error is one-sided, so the picture is biased rather than
merely coarse.

Convexity fixes this. A two-objective Pareto boundary in minimization coordinates is
convex, which turns the interpolation problem into a bounding problem with an exact
answer on both sides:
- Every chord between two attainable points lies on or above the boundary, so the
  piecewise-linear plot is an upper bound.
- Every tangent line lies on or below the boundary, globally and not just near its
  point of contact, so the upper envelope of the tangents is a lower bound.
Between consecutive points the true frontier is therefore trapped in a lens, and the
lens width is a measured error bar rather than a guess. Shade it and the plot reports
its own resolution; take a curve through it and the curve is certified.

The tangent slopes are not an extra burden: a weighted-sum sweep already knows them.
Minimizing
   f_0 + λ f_1
puts the solution where the boundary's tangent has slope
   df_0 / df_1 = -λ,
so the sweep's own control variable is the derivative, exactly, by the envelope theorem.
An epsilon-constraint sweep gets the same number from the KKT multiplier on its bound.
Supplying these turns the interpolant from a fit into an interpolation of both value and
slope.

Everything here works in Cvxium's native minimization convention. A caller plotting a
maximized objective negates both axes, which swaps the two bounds: the chord becomes the
lower bound and the tangent envelope the upper one.

"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator


@dataclass(frozen=True)
class FrontierEnvelope:
    """Smooth frontier curve together with the bounds that certify it.

    The four arrays share the dense grid ``x`` and use the same coordinate convention as
    the data passed to ``interpolate_frontier``.

    Fields
    ------
     x : npt.NDArray[np.float64]
        Dense grid of length num, spanning the auxiliary objective from the smallest to
        the largest value among the input points.
     chord : npt.NDArray[np.float64]
        The piecewise-linear interpolant through the points, which is what joining them
        with straight segments draws. An upper bound on a convex frontier.
     tangent : npt.NDArray[np.float64]
        Upper envelope of the verified tangent lines. A lower bound on a convex
        frontier; -inf everywhere when no slope was supplied or none survived
        verification.
     curve : npt.NDArray[np.float64]
        The smooth interpolant, clipped into [tangent, chord] so that it cannot assert
        anything the bounds rule out.
     max_band_width : float
        Widest gap chord - tangent, as a fraction of the range of the primary objective
        across the input points: 0.02 means the frontier is pinned everywhere to within
        2% of its own vertical extent. A rigorous replacement for judging by eye whether
        the grid was fine enough. inf when there is no tangent bound, and 0.0 when the
        primary objective is constant.

    """

    x: npt.NDArray[np.float64]
    chord: npt.NDArray[np.float64]
    tangent: npt.NDArray[np.float64]
    curve: npt.NDArray[np.float64]
    max_band_width: float


def _normalize_samples(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    slopes: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate the samples, sort them by x, and collapse duplicate abscissae.

    Frontier points arrive in solve order, and two of them can land on the same x: a
    degenerate stretch, or a corner coinciding with the first swept point. An
    interpolator needs strictly increasing knots, so each tie collapses to the point
    with the lowest y, which is the one actually on the frontier, along with that
    point's slope.

    Parameters
    ----------
     x : npt.NDArray[np.float64]
        Auxiliary objective at each point, of length n.
     y : npt.NDArray[np.float64]
        Primary objective at each point, of length n.
     slopes : npt.NDArray[np.float64] | None
        Tangent slope at each point, of length n, or ``None`` when none are known.

    Returns
    -------
     x : npt.NDArray[np.float64]
        The distinct abscissae, strictly increasing.
     y : npt.NDArray[np.float64]
        The primary objective at each abscissa.
     m : npt.NDArray[np.float64]
        The slope at each abscissa, nan wherever none is known.

    Raises
    ------
     ValueError
        When x and y are not one-dimensional arrays of the same length, contain
        non-finite values, or have fewer than two distinct x values, or when ``slopes``
        does not match x in shape.

    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(
            f"x and y must be one-dimensional and the same length; got "
            f"{x.shape} and {y.shape}."
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite.")
    if slopes is None:
        m = np.full(x.shape, np.nan, dtype=np.float64)
    else:
        m = np.asarray(slopes, dtype=np.float64)
        if m.shape != x.shape:
            raise ValueError(
                f"slopes must match x in shape; got {m.shape} and {x.shape}."
            )

    order = np.argsort(x, kind="stable")
    x, y, m = x[order], y[order].copy(), m[order].copy()
    keep = np.ones(x.size, dtype=bool)
    keep[1:] = x[1:] > x[:-1]
    for i in np.flatnonzero(~keep):
        j = i - 1
        while not keep[j]:
            j -= 1
        if y[i] < y[j]:
            y[j], m[j] = y[i], m[i]
    x, y, m = x[keep], y[keep], m[keep]
    if x.size < 2:
        raise ValueError(
            f"need at least two distinct x values to interpolate; got {x.size}."
        )
    return x, y, m


def _supports_every_sample(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Identify the tangents that lie below every sampled point.

    A reported slope is a valid bound only if the point that reported it really
    minimizes
       y + λ x;
    equivalently, its tangent line must pass on or below every other solved point. That
    is not automatic. A barrier method never reaches a vertex of the attainable set, so
    a weight steeper than anything the frontier attains leaves its solution just inside
    the corner, and a steep line through a point slightly inside the corner passes well
    above the corner itself. Trusting it would make the certificate false, at exactly
    the end of the sweep where a reader is most likely to be misled.

    So the claim is checked rather than assumed, against the only evidence available:
    the other solved points. A tangent that fails is dropped, which loosens the bound
    but never corrupts it. The check is one n × n comparison over a handful of points,
    with a tolerance of 1e-6 max(max |y|, 1) to absorb the accuracy of the solves that
    produced the samples.

    Parameters
    ----------
     x : npt.NDArray[np.float64]
        Auxiliary objective at each sample, of length n.
     y : npt.NDArray[np.float64]
        Primary objective at each sample, of length n.
     m : npt.NDArray[np.float64]
        Tangent slope at each sample, nan where unknown.

    Returns
    -------
     supports : npt.NDArray[np.bool_]
        Mask of length n, True where the slope is known and its tangent passes on or
        below every sample, to within the tolerance.

    """
    known = np.isfinite(m)
    if not bool(np.any(known)):
        return known
    # Slack matched to solver accuracy, not to floating point: every sample is
    # itself the output of an iterative solve that stopped at a tolerance, so a
    # point can sit a hair below a genuinely supporting tangent. Too tight a
    # test here rejects honest tangents -- including the robust corner's
    # horizontal one -- and leaves the bound vacuous.
    scale = float(np.max(np.abs(y))) if y.size else 0.0
    tol = 1e-6 * max(scale, 1.0)
    # lines[i, j] is the tangent at point i evaluated at point j's abscissa.
    lines = np.where(
        known[:, None],
        y[:, None] + np.where(known, m, 0.0)[:, None] * (x[None, :] - x[:, None]),
        -np.inf,
    )
    return known & np.all(lines <= y[None, :] + tol, axis=1)


def interpolate_frontier(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    slopes: npt.NDArray[np.float64] | None = None,
    num: int = 200,
) -> FrontierEnvelope:
    """Interpolate a convex Pareto frontier and bound the interpolation error.

    Parameters
    ----------
     x : npt.NDArray[np.float64]
        Auxiliary objective (the swept axis) at each solved point, of length n, in
        minimization coordinates. Need not be sorted; duplicates are collapsed, keeping
        the lowest y at each x.
     y : npt.NDArray[np.float64]
        Primary objective at each solved point, of length n, in minimization
        coordinates.
     slopes : npt.NDArray[np.float64] | None
        Tangent slope dy / dx at each point, of length n. For a weighted-sum sweep
        minimizing y + λ x this is -λ. Non-finite entries mean "unknown here" and are
        skipped when building the tangent bound, so a frontier that knows only some of
        its slopes still gets a partial certificate. Supplied slopes are verified
        against the other samples before being trusted (see ``_supports_every_sample``)
        and silently dropped if they fail, so an inaccurate subproblem loosens the bound
        rather than invalidating it. Wherever no verified slope is available, including
        when ``slopes`` is ``None``, the smooth curve uses shape-preserving (PCHIP)
        slopes.
     num : int
        Number of points in the returned dense grid.

    Returns
    -------
     envelope : FrontierEnvelope
        The dense grid, the chord and tangent bounds, the certified smooth curve, and
        the widest band width.

    Raises
    ------
     ValueError
        When ``num`` < 2, or when the samples are invalid (see ``_normalize_samples``):
        arrays of different lengths, non-finite x or y, or fewer than two distinct x
        values.

    Notes
    -----
    Convexity is assumed, not checked: it is a property of the problem (a convex
    decision set and convex objectives, in minimization coordinates), not of the
    samples, and a near-degenerate sample can violate it through solver noise without
    the frontier being non-convex. When the samples do contradict it, the tangent bound
    can rise above the chord between two samples, and the curve is clipped to the chord
    there. Because the verified tangents lie below every sample, ``max_band_width``
    stays nonnegative to within the tolerance and does not reveal such a crossing;
    compare ``tangent`` with ``chord`` directly to detect one.

    """
    if num < 2:
        raise ValueError(f"num must be at least 2, got {num}.")
    x, y, m = _normalize_samples(x, y, slopes)

    grid = np.linspace(x[0], x[-1], num)

    # Upper bound: the piecewise-linear interpolant a plot draws today. Every
    # chord of a convex function lies on or above it.
    chord = np.interp(grid, x, y)

    # Lower bound: for a convex function each tangent lies globally below the
    # curve, so the highest tangent at any abscissa is the tightest such bound.
    # Points whose slope is unknown contribute nothing and are dropped.
    known = np.isfinite(m) & _supports_every_sample(x, y, m)
    if bool(np.any(known)):
        lines = y[known, None] + m[known, None] * (grid[None, :] - x[known, None])
        tangent = lines.max(axis=0)
    else:
        tangent = np.full(grid.shape, -np.inf)

    # Smooth curve: Hermite through value AND slope where the slope is known.
    # PCHIP supplies shape-preserving slopes for the rest, which is what a
    # slope-free interpolator would have used everywhere.
    if not bool(np.all(known)):
        fallback = PchipInterpolator(x, y).derivative()(x)
        m = np.where(known, m, fallback)
    curve = CubicHermiteSpline(x, y, m)(grid)
    # A Hermite segment can bulge outside the certified band even with exact
    # endpoint data; the bounds are proven, so they win.
    curve = np.clip(curve, tangent, chord)

    y_range = float(y.max() - y.min())
    if not np.isfinite(tangent).all():
        max_band_width = np.inf
    elif y_range > 0.0:
        max_band_width = float(np.max(chord - tangent) / y_range)
    else:
        max_band_width = 0.0

    return FrontierEnvelope(
        x=grid,
        chord=chord,
        tangent=tangent,
        curve=curve,
        max_band_width=max_band_width,
    )
