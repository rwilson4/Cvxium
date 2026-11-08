"""Tests for the convex frontier interpolator and its certificate."""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt

from cvxium.frontier.frontier_interpolation import (
    interpolate_frontier,
)


def _parabola(
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A convex test frontier with slopes known in closed form."""
    return x**2, 2.0 * x


class InterpolateFrontierTest(unittest.TestCase):
    """The bounds are the contract; the smooth curve only has to respect them."""

    def test_bounds_bracket_a_known_convex_frontier(self) -> None:
        """Chord above, tangent envelope below, everywhere in between."""
        x = np.linspace(-1.0, 1.0, 6)
        y, slopes = _parabola(x)

        env = interpolate_frontier(x, y, slopes=slopes, num=401)

        truth = env.x**2
        # The proof, sampled: a chord of a convex function never dips below it,
        # and a tangent never rises above it.
        self.assertTrue(np.all(env.chord >= truth - 1e-12))
        self.assertTrue(np.all(env.tangent <= truth + 1e-12))
        # Both bounds are tight AT the knots, where chord and tangent touch.
        self.assertAlmostEqual(float(env.chord[0]), float(truth[0]))
        self.assertAlmostEqual(float(env.tangent[0]), float(truth[0]))

    def test_slopes_buy_an_order_of_magnitude_over_the_chord(self) -> None:
        """Hermite with exact slopes beats the piecewise-linear plot outright.

        This is the whole reason to thread ``lambda`` through: the sweep
        already knows the derivative, so using it is free accuracy rather than
        a smoothing heuristic.
        """
        x = np.linspace(-1.0, 1.0, 6)
        y, slopes = _parabola(x)

        env = interpolate_frontier(x, y, slopes=slopes, num=401)

        truth = env.x**2
        chord_error = float(np.max(np.abs(env.chord - truth)))
        curve_error = float(np.max(np.abs(env.curve - truth)))
        self.assertLess(curve_error, chord_error / 10.0)

    def test_band_width_shrinks_quadratically_as_the_grid_refines(self) -> None:
        # The certificate has to be worth reading: doubling the grid should
        # visibly tighten it, not merely not-loosen it. For a smooth convex
        # curve the chord-to-tangent gap is O(h^2), so 2x the points is ~4x
        # tighter; assert a conservative 3x so curvature variation cannot flake.
        coarse_x = np.linspace(-1.0, 1.0, 5)
        fine_x = np.linspace(-1.0, 1.0, 9)

        coarse = interpolate_frontier(coarse_x, *_parabola(coarse_x))
        fine = interpolate_frontier(fine_x, *_parabola(fine_x))

        self.assertGreater(coarse.max_band_width, 3.0 * fine.max_band_width)

    def test_unknown_slopes_are_skipped_not_fatal(self) -> None:
        """A frontier that knows only some of its slopes still gets a bound.

        The nominal corner of a weighted-sum sweep has a vertical tangent
        (``lambda -> infinity``), so it genuinely cannot report a finite slope.
        That must cost only the tangent at that one point.
        """
        x = np.linspace(-1.0, 1.0, 6)
        y, slopes = _parabola(x)
        slopes[-1] = np.inf

        env = interpolate_frontier(x, y, slopes=slopes, num=201)

        truth = env.x**2
        self.assertTrue(np.all(env.tangent <= truth + 1e-12))
        self.assertTrue(np.all(np.isfinite(env.tangent)))
        self.assertTrue(np.isfinite(env.max_band_width))

    def test_a_slope_that_does_not_support_the_samples_is_dropped(self) -> None:
        """An inaccurate subproblem must loosen the bound, never break it.

        This is not hypothetical. A weight steeper than any slope the frontier
        attains drives the solution toward a vertex, and a barrier method stops
        just inside it; the steep line through that slightly-interior point
        cuts well above the vertex. Reported blindly it would certify a bound
        the frontier violates, at the very end of the sweep where a reader is
        least able to tell.
        """
        x = np.linspace(-1.0, 1.0, 6)
        y, slopes = _parabola(x)
        honest = interpolate_frontier(x, y, slopes=slopes, num=201)
        # Point 2's tangent is rotated far past anything this curve attains.
        slopes[2] = -50.0

        env = interpolate_frontier(x, y, slopes=slopes, num=201)

        truth = env.x**2
        self.assertTrue(np.all(env.tangent <= truth + 1e-12))
        # Dropped, not accommodated: the bound loosens to exactly what the
        # remaining five honest tangents support.
        self.assertGreater(env.max_band_width, honest.max_band_width)

    def test_samples_contradicting_convexity_certify_nothing(self) -> None:
        """When every tangent is rejected, say so rather than bound with none.

        A frontier sampled more finely than the solver resolves comes back
        bending the wrong way, and then no tangent supports the samples -- for a
        concave set of points every tangent but the one at its own knot cuts
        above the curve. Reporting a width from the survivors would be a bound
        derived from nothing, so ``inf`` is the honest answer. The curve is
        still drawn; only the guarantee is withheld.
        """
        x = np.linspace(0.0, 1.0, 6)
        y = -(x**2)
        slopes = -2.0 * x

        env = interpolate_frontier(x, y, slopes=slopes, num=101)

        self.assertTrue(np.all(np.isneginf(env.tangent)))
        self.assertEqual(env.max_band_width, np.inf)
        self.assertTrue(np.all(np.isfinite(env.curve)))

    def test_without_slopes_there_is_no_lower_bound(self) -> None:
        # PCHIP still smooths, but nothing is certified from below, and
        # max_band_width must say so rather than quietly reporting a number.
        x = np.linspace(-1.0, 1.0, 6)
        y, _ = _parabola(x)

        env = interpolate_frontier(x, y, num=101)

        self.assertTrue(np.all(np.isneginf(env.tangent)))
        self.assertEqual(env.max_band_width, np.inf)
        self.assertTrue(np.all(np.isfinite(env.curve)))

    def test_curve_never_escapes_the_certified_band(self) -> None:
        # A Hermite segment can bulge past its endpoints' bounds; the bounds are
        # proven and the curve is not, so the curve gets clipped.
        x = np.asarray([0.0, 0.1, 1.0, 3.0])
        y = np.asarray([1.0, 0.2, 0.05, 0.0])
        slopes = np.asarray([-20.0, -2.0, -0.05, -0.001])

        env = interpolate_frontier(x, y, slopes=slopes, num=301)

        self.assertTrue(np.all(env.curve <= env.chord + 1e-12))
        self.assertTrue(np.all(env.curve >= env.tangent - 1e-12))

    def test_unsorted_and_duplicated_input_is_normalized(self) -> None:
        # Frontier points arrive in solve order, and a corner can coincide with
        # a swept point. Sorting is the interpolator's job; on a tie the better
        # (lower) y wins, since that is the one actually on the frontier.
        x = np.asarray([1.0, -1.0, 0.0, 0.0])
        y = np.asarray([1.0, 1.0, 0.5, 0.0])
        slopes = np.asarray([2.0, -2.0, np.nan, 0.0])

        env = interpolate_frontier(x, y, slopes=slopes, num=101)

        self.assertAlmostEqual(float(env.x[0]), -1.0)
        self.assertAlmostEqual(float(env.x[-1]), 1.0)
        # The kept point at x = 0 is the y = 0 one, so the chord through the
        # middle is the lower pair of segments, not the y = 0.5 one.
        mid = int(np.argmin(np.abs(env.x)))
        self.assertAlmostEqual(float(env.chord[mid]), 0.0, places=6)

    def test_rejects_input_it_cannot_interpolate(self) -> None:
        x = np.linspace(0.0, 1.0, 4)
        y = x**2

        with self.assertRaisesRegex(ValueError, "at least two distinct"):
            interpolate_frontier(np.zeros(3), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "same length"):
            interpolate_frontier(x, y[:-1])
        with self.assertRaisesRegex(ValueError, "must be finite"):
            interpolate_frontier(np.asarray([0.0, np.nan, 1.0]), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "num must be at least 2"):
            interpolate_frontier(x, y, num=1)
        with self.assertRaisesRegex(ValueError, "slopes must match"):
            interpolate_frontier(x, y, slopes=np.zeros(3))


if __name__ == "__main__":
    unittest.main()
