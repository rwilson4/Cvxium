"""Tests for the Cvxium exception classes."""

import unittest

import numpy as np

from cvxium.exceptions import (
    BacktrackingLineSearchError,
    ConstraintBoundaryError,
    InteriorPointMethodError,
    InvalidDescentDirectionError,
    OptimizationError,
    SevereCurvatureError,
)


class ExceptionsTest(unittest.TestCase):
    """Fields and messages the solvers and their callers rely on."""

    def test_constraint_boundary_error_formats_its_message(self) -> None:
        """The inherited constructor and formatter keep the message verbatim."""
        error = ConstraintBoundaryError(message="too close to the boundary")
        self.assertIsInstance(error, BacktrackingLineSearchError)
        self.assertEqual(error.message, "too close to the boundary")
        self.assertEqual(str(error), "too close to the boundary")

    def test_invalid_descent_direction_error_reports_the_derivative(self) -> None:
        """The message shows the offending directional derivative with a delta."""
        error = InvalidDescentDirectionError(
            message="not a descent direction", grad_ft_dot_delta_x=0.25
        )
        self.assertEqual(error.message, "not a descent direction")
        self.assertEqual(error.grad_ft_dot_delta_x, 0.25)
        self.assertEqual(
            str(error),
            "not a descent direction (∇f^T Δx = 0.25 > 0, but should be <= 0)",
        )

    def test_severe_curvature_error_reports_both_improvements(self) -> None:
        """The message shows the required and the actual improvement."""
        error = SevereCurvatureError(
            message="insufficient decrease",
            required_improvement=1e-3,
            actual_improvement=-2e-4,
        )
        self.assertEqual(error.message, "insufficient decrease")
        self.assertEqual(
            str(error),
            "insufficient decrease (required improvement >= 0.001; "
            "actual improvement = -0.0002)",
        )

    def test_interior_point_method_error_carries_every_field(self) -> None:
        """The base constructor sets the shared fields; the subclass adds its own."""
        x = np.array([1.0, 2.0])
        error = InteriorPointMethodError(
            message="stalled", remaining_steps=3, suboptimality=0.5, last_iterate=x
        )
        self.assertIsInstance(error, OptimizationError)
        self.assertEqual(error.message, "stalled")
        self.assertEqual(error.remaining_steps, 3)
        self.assertEqual(error.suboptimality, 0.5)
        self.assertIs(error.last_iterate, x)
        self.assertIsNone(error.result)
        self.assertEqual(
            str(error), "stalled (3 step(s) remaining; suboptimality = 0.5)"
        )
