"""Test the per-x ObjectivePoint cache in the barrier adapters."""

import unittest

import numpy as np

from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
)
from cvxium.problems.objective import (
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.solvers.barrier_solver import (
    _BarrierAdapter,
)
from cvxium.solvers.settings import (
    OptimizationSettings,
)


def _qp_problem() -> Problem:
    """Tiny PSD QP with the unconstrained optimum interior to ``x >= 0``."""
    M = 3
    w_target = np.ones(M)
    c = -w_target  # so x* (unconstrained) = w_target, interior to lb=0 (Q = I)
    return Problem(
        obj=QuadraticObjective(c=c, hessian_contribution=DiagonalSystem(np.ones(M))),
        ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
    )


class ObjectivePointCacheInvalidationTest(unittest.TestCase):
    """Pin that ``_BarrierAdapter._point`` keys on array *contents*, not identity.

    ``centering_step`` advances the iterate in place (``x += delta_x``), reusing
    the same array object across iterations. An identity-keyed cache freezes on
    the first iterate and returns a stale ``ObjectivePoint`` for every
    subsequent one, silently degrading Newton's method to a frozen-Jacobian
    iteration (linear convergence, FP-floor bouncing). The cache must recompute
    when ``x``'s contents change, even though the object is unchanged.
    """

    def test_point_recomputes_after_in_place_mutation(self) -> None:
        """An in-place update of ``x`` must invalidate the cached point."""
        problem = _qp_problem()
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        x = np.full(3, 0.5)

        obj_before = adapter.evaluate_objective(x)
        grad_before = adapter.gradient(x).copy()

        # Advance the iterate in place, exactly as ``centering_step`` does --
        # same array object, new contents.
        x += 0.25

        obj_after = adapter.evaluate_objective(x)
        grad_after = adapter.gradient(x)

        # Ground truth: a fresh (un-warmed) adapter evaluated at the new point.
        fresh = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        np.testing.assert_allclose(
            obj_after, fresh.evaluate_objective(x.copy()), rtol=1e-6, atol=1e-12
        )
        np.testing.assert_allclose(grad_after, fresh.gradient(x.copy()))
        # Guard against a no-op test: the values must actually have changed
        # (an identity-keyed cache would return the stale ``x = 0.5`` results).
        self.assertNotAlmostEqual(obj_after, obj_before)
        assert not np.allclose(grad_after, grad_before)
