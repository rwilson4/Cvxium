"""A Phase I numerical breakdown is the answer "no feasible point found" (#72).

Under ``on_infeasible="raise"``, ``Solver._solve_phase1`` converts a Phase I
breakdown into ``ProblemMarginallyFeasibleError``. The primal-dual solver raises the
breakdown bare; the barrier engine wraps it as ``InteriorPointMethodError`` from
``CenteringStepError``, which the conversion once missed, so the two solvers gave
different answers on the same problem.
"""

from __future__ import annotations

import unittest

import numpy as np
from parameterized import parameterized

from cvxium.exceptions import (
    BacktrackingLineSearchError,
    CenteringStepError,
    InteriorPointMethodError,
    InvalidDescentDirectionError,
    NewtonStepError,
    RecessionDirectionError,
)
from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import QuadraticObjective
from cvxium.problems.problem import Problem
from cvxium.solvers.barrier_solver import BarrierSolver
from cvxium.solvers.optimization import (
    ProblemCertifiablyInfeasibleError,
    ProblemMarginallyFeasibleError,
)
from cvxium.solvers.primal_dual_solver import PrimalDualSolver
from cvxium.solvers.solver import _numerical_breakdown


def _empty_interior_problem() -> Problem:
    """Issue #72's reproduction: g'x = 1 as two opposed rows, two variables bounded.

    The feasible set is nonempty with no interior; the barrier's Phase I breaks
    down in its line search ("not a descent direction").
    """
    n = 5
    rng = np.random.default_rng(0)
    g = rng.uniform(0.5, 1.5, n)
    A = np.vstack([g, -g, rng.standard_normal((3, n))])
    h = np.r_[1.0, -1.0, 10.0, 10.0, 10.0]
    lb = np.r_[0.0, 0.0, -np.inf, -np.inf, -np.inf]
    return Problem.minimize(
        QuadraticObjective(
            c=-np.ones(n), hessian_contribution=DiagonalSystem(np.ones(n))
        )
    ).subject_to(
        ineq=[BoundsConstraintLayer(lb=lb), LinearInequalityConstraintLayer(A, cu=h)]
    )


def _wrapped(cause: BaseException) -> InteriorPointMethodError:
    """``cause`` as the barrier engine wraps it: IPM from centering from cause."""
    centering = CenteringStepError("centering failed", 10.0, np.zeros(2))
    centering.__cause__ = cause
    error = InteriorPointMethodError(
        message="Phase I centering step failed",
        remaining_steps=3,
        suboptimality=10.0,
        last_iterate=np.zeros(2),
    )
    error.__cause__ = centering
    return error


def _ray(unbounded: bool) -> RecessionDirectionError:
    return RecessionDirectionError(
        "ray",
        suboptimality=10.0,
        last_iterate=np.zeros(2),
        direction=np.array([0.0, 1.0]),
        objective_slope=-1.0 if unbounded else 0.0,
    )


class NumericalBreakdownTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("newton", NewtonStepError("H is singular")),
            ("line_search", BacktrackingLineSearchError("small steps")),
            ("descent", InvalidDescentDirectionError("not a descent direction", 0.5)),
        ]
    )
    def test_a_wrapped_breakdown_is_found(
        self, _name: str, breakdown: BaseException
    ) -> None:
        self.assertIs(_numerical_breakdown(_wrapped(breakdown)), breakdown)

    @parameterized.expand([("flat", False), ("unbounded", True)])
    def test_a_recession_ray_is_not_a_breakdown(
        self, _name: str, unbounded: bool
    ) -> None:
        """A flat ray must reach the box retry; neither ray is a breakdown."""
        ray = _ray(unbounded)
        ray.__cause__ = NewtonStepError("beneath the ray")
        self.assertIsNone(_numerical_breakdown(_wrapped(ray)))

    def test_a_failure_without_a_breakdown_is_left_alone(self) -> None:
        self.assertIsNone(
            _numerical_breakdown(
                InteriorPointMethodError(
                    message="ran out of steps",
                    remaining_steps=0,
                    suboptimality=10.0,
                    last_iterate=np.zeros(2),
                )
            )
        )


class Phase1BreakdownVerdictTest(unittest.TestCase):
    @parameterized.expand(
        [("barrier", BarrierSolver), ("primal_dual", PrimalDualSolver)]
    )
    def test_both_solvers_give_the_marginal_verdict(
        self, _name: str, solver: type[BarrierSolver] | type[PrimalDualSolver]
    ) -> None:
        with self.assertRaises(ProblemMarginallyFeasibleError) as ctx:
            _empty_interior_problem().solve(solver=solver())
        self.assertNotIsInstance(ctx.exception, ProblemCertifiablyInfeasibleError)

    def test_the_barrier_verdict_keeps_the_breakdown_and_the_last_iterate(
        self,
    ) -> None:
        problem = _empty_interior_problem()
        with self.assertRaises(ProblemMarginallyFeasibleError) as ctx:
            problem.solve(solver=BarrierSolver())
        error = ctx.exception
        self.assertIsInstance(error.__cause__, InteriorPointMethodError)
        breakdown = _numerical_breakdown(error.__cause__)  # type: ignore[arg-type]
        self.assertIsInstance(breakdown, BacktrackingLineSearchError)
        result = error.result
        self.assertIsNone(result.objective)
        self.assertEqual(result.solution.shape, (problem.dimension,))
        self.assertIn("last Phase I iterate", result.message)

    def test_continue_lets_the_barrier_breakdown_propagate(self) -> None:
        """``continue`` wants a best-effort point, which a breakdown cannot give."""
        with self.assertRaises(InteriorPointMethodError) as ctx:
            _empty_interior_problem().solve(
                solver=BarrierSolver(), on_infeasible="continue"
            )
        self.assertIsNotNone(_numerical_breakdown(ctx.exception))
