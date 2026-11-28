"""Problems whose free variables leave a lineality space (issue #71).

With fewer linear rows than unknowns and no bounds, the constraints and a slack (or
row-space) objective are constant along null(A), so the barrier Hessian is singular
there. The Newton system is singular but consistent; both solvers once refused it, and
the primal-dual solver then reported a comfortably feasible problem as marginally
feasible.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import scipy.optimize
from parameterized import parameterized

from cvxium.exceptions import NewtonStepError
from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.linear_constraints import LinearInequalityConstraintLayer
from cvxium.problems.objective import LinearObjective, QuadraticObjective
from cvxium.problems.problem import Problem
from cvxium.problems.result import SolveStatus
from cvxium.solvers.barrier_solver import BarrierSolver, _BarrierAdapter
from cvxium.solvers.primal_dual_solver import PrimalDualSolver

SOLVERS = [("barrier", BarrierSolver), ("primal_dual", PrimalDualSolver)]
SHAPES = [(3, 2), (20, 10), (200, 100)]
CASES = [
    (f"{name}_{n}x{m}", solver, n, m) for name, solver in SOLVERS for n, m in SHAPES
]


def _rows(n: int, m: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """m < n random rows with x = 1 strictly feasible by 1; x = 0 is infeasible."""
    rng = np.random.default_rng(n + m)
    A = rng.standard_normal((m, n))
    return A, A @ np.ones(n) - 1.0


class LinealitySpaceTest(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_a_feasibility_problem_finds_a_point(
        self,
        _name: str,
        solver: type[BarrierSolver] | type[PrimalDualSolver],
        n: int,
        m: int,
    ) -> None:
        A, h = _rows(n, m)
        problem = Problem(ineq=[LinearInequalityConstraintLayer(A, cu=h)])
        result = problem.solve(solver=solver())
        self.assertEqual(result.status, SolveStatus.FEASIBLE, result.message)
        self.assertTrue(problem.is_feasible(result.solution))

    @parameterized.expand(CASES)
    def test_a_qp_reaches_phase_ii(
        self,
        _name: str,
        solver: type[BarrierSolver] | type[PrimalDualSolver],
        n: int,
        m: int,
    ) -> None:
        A, h = _rows(n, m)
        problem = Problem.minimize(
            QuadraticObjective(
                c=-np.ones(n), hessian_contribution=DiagonalSystem(np.ones(n))
            )
        ).subject_to(ineq=[LinearInequalityConstraintLayer(A, cu=h)])
        result = problem.solve(solver=solver())
        self.assertEqual(result.status, SolveStatus.OPTIMAL, result.message)
        self.assertTrue(problem.is_feasible(result.solution))

    @parameterized.expand(CASES)
    def test_a_bounded_lp_matches_highs(
        self,
        _name: str,
        solver: type[BarrierSolver] | type[PrimalDualSolver],
        n: int,
        m: int,
    ) -> None:
        """c = A^T y with y < 0: bounded, with the optimal set unbounded along null(A)."""
        A, h = _rows(n, m)
        c = A.T @ -np.random.default_rng(1).uniform(0.1, 1.0, m)
        reference = scipy.optimize.linprog(c, A_ub=A, b_ub=h, bounds=[(None, None)] * n)
        assert reference.status == 0
        problem = Problem.minimize(LinearObjective(c=c)).subject_to(
            ineq=[LinearInequalityConstraintLayer(A, cu=h)]
        )
        result = problem.solve(solver=solver())
        self.assertEqual(result.status, SolveStatus.OPTIMAL, result.message)
        self.assertAlmostEqual(
            result.objective_value,
            reference.fun,
            delta=1e-5 * max(1.0, abs(reference.fun)),
        )

    @parameterized.expand(SOLVERS)
    def test_an_unbounded_lp_is_certified(
        self, _name: str, solver: type[BarrierSolver] | type[PrimalDualSolver]
    ) -> None:
        """c = A^T y with y > 0: unbounded below (HiGHS agrees)."""
        A, h = _rows(3, 2)
        c = A.T @ np.array([0.5, 0.25])
        problem = Problem.minimize(LinearObjective(c=c)).subject_to(
            ineq=[LinearInequalityConstraintLayer(A, cu=h)]
        )
        result = problem.solve(solver=solver())
        self.assertEqual(result.status, SolveStatus.UNBOUNDED, result.message)


class BarrierSingularRetryTest(unittest.TestCase):
    """The barrier's damped retry raises the original error when it fails too."""

    def test_the_original_error_survives_a_failed_retry(self) -> None:
        A, h = _rows(3, 2)
        problem = Problem(ineq=[LinearInequalityConstraintLayer(A, cu=h)])
        original = NewtonStepError("H is singular (original)")
        calls: list[int] = []

        def failing(*_args: object, **_kwargs: object) -> None:
            calls.append(1)
            raise original if len(calls) == 1 else NewtonStepError("retry failed")

        with (
            patch.object(_BarrierAdapter, "_solve_newton_system", side_effect=failing),
            self.assertRaises(Exception) as ctx,
        ):
            problem.solve(solver=BarrierSolver())
        self.assertEqual(len(calls), 2)
        cause: BaseException | None = ctx.exception
        while cause is not None and cause is not original:
            cause = cause.__cause__
        self.assertIs(cause, original)
