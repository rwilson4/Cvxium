"""Regression tests for QPs whose Hessian Q is singular.

With Q positive semidefinite but singular (some variables enter only linearly), the
Lagrangian is linear along null(Q), so the mini-Newton of ``evaluate_dual`` cannot
minimize it. That used to leave every dual bound at -inf, so:
- ``BarrierSolver`` reported multipliers from an arbitrary step (the "best" of all-inf
  dual values), far from stationary;
- ``PrimalDualSolver`` could not certify its gap, and reported ``MAX_ITERATIONS`` for
  a solve that had converged.
The fix certifies the bound through the Wolfe-dual correction, and the primal-dual
step solve refines once, since the reduced system is nearly singular along null(Q)
late in the solve.
"""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium.linalg.systems import DenseSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import QuadraticObjective
from cvxium.problems.problem import Problem
from cvxium.problems.result import SolveResult, SolveStatus
from cvxium.solvers.barrier_solver import BarrierSolver
from cvxium.solvers.primal_dual_solver import PrimalDualSolver

N, RANK, M = 30, 10, 60


def _problem(
    with_equality: bool,
) -> tuple[
    Problem, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """A random QP with rank-10 Q over 30 variables, bounds and 60 inequalities."""
    rng = np.random.default_rng(1)
    F = rng.standard_normal((N, RANK))
    F[RANK:] = 0.0  # the last 20 variables have no curvature
    Q = F @ F.T
    q = rng.standard_normal(N)
    G = rng.standard_normal((M, N))
    h = G @ rng.uniform(0.5, 1.0, N) + 0.5
    eq = (
        EqualityConstraintLayer(A=np.ones((1, N)), b=np.array([0.7 * N]))
        if with_equality
        else None
    )
    problem = Problem.minimize(
        QuadraticObjective(c=q, hessian_contribution=DenseSystem(Q))
    ).subject_to(
        eq=eq,
        ineq=[
            BoundsConstraintLayer(lb=0.0, dimension=N),
            LinearInequalityConstraintLayer(G, cu=h),
        ],
    )
    return problem, Q, q, G


def _stationarity_residual(
    problem: Problem,
    Q: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    G: npt.NDArray[np.float64],
    result: SolveResult,
) -> float:
    """Relative ‖Q x + q + G^T y - z + A^T ν‖ at the returned point and multipliers."""
    x = result.solution
    lam = result.inequality_multipliers
    z, y = lam[:N], lam[N:]
    terms = [Q @ x, q, G.T @ y, -z]
    if problem.eq is not None:
        terms.append(problem.eq.A.T @ result.equality_multipliers)
    scale = 1.0 + max(float(np.abs(t).max()) for t in terms)
    return float(np.abs(sum(terms)).max()) / scale


_CASES = [
    (f"{solver.__name__}_{'eq' if eq else 'ineq'}", solver, eq)
    for solver in (BarrierSolver, PrimalDualSolver)
    for eq in (False, True)
]


class SingularQuadraticTest(unittest.TestCase):
    """Both interior-point solvers certify and report duals with a singular Q."""

    @parameterized.expand(_CASES)
    def test_certified_optimal_with_stationary_duals(
        self, _: str, solver_cls: type, with_equality: bool
    ) -> None:
        problem, Q, q, G = _problem(with_equality)
        result = problem.solve(solver=solver_cls())
        self.assertIs(result.status, SolveStatus.OPTIMAL)
        self.assertTrue(np.isfinite(result.dual_value))
        gap = result.objective_value - result.dual_value
        self.assertGreaterEqual(gap, -1e-9)
        self.assertLessEqual(gap, 1e-5)
        self.assertLessEqual(_stationarity_residual(problem, Q, q, G, result), 1e-4)

    @parameterized.expand([("ineq", False), ("eq", True)])
    def test_dual_bounds_the_other_solvers_optimum(
        self, _name: str, with_equality: bool
    ) -> None:
        # Weak duality across solvers: each certified bound sits below the other
        # solver's (feasible) objective value.
        problem = _problem(with_equality)[0]
        barrier = problem.solve(solver=BarrierSolver())
        primal_dual = problem.solve(solver=PrimalDualSolver())
        self.assertLessEqual(barrier.dual_value, primal_dual.objective_value + 1e-9)
        self.assertLessEqual(primal_dual.dual_value, barrier.objective_value + 1e-9)


if __name__ == "__main__":
    unittest.main()
