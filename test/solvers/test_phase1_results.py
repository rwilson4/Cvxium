"""The main result carries the Phase I results that found its start, nested by layer.

Phase I cascades over the inequality layers: each layer's Phase I starts from a point
the earlier layers' Phase I found. ``SolveResult.phase1`` records that chain, so
``result.phase1.phase1`` is the Phase I that seeded ``result.phase1``.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np
from parameterized import parameterized

from cvxium.linalg.systems import DiagonalSystem
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

if TYPE_CHECKING:
    from cvxium.solvers.solver import Solver

N = 6


def _problem(
    with_equality: bool = False, margin: float = 0.05
) -> tuple[Problem, np.ndarray]:
    """Bounds and two linear layers around an interior point x_c; returns x_c.

    The margin is thin, so a point that satisfies the bounds (such as the bounds
    layer's own interior point) does not satisfy the linear layers, and each needs
    its own Phase I.
    """
    rng = np.random.default_rng(0)
    G1, G2 = rng.standard_normal((4, N)), rng.standard_normal((3, N))
    x_c = rng.uniform(1.0, 2.0, N)
    eq = (
        EqualityConstraintLayer(A=np.ones((1, N)), b=np.array([x_c.sum()]))
        if with_equality
        else None
    )
    problem = Problem.minimize(
        QuadraticObjective(
            c=rng.standard_normal(N), hessian_contribution=DiagonalSystem(np.ones(N))
        )
    ).subject_to(
        eq=eq,
        ineq=[
            BoundsConstraintLayer(lb=0.0, dimension=N),
            LinearInequalityConstraintLayer(G1, cu=G1 @ x_c + margin),
            LinearInequalityConstraintLayer(G2, cu=G2 @ x_c + margin),
        ],
    )
    return problem, x_c


def _chain(result: SolveResult) -> list[SolveResult]:
    """The Phase I results below ``result``, outermost first."""
    levels = []
    current = result.phase1
    while current is not None:
        levels.append(current)
        current = current.phase1
    return levels


_SOLVERS: list[tuple[str, type[Solver]]] = [
    ("barrier", BarrierSolver),
    ("primal_dual", PrimalDualSolver),
]


class Phase1ResultTest(unittest.TestCase):
    """``SolveResult.phase1`` holds the Phase I cascade."""

    @parameterized.expand(_SOLVERS)
    def test_a_feasible_start_records_no_phase1(
        self, _name: str, solver: type[Solver]
    ) -> None:
        problem, x_c = _problem()
        result = problem.solve(solver=solver(), x0=x_c)
        self.assertIs(result.status, SolveStatus.OPTIMAL)
        self.assertIsNone(result.phase1)

    @parameterized.expand(
        [
            (f"{name}_{eq}", solver, eq)
            for name, solver in _SOLVERS
            for eq in (False, True)
        ]
    )
    def test_an_infeasible_start_records_the_cascade(
        self, _name: str, solver: type[Solver], with_equality: bool
    ) -> None:
        problem, _ = _problem(with_equality)
        result = problem.solve(solver=solver(), x0=-np.ones(N))
        self.assertIs(result.status, SolveStatus.OPTIMAL)
        levels = _chain(result)
        # At least the active layer's Phase I and one beneath it, and some level
        # did real work (not the barrier's former trivial pass-through).
        self.assertGreaterEqual(len(levels), 2)
        self.assertTrue(any(level.n_iterations > 0 for level in levels))
        for level in levels:
            self.assertIs(level.status, SolveStatus.FEASIBLE, level.message)
        # The outermost level found the main solve's start: feasible for every layer.
        self.assertTrue(problem.is_feasible(levels[0].solution))
        # Deeper levels are feasible for the leading layers they cascaded over.
        leading = Problem(ineq=problem.ineq[:-1], eq=problem.eq)
        self.assertTrue(leading.is_feasible(levels[-1].solution))

    def test_the_cascade_survives_serialization(self) -> None:
        problem, _ = _problem()
        result = problem.solve(solver=PrimalDualSolver(), x0=-np.ones(N))
        restored = SolveResult.deserialize(result.serialize())
        original, copied = _chain(result), _chain(restored)
        self.assertEqual(len(copied), len(original))
        for a, b in zip(original, copied, strict=True):
            self.assertEqual(a.n_iterations, b.n_iterations)
            np.testing.assert_array_equal(a.solution, b.solution)

    @parameterized.expand(_SOLVERS)
    def test_a_feasibility_problem_returns_its_own_cascade(
        self, _name: str, solver: type[Solver]
    ) -> None:
        # With no objective the result is the Phase I result itself; its phase1 is
        # the cascade beneath the last layer.
        problem, _ = _problem()
        feasibility = Problem(ineq=problem.ineq)
        result = feasibility.solve(solver=solver(), x0=-np.ones(N))
        self.assertTrue(problem.is_feasible(result.solution))
        self.assertGreaterEqual(len(_chain(result)), 1)


if __name__ == "__main__":
    unittest.main()
