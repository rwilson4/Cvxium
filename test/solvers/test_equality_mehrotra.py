"""Tests for the Mehrotra predictor-corrector step on PrimalDualSolver's equality path.

The equality path used to take one centering step with a residual line search per
iteration. It now takes the infeasible-start Mehrotra step of the no-equality path,
with three supports, each tested here:
- the duals start at λ = 1, moved onto the central path's direction at the
  objective's magnitude when λ = 1 is the wrong scale, so a badly scaled objective
  does not start far from dual feasibility;
- each direction is refined once on the full (unreduced) KKT system, whose
  elimination of Δλ divides by f and so amplifies error on near-active rows;
- a step that would raise the surrogate gap falls back to the centering step with
  its residual line search (exercised by the c x 1e6 cases of
  ``test_scale_invariance``, where it cuts iterations from 25 / 69 / 21 to 17 / 20 / 16).
"""

from __future__ import annotations

import unittest

import numpy as np

from cvxium.linalg.systems import DenseSystem, DiagonalSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import LinearObjective, QuadraticObjective
from cvxium.problems.problem import Problem
from cvxium.problems.quadratic_constraints import SquaredNormConstraintLayer
from cvxium.problems.result import SolveStatus
from cvxium.solvers.primal_dual_solver import PrimalDualSolver, _PrimalDualKernel
from cvxium.solvers.settings import OptimizationSettings, OuterSettings
from test.solvers.test_scale_invariance import _scaled_lp


def _equality_lp(c_scale: float) -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(5)
    n, m = 8, 10
    G = rng.standard_normal((m, n))
    x0 = rng.uniform(0.5, 1.5, n)
    problem = Problem(
        obj=LinearObjective(c=c_scale * rng.standard_normal(n)),
        ineq=[
            BoundsConstraintLayer(lb=0.0, dimension=n),
            LinearInequalityConstraintLayer(G, cu=G @ x0 + 1.0),
        ],
        eq=EqualityConstraintLayer(A=np.ones((1, n)), b=np.array([x0.sum()])),
    )
    return problem, x0


class CentralPathDualsTest(unittest.TestCase):
    """The starting duals scale with the objective, or fall back to λ = 1."""

    def test_duals_scale_linearly_with_a_large_objective(self) -> None:
        base_problem, x0, _ = _scaled_lp(0, c_scale=1e3)
        scaled_problem, _, _ = _scaled_lp(0, c_scale=1e5)
        base = PrimalDualSolver._central_path_duals(base_problem, x0)
        scaled = PrimalDualSolver._central_path_duals(scaled_problem, x0)
        self.assertTrue(np.all(base > 0.0))
        self.assertFalse(np.allclose(base, 1.0))  # the scaled start was chosen
        np.testing.assert_allclose(scaled, 100.0 * base, rtol=1e-10)

    def test_a_well_scaled_problem_keeps_unit_duals(self) -> None:
        problem, x0 = _equality_lp(1.0)
        lmbda = PrimalDualSolver._central_path_duals(problem, x0)
        np.testing.assert_array_equal(lmbda, np.ones(lmbda.size))

    def test_keeps_unit_duals_when_they_are_merely_small(self) -> None:
        # minimize x^2 / 2 - 10 x s.t. x^2 <= 1 from x = -0.2: ‖G^T 1‖ / ‖∇f0‖ is 0.04,
        # inside the range. Rescaling here (∇φ points along ∇f0) would double the
        # dual residual and stall the solve (test_mpc_damping_returns_to_zero).
        problem = Problem(
            obj=QuadraticObjective(
                c=np.array([-10.0]), hessian_contribution=DiagonalSystem(np.ones(1))
            ),
            ineq=[SquaredNormConstraintLayer(phi=1.0, dimension=1)],
        )
        lmbda = PrimalDualSolver._central_path_duals(problem, np.array([-0.2]))
        np.testing.assert_array_equal(lmbda, np.ones(1))

    def test_opposed_bounds_do_not_read_as_a_weak_unit_start(self) -> None:
        # At the center of a box each variable's two bound gradients, -e_i and +e_i,
        # cancel, so G^T 1 = 0 although the bounds are as strong as ever. The
        # random-sign probe sees through that and keeps λ = 1.
        n = 6
        problem = Problem(
            obj=QuadraticObjective(
                c=np.ones(n), hessian_contribution=DiagonalSystem(np.ones(n))
            ),
            ineq=[BoundsConstraintLayer(lb=-1.0, ub=1.0, dimension=n)],
        )
        x = np.zeros(n)
        gtl = _PrimalDualKernel.g_transpose_times(problem, x, np.ones(2 * n))
        self.assertEqual(float(np.linalg.norm(gtl)), 0.0)
        lmbda = PrimalDualSolver._central_path_duals(problem, x)
        np.testing.assert_array_equal(lmbda, np.ones(2 * n))

    def test_matches_magnitudes_on_the_centering_direction(self) -> None:
        # At c x 1e6 the unit start exerts a millionth of the objective's force.
        problem, x0 = _equality_lp(1e6)
        lmbda = PrimalDualSolver._central_path_duals(problem, x0)
        f = _PrimalDualKernel.concat_constraints(problem, x0)
        grad = problem.obj.gradient(x0)  # type: ignore[union-attr]
        gtl = _PrimalDualKernel.g_transpose_times(problem, x0, lmbda)
        ratio = float(np.linalg.norm(gtl) / np.linalg.norm(grad))
        self.assertAlmostEqual(ratio, 1.0, places=10)
        # ... along the central-path direction λ̂ = -1 / f.
        np.testing.assert_allclose(lmbda * (-f), (lmbda * (-f))[0], rtol=1e-12)
        result = PrimalDualSolver(
            OptimizationSettings(outer=OuterSettings(objective_scale="relative"))
        ).solve(problem, x0)
        self.assertIs(result.status, SolveStatus.OPTIMAL, result.message)


class FullSystemRefinementTest(unittest.TestCase):
    """The equality-path direction solves the unreduced KKT system near the boundary."""

    def test_full_kkt_residual_is_small_with_near_active_rows(self) -> None:
        rng = np.random.default_rng(9)
        n, m = 12, 20
        F = rng.standard_normal((n, 4))
        # Nonsingular: the layer's Hessian is a low-rank update on the objective's
        # DenseSystem, which must factor on its own (bounds do that in practice).
        Q = F @ F.T + 0.1 * np.eye(n)
        G = rng.standard_normal((m, n))
        x = rng.uniform(0.5, 1.5, n)
        slack = np.where(np.arange(m) < 8, 1e-9, 1.0)  # eight rows nearly active
        problem = Problem(
            obj=QuadraticObjective(
                c=rng.standard_normal(n), hessian_contribution=DenseSystem(Q)
            ),
            ineq=[LinearInequalityConstraintLayer(G, cu=G @ x + slack)],
            eq=EqualityConstraintLayer(A=np.ones((1, n)), b=np.array([x.sum()])),
        )
        f = _PrimalDualKernel.concat_constraints(problem, x)
        # Active rows weigh λ / -f = 1e12; the rest 1, so H is well defined.
        lmbda = np.where(np.arange(m) < 8, 1e3, 1.0)
        r_dual = rng.standard_normal(n)
        r_cent = rng.standard_normal(m)
        r_pri = rng.standard_normal(1)
        dx, dlam, dnu = PrimalDualSolver()._compute_pd_step_equality(
            problem, x, lmbda, f, r_dual, r_cent, r_pri
        )
        A = problem.eq.A  # type: ignore[union-attr]

        def top_residual(dx: np.ndarray, dlam: np.ndarray, dnu: np.ndarray) -> float:
            top = Q @ dx + G.T @ dlam + A.T @ dnu + r_dual
            scale = 1.0 + max(np.abs(G.T @ dlam).max(), np.abs(r_dual).max())
            return float(np.abs(top).max() / scale)

        # The unrefined reduced solve, done exactly (dense): the elimination's
        # division by f ~ 1e-9 leaves a visible error in the full system.
        H_pd = Q + G.T @ np.diag(lmbda / -f) @ G
        kkt = np.block([[H_pd, A.T], [A, np.zeros((1, 1))]])
        sol = np.linalg.solve(kkt, np.r_[-r_dual - G.T @ (r_cent / f), -r_pri])
        dx0, dnu0 = sol[:n], sol[n:]
        dlam0 = (r_cent - lmbda * (G @ dx0)) / f
        unrefined = top_residual(dx0, dlam0, dnu0)

        refined = top_residual(dx, dlam, dnu)
        self.assertGreater(unrefined, 1e-6)
        self.assertLess(refined, unrefined / 100.0)
        middle = -lmbda * (G @ dx) - f * dlam + r_cent
        bottom = A @ dx + r_pri
        self.assertLessEqual(np.abs(middle).max(), 1e-8 * (1.0 + np.abs(r_cent).max()))
        self.assertLessEqual(np.abs(bottom).max(), 1e-10)


if __name__ == "__main__":
    unittest.main()
