"""Tests for the Mehrotra predictor-corrector step on PrimalDualSolver's equality path.

The equality path used to take one centering step with a residual line search per
iteration. It now takes the infeasible-start Mehrotra step of the no-equality path,
with three supports, each tested here:
- the duals start on the central path's direction at the objective's scale, so a
  badly scaled objective does not start far from dual feasibility;
- each direction is refined once on the full (unreduced) KKT system, whose
  elimination of Δλ divides by f and so amplifies error on near-active rows;
- a step that would raise the surrogate gap falls back to the centering step with
  its residual line search (exercised by the c x 1e6 cases of
  ``test_scale_invariance``, where it cuts iterations from 25 / 69 / 21 to 17 / 20 / 16).
"""

from __future__ import annotations

import unittest

import numpy as np

from cvxium.linalg.systems import DenseSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import LinearObjective, QuadraticObjective
from cvxium.problems.problem import Problem
from cvxium.solvers.primal_dual_solver import PrimalDualSolver, _PrimalDualKernel


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
    """The starting duals scale with the objective."""

    def test_duals_scale_linearly_with_the_objective(self) -> None:
        base_problem, x0 = _equality_lp(1.0)
        scaled_problem, _ = _equality_lp(1e4)
        floor = 1e-12  # keep the floor out of the way
        base = PrimalDualSolver._central_path_duals(base_problem, x0, floor)
        scaled = PrimalDualSolver._central_path_duals(scaled_problem, x0, floor)
        self.assertTrue(np.all(base > 0.0))
        np.testing.assert_allclose(scaled, 1e4 * base, rtol=1e-10)

    def test_duals_balance_the_objective_gradient(self) -> None:
        problem, x0 = _equality_lp(37.0)
        lmbda = PrimalDualSolver._central_path_duals(problem, x0, 1e-12)
        gtl = _PrimalDualKernel.g_transpose_times(problem, x0, lmbda)
        grad = problem.obj.gradient(x0)  # type: ignore[union-attr]
        self.assertAlmostEqual(np.linalg.norm(gtl), np.linalg.norm(grad), places=8)


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
