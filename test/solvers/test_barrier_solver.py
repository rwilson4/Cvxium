"""Test BarrierSolver."""

import unittest
import warnings
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import scipy.sparse
from parameterized import parameterized

from cvxium.diagnostics.trace import ListTrace
from cvxium.exceptions import (
    CenteringStepError,
    InteriorPointMethodError,
)
from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    MatrixSystem,
    SparseSaddleFactory,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    ImbalanceConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    Objective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.problems.quadratic_constraints import (
    NormConstraintLayer,
    SquaredNormConstraintLayer,
)
from cvxium.problems.result import (
    SolveResult,
    SolveStatus,
)
from cvxium.solvers.barrier_solver import (
    _DUAL_REGION_DECREMENT_SQ_MAX,
    BarrierSolver,
    _BarrierAdapter,
    _raise_to_objective_hint,
)
from cvxium.solvers.optimization import (
    ProblemCertifiablyInfeasibleError,
    _CenteringOutcome,
)
from cvxium.solvers.settings import (
    BarrierInitStrategy,
    InnerSettings,
    OptimizationSettings,
    OuterSettings,
)
from cvxium.solvers.solver import EqualitySolver

if TYPE_CHECKING:
    from cvxium.problems.constraint_layers import (
        ConstraintLayer,
    )


def _make_psd_matrix(rng: np.random.Generator, M: int) -> npt.NDArray[np.float64]:
    """Sample a random symmetric PD matrix with bounded condition number."""
    A = rng.standard_normal((M, M))
    return np.asarray(A @ A.T + M * np.eye(M))


class _LogObjective(Objective):
    r"""``f_0(w) = -sum log(w_i)`` -- strictly convex, restricted domain ``w > 0``.

    A restricted-domain objective for exercising the dual mini-Newton's in-domain
    backtrack. ``evaluate`` returns ``+inf`` off-domain (the extended-value
    extension), so an overshooting Newton step is rejected by the value check. The
    diagonal Hessian ``diag(1 / w^2)`` makes the baseline solve a diagonal scale.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        if np.any(w <= 0.0):
            return float("inf")
        return float(-np.sum(np.log(w)))

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(-1.0 / w)

    def hessian_multiply(
        self, w: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.asarray(v / np.square(w))

    def hessian_contribution(self, w: npt.NDArray[np.float64]) -> MatrixSystem:
        r"""``nabla^2 f_0 = diag(1 / w^2)`` as a ``DiagonalSystem``."""
        return DiagonalSystem(1.0 / np.square(w))

    def btls_keep_in_domain(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
    ) -> float:
        r"""Largest ``s`` with ``w + s\,dw > 0`` (domain ``w > 0``); ``inf`` if none binds."""
        biting = dw < 0.0
        if not np.any(biting):
            return float("inf")
        return float(np.min(w[biting] / -dw[biting]))


class BarrierSolverTest(unittest.TestCase):
    """Tests for `BarrierSolver` -- IPM over a `Problem`, auto-dispatching on equality."""

    def test_accepts_problem_with_equality(self) -> None:
        """`solve` auto-dispatches to the equality path when the Problem has ``A w = b``.

        The unified ``BarrierSolver`` no longer rejects an equality Problem (the
        old ``EqualityBarrierSolver``-era behavior); it solves it. Here the
        objective pushes both coordinates toward zero, so the optimum is the
        interior midpoint ``(0.5, 0.5)`` of the ``w_0 + w_1 = 1`` segment.
        """
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(2), hessian_contribution=DiagonalSystem(np.ones(2))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=2)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = BarrierSolver().solve(problem=problem, x0=np.array([0.5, 0.5]))
        # Minimizing 0.5 ||w||^2 on w_0 + w_1 = 1, w >= 0 gives w = (0.5, 0.5).
        np.testing.assert_allclose(result.solution, [0.5, 0.5], rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual((A @ result.solution).item(), 1.0, delta=1e-6)

    def test_rejects_equality_collaborator_without_equality_problem(self) -> None:
        """`_BarrierAdapter` rejects an ``equality`` collaborator on an eq-less Problem.

        Passing an ``EqualitySolver`` while ``problem.eq is None`` is an
        inconsistency caught at construction, rather than failing opaquely later in
        ``calculate_newton_step``.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(2), hessian_contribution=DiagonalSystem(np.ones(2))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=2)],
        )
        stray = EqualitySolver(
            constraint=EqualityConstraintLayer(
                A=np.array([[1.0, 1.0]]), b=np.array([1.0])
            )
        )
        with self.assertRaisesRegex(ValueError, "no equality layer"):
            _BarrierAdapter(
                problem=problem, settings=OptimizationSettings(), equality=stray
            )

    def test_rejects_problem_without_constraints(self) -> None:
        """`solve` raises when the Problem has no inequality constraint layers."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
        )
        with self.assertRaisesRegex(ValueError, "at least one inequality"):
            BarrierSolver().solve(problem=problem, x0=np.zeros(3))

    def test_one_solver_handles_inequality_and_equality_problems(self) -> None:
        """A single ``BarrierSolver`` instance solves both problem shapes.

        Exercises the auto-dispatch: the same instance first solves an
        inequality-only Problem (no-equality branch of ``_BarrierAdapter``) and
        then an inequality+equality Problem (equality branch), each to its known
        optimum. Proves the merged adapter branches correctly per-Problem rather
        than per-solver.
        """
        solver = BarrierSolver()

        # Inequality-only: min 0.5||w - target||^2 s.t. w >= 0 with target > 0,
        # so the unconstrained optimum (= target) is feasible.
        target = np.array([1.0, 2.0, 3.0])
        ineq_problem = Problem(
            obj=QuadraticObjective(
                c=-target, hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        ineq_result = solver.solve(problem=ineq_problem, x0=np.full(3, 0.5))
        np.testing.assert_allclose(ineq_result.solution, target, rtol=1e-5, atol=1e-5)

        # Inequality+equality: min 0.5||w||^2 s.t. w >= 0, sum(w) = 1. The optimum
        # is the uniform point (1/3, 1/3, 1/3) on the simplex.
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        eq_problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        eq_result = solver.solve(problem=eq_problem, x0=np.full(3, 1.0 / 3.0))
        np.testing.assert_allclose(
            eq_result.solution, np.full(3, 1.0 / 3.0), rtol=1e-5, atol=1e-5
        )
        self.assertAlmostEqual((A @ eq_result.solution).item(), 1.0, delta=1e-6)

    @parameterized.expand(
        [
            (1101, 5),
            (1102, 25),
            (1103, 50),
        ],
    )
    def test_solves_bound_constrained_quadratic_unconstrained_optimum(
        self, seed: int, M: int
    ) -> None:
        """When the unconstrained optimum is feasible, BarrierSolver finds it."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        # Pick c so that the unconstrained optimum w* = -Q^{-1} c is positive.
        # If we set w_target > 0 and c = -Q @ w_target, then w* = w_target.
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = rng.uniform(0.5, 1.0, size=M)
        result = BarrierSolver().solve(problem=problem, x0=x0)

        np.testing.assert_allclose(result.solution, w_target, rtol=1e-5, atol=1e-5)

    @parameterized.expand(
        [
            (1131, 5),
            (1132, 25),
        ],
    )
    def test_solve_without_x0_constructs_phase1(self, seed: int, M: int) -> None:
        """With no ``x0``, BarrierSolver builds a feasible start from its layers.

        Exercises ``_solve_feasibility`` -- Phase I via the last constraint
        layer's ``make_phase1_solver`` dispatch.
        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        result = BarrierSolver().solve(problem=problem)

        np.testing.assert_allclose(result.solution, w_target, rtol=1e-5, atol=1e-5)

    @parameterized.expand(
        [
            (1111, 5),
            (1112, 20),
        ],
    )
    def test_runs_phase1_when_x0_infeasible(self, seed: int, M: int) -> None:
        """When ``x0`` is infeasible, BarrierSolver runs Phase I and still solves.

        Phase I now uses the layered ``make_phase1_solver`` dispatch (the
        recursive ``Problem.make_phase1`` route that failed at ``t = 1`` was
        retired); see ``BarrierSolver._solve_feasibility``.
        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        # x0 is infeasible -- some components below the lower bound.
        x0 = rng.uniform(-2.0, -0.5, size=M)
        assert not problem.is_feasible(x0)

        result = BarrierSolver().solve(problem=problem, x0=x0)
        # Should still converge to the unconstrained optimum w_target.
        np.testing.assert_allclose(result.solution, w_target, rtol=1e-5, atol=1e-5)

    def test_phase1_with_linear_inequality_active_layer(self) -> None:
        """Structured Phase I feasibilizes a multi-row LinearInequalityConstraintLayer.

        The active (last) layer is a two-row ``LinearInequalityConstraintLayer``, a
        ``SlackConstraintLayer`` whose ``hessian_contribution`` is a rank-p
        ``LowRankUpdatedSystem``. From an infeasible ``x0`` the structured slack-form
        Phase I must route through ``_SlackActiveLayer``'s rank-p branch and land
        strictly feasible, converging to the same optimum reached from a feasible
        start. No prior test exercised this branch end-to-end (CDM's sales solver is
        the first production consumer).
        """
        M = 2
        Q = np.eye(M)
        w_target = np.array([2.0, 2.0])
        c = -Q @ w_target  # minimizes (1/2) ||w - w_target||^2 up to a constant
        # Two-row linear inequality: w0 + w1 <= 1 (binds) and w0 - w1 <= 5.
        A = np.array([[1.0, 1.0], [1.0, -1.0]])
        cu = np.array([1.0, 5.0])
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[
                BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M),
                LinearInequalityConstraintLayer(A=A, cu=cu),
            ],
        )
        feasible_x0 = np.array([0.1, 0.1])  # strictly inside every constraint
        assert problem.is_feasible(feasible_x0)
        reference = BarrierSolver().solve(problem=problem, x0=feasible_x0).solution

        infeasible_x0 = np.array([3.0, 3.0])  # bounds-ok but violates w0 + w1 <= 1
        assert not problem.is_feasible(infeasible_x0)
        result = BarrierSolver().solve(problem=problem, x0=infeasible_x0)

        # Phase I produced a strictly feasible iterate and Phase II reached the
        # same constrained optimum as the feasible-start solve.
        assert problem.is_feasible(result.solution)
        np.testing.assert_allclose(result.solution, reference, rtol=1e-5, atol=1e-5)

    def test_solves_corner_case_lp(self) -> None:
        """LP `min c^T w  s.t. lb <= w <= ub` finds a boundary solution.

        With c = [1, 1, 1], the optimum is at the lower bound ``lb`` for each
        component (driven by the upper-bound barrier from ub). BarrierSolver
        should converge near the lower bound.

        """
        M = 5
        c = np.ones(M)
        problem = Problem(
            obj=LinearObjective(c=c),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M)],
        )
        x0 = 5.0 * np.ones(M)  # interior
        result = BarrierSolver().solve(problem=problem, x0=x0)

        # Solution should be near the lower bound (within IPM tolerance).
        np.testing.assert_allclose(result.solution, np.zeros(M), atol=1e-3)

    def test_an_infinite_upper_bound_is_no_bound(self) -> None:
        r"""Infinite entries in ``ub`` leave those components uncapped.

        The same LP as above with two components uncapped. An infinite bound kept
        as a row made the barrier `-\infty` everywhere and the LP dual NaN, which
        the outer loop read as a zero duality gap after one centering step.
        """
        M = 5
        ub = np.array([10.0, np.inf, 10.0, np.inf, 10.0])
        problem = Problem(
            obj=LinearObjective(c=np.ones(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=ub)],
        )
        result = BarrierSolver().solve(problem=problem, x0=5.0 * np.ones(M))

        np.testing.assert_allclose(result.solution, np.zeros(M), atol=1e-3)
        objective = result.objective
        assert objective is not None
        self.assertTrue(np.isfinite(objective.suboptimality))
        self.assertLess(objective.suboptimality, 1e-3)


class StructuredBarrierSolveTest(unittest.TestCase):
    r"""The barrier adapter routes a Problem with a structured (MatrixSystem-declaring) objective through ``build_hessian_system``.

    Proves the wired ``_BarrierAdapter`` path drives a full barrier solve to the
    analytic optimum.
    """

    @parameterized.expand(
        [
            (3121, 5),
            (3122, 25),
            (3123, 50),
        ],
    )
    def test_end_to_end_diagonal_qp(self, seed: int, M: int) -> None:
        """A full barrier solve over a structured Problem hits the analytic optimum."""
        rng = np.random.default_rng(seed)
        q = rng.uniform(0.5, 3.0, size=M)
        w_target = rng.uniform(1.0, 5.0, size=M)  # positive => bound inactive
        c = -q * w_target  # so w* = -c / q = w_target
        diagonal = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DiagonalSystem(q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = rng.uniform(0.5, 1.0, size=M)

        result = BarrierSolver().solve(problem=diagonal, x0=x0)

        np.testing.assert_allclose(result.solution, w_target, rtol=1e-5, atol=1e-5)


def _project_onto_equality(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Closest point to ``x`` satisfying ``A y = b`` (least-norm projection)."""
    nu = np.linalg.solve(A @ A.T, A @ x - b)
    return np.asarray(x - A.T @ nu)


class BarrierSolverWithEqualityTest(unittest.TestCase):
    """Tests for `BarrierSolver` -- IPM with equality constraints."""

    def test_accepts_problem_without_equality(self) -> None:
        """``BarrierSolver`` solves a no-equality Problem.

        The old class rejected an equality-less Problem; the unified solver
        auto-dispatches to the plain barrier path instead.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        result = BarrierSolver().solve(problem=problem, x0=np.ones(3))
        # min 0.5||w||^2 s.t. w >= 0 has optimum at the origin. The barrier method
        # stops slightly interior at the duality-gap tolerance, so allow 1e-3.
        np.testing.assert_allclose(result.solution, np.zeros(3), atol=1e-3)

    def test_rejects_problem_without_constraints(self) -> None:
        """`solve` raises when the Problem has no inequality constraints."""
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        with self.assertRaisesRegex(ValueError, "NewtonSolver"):
            BarrierSolver().solve(
                problem=problem, x0=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            )

    @parameterized.expand(
        [
            (104, 100, 20),
            (204, 200, 30),
        ],
    )
    def test_certifies_infeasible_structured_phase1(
        self, seed: int, M: int, p: int
    ) -> None:
        r"""Certify infeasibility through the structured direct-form Phase I.

        Builds a Farkas certificate ``nu`` with ``A^T nu >= 0`` and ``b^T nu <
        0``, so the calibration set ``{A w = b, w >= 0}`` is empty. The
        direct-form Phase I (norm active) has no strictly feasible ``w > 0``
        start, so it recurses into the slack-form feasibility problem ``min s
        s.t. A w = b, w >= -s`` -- an all-linear LP whose dual certifies
        infeasibility. The barrier central-path multipliers are only
        ~1e-3-stationary, so ``evaluate_dual``'s all-linear branch must accept
        them at that tolerance (parity with ``LayeredFeasibilitySolver``); a
        1e-10 gate returns ``-inf`` at every step, and at this scale the barrier
        then marches ``t`` into the ill-conditioned active vertex and fails
        centering before any clean dual check, raising
        ``InteriorPointMethodError`` instead of certifying. The certificate must
        come from the iterate's own near-stationary multipliers.

        ``make_phase1`` drives the structured Phase I
        directly, mirroring the reroute the solver takes for a direct-form
        active layer (bypassing the legacy ``make_phase1_solver`` fallback).

        """
        rng = np.random.default_rng(seed)
        nu = rng.standard_normal(p)
        A = rng.standard_normal((p, M))
        # Flip columns so A^T nu >= 0 (each column's inner product with nu >= 0).
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]
        assert np.all(A.T @ nu >= 0)
        # Scale A so the geometry matches the production calibration problem (the
        # conditioning of the slack-form vertex is what makes centering fail).
        A /= np.dot(A @ np.ones(M), nu)
        # Choose b with b^T nu < 0 so {A w = b, w >= 0} is empty (Farkas).
        b = rng.standard_normal(p)
        b = b - (np.dot(nu, b) / np.dot(nu, nu) + 0.1) * nu
        assert np.dot(b, nu) < 0

        problem = Problem(
            obj=LinearObjective(c=np.zeros(M)),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=M),
                SquaredNormConstraintLayer(phi=float(M) * 1000.0, dimension=M),
            ],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        phase1_problem, x0_phase1 = problem.make_phase1(np.zeros(M))
        with self.assertRaises(ProblemCertifiablyInfeasibleError) as exc_info:
            BarrierSolver().solve(problem=phase1_problem, x0=x0_phase1)
        # The infeasibility certificate must carry a finite dual value (the
        # tolerance fix), not inf/nan -- that finite value is the certificate.
        assert np.isfinite(exc_info.exception.result.dual_value)

    def test_on_infeasible_continue_minimizes_violation(self) -> None:
        r"""``on_infeasible="continue"`` drives Phase I to the slack optimum.

        ``x1 + x2 = 10`` with each entry in ``[0, 1]`` is infeasible: the smallest
        achievable bound violation (the Phase I slack) is ``s* = 4`` (both entries
        pinned at the mean ``5``, exceeding the upper bound ``1`` by ``4``). The
        default ``"raise"`` aborts at the first dual certificate -- an intermediate,
        not-yet-minimized slack -- while ``"continue"`` runs the slack down to
        ``s*`` and attaches that best-effort result to the raised error. Starting
        from ``[10, 0]`` (on the manifold) makes the initial slack ``~9``, so the
        two policies reach visibly different slacks.
        """
        problem = Problem(
            ineq=[BoundsConstraintLayer(lb=0.0, ub=1.0, dimension=2)],
            eq=EqualityConstraintLayer(A=np.array([[1.0, 1.0]]), b=np.array([10.0])),
        )
        x0 = np.array([10.0, 0.0])  # on the equality manifold A x = b

        with self.assertRaises(ProblemCertifiablyInfeasibleError) as early:
            problem.solve(BarrierSolver(), x0=x0, on_infeasible="raise")
        with self.assertRaises(ProblemCertifiablyInfeasibleError) as full:
            problem.solve(BarrierSolver(), x0=x0, on_infeasible="continue")

        # The certificate carries a finite dual value (not inf/nan).
        assert np.isfinite(full.exception.result.dual_value)
        # Continuing reaches the true minimum violation s* = 4 ...
        self.assertAlmostEqual(full.exception.result.objective_value, 4.0, delta=0.5)
        # ... and never reports a larger slack than the early-abort iterate.
        assert (
            full.exception.result.objective_value
            <= early.exception.result.objective_value + 1e-3
        )

    def test_direct_form_solves_over_rank_update_background(self) -> None:
        """Direct-form Phase I handles a rank-update background with no legacy fallback.

        A ``SquaredNormConstraintLayer`` active (hoisted to the Phase-I objective)
        over an ``ImbalanceConstraintLayer`` background. The direct form hands the
        background to the general ``BarrierSolver`` unchanged, so it solves
        structured without any legacy fallback -- and reaches a point feasible for
        the original problem.
        """
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        rng = np.random.default_rng(0)
        imbalance = ImbalanceConstraintLayer(
            B=rng.standard_normal((2, 3)),
            c=rng.standard_normal(2),
            psi=100.0,
        )
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[imbalance, SquaredNormConstraintLayer(phi=10.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        phase1_problem, x0_phase1 = problem.make_phase1(np.zeros(3))
        result = BarrierSolver().solve(problem=phase1_problem, x0=x0_phase1)
        x_recovered = np.asarray(result.solution)[:3]
        assert problem.is_feasible(x_recovered)

    def test_slack_form_solves_over_rank_update_background(self) -> None:
        """Slack-form Phase I handles a rank-update background (the D1 reroute).

        A ``BoundsConstraintLayer`` (slack-form) active over an
        ``ImbalanceConstraintLayer`` (rank-update) background. The lifted slack
        assembly composes the rank-update background via ``BlockDiagonalSystem``;
        with the background strictly feasible at the seed, Phase I makes the
        Bounds active feasible and reaches a point feasible for the original
        problem -- no legacy ``make_phase1_solver`` fallback. (Before the gate
        relaxation this case fell back to the legacy path.)
        """
        rng = np.random.default_rng(7)
        M = 6
        # Loose Imbalance so the seed is comfortably background-feasible.
        imbalance = ImbalanceConstraintLayer(
            B=rng.standard_normal((2, M)),
            c=rng.standard_normal(2),
            psi=100.0,
        )
        A = np.ones((1, M))
        b = np.array([float(M)])  # mean weight 1
        problem = Problem(
            obj=LinearObjective(c=np.zeros(M)),
            ineq=[imbalance, BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        # Equality-feasible (sum == M) but Bounds-infeasible (a negative entry),
        # so Phase I must feasibilize the Bounds active; the loose Imbalance
        # background is satisfied at the seed.
        x0 = np.full(M, 1.3)
        x0[-1] = b[0] - float(x0[:-1].sum())  # on-manifold; last entry < 0
        result = BarrierSolver().solve(problem=problem, x0=x0)
        w = np.asarray(result.solution)
        assert problem.is_feasible(w)

    def test_slack_form_stages_background_infeasible_seed(self) -> None:
        """Slack-form Phase I stages background layers infeasible at the seed.

        The slack-form Phase I makes only the active (last) layer feasible, so it
        recursively feasibilizes the background layers first. Here the active is a
        loose Imbalance over a Bounds background and the seed violates Bounds, so
        the cascade feasibilizes Bounds, then the Imbalance, reaching a point
        feasible for the whole problem -- rather than rejecting the
        background-infeasible seed.
        """
        rng = np.random.default_rng(13)
        M = 6
        imbalance = ImbalanceConstraintLayer(
            B=rng.standard_normal((2, M)),
            c=rng.standard_normal(2),
            psi=100.0,  # loose -> feasible
        )
        A = np.ones((1, M))
        b = np.array([float(M)])
        problem = Problem(
            obj=LinearObjective(c=np.zeros(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M), imbalance],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        # Equality-feasible but Bounds-infeasible (a negative entry): the Bounds
        # background is infeasible at the seed, so the cascade must stage it.
        x0 = np.full(M, 1.3)
        x0[-1] = b[0] - float(x0[:-1].sum())  # on-manifold; last entry < 0
        res = BarrierSolver().solve(problem=problem, x0=x0)
        assert problem.is_feasible(np.asarray(res.solution))

    @parameterized.expand([(11, 8, 2), (12, 20, 4)])
    def test_direct_form_feasible_via_structured_phase1(
        self, seed: int, M: int, p: int
    ) -> None:
        """A feasible SquaredNorm-active problem solves through the structured Phase I.

        With ``x0=None`` the solver runs Phase I; the direct-form (SquaredNorm)
        active now routes through ``make_phase1`` + ``BarrierSolver`` (the
        Option-A reroute) rather than the legacy ``MinimizeConstraintValueSolver``.

        """
        rng = np.random.default_rng(seed)
        w_target = rng.uniform(0.5, 2.0, size=M)  # strictly positive
        A = rng.standard_normal((p, M))
        b = A @ w_target  # feasible: A w_target = b with w_target > 0
        phi_bound = 100.0 * M  # loose: ||w_target||^2 ~ 1.5 M << phi_bound
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(M), hessian_contribution=DiagonalSystem(np.ones(M))
            ),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=M),
                SquaredNormConstraintLayer(phi=phi_bound, dimension=M),
            ],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = BarrierSolver().solve(problem=problem)
        w = np.asarray(result.solution)
        np.testing.assert_array_less(-1e-8, w)  # w >= 0
        np.testing.assert_allclose(A @ w, b, rtol=1e-6, atol=1e-6)
        assert float(np.dot(w, w)) < phi_bound

    @parameterized.expand(
        [
            (1201, 5, 1),
            (1202, 25, 5),
        ],
    )
    def test_solves_bound_constrained_quadratic_with_equality(
        self, seed: int, M: int, p: int
    ) -> None:
        """Solve `min (1/2) w^T Q w + c^T w  s.t. A w = b, w >= 0`.

        Construct so the unconstrained-equality optimum is positive (so the bound
        constraint is inactive and the solution matches a closed-form KKT solve
        without the bound).

        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        # Pick w_target > 0 and a compatible equality (A w_target = b).
        w_target = rng.uniform(2.0, 5.0, size=M)
        A = rng.standard_normal((p, M))
        b = A @ w_target
        # Pick c so that the equality-constrained optimum (without bounds) is w_target.
        # Stationarity: Q w + c + A^T nu = 0; choose nu = 0, so c = -Q w_target.
        c = -Q @ w_target

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = _project_onto_equality(A, b, rng.uniform(1.5, 4.0, size=M))
        # Make sure the projected x0 is feasible for the bound constraint.
        if not problem.is_feasible(x0):
            self.skipTest(f"projected x0 not bound-feasible (seed={seed}, M={M})")
        result = BarrierSolver().solve(problem=problem, x0=x0)

        np.testing.assert_allclose(result.solution, w_target, rtol=1e-4, atol=1e-4)
        # Equality residual is tiny.
        np.testing.assert_allclose(A @ result.solution, b, rtol=1e-6, atol=1e-6)

    @parameterized.expand(
        [
            (1231, 5, 1),
            (1232, 25, 5),
        ],
    )
    def test_solve_without_x0_constructs_phase1(
        self, seed: int, M: int, p: int
    ) -> None:
        """With no ``x0``, BarrierSolver builds an equality-feasible start.

        The constructed Phase I start must satisfy ``A w = b`` and ``w > 0``;
        ``_solve_feasibility`` drives the supplied ``EqualitySolver`` through the
        last layer's ``make_phase1_solver`` to obtain it.
        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(2.0, 5.0, size=M)
        A = rng.standard_normal((p, M))
        b = A @ w_target
        c = -Q @ w_target

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = BarrierSolver().solve(problem=problem)

        np.testing.assert_allclose(result.solution, w_target, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(A @ result.solution, b, rtol=1e-6, atol=1e-6)

    def test_dual_value_matches_primal_at_convergence(self) -> None:
        r"""Zero duality gap at the BarrierSolver optimum.

        Regression test for the bug where `evaluate_dual` silently dropped the
        equality terms (`A^T \nu` in ``grad_residual`` and ``-\nu^T b`` in the
        return value), producing wrong dual values on QP+equality problems.

        Setup: ``min (1/2) w^T w  s.t.  ones^T w = 3, w >= -100``. The bound is
        essentially inactive at the optimum ``w* = (1, 1, 1)``; the equality
        multiplier converges to ``\nu* = -1`` so ``\nu^T b = -3`` is large
        relative to the IPM tolerance. With the bug, ``dual_value`` would
        differ from ``objective_value`` by ``\nu^T b ~ 3``; with the fix the
        gap is bounded by the IPM outer tolerance (~1e-3).

        """
        M = 3
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(M), hessian_contribution=DiagonalSystem(np.ones(M))
            ),
            ineq=[BoundsConstraintLayer(lb=-100.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([3.0])),
        )
        x0 = np.array([1.5, 0.5, 1.0])  # equality-feasible interior

        result = BarrierSolver().solve(problem=problem, x0=x0)

        # Primal-dual gap at the optimum is small; the bug would surface here as
        # a ~3-unit gap.
        self.assertAlmostEqual(result.dual_value, result.objective_value, delta=1e-3)


class BarrierSolverWithEqualityStructuredTest(unittest.TestCase):
    r"""The equality barrier adapter routes a structured Problem through the KKT solve.

    Proves the equality-aware ``_BarrierAdapter`` assembles via ``build_hessian_system``
    and routes the resulting ``MatrixSystem`` through the saddle-system seam
    (``EqualityConstraintLayer.saddle_system``), and that a full
    equality-constrained barrier solve reaches the analytic optimum.
    """

    @parameterized.expand(
        [
            (3211, 5, 1),
            (3212, 25, 5),
        ],
    )
    def test_equality_end_to_end_matches_optimum(
        self, seed: int, M: int, p: int
    ) -> None:
        """A full equality-constrained barrier solve through a structured Problem."""
        rng = np.random.default_rng(seed)
        q = rng.uniform(0.5, 3.0, size=M)
        w_target = rng.uniform(2.0, 5.0, size=M)  # positive => bound inactive
        A = rng.standard_normal((p, M))
        b = A @ w_target
        c = -q * w_target  # equality optimum (no bounds) = w_target

        diagonal = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DiagonalSystem(q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = _project_onto_equality(A, b, rng.uniform(1.5, 4.0, size=M))
        if not diagonal.is_feasible(x0):
            self.skipTest(f"projected x0 not bound-feasible (seed={seed}, M={M})")

        result = BarrierSolver().solve(problem=diagonal, x0=x0)

        np.testing.assert_allclose(result.solution, w_target, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(A @ result.solution, b, rtol=1e-6, atol=1e-6)


class EvaluateBarrierObjectiveInfeasibilityGuardTest(unittest.TestCase):
    """STAB-7: ``evaluate_barrier_objective`` returns ``+inf`` cleanly on infeasible x.

    Without the guard, ``-np.sum(np.log(-c))`` raises a
    ``RuntimeWarning: invalid value encountered in log`` and returns
    ``nan`` when any ``c_i >= 0``. Mirrors the existing guard on
    ``ConstraintLayer.barrier_objective`` (constraint_layers.py:205).
    """

    def test_returns_inf_on_infeasible_iterate_no_warning(self) -> None:
        """Marginally infeasible ``x`` yields ``+inf`` without RuntimeWarning."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        # x_i = 0 makes c_i = -x_i = 0 (boundary), so c >= 0 fires the guard.
        x_infeasible = np.zeros(3)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = adapter.evaluate_barrier_objective(x_infeasible, t=1.0)

        assert value == np.inf

    def test_returns_finite_on_feasible_iterate(self) -> None:
        """Strictly feasible ``x`` returns the standard barrier value."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        x_feasible = np.ones(3)
        # f0 = 0.5 * ||x||^2 = 1.5, c = -x = [-1, -1, -1], -log(-c) sums to 0.
        expected = 1.0 * 1.5 - 3 * float(np.log(1.0))

        value = adapter.evaluate_barrier_objective(x_feasible, t=1.0)

        np.testing.assert_allclose(value, expected, rtol=1e-12)


class BarrierAdapterBtlsKeepFeasibleTest(unittest.TestCase):
    """``_BarrierAdapter.btls_keep_feasible`` walks layers instead of inheriting the inner-iterate-halving fallback."""

    def test_interior_step_returns_layer_ceiling(self) -> None:
        """Override walks layers and returns the closed-form ceiling (10) instead of the inherited iterative cap (≤ 1/beta = 2).

        Pre-override behavior: the inherited ``Optimizer.btls_keep_feasible``
        fallback iterates ``btls_s = 1/beta, 1, beta, ...`` and could only
        return values ``<= 1/beta``. The override walks the layers and
        returns each layer's closed-form ``btls_max_step`` directly; for
        ``BoundsConstraintLayer`` that's exactly ``(w - lb) / -dw``, here
        ``10``. Caller's ``min(1.0, beta * 10) = 1.0``, so the quadratic-
        phase Newton takes full steps.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        x = np.full(3, 10.0)  # well inside the lb=0 box
        delta_x = -np.ones(3)  # boundary at alpha=10

        self.assertAlmostEqual(adapter.btls_keep_feasible(x, delta_x), 10.0)

    def test_unbinding_direction_returns_inf(self) -> None:
        """When ``delta_x`` moves away from every bound, return ``np.inf``."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        x = np.ones(3)
        delta_x = np.ones(3)  # all positive; lower bound never binds.

        assert adapter.btls_keep_feasible(x, delta_x) == np.inf

    def test_binding_step_returns_bounds_ceiling(self) -> None:
        """When ``delta_x`` would step past ``lb``, return the per-coordinate ceiling."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        x = np.array([1.0, 2.0, 3.0])
        delta_x = np.array([-2.0, -1.0, -1.0])  # x[0] hits 0 at alpha=0.5

        self.assertAlmostEqual(adapter.btls_keep_feasible(x, delta_x), 0.5)


class BarrierSolverEqualityInfeasibleX0Test(unittest.TestCase):
    """``BarrierSolver`` projects an off-manifold ``x0`` onto ``A x = b``.

    The feasible-start barrier maintains ``A x = b`` but never corrects an
    off-manifold iterate (its KKT step carries a zero equality RHS). Rather than
    reject a supplied equality-violating ``x0``, ``solve`` projects it onto the
    manifold (least-norm toward ``x0``, via ``EqualitySolver.solve``) before the
    barrier loop -- so an off-manifold ``x0`` solves normally and lands at the
    same optimum as a ``None`` (Phase-I) start. An on-manifold ``x0`` is a no-op
    under the projection.
    """

    def test_equality_infeasible_x0_is_projected(self) -> None:
        """An inequality-feasible but equality-violating ``x0`` is projected, not rejected."""
        M = 4
        c = -np.ones(M)
        A_eq = np.ones((1, M))
        b_eq = np.array([float(M)])  # sum(x) = M, satisfied only by x = ones
        problem = Problem(
            obj=QuadraticObjective(
                c=c, hessian_contribution=DiagonalSystem(np.ones(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        # x0 is inequality-feasible (interior of the box) but equality-
        # infeasible: sum(x0) = 20 != 4.
        x0 = np.full(M, 5.0)
        assert problem.is_feasible(x0)  # inequalities OK
        assert abs(A_eq @ x0 - b_eq) > 0.5  # equality violated by ~16

        # The solver projects x0 onto the manifold and solves. The unique
        # equality-feasible optimum here is x = ones; the off-manifold start must
        # reach the same solution as a None (Phase-I) start and satisfy A x = b.
        result = BarrierSolver().solve(problem=problem, x0=x0)
        none_start = BarrierSolver().solve(problem=problem, x0=None)
        np.testing.assert_allclose(result.solution, np.ones(M), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            result.solution, none_start.solution, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(A_eq @ result.solution, b_eq, atol=1e-6)

    def test_equality_feasible_x0_short_circuits(self) -> None:
        """When ``x0`` IS on the equality manifold AND inequality-feasible, behavior unchanged."""
        M = 4
        c = -np.ones(M)
        A_eq = np.ones((1, M))
        b_eq = np.array([float(M)])
        problem = Problem(
            obj=QuadraticObjective(
                c=c, hessian_contribution=DiagonalSystem(np.ones(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        # x0 satisfies both inequalities and equality (sum = 4).
        x0 = np.ones(M)
        result = BarrierSolver().solve(problem=problem, x0=x0)
        np.testing.assert_allclose(result.solution, np.ones(M), rtol=1e-6, atol=1e-6)

    def test_marginal_equality_residual_passes(self) -> None:
        """Equality residual at the outer-tolerance threshold short-circuits cleanly (does not raise)."""
        M = 4
        c = -np.ones(M)
        A_eq = np.ones((1, M))
        b_eq = np.array([float(M)])
        problem = Problem(
            obj=QuadraticObjective(
                c=c, hessian_contribution=DiagonalSystem(np.ones(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        # Perturb each coordinate by a tiny amount so that
        # ||A x0 - b||_2 = ||4 * 1e-9||_2 = 4e-9, well below the
        # 1e-6 * max(||b||=4, 1) = 4e-6 threshold.
        x0 = np.ones(M) + 1e-9
        result = BarrierSolver().solve(problem=problem, x0=x0)
        np.testing.assert_allclose(result.solution, np.ones(M), rtol=1e-5, atol=1e-5)


class MinimizeLagrangianTest(unittest.TestCase):
    """``_BarrierAdapter._minimize_lagrangian`` mini-Newton over `L(x, lambda)`.

    Drives `\\nabla_x L \\to 0` from a warm start. Quadratic `L` converges in one
    Newton step; singular `H_L` returns ``None`` so the caller falls back to the
    plug-in ``L(x^star, lambda)`` form.

    """

    def test_recovers_quadratic_minimizer_in_one_step(self) -> None:
        r"""For ``f0 = (1/2) x^T Q x`` with bound constraints, ``x*_L = Q^{-1} lambda``.

        With ``f_i(x) = -x_i`` (lb=0 bounds), ``L = (1/2) x^T Q x - lambda^T x`` is
        quadratic in ``x`` so a single Newton step from any feasible warm-start lands
        exactly on the analytical minimizer.
        """
        rng = np.random.default_rng(2701)
        M = 6
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        x0 = np.ones(M)  # arbitrary warm start (strictly feasible)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        expected = np.linalg.solve(Q, lmbda)
        assert x_min is not None
        np.testing.assert_allclose(x_min, expected, rtol=1e-10, atol=1e-10)
        # Quadratic L: one Newton step lands exactly, so delta^2 = 0 at x_min.
        assert delta2 is not None
        self.assertAlmostEqual(delta2, 0.0, delta=1e-12)

    def test_short_circuits_when_gradient_already_zero(self) -> None:
        r"""When ``\nabla_x L(x_0) \approx 0`` the loop returns ``x_0`` on iteration 0.

        Simulates the centering-step hot path: at ``x*`` the KKT stationarity gives
        ``\nabla_x L \approx 0``, so the mini-Newton should not take any steps.
        """
        rng = np.random.default_rng(2702)
        M = 4
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        # Pick x0 so the gradient `Q x0 - lambda` is exactly zero.
        x0 = np.asarray(np.linalg.solve(Q, lmbda), dtype=np.float64)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        assert x_min is not None
        np.testing.assert_array_equal(x_min, x0)
        # Gradient already ~0 => the pre-check returns delta^2 = 0 without stepping.
        assert delta2 == 0.0

    def test_recovers_minimizer_with_quadratic_layer_curvature(self) -> None:
        r"""``NormConstraintLayer`` contributes curvature ``2 lambda B^T B`` to ``H_L``.

        ``L = (1/2) x^T Q x + c^T x + lambda (||B x - c_n||^2 - phi^2)`` is quadratic in
        ``x``; one Newton step lands on the analytical minimizer
        ``(Q + 2 lambda B^T B)^{-1} (2 lambda B^T c_n - c)``. Exercises the
        ``solve_rank_p_update`` composition over the objective baseline solve.
        """
        rng = np.random.default_rng(2703)
        M = 5
        q = 3
        Q = _make_psd_matrix(rng, M)
        c_obj = rng.standard_normal(M)
        B = rng.standard_normal((q, M))
        c_norm = rng.standard_normal(q)
        # phi large enough that the ball is non-binding at x0 = zeros (strictly feasible).
        phi = float(np.linalg.norm(c_norm)) + 5.0
        problem = Problem(
            obj=QuadraticObjective(c=c_obj, hessian_contribution=DenseSystem(Q)),
            ineq=[NormConstraintLayer(B=B, c=c_norm, phi=phi)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.array([0.7])
        x0 = np.zeros(M)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        H = Q + 2.0 * lmbda[0] * B.T @ B
        rhs = 2.0 * lmbda[0] * B.T @ c_norm - c_obj
        expected = np.linalg.solve(H, rhs)
        assert x_min is not None
        np.testing.assert_allclose(x_min, expected, rtol=1e-10, atol=1e-10)
        assert delta2 is not None
        self.assertAlmostEqual(delta2, 0.0, delta=1e-12)

    def test_returns_none_when_hessian_singular(self) -> None:
        r"""LinearObjective + affine-only constraints has zero curvature => mini-Newton bails.

        ``H_L = 0`` (zero objective Hessian, no layer-curvature contributions); the
        baseline solve is ``solve_diagonal(eta=0)`` which produces non-finite output.
        ``_minimize_lagrangian`` detects the non-finite delta and returns ``None`` so
        the caller can fall back to the plug-in ``L(x^star, lambda)`` form.
        """
        M = 3
        problem = Problem(
            obj=LinearObjective(c=np.ones(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        # gradient(x) = c = ones; mini-Newton needs to solve H_L delta = -ones.
        lmbda = np.full(M, 0.5)
        x0 = np.ones(M)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        assert x_min is None
        assert delta2 is None

    def test_returns_none_for_linear_objective_with_quadratic_layer(self) -> None:
        r"""LP objective + ``NormConstraintLayer`` falls back to plug-in despite PD ``H_L``.

        Architecturally, ``_lagrangian_hessian_solve`` assembles ``H_L`` as a
        structured ``MatrixSystem`` -- the objective's ``hessian_contribution`` as the
        base, with the layer's ``hessian_curvature`` composed on top.
        ``LinearObjective`` contributes no curvature (a zero base), so the layer's
        low-rank (Woodbury) update rides a singular base and the solve is non-finite
        even though the assembled ``H_L = 2 lambda B^T B`` is analytically PD.
        ``_minimize_lagrangian`` detects the non-finite delta and returns ``None`` so
        the caller falls back to the plug-in ``L(x^star, lambda)`` form (which is
        correct at centering convergence).

        Pinning current behavior; a more sophisticated dispatch could recognize the
        zero-curvature base case and route the solve through the constraint layer's
        curvature directly.
        """
        rng = np.random.default_rng(2704)
        M = 4
        q = 5
        c_obj = rng.standard_normal(M)
        B = rng.standard_normal((q, M))
        c_norm = rng.standard_normal(q)
        phi = float(np.linalg.norm(c_norm)) + 5.0
        problem = Problem(
            obj=LinearObjective(c=c_obj),
            ineq=[NormConstraintLayer(B=B, c=c_norm, phi=phi)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.array([0.6])
        x0 = np.zeros(M)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        assert x_min is None
        assert delta2 is None

    def test_recovers_minimizer_with_two_quadratic_layers(self) -> None:
        r"""QP + ``NormConstraintLayer`` + ``SquaredNormConstraintLayer`` composes two curvature contributions.

        ``L = (1/2) x^T Q x + c^T x + lambda_1 (||B x - c_n||^2 - phi_1^2)
              + lambda_2 (x^T x - phi_2)``. ``H_L = Q + 2 lambda_1 B^T B + 2 lambda_2 I``
        is PD; analytical minimizer is ``H_L^{-1} (2 lambda_1 B^T c_n - c)``. Exercises
        two ``solve_rank_p_update`` compositions stacked over the QP baseline.
        """
        rng = np.random.default_rng(2705)
        M = 5
        q = 3
        Q = _make_psd_matrix(rng, M)
        c_obj = rng.standard_normal(M)
        B = rng.standard_normal((q, M))
        c_norm = rng.standard_normal(q)
        phi_1 = float(np.linalg.norm(c_norm)) + 5.0
        phi_2 = 100.0  # large enough to be non-binding
        problem = Problem(
            obj=QuadraticObjective(c=c_obj, hessian_contribution=DenseSystem(Q)),
            ineq=[
                NormConstraintLayer(B=B, c=c_norm, phi=phi_1),
                SquaredNormConstraintLayer(phi=phi_2, dimension=M),
            ],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.array([0.7, 0.4])
        x0 = np.zeros(M)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        H = Q + 2.0 * lmbda[0] * B.T @ B + 2.0 * lmbda[1] * np.eye(M)
        rhs = 2.0 * lmbda[0] * B.T @ c_norm - c_obj
        expected = np.linalg.solve(H, rhs)
        assert x_min is not None
        np.testing.assert_allclose(x_min, expected, rtol=1e-10, atol=1e-10)
        assert delta2 is not None
        self.assertAlmostEqual(delta2, 0.0, delta=1e-12)

    def test_backtracks_to_stay_in_objective_domain(self) -> None:
        r"""A dual Newton step that would overshoot off ``dom f_0`` is capped in-domain.

        ``f_0 = -sum log(w)`` (domain ``w > 0``) with an upper-bound layer gives
        ``L = -sum log(w) + lambda^T (w - ub)``, minimized at ``w = 1 / lambda``.
        From ``w0 = 2`` with ``lambda = 1`` the full Newton step is ``delta = -2``,
        which lands exactly on ``w = 0`` (the domain boundary). Without the
        feasibility cap the next iteration's gradient/solve would be non-finite,
        bailing to ``None`` (the loose plug-in dual). ``_LogObjective.btls_keep_in_domain``
        reports the closed-form distance to the boundary (``1``), so the Armijo line
        search caps the step strictly inside ``dom f_0`` (``s <= beta * 1``) and the
        mini-Newton converges to the minimizer ``w = 1`` while staying positive.
        """
        M = 3
        problem = Problem(
            obj=_LogObjective(dimension=M),
            ineq=[BoundsConstraintLayer(ub=10.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.ones(M)
        x0 = np.full(M, 2.0)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        assert x_min is not None
        assert np.all(x_min > 0.0)  # never left dom f_0
        np.testing.assert_allclose(x_min, np.ones(M), rtol=1e-9, atol=1e-9)
        assert delta2 is not None
        assert delta2 >= 0.0


class NewtonDecrementSqTest(unittest.TestCase):
    r"""``_BarrierAdapter._newton_decrement_sq``: `\delta^2(x) = \nabla L^T H_L^{-1} \nabla L`.

    Returned by ``evaluate_dual`` as the correction to ``L(x_min, lambda)`` -- a valid
    upper bound on `L(x_min) - \inf_x L` for self-concordant ``L`` in the
    quadratic-convergence region (B&V Eq 9.42). For quadratic ``L`` at the exact
    minimizer the gradient (and hence the decrement) is zero, so the correction is
    a no-op at convergence.

    """

    def test_zero_when_gradient_tiny(self) -> None:
        r"""Short-circuits to ``0`` when `\|\nabla_x L\| < 1e-10` without touching the Hessian solve."""
        rng = np.random.default_rng(2801)
        M = 4
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        # x* = Q^{-1} lambda makes `\nabla L = Q x - lambda` exactly zero.
        x_star = np.asarray(np.linalg.solve(Q, lmbda), dtype=np.float64)

        decrement_sq = adapter._newton_decrement_sq(x_star, lmbda)

        assert decrement_sq == 0.0

    def test_matches_quadratic_form_off_optimum(self) -> None:
        r"""At an off-optimum ``x``, `\delta^2 = g^T H^{-1} g` for ``g = Q x - lambda``."""
        rng = np.random.default_rng(2802)
        M = 5
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        x = np.ones(M)  # arbitrary point with non-zero gradient

        decrement_sq = adapter._newton_decrement_sq(x, lmbda)

        g = Q @ x - lmbda
        expected = float(g @ np.linalg.solve(Q, g))
        assert decrement_sq is not None
        np.testing.assert_allclose(decrement_sq, expected, rtol=1e-10, atol=1e-12)

    def test_matches_dual_gap_for_pure_quadratic(self) -> None:
        r"""For quadratic ``L``, `L(x) - \tfrac12 \delta^2(x) = \inf_x L` exactly.

        The Newton decrement correction guarantees a lower bound on `\inf_x L`; the
        gap between ``L(x) - delta^2`` (what ``evaluate_dual`` returns) and the true
        `\inf_x L` is exactly `\tfrac12 \delta^2` for pure-quadratic ``L``.
        """
        rng = np.random.default_rng(2803)
        M = 4
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        x = np.ones(M)

        decrement_sq = adapter._newton_decrement_sq(x, lmbda)

        L_x = 0.5 * x @ Q @ x - lmbda @ x
        x_min = np.linalg.solve(Q, lmbda)
        inf_L = 0.5 * x_min @ Q @ x_min - lmbda @ x_min
        assert decrement_sq is not None
        # L(x) - delta^2 underestimates inf_x L by exactly (1/2) delta^2.
        np.testing.assert_allclose(
            (L_x - decrement_sq), inf_L - 0.5 * decrement_sq, rtol=1e-10, atol=1e-12
        )

    def test_returns_none_on_singular_hessian(self) -> None:
        r"""LP at non-stationary ``x``: Hessian solve fails -> returns ``None``.

        Caller (``evaluate_dual``) then skips the correction rather than collapsing
        the dual to `-\infty`.
        """
        M = 3
        problem = Problem(
            obj=LinearObjective(c=np.ones(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.full(M, 0.5)
        # gradient = c + (-1)*lambda = ones - 0.5*ones = 0.5*ones, large enough to
        # avoid the short-circuit and force the Hessian solve.
        x = np.ones(M)

        decrement_sq = adapter._newton_decrement_sq(x, lmbda)

        assert decrement_sq is None


class AllLinearDispatchTest(unittest.TestCase):
    r"""``_BarrierAdapter.evaluate_dual`` analytical LP dual when ``_all_linear()``.

    For LP/equality-LP problems, ``L`` is affine in ``x``: `\inf_x L = -\infty`
    unless the gradient residual ``r = c + A_ineq^T lambda + A_eq^T nu`` is zero,
    in which case ``L(x)`` is the exact dual at every ``x``. This sidesteps the
    mini-Newton path (and its Woodbury-anchor degeneracy on `H_L = 0` LPs).

    """

    def test_is_linear_class_attributes(self) -> None:
        """``is_linear`` is set as expected on the objective and layer types."""
        assert LinearObjective.is_linear is True
        assert QuadraticObjective.is_linear is False
        assert BoundsConstraintLayer.is_linear is True
        assert NormConstraintLayer.is_linear is False
        assert SquaredNormConstraintLayer.is_linear is False

    def test_all_linear_true_for_lp_problem(self) -> None:
        """LP + Bounds layer => ``_all_linear()`` is ``True``."""
        problem = Problem(
            obj=LinearObjective(c=np.ones(3)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())

        assert adapter._all_linear() is True

    def test_all_linear_false_for_qp_problem(self) -> None:
        """QP objective taints ``_all_linear()`` even with affine constraints."""
        rng = np.random.default_rng(2901)
        Q = _make_psd_matrix(rng, 3)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(3), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())

        assert adapter._all_linear() is False

    def test_all_linear_false_for_lp_with_quadratic_layer(self) -> None:
        """Norm-constraint layer taints ``_all_linear()`` even with LP objective."""
        rng = np.random.default_rng(2902)
        M = 3
        B = rng.standard_normal((2, M))
        c_norm = rng.standard_normal(2)
        problem = Problem(
            obj=LinearObjective(c=np.ones(M)),
            ineq=[NormConstraintLayer(B=B, c=c_norm, phi=10.0)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())

        assert adapter._all_linear() is False

    def test_evaluate_dual_returns_minus_inf_when_residual_nonzero(self) -> None:
        r"""LP with `r = c + (-I)^T lambda \ne 0` => dual is `-\infty`."""
        M = 3
        problem = Problem(
            obj=LinearObjective(c=np.ones(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        # gradient residual = ones - lambda; lambda != ones => residual != 0.
        lmbda = np.array([0.3, 0.5, 0.8])
        x_star = np.ones(M)

        dual = adapter.evaluate_dual(lmbda=lmbda, nu=np.zeros(0), x_star=x_star)

        assert dual == float("-inf")

    def test_evaluate_dual_returns_lagrangian_when_residual_zero(self) -> None:
        r"""LP with stationary ``lambda`` (`r = 0`) => dual equals `-b^T lambda` at any x."""
        M = 3
        c = np.array([1.0, 2.0, 3.0])
        problem = Problem(
            obj=LinearObjective(c=c),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        # f_i(x) = -x_i so grad_f_i = -e_i; r = c - lambda = 0 when lambda = c.
        lmbda = c.copy()
        # Any x works; pick a non-trivial one to confirm L(x) is constant in x.
        x_star = np.array([2.5, 4.0, 1.2])

        dual = adapter.evaluate_dual(lmbda=lmbda, nu=np.zeros(0), x_star=x_star)

        # L(x) = c^T x + lambda^T (-x) = (c - lambda)^T x = 0 (since lambda = c).
        self.assertAlmostEqual(dual, 0.0, delta=1e-12)

    def test_evaluate_dual_for_equality_lp_returns_minus_inf_when_residual_nonzero(
        self,
    ) -> None:
        r"""Equality LP with `r = c + A_eq^T nu - lambda \ne 0` => dual is `-\infty`."""
        M = 3
        A_eq = np.array([[1.0, 1.0, 1.0]])
        b_eq = np.array([1.0])
        problem = Problem(
            obj=LinearObjective(c=np.ones(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.array([0.1, 0.1, 0.1])
        nu = np.array([0.0])
        # r = ones + A^T*0 - 0.1*ones = 0.9*ones != 0.
        x_star = np.full(M, 1.0 / 3.0)  # feasible: sum = 1.

        dual = adapter.evaluate_dual(lmbda=lmbda, nu=nu, x_star=x_star)

        assert dual == float("-inf")

    def test_evaluate_dual_for_equality_lp_returns_finite_when_residual_zero(
        self,
    ) -> None:
        r"""Equality LP at stationary `(lambda, nu)` => dual is `-b_eq^T nu` (plus the
        ``lambda^T f(x) - b_eq^T nu`` plug-in, which at primal-feasible ``x`` reduces
        to `-b_eq^T nu` when ``lambda = 0``)."""
        M = 3
        A_eq = np.array([[1.0, 1.0, 1.0]])
        b_eq = np.array([1.0])
        c = np.array([1.0, 1.0, 1.0])  # parallel to A_eq row
        problem = Problem(
            obj=LinearObjective(c=c),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A_eq, b=b_eq),
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        # r = c + A_eq^T*nu - lambda; choose lambda = 0, nu = -1 => r = ones - ones = 0.
        lmbda = np.zeros(M)
        nu = np.array([-1.0])
        x_star = np.full(M, 1.0 / 3.0)  # primal-feasible

        dual = adapter.evaluate_dual(lmbda=lmbda, nu=nu, x_star=x_star)

        # L(x) = c^T x + 0 + nu*(A_eq x - b_eq) = 1 + (-1)*(1 - 1) = 1.
        # Equivalent to -b_eq^T nu = -1 * (-1) = 1 at convergence.
        self.assertAlmostEqual(dual, 1.0, delta=1e-12)


class CenteringStepQuadraticPhaseUnitStepTest(unittest.TestCase):
    """Pin the BTLS skip in ``BaseInteriorPointMethodSolver.centering_step``.

    Once the inner Newton loop enters the quadratic-convergence phase
    (``lambda < eta``), Boyd and Vandenberghe 9.5.3 guarantees the unit
    step satisfies the Armijo sufficient-decrease test. At high barrier
    parameter ``t``, ``ft = t * f0 + sum log(-fi)`` grows large enough
    that ``ft(x) - ft(x + delta_x)`` rounds to noise in double precision,
    and the BTLS Armijo check rejects ``s=1`` -- halving down to
    ``min_step`` and either raising or returning a step that barely moves
    ``x``. The fix short-circuits BTLS in the quadratic phase: take the
    unit step (capped at the feasibility limit) directly.
    """

    @staticmethod
    def _qp_problem() -> Problem:
        """Tiny PSD QP with the unconstrained optimum interior to ``x >= 0``."""
        M = 3
        Q = np.eye(M)
        w_target = np.ones(M)
        c = -Q @ w_target  # so x* (unconstrained) = w_target, interior to lb=0
        return Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )

    def test_quadratic_phase_skips_btls_at_high_t(self) -> None:
        """At ``t = 1e10`` with a tolerance below the FP floor of
        ``ft(x) - ft(x + delta_x)``, the centering loop must converge
        without ever calling ``backtracking_line_search``. The first
        Newton iterate already satisfies ``lambda < eta`` so the
        quadratic-phase branch fires immediately.
        """
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        # Tighten ``tolerance`` past the FP floor so the iteration cannot
        # short-circuit on ``suboptimality < tolerance`` and must actually
        # exercise the post-tolerance code path.
        settings = OptimizationSettings(
            inner=InnerSettings(
                tolerance=1e-20,
                tolerance_soft=1e-20,
                max_iterations=50,
            )
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        btls_calls = 0
        original_btls = adapter.backtracking_line_search

        def spy_btls(
            x: npt.NDArray[np.float64],
            delta_x: npt.NDArray[np.float64],
            t: float,
            lambda_squared: float,
        ) -> float:
            nonlocal btls_calls
            btls_calls += 1
            return original_btls(x, delta_x, t, lambda_squared)

        t = 1e10
        x0 = np.ones(3)

        with patch.object(adapter, "backtracking_line_search", new=spy_btls):
            result = adapter.centering_step(x0=x0, t=t, last_step=False)

        assert btls_calls == 0
        # Pre-fix this scenario raised ``CenteringStepError`` (BTLS hit
        # ``min_step``); post-fix it returns cleanly. ``status`` may be 0
        # (suboptimality fell under ``tolerance``) or the loop runs out of
        # iterations -- either way the BTLS-skip path is what matters,
        # and the spy proves it.
        assert result.nits <= settings.inner.max_iterations

    def test_converging_on_the_crossing_iteration_leaves_no_phantom_index(
        self,
    ) -> None:
        """Converging in the same iteration that crosses the quadratic
        threshold must not record a ``quadratic_from`` with nothing behind
        it.

        ``quadratic_from`` used to be set to ``len(btls_steps)`` at the
        moment the threshold was detected -- before that iteration's step
        is appended. An iterate that also satisfies ``tolerance`` in that
        same iteration returns early with no step ever appended, leaving
        ``quadratic_from`` pointing one past the end of ``btls_steps`` and
        silently dropping the transition instead of reporting it via
        ``None``, "the threshold was never crossed".
        """
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        settings = OptimizationSettings(
            inner=InnerSettings(
                tolerance=1e-7,
                tolerance_soft=1e-7,
                max_iterations=50,
            )
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        def fake_newton_step(
            x: npt.NDArray[np.float64], t: float
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            return np.zeros_like(x), np.zeros(0)

        def fake_decrement_squared(
            x: npt.NDArray[np.float64],
            t: float,
            delta_x: npt.NDArray[np.float64],
        ) -> float:
            # Below eta^2 (crosses into the quadratic phase) and below
            # 2 * tolerance (already converged), on the very first iterate.
            return 1e-8

        with (
            patch.object(adapter, "calculate_newton_step", new=fake_newton_step),
            patch.object(
                adapter, "newton_decrement_squared", new=fake_decrement_squared
            ),
        ):
            result = adapter.centering_step(x0=np.ones(3), t=1e10, last_step=False)

        assert result.status == 0
        assert result.nits == 1
        assert result.btls_steps == []
        assert result.quadratic_from is None

    def test_quadratic_phase_stall_detector_bails_to_best_iterate(self) -> None:
        """At high ``t`` the centering loop can drop into the quadratic
        phase and then bounce around the floating-point floor of
        ``ft(x) - ft(x + delta_x)`` without ever satisfying
        ``suboptimality < tolerance``. With ``tolerance`` below the floor
        but ``tolerance_soft`` above it, the stall detector must bail to
        the best iterate seen after ``stall_patience`` consecutive
        non-improving iterations -- returning ``status=1`` rather than
        burning the full ``max_iterations`` budget.

        Drives the FSM by monkeypatching ``newton_decrement_squared`` to a
        scripted sequence: the quadratic threshold is crossed on iter 1,
        iters 2-3 genuinely improve (iter 3 is the global best), then
        five flat iterates fail to beat the best. With
        ``stall_patience=5``, the detector fires on iter 8.
        """
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        stall_patience = 5
        settings = OptimizationSettings(
            inner=InnerSettings(
                tolerance=1e-20,
                tolerance_soft=1e-3,
                max_iterations=200,
                stall_patience=stall_patience,
            )
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        # 2 * sub-opt sequence: quad threshold cross on iter 1, improve,
        # global best, then five flat iterates (no strict improvement).
        lambda_squared_sequence = [
            2e-3,  # iter 1: quad threshold crossed, sub-opt = 1e-3
            2e-10,  # iter 2: improved, sub-opt = 1e-10
            2e-12,  # iter 3: improved, sub-opt = 1e-12 -- BEST
            2e-10,  # iter 4: no improvement, counter=1
            2e-10,  # iter 5: counter=2
            2e-10,  # iter 6: counter=3
            2e-10,  # iter 7: counter=4
            2e-10,  # iter 8: counter=5 -> stall detector fires
        ]
        call_idx = {"i": 0}

        def fake_newton_step(
            x: npt.NDArray[np.float64], t: float
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            return np.zeros_like(x), np.zeros(0)

        def fake_decrement_squared(
            x: npt.NDArray[np.float64],
            t: float,
            delta_x: npt.NDArray[np.float64],
        ) -> float:
            val = lambda_squared_sequence[call_idx["i"]]
            call_idx["i"] += 1
            return val

        btls_calls = 0
        original_btls = adapter.backtracking_line_search

        def spy_btls(
            x: npt.NDArray[np.float64],
            delta_x: npt.NDArray[np.float64],
            t: float,
            lambda_squared: float,
        ) -> float:
            nonlocal btls_calls
            btls_calls += 1
            return original_btls(x, delta_x, t, lambda_squared)

        with (
            patch.object(adapter, "calculate_newton_step", new=fake_newton_step),
            patch.object(
                adapter, "newton_decrement_squared", new=fake_decrement_squared
            ),
            patch.object(adapter, "backtracking_line_search", new=spy_btls),
        ):
            result = adapter.centering_step(x0=np.ones(3), t=1e10, last_step=False)

        assert result.status == 1
        assert "stalled" in result.message
        assert result.nits == 8
        # Every iter is in the quadratic phase; BTLS must never be called.
        assert btls_calls == 0
        # Bail returns the best (iter-3) iterate's suboptimality.
        np.testing.assert_allclose(
            min(result.suboptimalities), 1e-12, rtol=1e-6, atol=1e-12
        )

    def test_quadratic_phase_stall_detector_fires_on_last_step(self) -> None:
        """The stall detector must fire on the *last* centering step too.

        The highest-``t`` step is the one most prone to the FP-floor bounce,
        and there is no further centering step to make up the gap -- so
        bailing to the best iterate (already below ``tolerance_soft``) with
        ``status=1`` is preferable to burning the full ``max_iterations``
        budget. Drives the identical scripted FSM as
        ``test_quadratic_phase_stall_detector_bails_to_best_iterate`` but
        with ``last_step=True``, and pins the same bail outcome.
        """
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        stall_patience = 5
        settings = OptimizationSettings(
            inner=InnerSettings(
                tolerance=1e-20,
                tolerance_soft=1e-3,
                max_iterations=200,
                stall_patience=stall_patience,
            )
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        # quad threshold cross on iter 1, improve, global best (iter 3),
        # then five flat iterates -> stall detector fires on iter 8.
        lambda_squared_sequence = [
            2e-3,
            2e-10,
            2e-12,
            2e-10,
            2e-10,
            2e-10,
            2e-10,
            2e-10,
        ]
        call_idx = {"i": 0}

        def fake_newton_step(
            x: npt.NDArray[np.float64], t: float
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            return np.zeros_like(x), np.zeros(0)

        def fake_decrement_squared(
            x: npt.NDArray[np.float64],
            t: float,
            delta_x: npt.NDArray[np.float64],
        ) -> float:
            val = lambda_squared_sequence[call_idx["i"]]
            call_idx["i"] += 1
            return val

        btls_calls = 0
        original_btls = adapter.backtracking_line_search

        def spy_btls(
            x: npt.NDArray[np.float64],
            delta_x: npt.NDArray[np.float64],
            t: float,
            lambda_squared: float,
        ) -> float:
            nonlocal btls_calls
            btls_calls += 1
            return original_btls(x, delta_x, t, lambda_squared)

        with (
            patch.object(adapter, "calculate_newton_step", new=fake_newton_step),
            patch.object(
                adapter, "newton_decrement_squared", new=fake_decrement_squared
            ),
            patch.object(adapter, "backtracking_line_search", new=spy_btls),
        ):
            result = adapter.centering_step(x0=np.ones(3), t=1e10, last_step=True)

        assert result.status == 1
        assert "stalled" in result.message
        assert result.nits == 8
        assert btls_calls == 0
        np.testing.assert_allclose(
            min(result.suboptimalities), 1e-12, rtol=1e-6, atol=1e-12
        )

    def test_quadratic_phase_stall_detector_fires_on_microscopic_creep(self) -> None:
        """The stall detector must fire on a microscopic asymptotic creep.

        A slow crawl toward a floor above ``tolerance`` -- each iterate a hair
        below the last, so it always sets a new all-time best -- would reset a
        naive ``iters_since_improvement`` counter every iteration and never
        trip the detector, burning the full ``max_iterations`` budget. With
        ``stall_rel_improvement`` the counter resets only on a >=1% relative
        drop, so sub-1% creep counts as a stall.

        Sequence: quad threshold crossed on iter 1, one genuine drop on iter
        2, then five sub-1% creep iterates -> the detector fires on iter 7.
        """
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        settings = OptimizationSettings(
            inner=InnerSettings(
                tolerance=1e-20,
                tolerance_soft=1e-3,
                max_iterations=200,
                stall_patience=5,
                stall_rel_improvement=0.01,
            )
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        # lambda_squared = 2 * sub-opt. After iter 2's genuine drop, each
        # iterate improves by ~0.2% -- below the 1% floor, so it counts as a
        # stall.
        lambda_squared_sequence = [
            2e-3,  # iter 1: quad cross, sub-opt 1e-3
            1e-5,  # iter 2: sub-opt 5e-6 -- genuine drop, counter reset
            2 * 4.99e-6,  # iter 3: creep, counter=1
            2 * 4.98e-6,  # iter 4: counter=2
            2 * 4.97e-6,  # iter 5: counter=3
            2 * 4.96e-6,  # iter 6: counter=4
            2 * 4.95e-6,  # iter 7: counter=5 -> stall fires
        ]
        call_idx = {"i": 0}

        def fake_newton_step(
            x: npt.NDArray[np.float64], t: float
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            return np.zeros_like(x), np.zeros(0)

        def fake_decrement_squared(
            x: npt.NDArray[np.float64],
            t: float,
            delta_x: npt.NDArray[np.float64],
        ) -> float:
            val = lambda_squared_sequence[call_idx["i"]]
            call_idx["i"] += 1
            return val

        with (
            patch.object(adapter, "calculate_newton_step", new=fake_newton_step),
            patch.object(
                adapter, "newton_decrement_squared", new=fake_decrement_squared
            ),
        ):
            result = adapter.centering_step(x0=np.ones(3), t=1e10, last_step=False)

        assert result.status == 1
        assert "stalled" in result.message
        assert result.nits == 7
        np.testing.assert_allclose(
            min(result.suboptimalities), 4.95e-6, rtol=1e-6, atol=1e-12
        )

    def test_max_iterations_returns_best_within_soft_tolerance(self) -> None:
        """Hitting ``max_iterations`` within soft tolerance returns, not raises.

        When the loop runs out of iterations but the best iterate is below
        ``tolerance_soft``, the centering step must return it with
        ``status=1`` rather than raising ``CenteringStepError`` -- the same
        soft-tolerance contract the stall detector and BTLS-error fallback
        honor.

        Drives a meaningful (>1%) improvement every iteration so the stall
        detector never fires; with ``max_iterations=4`` the loop exits having
        reached 1e-6 (< ``tolerance_soft`` 1e-3) but not ``tolerance``
        (1e-20).
        """
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        settings = OptimizationSettings(
            inner=InnerSettings(
                tolerance=1e-20,
                tolerance_soft=1e-3,
                max_iterations=4,
                stall_patience=5,
                stall_rel_improvement=0.01,
            )
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        # Quad cross on iter 1, then a 10x drop each iter (all meaningful, so
        # the stall counter never accumulates), ending at 1e-6 after 4 iters.
        lambda_squared_sequence = [2e-3, 2e-4, 2e-5, 2e-6]
        call_idx = {"i": 0}

        def fake_newton_step(
            x: npt.NDArray[np.float64], t: float
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            return np.zeros_like(x), np.zeros(0)

        def fake_decrement_squared(
            x: npt.NDArray[np.float64],
            t: float,
            delta_x: npt.NDArray[np.float64],
        ) -> float:
            val = lambda_squared_sequence[call_idx["i"]]
            call_idx["i"] += 1
            return val

        with (
            patch.object(adapter, "calculate_newton_step", new=fake_newton_step),
            patch.object(
                adapter, "newton_decrement_squared", new=fake_decrement_squared
            ),
        ):
            result = adapter.centering_step(x0=np.ones(3), t=1e10, last_step=False)

        assert result.status == 1
        assert "max_iterations" in result.message
        assert result.nits == 4
        np.testing.assert_allclose(
            min(result.suboptimalities), 1e-6, rtol=1e-6, atol=1e-12
        )


class NewtonDecrementSquaredTest(unittest.TestCase):
    """The gradient-form Newton decrement equals the Hessian form.

    With ``inner.newton_decrement_via_gradient=True``,
    ``newton_decrement_squared`` computes ``-grad_ft . delta_x`` instead of a
    second ``hessian_multiply``; for the Newton step the two are mathematically
    identical (``H delta_x = -grad_ft``). These tests pin that equivalence so
    the cheaper form can't silently drift from the Hessian-quadratic form.
    """

    _GRADIENT = OptimizationSettings(
        inner=InnerSettings(newton_decrement_via_gradient=True)
    )

    def test_matches_hessian_form_no_equality(self) -> None:
        rng = np.random.default_rng(0)
        M = 6
        problem = Problem(
            obj=QuadraticObjective(
                c=rng.standard_normal(M),
                hessian_contribution=DenseSystem(_make_psd_matrix(rng, M)),
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(
            problem=problem, settings=NewtonDecrementSquaredTest._GRADIENT
        )
        x = rng.uniform(1.0, 2.0, size=M)  # strictly feasible for lb=0
        t = 5.0
        delta_x, _ = adapter.calculate_newton_step(x, t)
        hessian_form = max(
            0.0, float(np.dot(delta_x, adapter.hessian_multiply(x, t, delta_x)))
        )
        np.testing.assert_allclose(
            adapter.newton_decrement_squared(x, t, delta_x),
            hessian_form,
            rtol=1e-7,
            atol=1e-12,
        )

    def test_matches_hessian_form_with_equality(self) -> None:
        rng = np.random.default_rng(1)
        M = 5
        A = np.ones((1, M))
        x = rng.uniform(1.0, 2.0, size=M)  # strictly feasible for lb=0
        problem = Problem(
            obj=QuadraticObjective(
                c=rng.standard_normal(M),
                hessian_contribution=DenseSystem(_make_psd_matrix(rng, M)),
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=A @ x),  # x lies on the manifold
        )
        adapter = _BarrierAdapter(
            problem=problem, settings=NewtonDecrementSquaredTest._GRADIENT
        )
        t = 5.0
        delta_x, _ = adapter.calculate_newton_step(x, t)
        hess_vec = adapter.hessian_multiply(x, t, delta_x)
        hessian_form = max(0.0, float(np.dot(delta_x, hess_vec)))
        np.testing.assert_allclose(
            adapter.newton_decrement_squared(x, t, delta_x),
            hessian_form,
            rtol=1e-7,
            atol=1e-12,
        )


class SolveBestBoundsFallbackTest(unittest.TestCase):
    """``solve`` reports the best primal/dual bounds found on a centering failure.

    When a later centering step raises ``CenteringStepError``, the outer loop
    returns the lowest-objective (best primal) iterate certified by the
    highest dual (best lower bound) -- which may come from *different* steps --
    reporting ``best_primal.objective - best_dual.dual`` as the suboptimality,
    when that gap is within ``outer.tolerance_soft``; otherwise it raises.
    """

    @staticmethod
    def _result(
        obj: float,
        dual: float,
        sol_marker: float,
        mult_marker: float,
        status: Literal[0, 1, 2] = 0,
    ) -> _CenteringOutcome:
        """A centering result with markers so tests can tell which step came back."""
        return _CenteringOutcome(
            solution=np.full(3, sol_marker),
            objective_value=obj,
            barrier_objective_value=obj,
            dual_value=dual,
            equality_multipliers=np.zeros(0),
            inequality_multipliers=np.full(3, mult_marker),
            suboptimalities=[obj - dual],
            nits=2,
            status=status,
            message="completed",
        )

    @staticmethod
    def _pinned_t0(x0: npt.NDArray[np.float64]) -> float:
        """Pin the barrier init so num_steps is comfortably > 1."""
        return 1.0

    @staticmethod
    def _adapter() -> _BarrierAdapter:
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        return _BarrierAdapter(problem=problem)

    def test_pairs_best_primal_with_best_dual_from_different_steps(self) -> None:
        """The lowest-obj step and highest-dual step can differ; both are used."""
        adapter = SolveBestBoundsFallbackTest._adapter()
        # step 2 has the best (lowest) objective; step 3 (a soft status==1 step)
        # has the best (highest) dual. The returned solution must come from #2 and
        # the dual/multipliers from #3, with gap = 5.0 - 4.9999 = 1e-4 < 1e-3.
        # All duals are valid lower bounds (<= min objective 5.0 ~ p*).
        results: dict[int, tuple[float, float, float, float, Literal[0, 1, 2]]] = {
            1: (10.0, 2.0, 1.0, 1.0, 0),
            2: (5.0, 4.0, 2.0, 2.0, 0),  # best primal (obj 5.0)
            3: (5.5, 4.9999, 3.0, 3.0, 1),  # best dual (4.9999), soft step
        }

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if last_step:
                raise CenteringStepError(
                    "forced last-step failure", suboptimality=0.1, last_iterate=x
                )
            obj, dual, sol, mult, status = results.get(
                outer_nit, (6.0, 3.0, float(outer_nit), float(outer_nit), 0)
            )
            return SolveBestBoundsFallbackTest._result(obj, dual, sol, mult, status)

        with (
            patch.object(
                adapter,
                "initialize_barrier_parameter",
                new=SolveBestBoundsFallbackTest._pinned_t0,
            ),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        assert result.status is SolveStatus.ACCEPTABLE
        np.testing.assert_allclose(result.suboptimality, 1e-4, rtol=1e-6, atol=1e-12)
        # Primal point from step 2, dual + multipliers from step 3.
        np.testing.assert_array_equal(result.solution, np.full(3, 2.0))
        self.assertAlmostEqual(result.objective_value, 5.0)
        self.assertAlmostEqual(result.dual_value, 4.9999)
        np.testing.assert_array_equal(result.inequality_multipliers, np.full(3, 3.0))
        assert "acceptable tolerance" in result.message

    def test_raises_when_best_bounds_gap_too_loose(self) -> None:
        adapter = SolveBestBoundsFallbackTest._adapter()
        # Every step brackets only to gap 1.0 >> outer.tolerance_soft (1e-3).

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if last_step:
                raise CenteringStepError(
                    "forced last-step failure", suboptimality=0.1, last_iterate=x
                )
            return SolveBestBoundsFallbackTest._result(
                5.0, 4.0, float(outer_nit), float(outer_nit)
            )

        with (
            patch.object(
                adapter,
                "initialize_barrier_parameter",
                new=SolveBestBoundsFallbackTest._pinned_t0,
            ),
            patch.object(adapter, "centering_step", new=scripted),
            self.assertRaises(InteriorPointMethodError) as exc,
        ):
            adapter.solve(x0=np.full(3, 0.5))
        # The best result found is attached so the caller can recover it.
        attached = exc.exception.result
        assert attached is not None
        self.assertAlmostEqual(attached.suboptimality, 1.0)
        np.testing.assert_array_equal(attached.solution, np.full(3, 1.0))
        assert "tolerance_soft" in str(exc.exception)
        # The attached result must report the failure honestly: status 2 (not the
        # "acceptable tolerance" status 1) and a message that does not claim the
        # soft tolerance was met.
        assert attached.status is SolveStatus.MAX_ITERATIONS
        assert "acceptable tolerance" not in attached.message
        assert "tolerance_soft" in attached.message

    def test_relative_gap_hint_in_failure_message(self) -> None:
        """With ``tolerance_rel`` set, a tiny *relative* gap is called out."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        settings = OptimizationSettings(outer=OuterSettings(tolerance_rel=1e-3))
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        # Absolute gap 10 exceeds tolerance_soft (1e-3), but relative to the
        # objective (1e6) it is 1e-5 <= tolerance_rel (1e-3).
        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if last_step:
                raise CenteringStepError(
                    "forced last-step failure", suboptimality=0.1, last_iterate=x
                )
            return SolveBestBoundsFallbackTest._result(
                1e6, 1e6 - 10.0, float(outer_nit), float(outer_nit)
            )

        with (
            patch.object(
                adapter,
                "initialize_barrier_parameter",
                new=SolveBestBoundsFallbackTest._pinned_t0,
            ),
            patch.object(adapter, "centering_step", new=scripted),
            self.assertRaises(InteriorPointMethodError) as exc,
        ):
            adapter.solve(x0=np.full(3, 0.5))
        assert "relative gap" in str(exc.exception)
        assert exc.exception.result is not None

    def test_raises_when_no_step_completed(self) -> None:
        adapter = SolveBestBoundsFallbackTest._adapter()

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            raise CenteringStepError(
                "forced failure on every step", suboptimality=0.1, last_iterate=x
            )

        with (
            patch.object(
                adapter,
                "initialize_barrier_parameter",
                new=SolveBestBoundsFallbackTest._pinned_t0,
            ),
            patch.object(adapter, "centering_step", new=scripted),
            self.assertRaises(InteriorPointMethodError),
        ):
            adapter.solve(x0=np.full(3, 0.5))

    def test_early_stops_when_gap_within_tolerance(self) -> None:
        """Once the best-bounds gap is within ``tolerance``, the outer loop stops
        instead of marching ``t`` through the remaining ``num_steps`` into the
        ill-conditioned high-``t`` regime."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        trace = ListTrace()
        settings = OptimizationSettings(
            outer=OuterSettings(tolerance=1e-3), trace=trace
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)
        calls: list[int] = []

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            calls.append(outer_nit)
            # Step 1 brackets loosely (gap 8); step 2 closes the gap to 0, which is
            # within tolerance, so the loop must stop before step 3.
            if outer_nit == 1:
                return SolveBestBoundsFallbackTest._result(10.0, 2.0, 1.0, 1.0, 0)
            return SolveBestBoundsFallbackTest._result(5.0, 5.0, 2.0, 2.0, 0)

        with (
            patch.object(
                adapter,
                "initialize_barrier_parameter",
                new=SolveBestBoundsFallbackTest._pinned_t0,
            ),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        # Stopped at step 2; never ran step 3+ even though num_steps > 2.
        assert calls == [1, 2]
        assert result.status is SolveStatus.OPTIMAL
        np.testing.assert_allclose(result.suboptimality, 0.0, rtol=1e-6, atol=1e-12)
        # Reports the converged step-2 primal.
        np.testing.assert_array_equal(result.solution, np.full(3, 2.0))
        # The convergence event fired exactly once, at step 2.
        converged = [r for r in trace.records() if r["event"] == "converged"]
        assert len(converged) == 1
        assert converged[0]["outer_nit"] == 2


class InitializeBarrierParameterFloorTest(unittest.TestCase):
    """The gradient-balance ``t0`` floor is its own setting, not ``tolerance_soft``.

    Regression for the D105770599 / D107458456 interaction: D105770599 floored
    ``t0`` at ``outer.tolerance_soft`` (fine when that defaulted to 1e-3), but
    D107458456 raised CDM's ``tolerance_soft`` to 100.0 for acceptance at scale,
    which retroactively forced ``t0 = 100`` -- starting the first centering far
    off the central path, where Newton diverges and BTLS collapses. The floor is
    now ``outer.min_initial_barrier_parameter`` (default 1e-3), independent of
    ``tolerance_soft`` and tunable by the caller.
    """

    def test_large_tolerance_soft_does_not_inflate_t0(self) -> None:
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        x0 = np.full(3, 0.5)
        t_default = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(outer=OuterSettings(tolerance_soft=1e-3)),
        ).initialize_barrier_parameter(x0)
        t_loose = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(outer=OuterSettings(tolerance_soft=100.0)),
        ).initialize_barrier_parameter(x0)
        # tolerance_soft no longer feeds the floor, so t0 is unchanged...
        self.assertAlmostEqual(t_loose, t_default)
        # ...and well below the acceptance scale (the old bug returned 100.0).
        assert t_loose < 100.0

    def test_min_initial_barrier_parameter_floors_t0(self) -> None:
        """``min_initial_barrier_parameter`` is honored as the lower bound on t0."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        x0 = np.full(3, 0.5)
        # The natural gradient-balance t0 (a negligible floor lets the ratio win).
        natural = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(min_initial_barrier_parameter=1e-12)
            ),
        ).initialize_barrier_parameter(x0)
        # A floor above the natural t0 binds, pinning t0 to the floor.
        big_floor = natural + 1.0
        t0 = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(min_initial_barrier_parameter=big_floor)
            ),
        ).initialize_barrier_parameter(x0)
        self.assertAlmostEqual(t0, big_floor)


class BarrierSolverFeasibilityModeTest(unittest.TestCase):
    r"""A ``BarrierSolver`` stops early when the objective carries a feasibility threshold.

    An objective with ``feasibility_threshold`` set puts the barrier loop in
    feasibility mode: it stops at the first iterate whose value crosses the
    threshold, so the same solver doubles as a feasibility solver. An ordinary
    objective (no threshold) runs a full optimization solve.
    """

    def test_threshold_stops_at_negative_objective(self) -> None:
        """``min x  s.t.  x >= -5``: a threshold objective stops once ``x < 0``."""
        x0 = np.array([3.0])
        feasibility = BarrierSolver().solve(
            problem=Problem(
                obj=LinearObjective(c=np.array([1.0])).with_feasibility_threshold(0.0),
                ineq=[BoundsConstraintLayer(lb=-5.0, dimension=1)],
            ),
            x0=x0,
        )
        optimum = BarrierSolver().solve(
            problem=Problem(
                obj=LinearObjective(c=np.array([1.0])),
                ineq=[BoundsConstraintLayer(lb=-5.0, dimension=1)],
            ),
            x0=x0,
        )

        # Threshold mode returns a strictly feasible point (objective < 0) but does
        # not chase the optimum at the lower bound.
        assert float(feasibility.solution[0]) < 0.0
        assert float(feasibility.solution[0]) > float(optimum.solution[0]) - 1e-9
        # An ordinary objective drives to the lower bound.
        self.assertAlmostEqual(float(optimum.solution[0]), -5.0, delta=1e-2)

    @parameterized.expand([(1111, 5), (1112, 20)])
    def test_feasibility_recovers_infeasible_x0(self, seed: int, M: int) -> None:
        """A feasibility solve recovers a feasible point from an infeasible ``x0``.

        ``BarrierSolver.solve`` on an ``obj=None`` problem dispatches to the
        feasibility solve, which feasibilizes the bounds-violating start and
        returns a strictly feasible point.
        """
        rng = np.random.default_rng(seed)
        problem = Problem(ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)])
        x0 = rng.uniform(-2.0, -0.5, size=M)
        assert not problem.is_feasible(x0)

        result = BarrierSolver().solve(problem, x0=x0)
        assert problem.is_feasible(np.asarray(result.solution))

    def test_equality_threshold_stops_at_negative_objective(self) -> None:
        """Equality variant: ``min x0 + x1  s.t.  x0 = x1, x >= -5``."""
        obj = LinearObjective(c=np.array([1.0, 1.0])).with_feasibility_threshold(0.0)
        problem = Problem(
            obj=obj,
            ineq=[BoundsConstraintLayer(lb=-5.0, dimension=2)],
            eq=EqualityConstraintLayer(A=np.array([[1.0, -1.0]]), b=np.array([0.0])),
        )
        x0 = np.array([3.0, 3.0])  # on the manifold x0 = x1, strictly feasible

        feasibility = BarrierSolver().solve(problem, x0=x0)
        # Objective (the feasibility residual) is driven strictly below zero.
        assert float(obj.evaluate(np.asarray(feasibility.solution))) < 0.0

    def test_already_feasible_x0_returns_zero_iteration_result(self) -> None:
        """An already-feasible threshold solve returns a zero-iteration result.

        When the starting iterate already satisfies the threshold, no centering
        runs; the solver returns a zero-iteration ``SolveResult``.
        """
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0])).with_feasibility_threshold(0.0),
            ineq=[BoundsConstraintLayer(lb=-5.0, dimension=1)],
        )
        x0 = np.array([-1.0])  # objective = -1 < 0 -> already feasible

        result = BarrierSolver().solve(problem, x0=x0)

        assert isinstance(result, SolveResult)
        assert result.n_iterations == 0
        self.assertAlmostEqual(float(result.solution[0]), -1.0)


class BarrierInitStrategyTest(unittest.TestCase):
    """``OuterSettings.barrier_init`` selects the barrier ``t0`` strategy.

    ``GRADIENT_BALANCE`` (the default) preserves the adaptive gradient-balance
    ``t0`` the barrier adapters have always used; ``UNIT`` falls back to the base
    ``t0 = 1`` for problem classes the adaptive ratio does not suit.
    """

    def test_default_is_gradient_balance(self) -> None:
        """The default is ``GRADIENT_BALANCE`` -- unchanged from before the seam."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        x0 = np.full(3, 0.5)
        t_default = _BarrierAdapter(
            problem=problem, settings=OptimizationSettings()
        ).initialize_barrier_parameter(x0)
        t_explicit = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(barrier_init=BarrierInitStrategy.GRADIENT_BALANCE)
            ),
        ).initialize_barrier_parameter(x0)
        self.assertAlmostEqual(t_default, t_explicit)
        # The QP fixture's gradient-balance t0 is genuinely adaptive, not the
        # unit fallback -- so this asserts the default really is the ratio.
        assert not np.isclose(t_default, 1.0, rtol=1e-6, atol=1e-12)

    def test_unit_strategy_returns_one(self) -> None:
        """``UNIT`` bypasses the ratio and returns the base ``t0 = 1``."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        x0 = np.full(3, 0.5)
        t0 = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(barrier_init=BarrierInitStrategy.UNIT)
            ),
        ).initialize_barrier_parameter(x0)
        self.assertAlmostEqual(t0, 1.0)

    def test_equality_unit_strategy_returns_one(self) -> None:
        """The equality adapter's ``UNIT`` skips the Schur refinement -> ``t0 = 1``."""
        M = 3
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(M), hessian_contribution=DenseSystem(np.eye(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([1.0])),
        )
        x0 = np.full(M, 1.0 / M)  # on the manifold A x = b, strictly feasible
        t0 = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(barrier_init=BarrierInitStrategy.UNIT)
            ),
        ).initialize_barrier_parameter(x0)
        self.assertAlmostEqual(t0, 1.0)

    def test_no_equality_rejects_equality_schur(self) -> None:
        """``EQUALITY_SCHUR`` has no ``A`` to work with on a no-equality solve."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        adapter = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(barrier_init=BarrierInitStrategy.EQUALITY_SCHUR)
            ),
        )
        with self.assertRaisesRegex(ValueError, "EQUALITY_SCHUR"):
            adapter.initialize_barrier_parameter(np.full(3, 0.5))

    def test_objective_value_gap_degrades_to_unit_without_estimate(self) -> None:
        """``OBJECTIVE_VALUE_GAP`` with no estimate degrades to ``t0 = 1``."""
        problem = CenteringStepQuadraticPhaseUnitStepTest._qp_problem()
        t0 = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(
                    barrier_init=BarrierInitStrategy.OBJECTIVE_VALUE_GAP
                )
            ),
        ).initialize_barrier_parameter(np.full(3, 0.5))
        self.assertAlmostEqual(t0, 1.0)

    def test_objective_value_gap_uses_estimate(self) -> None:
        """``t0 = m / (mu (f0 - p_star))``, clamped, from the value estimate."""
        M = 3
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(M), hessian_contribution=DenseSystem(np.eye(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = np.ones(M)  # strictly feasible; f0 = 0.5 * ||x0||^2 = 1.5
        adapter = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(
                    barrier_init=BarrierInitStrategy.OBJECTIVE_VALUE_GAP,
                    objective_value_estimate=1.4,
                )
            ),
        )
        assert adapter.num_ineq_constraints == M  # one lower bound per coordinate
        # gap = f0 - p_star = 1.5 - 1.4 = 0.1; t0 = 3 / (10 * 0.1) = 3.0
        # (below the cap m/(mu eps) = 3e5 and above the floor 1.0).
        self.assertAlmostEqual(adapter.initialize_barrier_parameter(x0), 3.0)

    def test_objective_value_gap_near_optimal_skips_to_end(self) -> None:
        """When ``f0 <= p_star`` the init jumps to ``t_max / mu^3``."""
        M = 3
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(M), hessian_contribution=DenseSystem(np.eye(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = np.ones(M)  # f0 = 1.5
        adapter = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(
                    barrier_init=BarrierInitStrategy.OBJECTIVE_VALUE_GAP,
                    objective_value_estimate=1.5,  # equals f0 -> near-optimal
                )
            ),
        )
        # t_max = m / (mu eps) = 3 / (10 * 1e-6) = 3e5; t0 = t_max / mu^3 = 300.0.
        np.testing.assert_allclose(
            adapter.initialize_barrier_parameter(x0), 300.0, rtol=1e-6, atol=1e-12
        )

    def test_equality_objective_value_gap_uses_estimate(self) -> None:
        """The equality adapter also honors ``OBJECTIVE_VALUE_GAP``."""
        M = 3
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(M), hessian_contribution=DenseSystem(np.eye(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([3.0])),
        )
        x0 = np.ones(M)  # on the manifold sum = 3, strictly feasible; f0 = 1.5
        adapter = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(
                    barrier_init=BarrierInitStrategy.OBJECTIVE_VALUE_GAP,
                    objective_value_estimate=1.4,
                )
            ),
        )
        assert adapter.num_ineq_constraints == M
        # Same as the no-equality case: gap = 0.1, t0 = 3 / (10 * 0.1) = 3.0.
        self.assertAlmostEqual(adapter.initialize_barrier_parameter(x0), 3.0)

    def test_equality_default_matches_equality_schur(self) -> None:
        """The equality adapter's default (None) is the equality-aware Schur init."""
        M = 3
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(M), hessian_contribution=DenseSystem(np.eye(M))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([1.0])),
        )
        x0 = np.full(M, 1.0 / M)  # on the manifold A x = b, strictly feasible
        t_default = _BarrierAdapter(
            problem=problem, settings=OptimizationSettings()
        ).initialize_barrier_parameter(x0)
        t_schur = _BarrierAdapter(
            problem=problem,
            settings=OptimizationSettings(
                outer=OuterSettings(barrier_init=BarrierInitStrategy.EQUALITY_SCHUR)
            ),
        ).initialize_barrier_parameter(x0)
        self.assertAlmostEqual(t_default, t_schur)

    def test_equality_can_select_plain_gradient_balance(self) -> None:
        """``GRADIENT_BALANCE`` on the equality adapter is the plain ratio (ignores A)."""
        M = 3
        objective = QuadraticObjective(
            c=-np.ones(M), hessian_contribution=DenseSystem(np.eye(M))
        )
        constraints: list[ConstraintLayer] = [
            BoundsConstraintLayer(lb=0.0, dimension=M)
        ]
        x0 = np.full(M, 1.0 / M)
        gb = OptimizationSettings(
            outer=OuterSettings(barrier_init=BarrierInitStrategy.GRADIENT_BALANCE)
        )
        t_eq = _BarrierAdapter(
            problem=Problem(
                obj=objective,
                ineq=constraints,
                eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([1.0])),
            ),
            settings=gb,
        ).initialize_barrier_parameter(x0)
        t_noeq = _BarrierAdapter(
            problem=Problem(obj=objective, ineq=constraints),
            settings=gb,
        ).initialize_barrier_parameter(x0)
        # The equality adapter's GRADIENT_BALANCE ignores A, so it matches the
        # no-equality adapter's ratio on the same objective and inequalities.
        self.assertAlmostEqual(t_eq, t_noeq)


class DirectFormBarrierHintTest(unittest.TestCase):
    r"""A direct-form Phase I raises ``t0`` to the active layer's ``barrier_parameter_hint``.

    ``Problem.make_phase1`` hoists a ``DirectConstraintLayer`` active to the
    objective; the legacy ``MinimizeConstraintValueSolver`` floored ``t0`` at the
    layer's ``barrier_parameter_hint``. The structured barrier adapters must do
    the same so direct-form Phase I does not under-progress when MCVS is retired.

    For a ``SquaredNormConstraintLayer`` objective (``f_0 = w^T w - phi``) over a
    ``BoundsConstraintLayer(lb=0)`` background at a uniform ``x0 = a * 1``, the
    geometry gives a clean identity: the gradient-balance ratio is ``1/(2 a^2)``
    while the hint ``m / max(tol, a^2 M) = 1/a^2`` (for ``a^2 M > tol``) is
    exactly twice it -- so the hint always dominates and ``t0`` should equal it.
    """

    @staticmethod
    def _direct_form_problem(M: int, phi: float = 100.0) -> Problem:
        """A direct-form Phase I problem: SquaredNorm objective, Bounds background."""
        return Problem(
            obj=SquaredNormConstraintLayer(phi=phi, dimension=M),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )

    def test_helper_floors_t0_at_direct_form_hint(self) -> None:
        """``_raise_to_objective_hint`` raises a small ``t0`` and leaves a large one."""
        M = 10
        x0 = 0.02 * np.ones(M)
        active = SquaredNormConstraintLayer(phi=100.0, dimension=M)
        problem = Problem(
            obj=active,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        hint = active.barrier_parameter_hint(x0, num_ineq=M, outer_tolerance=1e-6)
        assert hint is not None
        # A t0 below the hint is raised to it; one above is left untouched.
        self.assertAlmostEqual(
            _raise_to_objective_hint(1.0, problem, x0, M, 1e-6), hint
        )
        self.assertAlmostEqual(
            _raise_to_objective_hint(hint * 10.0, problem, x0, M, 1e-6), hint * 10.0
        )

    def test_helper_noop_when_objective_not_direct_form(self) -> None:
        """A plain (non-``DirectConstraintLayer``) objective leaves ``t0`` unchanged."""
        M = 5
        problem = Problem(
            obj=LinearObjective(c=np.zeros(M)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = np.ones(M)
        self.assertAlmostEqual(_raise_to_objective_hint(3.0, problem, x0, M, 1e-6), 3.0)

    def test_barrier_adapter_raises_t0_to_hint(self) -> None:
        """The no-equality adapter floors gradient-balance ``t0`` at the hint."""
        M = 12
        a = 0.02
        x0 = a * np.ones(M)  # strictly feasible (w > 0)
        active = SquaredNormConstraintLayer(phi=100.0, dimension=M)
        problem = Problem(
            obj=active,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        settings = OptimizationSettings()  # default barrier_init -> GRADIENT_BALANCE
        adapter = _BarrierAdapter(problem=problem, settings=settings)
        hint = active.barrier_parameter_hint(
            x0,
            num_ineq=adapter.num_ineq_constraints,
            outer_tolerance=settings.outer.tolerance,
        )
        assert hint is not None
        t0 = adapter.initialize_barrier_parameter(x0)
        # Hint = 1/a^2 dominates the bare gradient-balance ratio 1/(2 a^2).
        np.testing.assert_allclose(t0, hint, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(hint, 1.0 / (a * a), rtol=1e-6, atol=1e-12)

    def test_equality_adapter_raises_gradient_balance_t0_to_hint(self) -> None:
        """The equality adapter (GRADIENT_BALANCE) floors ``t0`` at the hint."""
        M, p = 20, 4
        a = 0.02
        rng = np.random.default_rng(7)
        x0 = a * np.ones(M)
        A = rng.standard_normal((p, M))
        b = A @ x0  # x0 lies on the equality manifold
        active = SquaredNormConstraintLayer(phi=1000.0, dimension=M)
        problem = Problem(
            obj=active,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        settings = OptimizationSettings(
            outer=OuterSettings(barrier_init=BarrierInitStrategy.GRADIENT_BALANCE)
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)
        hint = active.barrier_parameter_hint(
            x0,
            num_ineq=adapter.num_ineq_constraints,
            outer_tolerance=settings.outer.tolerance,
        )
        assert hint is not None
        np.testing.assert_allclose(
            adapter.initialize_barrier_parameter(x0), hint, rtol=1e-6, atol=1e-12
        )

    def test_equality_schur_default_respects_hint_floor(self) -> None:
        """The equality adapter's default (EQUALITY_SCHUR) still applies the hint floor."""
        M, p = 20, 4
        a = 0.02
        rng = np.random.default_rng(9)
        x0 = a * np.ones(M)
        A = rng.standard_normal((p, M))
        b = A @ x0
        active = SquaredNormConstraintLayer(phi=1000.0, dimension=M)
        problem = Problem(
            obj=active,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        settings = OptimizationSettings()  # default -> EQUALITY_SCHUR
        adapter = _BarrierAdapter(problem=problem, settings=settings)
        hint = active.barrier_parameter_hint(
            x0,
            num_ineq=adapter.num_ineq_constraints,
            outer_tolerance=settings.outer.tolerance,
        )
        assert hint is not None
        # The Schur-complement t0 sits below the hint here, so the floor binds
        # exactly -- a stronger check than ``>= hint`` (which would pass even if
        # the floor were removed).
        np.testing.assert_allclose(
            adapter.initialize_barrier_parameter(x0), hint, rtol=1e-6, atol=1e-12
        )

    def test_unit_strategy_does_not_apply_hint(self) -> None:
        """``UNIT`` is the literal escape hatch: ``t0 = 1`` even with a hint-provider."""
        M = 12
        x0 = 0.02 * np.ones(M)
        problem = DirectFormBarrierHintTest._direct_form_problem(M)
        settings = OptimizationSettings(
            outer=OuterSettings(barrier_init=BarrierInitStrategy.UNIT)
        )
        adapter = _BarrierAdapter(problem=problem, settings=settings)
        self.assertAlmostEqual(adapter.initialize_barrier_parameter(x0), 1.0)


class FeasibilityThresholdEarlyStopTest(unittest.TestCase):
    """The barrier loop halts once the objective crosses its ``feasibility_threshold``."""

    def test_stops_at_threshold_above_the_optimum(self) -> None:
        """A threshold-carrying objective stops the solve short of the true optimum."""
        bounds = BoundsConstraintLayer(lb=-10.0, dimension=1)  # w >= -10
        x0 = np.array([5.0])  # strictly feasible start
        # Plain solve drives w toward the lower bound (~ -10).
        plain = BarrierSolver().solve(
            Problem(obj=LinearObjective(c=np.array([1.0])), ineq=[bounds]),
            x0=x0,
        )
        # Attaching feasibility_threshold=0 makes the solve stop as soon as the
        # objective drops below 0, well short of -10.
        feasible = BarrierSolver().solve(
            Problem(
                obj=LinearObjective(c=np.array([1.0])).with_feasibility_threshold(0.0),
                ineq=[bounds],
            ),
            x0=x0,
        )
        assert feasible.status is SolveStatus.MAX_ITERATIONS
        assert feasible.objective_value < 0.0  # crossed the threshold
        # Stopped short of the optimum that the plain solve reaches.
        assert feasible.objective_value > plain.objective_value


class OnInfeasibleContinueTest(unittest.TestCase):
    r"""``on_infeasible="continue"`` drives the Phase I to its optimum before raising.

    The default ``"raise"`` quits as soon as an infeasibility certificate appears
    (dual > 0).  ``"continue"`` re-solves the Phase I as a plain optimization so
    the slack ``s`` is driven all the way to its minimum ``s*`` before the
    ``ProblemCertifiablyInfeasibleError`` is raised -- giving the caller the
    best-effort minimum-violation weights.

    Problem geometry (used across tests):
        w1 + w2 = 1,  w1, w2 >= 0
        |w1 - w2 - 2| <= 0.5   (imbalance on B = [[1, -1]], c = [2], psi = 0.5)

    Since w1 + w2 = 1 forces w1 - w2 in [-1, 1], the imbalance target 2 ± 0.5
    is unreachable.  The Phase I slack form minimizes ``s`` s.t.
    ``|w1 - w2 - 2| <= 0.5 + s``; the optimum is at w* = (1, 0) with
    ``|1 - 0 - 2| = 1``, giving ``s* = 0.5``.
    """

    @staticmethod
    def _make_infeasible_problem() -> Problem:
        eq = EqualityConstraintLayer(A=np.array([[1.0, 1.0]]), b=np.array([1.0]))
        return Problem(
            eq=eq,
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=2),
                ImbalanceConstraintLayer(
                    B=np.array([[1.0, -1.0]]),
                    c=np.array([2.0]),
                    psi=0.5,
                ),
            ],
        )

    @staticmethod
    def _make_feasible_problem() -> Problem:
        eq = EqualityConstraintLayer(A=np.array([[1.0, 1.0]]), b=np.array([1.0]))
        return Problem(
            eq=eq,
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=2),
                ImbalanceConstraintLayer(
                    B=np.array([[1.0, -1.0]]),
                    c=np.array([0.0]),
                    psi=0.6,
                ),
            ],
        )

    def test_feasible_problem_returns_normally(self) -> None:
        """``on_infeasible="continue"`` is a no-op when the problem is feasible."""
        problem = OnInfeasibleContinueTest._make_feasible_problem()
        result = problem.solve(BarrierSolver(), on_infeasible="continue")
        assert problem.is_feasible(np.asarray(result.solution))

    def test_infeasible_raises_certifiable_error(self) -> None:
        """``on_infeasible="continue"`` still raises for a certifiably infeasible problem."""
        problem = OnInfeasibleContinueTest._make_infeasible_problem()
        with self.assertRaises(ProblemCertifiablyInfeasibleError):
            problem.solve(BarrierSolver(), on_infeasible="continue")

    def test_continue_result_is_at_phase1_optimum(self) -> None:
        r"""The result attached to the error is the optimal Phase I (minimum slack).

        ``on_infeasible="raise"`` quits early (possibly before the Phase I has
        converged to its slack minimum), while ``on_infeasible="continue"`` drives
        the slack ``s`` all the way to ``s*``.  The ``objective_value`` of the
        "continue" result must therefore be at most that of the "raise" result.

        For this geometry the Phase I slack minimizer is exactly s* = 0.5, so we
        also verify the result is close to that known value.
        """
        problem = OnInfeasibleContinueTest._make_infeasible_problem()
        with self.assertRaises(ProblemCertifiablyInfeasibleError) as exc_raise:
            problem.solve(BarrierSolver(), on_infeasible="raise")
        with self.assertRaises(ProblemCertifiablyInfeasibleError) as exc_continue:
            problem.solve(BarrierSolver(), on_infeasible="continue")

        obj_raise = exc_raise.exception.result.objective_value
        obj_continue = exc_continue.exception.result.objective_value
        # "continue" must be at least as good (lower slack) as "raise".
        assert obj_continue <= obj_raise + 1e-6
        # Phase I optimal slack is s* = 0.5; "continue" must be close.
        assert abs(obj_continue - 0.5) < 1e-3

    def test_continue_result_carries_recovered_weights(self) -> None:
        """The error's result is already de-augmented: solution has the original M entries.

        ``finalize_solution`` drops the slack-form Phase I's trailing slack before
        the result is attached to the error, so callers get valid weights directly
        rather than an M+1 vector they must trim.
        """
        problem = OnInfeasibleContinueTest._make_infeasible_problem()
        with self.assertRaises(ProblemCertifiablyInfeasibleError) as exc_info:
            problem.solve(BarrierSolver(), on_infeasible="continue")
        result = exc_info.exception.result
        # The slack has already been dropped: solution is at the original dimension.
        assert len(result.solution) == problem.dimension
        # The recovered weights satisfy the equality constraints directly.
        w = np.asarray(result.solution)
        assert problem.eq is not None
        assert np.allclose(problem.eq.A @ w, problem.eq.b, atol=1e-6)


class TestEvaluateDualDecrementReuseAndRegionGuard(unittest.TestCase):
    r"""``evaluate_dual`` with the reused ``delta^2`` and the region guard.

    The mini-Newton now returns ``(x_min, delta^2)`` and ``evaluate_dual`` returns
    ``L(x_min) - delta^2`` only inside the self-concordant quadratic region
    (``delta^2 <= 0.68^2``); outside it (or when ``delta^2`` is unavailable) it
    returns ``-inf`` rather than a possibly-invalid bound.

    """

    @parameterized.expand([(3201,), (3202,), (3203,)])
    def test_dual_is_lower_bound_on_inf_L(self, seed: int) -> None:
        r"""``evaluate_dual`` returns a valid lower bound on ``inf_x L`` (hence on primal).

        For a strictly-convex QP Lagrangian the exact infimum is
        ``inf_x L = L(x_min) - (1/2) delta^2(x0)`` at the analytical minimizer. The
        returned dual ``L(x_min) - delta^2`` must not exceed ``inf_x L`` (weak
        duality: ``inf_x L <= primal optimum``), i.e. it is a conservative bound.
        """
        rng = np.random.default_rng(seed)
        M = 5
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        # Warm start near the minimizer so the mini-Newton stays in-region (finite).
        x_min_analytic = np.linalg.solve(Q, lmbda)
        x_star = x_min_analytic + 1e-3 * rng.standard_normal(M)

        dual = adapter.evaluate_dual(lmbda=lmbda, nu=np.zeros(0), x_star=x_star)

        inf_L = float(
            0.5 * x_min_analytic @ Q @ x_min_analytic - lmbda @ x_min_analytic
        )
        assert dual != float("-inf")
        # The corrected dual is a lower bound on inf_x L (looser by (1/2) delta^2).
        assert dual <= inf_L + 1e-9

    def test_minimize_lagrangian_leaves_region_after_one_far_step(self) -> None:
        r"""A far warm start with ``max_iter=1`` leaves ``delta^2`` out of region.

        ``L = -sum log(w) + lambda^T (w - ub)`` is self-concordant but not quadratic,
        so from a far warm start a single damped Newton step does not reach the
        quadratic region: the squared decrement at the final iterate exceeds
        ``0.68^2``. This is the precondition the region guard keys off of.
        """
        M = 3
        problem = Problem(
            obj=_LogObjective(dimension=M),
            ineq=[BoundsConstraintLayer(ub=1e6, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = np.ones(M)
        # Far from the minimizer w = 1/lambda = 1.
        x0 = np.full(M, 200.0)

        x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda, max_iter=1)

        assert x_min is not None
        assert delta2 is not None
        assert delta2 > _DUAL_REGION_DECREMENT_SQ_MAX

    def test_minimize_lagrangian_aborts_when_domain_cap_blocks_step(self) -> None:
        r"""A domain cap too tight for the line search aborts to ``(None, None)``.

        When ``btls_keep_in_domain`` returns a max step so small that
        ``beta * max_step < min_step``, ``LineSearch.find_step`` raises
        ``ConstraintBoundaryError``; the mini-Newton loop catches it and returns
        ``(None, None)``. This pins the new conservative behavior: a near-domain-
        boundary warm start now aborts to the plug-in ``L(x_star)`` fallback the
        region guard relies on, rather than forcing a damped step.
        """
        rng = np.random.default_rng(3401)
        M = 4
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        # Far from the minimizer w = Q^{-1} lambda so the loop reaches the line
        # search (gradient is well above the cheap convergence floor).
        x0 = np.full(M, 50.0)
        # A max step so small that beta * max_step < min_step, forcing the line
        # search to raise ConstraintBoundaryError on the first iterate.
        beta = adapter.settings.armijo.beta
        min_step = adapter.settings.armijo.min_step
        tiny_cap = min_step / beta / 100.0

        obj = problem.obj
        assert obj is not None
        with patch.object(
            obj,
            "btls_keep_in_domain",
            return_value=tiny_cap,
        ):
            x_min, delta2 = adapter._minimize_lagrangian(x0=x0, lmbda=lmbda)

        self.assertIsNone(x_min)
        self.assertIsNone(delta2)

    def test_region_guard_returns_minus_inf_when_out_of_region(self) -> None:
        r"""When the mini-Newton lands out of region, ``evaluate_dual`` returns -inf.

        Forces the out-of-region branch by having ``_minimize_lagrangian`` return a
        ``delta^2`` above ``0.68^2``. The guard must return ``-inf`` (a conservative
        deferral) rather than the possibly-invalid ``L(x_min) - delta^2``.
        """
        rng = np.random.default_rng(3301)
        M = 4
        Q = _make_psd_matrix(rng, M)
        problem = Problem(
            obj=QuadraticObjective(c=np.zeros(M), hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        lmbda = rng.uniform(0.1, 1.5, size=M)
        x_star = np.ones(M)

        out_of_region = (x_star, _DUAL_REGION_DECREMENT_SQ_MAX + 1.0)
        with patch.object(adapter, "_minimize_lagrangian", return_value=out_of_region):
            dual = adapter.evaluate_dual(lmbda=lmbda, nu=np.zeros(0), x_star=x_star)

        assert dual == float("-inf")


class BarrierSolverSparsePhase1Test(unittest.TestCase):
    """A sparse equality ``A`` survives Phase I end-to-end."""

    def test_sparse_equality_solve_through_phase1(self) -> None:
        # min c^T x  s.t.  sum(x) = 3, 0 <= x <= 10, with a sparse A.
        A = scipy.sparse.csr_matrix(np.array([[1.0, 1.0, 1.0]]))
        b = np.array([3.0])
        c = np.array([1.0, 2.0, 3.0])
        problem = Problem(
            obj=LinearObjective(c=c),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b, saddle_factory=SparseSaddleFactory()),
        )
        settings = OptimizationSettings(
            outer=OuterSettings(barrier_init=BarrierInitStrategy.UNIT)
        )
        # x0 sits on the lower bound (x[1] = x[2] = 0), so it is not strictly
        # interior -- the solver must run Phase I, which now handles sparse A.
        x0 = np.array([3.0, 0.0, 0.0])
        result = BarrierSolver(settings=settings).solve(problem=problem, x0=x0)

        self.assertTrue(result.succeeded)
        np.testing.assert_allclose(A.toarray() @ result.solution, b, atol=1e-5)
        self.assertGreater(result.solution.min(), -1e-6)
        # The minimizer loads the cheapest coordinate: x = [3, 0, 0], cost 3.
        self.assertTrue(np.isclose(float(c @ result.solution), 3.0, atol=1e-3))
