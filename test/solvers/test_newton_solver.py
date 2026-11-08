"""Test the NewtonSolver (unconstrained and equality-constrained)."""

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium.exceptions import (
    CenteringStepError,
    NewtonStepError,
)
from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.solvers.newton_solver import NewtonSolver


def _make_psd_matrix(rng: np.random.Generator, M: int) -> npt.NDArray[np.float64]:
    """Sample a random symmetric PD matrix with bounded condition number."""
    A = rng.standard_normal((M, M))
    return np.asarray(A @ A.T + M * np.eye(M))


class NewtonSolverTest(unittest.TestCase):
    """Tests for `NewtonSolver`."""

    @parameterized.expand(
        [
            (901, 5),
            (902, 25),
            (903, 100),
        ]
    )
    def test_solves_quadratic_to_optimum(self, seed: int, M: int) -> None:
        """For `min (1/2) w^T Q w + c^T w` the optimum is `w* = -Q^{-1} c`."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        x0 = rng.standard_normal(M)

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
        )
        result = NewtonSolver().solve(problem=problem, x0=x0)

        expected = np.linalg.solve(Q, -c)
        np.testing.assert_allclose(result.solution, expected, rtol=1e-8, atol=1e-8)

    @parameterized.expand([(911, 10), (912, 50)])
    def test_quadratic_converges_in_one_iteration(self, seed: int, M: int) -> None:
        """For a quadratic objective, Newton's method should converge in one step."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        x0 = rng.standard_normal(M)

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
        )
        result = NewtonSolver().solve(problem=problem, x0=x0)
        # Newton on a strictly convex quadratic is exact in one step (full step
        # accepted by line search).
        assert result.n_iterations <= 2

    def test_rejects_problem_with_inequality_constraints(self) -> None:
        """`solve` raises when the Problem has any inequality constraint."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        with self.assertRaisesRegex(ValueError, "no inequality constraints"):
            NewtonSolver().solve(problem=problem, x0=np.ones(3))

    def test_one_solver_handles_both_unconstrained_and_equality(self) -> None:
        """A single `NewtonSolver` instance auto-dispatches on ``problem.eq``.

        The same solver object solves an unconstrained problem (``eq=None``) and
        an equality-constrained problem to their respective optima.
        """
        rng = np.random.default_rng(9999)
        M, p = 20, 4
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        solver = NewtonSolver()

        # Unconstrained: optimum is -Q^{-1} c.
        x0 = rng.standard_normal(M)
        unconstrained = solver.solve(
            problem=Problem(
                obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ),
            x0=x0,
        )
        np.testing.assert_allclose(
            unconstrained.solution, np.linalg.solve(Q, -c), rtol=1e-8, atol=1e-8
        )

        # Equality-constrained: same solver, KKT optimum.
        A = rng.standard_normal((p, M))
        b = rng.standard_normal(p)
        x0_eq = _project_onto_equality(A, b, rng.standard_normal(M))
        equality = solver.solve(
            problem=Problem(
                obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
                eq=EqualityConstraintLayer(A=A, b=b),
            ),
            x0=x0_eq,
        )
        kkt = np.block([[Q, A.T], [A, np.zeros((p, p))]])
        expected = np.linalg.solve(kkt, np.concatenate([-c, b]))[:M]
        np.testing.assert_allclose(equality.solution, expected, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(A @ equality.solution, b, rtol=1e-7, atol=1e-7)

    def test_rejects_missing_x0(self) -> None:
        """`solve` raises when ``x0`` is ``None`` -- Newton needs an explicit start."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
        )
        with self.assertRaisesRegex(ValueError, "explicit starting point x0"):
            NewtonSolver().solve(problem=problem, x0=None)

    @parameterized.expand([(7001, 10), (7002, 25), (7003, 50)])
    def test_gradient_called_once_per_newton_iteration(self, seed: int, M: int) -> None:
        """Adapter's id(x)-keyed cache prevents the centering loop from recomputing
        the objective gradient at the same iterate. Without caching, each Newton
        iteration calls `_objective.gradient(x)` twice (once in `newton_step`,
        once in `backtracking_line_search` via `gradient_barrier`). With caching,
        each iteration triggers exactly one underlying call.
        """

        class _GradientCountingQuadratic(QuadraticObjective):
            gradient_calls: int = 0

            def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                self.gradient_calls += 1
                return super().gradient(w)

        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        x0 = rng.standard_normal(M)

        obj = _GradientCountingQuadratic(c=c, hessian_contribution=DenseSystem(Q))
        result = NewtonSolver().solve(
            problem=Problem(
                obj=obj,
            ),
            x0=x0,
        )
        assert obj.gradient_calls == result.n_iterations, (
            f"expected {result.n_iterations} gradient calls (one per Newton iter); "
            f"got {obj.gradient_calls}"
        )

    def test_rejects_singular_objective(self) -> None:
        """`LinearObjective` has a zero Hessian -- its ``hessian_contribution`` is a
        ``ZeroSystem``, whose ``solve`` raises `NewtonStepError`. The IPM loop
        re-raises as `CenteringStepError` with the `NewtonStepError` chained as
        ``__cause__``. Guards against silent NaN/Inf if the underlying contract
        changes.
        """
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 2.0, 3.0])),
        )
        with self.assertRaisesRegex(CenteringStepError, "Failed to calculate") as ctx:
            NewtonSolver().solve(problem=problem, x0=np.zeros(3))
        assert isinstance(ctx.exception.__cause__, NewtonStepError)
        assert "not invertible" in str(ctx.exception.__cause__)


def _project_onto_equality(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the closest point to ``x`` satisfying ``A y = b`` (least-norm projection)."""
    nu = np.linalg.solve(A @ A.T, A @ x - b)
    return np.asarray(x - A.T @ nu)


class NewtonSolverWithEqualityTest(unittest.TestCase):
    """Tests for `NewtonSolver` on equality-constrained problems."""

    @parameterized.expand(
        [
            (1001, 5, 1),
            (1002, 25, 5),
            (1003, 100, 20),
        ]
    )
    def test_solves_equality_quadratic(self, seed: int, M: int, p: int) -> None:
        """For `min (1/2) w^T Q w + c^T w` s.t. `A w = b`, Newton finds the KKT point."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        A = rng.standard_normal((p, M))
        b = rng.standard_normal(p)
        x0_raw = rng.standard_normal(M)
        x0 = _project_onto_equality(A, b, x0_raw)

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = NewtonSolver().solve(problem=problem, x0=x0)

        # Closed-form KKT solution:
        #   Q w + c + A^T nu = 0
        #   A w = b
        # Combine into block KKT system and solve directly as the reference.
        kkt = np.block([[Q, A.T], [A, np.zeros((p, p))]])
        rhs = np.concatenate([-c, b])
        expected = np.linalg.solve(kkt, rhs)[:M]

        np.testing.assert_allclose(result.solution, expected, rtol=1e-7, atol=1e-7)
        # Equality residual should be tiny.
        np.testing.assert_allclose(A @ result.solution, b, rtol=1e-7, atol=1e-7)

    @parameterized.expand([(1011, 10, 3), (1012, 50, 8)])
    def test_quadratic_converges_quickly(self, seed: int, M: int, p: int) -> None:
        """For an equality-constrained quadratic, Newton converges in 1-2 steps."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        A = rng.standard_normal((p, M))
        b = rng.standard_normal(p)
        x0 = _project_onto_equality(A, b, rng.standard_normal(M))

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = NewtonSolver().solve(problem=problem, x0=x0)
        assert result.n_iterations <= 2

    @parameterized.expand([(7101, 10, 3), (7102, 25, 5)])
    def test_gradient_called_once_per_newton_iteration(
        self, seed: int, M: int, p: int
    ) -> None:
        """Equality adapter caches gradient per iteration; same contract as the
        unconstrained `_ObjectiveNewtonAdapter`. Without caching, each Newton iter
        triggers two `_objective.gradient(x)` calls (newton_step + backtracking).
        """

        class _GradientCountingQuadratic(QuadraticObjective):
            gradient_calls: int = 0

            def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                self.gradient_calls += 1
                return super().gradient(w)

        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        A = rng.standard_normal((p, M))
        b = rng.standard_normal(p)
        x0 = _project_onto_equality(A, b, rng.standard_normal(M))

        obj = _GradientCountingQuadratic(c=c, hessian_contribution=DenseSystem(Q))
        result = NewtonSolver().solve(
            problem=Problem(obj=obj, eq=EqualityConstraintLayer(A=A, b=b)),
            x0=x0,
        )
        assert obj.gradient_calls == result.n_iterations, (
            f"expected {result.n_iterations} gradient calls (one per Newton iter); "
            f"got {obj.gradient_calls}"
        )

    def test_rejects_singular_objective(self) -> None:
        """`LinearObjective` has a zero Hessian -- the KKT solve uses the
        objective's ``hessian_contribution`` (a ``ZeroSystem``), whose ``solve``
        raises `NewtonStepError`. The IPM loop re-raises as `CenteringStepError`.
        Guards against silent NaN/Inf if the underlying contract changes.
        """
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 2.0, 3.0])),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        with self.assertRaisesRegex(CenteringStepError, "Failed to calculate") as ctx:
            NewtonSolver().solve(problem=problem, x0=x0)
        assert isinstance(ctx.exception.__cause__, NewtonStepError)
        assert "not invertible" in str(ctx.exception.__cause__)
