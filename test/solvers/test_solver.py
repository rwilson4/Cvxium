"""Test the Solver abstraction."""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium._typing import none_throws
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
)
from cvxium.problems.objective import LinearObjective
from cvxium.problems.problem import (
    OnInfeasible,
    Problem,
)
from cvxium.problems.quadratic_constraints import (
    NormConstraintLayer,
)
from cvxium.problems.result import (
    PrimalDualDiagnostics,
    SolveResult,
    SolveStatus,
)
from cvxium.solvers.barrier_solver import (
    BarrierSolver,
)
from cvxium.solvers.newton_solver import NewtonSolver
from cvxium.solvers.optimization import (
    ProblemCertifiablyInfeasibleError,
)
from cvxium.solvers.primal_dual_solver import (
    PrimalDualSolver,
)
from cvxium.solvers.settings import (
    ArmijoSettings,
    InnerSettings,
    ObservabilitySettings,
    OptimizationSettings,
    OuterSettings,
)
from cvxium.solvers.solver import Solver


def _feasible(x0: npt.NDArray[np.float64]) -> SolveResult:
    """A bare feasibility ``SolveResult`` for trivial test solvers."""
    return SolveResult(
        solution=x0,
        status=SolveStatus.FEASIBLE,
        message="",
        solver="",
        n_iterations=0,
    )


class SolverTest(unittest.TestCase):
    """Tests for the `Solver` abstract base class."""

    def test_abc_cannot_be_instantiated(self) -> None:
        """Cannot instantiate `Solver` directly -- it's abstract."""
        abstract_cls: type[Solver] = Solver  # type: ignore[type-abstract]
        with self.assertRaisesRegex(TypeError, "abstract"):
            abstract_cls()

    def test_subclass_must_implement_solve(self) -> None:
        """A subclass that omits `solve` is still abstract."""

        class IncompleteSolver(Solver):
            pass

        incomplete_cls: type[Solver] = IncompleteSolver  # type: ignore[type-abstract]
        with self.assertRaisesRegex(TypeError, "abstract"):
            incomplete_cls()

    def test_settings_default_is_none(self) -> None:
        """`settings` defaults to ``None`` when not supplied."""

        class TrivialSolver(Solver):
            def solve(
                self,
                problem: Problem,
                x0: npt.NDArray[np.float64] | None = None,
                on_infeasible: OnInfeasible = "raise",
            ) -> SolveResult:
                return _feasible(none_throws(x0))

        solver = TrivialSolver()
        assert solver.settings is None

    @parameterized.expand([(NewtonSolver,), (PrimalDualSolver,)])
    def test_continue_policy_unsupported_raises(
        self,
        make_solver: type[Solver],
    ) -> None:
        """Solvers without a feasibility cascade reject ``on_infeasible="continue"``.

        Only the barrier (IPM) feasibility path can drive the Phase I slack to its
        optimum; the others raise ``NotImplementedError`` rather than silently
        ignoring the requested policy.
        """
        problem = Problem(
            obj=LinearObjective(c=np.zeros(2)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=2)],
        )
        with self.assertRaisesRegex(NotImplementedError, "continue"):
            problem.solve(make_solver(), on_infeasible="continue")

    def test_settings_is_stored(self) -> None:
        """An ``OptimizationSettings`` passed to ``__init__`` is exposed as ``settings``."""

        class TrivialSolver(Solver):
            def solve(
                self,
                problem: Problem,
                x0: npt.NDArray[np.float64] | None = None,
                on_infeasible: OnInfeasible = "raise",
            ) -> SolveResult:
                return _feasible(none_throws(x0))

        settings = OptimizationSettings()
        solver = TrivialSolver(settings=settings)
        assert solver.settings is settings

    def test_concrete_subclass_can_solve(self) -> None:
        """A trivial concrete subclass returns the initial guess unchanged."""

        class IdentitySolver(Solver):
            def solve(
                self,
                problem: Problem,
                x0: npt.NDArray[np.float64] | None = None,
                on_infeasible: OnInfeasible = "raise",
            ) -> SolveResult:
                return _feasible(none_throws(x0))

        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 2.0, 3.0])),
        )
        result = IdentitySolver().solve(problem, x0=np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(result.solution, np.array([1.0, 1.0, 1.0]))


class NestedSettingsDefaultsTest(unittest.TestCase):
    """Pin the default values of the nested settings dataclasses."""

    def test_outer_settings_defaults(self) -> None:
        s = OuterSettings()
        assert s.tolerance == 1e-6
        assert s.tolerance_soft == 1e-3
        assert s.barrier_multiplier == 10.0

    def test_inner_settings_defaults(self) -> None:
        s = InnerSettings()
        assert s.tolerance == 1e-6
        assert s.tolerance_soft == 1e-4
        assert s.max_iterations == 1_000
        assert s.stall_patience == 5
        assert s.stall_rel_improvement == 0.01

    def test_armijo_settings_defaults(self) -> None:
        s = ArmijoSettings()
        assert s.alpha == 0.01
        assert s.beta == 0.5
        assert s.min_step == 1e-3
        assert s.use_self_concordant_step_cap is False

    def test_observability_settings_defaults(self) -> None:
        s = ObservabilitySettings()
        assert s.verbose is False

    def test_overrides_take_effect(self) -> None:
        """Constructors accept overrides at the field level."""
        outer = OuterSettings(tolerance=1e-8, barrier_multiplier=5.0)
        assert outer.tolerance == 1e-8
        assert outer.barrier_multiplier == 5.0
        assert outer.tolerance_soft == 1e-3

        armijo = ArmijoSettings(use_self_concordant_step_cap=True)
        assert armijo.use_self_concordant_step_cap is True
        assert armijo.alpha == 0.01


class OptimizationSettingsNestedViewsTest(unittest.TestCase):
    """Pin the nested fields on ``OptimizationSettings``."""

    def test_default_construction_populates_views(self) -> None:
        settings = OptimizationSettings()
        assert settings.outer == OuterSettings()
        assert settings.inner == InnerSettings()
        assert settings.armijo == ArmijoSettings()
        assert settings.observability == ObservabilitySettings()


class OptimizationSettingsNestedKwargsTest(unittest.TestCase):
    """Pin the nested-kwarg constructor path on ``OptimizationSettings``."""

    def test_nested_outer_used_directly(self) -> None:
        outer = OuterSettings(
            tolerance=1e-8, tolerance_soft=1e-5, barrier_multiplier=20.0
        )
        settings = OptimizationSettings(outer=outer)
        assert settings.outer is outer
        assert settings.outer.tolerance == 1e-8

    def test_nested_inner_used_directly(self) -> None:
        inner = InnerSettings(tolerance=1e-7, tolerance_soft=1e-5, max_iterations=200)
        settings = OptimizationSettings(inner=inner)
        assert settings.inner is inner
        assert settings.inner.max_iterations == 200

    def test_nested_armijo_used_directly(self) -> None:
        armijo = ArmijoSettings(
            alpha=0.05, beta=0.7, min_step=1e-5, use_self_concordant_step_cap=True
        )
        settings = OptimizationSettings(armijo=armijo)
        assert settings.armijo is armijo
        assert settings.armijo.use_self_concordant_step_cap is True

    def test_nested_observability_used_directly(self) -> None:
        observability = ObservabilitySettings(verbose=True)
        settings = OptimizationSettings(observability=observability)
        assert settings.observability is observability
        assert settings.observability.verbose is True

    def test_categories_compose_independently(self) -> None:
        """Mixing nested kwargs across categories preserves each."""
        settings = OptimizationSettings(
            outer=OuterSettings(tolerance=1e-8),
            inner=InnerSettings(tolerance=1e-9),
        )
        assert settings.outer.tolerance == 1e-8
        assert settings.inner.tolerance == 1e-9


class SolveFeasibilityTest(unittest.TestCase):
    """`Solver._solve_feasibility` -- the recursive feasibility engine."""

    def test_raises_when_objective_present(self) -> None:
        """A problem with an objective is not a feasibility problem."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        with self.assertRaisesRegex(ValueError, "feasibility problem"):
            BarrierSolver()._solve_feasibility(problem, x0=np.zeros(3))

    def test_equality_only_projects_onto_manifold(self) -> None:
        """An obj=None equality-only problem solves by projecting onto ``A x = b``."""
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        problem = Problem(eq=EqualityConstraintLayer(A=A, b=b))
        result = BarrierSolver()._solve_feasibility(problem, x0=np.zeros(3))
        assert isinstance(result, SolveResult)
        np.testing.assert_allclose(A @ np.asarray(result.solution), b, atol=1e-9)

    def test_inequality_problem_returns_feasible_result(self) -> None:
        """An obj=None inequality problem solves to a strictly feasible point."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=1.0, dimension=3)])
        result = BarrierSolver()._solve_feasibility(problem, x0=np.zeros(3))
        solution = np.asarray(result.solution)
        assert solution.shape == (3,)
        assert problem.is_feasible(solution)

    def test_already_feasible_warm_start_is_returned(self) -> None:
        """A strictly feasible warm start is returned without a Phase I solve."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)])
        x0 = np.array([1.0, 2.0, 3.0])  # strictly feasible
        result = BarrierSolver()._solve_feasibility(problem, x0=x0)
        np.testing.assert_allclose(np.asarray(result.solution), x0)

    def test_infeasible_problem_raises(self) -> None:
        """An infeasible obj=None problem raises with the certificate."""
        M = 4
        A = np.ones((1, M))
        b = np.array([float(M)])  # mean weight 1, but lb=2 forces mean >= 2
        problem = Problem(
            ineq=[BoundsConstraintLayer(lb=2.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        with self.assertRaises(ProblemCertifiablyInfeasibleError):
            BarrierSolver()._solve_feasibility(problem, x0=np.zeros(M))

    def test_lone_direct_layer_short_circuits_to_analytic_interior(self) -> None:
        """A sole ellipsoid returns its analytic center with no Phase I solve.

        This is the direct-form Phase I shape that hoists the only layer into the
        objective, leaving zero inequalities -- a shape the barrier engine cannot
        solve. ``interior_point`` short-circuits it: the ellipsoid center is
        returned in zero iterations. ``x0`` is outside the ball, so the ordinary
        already-feasible short-circuit does not fire.
        """
        layer = NormConstraintLayer(B=np.eye(3), c=np.zeros(3), phi=1.0)
        problem = Problem(ineq=[layer])
        x0 = 2.0 * np.ones(3)  # ||x0||^2 = 12 > phi^2 = 1: not feasible
        result = BarrierSolver()._solve_feasibility(problem, x0=x0)
        solution = np.asarray(result.solution)
        np.testing.assert_allclose(solution, np.zeros(3), atol=1e-12)
        assert problem.is_feasible(solution)
        assert result.n_iterations == 0
        assert result.solver == "hook"
        assert result.message == "Active layer provided strictly feasible point"

    def test_lone_direct_layer_short_circuits_without_x0(self) -> None:
        """A sole ellipsoid short-circuits to its center even with no ``x0`` given."""
        layer = NormConstraintLayer(B=np.eye(2), c=np.array([1.0, 0.0]), phi=1.0)
        problem = Problem(ineq=[layer])
        result = BarrierSolver()._solve_feasibility(problem, x0=None)
        solution = np.asarray(result.solution)
        # center = B^{-1} c = [1, 0]; strictly interior (zero residual).
        np.testing.assert_allclose(solution, np.array([1.0, 0.0]), atol=1e-12)
        assert problem.is_feasible(solution)
        assert result.n_iterations == 0

    def test_ellipsoid_with_background_does_not_short_circuit(self) -> None:
        """With a background layer present, the analytic center is not taken verbatim.

        The ellipsoid center (origin) violates the background bound ``w >= 0.1``, so
        the short-circuit must be skipped (``len(ineq) > 1``) and the combined
        problem solved through the ordinary Phase I to a point feasible for *both*.
        """
        ball = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=1.0)
        bound = BoundsConstraintLayer(lb=0.1, dimension=2)
        problem = Problem(ineq=[bound, ball])  # ball is the active (last) layer
        result = BarrierSolver()._solve_feasibility(problem, x0=np.zeros(2))
        solution = np.asarray(result.solution)
        assert problem.is_feasible(solution)  # inside the ball AND w >= 0.1
        assert np.all(solution >= 0.1 - 1e-8)
        # Not the ellipsoid center, which would violate the bound.
        assert not np.allclose(solution, np.zeros(2))

    def test_lone_direct_layer_short_circuits_with_equality(self) -> None:
        """A sole ellipsoid + equality short-circuits to its constrained center.

        With an equality manifold present the hook is handed ``problem.eq`` and
        returns the equality-constrained least-squares point (the closest point of
        the ball's center manifold to the constraint), not the unconstrained
        center. Here ``min ||w||^2 s.t. w1 + w2 = 2`` gives ``w = [1, 1]``, which is
        strictly inside the radius-5 ball, so it is returned in zero iterations.
        """
        ball = NormConstraintLayer(B=np.eye(2), c=np.zeros(2), phi=5.0)
        eq = EqualityConstraintLayer(A=np.array([[1.0, 1.0]]), b=np.array([2.0]))
        problem = Problem(ineq=[ball], eq=eq)
        result = BarrierSolver()._solve_feasibility(problem, x0=None)
        solution = np.asarray(result.solution)
        np.testing.assert_allclose(solution, np.array([1.0, 1.0]), atol=1e-10)
        assert problem.is_feasible(solution)  # inside the ball
        np.testing.assert_allclose(eq.A @ solution, eq.b, atol=1e-10)  # on manifold
        assert result.n_iterations == 0
        assert result.solver == "hook"


class SolveObjNoneRoutesToFeasibilityTest(unittest.TestCase):
    """``solver.solve(Problem(obj=None, ...))`` routes to the feasibility solve."""

    def test_barrier_routes_obj_none(self) -> None:
        """BarrierSolver.solve on an obj=None problem returns a feasible result."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=1.0, dimension=3)])
        result = BarrierSolver().solve(problem, x0=np.zeros(3))
        assert isinstance(result, SolveResult)
        assert result.solver == "barrier"
        assert problem.is_feasible(np.asarray(result.solution))

    def test_equality_barrier_routes_obj_none(self) -> None:
        """BarrierSolver.solve on an obj=None problem returns a feasible result."""
        M = 5
        problem = Problem(
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([float(M)])),
        )
        result = BarrierSolver().solve(problem)
        assert isinstance(result, SolveResult)
        assert result.solver == "barrier"
        assert problem.is_feasible(np.asarray(result.solution))

    def test_primal_dual_routes_obj_none(self) -> None:
        """PrimalDualSolver.solve on an obj=None problem returns a feasible result."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=1.0, dimension=3)])
        result = PrimalDualSolver().solve(problem, x0=np.zeros(3))
        assert isinstance(result, SolveResult)
        assert isinstance(result.diagnostics, PrimalDualDiagnostics)
        assert problem.is_feasible(np.asarray(result.solution))

    def test_equality_primal_dual_routes_obj_none(self) -> None:
        """PrimalDualSolver.solve on an obj=None equality problem returns a feasible result."""
        M = 5
        problem = Problem(
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([float(M)])),
        )
        result = PrimalDualSolver().solve(problem, x0=np.ones(M))
        assert isinstance(result, SolveResult)
        assert isinstance(result.diagnostics, PrimalDualDiagnostics)
        assert problem.is_feasible(np.asarray(result.solution))
        # The solution must also satisfy A x = b (here sum(x) = M); is_feasible
        # only checks the inequality layers.
        np.testing.assert_allclose(
            float(np.asarray(result.solution).sum()), float(M), atol=1e-6
        )
        # x0 is strictly feasible (on the manifold A x = b, all bounds slack), so
        # the feasibility solve short-circuits and _feasible_result repackages it
        # without a Phase I -- verifying the equality-variant lift end-to-end.
        assert result.n_iterations == 0

    def test_equality_obj_none_off_manifold_x0_is_not_short_circuited(self) -> None:
        """An inequality-feasible but off-manifold ``x0`` must not short-circuit.

        ``Problem.is_feasible`` ignores equality, so the feasibility short-circuit
        must additionally require ``A x = b``; otherwise an off-manifold ``x0``
        would be returned verbatim, violating the equality constraint.
        """
        M = 3
        problem = Problem(
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([float(M)])),
        )
        # Strictly inequality-feasible (all > 0) but off the manifold: sum 4, not M=3.
        x0 = np.array([1.0, 1.0, 2.0])
        result = PrimalDualSolver().solve(problem, x0=x0)
        solution = np.asarray(result.solution)
        assert problem.is_feasible(solution)
        # The returned solution must satisfy A x = b (here sum(x) = M), not just
        # the inequalities.
        np.testing.assert_allclose(float(solution.sum()), float(M), atol=1e-6)

    def test_barrier_already_feasible_returns_rich_result(self) -> None:
        """An already-feasible obj=None start is repackaged as the rich result."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)])
        x0 = np.array([1.0, 2.0, 3.0])  # already strictly feasible
        result = BarrierSolver().solve(problem, x0=x0)
        assert isinstance(result, SolveResult)
        assert result.solver == "barrier"
        assert result.n_iterations == 0
        np.testing.assert_allclose(np.asarray(result.solution), x0)

    def test_primal_dual_already_feasible_returns_pd_result(self) -> None:
        """An already-feasible obj=None PD start repackages into a PrimalDualResult."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)])
        x0 = np.array([1.0, 2.0, 3.0])  # already strictly feasible
        result = PrimalDualSolver().solve(problem, x0=x0)
        assert isinstance(result, SolveResult)
        assert isinstance(result.diagnostics, PrimalDualDiagnostics)
        assert result.n_iterations == 0
        np.testing.assert_allclose(np.asarray(result.solution), x0)
