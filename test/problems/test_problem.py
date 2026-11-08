"""Test the Problem composition container."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import scipy.sparse
from parameterized import parameterized

from cvxium._typing import none_throws
from cvxium.linalg.systems import (
    BandedSystem,
    DenseSystem,
    DiagonalSystem,
    MatrixSystem,
    SparseSaddleFactory,
    ZeroSystem,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    NegatedObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import (
    OnInfeasible,
    Problem,
    Sense,
)
from cvxium.problems.quadratic_constraints import (
    SquaredNormConstraintLayer,
)
from cvxium.problems.result import (
    ObjectiveOutcome,
    SolveResult,
    SolveStatus,
)
from cvxium.solvers.solver import Solver


def _barrier_lmbda(
    problem: Problem, x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Barrier per-constraint weights ``lmbda = -1/f`` stacked over the layers.

    The central-path weighting that makes ``build_hessian_system`` reproduce the
    barrier Hessian: each constraint value ``f_i = layer.constraints(x)_i`` (< 0 in
    the interior) maps to ``-1/f_i``, stacked in constraint order.
    """
    if not problem.ineq:
        return np.zeros(0)
    f = np.concatenate([layer.constraints(x) for layer in problem.ineq])
    return np.asarray(-1.0 / f)


class ProblemConstructionTest(unittest.TestCase):
    """obj/eq/ineq construction."""

    def test_fields_are_set(self) -> None:
        """obj/eq/ineq populate the corresponding fields."""
        obj = LinearObjective(c=np.zeros(5))
        layer = EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0]))
        bounds = BoundsConstraintLayer(lb=0.0, dimension=5)
        problem = Problem(obj=obj, eq=layer, ineq=[bounds])
        assert problem.obj is obj
        assert problem.eq is layer
        assert problem.ineq == [bounds]

    def test_defaults(self) -> None:
        """ineq defaults to empty; eq defaults to None."""
        problem = Problem(obj=LinearObjective(c=np.zeros(3)))
        assert problem.ineq == []
        assert problem.eq is None

    def test_obj_defaults_to_none_for_a_feasibility_problem(self) -> None:
        """Omitting ``obj`` yields a feasibility problem when a constraint is present."""
        bounds = BoundsConstraintLayer(lb=0.0, dimension=3)
        problem = Problem(ineq=[bounds])
        assert problem.obj is None
        assert problem.ineq == [bounds]

    def test_feasibility_problem_requires_a_constraint(self) -> None:
        """``obj=None`` with neither eq nor ineq has nothing to satisfy and raises."""
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            Problem()


class ProblemBuilderTest(unittest.TestCase):
    """The fluent builder: ``Problem.minimize`` and ``Problem.subject_to``."""

    def test_minimize_builds_unconstrained(self) -> None:
        """``minimize`` yields an unconstrained problem carrying the objective."""
        obj = LinearObjective(c=np.zeros(5))
        problem = Problem.minimize(obj)
        assert problem.obj is obj
        assert problem.eq is None
        assert problem.ineq == []
        assert problem.dimension == 5

    def test_class_subject_to_eq_only(self) -> None:
        """Class-level ``subject_to`` with eq only builds an eq feasibility problem."""
        eq = EqualityConstraintLayer(A=np.ones((1, 4)), b=np.array([1.0]))
        problem = Problem.subject_to(eq=eq)
        assert problem.obj is None
        assert problem.eq is eq
        assert problem.ineq == []
        assert problem.dimension == 4

    def test_class_subject_to_ineq_only(self) -> None:
        """Class-level ``subject_to`` with ineq only builds an ineq feasibility problem."""
        bounds = BoundsConstraintLayer(lb=0.0, dimension=6)
        problem = Problem.subject_to(ineq=[bounds])
        assert problem.obj is None
        assert problem.eq is None
        assert problem.ineq == [bounds]
        assert problem.dimension == 6

    def test_class_subject_to_eq_and_ineq(self) -> None:
        """Class-level ``subject_to`` with both builds a feasibility problem with both."""
        eq = EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0]))
        bounds = BoundsConstraintLayer(lb=0.0, dimension=5)
        problem = Problem.subject_to(eq=eq, ineq=[bounds])
        assert problem.obj is None
        assert problem.eq is eq
        assert problem.ineq == [bounds]
        assert problem.dimension == 5

    def test_minimize_then_subject_to_carries_objective(self) -> None:
        """``minimize(obj).subject_to(...)`` carries the objective plus the constraints."""
        obj = LinearObjective(c=np.zeros(5))
        eq = EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0]))
        bounds = BoundsConstraintLayer(lb=0.0, dimension=5)
        problem = Problem.minimize(obj).subject_to(eq=eq, ineq=[bounds])
        assert problem.obj is obj
        assert problem.eq is eq
        assert problem.ineq == [bounds]
        assert problem.dimension == 5

    def test_minimize_then_empty_subject_to_stays_unconstrained(self) -> None:
        """``minimize(obj).subject_to()`` with no constraints stays unconstrained."""
        obj = LinearObjective(c=np.zeros(3))
        problem = Problem.minimize(obj).subject_to()
        assert problem.obj is obj
        assert problem.eq is None
        assert problem.ineq == []
        assert problem.dimension == 3

    def test_subject_to_is_immutable(self) -> None:
        """Chaining returns a NEW object; the original ``minimize`` result is unchanged."""
        obj = LinearObjective(c=np.zeros(5))
        base = Problem.minimize(obj)
        bounds = BoundsConstraintLayer(lb=0.0, dimension=5)
        chained = base.subject_to(ineq=[bounds])
        assert chained is not base
        # The original is untouched (replace, not in-place accumulate).
        assert base.ineq == []
        assert base.eq is None
        assert chained.ineq == [bounds]

    def test_subject_to_replace_semantics(self) -> None:
        """``subject_to`` replaces (does not accumulate) the receiver's constraints."""
        obj = LinearObjective(c=np.zeros(5))
        first = BoundsConstraintLayer(lb=0.0, dimension=5)
        second = BoundsConstraintLayer(ub=1.0, dimension=5)
        problem = (
            Problem.minimize(obj).subject_to(ineq=[first]).subject_to(ineq=[second])
        )
        assert problem.ineq == [second]

    def test_builder_dimension_mismatch_raises(self) -> None:
        """A builder chain whose objective and constraints disagree on M raises."""
        obj = LinearObjective(c=np.zeros(5))
        bounds = BoundsConstraintLayer(lb=0.0, dimension=4)
        with self.assertRaisesRegex(ValueError, "disagree on dimension"):
            Problem.minimize(obj).subject_to(ineq=[bounds])

    def test_class_subject_to_no_constraints_raises(self) -> None:
        """``Problem.subject_to()`` with nothing has nothing to satisfy and raises."""
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            Problem.subject_to()

    def test_builder_is_subclass_aware(self) -> None:
        """The builder returns the receiver's subclass, not a plain ``Problem``.

        ``minimize`` already uses ``cls(...)``; ``subject_to`` must likewise build
        the ``owner`` class so a ``SubProblem`` chain returns ``SubProblem``
        instances rather than collapsing to the base ``Problem``.
        """

        class _SubProblem(Problem):
            pass

        obj = LinearObjective(c=np.zeros(5))
        eq = EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0]))

        # minimize(...).subject_to(...) preserves the subclass throughout.
        chained = _SubProblem.minimize(obj).subject_to(eq=eq)
        assert isinstance(chained, _SubProblem)

        # Class-level subject_to on the subclass also returns the subclass.
        feasibility = _SubProblem.subject_to(eq=eq)
        assert isinstance(feasibility, _SubProblem)


class _CannedSolver(Solver):
    """A `Solver` that ignores the problem and returns a fixed objective outcome.

    Isolates ``Problem.solve``'s sense handling: the outcome it hands back
    describes a ``-g`` minimization, so a maximization must flip it while a
    minimization must pass it through untouched.
    """

    def __init__(self, outcome: ObjectiveOutcome | None) -> None:
        super().__init__()
        self._outcome = outcome

    def solve(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        return SolveResult(
            solution=np.arange(float(problem.dimension)),
            status=SolveStatus.OPTIMAL,
            message="canned",
            solver="canned",
            n_iterations=1,
            objective=self._outcome,
        )


class ProblemMaximizeTest(unittest.TestCase):
    """``Problem.maximize`` and the sense-aware reporting in ``Problem.solve``."""

    def test_maximize_wraps_in_negated_objective_and_sets_sense(self) -> None:
        """``maximize(g)`` carries ``NegatedObjective(g)`` with the MAXIMIZE sense."""
        g = QuadraticObjective(
            c=np.zeros(4), hessian_contribution=DenseSystem(np.eye(4))
        )
        problem = Problem.maximize(g)
        assert problem.sense is Sense.MAXIMIZE
        assert isinstance(problem.obj, NegatedObjective)
        assert problem.obj.objective is g
        assert problem.dimension == 4

    def test_minimize_sense_defaults_to_minimize(self) -> None:
        """``minimize`` and the bare constructor keep the MINIMIZE default."""
        g = LinearObjective(c=np.zeros(3))
        assert Problem.minimize(g).sense is Sense.MINIMIZE
        assert Problem(obj=g).sense is Sense.MINIMIZE

    def test_maximize_sense_requires_a_negated_objective(self) -> None:
        """``Sense.MAXIMIZE`` on a bare (un-negated) objective is rejected."""
        g = LinearObjective(c=np.zeros(3))
        with self.assertRaisesRegex(ValueError, "NegatedObjective"):
            Problem(obj=g, sense=Sense.MAXIMIZE)
        # A feasibility problem (obj=None) carrying a leftover MAXIMIZE sense is
        # accepted: barrier/PD Phase I builds one via replace(problem, obj=None),
        # and with no objective there is no value for solve to sign-flip.
        feasibility = Problem(
            obj=None,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            sense=Sense.MAXIMIZE,
        )
        assert feasibility.obj is None
        # The wrapper form -- exactly what ``maximize`` builds -- is accepted.
        problem = Problem(obj=NegatedObjective(g), sense=Sense.MAXIMIZE)
        assert problem.sense is Sense.MAXIMIZE

    def test_maximize_then_subject_to_preserves_sense(self) -> None:
        """Chaining constraints onto a maximization keeps the sense and the wrapper."""
        g = LinearObjective(c=np.zeros(5))
        eq = EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0]))
        bounds = BoundsConstraintLayer(lb=0.0, dimension=5)
        problem = Problem.maximize(g).subject_to(eq=eq, ineq=[bounds])
        assert problem.sense is Sense.MAXIMIZE
        assert isinstance(problem.obj, NegatedObjective)
        assert problem.obj.objective is g
        assert problem.eq is eq
        assert problem.ineq == [bounds]

    def test_solve_reports_the_maximization_value_and_dual_bound(self) -> None:
        """A maximization flips the ``-g`` outcome back: value and dual negate."""
        g = LinearObjective(c=np.zeros(2))
        solver = _CannedSolver(
            ObjectiveOutcome(value=-4.0, suboptimality=1e-9, dual_value=-4.5)
        )
        result = Problem.maximize(g).solve(solver=solver)
        self.assertAlmostEqual(result.objective_value, 4.0)  # g(w*)
        self.assertAlmostEqual(result.dual_value, 4.5)  # upper bound on g

    def test_solve_minimization_value_passes_through_unchanged(self) -> None:
        """A minimization is not flipped; the raw solver outcome is reported."""
        g = LinearObjective(c=np.zeros(2))
        solver = _CannedSolver(
            ObjectiveOutcome(value=-4.0, suboptimality=1e-9, dual_value=-4.5)
        )
        result = Problem.minimize(g).solve(solver=solver)
        self.assertAlmostEqual(result.objective_value, -4.0)
        self.assertAlmostEqual(result.dual_value, -4.5)

    def test_solve_maximization_leaves_the_solution_point_untouched(self) -> None:
        """The sense flip touches only the reported value, never ``solution``."""
        g = LinearObjective(c=np.zeros(3))
        solver = _CannedSolver(ObjectiveOutcome(value=-1.0, suboptimality=0.0))
        result = Problem.maximize(g).solve(solver=solver)
        np.testing.assert_array_equal(result.solution, np.arange(3.0))

    def test_solve_maximization_with_none_objective_passes_through(self) -> None:
        """A MAXIMIZE solve whose solver reports no objective is left untouched."""
        # ``Problem.solve`` guards ``result.objective is not None`` before negating
        # in the MAXIMIZE branch; a solver that returns no outcome (objective=None)
        # must pass straight through rather than crash on the sense flip.
        g = LinearObjective(c=np.zeros(2))
        solver = _CannedSolver(None)
        result = Problem.maximize(g).solve(solver=solver)
        assert result.objective is None


class ProblemDimensionIntegrityTest(unittest.TestCase):
    """``__post_init__`` cross-checks that the components agree on the dimension."""

    def test_consistent_components_construct(self) -> None:
        """Components that all share a common dimension construct without error."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(5)),
            eq=EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0])),
            ineq=[LinearInequalityConstraintLayer(A=np.ones((2, 5)), cu=np.zeros(2))],
        )
        assert problem.dimension == 5

    def test_objective_equality_mismatch_raises(self) -> None:
        """An objective and equality that disagree on dimension raise."""
        with self.assertRaisesRegex(ValueError, "disagree on dimension"):
            Problem(
                obj=LinearObjective(c=np.zeros(5)),
                eq=EqualityConstraintLayer(A=np.ones((1, 4)), b=np.array([1.0])),
            )

    def test_objective_inequality_mismatch_raises(self) -> None:
        """An objective and an inequality layer that disagree raise, naming both."""
        with self.assertRaisesRegex(
            ValueError,
            "objective has 5, but the inequality layer 0 has 4",
        ):
            Problem(
                obj=LinearObjective(c=np.zeros(5)),
                ineq=[
                    LinearInequalityConstraintLayer(A=np.ones((2, 4)), cu=np.zeros(2))
                ],
            )

    def test_equality_inequality_mismatch_raises(self) -> None:
        """An equality and an inequality layer that disagree raise."""
        with self.assertRaisesRegex(ValueError, "disagree on dimension"):
            Problem(
                eq=EqualityConstraintLayer(A=np.ones((1, 5)), b=np.array([1.0])),
                ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            )

    def test_inequality_layers_mismatch_raises(self) -> None:
        """Two inequality layers that disagree with each other raise."""
        with self.assertRaisesRegex(ValueError, "disagree on dimension"):
            Problem(
                ineq=[
                    BoundsConstraintLayer(lb=0.0, dimension=5),
                    BoundsConstraintLayer(lb=0.0, dimension=3),
                ]
            )

    @parameterized.expand([(3,), (7,), (20,)])
    def test_scalar_bounds_layer_pins_explicit_dimension(self, M: int) -> None:
        """A scalar-bounds layer pins the ``dimension`` it was constructed with."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)])
        assert problem.dimension == M

    def test_vector_bounds_layer_infers_dimension(self) -> None:
        """A vector-bounds layer infers its ``dimension`` from the array length."""
        M = 4
        problem = Problem(ineq=[BoundsConstraintLayer(lb=np.zeros(M))])
        assert problem.dimension == M
        assert problem.ineq[0].dimension == M

    def test_zero_dimension_component_raises(self) -> None:
        """A component whose dimension is zero raises rather than constructing.

        A degenerate zero-size component would otherwise construct and fail
        later with a cryptic shape error deep inside a solve.
        """
        with self.assertRaisesRegex(ValueError, "dimension must be positive"):
            Problem(obj=LinearObjective(c=np.zeros(0)))

    def test_zero_dimension_equality_raises(self) -> None:
        """An equality matrix with zero columns raises at construction time."""
        with self.assertRaisesRegex(ValueError, "dimension must be positive"):
            Problem(eq=EqualityConstraintLayer(A=np.zeros((0, 0)), b=np.zeros(0)))


class ProblemDimensionDerivationTest(unittest.TestCase):
    """``dimension`` is a derived read-only property reading from the components."""

    @parameterized.expand([(1,), (5,), (50,)])
    def test_objective_only_unconstrained(self, M: int) -> None:
        """An unconstrained problem derives its dimension from the objective."""
        problem = Problem(obj=LinearObjective(c=np.zeros(M)))
        assert problem.dimension == M

    @parameterized.expand([(2,), (4,), (9,)])
    def test_equality_only_feasibility(self, M: int) -> None:
        """An equality-only feasibility problem derives its dimension from the eq."""
        problem = Problem(
            eq=EqualityConstraintLayer(A=np.ones((1, M)), b=np.array([1.0]))
        )
        assert problem.dimension == M

    @parameterized.expand([(2,), (4,), (9,)])
    def test_inequality_only_feasibility(self, M: int) -> None:
        """An inequality-only feasibility problem derives its dimension from ineq[0]."""
        problem = Problem(ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)])
        assert problem.dimension == M

    def test_constrained_problem(self) -> None:
        """A fully constrained problem derives a single consistent dimension."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(6)),
            eq=EqualityConstraintLayer(A=np.ones((1, 6)), b=np.array([1.0])),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=6)],
        )
        assert problem.dimension == 6


class ProblemTest(unittest.TestCase):
    """Tests for `Problem`."""

    def test_default_constraints_is_empty_list(self) -> None:
        """`constraints` defaults to an empty list."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
        )
        assert problem.ineq == []
        assert problem.eq is None

    def test_solve_delegates_to_solver(self) -> None:
        """`problem.solve(solver=...)` forwards to ``solver.solve(problem, x0)``."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(2)),
        )
        solver = MagicMock()
        x0 = np.array([0.5, 0.5])

        result = problem.solve(solver=solver, x0=x0)

        solver.solve.assert_called_once_with(problem, x0, on_infeasible="raise")
        assert result is solver.solve.return_value

    def test_solve_defaults_x0_to_none(self) -> None:
        """`problem.solve(solver=...)` forwards ``x0=None`` when omitted."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(2)),
        )
        solver = MagicMock()

        problem.solve(solver=solver)

        solver.solve.assert_called_once_with(problem, None, on_infeasible="raise")

    def test_solve_forwards_on_infeasible(self) -> None:
        """`problem.solve(..., on_infeasible=...)` forwards the policy to the solver."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(2)),
        )
        solver = MagicMock()
        x0 = np.array([0.5, 0.5])

        problem.solve(solver=solver, x0=x0, on_infeasible="continue")

        solver.solve.assert_called_once_with(problem, x0, on_infeasible="continue")

    def test_is_feasible_with_no_constraints(self) -> None:
        """Empty constraint list is vacuously feasible."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
        )
        assert problem.is_feasible(np.array([1.0, 2.0, 3.0])) is True

    def test_is_feasible_with_satisfied_layer(self) -> None:
        """`is_feasible` returns True when each layer is strictly satisfied."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        assert problem.is_feasible(np.array([1.0, 2.0, 3.0])) is True

    def test_is_feasible_with_violated_layer(self) -> None:
        """`is_feasible` returns False when any layer is violated."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[BoundsConstraintLayer(lb=1.0, dimension=3)],
        )
        # 0.5 < 1.0 violates the lower-bound constraint.
        assert problem.is_feasible(np.array([0.5, 2.0, 3.0])) is False

    def test_num_inequality_constraints_empty(self) -> None:
        """No layers means no inequality constraints."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
        )
        assert problem.num_inequality_constraints() == 0

    def test_num_inequality_constraints_sums_layers(self) -> None:
        """Counts are summed across layers using the problem's dimension."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(5)),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=5),  # 5 lower bounds
                BoundsConstraintLayer(ub=1.0, dimension=5),  # 5 upper bounds
            ],
        )
        assert problem.num_inequality_constraints() == 10


class ProblemMakePhase1Test(unittest.TestCase):
    """Tests for `Problem.make_phase1` -- Phase I dispatch."""

    def test_no_constraints_raises(self) -> None:
        """`make_phase1` raises when there are no inequality constraints."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
        )
        with self.assertRaisesRegex(ValueError, "at least one inequality"):
            problem.make_phase1(x0=np.zeros(3))

    def test_direct_form_when_active_is_direct(self) -> None:
        """A `DirectConstraintLayer` active produces direct-form Phase I."""
        active = SquaredNormConstraintLayer(phi=5.0, dimension=3)
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[active],
        )
        phase1, x0_phase1 = problem.make_phase1(x0=np.array([1.0, 1.0, 1.0]))

        # Direct form: same dimension, the active layer itself becomes the
        # objective (DirectConstraintLayer is also an Objective), no constraint
        # layers.
        assert phase1.dimension == 3
        assert phase1.obj is active
        assert phase1.ineq == []
        # Same x0.
        np.testing.assert_array_equal(x0_phase1, np.array([1.0, 1.0, 1.0]))

    def test_direct_form_preserves_background_layers(self) -> None:
        """Background layers are preserved; the active is hoisted to the objective."""
        bg = BoundsConstraintLayer(lb=0.0, dimension=3)
        active = SquaredNormConstraintLayer(phi=5.0, dimension=3)
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[bg, active],
        )
        phase1, _ = problem.make_phase1(x0=np.array([1.0, 1.0, 1.0]))

        assert phase1.obj is active
        assert phase1.ineq == [bg]

    def test_slack_form_when_active_is_slack(self) -> None:
        """A `SlackConstraintLayer` active produces slack-form Phase I."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        x0 = np.array([1.0, 1.0, 1.0])
        phase1, x0_phase1 = problem.make_phase1(x0=x0)

        # Slack form: dimension augmented to M+1.
        assert phase1.dimension == 4
        # Objective is LinearObjective(c=[0,0,0,1]).
        assert isinstance(phase1.obj, LinearObjective)
        np.testing.assert_array_equal(phase1.obj.c, np.array([0.0, 0.0, 0.0, 1.0]))
        # x0 augmented with initial slack at index M=3.
        assert x0_phase1.shape == (4,)
        np.testing.assert_array_equal(x0_phase1[:3], x0)

    def test_slack_form_constraint_structure(self) -> None:
        """Slack-form constraints: lifted background + slack upper + slack active.

        Verifies the role of each constraint by probing its output on an augmented
        ``x_aug = [x; s]`` rather than asserting against private wrapper types.

        """
        bg = BoundsConstraintLayer(lb=0.0, dimension=3)
        active = BoundsConstraintLayer(ub=10.0, dimension=3)
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[bg, active],
        )
        x = np.array([5.0, 5.0, 5.0])
        phase1, _ = problem.make_phase1(x0=x)

        assert len(phase1.ineq) == 3
        # Augmented x_aug = [x, s]; pick s arbitrarily within the slack domain.
        s = 2.0
        x_aug = np.concatenate([x, [s]])

        # Constraint 0: lifted background. Its values on x_aug equal bg's values on x
        # (the lift ignores the slack component).
        np.testing.assert_array_equal(
            phase1.ineq[0].constraints(x_aug),
            bg.constraints(x),
        )

        # Constraint 1: slack upper bound. Single scalar constraint that depends only
        # on the slack component.
        assert phase1.ineq[1].num_constraints(4) == 1
        assert phase1.ineq[1].constraints(x_aug).shape == (1,)

        # Constraint 2: slack-coupled active. Its values equal active(x) - s, so the
        # active constraint relaxes by the slack.
        np.testing.assert_allclose(
            phase1.ineq[2].constraints(x_aug),
            active.constraints(x) - s,
        )

    def test_slack_form_augments_equality(self) -> None:
        """When equality is supplied, slack-form augments A with a zero column."""
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([3.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        phase1, _ = problem.make_phase1(x0=np.array([1.0, 1.0, 1.0]))

        assert phase1.eq is not None
        # Augmented A: original A then a zero column.
        assert phase1.eq.A.shape == (1, 4)
        np.testing.assert_array_equal(phase1.eq.A[:, :3], A)
        np.testing.assert_array_equal(phase1.eq.A[:, 3], np.array([0.0]))
        np.testing.assert_array_equal(phase1.eq.b, b)

    def test_slack_form_augments_sparse_equality_and_keeps_factory(self) -> None:
        """Slack-form Phase I keeps a sparse A sparse and preserves the factory."""
        A = scipy.sparse.csr_matrix(np.array([[1.0, 1.0, 1.0]]))
        b = np.array([3.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b, saddle_factory=SparseSaddleFactory()),
        )
        phase1, _ = problem.make_phase1(x0=np.array([1.0, 1.0, 1.0]))

        assert phase1.eq is not None
        # The augmented A stays sparse (no densification) and gains a zero column.
        self.assertTrue(scipy.sparse.issparse(phase1.eq.A))
        self.assertEqual(phase1.eq.A.shape, (1, 4))
        np.testing.assert_array_equal(phase1.eq.A.toarray()[:, :3], A.toarray())
        np.testing.assert_array_equal(phase1.eq.A.toarray()[:, 3], np.array([0.0]))
        # The sparse saddle factory carries into the Phase I equality layer.
        self.assertIsInstance(phase1.eq.saddle_factory, SparseSaddleFactory)

    def test_slack_form_initial_slack_satisfies_active_shifted(self) -> None:
        """The initial augmented x0 should put `s_0 > max f_i(x0)` so shifted constraints are strictly negative."""
        active = BoundsConstraintLayer(lb=0.0, dimension=3)
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[active],
        )
        x0 = np.array([0.5, 1.0, 0.1])  # all positive => active.constraints(x0) = -x0
        phase1, x0_aug = problem.make_phase1(x0=x0)

        # The slack-active layer should be feasible at x0_aug.
        slack_active = phase1.ineq[-1]
        c_active = slack_active.constraints(x0_aug)
        assert np.all(
            c_active < 0
        ), f"Active constraints not strictly feasible: {c_active}"
        # And the slack-upper-bound layer should also be feasible.
        slack_upper = phase1.ineq[-2]
        c_upper = slack_upper.constraints(x0_aug)
        assert np.all(c_upper < 0), f"Slack upper not strictly feasible: {c_upper}"


class ProblemMakePhase1ProjectionTest(unittest.TestCase):
    """``make_phase1`` projects ``x0`` onto the equality manifold ``A x = b``.

    A feasible-start barrier maintains ``A x = b`` but never corrects an
    off-manifold start, so the Phase I start must already be on the manifold.
    The projection runs before the slack/direct dispatch, so it applies to both
    forms; it is idempotent on an already-on-manifold ``x0``.
    """

    def test_slack_form_projects_off_manifold_x0(self) -> None:
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([3.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DenseSystem(np.eye(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],  # slack-form active
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = np.array([0.5, 0.5, 0.5])  # A x0 = 1.5 != 3 -> off the manifold
        assert not np.isclose((A @ x0).item(), float(b[0]))
        _, x0_aug = problem.make_phase1(x0=x0)
        # The recovered w-part lands on the manifold (the slack column is zero).
        np.testing.assert_allclose(A @ x0_aug[:3], b, atol=1e-9)

    def test_direct_form_projects_off_manifold_x0(self) -> None:
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([3.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DenseSystem(np.eye(3))
            ),
            ineq=[
                SquaredNormConstraintLayer(phi=5.0, dimension=3)
            ],  # direct-form active
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = np.array([0.5, 0.5, 0.5])
        _, x0_phase1 = problem.make_phase1(x0=x0)
        np.testing.assert_allclose(A @ x0_phase1, b, atol=1e-9)

    def test_on_manifold_x0_is_unchanged(self) -> None:
        """Projection is idempotent: an on-manifold x0 is returned as-is."""
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([3.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DenseSystem(np.eye(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = np.array([1.0, 1.0, 1.0])  # A x0 = 3 = b -> on the manifold
        _, x0_aug = problem.make_phase1(x0=x0)
        np.testing.assert_allclose(x0_aug[:3], x0, atol=1e-9)

    def test_no_equality_leaves_x0_untouched(self) -> None:
        """Without equality there is no manifold to project onto."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DenseSystem(np.eye(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        x0 = np.array([0.5, 0.5, 0.5])
        _, x0_aug = problem.make_phase1(x0=x0)
        np.testing.assert_array_equal(x0_aug[:3], x0)


class _BandedBounds(BoundsConstraintLayer):
    """Test-only layer that returns a (non-absorbable) banded system."""

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        return BandedSystem(np.ones((2, len(w))))


class BuildHessianSystemTest(unittest.TestCase):
    """Tests for Problem.build_hessian_system: assembled Hessian + structural gating."""

    def test_build_hessian_system_diagonal_only_matches_dense(self) -> None:
        r"""Diagonal objective + bounds assemble to `t Q + diag(1/(x-lb)^2)`."""
        eta = np.array([2.0, 3.0, 5.0, 7.0])
        Q = np.diag(eta)
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(4), hessian_contribution=DiagonalSystem(eta)
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=4)],
        )
        x = np.array([1.0, 2.0, 0.5, 4.0])  # strictly feasible (x > lb = 0)
        t = 1.5

        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), t, _barrier_lmbda(problem, x)
        )

        H_dense = t * Q + np.diag(1.0 / np.square(x))
        b = np.array([1.0, -2.0, 3.0, 0.5])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-10)

    def test_build_hessian_system_with_low_rank_matches_dense(self) -> None:
        """A low-rank layer's rank-1 update lands via the universal wrapper.

        ``SquaredNormConstraintLayer`` contributes a ``DiagonalSystem`` carrying
        one low-rank update; the assembled system must equal the full dense barrier
        Hessian including the rank-1 outer product.
        """
        rng = np.random.default_rng(7)
        diag_q = rng.uniform(1.0, 4.0, size=5)
        Q = np.diag(diag_q)
        phi = 100.0
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(5), hessian_contribution=DiagonalSystem(diag_q)
            ),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=5),
                SquaredNormConstraintLayer(phi=phi, dimension=5),
            ],
        )
        x = np.array([1.0, 2.0, 3.0, 1.5, 0.5])  # x > 0 and ||x||^2 < phi
        t = 2.0

        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), t, _barrier_lmbda(problem, x)
        )

        # Dense ground truth: t Q + diag(1/x^2) + (-2/c) I + (4/c^2) x x^T.
        c = float(np.dot(x, x)) - phi  # < 0
        H_dense = (
            t * Q
            + np.diag(1.0 / np.square(x))
            + (-2.0 / c) * np.eye(5)
            + (4.0 / (c * c)) * np.outer(x, x)
        )
        b = rng.standard_normal(5)
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-9
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-9)

    def test_caller_declared_low_rank_q(self) -> None:
        r"""A low-rank `Q = kappa kappa^T` declared as `ZeroSystem(...).low_rank_update(...)`.

        Demonstrates that the caller declares the structure of ``Q`` -- here a
        rank-2 ``Q`` carried as a universal low-rank update on a zero base -- and
        ``Problem`` assembles it correctly (the objective's ``t`` scaling
        folds into the update via the unit-weight sqrt path).
        """
        rng = np.random.default_rng(11)
        kappa = rng.standard_normal((4, 2))
        Q = kappa @ kappa.T  # rank-2, PSD
        contribution = ZeroSystem(4).low_rank_update(kappa)
        objective = QuadraticObjective(c=np.zeros(4), hessian_contribution=contribution)
        problem = Problem(
            obj=objective,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=4)],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0])
        t = 2.0

        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), t, _barrier_lmbda(problem, x)
        )

        H_dense = t * Q + np.diag(1.0 / np.square(x))
        b = rng.standard_normal(4)
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-9
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-9)

    def test_build_hessian_system_pd_weights_match_dense(self) -> None:
        r"""Primal-dual weighting: ``objective_scale=1`` with arbitrary dual ``lmbda``.

        The same assembler builds the condensed primal-dual Hessian `\nabla^2 f_0 +
        \sum_i \lambda_i C_i` when handed the dual variables instead of the barrier
        ``-1/f``. For a diagonal QP under lower bounds (linear, so `C_i` is diagonal),
        coordinate `i` gets `q_i + \lambda_i / x_i`.
        """
        q = np.array([2.0, 3.0, 5.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(q)
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        x = np.array([1.0, 2.0, 4.0])  # strictly feasible (x > 0)
        lmbda = np.array([0.5, 1.5, 0.7])  # arbitrary positive duals (not -1/f)

        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, lmbda
        )

        # f = -x (lb = 0); C_i diagonal = -1/f_i = 1/x_i. H_pd = diag(q + lmbda/x).
        H_dense = np.diag(q + lmbda / x)
        b = np.array([1.0, -1.0, 2.0])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-12
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-12)

    def test_build_hessian_system_pd_weights_low_rank(self) -> None:
        r"""Primal-dual weighting with a rank-1 layer (squared norm).

        `H_{pd} = \nabla^2 f_0 + \mathrm{diag}(\lambda_b / x) + \lambda_s (2 I -
        (4/c) x x^T)`, exercising both the diagonal and the low-rank channels under
        arbitrary dual weights.
        """
        rng = np.random.default_rng(23)
        q = rng.uniform(1.0, 4.0, size=4)
        phi = 100.0
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(4), hessian_contribution=DiagonalSystem(q)
            ),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=4),
                SquaredNormConstraintLayer(phi=phi, dimension=4),
            ],
        )
        x = np.array([1.0, 2.0, 3.0, 1.5])  # x > 0 and ||x||^2 < phi
        lam_bounds = rng.uniform(0.2, 2.0, size=4)
        lam_sn = 0.8
        lmbda = np.concatenate([lam_bounds, np.array([lam_sn])])

        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, lmbda
        )

        c = float(np.dot(x, x)) - phi  # < 0
        H_dense = (
            np.diag(q + lam_bounds / x)
            + lam_sn * 2.0 * np.eye(4)
            - (4.0 * lam_sn / c) * np.outer(x, x)
        )
        b = rng.standard_normal(4)
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-9
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-9)

    def test_banded_objective_assembles_with_diagonal_layer(self) -> None:
        """A banded objective + a diagonal bounds layer assemble to a banded sum.

        The barrier diagonal folds into the band as a width-0 band, so the
        assembled Hessian stays banded rather than densifying -- the generality
        proof that a non-diagonal objective curvature composes with the barrier.
        """
        H_band = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
        ab = np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]])  # upper compact storage
        objective = QuadraticObjective(
            c=np.zeros(3), hessian_contribution=BandedSystem(ab)
        )
        problem = Problem(
            obj=objective,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        x = np.array([1.0, 2.0, 3.0])  # strictly feasible (x > lb = 0)
        t = 1.5

        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), t, _barrier_lmbda(problem, x)
        )

        H_dense = t * H_band + np.diag(1.0 / np.square(x))
        b = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-10)

    def test_linear_objective_contributes_zero(self) -> None:
        """A `LinearObjective` (zero Hessian) assembles purely from the layers."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        x = np.array([1.0, 2.0, 4.0])
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, _barrier_lmbda(problem, x)
        )
        # Only the bounds barrier diagonal 1/x^2 remains.
        H_dense = np.diag(1.0 / np.square(x))
        b = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )

    def test_assembly_uses_stable_eta_inverse_single_source(self) -> None:
        r"""The bounds layer's stable `eta_inverse = slack / lmbda` flows through assembly.

        With a ``LinearObjective`` (contributes ``ZeroSystem``), the only diagonal
        source is the bounds layer, which supplies a stable ``eta_inverse =
        slack / lmbda`` (`= slack^2` at the barrier weight ``lmbda = 1/slack``).
        Near the boundary this stays accurate where recomputing ``1/eta`` would
        drift.
        """
        s = np.array([1e-7, 2e-7, 3e-7])  # tiny slacks
        x = s  # lb = 0, so slack = x - 0 = s
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, _barrier_lmbda(problem, x)
        )
        # H = diag(1/s^2); H^{-1} = diag(s^2). The stable eta_inverse = s/lmbda
        # reproduces s^2 to machine precision (lmbda = -1/f = 1/s at the barrier).
        np.testing.assert_allclose(system.solve(np.ones(3)), s * s, rtol=1e-12)

    def test_assembly_combines_eta_inverse_across_sources(self) -> None:
        r"""Two diagonal sources both supplying `eta_inverse` combine correctly.

        Objective (diagonal Q, declaring its reciprocal) plus a bounds layer: both
        contribute a diagonal with ``eta_inverse``, so assembly takes the
        factor-out path. The result must still equal the true `1/(t Q + 1/s^2)`.
        """
        q = np.array([2.0, 3.0, 5.0])
        objective = QuadraticObjective(
            c=np.zeros(3),
            hessian_contribution=DiagonalSystem(q, eta_inverse=1.0 / q),
        )
        problem = Problem(
            obj=objective,
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        x = np.array([1.0, 2.0, 0.5])
        t = 1.5
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), t, _barrier_lmbda(problem, x)
        )
        H_dense = np.diag(t * q + 1.0 / np.square(x))
        b = np.array([1.0, -1.0, 2.0])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-12
        )

    def test_banded_layer_assembles_with_diagonal_objective(self) -> None:
        """A banded layer combines with a diagonal objective in either order.

        ``plus`` is commutative, so the diagonal objective (reduced first) and
        the banded layer assemble into a banded sum rather than raising -- the
        objective's diagonal folds into the band as a width-0 band.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3),
                hessian_contribution=DiagonalSystem(np.ones(3)),
            ),
            ineq=[_BandedBounds(lb=0.0, dimension=3)],
        )
        x = np.array([1.0, 2.0, 3.0])
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, _barrier_lmbda(problem, x)
        )
        # diag(1) [objective, t=1] + the layer's banded ones matrix.
        H_dense = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        b = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-10)

    def test_diagonal_objective_and_layers_assemble(self) -> None:
        """Zero objective + diagonal/low-rank layers assemble and solve."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=3),
                SquaredNormConstraintLayer(phi=10.0, dimension=3),
            ],
        )
        x = np.array([1.0, 1.0, 1.0])  # x > 0 and ||x||^2 = 3 < phi
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, _barrier_lmbda(problem, x)
        )
        # Densify the assembled system via multiply, then confirm solve inverts it.
        H_dense = np.column_stack([system.multiply(e) for e in np.eye(3)])
        b = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )
