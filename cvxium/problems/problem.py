"""Problem statement for Cvxium: an objective, inequality layers, and equalities.

Defines ``Problem``, the passive statement of a convex optimization problem
   minimize    f0(w)
   subject to  A w = b,
               f_i(w) ≤ 0 for each inequality layer,
kept separate from the ``Solver`` that solves it. A ``Problem`` is built fluently with
``Problem.minimize``, ``Problem.maximize`` and ``subject_to``, and it carries the two
pieces of assembly the solvers share: the weighted Newton-system Hessian
(``build_hessian_system``) and the Phase I problem that finds a strictly feasible start
(``make_phase1``).

``Sense`` records whether the problem minimizes or maximizes; ``OnInfeasible`` names the
policy a feasibility solve follows on an infeasibility certificate.
``_SubjectToDescriptor`` lets ``subject_to`` work on both the class and an instance.

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, overload

import numpy as np
import numpy.typing as npt
import scipy.sparse

from ..problems.constraint_layers import (
    ConstraintLayer,
    DirectConstraintLayer,
    SlackConstraintLayer,
)
from ..problems.linear_constraints import (
    EqualityConstraintLayer,
)
from ..problems.objective import (
    LinearObjective,
    NegatedObjective,
    Objective,
    ObjectivePoint,
)
from ..solvers.phase1 import (
    LiftedConstraintLayer,
    _SlackActiveLayer,
    _SlackUpperBoundLayer,
)

if TYPE_CHECKING:
    from ..linalg.systems import MatrixSystem
    from ..problems.result import SolveResult
    from ..solvers.solver import Solver


class _SubjectToCallable(Protocol):
    """Callable returned by the ``subject_to`` descriptor.

    A ``typing.Protocol`` rather than a bare ``Callable[...]``, so that ``eq`` and
    ``ineq`` keep their names: ``Callable`` erases parameter names into positional-only
    slots, which would reject the keyword forms (``subject_to(eq=..., ineq=[...])``)
    under the type checker.

    """

    def __call__(
        self,
        eq: EqualityConstraintLayer | None = None,
        ineq: list[ConstraintLayer] | None = None,
    ) -> Problem:
        """Build a new ``Problem`` with the given constraints.

        Parameters
        ----------
         eq : EqualityConstraintLayer | None
            Linear equality constraint A w = b, or ``None`` for none.
         ineq : list[ConstraintLayer] | None
            Inequality-constraint layers; ``None`` means none.

        Returns
        -------
         problem : Problem
            A new frozen ``Problem`` carrying the constraints.

        """
        ...


class _SubjectToDescriptor:
    """Class-or-instance method descriptor backing ``Problem.subject_to``.

    Neither a plain method nor a ``classmethod`` can serve both access modes: a plain
    method requires an instance, and a ``classmethod`` discards the instance it is
    accessed on. ``Problem.subject_to`` needs both:
    - On the class (``Problem.subject_to(...)``), it builds a feasibility ``Problem``
      with ``obj=None``.
    - On an instance (``some_problem.subject_to(...)``), it builds a new ``Problem``
      that carries the instance's objective and sense.
    ``__get__`` receives ``None`` for class access and the ``Problem`` for instance
    access, and binds the matching objective into the returned builder. Both forms
    return a new frozen ``Problem``; nothing is mutated.

    The two ``__get__`` overloads type the access modes separately, so the chain
    ``Problem.minimize(obj).subject_to(...)`` type-checks.

    """

    @overload
    def __get__(self, instance: None, owner: type[Problem]) -> _SubjectToCallable: ...

    @overload
    def __get__(
        self, instance: Problem, owner: type[Problem]
    ) -> _SubjectToCallable: ...

    def __get__(
        self, instance: Problem | None, owner: type[Problem]
    ) -> _SubjectToCallable:
        """Bind the ``subject_to`` builder to the access mode.

        Class access binds no objective and the minimize sense (moot for a feasibility
        problem); instance access binds the instance's objective and sense.

        Parameters
        ----------
         instance : Problem | None
            The ``Problem`` the attribute was accessed on, or ``None`` for class access.
         owner : type[Problem]
            The class the attribute was accessed through; the builder constructs this
            class.

        Returns
        -------
         subject_to : _SubjectToCallable
            The builder, which returns a new ``Problem``.

        """
        # Class access carries no objective; instance access carries the
        # instance's objective (and its minimize/maximize sense) into the new
        # problem. Class access is a feasibility problem, whose sense is moot;
        # the minimize default keeps it well-defined.
        obj = instance.obj if instance is not None else None
        sense = instance.sense if instance is not None else Sense.MINIMIZE

        def subject_to(
            eq: EqualityConstraintLayer | None = None,
            ineq: list[ConstraintLayer] | None = None,
        ) -> Problem:
            """Attach equality and inequality constraints, returning a new ``Problem``.

            Accessed on the class, this builds a feasibility problem with ``obj=None``;
            accessed on an instance, it builds a new problem carrying that instance's
            objective and sense. Either way it returns a new frozen ``Problem`` and
            leaves the receiver unchanged.

            The semantics are replace, not accumulate: ``eq`` and ``ineq`` become the
            new problem's constraints outright, discarding any the receiver had.

            Parameters
            ----------
             eq : EqualityConstraintLayer | None
                Linear equality constraint A w = b, or ``None`` for none.
             ineq : list[ConstraintLayer] | None
                Inequality-constraint layers; ``None`` is treated as an empty list.

            Returns
            -------
             problem : Problem
                A new frozen ``Problem`` carrying the bound objective (``None`` under
                class access) and sense, together with ``eq`` and ``ineq``.

            Raises
            ------
             ValueError
                Propagated from ``Problem.__post_init__``: when the components disagree
                on the dimension, or when class access supplies no constraint at all.

            """
            return owner(
                obj=obj,
                eq=eq,
                ineq=ineq if ineq is not None else [],
                sense=sense,
            )

        return subject_to


# Policy for how a feasibility solve responds to an infeasibility certificate.
# ``"raise"`` (the default) raises ``ProblemInfeasibleError`` as soon as the
# certificate is detected -- the cheapest way to learn the problem is infeasible.
# ``"continue"`` instead drives the slack-form Phase I to its optimum (the
# minimum-imbalance point) before raising, attaching that best-effort result to
# the exception so the caller can recover it.
OnInfeasible = Literal["raise", "continue"]


class Sense(Enum):
    """Optimization sense of a ``Problem``: minimize or maximize.

    Cvxium's solvers only minimize, so a ``Problem`` minimizes by default.
    ``Problem.maximize`` records ``MAXIMIZE`` alongside a ``NegatedObjective``, and
    ``Problem.solve`` reads the sense back to report the objective value as g(w*) rather
    than the -g(w*) the solver computed. The sense changes only how the result is
    reported, never the point the solver converges to.

    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class Problem:
    """Convex optimization problem in composition form.

    A passive container holding an ``Objective`` (or ``None`` for a feasibility
    problem), an ordered list of inequality ``ConstraintLayer`` objects, and an optional
    linear equality constraint. It states what to optimize, separately from how (the
    ``Solver``):
       minimize    f0(w)
       subject to  A w = b                   (when ``eq`` is given),
                   f_i(w) ≤ 0                (for each layer in ``ineq``).
    A ``Problem`` performs no optimization itself. Besides the builders (``minimize``,
    ``maximize``, ``subject_to``) and simple queries (``dimension``, ``is_feasible``,
    ``num_inequality_constraints``), it assembles the Newton-system Hessian
    (``build_hessian_system``) and the Phase I problem (``make_phase1``) for the
    solvers. ``problem.solve(solver)`` delegates to ``solver.solve(problem)`` and
    reports the result in the problem's sense.

    The decision-variable dimension M is not stored: ``dimension`` reads it from
    whichever component is present, and ``__post_init__`` checks at construction that
    every component agrees on it.

    Fields
    ------
     obj : Objective | None
        The objective f0(w) to minimize, or ``None`` for a feasibility problem (find any
        w satisfying the constraints). A feasibility problem must carry at least one
        equality or inequality constraint, or construction raises ``ValueError``.
     eq : EqualityConstraintLayer | None
        Linear equality constraint A w = b, or ``None`` for no equalities.
     ineq : list[ConstraintLayer]
        Inequality-constraint layers, in caller-chosen order; the last is the active
        layer for Phase I. May be empty.
     sense : Sense
        Whether ``obj`` is minimized (the default) or maximized. The solvers always
        minimize, so a maximization carries a ``NegatedObjective`` and ``solve`` flips
        the reported value back. With an objective present, ``Sense.MAXIMIZE`` is valid
        only on a ``NegatedObjective``, and construction raises ``ValueError``
        otherwise, so ``Problem.maximize`` is the supported way to build a maximization.
        A feasibility problem may carry either sense harmlessly, since there is no
        objective value to report.

    """

    obj: Objective | None = None
    eq: EqualityConstraintLayer | None = None
    ineq: list[ConstraintLayer] = field(default_factory=list)
    sense: Sense = Sense.MINIMIZE

    @classmethod
    def minimize(cls, obj: Objective) -> Problem:
        """Begin a builder chain for the problem minimize obj.

        Chain ``.subject_to(eq=..., ineq=[...])`` onto the result to attach constraints
        while keeping the objective. Each builder call returns a new frozen ``Problem``.

        Parameters
        ----------
         obj : Objective
            The objective f0(w) to minimize.

        Returns
        -------
         problem : Problem
            An unconstrained ``Problem`` carrying ``obj``.

        """
        return cls(obj=obj)

    @classmethod
    def maximize(cls, obj: Objective) -> Problem:
        """Begin a builder chain for the problem maximize obj.

        Cvxium's solvers only minimize, so maximizing a concave g is done by minimizing
        -g. The returned ``Problem`` carries ``NegatedObjective(obj)`` with
        ``sense=Sense.MAXIMIZE``, so ``solve`` reports the objective value as g(w*), not
        the -g(w*) the solver computes. The solution point is the same either way.

        For the round trip to cost no extra arithmetic, ``obj`` should advertise its
        concave (negative-definite) curvature honestly, so that
        ``obj.hessian_contribution(w)`` is a ``negated()`` view of a positive-definite
        system. ``NegatedObjective`` then negates it once more, collapsing the two sign
        flips, and the solver only ever factors the positive-definite system.

        Chain ``.subject_to(eq=..., ineq=[...])`` onto the result to attach constraints;
        the maximization sense is kept across the chain.

        Parameters
        ----------
         obj : Objective
            The concave objective g(w) to maximize.

        Returns
        -------
         problem : Problem
            An unconstrained maximization ``Problem`` carrying ``NegatedObjective(obj)``
            and ``sense=Sense.MAXIMIZE``.

        """
        return cls(obj=NegatedObjective(obj), sense=Sense.MAXIMIZE)

    # A class-or-instance method: ``Problem.subject_to(...)`` builds a
    # feasibility problem (``obj=None``); ``some_problem.subject_to(...)``
    # builds a new problem carrying that instance's objective. Both return a
    # new frozen ``Problem`` with replace (not accumulate) constraint semantics.
    subject_to: ClassVar[_SubjectToDescriptor] = _SubjectToDescriptor()

    def __post_init__(self) -> None:
        """Validate the constraint requirement, the sense, and the dimensions.

        Rejects a feasibility problem with no constraints, and a ``Sense.MAXIMIZE``
        problem whose objective is not a ``NegatedObjective``: the solver would minimize
        the bare objective while ``solve`` negated the reported value, a
        sign-inconsistent result. The sense check applies only when an objective is
        present, because a feasibility sub-problem derived from a maximization (by
        replacing ``obj`` with ``None``) inherits the parent's sense harmlessly.

        Raises
        ------
         ValueError
            When ``obj``, ``eq`` and ``ineq`` are all absent; when ``sense`` is
            ``Sense.MAXIMIZE`` on an objective that is not a ``NegatedObjective``; or,
            propagated from ``_check_dimension_integrity``, when the components disagree
            on the dimension or the dimension is not positive.

        """
        if self.obj is None and self.eq is None and not self.ineq:
            raise ValueError(
                "A feasibility problem (obj=None) requires at least one equality "
                "or inequality constraint to satisfy."
            )
        # ``sense`` is a plain field, so a caller could set ``Sense.MAXIMIZE``
        # directly on an un-negated objective; the solver would then minimize that
        # objective while ``solve`` negated the reported value, a sign-inconsistent
        # result. Enforce the invariant ``Problem.maximize`` establishes -- a
        # maximization always carries the ``NegatedObjective(-g)`` wrapper -- so
        # ``maximize`` is the only supported way to build one. The check is scoped
        # to problems that actually carry an objective: a feasibility solve derives
        # its sub-problem via ``replace(problem, obj=None)`` (barrier/PD Phase I),
        # which inherits the parent ``sense`` harmlessly -- there is no objective
        # value for ``solve`` to sign-flip, so the leftover ``MAXIMIZE`` is inert.
        if (
            self.obj is not None
            and self.sense is Sense.MAXIMIZE
            and not isinstance(self.obj, NegatedObjective)
        ):
            raise ValueError(
                "sense=Sense.MAXIMIZE is only valid on a Problem whose objective is a "
                "NegatedObjective -- the -g wrapper Problem.maximize installs. Build a "
                "maximization with Problem.maximize(g) rather than setting sense "
                "directly, so the solver minimizes -g while solve reports g(w*) "
                "consistently."
            )
        self._check_dimension_integrity()

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M, derived from the components.

        A ``Problem`` does not store M; it reads it from the objective if present, else
        the equality constraint, else the first inequality layer. Construction checked
        that the components agree, so the choice does not matter.

        Returns
        -------
         M : int
            The decision-variable dimension.

        Raises
        ------
         ValueError
            When the problem has no components, which ``__post_init__`` rules out.

        """
        if self.obj is not None:
            return self.obj.dimension
        if self.eq is not None:
            return self.eq.dimension
        if self.ineq:
            return self.ineq[0].dimension
        # Unreachable: __post_init__ guarantees at least one component exists.
        raise ValueError("Problem has no components to derive its dimension from.")

    def _check_dimension_integrity(self) -> None:
        """Check that every component agrees on a positive dimension.

        Each component (objective, equality constraint, inequality layer) reports the
        dimension fixed at its construction: a coefficient vector's length, a matrix's
        column count, a bounds length, or the slack-augmented M + 1 of a Phase I layer.
        A disagreement is raised here, at construction, rather than surfacing as an
        obscure shape failure deep inside a solve.

        Raises
        ------
         ValueError
            When two components report different dimensions, or the common dimension is
            not positive.

        """
        components: list[tuple[str, int]] = []
        if self.obj is not None:
            components.append(("objective", self.obj.dimension))
        if self.eq is not None:
            components.append(("equality constraint", self.eq.dimension))
        for i, layer in enumerate(self.ineq):
            components.append((f"inequality layer {i}", layer.dimension))
        # __post_init__ guarantees at least one component is present.
        reference_name, reference_dimension = components[0]
        for name, component_dimension in components[1:]:
            if component_dimension != reference_dimension:
                raise ValueError(
                    f"Problem components disagree on dimension: "
                    f"{reference_name} has {reference_dimension}, but the "
                    f"{name} has {component_dimension}."
                )
        # The components agree on a common dimension; reject a degenerate
        # zero-size (or negative) dimension here rather than letting it surface
        # as a cryptic shape error deep inside a solve.
        if reference_dimension <= 0:
            raise ValueError(
                f"Problem dimension must be positive, got {reference_dimension}."
            )

    def solve(
        self,
        solver: Solver,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Run a solver on this problem, reporting the result in this problem's sense.

        For a minimization this returns the solver's result unchanged. For a
        maximization (built with ``maximize``) the solver minimizes the carried -g, so
        the objective outcome is negated back (``ObjectiveOutcome.negated``) and
        ``objective_value`` reads g(w*). The solution point is never altered.

        This is the sense-aware entry point: calling ``solver.solve(problem)`` directly
        skips the flip and reports the raw -g minimization.

        Parameters
        ----------
         solver : Solver
            The algorithm to run, for example ``BarrierSolver()``.
         x0 : npt.NDArray[np.float64] | None
            Initial guess of length M, forwarded to the solver; ``None`` lets the solver
            choose its start.
         on_infeasible : OnInfeasible
            How a feasibility solve responds to an infeasibility certificate:
            ``"raise"`` raises as soon as the certificate appears; ``"continue"`` first
            drives the Phase I slack to its optimum and attaches that best-effort result
            to the raised exception.

        Returns
        -------
         result : SolveResult
            The solver's result, with ``result.objective`` in this problem's sense. Only
            that outcome is flipped: for a maximization, the ``diagnostics`` trajectory
            and any nested Phase I result (``phase1``) stay in the solver's internal
            minimization sense, so they record what the solver actually did.

        Raises
        ------
         NotImplementedError
            Propagated from the solver when it cannot honor
            ``on_infeasible="continue"``.
         ValueError
            Propagated from the solver when it does not support the problem's shape.
         ProblemInfeasibleError
            Propagated from the solver when the problem is infeasible.

        """
        result = solver.solve(self, x0, on_infeasible=on_infeasible)
        if self.sense is Sense.MAXIMIZE and result.objective is not None:
            return replace(result, objective=result.objective.negated())
        return result

    def is_feasible(self, w: npt.NDArray[np.float64]) -> bool:
        """Check whether a point strictly satisfies every inequality layer.

        Equality constraints are not checked; a caller that needs A w = b checks the
        residual itself, as ``Solver._solve_feasibility`` does.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         feasible : bool
            True when every layer is strictly satisfied at w, and True when there are no
            inequality layers.

        """
        return all(layer.is_satisfied(w) for layer in self.ineq)

    def num_inequality_constraints(self) -> int:
        """Count the inequality constraints across all layers.

        Returns
        -------
         m : int
            The sum of ``num_constraints`` over the layers in ``ineq``, at this
            problem's dimension.

        """
        return sum(layer.num_constraints(self.dimension) for layer in self.ineq)

    def build_hessian_system(
        self,
        x: npt.NDArray[np.float64],
        point: ObjectivePoint,
        objective_scale: float,
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Assemble the weighted Hessian of the Newton system as a ``MatrixSystem``.

        Builds
           s ∇^2 f0(x) + ∑ λ_i C_i(x),   C_i = ∇^2 f_i - (1 / f_i) ∇f_i ∇f_i^T,
        where s is ``objective_scale`` and C_i is the per-constraint curvature atom of
        ``ConstraintLayer.hessian_contribution``. One contract serves both
        interior-point methods:
        - Barrier: ``objective_scale`` = t and ``lmbda`` = -1 / f, giving
          t ∇^2 f0 + ∑ ∇^2 φ_i, the Hessian of the barrier-augmented objective.
        - Primal-dual: ``objective_scale`` = 1 and ``lmbda`` the dual variables, giving
          the condensed modified Hessian ∇^2 f0 + ∑ λ_i C_i.

        The objective's curvature (scaled by ``objective_scale``) and each layer's
        contribution (weighted by its slice of ``lmbda``) are summed with
        ``MatrixSystem.plus``. Same-shape bases combine in closed form (diagonal plus
        diagonal stays diagonal), and low-rank updates ride on top of any base.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current iterate, of length M, at which the layers' contributions are
            evaluated.
         point : ObjectivePoint
            The objective's per-point context, which must be ``self.obj.at(x)``; passed
            in so the per-iterate forward pass is computed once and shared.
         objective_scale : float
            Scalar multiplying the objective's curvature (t for barrier, 1 for
            primal-dual).
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights, of length ``num_inequality_constraints()``, stacked
            in constraint order across the layers (-1 / f for barrier, the dual
            variables for primal-dual).

        Returns
        -------
         system : MatrixSystem
            The assembled Hessian.

        Raises
        ------
         IncompatibleStructureError
            Propagated from ``MatrixSystem.plus`` when two contributions have base
            shapes that do not compose, rather than silently densifying.

        """
        M = self.dimension
        systems: list[MatrixSystem] = [
            point.hessian_contribution().scaled(objective_scale)
        ]
        offset = 0
        for layer in self.ineq:
            n_c = layer.num_constraints(M)
            systems.append(layer.hessian_contribution(x, lmbda[offset : offset + n_c]))
            offset += n_c
        return reduce(lambda left, right: left.plus(right), systems)

    def make_phase1(
        self,
        x0: npt.NDArray[np.float64],
    ) -> tuple[Problem, npt.NDArray[np.float64]]:
        """Construct a Phase I problem whose solution is a strictly feasible point.

        Takes the last layer in ``ineq`` as the active layer (the one to make feasible)
        and the rest as background layers, and dispatches on the active layer's type:
        - ``DirectConstraintLayer``: direct form. The active layer, which is also an
          ``Objective``, becomes the objective, with the background layers and the
          equality constraint kept as constraints. No augmentation.
        - ``SlackConstraintLayer``: slack form. The variable is augmented to
          w_aug = [w; s], the objective minimizes s, the background layers are lifted to
          w_aug, the active layer is relaxed to f_i(w) ≤ s, and s is bounded above (see
          ``_make_phase1_slack_form``).
        ``Solver._solve_feasibility`` drives the Phase I solve.

        When equality constraints are present, ``x0`` is first projected onto A w = b
        (with ``EqualitySolver``), since a feasible-start barrier keeps A w = b but
        never corrects an off-manifold iterate. Both forms start from the projected
        point. The slack form's augmented matrix [A 0] leaves A w = b unchanged, so
        projecting w suffices.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Starting iterate, of length M. For the slack form it also sets the initial
            slack, and it should already satisfy the background layers.

        Returns
        -------
         phase1_problem : Problem
            The Phase I problem.
         x0_phase1 : npt.NDArray[np.float64]
            Starting iterate for ``phase1_problem``: the (projected) x0 for the direct
            form, or [x0; s0], of length M + 1, for the slack form.

        Raises
        ------
         ValueError
            When ``ineq`` is empty; a problem without inequality layers needs no Phase
            I.
         NotImplementedError
            When the active layer is neither a ``SlackConstraintLayer`` nor a
            ``DirectConstraintLayer``.
         ProblemInfeasibleError
            Propagated from ``EqualitySolver.solve`` when b is not in the range of A.

        """
        if not self.ineq:
            raise ValueError(
                "make_phase1 requires at least one inequality constraint layer; "
                "no Phase I is needed for unconstrained problems."
            )
        # Project ``x0`` onto the equality manifold ``A x = b`` so the Phase I
        # start is equality-consistent. A feasible-start barrier maintains
        # ``A x = b`` but never corrects an off-manifold iterate, so it needs an
        # on-manifold start; ``EqualitySolver.solve`` returns exactly the
        # least-norm projection (idempotent when ``x0`` is already on the
        # manifold). The slack form's augmented matrix ``[A | 0]`` leaves
        # ``A w = b`` unchanged, so projecting the original ``w`` suffices.
        if self.eq is not None:
            # Imported lazily to avoid a circular import: ``solver`` (which
            # defines ``EqualitySolver``) imports this module at top level.
            from ..solvers.solver import (
                EqualitySolver,
            )

            x0 = np.asarray(EqualitySolver(constraint=self.eq).solve(x0=x0).solution)
        active = self.ineq[-1]
        background = self.ineq[:-1]

        if isinstance(active, DirectConstraintLayer):
            return self._make_phase1_direct_form(active, background), x0

        if isinstance(active, SlackConstraintLayer):
            return self._make_phase1_slack_form(active, background, x0)

        raise NotImplementedError(
            f"make_phase1 cannot dispatch on active layer of type "
            f"{type(active).__name__}; expected SlackConstraintLayer or "
            "DirectConstraintLayer."
        )

    def _make_phase1_direct_form(
        self,
        active: DirectConstraintLayer,
        background: list[ConstraintLayer],
    ) -> Problem:
        """Build a direct-form Phase I problem with the active layer as its objective.

        ``DirectConstraintLayer`` subclasses ``Objective``, so the layer is the Phase I
        objective directly, with no adapter. Minimizing its value f(w), subject to the
        background layers and the equality constraint, drives it below zero.

        Parameters
        ----------
         active : DirectConstraintLayer
            The layer to make feasible.
         background : list[ConstraintLayer]
            The remaining inequality layers, kept as constraints.

        Returns
        -------
         phase1_problem : Problem
            The direct-form Phase I problem, of dimension M.

        """
        return Problem(
            obj=active,
            ineq=list(background),
            eq=self.eq,
        )

    def _make_phase1_slack_form(
        self,
        active: SlackConstraintLayer,
        background: list[ConstraintLayer],
        x0: npt.NDArray[np.float64],
    ) -> tuple[Problem, npt.NDArray[np.float64]]:
        """Build a slack-form Phase I problem over the augmented variable [w; s].

        The Phase I problem is
           minimize    s
           subject to  f_i(w) ≤ s               (active layer, ``_SlackActiveLayer``),
                       s ≤ s_max                (``_SlackUpperBoundLayer``),
                       background layers on w   (each a ``LiftedConstraintLayer``),
                       [A 0] [w; s] = b         (when equalities are present),
        with
           s0 = max_i f_i(x0) + 1,   s_max = s0 + 1,
        so the start [x0; s0] strictly satisfies the active layer and the upper bound.
        The upper bound keeps the slack bounded so the barrier subproblem has a
        minimizer. The augmented equality matrix keeps the sparsity of A and its saddle
        factory, so a sparse equality problem stays on the sparse KKT path.

        Parameters
        ----------
         active : SlackConstraintLayer
            The layer to make feasible.
         background : list[ConstraintLayer]
            The remaining inequality layers, which x0 should already satisfy strictly.
         x0 : npt.NDArray[np.float64]
            Starting point, of length M.

        Returns
        -------
         phase1_problem : Problem
            The slack-form Phase I problem, of dimension M + 1, whose active layer is
            the relaxed active layer.
         x0_aug : npt.NDArray[np.float64]
            The starting point [x0; s0], of length M + 1.

        """
        M = self.dimension
        eps = 1.0
        s0 = float(np.max(active.constraints(x0))) + eps
        s_max = s0 + eps

        lifted_background: list[ConstraintLayer] = [
            LiftedConstraintLayer(base=layer, base_dimension=M) for layer in background
        ]
        slack_active = _SlackActiveLayer(base=active, base_dimension=M, s_max=s_max)
        slack_upper = _SlackUpperBoundLayer(s_max=s_max, base_dimension=M)

        phase1_constraints: list[ConstraintLayer] = [
            *lifted_background,
            slack_upper,
            slack_active,
        ]

        # Augment the equality matrix with a zero column for the slack variable,
        # preserving both the sparsity of ``A`` and the layer's saddle factory
        # (so a sparse equality problem stays on the sparse KKT path in Phase I).
        phase1_eq: EqualityConstraintLayer | None = None
        if self.eq is not None:
            A = self.eq.A
            if scipy.sparse.issparse(A):
                zero_col = scipy.sparse.csr_matrix((A.shape[0], 1))
                A_aug = scipy.sparse.hstack([A, zero_col], format="csr")
            else:
                A_aug = np.hstack([A, np.zeros((A.shape[0], 1))])
            phase1_eq = EqualityConstraintLayer(
                A=A_aug, b=self.eq.b, saddle_factory=self.eq.saddle_factory
            )

        # Phase I objective is the slack variable: f(w_aug) = w_aug[M].
        c_obj = np.zeros(M + 1)
        c_obj[M] = 1.0
        phase1_objective = LinearObjective(c=c_obj)

        x0_aug = np.zeros(M + 1)
        x0_aug[:M] = x0
        x0_aug[M] = s0

        phase1_problem = Problem(
            obj=phase1_objective,
            ineq=phase1_constraints,
            eq=phase1_eq,
        )
        return phase1_problem, x0_aug
