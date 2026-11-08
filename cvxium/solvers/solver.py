"""Solver abstraction: an algorithm that consumes a Problem and produces a Result.

Defines ``Solver``, the abstract base for Cvxium's algorithms (``NewtonSolver``,
``BarrierSolver``, ``PrimalDualSolver``, ``HsdSolver``). Each takes a ``Problem``, the
mathematical statement, and returns a ``SolveResult``. The base also carries the
feasibility machinery the interior-point solvers share: ``_solve_feasibility`` finds a
strictly feasible starting point, building a Phase I problem with
``Problem.make_phase1`` and solving it through the same solver with ``_solve_phase1``,
and ``_feasible_result`` lifts a point that needed no Phase I into a full result.

``EqualitySolver`` handles the linear equality constraints A x = b: it projects a point
onto the affine set, through an SVD for a dense A or a KKT saddle system for a sparse
one, and supplies the equality-aware initial barrier parameter.

"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

from .._typing import none_throws
from ..exceptions import (
    BacktrackingLineSearchError,
    NewtonStepError,
    ProblemInfeasibleError,
)
from ..linalg.numerical_helpers import (
    tikhonov_filter,
)
from ..linalg.systems import DiagonalSystem
from ..problems.constraint_layers import (
    DirectConstraintLayer,
)
from ..problems.linear_constraints import (
    EqualityConstraintLayer,
)
from ..problems.problem import (
    OnInfeasible,
    Problem,
)
from ..problems.result import (
    BarrierDiagnostics,
    ObjectiveOutcome,
    SolveResult,
    SolveStatus,
)
from ..solvers.optimization import (
    FeasibilitySolver,
    ProblemCertifiablyInfeasibleError,
    ProblemMarginallyFeasibleError,
    _configure_verbose_logging,
    _duals_from,
)
from ..solvers.settings import (
    OptimizationSettings,
)


class Solver(ABC):
    """Abstract algorithm that solves a ``Problem``.

    A ``Solver`` encapsulates an optimization algorithm (Newton's method, the barrier
    method, primal-dual, the homogeneous self-dual method). It is decoupled from the
    mathematical statement of the problem, which lives on a ``Problem`` (objective,
    inequality layers, and equality constraints).

    Concrete subclasses implement a single ``solve(problem, x0)`` entry point. The same
    ``Problem`` can be handed to different solvers without modification, and the same
    solver can be applied to any ``Problem`` whose shape it supports; a subclass raises
    ``ValueError`` when handed a ``Problem`` it cannot solve.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm-level settings (tolerances, iteration limits, line-search parameters).
        ``None`` uses the solver's defaults.

    """

    def __init__(self, settings: OptimizationSettings | None = None) -> None:
        self.settings: OptimizationSettings | None = settings

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Solve ``problem`` starting from the initial guess ``x0``.

        Parameters
        ----------
         problem : Problem
            The mathematical problem to solve.
         x0 : npt.NDArray[np.float64] | None
            Initial guess of length ``problem.dimension``, or ``None`` to let the solver
            choose a start. The barrier and primal-dual solvers accept ``None`` or an
            infeasible ``x0`` and construct a strictly feasible start with
            ``_solve_feasibility``; ``HsdSolver`` ignores ``x0`` and builds its own;
            ``NewtonSolver`` requires an explicit start and raises ``ValueError`` when
            ``x0`` is ``None``.
         on_infeasible : OnInfeasible
            How a feasibility solve responds to an infeasibility certificate:
            ``"raise"`` as soon as it appears, or ``"continue"`` to the slack optimum
            first. Solvers that cannot honor ``"continue"`` raise
            ``NotImplementedError``.

        Returns
        -------
         result : SolveResult
            Solver result, including the final iterate and diagnostic data.

        Raises
        ------
         ValueError
            When the solver does not support the shape of ``problem`` (for example, a
            ``NewtonSolver`` handed inequality constraints, or a ``BarrierSolver``
            handed none).
         NotImplementedError
            When ``on_infeasible="continue"`` and the solver cannot honor it
            (``NewtonSolver``, ``PrimalDualSolver`` and ``HsdSolver``).
         ProblemInfeasibleError
            When a feasibility solve shows the problem infeasible
            (``ProblemCertifiablyInfeasibleError``) or cannot locate a feasible point
            (``ProblemMarginallyFeasibleError``).

        """
        ...

    def _validate_x0(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64],
    ) -> None:
        """Reject a supplied ``x0`` the solver cannot start from.

        ``_solve_feasibility`` calls this hook before testing whether a supplied ``x0``
        is already feasible. The default accepts every ``x0``, and no solver in Cvxium
        overrides it: ``BarrierSolver``, whose feasible-start barrier maintains A x = b
        but never corrects it, projects a supplied ``x0`` onto the equality manifold in
        its ``solve`` instead, and the primal-dual solver carries the equality residual
        in its Newton step and drives it to zero. An override should raise
        ``ValueError`` for an unusable ``x0``.

        Parameters
        ----------
         problem : Problem
            The problem being solved.
         x0 : npt.NDArray[np.float64]
            The supplied starting point, of length M.

        """
        return

    def _solve_feasibility(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Find a strictly feasible point for a problem with no objective.

        A feasibility problem asks for any point satisfying the constraints. The
        technique follows from its structure, tried in order:
        1. Equality only (no inequality layers): one projection of ``x0`` onto
           A x = b by ``EqualitySolver.solve``, with no barrier and no iteration.
        2. A supplied ``x0`` that strictly satisfies every inequality layer and lies on
           the equality manifold (every component of A x0 - b below 1e-10 in absolute
           value, the gate ``EqualitySolver.solve`` uses) is returned as is.
        3. When the active layer is the only inequality layer and is a
           ``DirectConstraintLayer`` whose ``interior_point`` supplies a point (given
           the equality constraints), that point is returned after the inequality and
           the equality residual are rechecked.
        4. Otherwise Phase I runs on the active layer, the last in ``problem.ineq``: a
           direct-form problem when it is a ``DirectConstraintLayer`` and a slack-form
           problem when it is a ``SlackConstraintLayer``, built by
           ``Problem.make_phase1`` and solved by ``_solve_phase1``.

        Parameters
        ----------
         problem : Problem
            The feasibility problem (``obj`` is ``None``), with at least one constraint.
         x0 : npt.NDArray[np.float64] | None
            Starting point of length M, which need not be feasible. ``None`` starts from
            the minimum-norm solution of A x = b in the equality-only case, and from the
            origin (projected onto A x = b when equality constraints are present)
            otherwise.
         on_infeasible : OnInfeasible
            How Phase I responds to an infeasibility certificate, forwarded to
            ``_solve_phase1`` and the inner solves.

        Returns
        -------
         result : SolveResult
            A result whose ``solution`` is strictly feasible. Phase I returns the full
            result of its solve, with any slack removed; cases 1 to 3 return a bare
            FEASIBLE result with no objective, which ``BarrierSolver`` and
            ``PrimalDualSolver`` lift with ``_feasible_result``.

        Raises
        ------
         ValueError
            When ``problem`` has an objective or no constraints. Also propagated from
            ``self.solve`` when a sole ``DirectConstraintLayer`` offers no interior
            point (see Notes).
         ProblemInfeasibleError
            When the equality constraints are inconsistent. Propagated from
            ``EqualitySolver.solve``, directly or through ``Problem.make_phase1``.
         ProblemCertifiablyInfeasibleError
            When Phase I cannot reach a strictly feasible point. Raised by
            ``_solve_phase1`` or propagated from the inner solve. Raising keeps an
            infeasible iterate out of a main loop, where the λ / (-f) weights or the log
            barrier go non-finite.
         ProblemMarginallyFeasibleError
            When Phase I breaks down numerically under ``on_infeasible="raise"``.
            Propagated from ``_solve_phase1``.
         NewtonStepError
            When Phase I breaks down numerically under ``on_infeasible="continue"``.
            Propagated from ``_solve_phase1``.
         BacktrackingLineSearchError
            As for ``NewtonStepError``.

        Notes
        -----
        Phase I solves through ``self``, the technique the caller chose (barrier or
        primal-dual). Both honor the zero feasibility threshold that ``_solve_phase1``
        attaches to the Phase I objective, so the solve stops at the first strictly
        feasible iterate rather than chasing the optimum.

        The recursion terminates by reducing the layer count. With a
        ``DirectConstraintLayer`` active, ``make_phase1`` hoists the layer to the
        objective and keeps the other (background) layers as constraints; ``self.solve``
        then seeds that problem with its own feasibility solve, a recursion on one fewer
        layer. With a ``SlackConstraintLayer`` active, the background layers stay hard
        constraints, so they are made feasible first, here, by a recursion on one fewer
        layer; the slack-form problem built from that point has a strictly feasible
        augmented start by construction. Case 2 also ends the recursion: when
        ``self.solve`` seeds a Phase I problem whose start is already feasible, the call
        back here returns at once.

        Case 3 exists because a sole ``DirectConstraintLayer`` leaves a direct-form
        Phase I problem with no inequality layers (the layer is hoisted to the
        objective), a shape the barrier and primal-dual solvers reject with
        ``ValueError``. A layer with background layers cannot use case 3, since its own
        interior point need not satisfy them.

        """
        if problem.obj is not None:
            raise ValueError(
                "_solve_feasibility requires a feasibility problem (obj=None); got "
                "a problem with an objective."
            )
        if not problem.ineq and problem.eq is None:
            raise ValueError(
                "_solve_feasibility requires at least one constraint to satisfy."
            )
        # Equality only: a single projection onto the manifold.
        if not problem.ineq:
            return EqualitySolver(
                constraint=none_throws(problem.eq), settings=self.settings
            ).solve(x0=x0)
        # A strictly feasible warm start needs no Phase I. This short-circuit also
        # terminates the recursion: when ``self.solve`` seeds its own Phase I by
        # calling back here on the already-feasible augmented start, this returns
        # immediately instead of transforming the problem again.
        if x0 is not None:
            self._validate_x0(problem, x0)
            # ``Problem.is_feasible`` checks only the inequality layers, so also
            # require ``x0`` to lie on the equality manifold ``A x = b`` before
            # short-circuiting. Otherwise an inequality-feasible but off-manifold
            # ``x0`` (which infeasible-start solvers like primal-dual accept) would
            # be returned verbatim as a "feasible" solution that violates the
            # equality constraints, skipping the solve that drives the residual to
            # zero.
            # Use the same absolute residual threshold ``EqualitySolver.solve``
            # applies when deciding ``x0`` already lies on the manifold
            # (``|A x0 - b| < 1e-10``), not ``np.allclose``'s looser defaults
            # (atol=1e-8, rtol=1e-5). Otherwise this gate could accept an ``x0``
            # that ``EqualitySolver`` would still refine, returning a point with a
            # larger equality residual than an actual solve would.
            on_manifold = problem.eq is None or bool(
                np.all(np.abs(problem.eq.A @ x0 - problem.eq.b) < 1e-10)
            )
            if problem.is_feasible(x0) and on_manifold:
                return SolveResult(
                    solution=x0,
                    status=SolveStatus.FEASIBLE,
                    message="",
                    solver="",
                    n_iterations=0,
                )
        active = problem.ineq[-1]
        if isinstance(active, DirectConstraintLayer):
            # When the active layer is the *only inequality* it may know a
            # strictly interior point in closed form (e.g. an ellipsoid's center,
            # or -- given the equality manifold -- its equality-constrained
            # least-squares point). Taking that point certifies feasibility with no
            # Phase I solve, and sidesteps the direct-form Phase I whose
            # objective-only shape (the sole layer hoisted to the objective,
            # leaving zero inequality layers) the barrier engine cannot solve. The
            # equality manifold is handed to the layer so it can return a point on
            # it; ``is_feasible`` checks only the inequalities, so the manifold
            # residual is re-checked here (same ``1e-10`` gate the x0 short-circuit
            # uses) before accepting. Fall through to the iterative Phase I when the
            # layer offers no such point. The one structural guard that remains is
            # ``len(ineq) == 1``: with background inequalities the layer's own
            # point need not satisfy them, so it cannot be returned verbatim.
            if len(problem.ineq) == 1:
                interior = active.interior_point(problem.eq)
                if interior is not None and problem.is_feasible(interior):
                    on_manifold = problem.eq is None or bool(
                        np.all(np.abs(problem.eq.A @ interior - problem.eq.b) < 1e-10)
                    )
                    if on_manifold:
                        return SolveResult(
                            solution=interior,
                            status=SolveStatus.FEASIBLE,
                            message="Active layer provided strictly feasible point",
                            solver="hook",
                            n_iterations=0,
                        )
            # Direct form: ``make_phase1`` hoists the active layer to the objective;
            # ``self.solve`` then cascades the backgrounds via its own feasibility
            # seeding (a recursion on one fewer layer).
            phase1_problem, x0_phase1 = problem.make_phase1(
                x0 if x0 is not None else np.zeros(problem.dimension)
            )
            return self._solve_phase1(
                problem, phase1_problem, x0_phase1, on_infeasible=on_infeasible
            )
        # Slack form: the backgrounds stay hard, so feasibilize them first (a
        # recursion on one fewer layer), then build the slack problem -- whose
        # augmented start is feasible by construction -- and solve it.
        background = problem.ineq[:-1]
        if background:
            seed = np.asarray(
                self._solve_feasibility(
                    Problem(
                        eq=problem.eq,
                        ineq=background,
                    ),
                    x0,
                    on_infeasible=on_infeasible,
                ).solution
            )
        else:
            seed = x0 if x0 is not None else np.zeros(problem.dimension)
        phase1_problem, aug_x0 = problem.make_phase1(seed)
        return self._solve_phase1(
            problem, phase1_problem, aug_x0, on_infeasible=on_infeasible
        )

    def _solve_phase1(
        self,
        problem: Problem,
        phase1_problem: Problem,
        x0_phase1: npt.NDArray[np.float64],
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Solve a Phase I problem as a feasibility solve and recover the result.

        Attaches a zero feasibility threshold to the Phase I objective (so the solve
        stops at the first strictly feasible iterate), solves through ``self``, drops
        any augmenting slack from the solution, and raises
        ``ProblemCertifiablyInfeasibleError`` when the recovered point is not strictly
        feasible for the original problem.

        Parameters
        ----------
         problem : Problem
            The original feasibility problem, used to drop the slack and to check the
            recovered point.
         phase1_problem : Problem
            The Phase I problem from ``Problem.make_phase1``; it must carry an
            objective.
         x0_phase1 : npt.NDArray[np.float64]
            Starting point for ``phase1_problem``. The slack form appends the slack, so
            its length is M + 1; the direct form's is M.
         on_infeasible : OnInfeasible
            How the solve responds to an infeasibility certificate; it also decides
            whether a numerical breakdown is converted (see Notes).

        Returns
        -------
         result : SolveResult
            The Phase I result, with ``solution`` truncated to the original dimension M.

        Raises
        ------
         ProblemMarginallyFeasibleError
            When the Phase I solve breaks down numerically (``NewtonStepError`` or
            ``BacktrackingLineSearchError``) under ``on_infeasible="raise"``.
         NewtonStepError
            Propagated unchanged from the Phase I solve under
            ``on_infeasible="continue"``.
         BacktrackingLineSearchError
            Propagated unchanged from the Phase I solve under
            ``on_infeasible="continue"``.
         ProblemCertifiablyInfeasibleError
            When the recovered point is not strictly feasible for ``problem``; also
            propagated unchanged when the inner solve raises it.
         ValueError
            Propagated unchanged from the inner solve, for a problem the solver cannot
            represent.

        Notes
        -----
        Under ``on_infeasible="raise"``, a Phase I solve that breaks down numerically
        raises ``ProblemMarginallyFeasibleError`` rather than propagating the low-level
        failure. Phase I's whole contract is to answer feasible-or-not, so a breakdown
        here is the answer "could not find a feasible point". Such breakdowns cluster
        just inside the infeasible side, where the feasible set is empty or vanishingly
        thin and the iterates are driven onto the boundary; without the conversion,
        callers would see an internal error about slack signs or a descent direction,
        several layers below anything they control. The conversion deliberately does not
        claim a certificate: the marginal variant says the problem may be infeasible
        without proving it.

        The ``SolveResult`` that exception carries exists to satisfy its contract and
        holds the Phase I seed, not a computed point, since the numerical exceptions
        carry no iterate to attach. Its ``solution`` is not a minimum-violation answer,
        and it has no ``objective``.

        That is also why the conversion does not apply under
        ``on_infeasible="continue"``: there the caller has asked for a best-effort point
        and will consume ``solve_result.solution`` as one, possibly returning it as its
        own answer. Rather than pass off an untouched seed as a minimum-violation
        solution, that path lets the breakdown propagate unchanged.

        Only the numerical-failure family is converted. A ``ValueError`` reports a
        problem the solver cannot represent, such as a layer shape it does not support;
        that is a caller error, not a verdict about the feasible set, and relabelling it
        would hide a real defect behind a plausible-looking infeasibility. An
        infeasibility verdict raised by the inner solve is already the answer and
        propagates untouched. The conversion is also scoped to Phase I: the same
        breakdown in the optimization path surfaces as itself.

        """
        transformed = replace(
            phase1_problem,
            obj=none_throws(phase1_problem.obj).with_feasibility_threshold(0.0),
        )
        try:
            result = self.solve(transformed, x0_phase1, on_infeasible=on_infeasible)
        except (NewtonStepError, BacktrackingLineSearchError) as e:
            if on_infeasible != "raise":
                # The caller asked to continue to a best-effort point, and Phase
                # I has none to give: these exceptions carry only a message, no
                # iterate. Manufacturing one would hand back the untouched seed
                # -- for the direct form often just zeros -- and consumers read
                # ``solve_result.solution`` as a minimum-violation answer
                # (a caller may return it as its own result). A
                # breakdown that speaks for itself beats a verdict carrying a
                # point nobody computed.
                raise

            raise ProblemMarginallyFeasibleError(
                message=(
                    "Phase I could not find a feasible point: the solve broke down "
                    f"numerically ({type(e).__name__}: {e}). The problem may be "
                    "infeasible, or feasible by too thin a margin to locate."
                ),
                result=SolveResult(
                    solution=np.asarray(x0_phase1)[: problem.dimension],
                    status=SolveStatus.INFEASIBLE,
                    message=(
                        "Phase I broke down numerically; this result carries the "
                        "Phase I seed, not a computed point."
                    ),
                    solver=type(self).__name__,
                    n_iterations=0,
                ),
            ) from e
        # Slack-form augments the variable with a trailing slack; drop it. The
        # direct form's solution already has the original dimension, so the slice
        # is a no-op there. (This de-augments the *return* path for every solver,
        # including primal-dual, whose ``finalize_solution`` is the identity; the
        # barrier adapter additionally de-augments its ``on_infeasible``
        # exception/best-effort results, which are raised from inside the IPM loop
        # and never reach this line.)
        recovered = np.asarray(result.solution)[: problem.dimension]
        if not problem.is_feasible(recovered):
            # A Phase I transform always carries inequalities, so solving it
            # always yields a result with an objective and a dual value; use
            # none_throws so a violated invariant fails loudly at the source
            # rather than embedding nan in the infeasibility message.
            dual_value = none_throws(none_throws(result.objective).dual_value)
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    "Phase I converged but the recovered point is not strictly "
                    "feasible for the original problem; the problem is infeasible."
                    f" (Phase I dual value: {dual_value:.3e})."
                ),
                result=result,
            )
        return replace(result, solution=recovered)

    def _feasible_result(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
    ) -> SolveResult:
        """Wrap an already-feasible iterate as a zero-iteration feasibility result.

        ``_solve_feasibility`` returns a bare ``SolveResult`` with no objective when no
        Phase I runs: the supplied iterate was already feasible, or a sole direct layer
        supplied an interior point. ``BarrierSolver`` and ``PrimalDualSolver``, whose
        feasibility solves otherwise return a full result, lift that point here.

        The objective value reported is the largest constraint value of the active
        (last) layer at x, which is negative for a strictly feasible point, or 0.0 when
        there is no constraint value. The result carries zero multipliers, an infinite
        suboptimality, a dual value of -inf, empty ``BarrierDiagnostics``, and
        ``solver`` set to ``"barrier"``; ``PrimalDualSolver`` replaces the last two.

        Parameters
        ----------
         problem : Problem
            The feasibility problem that x satisfies.
         x : npt.NDArray[np.float64]
            A strictly feasible point, of length M.

        Returns
        -------
         result : SolveResult
            A FEASIBLE result at x with zero iterations.

        """
        # ``problem.ineq`` is non-empty here, but a layer may yield a zero-length
        # constraint vector at ``x``; ``np.max`` of an empty array raises, so treat
        # "no constraint values" the same as "no inequalities" (boundary 0.0).
        last_constraints = (
            problem.ineq[-1].constraints(x) if problem.ineq else np.empty(0)
        )
        objective_value = (
            float(np.max(last_constraints)) if last_constraints.size > 0 else 0.0
        )
        num_eq = problem.eq.num_eq_constraints if problem.eq is not None else 0
        return SolveResult(
            solution=np.asarray(x),
            status=SolveStatus.FEASIBLE,
            message="Starting iterate is already feasible; no Phase I needed.",
            solver="barrier",
            n_iterations=0,
            objective=ObjectiveOutcome(
                value=objective_value,
                suboptimality=float("inf"),
                dual_value=float("-inf"),
            ),
            duals=_duals_from(
                np.zeros(num_eq),
                np.zeros(problem.num_inequality_constraints()),
            ),
            diagnostics=BarrierDiagnostics(),
            phase1=None,
        )


class EqualitySolver(FeasibilitySolver):
    """Feasibility solver for linear equality constraints A x = b.

    Projects onto the affine set {x : A x = b}: it returns the nearest point to a
    supplied x0, or the minimum-norm point when none is given. A dense A is handled
    through its SVD, a sparse A through the KKT saddle system of the constraint layer.
    The equality data and the cached SVD live on an ``EqualityConstraintLayer``, so
    several solvers can share one layer and its warm SVD; ``A``, ``b`` and ``svd_A``
    delegate to it. ``initialize_barrier_parameter`` serves barrier solvers with
    equality constraints.

    Parameters
    ----------
     A : npt.NDArray[np.float64] | None
        Equality constraint matrix, of shape (p, M). Give either ``A`` and ``b`` or
        ``constraint``: passing ``constraint`` together with either of them, or neither
        form, makes the constructor raise ``ValueError``.
     b : npt.NDArray[np.float64] | None
        Equality right-hand side, of length p. A length that does not match ``A`` fails
        an assertion in ``EqualityConstraintLayer``.
     settings : OptimizationSettings | None
        Solver settings, read for the trace and verbosity; ``None`` uses the defaults.
     constraint : EqualityConstraintLayer | None
        A prebuilt equality layer, possibly shared with other solvers.

    """

    def __init__(
        self,
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        settings: OptimizationSettings | None = None,
        constraint: EqualityConstraintLayer | None = None,
    ) -> None:
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings
        if self.settings.observability.verbose:
            _configure_verbose_logging()

        # Accept either a pre-built ``EqualityConstraintLayer`` (so several
        # solvers can share one layer and its warm SVD) or the raw ``(A, b)``.
        if constraint is not None:
            if A is not None or b is not None:
                raise ValueError(
                    "EqualitySolver accepts either `constraint` or `(A, b)`, not both."
                )
            self.constraint: EqualityConstraintLayer = constraint
        elif A is not None and b is not None:
            self.constraint = EqualityConstraintLayer(A, b)
        else:
            raise ValueError(
                "EqualitySolver requires either a `constraint` layer or both "
                "`A` and `b`."
            )

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Return the equality matrix A, delegated to the constraint layer.

        Returns
        -------
         A : npt.NDArray[np.float64]
            The matrix, of shape (p, M); dense, or ``scipy.sparse`` when the layer holds
            a sparse matrix.

        """
        return self.constraint.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Return the equality right-hand side b, delegated to the constraint layer.

        Returns
        -------
         b : npt.NDArray[np.float64]
            The right-hand side, of length p.

        """
        return self.constraint.b

    @property
    def num_eq_constraints(self) -> int:
        """Return the number of equality constraints.

        Returns
        -------
         p : int
            The row count of A.

        """
        return self.constraint.num_eq_constraints

    @property
    def num_ineq_constraints(self) -> int:
        """Return the number of inequality constraints, which is zero.

        Returns
        -------
         m : int
            Always 0.

        """
        return 0

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Solve A x = b, projecting x0 onto the solution set when one is given.

        A supplied x0 that already satisfies A x = b (every component of A x0 - b below
        1e-10 in absolute value) is returned unchanged. Otherwise the result is the
        Euclidean projection of x0 onto {x : A x = b}, or the minimum-norm solution when
        x0 is ``None``. A dense A uses the cached SVD (see Notes); a sparse A goes
        through ``_project_via_saddle``.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Point to project, of length M, or ``None``.
         **kwargs : Any
            Ignored; accepted for the ``FeasibilitySolver.solve`` signature.

        Returns
        -------
         res : SolveResult
            A FEASIBLE result with zero iterations, whose ``solution`` satisfies the
            equality.

        Raises
        ------
         ProblemInfeasibleError
            When b is not in the range of A. For a dense A the test is relative,
               ‖U_r U_r^T b - b‖ > 1e-8 max(1, ‖b‖),
            where U_r holds the retained left singular vectors; for a sparse A it is
            propagated from ``_project_via_saddle``.

        Notes
        -----
        With x0 given, the projection solves
           minimize    ‖x - x0‖^2
           subject to  A x = b.
        The Lagrangian is
           L(x, ν) = ‖x - x0‖^2 + ν^T (A x - b),
        and setting its gradient in x to zero gives
           2 (x - x0) + A^T ν = 0,   so   x = x0 - (1/2) A^T ν.
        Substituting gives the dual function
           g(ν) = -(1/4) ν^T A A^T ν + ν^T (A x0 - b),
        which is maximized where
           A A^T ν_star = 2 (A x0 - b).
        The solution is then
           x_star = x0 - (1/2) A^T ν_star.

        A need not have full rank. With the thin SVD A = U diag(s) V^T, (A A^T)^{-1} is
        applied as
           U diag(1 / (s^2 + δ^2)) U^T,
        a Tikhonov-damped inverse (``numerical_helpers.tikhonov_filter``) that drops
        singular values at or below δ = 1e-12 s_max and damps those just above it, so a
        redundant row is ignored and a nearly redundant one is not amplified. Without
        x0, the minimum-norm solution is
           x = V diag(s / (s^2 + δ^2)) U^T b.
        The damping perturbs each retained mode by a relative δ^2 / (s^2 + δ^2),
        negligible unless s is near the floor.

        Both branches reuse the SVD cached on the layer. Given it, the minimum-norm
        branch costs O(k (p + M)), with k = min(p, M), and the x0 branch adds two
        products with A, O(p M).

        """
        if x0 is not None and np.all(np.abs(self.A @ x0 - self.b) < 1e-10):
            self.settings.trace.on_feasible_initial_guess()
            return SolveResult(
                solution=x0,
                status=SolveStatus.FEASIBLE,
                message="",
                solver="",
                n_iterations=0,
            )

        # A sparse A has no SVD path (svd_A is dense-only); project through the
        # injected saddle instead, which keeps A sparse when the layer carries a
        # SparseSaddleFactory.
        if scipy.sparse.issparse(self.A):
            return self._project_via_saddle(x0)

        # If A = U * s * Vh, then we seek to solve:
        # U * s * Vh * x = b
        #
        # Then s * Vh * x = U^T * b
        # Vh * x = (U^T * b) / s
        # x = Vh^T * (U^T * b) / s
        svd_start = time.time()

        U, s, Vh = self.svd_A()
        b = self.b
        # Tikhonov-damped SVD pseudo-inverse (see
        # ``numerical_helpers.tikhonov_filter``) rather than a hard rank cut then
        # ``1 / s^2``: a redundant constraint row (zero singular value) is
        # dropped, and a singular value that lands just above the floor is damped
        # instead of amplified, so a rank-deficient ``A`` stays numerically
        # stable. ``power == 2`` filters ``(A A^T)^{-1}``; ``power == 1`` the
        # pseudo-inverse of ``A`` for the min-norm branch.
        filt_aat, rank = tikhonov_filter(s, power=2)
        U_r = U[:, 0:rank]
        # Range test: ``b`` must lie in range(A) = range(U_r). Use a relative
        # residual (scaled by ``||b||``) rather than ``np.allclose``'s default
        # tolerances, which mis-fire for a poorly-scaled ``b``.
        residual = float(np.linalg.norm(U_r @ (U_r.T @ b) - b))
        if residual > 1e-8 * max(1.0, float(np.linalg.norm(b))):
            raise ProblemInfeasibleError("b was not in the range of A.")

        if x0 is not None:
            # Step 1: find nu such that (A A^T) nu = 2 (A x0 - b) =: rhs. With
            # A = U diag(s) V^T, the Tikhonov-regularized (A A^T + lmbda^2 I)^{-1}
            # acting on the range is U diag(1 / (s^2 + lmbda^2)) U^T (the
            # ``power == 2`` filter, zero outside the retained span). Up to the
            # negligible damping this is exact when A is full rank, and it stays
            # stable when A is not.
            rhs = 2.0 * (self.A @ x0 - b)
            nu_star = U @ (filt_aat * (U.T @ rhs))

            # Step 2: Calculate x as x0 - (1/2) * A^T nu_star.
            x = x0 - 0.5 * (self.A.T @ nu_star)
        else:
            filt_a, _ = tikhonov_filter(s, power=1)
            x = Vh.T @ (filt_a * (U.T @ b))

        svd_elapsed = 1000 * (time.time() - svd_start)
        self.settings.trace.on_svd_done(svd_elapsed)

        return SolveResult(
            solution=x,
            status=SolveStatus.FEASIBLE,
            message="",
            solver="",
            n_iterations=0,
        )

    def _project_via_saddle(self, x0: npt.NDArray[np.float64] | None) -> SolveResult:
        """Project onto A x = b for a sparse A, through the injected saddle system.

        Solves the same projection as ``solve``,
           minimize    ‖x - x0‖^2
           subject to  A x = b,
        with x0 = 0 when it is ``None`` (the minimum-norm solution), through its KKT
        system
           | 2 I  A^T | | x |   | 2 x0 |
           |  A    0  | | ν | = |  b   |.
        Routing it through ``EqualityConstraintLayer.saddle_system`` reuses the injected
        ``SaddleFactory``, so a ``SparseSaddleFactory`` keeps A sparse throughout and no
        SVD of A is formed. The solution is then checked directly, since a
        borderline-singular Schur complement can pass the Cholesky factorization with a
        bogus multiplier.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Point to project, of length M, or ``None``.

        Returns
        -------
         res : SolveResult
            A FEASIBLE result with zero iterations, whose ``solution`` satisfies the
            equality.

        Raises
        ------
         ProblemInfeasibleError
            When b is not in the range of A: translated from the ``NewtonStepError`` the
            saddle solve raises, or detected by the relative residual test
               ‖A x - b‖ > 1e-8 max(1, ‖b‖).

        """
        projection_start = time.time()
        M = self.constraint.dimension
        hessian = DiagonalSystem(2.0 * np.ones(M))
        b_top = 2.0 * x0 if x0 is not None else np.zeros(M)
        try:
            x, _ = self.constraint.saddle_system(hessian).solve(b_top, self.b)
        except NewtonStepError as e:
            raise ProblemInfeasibleError("b was not in the range of A.") from e
        # The Schur solve only flags infeasibility when its Cholesky falls
        # through to the SVD range test; a borderline-singular Schur complement
        # can slip past Cholesky with a bogus multiplier. Verify feasibility
        # directly and unconditionally.
        residual = float(np.linalg.norm(self.A @ x - self.b))
        if residual > 1e-8 * max(1.0, float(np.linalg.norm(self.b))):
            raise ProblemInfeasibleError("b was not in the range of A.")
        # on_svd_done is reused as a generic "projection done" bucket; no SVD
        # is formed in this path.
        self.settings.trace.on_svd_done(1000 * (time.time() - projection_start))
        return SolveResult(
            solution=x,
            status=SolveStatus.FEASIBLE,
            message="",
            solver="",
            n_iterations=0,
        )

    def svd_A(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Return the thin SVD of A, cached on the constraint layer.

        Returns
        -------
         U : npt.NDArray[np.float64]
            Left singular vectors, of shape (p, k), with k = min(p, M).
         s : npt.NDArray[np.float64]
            Singular values in descending order, of length k.
         Vh : npt.NDArray[np.float64]
            Transposed right singular vectors, of shape (k, M).

        Raises
        ------
         NotImplementedError
            When A is sparse. Propagated from ``EqualityConstraintLayer.svd_A``.

        """
        return self.constraint.svd_A()

    def initialize_barrier_parameter(
        self,
        gradient: npt.NDArray[np.float64],
        grad_phi: npt.NDArray[np.float64],
        num_ineq: int,
        outer_tol: float,
        super_init: float,
    ) -> float:
        """Compute an equality-aware initial barrier parameter t0.

        Serves any barrier solver that holds an ``EqualitySolver``; ``BarrierSolver``
        calls it under its ``EQUALITY_SCHUR`` strategy, the default when equality
        constraints are present.

        Following (Boyd and Vandenberghe, 2004, section 11.3.1), it picks the t that
        best centers x0, the least-squares solution over t and ν of
           minimize ‖t ∇f0(x0) + ∇φ(x0) + A^T ν‖,
        where φ is the log barrier. Eliminating ν projects onto the null space of A
        through
           P = I - A^T (A A^T)^{-1} A,
        which gives
           t_ls = -∇f0^T P ∇φ / (∇f0^T P ∇f0).
        The result is
           t0 = min(m / ε, max(super_init, t_ls)),
        where the cap m / ε is the value at which the duality-gap bound m / t already
        meets the outer tolerance ε.

        Parameters
        ----------
         gradient : npt.NDArray[np.float64]
            Objective gradient ∇f0(x0) at the initial iterate, of length M.
         grad_phi : npt.NDArray[np.float64]
            Barrier gradient ∇φ(x0) at the initial iterate, of length M.
         num_ineq : int
            Number of inequality constraints, m.
         outer_tol : float
            Outer duality-gap tolerance, ε.
         super_init : float
            Fallback value and lower bound for t0, from the caller's equality-unaware
            initializer.

        Returns
        -------
         t0 : float
            The initial barrier parameter.

        Raises
        ------
         NotImplementedError
            When A is sparse. Propagated from ``svd_A``.

        Notes
        -----
        (A A^T)^{-1} is applied through the same Tikhonov-damped SVD inverse as
        ``solve``. The denominator ∇f0^T P ∇f0, a Schur complement, is floored at 1e-16,
        so a gradient in the row space of A yields a huge t_ls that the cap then bounds.
        When t_ls is not finite (an extreme scale of A or of the gradients), super_init
        replaces it, so a non-finite t never reaches the barrier loop.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        A_grad = self.A @ gradient
        A_grad_phi = self.A @ grad_phi
        U, s, _ = self.svd_A()
        # Tikhonov-damped ``(A A^T)^{-1}`` (see
        # ``numerical_helpers.tikhonov_filter``, ``power == 2``). Damping bounds
        # the ``1 / s^2`` amplification, so the same relative floor as the
        # projection path is safe here, and no equality information is dropped
        # for a merely poorly-scaled ``A``.
        filt_aat, _ = tikhonov_filter(s, power=2)
        z_phi = U @ (filt_aat * (U.T @ A_grad_phi))
        z_grad = U @ (filt_aat * (U.T @ A_grad))
        schur_complement = float(np.dot(gradient, gradient) - np.dot(A_grad, z_grad))
        t_star_s = float(-np.dot(gradient, grad_phi) + np.dot(A_grad, z_phi))
        # Overflow guard: the Tikhonov filter bounds the reciprocal, but a
        # pathological ``A`` / gradient scale can still drive the dot products
        # into ``schur_complement`` / ``t_star_s`` to ``+/- inf`` (or ``nan``
        # from ``inf / inf`` cancellation in the schur term). Falling
        # through to ``min(num_ineq/outer_tol, max(super_init, t_candidate))``
        # then poisons the IPM with a non-finite ``t``. Detect the
        # non-finite case at the cause site and fall back to ``super_init``,
        # the inequality-only Schur initializer the caller already supplied.
        t_candidate = t_star_s / max(schur_complement, 1e-16)
        if not np.isfinite(t_candidate):
            t_candidate = super_init
        return min(num_ineq / outer_tol, max(super_init, t_candidate))
