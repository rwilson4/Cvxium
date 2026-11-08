"""Base optimizer classes and settings for Cvxium's barrier interior-point method.

Defines the class hierarchy that Cvxium's barrier solvers build on:
- ``Optimizer``: the abstract base, holding settings and an optional Phase I solver.
- ``FeasibilitySolver``: an optimizer that finds a feasible point;
  ``TrivialFeasibilitySolver`` is the pass-through placeholder.
- ``BaseInteriorPointMethodSolver``: the barrier method itself. ``solve`` runs the outer
  loop over the barrier parameter t, tracking the best primal and dual bounds;
  ``centering_step`` runs Newton's method with backtracking line search; subclasses
  supply the objective, constraints, Hessian products and Newton step.
- ``InteriorPointMethodSolver`` and ``FeasibilityInteriorPointSolver``: the optimization
  and feasibility variants.
- ``UnconstrainedNewtonSolver`` and ``EqualityConstrainedNewtonSolver``: Newton's method
  without inequality constraints, as a single centering step at t = 1.

It also defines the Phase I infeasibility errors and the private helpers that turn
centering-step outcomes into ``SolveResult`` objects. The solver settings live in
``settings``.

"""

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ..exceptions import (
    BacktrackingLineSearchError,
    CenteringStepError,
    ConstraintBoundaryError,
    InteriorPointMethodError,
    NewtonStepError,
    ProblemInfeasibleError,
)
from ..problems.result import (
    BarrierDiagnostics,
    Duals,
    ObjectiveOutcome,
    SolveResult,
    SolveStatus,
)
from ..solvers.line_search import (
    ArmijoBarrierMerit,
    LineSearch,
)
from ..solvers.settings import (
    OptimizationSettings,
)

logger: logging.Logger = logging.getLogger("cvxium.optimization")


def _configure_verbose_logging() -> None:
    """Attach a console handler to the ``cvxium`` logger unless one is already present.

    Called when an optimizer or ``EqualitySolver`` is constructed with
    ``observability.verbose`` set. Adds a ``logging.StreamHandler`` that writes bare
    messages (no timestamp or level prefix) to stderr, and sets the ``cvxium`` logger
    level to ``DEBUG``, so both ``INFO`` records (centering steps) and ``DEBUG`` records
    (Newton steps, line search) are shown.

    A no-op when the ``cvxium`` logger already has a ``logging.StreamHandler`` other
    than a ``NullHandler`` (a ``FileHandler`` counts, being a ``StreamHandler``
    subclass): the application has then configured logging itself. A handler of another
    type, such as an ``IterationHandler``, does not count, so the console handler is
    still added.

    """
    parent = logging.getLogger("cvxium")
    has_real_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in parent.handlers
    )
    if not has_real_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        parent.addHandler(handler)
        parent.setLevel(logging.DEBUG)


def _legacy_solve_status(status: Literal[0, 1, 2]) -> SolveStatus:
    """Map an integer status code onto a ``SolveStatus``.

    The centering step and the barrier loop report status as 0 (converged), 1
    (acceptable at the soft tolerance) or 2 (best-effort iterate). Code 2 maps to
    ``MAX_ITERATIONS``; solvers that stop early at a feasible point set ``FEASIBLE``
    themselves.

    Parameters
    ----------
     status : Literal[0, 1, 2]
        The integer status code.

    Returns
    -------
     solve_status : SolveStatus
        ``OPTIMAL`` for 0, ``ACCEPTABLE`` for 1, and ``MAX_ITERATIONS`` for 2.

    """
    if status == 0:
        return SolveStatus.OPTIMAL
    if status == 1:
        return SolveStatus.ACCEPTABLE
    return SolveStatus.MAX_ITERATIONS


def _newton_solveresult(outcome: "_CenteringOutcome") -> SolveResult:
    """Convert a centering-step outcome into a Newton-solver ``SolveResult``.

    Used at the public boundary of the Newton solvers, which report an objective outcome
    and, when present, duals, but no per-step diagnostics.

    Parameters
    ----------
     outcome : _CenteringOutcome
        The outcome returned by ``centering_step``.

    Returns
    -------
     result : SolveResult
        The result, with solver name ``"newton"``. Its suboptimality is the last entry
        of ``outcome.suboptimalities``, or inf when the list is empty.

    """
    return SolveResult(
        solution=outcome.solution,
        status=_legacy_solve_status(outcome.status),
        message=outcome.message,
        solver="newton",
        n_iterations=outcome.nits,
        objective=ObjectiveOutcome(
            value=outcome.objective_value,
            suboptimality=(
                outcome.suboptimalities[-1] if outcome.suboptimalities else float("inf")
            ),
            dual_value=outcome.dual_value,
        ),
        duals=_duals_from(outcome.equality_multipliers, outcome.inequality_multipliers),
    )


def _barrier_solveresult(
    *,
    solution: npt.NDArray[np.float64],
    objective_value: float,
    dual_value: float,
    equality_multipliers: npt.NDArray[np.float64],
    inequality_multipliers: npt.NDArray[np.float64],
    suboptimality: float,
    duality_gaps: list[float],
    nits: int,
    inner_nits: list[int],
    inner_suboptimalities: list[list[float]],
    inner_btls_steps: list[list[float]],
    inner_quadratic_from: list[int | None],
    status: Literal[0, 1, 2],
    message: str,
    phase1: SolveResult | None,
) -> SolveResult:
    """Build a barrier-method ``SolveResult`` from the per-iteration trajectories.

    Centralizes the mapping that the barrier method's result sites share, and fills
    ``BarrierDiagnostics`` from the trajectories, so no per-iteration information is
    lost.

    Parameters
    ----------
     solution : npt.NDArray[np.float64]
        The final iterate.
     objective_value : float
        The objective value at the solution.
     dual_value : float
        The Lagrange dual value certifying the solution.
     equality_multipliers : npt.NDArray[np.float64]
        The equality multipliers ν; empty when there are none.
     inequality_multipliers : npt.NDArray[np.float64]
        The inequality multipliers λ; empty when there are none.
     suboptimality : float
        The final duality gap.
     duality_gaps : list[float]
        The duality gap after each outer iteration.
     nits : int
        The number of outer iterations.
     inner_nits : list[int]
        The number of Newton iterations in each centering step.
     inner_suboptimalities : list[list[float]]
        The suboptimality at each Newton iteration, per centering step.
     inner_btls_steps : list[list[float]]
        The accepted line-search step at each Newton iteration, per centering step.
     inner_quadratic_from : list[int | None]
        For each centering step, the Newton iteration at which the quadratic phase
        began, or ``None`` if it never did.
     status : Literal[0, 1, 2]
        The integer status code (see ``_legacy_solve_status``).
     message : str
        Summary of the outcome.
     phase1 : SolveResult | None
        The Phase I result, when a Phase I solve ran.

    Returns
    -------
     result : SolveResult
        The result, with solver name ``"barrier"`` and populated ``BarrierDiagnostics``.

    """
    return SolveResult(
        solution=solution,
        status=_legacy_solve_status(status),
        message=message,
        solver="barrier",
        n_iterations=nits,
        objective=ObjectiveOutcome(
            value=objective_value,
            suboptimality=suboptimality,
            dual_value=dual_value,
        ),
        duals=_duals_from(equality_multipliers, inequality_multipliers),
        diagnostics=BarrierDiagnostics(
            duality_gaps=tuple(duality_gaps),
            inner_nits=tuple(inner_nits),
            inner_suboptimalities=tuple(
                tuple(subopts) for subopts in inner_suboptimalities
            ),
            inner_btls_steps=tuple(tuple(steps) for steps in inner_btls_steps),
            inner_quadratic_from=tuple(inner_quadratic_from),
        ),
        phase1=phase1,
    )


def _duals_from(
    equality_multipliers: npt.NDArray[np.float64],
    inequality_multipliers: npt.NDArray[np.float64],
) -> Duals | None:
    """Build a ``Duals`` from multiplier arrays, treating an empty array as absent.

    Parameters
    ----------
     equality_multipliers : npt.NDArray[np.float64]
        The equality multipliers ν, possibly empty.
     inequality_multipliers : npt.NDArray[np.float64]
        The inequality multipliers λ, possibly empty.

    Returns
    -------
     duals : Duals | None
        The multipliers, with each empty array replaced by ``None``; ``None`` when both
        are empty.

    """
    eq = equality_multipliers if equality_multipliers.size > 0 else None
    ineq = inequality_multipliers if inequality_multipliers.size > 0 else None
    if eq is None and ineq is None:
        return None
    return Duals(equality_multipliers=eq, inequality_multipliers=ineq)


@dataclass
class _CenteringOutcome:
    """Private record of one centering step's outcome.

    Returned by ``centering_step`` and read directly, by attribute name, by the barrier
    outer loop, ``_update_incumbent_bounds`` and ``_result_from_best_bounds``.
    ``SolveResult`` cannot hold the per-iteration suboptimality list (it carries a
    single suboptimality) or the barrier objective value, so the centering step produces
    this richer record and the public solver boundary converts it to a ``SolveResult``.

    Fields
    ------
     solution : npt.NDArray[np.float64]
        The final iterate.
     objective_value : float
        The objective value f0 at the solution.
     barrier_objective_value : float
        The barrier objective f_t at the solution.
     dual_value : float
        The Lagrange dual value at the multipliers implied by the solution.
     equality_multipliers : npt.NDArray[np.float64]
        The equality multipliers ν; empty when there are none.
     inequality_multipliers : npt.NDArray[np.float64]
        The inequality multipliers λ; empty when there are none.
     suboptimalities : list[float]
        The suboptimality at each Newton iteration.
     nits : int
        The number of Newton iterations.
     status : Literal[0, 1, 2]
        The integer status code (see ``_legacy_solve_status``).
     message : str
        Summary of the outcome.
     btls_steps : list[float]
        The accepted line-search step at each Newton iteration. Empty by default, since
        some error paths do not record the trajectory.
     quadratic_from : int | None
        The index into ``btls_steps`` at which the quadratic phase began, or ``None``
        (the default) if it never did.

    """

    solution: npt.NDArray[np.float64]
    objective_value: float
    barrier_objective_value: float
    dual_value: float
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimalities: list[float]
    nits: int
    status: Literal[0, 1, 2]
    message: str
    # Accepted step modifier per inner Newton iteration, and the index within
    # that list at which the quadratic phase began (None if never reached).
    # Defaulted because some return sites are error paths where the trajectory
    # is incidental.
    btls_steps: list[float] = field(default_factory=list)
    quadratic_from: int | None = None


class ProblemCertifiablyInfeasibleError(ProblemInfeasibleError):
    """Error raised when a problem is certifiably infeasible.

    Raised when a feasibility (Phase I) solve certifies that no feasible point exists,
    typically through a strictly positive dual value of the Phase I problem. Carries the
    best-effort result of the solve.

    Parameters
    ----------
     message : str
        Description of the failure, returned by ``str``.
     result : SolveResult
        The best-effort result of the solve.

    """

    def __init__(self, message: str, result: SolveResult) -> None:
        self.message = message
        self.result = result

    def __str__(self) -> str:
        """Return the error message.

        Returns
        -------
         message : str
            The message given at construction.

        """
        return self.message

    @property
    def solve_result(self) -> SolveResult:
        """Return the best-effort result of the solve.

        Returns
        -------
         result : SolveResult
            The result given at construction.

        """
        return self.result


class ProblemMarginallyFeasibleError(ProblemInfeasibleError):
    """Error raised when a problem may be infeasible but cannot be certified as such.

    Raised when a feasibility (Phase I) solve neither finds a strictly feasible point
    nor certifies infeasibility. Carries the best-effort result of the solve.

    Parameters
    ----------
     message : str
        Description of the failure, returned by ``str``.
     result : SolveResult
        The best-effort result of the solve.

    """

    def __init__(self, message: str, result: SolveResult) -> None:
        self.message = message
        self.result = result

    def __str__(self) -> str:
        """Return the error message.

        Returns
        -------
         message : str
            The message given at construction.

        """
        return self.message

    @property
    def solve_result(self) -> SolveResult:
        """Return the best-effort result of the solve.

        Returns
        -------
         result : SolveResult
            The result given at construction.

        """
        return self.result


class Optimizer(ABC):
    """Abstract base for Cvxium's iterative optimizers.

    Holds the algorithm settings and an optional Phase I solver, and declares the
    interface every optimizer shares: the constraint counts and ``solve``.

    Parameters
    ----------
     phase1_solver : FeasibilitySolver | None
        Solver that finds a feasible starting point when the initial guess is not
        feasible, or ``None`` when the optimizer needs none.
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults. When ``observability.verbose``
        is set, the constructor attaches a console handler to the ``cvxium`` logger.
     **kwargs : Any
        Accepted for cooperative construction and ignored.

    """

    def __init__(
        self,
        phase1_solver: "FeasibilitySolver | None" = None,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        self.phase1_solver = phase1_solver
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings
        if self.settings.observability.verbose:
            _configure_verbose_logging()

    @property
    @abstractmethod
    def num_eq_constraints(self) -> int:
        """Return the number of equality constraints.

        Returns
        -------
         p : int
            The number of equality constraints.

        """

    @property
    @abstractmethod
    def num_ineq_constraints(self) -> int:
        """Return the number of inequality constraints.

        Returns
        -------
         m : int
            The number of inequality constraints.

        """

    @abstractmethod
    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Solve the optimization problem.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess; ``None`` lets the implementation choose or require one.
         **kwargs : Any
            Implementation-specific options.

        Returns
        -------
         result : SolveResult
            The solution and the solve's diagnostics.

        """


class FeasibilitySolver(Optimizer):
    """Abstract optimizer that finds a feasible point rather than an optimal one.

    Serves as the Phase I solver of an ``Optimizer``. Its ``solve`` returns a feasible
    point, or raises when none is found.

    Parameters
    ----------
     phase1_solver : FeasibilitySolver | None
        Phase I solver of its own, or ``None``.
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults.
     **kwargs : Any
        Accepted for cooperative construction and ignored.

    """

    @abstractmethod
    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Find a feasible point.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess; ``None`` lets the implementation choose or require one.
         **kwargs : Any
            Implementation-specific options.

        Returns
        -------
         result : SolveResult
            The result, whose solution is the feasible point found.

        """


class TrivialFeasibilitySolver(FeasibilitySolver):
    """Pass-through feasibility solver for which every point is feasible.

    A placeholder Phase I solver for an optimizer whose feasibility is established
    elsewhere: ``BarrierSolver``'s internal adapter uses it, since ``BarrierSolver``
    finds a feasible point before constructing the adapter.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults.

    """

    def __init__(self, settings: OptimizationSettings | None = None) -> None:
        super().__init__(settings=settings)

    @property
    def num_eq_constraints(self) -> int:
        """Return zero, the number of equality constraints.

        Returns
        -------
         p : int
            Always 0.

        """
        return 0

    @property
    def num_ineq_constraints(self) -> int:
        """Return zero, the number of inequality constraints.

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
        """Return the initial guess unchanged, as a feasible point.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            The point to return. Required despite the ``None`` default.
         **kwargs : Any
            Ignored.

        Returns
        -------
         result : SolveResult
            A result with solution ``x0``, status ``FEASIBLE`` and zero iterations.

        Raises
        ------
         ValueError
            When ``x0`` is ``None``.

        """
        if x0 is None:
            raise ValueError("TrivialFeasibilitySolver requires an initial point x0.")
        return SolveResult(
            solution=x0,
            status=SolveStatus.FEASIBLE,
            message="",
            solver="",
            n_iterations=0,
        )


class BaseInteriorPointMethodSolver(Optimizer):
    """Base class for barrier interior-point solvers.

    Implements the barrier method (Boyd and Vandenberghe, 2004, section 11.3) for
       minimize    f0(x)
       subject to  f_i(x) ≤ 0,   i = 1, ..., m,
                   A x = b,
    by solving a sequence of centering problems
       minimize    f_t(x) = t f0(x) - ∑ log(-f_i(x))
       subject to  A x = b
    for increasing t. ``solve`` runs the outer loop: a Phase I solve for a strictly
    feasible start, then centering steps at t0, μ t0, μ^2 t0, and so on.
    ``centering_step`` runs the inner loop: Newton's method with backtracking line
    search.

    Subclasses supply the problem: ``evaluate_objective``, ``gradient``,
    ``constraints``, ``grad_constraints``, ``hessian_multiply``,
    ``calculate_newton_step`` (where structure is exploited), ``evaluate_dual``,
    ``is_feasible`` and the constraint counts. The same loop also serves as a Phase I
    solver: when ``_feasibility_threshold`` returns a value, ``solve`` stops at the
    first iterate ``is_feasible`` accepts.

    Parameters
    ----------
     phase1_solver : FeasibilitySolver | None
        Solver that turns the initial guess into a strictly feasible start. Required by
        ``solve``, which raises ``ValueError`` without one.
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults. When ``observability.verbose``
        is set, the constructor attaches a console handler to the ``cvxium`` logger.
     name : str | None
        Human-readable identifier used in log messages and trace events (for example,
        "Starting IPM (<name>)"). ``None`` falls back to
        ``settings.observability.label`` when set, and otherwise to the concrete class
        name.
     **kwargs : Any
        Accepted for cooperative construction and ignored.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    def __init__(
        self,
        phase1_solver: "FeasibilitySolver | None" = None,
        settings: OptimizationSettings | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.phase1_solver = phase1_solver
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings
        if name is not None:
            resolved_name = name
        elif self.settings.observability.label is not None:
            resolved_name = self.settings.observability.label
        else:
            resolved_name = self.__class__.__name__
        self.name: str = resolved_name
        if self.settings.observability.verbose:
            _configure_verbose_logging()

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Solve the problem with the barrier method.

        Solves
           minimize    f0(x)
           subject to  f_i(x) ≤ 0,   i = 1, ..., m,
                       A x = b,
        starting from the strictly feasible point the Phase I solver returns.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess, passed to the Phase I solver. It need not be feasible; a
            guess feasible for some of the constraints lets Phase I focus on the rest.
         **kwargs : Any
            Forwarded to the Phase I solver. ``on_infeasible`` (``"raise"``, the
            default, or ``"continue"``) is also read here, in feasibility mode:
            ``"continue"`` defers an infeasibility certificate until the schedule
            completes, so that the raised error carries the minimum-violation result
            rather than an intermediate iterate.

        Returns
        -------
         result : SolveResult
            The solution, with the objective, dual value, multipliers, duality gap and
            per-iteration ``BarrierDiagnostics``. In feasibility mode, the Phase I
            result itself when its point is already feasible.

        Raises
        ------
         ValueError
            When there is no Phase I solver.
         InteriorPointMethodError
            When no centering step completes, or when even the best primal and dual
            bounds miss ``outer.tolerance_soft``; in the latter case the best result is
            attached as ``result``. In feasibility mode, when a centering step fails
            while the surrogate gap still exceeds ``outer.tolerance_soft``.
         ProblemCertifiablyInfeasibleError
            In feasibility mode, when the dual certifies infeasibility; raised by
            ``check_for_infeasibility`` at once, or after the schedule under
            ``on_infeasible="continue"``.
         ProblemMarginallyFeasibleError
            In feasibility mode, when the schedule ends without a feasible point and
            without a certificate of infeasibility.
         ConstraintBoundaryError
            Propagated from ``btls_keep_feasible`` when a quadratic-phase Newton step
            cannot be kept strictly feasible (see ``centering_step``).

        Notes
        -----
        The outer loop runs
           N = ceil((log m - log(t0 ε)) / log μ) + 1
        centering steps, enough for the last one, at t = t0 μ^(N-1), to have m / t ≤ ε,
        where ε is ``outer.tolerance`` and μ is ``outer.barrier_multiplier``.

        Every barrier iterate is strictly feasible, so its objective value is an upper
        bound on the optimal value p_star; every set of barrier multipliers is
        nonnegative, so its dual value is a lower bound. The loop tracks the best of
        each independently, and the best pair may come from different steps. This
        matters because the per-step gap is not monotone: at high t the conditioning
        degrades and the dual value sags while the objective keeps improving. The loop
        stops early once the best-bounds gap is within ε, which often happens before the
        schedule's last step.

        The result's suboptimality is the smaller of the best-bounds gap and the barrier
        surrogate m / t of the step that produced the best objective; both bound the
        same suboptimality, and the surrogate is the only usable one when the dual is
        loose or -inf. Because the solution comes from the best-objective step and the
        multipliers from the best-dual step, the multipliers are valid dual bounds but
        need not be KKT-consistent with the solution, so they should not be used as
        exact Lagrange multipliers there (for example, for sensitivity analysis). A
        centering-step failure finalizes from the bounds gathered so far.

        In feasibility mode, the loop instead stops at the first strictly feasible
        iterate, and after each step checks the dual for a certificate of infeasibility.

        """
        if self.phase1_solver is None:
            raise ValueError("FeasibilitySolver not specified.")

        # A feasibility solve stops at the first strictly feasible iterate (Phase I)
        # rather than chasing the optimum. It is requested by the objective carrying
        # a feasibility threshold (``_feasibility_threshold()`` is not None); an
        # ordinary objective has none, so the loop runs to the optimum. The loop
        # branches on ``feasibility_mode`` throughout.
        feasibility_mode = self._feasibility_threshold() is not None

        # ``on_infeasible="continue"`` defers an infeasibility certificate so the
        # Phase I slack can be driven to its optimum (the minimum-violation point)
        # before the error is raised, attaching that best-effort result to the
        # exception. The default ``"raise"`` keeps the early-exit behavior. The
        # certificate is recorded here and acted on after the loop. Read from
        # ``kwargs`` (rather than an explicit parameter) so it keeps propagating to
        # nested Phase I solvers via the ``**kwargs`` forward below.
        on_infeasible = kwargs.get("on_infeasible", "raise")
        infeasibility_certified = False

        # The nested Phase I solver should return x such that A * x = b and fi(x) < 0,
        # i=1, ..., M.
        phase1_res = self.phase1_solver.solve(x0=x0, **kwargs)
        x = self.augment_previous_solution(phase1_res)

        # Applicable only for Phase I methods.
        if feasibility_mode and self.is_feasible(x):
            self.settings.trace.on_phase1_feasible(self.name)
            return phase1_res

        t = self.initialize_barrier_parameter(x0=x)
        num_steps = (
            int(
                np.ceil(
                    (
                        np.log(self.num_ineq_constraints)
                        - np.log(t * self.settings.outer.tolerance)
                    )
                    / np.log(self.settings.outer.barrier_multiplier)
                )
            )
            + 1
        )
        overall_start_time = time.time()
        self.settings.trace.on_ipm_start(self.name)

        inner_nits: list[int] = []
        inner_suboptimalities: list[list[float]] = []
        inner_btls_steps: list[list[float]] = []
        inner_quadratic_from: list[int | None] = []
        duality_gaps: list[float] = []
        status: Literal[0, 1, 2] = 0
        message = (
            "Interior Point Method completed successfully to the desired tolerance"
        )
        equality_multipliers = np.zeros(1)
        inequality_multipliers = np.zeros(1)
        nit = 0
        result: _CenteringOutcome | None = None
        # Track the best primal and best dual bounds *independently* across
        # centering steps. Every barrier iterate is strictly feasible, so each
        # ``objective_value`` is a valid upper bound on p*; every barrier
        # multiplier is >= 0, so each ``dual_value`` is a valid lower bound. The
        # tightest certificate from the run is therefore ``min(objective) -
        # max(dual)``, which can pair a primal and a dual from *different* steps.
        # This matters because the gap is not monotone: at high ``t`` the
        # conditioning degrades and the dual sags even while the primal keeps
        # improving, so the last (or single-best-gap) step is not the best
        # certificate. ``min``/``max`` self-select the trustworthy steps, so no
        # ``status`` gating is needed.
        #
        # ``best_primal_subopt`` is the barrier surrogate bound ``m / t`` (B&V
        # 11.3) for the central point that produced ``best_primal``, evaluated at
        # that step's ``t``. The reported suboptimality is
        # the *tighter* of the reconstructed gap (``best_primal.objective -
        # best_dual.dual``) and this surrogate -- both are valid upper bounds on
        # the same suboptimality, and the reconstructed dual can be loose or even
        # ``-inf`` (e.g. degenerate LP duals) where the surrogate is the only
        # usable bound, while the reconstructed gap is tighter when the dual is
        # well-behaved (e.g. a smooth, well-conditioned objective).
        best_primal: _CenteringOutcome | None = None
        best_primal_subopt: float = float("inf")
        best_dual: _CenteringOutcome | None = None
        for nit in range(num_steps):
            step_start_time = time.time()
            self.settings.trace.on_centering_start(nit + 1, t)

            try:
                result = self.centering_step(
                    x,
                    t,
                    last_step=(nit + 1 == num_steps),
                    outer_nit=nit + 1,
                )
            except CenteringStepError as e:
                # A centering step failed: finalize from the best bounds gathered
                # so far (or raise with them attached). Only meaningful for the
                # optimization path; Phase I keeps its own feasibility handling.
                if not feasibility_mode:
                    return self._result_from_best_bounds(
                        best_primal=best_primal,
                        best_primal_subopt=best_primal_subopt,
                        best_dual=best_dual,
                        nit=nit,
                        num_steps=num_steps,
                        duality_gaps=duality_gaps,
                        inner_nits=inner_nits,
                        inner_suboptimalities=inner_suboptimalities,
                        inner_btls_steps=inner_btls_steps,
                        inner_quadratic_from=inner_quadratic_from,
                        phase1_res=phase1_res,
                        fallback_iterate=x,
                        cause=e,
                    )
                x, status, message = self._accept_phase1_partial_or_raise(
                    e, nit=nit, num_steps=num_steps, t=t, duality_gaps=duality_gaps
                )
                break

            step_elapsed = 1000 * (time.time() - step_start_time)
            step_duality_gap = result.objective_value - result.dual_value
            self.settings.trace.on_centering_done(
                nit + 1, step_elapsed, step_duality_gap
            )

            x = result.solution
            equality_multipliers = result.equality_multipliers
            inequality_multipliers = result.inequality_multipliers
            inner_nits.append(result.nits)
            inner_suboptimalities.append(result.suboptimalities)
            inner_btls_steps.append(result.btls_steps)
            inner_quadratic_from.append(result.quadratic_from)
            best_primal, best_primal_subopt, best_dual = self._update_incumbent_bounds(
                result, t, best_primal, best_primal_subopt, best_dual
            )
            # Record the running-best certificate, not the raw per-step gap. Each
            # barrier iterate is strictly feasible (its objective is a valid upper
            # bound on p*) and each multiplier is >= 0 (its dual is a valid lower
            # bound), so ``min(primal) - max(dual)`` through this step -- floored at
            # the B&V 11.3 surrogate ``m / t`` -- is the honest gap achieved so far,
            # monotone non-increasing. The raw per-step gap is non-monotone and
            # inflates in the ill-conditioned high-``t`` regime even while the
            # certificate holds, so ``[-1]`` of a raw trace is not a trustworthy
            # convergence signal. This keeps ``duality_gaps`` consistent with the
            # best-bounds certificate the solve ultimately reports (``suboptimality``
            # / ``dual_value`` via ``_result_from_best_bounds``).
            duality_gaps.append(
                max(
                    0.0,
                    min(
                        best_primal.objective_value - best_dual.dual_value,
                        best_primal_subopt,
                    ),
                )
            )

            # Optimization early-stop: B&V 11.3.1 sizes ``num_steps`` to drive the
            # *surrogate* gap ``m / t`` below ``tolerance``, but the actual
            # reconstructed certificate ``best_primal.objective - best_dual.dual``
            # converges far sooner. Marching ``t`` past that point only enters the
            # ill-conditioned high-``t`` regime, where the dual sags and the gap
            # inflates (the non-monotone gap). Stop on the honest certificate once
            # it is within ``tolerance``. Phase I has its own feasibility-based
            # early-stop below.
            if not feasibility_mode:
                if best_primal is None or best_dual is None:
                    # Unreachable in practice -- the incumbent-bounds update just
                    # above sets both from the completed centering ``result``.
                    # Raise (rather than ``assert``, which ``python -O`` strips)
                    # so a broken invariant surfaces explicitly instead of a later
                    # ``NoneType`` attribute error.
                    raise InteriorPointMethodError(
                        message=(
                            "Interior point method reached the convergence check "
                            "without recorded primal/dual bounds"
                        ),
                        remaining_steps=num_steps - nit - 1,
                        suboptimality=float("inf"),
                        last_iterate=x,
                    )
                best_bounds_gap = max(
                    0.0, best_primal.objective_value - best_dual.dual_value
                )
                if best_bounds_gap <= self.settings.outer.tolerance:
                    self.settings.trace.on_converged(nit + 1, best_bounds_gap)
                    break

            # Applicable only for Phase I methods.
            if feasibility_mode and self.is_feasible(x):
                status = 2
                message = "Feasibility method successfully found a feasible point"
                self.settings.trace.on_early_stop(nit + 1)
                break

            # Dual can provide certificate of infeasibility, in which case we can quit
            # faster. Applicable only for Phase I methods.
            if feasibility_mode:
                ipm_result = _barrier_solveresult(
                    solution=self.finalize_solution(x),
                    objective_value=self.evaluate_objective(x),
                    dual_value=self.evaluate_dual(
                        lmbda=inequality_multipliers,
                        nu=equality_multipliers,
                        x_star=x,
                    ),
                    equality_multipliers=equality_multipliers,
                    inequality_multipliers=inequality_multipliers,
                    suboptimality=self.num_ineq_constraints
                    * self.settings.outer.barrier_multiplier
                    / t,
                    duality_gaps=list(duality_gaps),
                    nits=nit + 1,
                    inner_nits=list(inner_nits),
                    inner_suboptimalities=list(inner_suboptimalities),
                    inner_btls_steps=list(inner_btls_steps),
                    inner_quadratic_from=list(inner_quadratic_from),
                    status=0,
                    message="Intermediate result at infeasibility check",
                    phase1=phase1_res,
                )
                if on_infeasible == "continue":
                    # Defer the raise: keep driving the slack to its minimum so the
                    # eventual error carries the best-effort (minimum-violation)
                    # result rather than this intermediate iterate. Just record that
                    # the dual has certified infeasibility; the post-loop block
                    # raises ``ProblemCertifiablyInfeasibleError`` with the optimized
                    # result once the schedule completes.
                    if self._is_infeasible_certified(ipm_result):
                        infeasibility_certified = True
                else:
                    self.check_for_infeasibility(ipm_result)

            t *= self.settings.outer.barrier_multiplier

        overall_elapsed = 1000 * (time.time() - overall_start_time)
        self.settings.trace.on_ipm_done(self.name, overall_elapsed)

        # Applicable only for Phase I methods.
        if feasibility_mode and result is not None and not self.is_feasible(x):
            # The slack has been driven to its minimum over the full schedule (or to
            # the point where the dual certified infeasibility, under
            # ``on_infeasible="continue"``). This is the best-effort, minimum-
            # violation result attached to whichever infeasibility error we raise.
            best_effort = _barrier_solveresult(
                solution=self.finalize_solution(x),
                objective_value=self.evaluate_objective(x),
                dual_value=self.evaluate_dual(
                    lmbda=inequality_multipliers,
                    nu=equality_multipliers,
                    x_star=x,
                ),
                equality_multipliers=equality_multipliers,
                inequality_multipliers=inequality_multipliers,
                suboptimality=self.num_ineq_constraints
                * self.settings.outer.barrier_multiplier
                / t,
                duality_gaps=list(duality_gaps),
                nits=nit + 1,
                inner_nits=list(inner_nits),
                inner_suboptimalities=list(inner_suboptimalities),
                inner_btls_steps=list(inner_btls_steps),
                inner_quadratic_from=list(inner_quadratic_from),
                status=0,
                message="Problem may be infeasible",
                phase1=phase1_res,
            )
            # ``infeasibility_certified`` is only ever set under
            # ``on_infeasible="continue"`` (the ``"raise"`` path exits at the
            # in-loop certificate check above), so the default behavior is unchanged.
            if infeasibility_certified:
                raise ProblemCertifiablyInfeasibleError(
                    message=(
                        "Problem certifiably infeasible: the dual certified "
                        "infeasibility and the Phase I slack was minimized to "
                        f"{best_effort.objective_value:.3e}."
                    ),
                    result=best_effort,
                )
            raise ProblemMarginallyFeasibleError(
                message=(
                    "Problem may be infeasible: dual value was "
                    f"{result.dual_value} < 0, but last centering step resulted in "
                    f"a point with value {result.objective_value}"
                ),
                result=best_effort,
            )

        # Optimization path: finalize from the best primal/dual bounds gathered
        # across centering steps (the gap is non-monotone at high t, so the last
        # step is not necessarily the best). Reports the actual gap and, when even
        # the best bounds miss tolerance_soft, raises with the best result
        # attached. Phase I keeps its own feasibility-oriented assembly below.
        if not feasibility_mode:
            return self._result_from_best_bounds(
                best_primal=best_primal,
                best_primal_subopt=best_primal_subopt,
                best_dual=best_dual,
                nit=nit,
                num_steps=num_steps,
                duality_gaps=duality_gaps,
                inner_nits=inner_nits,
                inner_suboptimalities=inner_suboptimalities,
                inner_btls_steps=inner_btls_steps,
                inner_quadratic_from=inner_quadratic_from,
                phase1_res=phase1_res,
                fallback_iterate=x,
                cause=None,
            )

        obj_value = self.evaluate_objective(x)
        sub = self.num_ineq_constraints * self.settings.outer.barrier_multiplier / t
        try:
            dual_value = self.evaluate_dual(
                lmbda=inequality_multipliers,
                nu=equality_multipliers,
                x_star=x,
            )
        except (ValueError, TypeError) as e:
            logger.debug("Dual evaluation failed, using bound: %s", e)
            dual_value = obj_value - sub

        return _barrier_solveresult(
            solution=self.finalize_solution(x),
            objective_value=obj_value,
            dual_value=dual_value,
            equality_multipliers=equality_multipliers,
            inequality_multipliers=inequality_multipliers,
            suboptimality=sub,
            duality_gaps=duality_gaps,
            nits=nit + 1,
            inner_nits=inner_nits,
            inner_suboptimalities=inner_suboptimalities,
            inner_btls_steps=inner_btls_steps,
            inner_quadratic_from=inner_quadratic_from,
            status=status,
            message=message,
            phase1=phase1_res,
        )

    def _feasibility_threshold(self) -> float | None:
        """Return the objective's feasibility threshold, or None for an ordinary solve.

        A non-``None`` value puts ``solve`` and ``centering_step`` into feasibility
        mode: the solver is answering a feasibility question, so it stops at the first
        iterate ``is_feasible`` accepts instead of driving the objective to its minimum.
        The value itself is not compared here; ``is_feasible`` decides. The base returns
        ``None``; adapters that carry a ``Problem`` override this to surface the
        threshold of the problem's objective.

        Returns
        -------
         threshold : float | None
            The feasibility threshold, or ``None`` to disable feasibility mode.

        """
        return None

    def _update_incumbent_bounds(
        self,
        result: _CenteringOutcome,
        t: float,
        best_primal: "_CenteringOutcome | None",
        best_primal_subopt: float,
        best_dual: "_CenteringOutcome | None",
    ) -> "tuple[_CenteringOutcome, float, _CenteringOutcome]":
        """Fold a completed centering step into the best primal and dual bounds.

        Keeps the step with the lowest objective value (the best upper bound on p_star)
        and, independently, the step with the highest dual value (the best lower bound);
        they may be different steps. A ``nan`` dual value certifies nothing, so it is
        replaced by -inf before the comparison; left as ``nan``, it would win the first
        comparison, never be displaced, and make the reported gap read as 0.

        Parameters
        ----------
         result : _CenteringOutcome
            The completed step.
         t : float
            The barrier parameter of that step.
         best_primal : _CenteringOutcome | None
            The incumbent lowest-objective step, or ``None`` before the first step.
         best_primal_subopt : float
            The surrogate gap m / t of ``best_primal``'s step.
         best_dual : _CenteringOutcome | None
            The incumbent highest-dual step, or ``None`` before the first step.

        Returns
        -------
         best_primal : _CenteringOutcome
            The updated lowest-objective step.
         best_primal_subopt : float
            The surrogate gap m / t of that step, using its own t.
         best_dual : _CenteringOutcome
            The updated highest-dual step.

        """
        step_surrogate = self.num_ineq_constraints / t
        # A NaN dual certifies nothing. Comparisons against NaN are all False, so
        # left as NaN it would become the incumbent on the first step and stay
        # there, and the gap `max(0, min(nan, ...))` it feeds reads as 0.0 -- a
        # converged solve.
        if math.isnan(result.dual_value):
            result = replace(result, dual_value=-math.inf)
        if best_primal is None or result.objective_value < best_primal.objective_value:
            best_primal = result
            best_primal_subopt = step_surrogate
        if best_dual is None or result.dual_value > best_dual.dual_value:
            best_dual = result
        return best_primal, best_primal_subopt, best_dual

    def _result_from_best_bounds(
        self,
        best_primal: "_CenteringOutcome | None",
        best_primal_subopt: float,
        best_dual: "_CenteringOutcome | None",
        nit: int,
        num_steps: int,
        duality_gaps: list[float],
        inner_nits: list[int],
        inner_suboptimalities: list[list[float]],
        inner_btls_steps: list[list[float]],
        inner_quadratic_from: list[int | None],
        phase1_res: SolveResult,
        fallback_iterate: npt.NDArray[np.float64],
        cause: CenteringStepError | None = None,
    ) -> SolveResult:
        """Assemble the optimization result from the best primal and dual bounds.

        Finalizes an optimization (not feasibility) solve, both on normal completion
        (``cause`` is ``None``) and after a centering-step failure. The result pairs the
        lowest-objective step's solution with the highest-dual step's dual value and
        multipliers; see the Notes of ``solve`` for why, and for the caveat that the
        multipliers need not be KKT-consistent with the solution.

        The reported suboptimality is
           max(0, min(f0(x_best) - g_best, s_best)),
        the smaller of the best-bounds gap and the surrogate ``best_primal_subopt``. The
        status is 0 when it is within ``outer.tolerance``, 1 when within
        ``outer.tolerance_soft``, and 2 otherwise, in which case the result (with a
        failing message) is attached to a raised ``InteriorPointMethodError`` so the
        caller can inspect it or re-run with a looser tolerance.

        Parameters
        ----------
         best_primal : _CenteringOutcome | None
            The lowest-objective step, or ``None`` when no step completed.
         best_primal_subopt : float
            The surrogate gap m / t of that step.
         best_dual : _CenteringOutcome | None
            The highest-dual step, or ``None`` when no step completed.
         nit : int
            Zero-based index of the last outer iteration reached.
         num_steps : int
            The planned number of outer iterations.
         duality_gaps : list[float]
            The running best-bounds gap after each completed step.
         inner_nits : list[int]
            Newton iterations per completed step.
         inner_suboptimalities : list[list[float]]
            Per-iteration suboptimalities, per completed step.
         inner_btls_steps : list[list[float]]
            Per-iteration accepted line-search steps, per completed step.
         inner_quadratic_from : list[int | None]
            Start of the quadratic phase, per completed step.
         phase1_res : SolveResult
            The Phase I result, attached to the returned result.
         fallback_iterate : npt.NDArray[np.float64]
            Iterate reported when no step completed and there is no ``cause``.
         cause : CenteringStepError | None
            The centering-step failure that triggered finalization, or ``None``.

        Returns
        -------
         result : SolveResult
            The result, with status ``OPTIMAL`` or ``ACCEPTABLE``.

        Raises
        ------
         InteriorPointMethodError
            When no step completed, or when the suboptimality exceeds
            ``outer.tolerance_soft`` (with the result attached). Chained from ``cause``
            when there is one.

        """
        if best_primal is None or best_dual is None:
            # No centering step ever completed: no certified bound to report.
            # Fall back to the failed step's iterate when there was one, else the
            # ``fallback_iterate`` (the post-Phase-I feasible start) -- friendlier
            # than an empty array for the (likely unreachable) num_steps == 0 case.
            raise InteriorPointMethodError(
                message=(
                    "Interior point method failed before any centering step "
                    f"completed{self._cause_suffix(cause)}"
                ),
                remaining_steps=num_steps - nit - 1,
                suboptimality=float("inf"),
                last_iterate=(
                    cause.last_iterate if cause is not None else fallback_iterate
                ),
            ) from cause

        # Suboptimality of ``best_primal`` is bounded by BOTH the reconstructed
        # gap ``objective - best_dual.dual`` (weak duality) and the barrier
        # surrogate ``best_primal_subopt`` (B&V 11.3) -- take the tighter. The
        # reconstructed gap wins when the dual is well-behaved (tighter, honest);
        # the surrogate wins when the dual is loose or ``-inf`` (e.g. LP). Clamp
        # tiny floating-point negatives.
        gap = max(
            0.0,
            min(
                best_primal.objective_value - best_dual.dual_value,
                best_primal_subopt,
            ),
        )
        # Status ladder: 0 within ``tolerance``, 1 within ``tolerance_soft``, 2
        # when even the best bounds miss ``tolerance_soft``. The status-2 result
        # is attached to the raised error below, so it must report the failure
        # honestly -- callers inspecting ``exc.result.status`` / ``.message``
        # should never see an "acceptable tolerance" claim on a result that
        # failed the hard requirement.
        outer = self.settings.outer
        if gap <= outer.tolerance:
            status: Literal[0, 1, 2] = 0
            message = (
                "Interior Point Method converged to the desired tolerance "
                f"(duality gap {gap:.3g})."
            )
        elif gap <= outer.tolerance_soft:
            status = 1
            message = (
                "Interior Point Method reached an acceptable tolerance "
                f"(duality gap {gap:.3g})."
            )
        else:
            status = 2
            message = self._gap_failure_message(gap, best_primal.objective_value, cause)
        result = _barrier_solveresult(
            # The solution and the reported dual / multipliers may come from
            # different centering steps -- each is the tightest valid bound found.
            solution=self.finalize_solution(best_primal.solution),
            objective_value=best_primal.objective_value,
            dual_value=best_dual.dual_value,
            equality_multipliers=best_dual.equality_multipliers,
            inequality_multipliers=best_dual.inequality_multipliers,
            suboptimality=gap,
            duality_gaps=list(duality_gaps),
            nits=nit + 1,
            inner_nits=list(inner_nits),
            inner_suboptimalities=list(inner_suboptimalities),
            inner_btls_steps=list(inner_btls_steps),
            inner_quadratic_from=list(inner_quadratic_from),
            status=status,
            message=message,
            phase1=phase1_res,
        )
        if status != 2:
            return result

        # Even the tightest bounds do not bracket p* to the soft tolerance.
        # Attach the best result (status 2, failing message) so the caller can
        # recover it.
        raise InteriorPointMethodError(
            message=message,
            remaining_steps=num_steps - nit - 1,
            suboptimality=gap,
            last_iterate=best_primal.solution,
            result=result,
        ) from cause

    @staticmethod
    def _cause_suffix(cause: CenteringStepError | None) -> str:
        """Format a centering-step failure as a suffix for an error message.

        Parameters
        ----------
         cause : CenteringStepError | None
            The failure, or ``None``.

        Returns
        -------
         suffix : str
            " -- " followed by the failure's message, or an empty string when ``cause``
            is ``None``.

        """
        return f" -- {cause}" if cause is not None else ""

    def _gap_failure_message(
        self,
        gap: float,
        objective_value: float,
        cause: CenteringStepError | None,
    ) -> str:
        """Build the failure message for a gap that exceeds the soft tolerance.

        Adds a relative-gap hint when ``outer.tolerance_rel`` is set and the gap
        relative to |f0| is within it: the barrier guarantee is absolute, but a gap
        negligible relative to the objective's scale usually means the caller is
        over-optimizing rather than that the answer is bad.

        Parameters
        ----------
         gap : float
            The best duality gap found.
         objective_value : float
            The objective value f0 at the best solution, for the relative gap.
         cause : CenteringStepError | None
            The centering-step failure that triggered finalization, or ``None``.

        Returns
        -------
         message : str
            The failure message.

        """
        soft = self.settings.outer.tolerance_soft
        base = (
            "Interior point method did not reach tolerance_soft "
            f"({soft:.3g}); the best duality gap found was {gap:.3g}."
        )
        tol_rel = self.settings.outer.tolerance_rel
        rel_gap = gap / abs(objective_value) if objective_value != 0.0 else float("inf")
        if tol_rel is not None and rel_gap <= tol_rel:
            base += (
                f" Relative to the objective the gap is negligible (relative gap "
                f"{rel_gap:.3g} <= tolerance_rel {tol_rel:.3g}) -- the attached "
                "`.result` is very likely good enough; inspect it before "
                "re-running, or simply raise outer.tolerance_soft."
            )
        else:
            base += (
                " The best result found is attached as `.result`; inspect it "
                "(objective and duality gap) or re-run with a larger "
                "outer.tolerance_soft."
            )
        return base + self._cause_suffix(cause)

    def _accept_phase1_partial_or_raise(
        self,
        e: CenteringStepError,
        nit: int,
        num_steps: int,
        t: float,
        duality_gaps: list[float],
    ) -> tuple[npt.NDArray[np.float64], Literal[0, 1, 2], str]:
        """Accept a failed Phase I centering step's last iterate, or raise.

        Used only in feasibility mode; the optimization path uses
        ``_result_from_best_bounds`` instead. When the surrogate gap m μ / t, at the
        failed step's t, is below ``outer.tolerance_soft``, accept the step's last
        iterate with status 1; otherwise raise. When the failure carries multipliers,
        also append the duality gap at the accepted iterate to ``duality_gaps``.

        Parameters
        ----------
         e : CenteringStepError
            The centering-step failure.
         nit : int
            Zero-based index of the failed outer iteration.
         num_steps : int
            The planned number of outer iterations.
         t : float
            The barrier parameter of the failed step.
         duality_gaps : list[float]
            The gap history; appended to in place.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The failed step's last iterate.
         status : Literal[0, 1, 2]
            Always 1 (acceptable).
         message : str
            Summary, including the failure's message.

        Raises
        ------
         InteriorPointMethodError
            When the surrogate gap is at least ``outer.tolerance_soft``; chained from
            ``e``.

        """
        suboptimality = (
            self.num_ineq_constraints * self.settings.outer.barrier_multiplier / t
        )
        if suboptimality >= self.settings.outer.tolerance_soft:
            raise InteriorPointMethodError(
                message="Centering step failed",
                remaining_steps=num_steps - nit - 1,
                suboptimality=suboptimality,
                last_iterate=e.last_iterate,
            ) from e

        x = e.last_iterate
        status: Literal[0, 1, 2] = 1
        message = (
            "Interior Point Method reached an acceptable precision but "
            f"then ran into numerical difficulties -- {e.__str__()}"
        )
        nu = e.equality_multipliers
        lmbda = e.inequality_multipliers
        if nu is not None and lmbda is not None:
            try:
                duality_gaps.append(
                    self.evaluate_objective(x)
                    - self.evaluate_dual(lmbda=lmbda, nu=nu, x_star=x)
                )
            except (ValueError, TypeError) as exc:
                logger.debug("Duality gap evaluation failed: %s", exc)
        return x, status, message

    def augment_previous_solution(
        self,
        phase1_res: SolveResult,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Build the starting iterate from the Phase I result.

        The base returns the Phase I solution unchanged. A subclass whose iterate
        differs from its Phase I solver's (for example, by an added variable) overrides
        this to convert.

        Parameters
        ----------
         phase1_res : SolveResult
            The Phase I result.
         **kwargs : Any
            Unused by the base.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The starting iterate.

        """
        return phase1_res.solution

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert an iterate into the solution reported to the caller.

        The inverse of ``augment_previous_solution``; the base returns the iterate
        unchanged.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The iterate.

        Returns
        -------
         solution : npt.NDArray[np.float64]
            The solution in the caller's variables.

        """
        return x

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        """Choose the initial barrier parameter t0.

        The base returns 1. A subclass can often skip early centering steps by starting
        higher (Boyd and Vandenberghe, 2004, section 11.3.1); see
        ``BarrierInitStrategy``.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            The strictly feasible starting point.

        Returns
        -------
         t0 : float
            The initial barrier parameter.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        return 1.0

    def centering_step(
        self,
        x0: npt.NDArray[np.float64],
        t: float,
        last_step: bool,
        outer_nit: int = 0,
        **kwargs: Any,
    ) -> _CenteringOutcome:
        """Solve one centering problem with Newton's method.

        Solves
           minimize    f_t(x) = t f0(x) - ∑ log(-f_i(x))
           subject to  A x = b
        from a strictly feasible start, by Newton's method with backtracking line
        search.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Starting point, strictly feasible; copied, not modified.
         t : float
            The barrier parameter.
         last_step : bool
            Whether this is the outer loop's last centering step. A line-search failure
            on the last step always raises, since no later step can make up the gap; on
            an earlier step it returns the current iterate with status 1 when its
            suboptimality is below ``inner.tolerance_soft``.
         outer_nit : int
            One-based index of the outer iteration, for trace events.
         **kwargs : Any
            Unused.

        Returns
        -------
         outcome : _CenteringOutcome
            The final iterate with its objective, barrier objective, dual value and
            multipliers. Status 0 when the suboptimality fell below ``inner.tolerance``;
            status 1 when the solve stopped early with an acceptable iterate (stalled,
            hit the iteration limit, hit a line-search failure, or, in feasibility mode,
            found a feasible point).

        Raises
        ------
         CenteringStepError
            When a Newton step cannot be computed (chained from the
            ``NewtonStepError``), when the line search fails and no acceptable iterate
            is available, or when ``inner.max_iterations`` is reached with the best
            suboptimality not below ``inner.tolerance_soft``.
         ConstraintBoundaryError
            Propagated from ``btls_keep_feasible`` in the quadratic phase, which is
            called outside the line search's error handling.

        Notes
        -----
        The suboptimality at each iteration is λ^2 / 2, the standard estimate of f_t(x)
        - inf f_t from the squared Newton decrement λ^2 (see
        ``newton_decrement_squared``). Once λ ≤ η, with
           η = (1 - 2α) / 4
        and α the Armijo coefficient (Boyd and Vandenberghe, 2004, section 9.6.4), the
        iteration is in the quadratic phase: the unit step satisfies the Armijo
        condition in exact arithmetic, so the line search is skipped and the step is
           s = min(1, β s_feas),
        where s_feas is the maximum feasible step from ``btls_keep_feasible``. This
        avoids the line search shrinking s to ``min_step`` at high t, where the change
        in f_t rounds to noise.

        The loop returns early in four ways besides convergence, each with status 1:
        1. Stall: in the quadratic phase, ``inner.stall_patience`` consecutive
           iterations without a relative improvement of ``inner.stall_rel_improvement``
           over the best suboptimality, with that best below ``inner.tolerance_soft``.
           Returns the best iterate.
        2. Line-search failure on a step other than the last, with the current
           suboptimality below ``inner.tolerance_soft``. Returns the current iterate.
        3. Feasibility mode: the next iterate is feasible. Returns it.
        4. Iteration limit, with the best suboptimality below ``inner.tolerance_soft``.
           Returns the best iterate.

        The multipliers are λ_i = -1 / (t f_i(x)) and ν = ν_hat / t, where ν_hat is the
        equality multiplier of the Newton system. On an early return, λ includes the
        first-order correction for imperfect centering (see ``inequality_multipliers``).

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        x = x0.copy()
        eta = 0.25 * (1.0 - 2 * self.settings.armijo.alpha)
        in_quadratic_phase = False
        btls_steps: list[float] = []
        quadratic_from: int | None = None
        suboptimality = np.inf
        suboptimalities = []
        best_x = x.copy()
        best_suboptimality = np.inf
        best_nu_hat: npt.NDArray[np.float64] | None = None
        best_delta_x: npt.NDArray[np.float64] | None = None
        iters_since_improvement = 0
        for nit in range(self.settings.inner.max_iterations):
            newton_start = time.time()

            try:
                delta_x, nu_hat = self.calculate_newton_step(x, t)
            except NewtonStepError as e:
                raise CenteringStepError(
                    message="Failed to calculate Newton step.",
                    suboptimality=suboptimality,
                    last_iterate=x,
                ) from e

            newton_elapsed = 1000 * (time.time() - newton_start)

            lambda_squared = self.newton_decrement_squared(x, t, delta_x)
            suboptimality = 0.5 * lambda_squared
            suboptimalities.append(suboptimality)

            lambda_norm = np.sqrt(lambda_squared)
            quadratic_threshold_crossed = not in_quadratic_phase and lambda_norm <= eta
            if quadratic_threshold_crossed:
                in_quadratic_phase = True

            # Track the best iterate seen, but reset the stall counter only on
            # *meaningful* improvement. A microscopic creep toward the FP floor
            # (each iterate a hair below the last, but never reaching
            # ``tolerance``) would otherwise reset the counter every iteration
            # and defeat the stall detector, burning the full
            # ``max_iterations`` budget. Compute the threshold against the
            # prior best, before it is updated below.
            meaningful_improvement = suboptimality < best_suboptimality * (
                1.0 - self.settings.inner.stall_rel_improvement
            )
            if suboptimality < best_suboptimality:
                best_x = x.copy()
                best_suboptimality = suboptimality
                # Copy (not alias) for symmetry with ``best_x``: guards
                # against a future ``calculate_newton_step`` that reuses a
                # scratch buffer for ``delta_x`` / ``nu_hat`` between
                # iterations, which would otherwise corrupt the bailed-to
                # iterate on the stall-detector return below.
                best_nu_hat = nu_hat.copy()
                best_delta_x = delta_x.copy()
            if meaningful_improvement:
                iters_since_improvement = 0
            elif in_quadratic_phase:
                iters_since_improvement += 1
            self.settings.trace.on_newton_step(
                outer_nit=outer_nit,
                inner_nit=nit + 1,
                elapsed_ms=newton_elapsed,
                suboptimality=suboptimality,
                lambda_norm=lambda_norm,
                quadratic_convergence_threshold=quadratic_threshold_crossed,
            )

            if suboptimality < self.settings.inner.tolerance:
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(x, t, delta_x)
                return _CenteringOutcome(
                    solution=x,
                    objective_value=self.evaluate_objective(x),
                    barrier_objective_value=self.evaluate_barrier_objective(x, t),
                    dual_value=self._evaluate_dual_traced(
                        outer_nit=outer_nit, lmbda=lambda_star, nu=nu_star, x_star=x
                    ),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimalities=suboptimalities,
                    btls_steps=btls_steps,
                    quadratic_from=quadratic_from,
                    nits=nit + 1,
                    status=0,
                    message=(
                        "Newton's method completed successfully to the desired "
                        "tolerance."
                    ),
                )

            # Stall detector. At high ``t``, ``ft = t * f0 - sum log(-fi)``
            # grows large enough that ``ft(x) - ft(x + delta_x)`` rounds to
            # noise and ``suboptimality`` bounces around the FP floor below
            # ``tolerance_soft`` but above ``tolerance``. Once
            # ``stall_patience`` consecutive iterations in the quadratic
            # phase fail to meaningfully improve the best ``suboptimality``,
            # bail to the best iterate with ``status=1``. This fires on the
            # last centering step too: there is no further step to make up
            # the gap, and the best iterate -- already below
            # ``tolerance_soft`` -- is an acceptable final answer, far better
            # than burning the full ``max_iterations`` budget bouncing at the
            # FP floor (the highest-``t`` step is the most prone to it).
            if (
                in_quadratic_phase
                and iters_since_improvement >= self.settings.inner.stall_patience
                and best_suboptimality < self.settings.inner.tolerance_soft
            ):
                # In the quadratic phase the best-tracking branch above has
                # fired at least once (the first iterate beats ``inf``), so
                # both are non-None here; assert to satisfy mypy and pin the
                # invariant loudly.
                assert best_nu_hat is not None and best_delta_x is not None
                nu_star = best_nu_hat / t
                lambda_star = self.inequality_multipliers(
                    best_x, t, best_delta_x, centering_step_solved_perfectly=False
                )
                return _CenteringOutcome(
                    solution=best_x,
                    objective_value=self.evaluate_objective(best_x),
                    barrier_objective_value=self.evaluate_barrier_objective(best_x, t),
                    dual_value=self._evaluate_dual_traced(
                        outer_nit=outer_nit,
                        lmbda=lambda_star,
                        nu=nu_star,
                        x_star=best_x,
                    ),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimalities=suboptimalities,
                    btls_steps=btls_steps,
                    quadratic_from=quadratic_from,
                    nits=nit + 1,
                    status=1,
                    message=(
                        "Newton's method stalled at the floating-point floor "
                        "in the quadratic phase; returning best iterate "
                        f"(suboptimality={best_suboptimality:.3e})."
                    ),
                )

            # Update. In the quadratic phase, Boyd and Vandenberghe 9.5.3
            # guarantees the unit step satisfies Armijo. At high ``t``,
            # ``ft = t * f0 - sum log(-fi)`` grows large enough that
            # ``ft(x) - ft(x + delta_x)`` rounds to noise in the
            # sufficient-decrease check, and BTLS halves ``s`` down to
            # ``min_step`` chasing a signal the floating-point arithmetic
            # cannot deliver. Skip the Armijo loop and take the unit step
            # capped at the feasibility limit (with the usual ``beta``
            # backoff from the constraint boundary). We deliberately do not
            # call ``LineSearch.find_step`` here: it runs the Armijo
            # contraction, which is exactly the FP-thrashing path this branch
            # exists to bypass. The cap below mirrors the pre-Armijo
            # feasibility cap in ``LineSearch.find_step`` (``beta * max_step``
            # capped at the unit step); keep the two in sync if that backoff
            # strategy ever changes.
            if in_quadratic_phase:
                btls_s = min(
                    1.0,
                    self.settings.armijo.beta * self.btls_keep_feasible(x, delta_x),
                )
            else:
                try:
                    btls_s = self.backtracking_line_search(
                        x=x, delta_x=delta_x, t=t, lambda_squared=lambda_squared
                    )
                except BacktrackingLineSearchError as e:
                    nu_star = nu_hat / t
                    lambda_star = self.inequality_multipliers(
                        x, t, delta_x, centering_step_solved_perfectly=False
                    )
                    if (
                        not last_step
                        and suboptimality < self.settings.inner.tolerance_soft
                    ):
                        return _CenteringOutcome(
                            solution=x,
                            objective_value=self.evaluate_objective(x),
                            barrier_objective_value=self.evaluate_barrier_objective(
                                x, t
                            ),
                            dual_value=self._evaluate_dual_traced(
                                outer_nit=outer_nit,
                                lmbda=lambda_star,
                                nu=nu_star,
                                x_star=x,
                            ),
                            equality_multipliers=nu_star,
                            inequality_multipliers=lambda_star,
                            suboptimalities=suboptimalities,
                            btls_steps=btls_steps,
                            quadratic_from=quadratic_from,
                            nits=nit + 1,
                            status=1,
                            message=(
                                "Newton's method achieved an acceptable tolerance but "
                                f"then ran into numerical issues -- {e.__str__()}"
                            ),
                        )

                    raise CenteringStepError(
                        message=f"Backtracking line search failed -- {e.__str__()}",
                        suboptimality=suboptimality,
                        last_iterate=x,
                        equality_multipliers=nu_star,
                        inequality_multipliers=lambda_star,
                    ) from e

            # Record the phase transition here, not when it is first detected
            # above: an iterate that crosses the quadratic threshold and then
            # converges before reaching this point returns early with no step
            # appended. Setting ``quadratic_from`` at detection time would
            # leave it indexing a step that is never recorded, making
            # ``steps_past_quadratic_convergence`` silently drop the
            # transition instead of reporting "no evidence" via `None`.
            if in_quadratic_phase and quadratic_from is None:
                quadratic_from = len(btls_steps)
            btls_steps.append(btls_s)

            if self.settings.trace.want_btls_details():
                ft = self.evaluate_barrier_objective(x, t)
                ft_new = self.evaluate_barrier_objective(x + btls_s * delta_x, t)
                expected_improvement = (
                    self.settings.armijo.alpha
                    * self.settings.armijo.beta
                    * lambda_squared
                    / (1 + lambda_norm)
                )
                self.settings.trace.on_btls(
                    outer_nit=outer_nit,
                    inner_nit=nit + 1,
                    btls_s=btls_s,
                    improvement=ft - ft_new,
                    expected_improvement=expected_improvement,
                )

            # A feasibility solve (the objective carries a threshold) stops the
            # centering at the first feasible iterate -- otherwise driving the slack
            # to its minimum degenerates the Hessian.
            if (self._feasibility_threshold() is not None) and self.is_feasible(
                x + btls_s * delta_x
            ):
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(
                    x, t, delta_x, centering_step_solved_perfectly=False
                )
                return _CenteringOutcome(
                    solution=x + btls_s * delta_x,
                    objective_value=self.evaluate_objective(x + btls_s * delta_x),
                    barrier_objective_value=self.evaluate_barrier_objective(
                        x + btls_s * delta_x, t
                    ),
                    dual_value=self._evaluate_dual_traced(
                        outer_nit=outer_nit, lmbda=lambda_star, nu=nu_star, x_star=x
                    ),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimalities=suboptimalities,
                    btls_steps=btls_steps,
                    quadratic_from=quadratic_from,
                    nits=nit + 1,
                    status=1,
                    message=("Newton's method found a feasible point."),
                )

            x += btls_s * delta_x

        # Loop exhausted ``max_iterations``. If we nonetheless reached
        # acceptable (soft) tolerance, return the best iterate with
        # ``status=1`` rather than raising -- the same soft-tolerance contract
        # the stall detector and the BTLS-error fallback honor. A slow
        # asymptotic creep toward a floor above ``tolerance`` lands here if it
        # ever outruns the stall detector.
        if best_suboptimality < self.settings.inner.tolerance_soft:
            assert best_nu_hat is not None and best_delta_x is not None
            nu_star = best_nu_hat / t
            lambda_star = self.inequality_multipliers(
                best_x, t, best_delta_x, centering_step_solved_perfectly=False
            )
            return _CenteringOutcome(
                solution=best_x,
                objective_value=self.evaluate_objective(best_x),
                barrier_objective_value=self.evaluate_barrier_objective(best_x, t),
                dual_value=self._evaluate_dual_traced(
                    outer_nit=outer_nit, lmbda=lambda_star, nu=nu_star, x_star=best_x
                ),
                equality_multipliers=nu_star,
                inequality_multipliers=lambda_star,
                suboptimalities=suboptimalities,
                btls_steps=btls_steps,
                quadratic_from=quadratic_from,
                nits=self.settings.inner.max_iterations,
                status=1,
                message=(
                    "Newton's method hit max_iterations but reached acceptable "
                    f"tolerance; returning best iterate "
                    f"(suboptimality={best_suboptimality:.3e})."
                ),
            )

        raise CenteringStepError(
            "Centering step did not converge.",
            suboptimality=suboptimality,
            last_iterate=x,
        )

    def _evaluate_dual_traced(
        self,
        outer_nit: int,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate the Lagrange dual function, bracketed by timing trace events.

        Wraps ``evaluate_dual`` with the ``on_dual_eval_start`` and
        ``on_dual_eval_done`` trace events. The dual evaluation can itself be an inner
        Newton minimization of the Lagrangian (as in ``BarrierSolver``), so its cost is
        reported alongside the Newton and line-search timings.

        Parameters
        ----------
         outer_nit : int
            One-based index of the outer iteration, for the trace events.
         lmbda : npt.NDArray[np.float64]
            The inequality multipliers λ.
         nu : npt.NDArray[np.float64]
            The equality multipliers ν.
         x_star : npt.NDArray[np.float64]
            The iterate at which the multipliers were computed.

        Returns
        -------
         g : float
            The dual value.

        """
        self.settings.trace.on_dual_eval_start(outer_nit)
        dual_start = time.time()
        dual_value = self.evaluate_dual(lmbda=lmbda, nu=nu, x_star=x_star)
        dual_elapsed = 1000 * (time.time() - dual_start)
        self.settings.trace.on_dual_eval_done(outer_nit, dual_elapsed)
        return dual_value

    def backtracking_line_search(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        t: float,
        lambda_squared: float,
    ) -> float:
        """Run the Armijo backtracking line search on the barrier objective.

        Wires ``ArmijoBarrierMerit`` on f_t to ``LineSearch``, with
        ``btls_keep_feasible`` as the maximum-step finder and the Armijo settings for α,
        β and the minimum step. When ``armijo.use_self_concordant_step_cap`` is set, the
        first trial step is capped at 1 / (1 + λ); the cap is applied here because this
        method knows λ^2.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate, strictly feasible.
         delta_x : npt.NDArray[np.float64]
            The Newton step.
         t : float
            The barrier parameter.
         lambda_squared : float
            The squared Newton decrement λ^2, nonnegative.

        Returns
        -------
         btls_s : float
            The accepted step.

        Raises
        ------
         InvalidDescentDirectionError
            Propagated from ``LineSearch.find_step`` when ∇f_t^T Δx > 0.
         ConstraintBoundaryError
            Propagated from ``LineSearch.find_step`` or ``btls_keep_feasible`` when the
            feasible step is too small.
         SevereCurvatureError
            Propagated from ``LineSearch.find_step`` when no step above the minimum
            satisfies the Armijo condition.

        Notes
        -----
        The self-concordant cap is the damped Newton step (Boyd and Vandenberghe, 2004,
        section 9.6.4). In the damped phase it prevents overshooting into
        ill-conditioned regions; as λ shrinks it approaches 1. It is opt-in: on
        well-behaved barriers (linear and quadratic programs) the uncapped unit step is
        usually accepted and the cap only slows convergence, while on sharper barriers,
        such as second-order cone constraints, full damped-phase steps can blow up.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        alpha = self.settings.armijo.alpha
        beta = self.settings.armijo.beta
        min_step = self.settings.armijo.min_step

        if self.settings.armijo.use_self_concordant_step_cap:
            step_cap = 1.0 / (1.0 + math.sqrt(lambda_squared))
        else:
            step_cap = 1.0

        grad_ft_dot_delta_x = float(np.dot(delta_x, self.gradient_barrier(x, t)))
        merit = ArmijoBarrierMerit(
            evaluate_barrier_objective=lambda x_trial: self.evaluate_barrier_objective(
                x_trial, t
            ),
            ft0=self.evaluate_barrier_objective(x, t),
            neg_alpha_grad=alpha * lambda_squared,
            grad_ft_dot_delta_x=grad_ft_dot_delta_x,
        )
        line_search = LineSearch(
            merit=merit,
            max_step_finder=self.btls_keep_feasible,
            beta=beta,
            min_step=min_step,
        )
        return line_search.find_step(x, delta_x, step_cap=step_cap)

    def newton_decrement_squared(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        delta_x: npt.NDArray[np.float64],
    ) -> float:
        """Compute the squared Newton decrement λ^2 at the Newton step.

        The squared decrement is Δx^T H Δx, where H is the Hessian of f_t. Two forms are
        available, selected by ``inner.newton_decrement_via_gradient``:
        - Hessian form (the default): Δx^T H Δx through ``hessian_multiply``. A
          second Hessian-vector product per iteration, on top of the Newton solve, but
          the true quadratic-model decrement and the more reliable form when high
          precision is needed.
        - Gradient form: -∇f_t^T Δx. Since Δx solves the Newton system
             H Δx + A^T ν = -∇f_t,   A Δx = 0,
          the two forms are equal: the A^T ν term drops out because A Δx = 0. The
          barrier gradient can be far cheaper than a Hessian-vector product (about 20
          times on one large structured objective), but the identity degrades under
          inexact solves and ill-conditioning at high t, so it suits problems that do
          not need high precision.
        With equality constraints, neither form equals ∇f_t^T H^{-1} ∇f_t.

        Requires Δx to be the Newton step at x, which the only caller,
        ``centering_step``, guarantees. The result is clamped at 0 to absorb
        floating-point noise.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate.
         t : float
            The barrier parameter.
         delta_x : npt.NDArray[np.float64]
            The Newton step at x.

        Returns
        -------
         lambda_squared : float
            The squared Newton decrement, nonnegative.

        """
        if self.settings.inner.newton_decrement_via_gradient:
            return max(0.0, -float(np.dot(delta_x, self.gradient_barrier(x, t))))
        return max(0.0, float(np.dot(delta_x, self.hessian_multiply(x, t, delta_x))))

    def inequality_multipliers(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        delta_x: npt.NDArray[np.float64],
        centering_step_solved_perfectly: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Estimate the inequality multipliers λ from a centering iterate.

        At an exactly centered point x*(t), the multipliers
           λ_i = -1 / (t f_i(x))
        are dual feasible (Boyd and Vandenberghe, 2004, section 11.2). When the
        centering was not solved to tolerance, the estimate instead uses the first-order
        expansion of -1 / (t f_i(x + Δx)) about x,
           λ_i = -(1 / (t f_i(x))) (1 - ∇f_i(x)^T Δx / f_i(x)),
        anticipating the Newton step.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The iterate, strictly feasible.
         t : float
            The barrier parameter.
         delta_x : npt.NDArray[np.float64]
            The Newton step at x; used only when ``centering_step_solved_perfectly`` is
            ``False``.
         centering_step_solved_perfectly : bool
            Whether x is centered to tolerance.

        Returns
        -------
         lmbda : npt.NDArray[np.float64]
            The multiplier estimates, of length m.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        lmbda: npt.NDArray[np.float64] = np.asarray(-1.0 / (t * self.constraints(x)))
        if not centering_step_solved_perfectly:
            lmbda = lmbda * (
                1.0 - self.grad_constraints_multiply(x, delta_x) / self.constraints(x)
            )
        return lmbda

    @abstractmethod
    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the Newton step of the centering problem.

        Implementations solve the KKT system of
           minimize    f_t(x) = t f0(x) - ∑ log(-f_i(x))
           subject to  A x = b,
        which is
           | H   A^T | | Δx    |   | -∇f_t(x) |
           | A    0  | | ν_hat | = |    0     |
        where H is the Hessian of f_t at x. This is where a subclass exploits the
        problem's structure.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate, strictly feasible.
         t : float
            The barrier parameter.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The Newton step Δx.
         nu : npt.NDArray[np.float64]
            The multiplier ν_hat of the equality constraints in the Newton system, of
            length p; empty without equality constraints. ``centering_step`` reports
            ν_hat / t as the equality multiplier of the original problem.

        Raises
        ------
         NewtonStepError
            Implementations raise when the system cannot be solved.

        """

    @abstractmethod
    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Check whether a point is feasible enough to stop a feasibility solve.

        Consulted only in feasibility mode (see ``_feasibility_threshold``).

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point to test.

        Returns
        -------
         feasible : bool
            True when the point solves the feasibility problem.

        """

    def _is_infeasible_certified(self, result: SolveResult) -> bool:
        """Check whether a result carries a dual certificate of infeasibility.

        For a Phase I problem minimizing a slack, a strictly positive dual value is a
        positive lower bound on the minimal slack, which proves no feasible point
        exists. Subclasses may override this to refine the condition. ``solve`` consults
        it under ``on_infeasible="continue"``, and the barrier adapter's
        ``check_for_infeasibility`` consults it too.

        Parameters
        ----------
         result : SolveResult
            An intermediate Phase I result.

        Returns
        -------
         certified : bool
            True when the dual value is strictly positive.

        """
        return result.dual_value > 0.0

    def check_for_infeasibility(self, result: SolveResult) -> None:
        """Raise if an intermediate Phase I result certifies infeasibility.

        For some problems, the dual function can certify infeasibility. A subclass for
        such a problem overrides this to inspect the result after each Phase I centering
        step and raise when infeasibility is proven. The base does nothing.

        Parameters
        ----------
         result : SolveResult
            An intermediate Phase I result.

        Raises
        ------
         ProblemCertifiablyInfeasibleError
            Overrides raise when the result proves infeasibility.

        """
        pass

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Find a step that keeps x + s Δx strictly feasible.

        A convenience fallback for subclasses without a constraint-layer model. It
        backtracks from s = 1 / β by factors of β, evaluating ``constraints`` at each
        trial, until every constraint is strictly negative. Starting at 1 / β rather
        than 1 matters: the caller multiplies the result by β and caps it at 1, so a
        start at 1 could only ever yield β (a half step with the default β), even when
        the full step is feasible.

        A subclass that works over a ``Problem`` with constraint layers should override
        this to return the minimum of the layers' closed-form ``btls_max_step``, which
        may be inf when nothing binds; ``_BarrierAdapter.btls_keep_feasible`` is the
        canonical example.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate, strictly feasible.
         delta_x : npt.NDArray[np.float64]
            The search direction.

        Returns
        -------
         btls_s : float
            A strictly feasible step, at most 1 / β.

        Raises
        ------
         ConstraintBoundaryError
            When the trial step falls below ``armijo.min_step`` without becoming
            feasible.

        """
        btls_s = 1.0 / self.settings.armijo.beta
        while np.any(self.constraints(x + btls_s * delta_x) >= 0):
            btls_s *= self.settings.armijo.beta
            if btls_s < self.settings.armijo.min_step:
                raise ConstraintBoundaryError(
                    message="Descent step takes us too close to constraint boundaries.",
                )

        return btls_s

    @abstractmethod
    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective f0 at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which to evaluate.

        Returns
        -------
         f0 : float
            The objective value.

        """

    def evaluate_barrier_objective(self, x: npt.NDArray[np.float64], t: float) -> float:
        """Evaluate the barrier objective f_t at x.

        Computes
           f_t(x) = t f0(x) - ∑ log(-f_i(x)).
        Returns inf when any constraint is not strictly satisfied, as
        ``ConstraintLayer.barrier_objective`` does: inf is the extended-real value of
        the barrier there, and lets the line search reject the point cleanly, where the
        log would instead give ``nan`` and a runtime warning.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which to evaluate.
         t : float
            The barrier parameter.

        Returns
        -------
         ft : float
            The barrier objective; inf when x is not strictly feasible.

        """
        # Mirror ``ConstraintLayer.barrier_objective``'s infeasibility guard:
        # without it, a marginally infeasible iterate (any ``c_i >= 0``)
        # raises ``RuntimeWarning: invalid value encountered in log`` and
        # returns ``nan``, which then misclassifies "infeasible" as
        # "indeterminate" upstream. Returning ``+inf`` is the correct
        # extended-real value of ``-log(0+)`` and lets BTLS reject cleanly.
        c = self.constraints(x)
        if np.any(c >= 0):
            return np.inf
        return t * self.evaluate_objective(x) - np.sum(np.log(-c))

    @abstractmethod
    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the objective gradient ∇f0 at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which to evaluate.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The objective gradient.

        """

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Compute the barrier-objective gradient ∇f_t at x.

        Computes
           ∇f_t(x) = t ∇f0(x) - G^T (1 / f(x)),
        where G is the constraint Jacobian and 1 / f(x) is taken componentwise.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which to evaluate, strictly feasible.
         t : float
            The barrier parameter.

        Returns
        -------
         grad_ft : npt.NDArray[np.float64]
            The barrier-objective gradient.

        """
        return t * self.gradient(x) - (
            self.grad_constraints_transpose_multiply(x, 1.0 / self.constraints(x))
        )

    @abstractmethod
    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the barrier-objective Hessian ∇^2 f_t at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Hessian is evaluated.
         t : float
            The barrier parameter.
         y : npt.NDArray[np.float64]
            The vector to multiply.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product ∇^2 f_t(x) y.

        """

    @abstractmethod
    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the inequality constraint values f_i(x).

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which to evaluate.

        Returns
        -------
         f : npt.NDArray[np.float64]
            The constraint values, of length m; all negative when x is strictly
            feasible.

        """

    @abstractmethod
    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the inequality constraint Jacobian G at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which to evaluate.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The Jacobian, whose row i is ∇f_i(x).

        """

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian G.

        The default forms G with ``grad_constraints``; a structured override is usually
        faster.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Jacobian is evaluated.
         y : npt.NDArray[np.float64]
            The vector to multiply.

        Returns
        -------
         Gy : npt.NDArray[np.float64]
            The product G y, of length m.

        """
        return self.grad_constraints(x) @ y

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian G^T.

        The default forms G with ``grad_constraints``; a structured override is usually
        faster.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Jacobian is evaluated.
         y : npt.NDArray[np.float64]
            The vector of length m to multiply.

        Returns
        -------
         GTy : npt.NDArray[np.float64]
            The product G^T y.

        """
        return self.grad_constraints(x).T @ y

    @abstractmethod
    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate the Lagrange dual function g(λ, ν).

        Parameters
        ----------
         lmbda : npt.NDArray[np.float64]
            The inequality multipliers λ.
         nu : npt.NDArray[np.float64]
            The equality multipliers ν.
         x_star : npt.NDArray[np.float64]
            The iterate at which the multipliers were computed.

        Returns
        -------
         g : float
            The dual value g(λ, ν), a lower bound on the optimal value.

        Notes
        -----
        Ideally an implementation evaluates the dual at any (λ, ν), but that can be
        hard. The solver only needs it at the multipliers of a centering iterate, where
        it is the Lagrangian evaluated at x_star when x_star minimizes the Lagrangian;
        ``x_star`` is supplied for that case.

        """


class InteriorPointMethodSolver(BaseInteriorPointMethodSolver):
    """Barrier interior-point solver for optimization problems.

    A ``BaseInteriorPointMethodSolver`` whose ``is_feasible`` always returns ``False``:
    outside feasibility mode the solver runs to the optimum rather than stopping at a
    feasible point.

    Parameters
    ----------
     phase1_solver : FeasibilitySolver | None
        Solver that finds a strictly feasible start; required by ``solve``.
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults.
     name : str | None
        Identifier for log messages; ``None`` uses the label or the class name.
     **kwargs : Any
        Accepted for cooperative construction and ignored.

    """

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Return False, so the solver never stops at a merely feasible point.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point to test; ignored.

        Returns
        -------
         feasible : bool
            Always ``False``.

        """
        return False

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Solve the problem with the barrier method.

        Solves
           minimize    f0(x)
           subject to  f_i(x) ≤ 0,   i = 1, ..., m,
                       A x = b.
        See ``BaseInteriorPointMethodSolver.solve``.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess. When it is not strictly feasible, the Phase I solver finds a
            start that is.
         **kwargs : Any
            Forwarded to ``BaseInteriorPointMethodSolver.solve``.

        Returns
        -------
         result : SolveResult
            The solution and the solve's diagnostics.

        Raises
        ------
         ValueError
            Propagated from ``BaseInteriorPointMethodSolver.solve`` when there is no
            Phase I solver.
         InteriorPointMethodError
            Propagated from ``BaseInteriorPointMethodSolver.solve`` when the best bounds
            miss ``outer.tolerance_soft`` or no centering step completes.

        """
        return super().solve(x0, **kwargs)


class FeasibilityInteriorPointSolver(BaseInteriorPointMethodSolver, FeasibilitySolver):
    """Base class for a feasibility solver built on the barrier method.

    A feasibility solver can stop as soon as it finds a feasible point, rather than
    solving its Phase I problem to optimality. That early stop happens in
    ``BaseInteriorPointMethodSolver``'s feasibility mode, which a subclass enables by
    overriding ``_feasibility_threshold`` to return a value and ``is_feasible`` to
    recognize a feasible point; this class adds neither.

    Parameters
    ----------
     phase1_solver : FeasibilitySolver | None
        Solver that finds a start strictly feasible for the constraints this solver
        keeps; required by ``solve``.
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults.
     **kwargs : Any
        Passed to ``BaseInteriorPointMethodSolver`` (for example, ``name``).

    """

    def __init__(
        self,
        phase1_solver: "FeasibilitySolver | None" = None,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(phase1_solver=phase1_solver, settings=settings, **kwargs)

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Find a point satisfying every constraint, with the barrier method.

        Finds x with
           A x = b,   f_i(x) ≤ 0 for i = 1, ..., m,   f_j(x) ≤ 0 for j = 1, ..., n,
        given a start the Phase I solver makes strictly feasible for the first group.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess. A guess feasible for some of the constraints lets the Phase I
            solver focus on the rest.
         **kwargs : Any
            Forwarded to ``BaseInteriorPointMethodSolver.solve``.

        Returns
        -------
         result : SolveResult
            The result, whose solution is a feasible point.

        Raises
        ------
         ProblemCertifiablyInfeasibleError
            Propagated from ``BaseInteriorPointMethodSolver.solve`` in feasibility mode,
            when the dual certifies that no feasible point exists.
         ProblemMarginallyFeasibleError
            Propagated from ``BaseInteriorPointMethodSolver.solve`` in feasibility mode,
            when no feasible point is found and infeasibility cannot be certified.
         InteriorPointMethodError
            Propagated from ``BaseInteriorPointMethodSolver.solve`` when a centering
            step fails before reaching an acceptable gap.

        Notes
        -----
        Given x0 with A x0 = b and f_i(x0) < 0 for every i, the typical formulation
        solves
           minimize    s
           subject to  A x = b,
                       f_i(x) ≤ 0,   i = 1, ..., m,
                       f_j(x) ≤ s,   j = 1, ..., n.
        If the optimal s is positive, the problem is infeasible, and the Lagrange
        multipliers certify it. Otherwise the solution x is feasible for every
        constraint. The problem need not be solved to high precision: any feasible (x,
        s) with s < 0 is enough to stop.

        """
        return super().solve(x0, **kwargs)


class UnconstrainedNewtonSolver(BaseInteriorPointMethodSolver):
    """Newton's method for problems with no equality or inequality constraints.

    Without inequality constraints the barrier parameter plays no role, so the outer
    barrier loop is skipped and Newton's method is applied directly to f0, which is one
    centering step at t = 1.

    A subclass implements ``newton_step`` (solve H Δx = -∇f0 and return Δx),
    ``evaluate_objective``, ``gradient`` and ``hessian_vector_product``.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults.
     **kwargs : Any
        Passed to ``BaseInteriorPointMethodSolver`` (for example, ``name``).

    """

    def __init__(
        self,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(phase1_solver=None, settings=settings, **kwargs)

    @property
    def num_eq_constraints(self) -> int:
        """Return zero, the number of equality constraints.

        Returns
        -------
         p : int
            Always 0.

        """
        return 0

    @property
    def num_ineq_constraints(self) -> int:
        """Return zero, the number of inequality constraints.

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
        """Minimize the objective with Newton's method.

        Skips the barrier loop and runs a single centering step at t = 1.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess. Required despite the ``None`` default.
         **kwargs : Any
            Ignored.

        Returns
        -------
         result : SolveResult
            The solution, with solver name ``"newton"``.

        Raises
        ------
         ValueError
            When ``x0`` is ``None``.
         CenteringStepError
            Propagated from ``centering_step`` when a Newton step cannot be computed,
            the line search fails, or Newton's method does not reach an acceptable
            tolerance.

        """
        if x0 is None:
            raise ValueError("Initial guess x0 is required.")
        return _newton_solveresult(self.centering_step(x0, t=1.0, last_step=True))

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Return True, since every point is feasible without constraints.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point to test; ignored.

        Returns
        -------
         feasible : bool
            Always ``True``.

        """
        return True

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the empty vector of inequality constraint values.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point; ignored.

        Returns
        -------
         f : npt.NDArray[np.float64]
            An empty array.

        """
        return np.zeros(0)

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the empty inequality constraint Jacobian.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point, used only for its length M.

        Returns
        -------
         G : npt.NDArray[np.float64]
            An empty array of shape (0, M).

        """
        return np.zeros((0, x.size))

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Return inf, since no constraint limits the step.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate; ignored.
         delta_x : npt.NDArray[np.float64]
            The search direction; ignored.

        Returns
        -------
         btls_s : float
            Always inf.

        """
        return np.inf

    @abstractmethod
    def newton_step(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the Newton step for the objective.

        Implementations solve H(x) Δx = -∇f0(x), exploiting the Hessian's structure.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The Newton step.

        Raises
        ------
         NewtonStepError
            Implementations raise when the system cannot be solved.

        """

    @abstractmethod
    def hessian_vector_product(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the objective Hessian H(x).

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Hessian is evaluated.
         y : npt.NDArray[np.float64]
            The vector to multiply.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product H(x) y.

        """

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the Newton step through ``newton_step``, ignoring t.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate.
         t : float
            The barrier parameter; unused, since there is no barrier.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The Newton step.
         nu : npt.NDArray[np.float64]
            An empty array, since there are no equality constraints.

        Raises
        ------
         NewtonStepError
            Propagated from ``newton_step``.

        """
        delta_x = self.newton_step(x)
        return delta_x, np.zeros(0)

    def hessian_multiply(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the objective Hessian, ignoring t.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Hessian is evaluated.
         t : float
            The barrier parameter; unused, since there is no barrier.
         y : npt.NDArray[np.float64]
            The vector to multiply.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product H(x) y.

        """
        return self.hessian_vector_product(x, y)

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Return the objective value at the solution, which the dual equals here.

        Without constraints the Lagrangian is the objective itself, so at a minimizer
        the dual value equals the primal value. Exact only at the optimum: at an
        approximate solution the returned value is not a lower bound.

        Parameters
        ----------
         lmbda : npt.NDArray[np.float64]
            The inequality multipliers; ignored (empty).
         nu : npt.NDArray[np.float64]
            The equality multipliers; ignored (empty).
         x_star : npt.NDArray[np.float64]
            The solution.

        Returns
        -------
         g : float
            The objective value f0(x_star).

        """
        return self.evaluate_objective(x_star)


class EqualityConstrainedNewtonSolver(BaseInteriorPointMethodSolver):
    """Newton's method for problems with equality constraints A x = b only.

    Without inequality constraints the barrier parameter plays no role, so the outer
    barrier loop is skipped and Newton's method is applied directly to the
    equality-constrained problem, each step solving the KKT system rather than the
    Hessian system alone.

    A subclass implements ``A`` (of shape (p, M)), ``b`` (of length p), ``newton_step``
    (solve the KKT system and return Δx and ν), ``evaluate_objective``, ``gradient`` and
    ``hessian_vector_product``.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults.
     **kwargs : Any
        Passed to ``BaseInteriorPointMethodSolver`` (for example, ``name``).

    """

    def __init__(
        self,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(phase1_solver=None, settings=settings, **kwargs)

    @property
    @abstractmethod
    def A(self) -> npt.NDArray[np.float64]:
        """Return the equality constraint matrix A.

        Returns
        -------
         A : npt.NDArray[np.float64]
            The equality matrix, of shape (p, M).

        """

    @property
    @abstractmethod
    def b(self) -> npt.NDArray[np.float64]:
        """Return the equality right-hand side b.

        Returns
        -------
         b : npt.NDArray[np.float64]
            The right-hand side, of length p.

        """

    @property
    def num_eq_constraints(self) -> int:
        """Return the number of equality constraints, the length of b.

        Returns
        -------
         p : int
            The number of equality constraints.

        """
        return len(self.b)

    @property
    def num_ineq_constraints(self) -> int:
        """Return zero, the number of inequality constraints.

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
        """Minimize the objective subject to A x = b with Newton's method.

        Skips the barrier loop and runs a single centering step at t = 1. Each Newton
        step satisfies A Δx = 0, so the iterates keep the equality residual of x0.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Initial guess, which should satisfy A x0 = b (not checked). Required despite
            the ``None`` default.
         **kwargs : Any
            Ignored.

        Returns
        -------
         result : SolveResult
            The solution, with solver name ``"newton"`` and the equality multipliers.

        Raises
        ------
         ValueError
            When ``x0`` is ``None``.
         CenteringStepError
            Propagated from ``centering_step`` when a Newton step cannot be computed,
            the line search fails, or Newton's method does not reach an acceptable
            tolerance.

        """
        if x0 is None:
            raise ValueError("Initial guess x0 is required.")
        return _newton_solveresult(self.centering_step(x0, t=1.0, last_step=True))

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Return True, since there are no inequality constraints to violate.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point to test; ignored.

        Returns
        -------
         feasible : bool
            Always ``True``.

        """
        return True

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the empty vector of inequality constraint values.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point; ignored.

        Returns
        -------
         f : npt.NDArray[np.float64]
            An empty array.

        """
        return np.zeros(0)

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the empty inequality constraint Jacobian.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point, used only for its length M.

        Returns
        -------
         G : npt.NDArray[np.float64]
            An empty array of shape (0, M).

        """
        return np.zeros((0, x.size))

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Return inf, since no inequality constraint limits the step.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate; ignored.
         delta_x : npt.NDArray[np.float64]
            The search direction; ignored.

        Returns
        -------
         btls_s : float
            Always inf.

        """
        return np.inf

    @abstractmethod
    def newton_step(
        self, x: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the KKT system for the Newton step and the equality multiplier.

        Implementations solve
           | H(x)   A^T | | Δx |   | -∇f0(x) |
           |  A      0  | | ν  | = |    0    |
        exploiting the problem's structure.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The Newton step, of length M.
         nu : npt.NDArray[np.float64]
            The equality multipliers, of length p.

        Raises
        ------
         NewtonStepError
            Implementations raise when the system cannot be solved.

        """

    @abstractmethod
    def hessian_vector_product(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the objective Hessian H(x).

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Hessian is evaluated.
         y : npt.NDArray[np.float64]
            The vector to multiply.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product H(x) y.

        """

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the Newton step through ``newton_step``, ignoring t.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The current iterate.
         t : float
            The barrier parameter; unused, since there is no barrier (t = 1).

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The Newton step.
         nu : npt.NDArray[np.float64]
            The equality multipliers.

        Raises
        ------
         NewtonStepError
            Propagated from ``newton_step``.

        """
        delta_x, nu = self.newton_step(x)
        return delta_x, nu

    def hessian_multiply(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the objective Hessian, ignoring t.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The point at which the Hessian is evaluated.
         t : float
            The barrier parameter; unused, since there is no barrier (t = 1).
         y : npt.NDArray[np.float64]
            The vector to multiply.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product H(x) y.

        """
        return self.hessian_vector_product(x, y)

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Return the objective value at the solution, which the dual equals here.

        With only linear equality constraints, strong duality holds, and at a feasible
        minimizer with its multipliers the Lagrangian equals the objective, so the dual
        value equals the primal value. Exact only at the optimum: at an approximate
        solution the returned value is not a lower bound.

        Parameters
        ----------
         lmbda : npt.NDArray[np.float64]
            The inequality multipliers; ignored (empty).
         nu : npt.NDArray[np.float64]
            The equality multipliers; ignored.
         x_star : npt.NDArray[np.float64]
            The solution.

        Returns
        -------
         g : float
            The objective value f0(x_star).

        """
        return self.evaluate_objective(x_star)
