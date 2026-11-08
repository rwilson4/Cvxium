"""Settings for Cvxium's solvers: tolerances, iteration limits and observability.

``OptimizationSettings`` is the single object a caller passes to a solver. It groups
the knobs by phase:
- ``OuterSettings``: the barrier method's outer loop (tolerances, the barrier
  multiplier, and the choice of initial barrier parameter via ``BarrierInitStrategy``).
- ``InnerSettings``: the Newton centering steps (tolerances, iteration limit, stall
  detection).
- ``ArmijoSettings``: the backtracking line search on the barrier objective.
- ``LineSearchSettings``: the feasibility-preserving step search on a constraint layer.
- ``PrimalDualSettings`` and ``LevenbergSettings``: the primal-dual solvers' Newton
  damping.
- ``ObservabilitySettings``: verbose logging and the solve label.
The ``trace`` field selects where iteration events go; see ``trace``.

This module depends only on ``diagnostics.trace``, and the ``solvers`` package
initializer holds only a docstring, so constraint layers can import their settings
without importing the solver engine.

"""

from dataclasses import dataclass, field
from enum import Enum

from ..diagnostics.trace import (
    LoggingTrace,
    Trace,
)


@dataclass
class LineSearchSettings:
    """Settings for the feasibility-preserving step search on a constraint layer.

    Used by ``ConstraintLayer.btls_max_step`` to find the largest α keeping the layer's
    constraints strictly feasible. The default iterative search starts at α = 1 / β, so
    that the caller's β α can reach the full unit step, shrinks α by the factor β at
    each trial, and gives up below ``min_step``.

    Fields
    ------
     beta : float
        Shrink factor between trials, in (0, 1). Default 0.5.
     min_step : float
        Smallest step the iterative search tries before giving up. Default 1e-3.

    """

    beta: float = 0.5
    min_step: float = 1e-3


class BarrierInitStrategy(Enum):
    """Strategy for choosing the initial barrier parameter t0 of the barrier method.

    The first centering step is taken at t = t0, and the outer loop then multiplies t by
    ``barrier_multiplier`` after each step. The right t0 depends on the problem class,
    so it is configurable per solve through ``OuterSettings.barrier_init``. ``None``
    there selects ``BarrierSolver``'s default: ``GRADIENT_BALANCE`` without equality
    constraints, ``EQUALITY_SCHUR`` with them.

    The members:
    - ``UNIT``: t0 = 1, the base ``initialize_barrier_parameter``. A fixed, neutral
      start that assumes nothing about the objective's curvature.
    - ``GRADIENT_BALANCE``: t0 = ‖∇φ‖ / ‖∇f0‖, balancing the barrier and objective
      gradients at x0, floored at ``OuterSettings.min_initial_barrier_parameter``.
      Suited to a start deep in the interior with a large objective gradient, where it
      removes a heavy line-search damping cost in the first centering. It is adaptive,
      not conservative: it can put t0 well below 1 or above it. The latter is
      undesirable for a nonconvex objective whose curvature makes the Newton system
      indefinite at high t.
    - ``EQUALITY_SCHUR``: the equality-aware initialization of
      ``EqualitySolver.initialize_barrier_parameter`` (Boyd and Vandenberghe, 2004,
      section 11.3.1), which folds the equality Jacobian A into the gradient balance
      through a Schur complement. Requires equality constraints; ``BarrierSolver``
      raises ``ValueError`` when it is requested for a problem without them.
    - ``OBJECTIVE_VALUE_GAP``: the objective-gap heuristic
         t0 = m / (μ (f0(x0) - p_star)),
      where p_star is an under-estimate of the optimal value supplied through
      ``OuterSettings.objective_value_estimate``, m is the number of inequality
      constraints and μ is ``barrier_multiplier`` (Boyd and Vandenberghe, 2004,
      section 11.3.1). It skips early centering steps when x0 is already nearly
      optimal. The result is capped at m / (μ ε), one step from the end, where ε is
      the outer tolerance, and floored at 1, so at least one centering step runs. It
      degrades to t0 = 1 when no estimate is given. Use it when a good p_star is
      cheaply known (a closed-form under-estimate, a relaxation value, or an
      adjacent point in a sweep).

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    UNIT = "unit"
    GRADIENT_BALANCE = "gradient_balance"
    EQUALITY_SCHUR = "equality_schur"
    OBJECTIVE_VALUE_GAP = "objective_value_gap"


@dataclass
class OuterSettings:
    """Settings for the outer (barrier) loop of the interior-point method.

    Fields
    ------
     tolerance : float
        Target tolerance on the surrogate duality gap; the outer loop stops once the gap
        falls below it. Default 1e-6.
     tolerance_soft : float
        Fallback tolerance when ``tolerance`` cannot be reached: a gap in
        [``tolerance``, ``tolerance_soft``] is acceptable (medium accuracy). This is the
        hard requirement: when even the best gap found exceeds it, the solve raises.
        Default 1e-3.
     barrier_multiplier : float
        Factor μ by which the barrier parameter grows at each outer iteration. Default
        10.0.
     tolerance_rel : float | None
        Relative gap tolerance. A solver that honors it declares convergence once the
        gap is at most
           max(``tolerance``, ``tolerance_rel`` |f0|),
        an absolute floor or a fraction of the objective's scale, whichever is larger.
        The absolute floor keeps the test meaningful when the optimal value is near zero
        (for example, a bilinear matrix game with value zero), where a purely relative
        test would never pass. ``StationaritySolver`` uses it as an acceptance
        criterion. The barrier method keeps its acceptance absolute, since the barrier
        guarantee is an absolute bound (Boyd and Vandenberghe, 2004, section 11.3), and
        consults ``tolerance_rel`` only for its failure message: when a solve misses
        ``tolerance_soft`` but the relative gap is within ``tolerance_rel``, the raised
        ``InteriorPointMethodError`` notes that the result is probably good enough
        relative to the objective's scale. ``None`` (the default) disables the relative
        criterion.
     min_initial_barrier_parameter : float
        Lower bound on the initial barrier parameter t0 from the gradient-balance
        strategy, and on ``StationaritySolver``'s initial barrier parameter. The
        gradient ratio can be zero in degenerate cases, and t0 = 0 breaks the outer step
        count, whose log(t0 ε) is then -inf; the floor keeps t0 strictly positive. Keep
        it small: a large t0 starts the first centering far from the central path, where
        Newton's method diverges and the line search collapses. It is deliberately
        independent of ``tolerance_soft``, which may be raised to a large acceptance
        scale. Default 1e-3.
     barrier_init : BarrierInitStrategy | None
        Strategy for the initial barrier parameter t0 (see ``BarrierInitStrategy``).
        ``None`` (the default) selects ``GRADIENT_BALANCE`` without equality constraints
        and ``EQUALITY_SCHUR`` with them. Set it explicitly to override, for example
        ``UNIT`` for a problem class the adaptive ratio does not suit (such as a
        nonconvex objective whose curvature makes the high-t Newton system indefinite),
        or to pin a strategy against a future change of default.
     objective_value_estimate : float | None
        Under-estimate p_star of the optimal objective value, used by the
        ``OBJECTIVE_VALUE_GAP`` strategy and ignored by the others; with ``None`` (the
        default) that strategy degrades to t0 = 1. It must be an under-estimate: a
        p_star above the true optimum shrinks the gap and inflates t0 too aggressively.
     max_outer_iterations : int
        Maximum number of outer iterations before giving up with
        ``SolveStatus.MAX_ITERATIONS``. Used by ``StationaritySolver``, whose outer loop
        runs until a gap test passes and so needs an explicit cap; the barrier method
        instead computes its step count from the barrier schedule. Default 30.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    tolerance: float = 1e-6
    tolerance_soft: float = 1e-3
    barrier_multiplier: float = 10.0
    tolerance_rel: float | None = None
    min_initial_barrier_parameter: float = 1e-3
    barrier_init: BarrierInitStrategy | None = None
    objective_value_estimate: float | None = None
    max_outer_iterations: int = 30


@dataclass
class InnerSettings:
    """Settings for the inner (Newton centering) loop of the interior-point method.

    Fields
    ------
     tolerance : float
        Convergence tolerance on the centering step's suboptimality, λ^2 / 2 for the
        squared Newton decrement λ^2. Default 1e-6.
     tolerance_soft : float
        Fallback tolerance when ``tolerance`` cannot be reached: a centering step that
        ends with suboptimality below it is accepted. Default 1e-4.
     max_iterations : int
        Maximum number of Newton iterations per centering step. Default 1000.
     stall_patience : int
        Number of consecutive quadratic-phase iterations without meaningful improvement
        in suboptimality (see ``stall_rel_improvement``) after which the centering step
        stops and returns the best iterate it has seen. Guards against the high-t
        regime, where floating-point cancellation in
           f_t(x) = t f0(x) - ∑ log(-f_i(x))
        leaves Newton's method bouncing at the floating-point floor, below ``tolerance``
        in exact arithmetic but not in practice. The stop fires only when the best
        suboptimality is below ``tolerance_soft``; otherwise the loop runs on to
        ``max_iterations``, where the same soft-tolerance fallback applies. Default 5.
     stall_rel_improvement : float
        Smallest relative decrease in suboptimality, against the best so far, that
        counts as progress and resets the stall counter. Without it, a slow creep toward
        the floating-point floor would reset the counter every iteration and defeat the
        stall test, spending the whole ``max_iterations`` budget. Genuine
        quadratic-phase progress is orders of magnitude per step, so a 1% floor never
        cuts off real convergence. Default 0.01.
     newton_decrement_via_gradient : bool
        How to compute the squared Newton decrement, which drives the stopping test and
        the line search. ``False`` (the default) uses the Hessian form Δx^T H Δx through
        ``hessian_multiply``: a second Hessian-vector product per Newton step, but the
        true quadratic-model decrement, and more reliable when high precision is needed.
        ``True`` uses the gradient form -∇f_t^T Δx, which agrees at the exact Newton
        step and can be far cheaper (about 20 times on one large structured objective),
        but which degrades under inexact solves and ill-conditioning at high t. Prefer
        ``True`` only when high precision is not required.

    """

    tolerance: float = 1e-6
    tolerance_soft: float = 1e-4
    max_iterations: int = 1_000
    stall_patience: int = 5
    stall_rel_improvement: float = 0.01
    newton_decrement_via_gradient: bool = False


@dataclass
class ArmijoSettings:
    """Settings for the Armijo (sufficient-decrease) backtracking line search.

    Used by the barrier method's line search and by the primal-dual solvers' step
    search. Distinct from ``LineSearchSettings``, which controls the
    feasibility-preserving step search on individual constraint layers.

    Fields
    ------
     alpha : float
        Sufficient-decrease coefficient α. The barrier method accepts a trial step s
        when the barrier objective decreases by at least α s λ^2, where λ^2 is the
        squared Newton decrement. Default 0.01.
     beta : float
        Shrink factor β applied to the step between trials. Default 0.5.
     min_step : float
        Smallest step the search tries before declaring failure. Default 1e-3.
     use_self_concordant_step_cap : bool
        When ``True``, the barrier method's line search starts from
           min(1 / (1 + λ), β s_feas)
        instead of min(1, β s_feas), where s_feas is the maximum feasible step (Boyd and
        Vandenberghe, 2004, section 9.6.4). Useful on sharply curved barriers, such as
        second-order cone constraints; harmful on linear and quadratic programs. Default
        ``False``.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    alpha: float = 0.01
    beta: float = 0.5
    min_step: float = 1e-3
    use_self_concordant_step_cap: bool = False


@dataclass
class ObservabilitySettings:
    """Settings for solver observability (logging and tracing).

    Fields
    ------
     verbose : bool
        When ``True``, attach a console handler to the ``cvxium`` logger (see
        ``_configure_verbose_logging``) and show progress at ``INFO`` (centering steps)
        and ``DEBUG`` (Newton steps and line search). For finer control, leave it
        ``False`` and configure ``logging.getLogger("cvxium")`` directly. Default
        ``False``.
     label : str | None
        Human-readable solver name, reported in the ``solver`` field of the start and
        done trace events of the barrier, primal-dual and homogeneous self-dual solvers
        (for example, "Starting IPM (<label>)"). With ``None`` (the default), each
        solver uses its class name. This lets a caller replace an uninformative internal
        adapter name such as ``_BarrierAdapter`` with a meaningful one. A ``name``
        passed explicitly to a ``BaseInteriorPointMethodSolver`` takes precedence.

    """

    verbose: bool = False
    label: str | None = None


@dataclass
class LevenbergSettings:
    """Adaptive Levenberg-Marquardt damping schedule for the primal-dual step.

    Each iteration tries the undamped step first. Damping engages only when a feasible
    step cannot otherwise be found, and a ratio test of predicted against actual
    residual reduction relaxes it back toward zero. See ``PrimalDualSolver`` for the
    full algorithm.

    Fields
    ------
     max_attempts : int
        Maximum damping increases per iteration; when they are exhausted, the step
        raises ``NewtonStepError``. Default 40.
     grow : float
        Multiplier applied to the damping when a step is rejected. Default 4.0.
     shrink_good : float
        Multiplier applied to the damping when the ratio ρ exceeds ``rho_good``. Default
        0.25.
     shrink_ok : float
        Multiplier applied to the damping when ρ exceeds ``rho_ok`` but not
        ``rho_good``. Default 0.5.
     rho_good : float
        Ratio threshold above which the damping shrinks aggressively. Default 0.75.
     rho_ok : float
        Ratio threshold above which the damping shrinks gently. Default 0.25.
     drop_to_zero : float
        Relative threshold for switching damping off: below ``drop_to_zero`` max(1, ‖x‖)
        the damping snaps to exactly 0, restoring the undamped step. Default 1e-12.

    """

    max_attempts: int = 40
    grow: float = 4.0
    shrink_good: float = 0.25
    shrink_ok: float = 0.5
    rho_good: float = 0.75
    rho_ok: float = 0.25
    drop_to_zero: float = 1e-12


@dataclass
class PrimalDualSettings:
    """Settings specific to the primal-dual solvers.

    Grouped sub-settings, mirroring how ``OptimizationSettings`` nests its concerns. It
    holds the Levenberg damping schedule; further primal-dual settings would join it as
    sibling groups.

    Fields
    ------
     damping : LevenbergSettings
        Adaptive Levenberg-Marquardt damping schedule.

    """

    damping: LevenbergSettings = field(default_factory=LevenbergSettings)


@dataclass
class OptimizationSettings:
    """Algorithm settings shared by Cvxium's solvers.

    Groups the settings of each part of a solve; each solver reads the groups it uses.

    Fields
    ------
     outer : OuterSettings
        Outer (barrier) loop tolerances and barrier schedule.
     inner : InnerSettings
        Inner (Newton) loop tolerances, iteration cap and stall detection.
     armijo : ArmijoSettings
        Parameters of the Armijo backtracking line search.
     observability : ObservabilitySettings
        Logging and labeling settings, such as ``verbose``.
     line_search : LineSearchSettings
        Settings for the feasibility-preserving step search of
        ``ConstraintLayer.btls_max_step``, distinct from the Armijo line search.
     pd_fraction_to_boundary_min : float
        Floor τ_min of the fraction-to-boundary factor that caps each primal-dual step
        to keep the multipliers positive (see ``mehrotra_step_cap``). The step is
           s = γ s_max,   γ = max(τ_min, 1 - μ),   μ = η / m,
        where s_max is the largest step keeping the multipliers nonnegative and μ is the
        surrogate duality gap per multiplier. Far from the solution (μ ≥ 1 - τ_min) the
        cap is τ_min s_max. Near it (μ → 0), γ → 1, which keeps the step from pinning at
        the minimum step when an active multiplier is close to zero while its step keeps
        pulling it negative (Mehrotra, 1992; Wright, 1997). Default 0.99.
     primal_dual : PrimalDualSettings
        Settings specific to the primal-dual solvers (the Levenberg damping schedule);
        ignored by the barrier method.
     trace : Trace
        Sink for structured solver events (Phase I outcome, centering steps, Newton
        steps, line search, SVD timing). The default ``LoggingTrace`` emits each event
        as a ``cvxium.*`` log record in the format ``IterationHandler`` reads (USAGE.md,
        section 7). Subclass ``Trace`` to collect events directly, without logging.

    References
    ----------
    Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method, SIAM
    Journal on Optimization 2, 1992.

    Wright, Stephen J., Primal-Dual Interior-Point Methods, SIAM, 1997.

    """

    outer: OuterSettings = field(default_factory=OuterSettings)
    inner: InnerSettings = field(default_factory=InnerSettings)
    armijo: ArmijoSettings = field(default_factory=ArmijoSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    line_search: LineSearchSettings = field(default_factory=LineSearchSettings)
    pd_fraction_to_boundary_min: float = 0.99
    primal_dual: PrimalDualSettings = field(default_factory=PrimalDualSettings)
    trace: Trace = field(default_factory=LoggingTrace)
