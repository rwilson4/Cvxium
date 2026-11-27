"""Primal-dual interior-point solver for a ``Problem`` with inequality constraints.

Defines ``PrimalDualSolver``, which minimizes f0(w) subject to f_i(w) ≤ 0 and optional
equality constraints A w = b by the primal-dual interior-point method (Boyd and
Vandenberghe, 2004, section 11.7). Unlike the barrier method (``BarrierSolver``), it
carries the inequality duals λ as explicit state, follows the central path with Newton
steps on the perturbed KKT residual rather than nested centering, and sets the barrier
parameter adaptively from the surrogate duality gap η = -f^T λ,
   t = μ m / η.

After eliminating Δλ, the modified KKT Hessian is
   H_pd = ∇^2 f0 + ∑ λ_i C_i,   C_i = ∇^2 f_i - (1 / f_i) ∇f_i ∇f_i^T,
which is exactly ``Problem.build_hessian_system`` at objective scale 1 with the duals as
the weights. Each layer's ``hessian_contribution`` already supplies the atom C_i, so the
primal-dual method reuses the structured assembly the barrier method uses.

``PrimalDualSolver.solve`` dispatches on ``problem.eq``:
- No equality constraints: a Mehrotra predictor-corrector step per iteration
  (``_run``).
- Equality constraints: the infeasible-start form of the same Mehrotra step
  (``_run_equality``), which drives A x - b to zero along with the other residuals.
Both paths need a strictly inequality-feasible start. When ``x0`` is omitted or
infeasible, ``Solver._solve_feasibility`` builds one through ``Problem.make_phase1``,
cascading through this same solver. Both paths fall back to adaptive Levenberg-Marquardt
damping when the undamped step admits no feasible move.

The module also holds the private pieces the two paths share: ``_PrimalDualKernel``
(stacked constraint values and Jacobian products), ``_StallWatch`` (stall detection),
``_clamp_dual_after_update`` (dual-positivity repair), ``_PdStepInfo`` (per-step
diagnostics), and ``_pd_solveresult`` (result assembly).

References
----------
Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University Press,
2004.

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from .._typing import none_throws
from ..exceptions import NewtonStepError
from ..linalg.systems import DiagonalSystem
from ..problems.result import (
    InfeasibilityCertificate,
    ObjectiveOutcome,
    PrimalDualDiagnostics,
    SolveResult,
    SolveStatus,
)
from ..solvers.barrier_solver import (
    _BarrierAdapter,
)
from ..solvers.line_search import (
    ResidualNormMerit,
    fraction_to_boundary,
    mehrotra_boundary_factor,
    mehrotra_step_cap,
)
from ..solvers.optimization import (
    ProblemCertifiablyInfeasibleError,
    _dual_scale,
    _duals_from,
    _gap_scale,
    _legacy_solve_status,
    _primal_scale,
    _weak_duality_slack,
)
from ..solvers.settings import (
    LevenbergSettings,
    OptimizationSettings,
    OuterSettings,
)
from ..solvers.solver import Solver

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..problems.problem import (
        OnInfeasible,
        Problem,
    )

# Adaptive Levenberg-Marquardt damping for the PD Newton step. The undamped
# step (``damping = 0``) is tried first every iteration, so well-conditioned
# problems are solved with no regularization. Damping engages only when that
# step cannot produce a feasible move (``_feasible_step`` returns ``None``): the
# modified Hessian is regularized with ``damping * I`` so that ``||dx||`` is
# bounded, curing the near-singular-objective-Hessian failure (e.g. an objective
# whose curvature vanishes along some directions). After a successful step, a
# predicted-vs-actual residual-reduction ratio test relaxes the damping back
# toward zero. The schedule constants are configurable via
# ``OptimizationSettings.primal_dual.damping`` (``LevenbergSettings``); the
# no-equality solver takes a Mehrotra predictor-corrector step on this
# (optionally damped) factorization, the equality solver a single damped Newton
# step with a residual line search.


# The unit dual start is rescaled when its constraint force ‖G^T 1‖ and the
# objective's ‖∇f0‖ differ by more than this factor either way (see
# ``PrimalDualSolver._central_path_duals``). A heuristic threshold: a ratio of 0.04
# (a one-dimensional test problem whose unit start is fine) must not trigger it, and
# 1e-3 (an LP with its objective scaled by 1e3) must.
_DUAL_SCALE_RANGE: float = 100.0


class _StallWatch:
    """Stall detector for a primal-dual iteration that has stopped making progress.

    The loops exit only when the surrogate gap η and the residual norms are below
    tolerance at the same time, and that conjunction can become unreachable: the step
    length collapses to its floor, and the iterate creeps, with the gap inching down
    while dual feasibility drifts back up. Without a guard the loop spends its whole
    iteration budget there. In one observed case the solve was healthy for its first 12
    iterations (unit step, gap 5.9e-4, ‖r_dual‖ 1.7e-9), then spent 986 iterations at
    the floor, ending with the gap at 3.1e-6 and ‖r_dual‖ 46,000 times worse than it had
    already achieved. ``BarrierSolver`` guards the analogous Newton stall the same way.

    Progress is judged on the worst of the quantities the convergence test must drive
    below tolerance (max(η, ‖r_dual‖), plus ‖r_pri‖ on the equality path). Unlike the
    KKT residual r_t, which is measured at the current t, that metric is comparable
    across iterations. A run that keeps improving it never trips the guard, however
    small its steps, so the guard cannot cut short a solve that is still converging. Two
    safeguards, shared with the barrier detector:
    - Only a relative drop of ``rel_improvement`` counts as progress. A bare "did it
      improve at all" test is defeated by exactly the creep this exists to catch: the
      observed stall improved the gap about 0.1% per iteration.
    - The stall fires only once the best metric is below ``soft_tolerance``. Short of
      that there is no answer worth returning, so the loop keeps going to its budget.

    The best iterate (``best_x``, ``best_lmbda``, ``best_nu``, ``best_nit``) is recorded
    on any improvement, however small, so a stalled run can hand it back.

    Parameters
    ----------
     patience : int
        Number of consecutive iterations without relative improvement that count as a
        stall (``InnerSettings.stall_patience``).
     rel_improvement : float
        Relative drop in the metric that counts as progress
        (``InnerSettings.stall_rel_improvement``).
     soft_tolerance : float
        The best metric must be below this before a stall is reported
        (``OuterSettings.tolerance_soft``).

    """

    __slots__ = (
        "_best",
        "_patience",
        "_rel_improvement",
        "_since_best",
        "_soft_tolerance",
        "best_lmbda",
        "best_nit",
        "best_nu",
        "best_x",
    )

    def __init__(
        self, patience: int, rel_improvement: float, soft_tolerance: float
    ) -> None:
        self._patience = patience
        self._rel_improvement = rel_improvement
        self._soft_tolerance = soft_tolerance
        self._best: float = np.inf
        self._since_best: int = 0
        self.best_x: npt.NDArray[np.float64] | None = None
        self.best_lmbda: npt.NDArray[np.float64] | None = None
        # Equality multipliers, carried only by ``_run_equality``; ``None`` on
        # the inequality-only path, where there are none to restore.
        self.best_nu: npt.NDArray[np.float64] | None = None
        self.best_nit: int = 0

    def observe(
        self,
        metric: float,
        nit: int,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64] | None = None,
    ) -> bool:
        """Record an iterate and report whether the run has stalled.

        Call before taking the step, with the metric measured at ``x``.

        Parameters
        ----------
         metric : float
            The progress metric at ``x``: max(η, ‖r_dual‖), plus ‖r_pri‖ on the equality
            path.
         nit : int
            Iteration index of ``x``.
         x : npt.NDArray[np.float64]
            Current primal iterate; copied when it is the best so far.
         lmbda : npt.NDArray[np.float64]
            Current inequality duals; copied when ``x`` is the best so far.
         nu : npt.NDArray[np.float64] | None
            Current equality multipliers, or ``None`` on the inequality-only path.

        Returns
        -------
         stalled : bool
            True when ``patience`` consecutive iterations have passed without a relative
            improvement of ``rel_improvement`` and the best metric is below
            ``soft_tolerance``.

        """
        if metric < self._best * (1.0 - self._rel_improvement):
            self._since_best = 0
        else:
            self._since_best += 1
        # Track the best iterate on any improvement, however small: if the bail
        # does fire, the point of it is to hand back the best thing seen.
        if metric < self._best:
            self._best = metric
            self.best_x = x.copy()
            self.best_lmbda = lmbda.copy()
            self.best_nu = None if nu is None else nu.copy()
            self.best_nit = nit
        return (
            self._since_best >= self._patience
            and self._best < self._soft_tolerance
            and self.best_x is not None
        )

    @property
    def within_soft_tolerance(self) -> bool:
        """Return whether the best iterate seen is good enough to return.

        Returns
        -------
         within : bool
            True when an iterate has been recorded and its metric is below the soft
            tolerance.

        """
        return self.best_x is not None and self._best < self._soft_tolerance

    @property
    def best_metric(self) -> float:
        """Return the best metric seen so far.

        Returns
        -------
         best : float
            The smallest metric passed to ``observe``; inf before the first call.

        """
        return self._best


@dataclass
class _PdStepInfo:
    """Diagnostics for one accepted Mehrotra predictor-corrector step.

    Returned by ``PrimalDualSolver._mpc_solve_step``; ``_run`` applies the step and
    passes the diagnostics to the trace.

    Fields
    ------
     delta_x : npt.NDArray[np.float64]
        Primal search direction, of length M.
     delta_lmbda : npt.NDArray[np.float64]
        Dual search direction, of length m.
     step : float
        Accepted step length along (``delta_x``, ``delta_lmbda``).
     next_damping : float
        Levenberg damping carried to the next iteration, after relaxation; 0 when the
        step was undamped.
     sigma : float
        The Mehrotra centering parameter.
     rho : float | None
        Predicted-versus-actual residual-reduction ratio, or ``None`` when the step was
        undamped and the ratio test was skipped.
     damping_used : float
        Levenberg damping of the accepted solve.
     attempts : int
        Number of damping growths before a feasible step was found (0 when the first try
        was accepted).

    """

    delta_x: npt.NDArray[np.float64]
    delta_lmbda: npt.NDArray[np.float64]
    step: float
    next_damping: float
    sigma: float
    rho: float | None
    damping_used: float
    attempts: int


def _clamp_dual_after_update(
    lmbda: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Clamp roundoff-scale negative duals to zero, raising on real leaks.

    Both ``PrimalDualSolver`` paths cap the step with the Mehrotra fraction-to-boundary
    factor γ = max(τ_min, 1 - η / m), which keeps λ + s Δλ > 0 componentwise in exact
    arithmetic. Floating-point rounding can still leave a component slightly negative,
    and γ relaxes toward 1 in the convergence tail, so such drift is clamped to zero. A
    negative component beyond the drift tolerance
       1e-12 max(1, ‖λ‖_∞)
    means the cap is not doing its job (a bug), so this raises instead. The max(1, ...)
    floor keeps the tolerance meaningful when λ is small.

    Non-finite entries are rejected before any comparison: np.min returns NaN when any
    entry is NaN, and NaN < 0 is False, so a NaN in ``delta_lmbda`` would otherwise pass
    through the clamp silently.

    Parameters
    ----------
     lmbda : npt.NDArray[np.float64]
        Updated inequality duals λ + s Δλ, of length m.

    Returns
    -------
     lmbda : npt.NDArray[np.float64]
        The duals with roundoff-scale negative entries set to zero; the input itself
        when no entry is negative.

    Raises
    ------
     NewtonStepError
        When any entry is NaN or inf, or when some entry is below -1e-12 max(1, ‖λ‖_∞).

    """
    if not np.all(np.isfinite(lmbda)):
        n_bad = int(np.sum(~np.isfinite(lmbda)))
        raise NewtonStepError(
            f"Inequality multipliers contain non-finite entries: "
            f"{n_bad} of {lmbda.size} are NaN or Inf. The BTLS update "
            "almost certainly received a non-finite ``delta_lmbda`` -- "
            "check the upstream Newton step for division by zero or "
            "ill-conditioned solves."
        )
    lmbda_min = float(np.min(lmbda)) if lmbda.size > 0 else 0.0
    if lmbda_min < 0.0:
        lmbda_norm_inf = float(np.max(np.abs(lmbda)))
        scale = max(1.0, lmbda_norm_inf)
        if lmbda_min < -1e-12 * scale:
            raise NewtonStepError(
                f"Inequality multipliers leaked negative beyond drift "
                f"tolerance: min(lmbda) = {lmbda_min:.3e}, "
                f"||lmbda||_inf = {lmbda_norm_inf:.3e}, "
                f"drift tolerance = {1e-12 * scale:.3e} "
                f"(= 1e-12 * max(1, ||lmbda||_inf)). BTLS Stage 1's "
                "dual-positivity cap should have prevented this; the cap "
                "may be miscomputed."
            )
        lmbda = np.maximum(lmbda, 0.0)
    return lmbda


def _verify_recession_direction(
    adapter: _BarrierAdapter,
    x: npt.NDArray[np.float64],
    delta_x: npt.NDArray[np.float64],
    feasibility_error: float,
    tolerance: float,
) -> npt.NDArray[np.float64] | None:
    """Verify a descending recession ray from a feasible iterate."""
    if not feasibility_error <= tolerance:
        return None
    delta_x = adapter.prepare_recession_direction(delta_x)
    norm = float(np.linalg.norm(delta_x))
    if 0.0 < norm < math.inf:
        d = delta_x / norm
        slope = adapter.recession_slope(x, d)
        if slope is not None and slope < 0.0:
            return d
    witness = adapter.find_structured_recession_direction(x)
    if witness is not None:
        return witness[0]
    return None


def _pd_scales(
    outer: OuterSettings,
    problem: Problem,
    x: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64] | None = None,
    b: npt.NDArray[np.float64] | None = None,
) -> tuple[float, float, float]:
    """Return the scales of the gap, dual-residual and equality-residual tests.

    All three are 1 under the default absolute ``objective_scale`` (``None``), so
    nothing is evaluated; see ``OuterSettings.objective_scale`` for the others.

    Parameters
    ----------
     outer : OuterSettings
        The settings carrying ``objective_scale``.
     problem : Problem
        The problem, whose objective sets the gap and dual scales.
     x : npt.NDArray[np.float64]
        The iterate.
     A : npt.NDArray[np.float64] | None
        The equality matrix, or ``None`` without equality constraints.
     b : npt.NDArray[np.float64] | None
        The equality right-hand side, or ``None``.

    Returns
    -------
     scales : tuple[float, float, float]
        The gap, dual-residual and equality-residual scales, each positive.

    """
    if outer.objective_scale is None:
        return 1.0, 1.0, 1.0
    obj = none_throws(problem.obj)
    gradient = obj.gradient(x)
    gap = _gap_scale(outer, float(obj.evaluate(x)), gradient, x)
    dual = _dual_scale(outer, gradient)
    if A is None or b is None:
        return gap, dual, 1.0
    row_scale = np.asarray(abs(A) @ np.abs(x), dtype=np.float64)
    return gap, dual, _primal_scale(outer, b, row_scale)


def _certify_pd_status(
    status: Literal[0, 1, 2],
    message: str,
    objective_value: float,
    dual_value: float,
    feasibility_error: float,
    outer: OuterSettings,
    gap_scale: float = 1.0,
    feasibility_scale: float = 1.0,
) -> tuple[Literal[0, 1, 2], str]:
    """Require a finite bound and a feasible primal point for a success status.

    The gap and the feasibility error are measured against their scales (see
    ``OuterSettings.objective_scale``), both 1 by default.
    """
    gap = objective_value - dual_value
    if (
        math.isfinite(gap)
        and math.isfinite(feasibility_error)
        and gap >= -_weak_duality_slack(objective_value)
    ):
        error = max(gap / gap_scale, feasibility_error / feasibility_scale)
        if status == 0 and error <= outer.tolerance:
            return 0, message
        if error <= outer.tolerance_soft:
            if status == 0:
                return (
                    1,
                    f"Certified gap and primal feasibility reached soft tolerance ({error:.3e}).",
                )
            return 1, message
    diagnostic = (
        f"Optimality was not certified (primal-dual gap {gap:.3e}, "
        f"primal feasibility error {feasibility_error:.3e})."
    )
    return 2, diagnostic if status == 0 else f"{message} {diagnostic}"


def _pd_solveresult(
    *,
    solution: npt.NDArray[np.float64],
    objective_value: float,
    dual_value: float,
    equality_multipliers: npt.NDArray[np.float64],
    inequality_multipliers: npt.NDArray[np.float64],
    suboptimality: float,
    nits: int,
    status: Literal[0, 1, 2],
    message: str,
    surrogate_gaps: list[float],
    step_sizes: list[float],
    primal_residual_norms: list[float],
    dual_residual_norms: list[float],
    direction: npt.NDArray[np.float64] | None = None,
) -> SolveResult:
    """Build a primal-dual ``SolveResult`` from the iteration's outputs and histories.

    Shared by every result-construction site in ``PrimalDualSolver``. The per-iteration
    histories populate ``PrimalDualDiagnostics``.

    Parameters
    ----------
     solution : npt.NDArray[np.float64]
        The returned primal iterate, of length M.
     objective_value : float
        f0 at ``solution``.
     dual_value : float
        Lagrange dual function value (a lower bound on the optimal value).
     equality_multipliers : npt.NDArray[np.float64]
        Equality multipliers ν, of length p (empty without equality constraints).
     inequality_multipliers : npt.NDArray[np.float64]
        Inequality duals λ, of length m.
     suboptimality : float
        Surrogate duality gap at ``solution``.
     nits : int
        Number of iterations run.
     status : Literal[0, 1, 2]
        Integer status code, mapped by ``_legacy_solve_status``: 0 to ``OPTIMAL``, 1 to
        ``ACCEPTABLE``, 2 to ``MAX_ITERATIONS``.
     message : str
        Human-readable termination message.
     surrogate_gaps : list[float]
        Surrogate gap η at each iteration.
     step_sizes : list[float]
        Accepted step length at each iteration.
     primal_residual_norms : list[float]
        ‖A x - b‖ at each iteration (zeros without equality constraints).
     dual_residual_norms : list[float]
        ‖r_dual‖ at each iteration.
     direction : npt.NDArray[np.float64] | None
        Verified recession direction for an unbounded problem, if present.

    Returns
    -------
     result : SolveResult
        The result, with ``solver="primal_dual"``.

    """
    certificate = None
    result_status = _legacy_solve_status(status)
    if direction is not None:
        result_status = SolveStatus.UNBOUNDED
        certificate = InfeasibilityCertificate(
            kind="unbounded", primal_witness=direction
        )
    return SolveResult(
        solution=solution,
        status=result_status,
        certificate=certificate,
        message=message,
        solver="primal_dual",
        n_iterations=nits,
        objective=ObjectiveOutcome(
            value=objective_value,
            suboptimality=suboptimality,
            dual_value=dual_value,
        ),
        duals=_duals_from(equality_multipliers, inequality_multipliers),
        diagnostics=PrimalDualDiagnostics(
            surrogate_gaps=tuple(surrogate_gaps),
            step_sizes=tuple(step_sizes),
            primal_residual_norms=tuple(primal_residual_norms),
            dual_residual_norms=tuple(dual_residual_norms),
        ),
    )


class _PrimalDualKernel:
    """Stateless primitives shared by both ``PrimalDualSolver`` paths.

    Collects the stacked constraint values and Jacobian products that the Mehrotra
    (no-equality) and damped-Newton (equality) paths both use, as static methods over a
    ``Problem`` plus iterate and multiplier inputs. Private to the primal-dual solver.

    """

    @staticmethod
    def concat_constraints(
        problem: Problem,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Stack the constraint values f(x) of every inequality layer.

        Parameters
        ----------
         problem : Problem
            The problem whose inequality layers are evaluated.
         x : npt.NDArray[np.float64]
            Point at which the constraints are evaluated, of length M.

        Returns
        -------
         f : npt.NDArray[np.float64]
            The constraint values, of length m, in layer order.

        """
        return np.concatenate([layer.constraints(x) for layer in problem.ineq])

    @staticmethod
    def gradient_residual(
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint force G^T λ, summed across layers.

        Parameters
        ----------
         problem : Problem
            The problem whose inequality layers supply the Jacobian G.
         x : npt.NDArray[np.float64]
            Point at which G is evaluated, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m, in layer order.

        Returns
        -------
         GTlmbda : npt.NDArray[np.float64]
            The product G^T λ, of length M.

        """
        result = np.zeros(problem.dimension)
        offset = 0
        for layer in problem.ineq:
            n_c = layer.num_constraints(problem.dimension)
            result = result + layer.grad_constraints_transpose_multiply(
                x, lmbda[offset : offset + n_c]
            )
            offset += n_c
        return result

    @staticmethod
    def g_times(
        problem: Problem,
        x: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the stacked constraint Jacobian G.

        Parameters
        ----------
         problem : Problem
            The problem whose inequality layers supply G.
         x : npt.NDArray[np.float64]
            Point at which G is evaluated, of length M.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length m, in layer order.

        """
        return np.concatenate(
            [layer.grad_constraints_multiply(x, v) for layer in problem.ineq]
        )

    @staticmethod
    def g_transpose_times(
        problem: Problem,
        x: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed stacked constraint Jacobian.

        Computes ∑_k G_k^T u_k, where G_k is layer k's Jacobian and u_k its slice of u.

        Parameters
        ----------
         problem : Problem
            The problem whose inequality layers supply G.
         x : npt.NDArray[np.float64]
            Point at which G is evaluated, of length M.
         u : npt.NDArray[np.float64]
            Vector of length m to multiply, in layer order.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        result = np.zeros(problem.dimension)
        offset = 0
        for layer in problem.ineq:
            n_c = layer.num_constraints(problem.dimension)
            result = result + layer.grad_constraints_transpose_multiply(
                x, u[offset : offset + n_c]
            )
            offset += n_c
        return result

    @staticmethod
    def validate_pd_iterate(
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> None:
        """Check that the iterate is valid before the modified Hessian is assembled.

        The modified Hessian weights the constraints by λ / (-f); a non-finite or
        wrong-sign weight silently propagates non-finite entries into the Newton step,
        surfacing far from the cause. A valid iterate has f < 0 (strict primal
        feasibility) and λ ≥ 0 (dual feasibility) entrywise, both finite. The
        step-length rules maintain both, so a violation means an invalid iterate slipped
        through.

        Parameters
        ----------
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values, of length m.

        Raises
        ------
         NewtonStepError
            When some f_i is not finite or not strictly negative, or some λ_i is not
            finite or negative.

        """
        if not (np.all(np.isfinite(f)) and np.all(f < 0)):
            raise NewtonStepError(
                "PD slacks f are not strictly negative or not finite; the "
                "iterate is not strictly feasible. BTLS should have raised "
                "before the primal-dual Hessian was assembled."
            )
        if not (np.all(np.isfinite(lmbda)) and np.all(lmbda >= 0)):
            raise NewtonStepError(
                "PD multipliers lmbda are negative or not finite; the dual "
                "iterate has left the non-negative orthant."
            )


class PrimalDualSolver(Solver):
    """Primal-dual interior-point method for inequality-constrained problems.

    Solves
       minimize    f0(w)
       subject to  f_i(w) ≤ 0,   i = 1, ..., m,
                   A w = b       (optional),
    It updates the primal iterate and the explicit duals λ (and ν, with equality
    constraints) jointly by Newton steps on the perturbed KKT residual, following the
    primal-dual interior-point method (Boyd and Vandenberghe, 2004, section 11.7).

    ``solve`` dispatches on ``problem.eq``. Without equality constraints it runs a
    Mehrotra predictor-corrector iteration (``_run``). With A w = b it runs an
    infeasible-start iteration (``_run_equality``) that accepts an ``x0`` off the
    equality manifold and drives r_pri = A x - b to zero together with the dual and
    centering residuals. Either way the iterate must be strictly inequality-feasible,
    f_i(x0) < 0; the feasibility cascade builds such a start when ``x0`` is omitted or
    infeasible.

    Compared with ``BarrierSolver`` (the barrier method on the same ``Problem``), the
    iteration count tends to vary less with problem conditioning, which suits workflows
    that value predictable behavior. The per-iteration cost is comparable: the modified
    Hessian has the same structure as the barrier Hessian, so the same structured
    ``Objective`` and ``ConstraintLayer`` assembly and solvers apply.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses the defaults. The solver reads the outer
        tolerance, soft tolerance and barrier multiplier μ, the inner iteration cap and
        stall settings, the Armijo parameters, ``pd_fraction_to_boundary_min`` (τ_min),
        the ``primal_dual.damping`` schedule, and the trace.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    def solve(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Solve a problem by the primal-dual method, dispatching on equality.

        A feasibility problem (no objective) goes straight to the feasibility cascade
        (``Solver._solve_feasibility``), which runs through this solver; a result from
        the already-feasible short-circuit is relabeled as a primal-dual result.
        Otherwise the cascade first finds a strictly inequality-feasible start, ignoring
        the objective, and the start goes to ``_run`` (no equality constraints) or
        ``_run_equality``. Equality feasibility is not required of ``x0``, since the
        equality path drives A x - b to zero; this solver therefore keeps the default
        ``_validate_x0``.

        Parameters
        ----------
         problem : Problem
            The problem to solve; must have at least one inequality layer.
         x0 : npt.NDArray[np.float64] | None
            Initial guess, of length M. ``None`` or an inequality-infeasible point
            triggers Phase I.
         on_infeasible : OnInfeasible
            Accepted for ``Solver`` interface uniformity; only ``"raise"`` is supported.

        Returns
        -------
         result : SolveResult
            The solution, multipliers and primal-dual diagnostics.

        Raises
        ------
         NotImplementedError
            When ``on_infeasible`` is ``"continue"``.
         ValueError
            When ``problem.ineq`` is empty; use ``NewtonSolver`` for unconstrained or
            equality-only problems.
         ProblemCertifiablyInfeasibleError
            Propagated from ``Solver._solve_feasibility`` when no strictly feasible
            start exists, and from ``_run`` when the final dual value is positive.
         NewtonStepError
            Propagated from the step computation when no usable step can be found or an
            iterate becomes invalid.

        """
        if on_infeasible == "continue":
            raise NotImplementedError(
                "PrimalDualSolver does not support on_infeasible='continue'; use "
                "BarrierSolver (the IPM engine) for the continue policy."
            )
        if not problem.ineq:
            raise ValueError(
                "PrimalDualSolver requires at least one inequality constraint "
                "layer; use NewtonSolver for unconstrained or equality-only "
                "problems."
            )

        # A feasibility problem (no objective) is solved directly via the
        # feasibility cascade, which runs through this primal-dual technique.
        # This is shared by both the no-equality and equality paths.
        if problem.obj is None:
            result = self._solve_feasibility(problem, x0)
            # A PD feasibility solve carries PD diagnostics; an already-feasible
            # short-circuit returns a bare feasibility result, which is lifted into
            # a PD-flavored result here.
            if isinstance(result.diagnostics, PrimalDualDiagnostics):
                return result
            lifted = self._feasible_result(problem, np.asarray(result.solution))
            return replace(
                lifted,
                solver="primal_dual",
                diagnostics=PrimalDualDiagnostics(),
                phase1=result.phase1,  # keep the earlier layers' cascade
            )

        feasibility = self._solve_feasibility(replace(problem, obj=None), x0)
        x = np.asarray(feasibility.solution)
        # Dispatch on equality presence: the no-equality Mehrotra step, or its
        # infeasible-start form that also drives A x - b to zero.
        result = (
            self._run(problem, x)
            if problem.eq is None
            else self._run_equality(problem, x)
        )
        return replace(result, phase1=self._phase1_part(feasibility))

    def _run(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64],
    ) -> SolveResult:
        """Run the Mehrotra predictor-corrector iteration (no equality constraints).

        Starts from zero damping and λ on the central path's direction at the
        objective's scale when λ = 1 is the wrong scale (``_central_path_duals``;
        λ = 1 in Phase I). Each iteration
        evaluates the surrogate gap
        and the dual residual,
           η = -f^T λ,   r_dual = ∇f0(x) + G^T λ,
        sets t = μ m / η, and takes one damped Mehrotra predictor-corrector step
        (``_mpc_solve_step``).

        Parameters
        ----------
         problem : Problem
            The problem, with an objective and no equality constraints.
         x0 : npt.NDArray[np.float64]
            Strictly feasible starting point, of length M.

        Returns
        -------
         result : SolveResult
            The solution, multipliers, dual value and per-iteration diagnostics.

        Raises
        ------
         ProblemCertifiablyInfeasibleError
            When the dual value at the final iterate is positive, certifying
            infeasibility; the exception carries a result with status ``INFEASIBLE``.
         NewtonStepError
            Propagated from ``_mpc_solve_step`` and ``_clamp_dual_after_update``, unless
            the best iterate seen is already within the soft tolerance (see Notes).

        Notes
        -----
        The loop stops at the first of:
        - the objective falling below its feasibility threshold, if it has one
          (status 2; the duals are unrefined, so the dual value and suboptimality are
          reported as zero);
        - η ≤ 0 (status 0);
        - η < tol and ‖r_dual‖ < tol (status 0);
        - a stall detected by ``_StallWatch`` on max(η, ‖r_dual‖), which returns the
          best iterate seen (status 1);
        - a Newton system that breaks down (``NewtonStepError``) once the best
          iterate seen is within the soft tolerance, which returns that iterate
          (status 1). Near a degenerate optimum, where λ_i / s_i diverges on several
          constraints at once, the low-rank update's capacitance matrix can lose
          definiteness in floating point after the iterates are already good; before
          that point the error propagates;
        - the iteration cap (status 1).
        After the loop, the dual value is evaluated with the barrier method's
        warm-started mini-Newton (``_BarrierAdapter.evaluate_dual``), a lower bound on
        inf_x L(x, λ). A plug-in value of L at the current x would only be an upper
        bound, so an infeasibility test built on it could wrongly certify a feasible
        problem infeasible.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        settings = (
            self.settings if self.settings is not None else OptimizationSettings()
        )
        m = problem.num_inequality_constraints()
        mu = settings.outer.barrier_multiplier
        outer_tol = settings.outer.tolerance
        max_iter = settings.inner.max_iterations
        beta = settings.armijo.beta
        min_step = settings.armijo.min_step
        tau_min = settings.pd_fraction_to_boundary_min
        lm = settings.primal_dual.damping

        x = x0.copy()
        # Start the duals at the objective's scale; in Phase I, whose objective is the
        # unit-scaled slack, λ = 1 already is (see ``_run_equality``).
        lmbda = (
            np.ones(m)
            if none_throws(problem.obj).feasibility_threshold is not None
            else self._central_path_duals(problem, x)
        )
        # Levenberg damping, carried across iterations; 0 = exact Newton step.
        damping = 0.0

        surrogate_gaps: list[float] = []
        step_sizes: list[float] = []
        dual_residual_norms: list[float] = []
        primal_residual_norms: list[float] = []
        status: Literal[0, 1, 2] = 0
        message = "Primal-dual method converged."
        stall = _StallWatch(
            patience=settings.inner.stall_patience,
            rel_improvement=settings.inner.stall_rel_improvement,
            soft_tolerance=settings.outer.tolerance_soft,
        )

        trace = settings.trace
        label = settings.observability.label
        solver_name = label if label is not None else self.__class__.__name__
        overall_start_time = time.time()
        trace.on_pd_start(
            solver_name, num_vars=problem.dimension, num_ineq=m, tol=outer_tol
        )

        # Feasibility-objective early-stop threshold: when minimizing a threshold-
        # carrying objective, the solve answers a feasibility question and halts
        # once the value crosses the threshold. ``None`` for an ordinary objective.
        feasibility_threshold = none_throws(problem.obj).feasibility_threshold

        adapter = _BarrierAdapter(problem, settings)
        check_ray = feasibility_threshold is None
        direction = None
        nit = 0
        for nit in range(max_iter):
            iter_start = time.time()
            if (
                feasibility_threshold is not None
                and none_throws(problem.obj).feasibility_residual(x)
                < feasibility_threshold
            ):
                status = 2
                message = (
                    "Feasibility objective crossed its threshold; "
                    "the problem is feasible"
                )
                # Mirror the barrier loop's early-stop trace event so tracing
                # consumers can distinguish a feasibility stop from an ordinary
                # status=2 exit across all three solver paths.
                trace.on_early_stop(nit + 1)
                break
            f = _PrimalDualKernel.concat_constraints(problem, x)
            r_dual = self._compute_r_dual(problem, x, lmbda)

            eta = float(-np.dot(f, lmbda))
            dual_norm = float(np.linalg.norm(r_dual))
            residual_norm = float(np.sqrt(dual_norm * dual_norm + eta * eta))
            surrogate_gaps.append(eta)
            dual_residual_norms.append(dual_norm)
            primal_residual_norms.append(0.0)  # no equality constraints
            gap_scale, dual_scale, _ = _pd_scales(settings.outer, problem, x)

            if eta <= 0.0:
                status = 0 if dual_norm < outer_tol * dual_scale else 1
                message = (
                    f"Iteration stopped at non-positive surrogate gap ({eta:.3e})."
                )
                break

            if eta < outer_tol * gap_scale and dual_norm < outer_tol * dual_scale:
                status = 0
                message = (
                    f"converged: surrogate gap {eta:.3e} and ||r_dual|| "
                    f"{dual_norm:.2e} both < tol {outer_tol:.1e}"
                )
                break

            # Both quantities above have to clear ``outer_tol`` together, and
            # they can stop being able to -- see ``_StallWatch``. Judge progress
            # on the worse of the two, and hand back the best iterate rather
            # than the last, which is not the same thing once dual feasibility
            # starts drifting back up.
            if stall.observe(
                max(eta / gap_scale, dual_norm / dual_scale), nit, x, lmbda
            ):
                status = 1
                message = (
                    f"stalled: no relative improvement in max(gap, ||r_dual||) "
                    f"over {settings.inner.stall_patience} iterations; "
                    f"returning best iterate from iteration {stall.best_nit} "
                    f"(max(gap, ||r_dual||)={stall.best_metric:.3e})"
                )
                # The returned iterate is ``best_nit``'s, while ``nits`` and the
                # history lists run to here -- so ``suboptimality`` will not
                # match ``surrogate_gaps[-1]``. That is deliberate, and matches
                # what BarrierSolver's stall return already does. ``nits``
                # answers "how much work did this cost", which is the detection
                # point; the histories are the evidence *of* the stall, and
                # truncating them at the best iterate would throw away the
                # drifting tail that shows why the solver gave up. The message
                # names ``best_nit`` so the two can be lined up.
                x = none_throws(stall.best_x)
                lmbda = none_throws(stall.best_lmbda)
                trace.on_early_stop(nit + 1)
                break

            t = mu * m / eta
            r_cent = -lmbda * f - 1.0 / t
            # True KKT residual r_t = [r_dual; r_cent] (no equality block here);
            # the surrogate-gap-based ``residual_norm`` above is a separate proxy
            # reported to the trace.
            cent_norm = float(np.linalg.norm(r_cent))
            rt_norm = float(np.sqrt(dual_norm * dual_norm + cent_norm * cent_norm))

            dir_start = time.time()
            try:
                info = self._mpc_solve_step(
                    problem,
                    x,
                    lmbda,
                    f,
                    r_dual,
                    r_cent,
                    eta=eta,
                    t=t,
                    beta=beta,
                    min_step=min_step,
                    tau_min=tau_min,
                    damping=damping,
                    lm=lm,
                )
            except NewtonStepError as e:
                # Near a degenerate optimum the Newton system can break down
                # after the iterates are already good enough. Hand back the best
                # one, as a stall would, rather than discard it.
                if not stall.within_soft_tolerance:
                    raise
                status = 1
                message = (
                    f"the Newton system broke down ({e}); returning best iterate "
                    f"from iteration {stall.best_nit} "
                    f"(max(gap, ||r_dual||)={stall.best_metric:.3e})"
                )
                x = none_throws(stall.best_x)
                lmbda = none_throws(stall.best_lmbda)
                trace.on_early_stop(nit + 1)
                break
            dir_ms = 1000.0 * (time.time() - dir_start)
            damping = info.next_damping
            if info.attempts > 0:
                trace.on_pd_damping_engaged(nit, info.damping_used, info.attempts)
            if check_ray:
                direction = _verify_recession_direction(
                    adapter, x, info.delta_x, float(np.max(f, initial=0.0)), outer_tol
                )
                if direction is not None:
                    status = 2
                    message = (
                        "Verified a feasible recession direction with negative slope."
                    )
                    break
            step_sizes.append(info.step)
            x = x + info.step * info.delta_x
            lmbda = _clamp_dual_after_update(lmbda + info.step * info.delta_lmbda)
            # Post-step KKT residual (at the same ``t``) for the line-search view.
            # Costs one extra residual evaluation, so gate it behind opt-in
            # detail logging -- mirrors BarrierSolver's ``want_btls_details``.
            rt_new = (
                self._residual_norm(problem, x, lmbda, t)
                if trace.want_pd_step_details()
                else None
            )
            trace.on_pd_iteration(
                nit=nit,
                eta=eta,
                dual_norm=dual_norm,
                pri_norm=0.0,
                residual_norm=residual_norm,
                t=t,
                step=info.step,
                elapsed_ms=1000.0 * (time.time() - iter_start),
                sigma=info.sigma,
                rho=info.rho,
                damping=info.damping_used,
                cent_norm=cent_norm,
                rt_norm=rt_norm,
                dir_ms=dir_ms,
                rt_new=rt_new,
            )
        else:
            status = 1
            message = f"iteration cap ({max_iter}) reached without convergence"

        # Feasibility early-stop short-circuit: the loop broke the instant the
        # objective crossed its threshold, before ``lmbda`` was refined past its
        # ``np.ones(m)`` seed. The dual-side fields (``dual_value``,
        # ``suboptimality``, residual norms) would only echo that arbitrary
        # multiplier, so report the meaningful primal certificate (``solution`` /
        # ``objective_value`` / ``status``) and zero the dual fields rather than
        # evaluate them on an unrefined multiplier.
        if status == 2 and direction is None:
            objective_value = none_throws(problem.obj).evaluate(x)
            trace.on_pd_done(
                solver=solver_name,
                nits=nit + 1,
                status=status,
                elapsed_ms=1000.0 * (time.time() - overall_start_time),
                reason=message,
            )
            # The objective crossed its feasibility threshold: report the point as
            # FEASIBLE, not by the integer code 2, which maps to MAX_ITERATIONS
            # (issue #52).
            return replace(
                _pd_solveresult(
                    solution=x,
                    objective_value=float(objective_value),
                    dual_value=0.0,
                    equality_multipliers=np.zeros(0),
                    inequality_multipliers=lmbda,
                    suboptimality=0.0,
                    nits=nit + 1,
                    status=status,
                    message=message,
                    surrogate_gaps=surrogate_gaps,
                    step_sizes=step_sizes,
                    primal_residual_norms=primal_residual_norms,
                    dual_residual_norms=dual_residual_norms,
                ),
                status=SolveStatus.FEASIBLE,
            )

        # Surrogate-gap proxy at the solution (r_dual and r_primal not re-evaluated).
        f_final = _PrimalDualKernel.concat_constraints(problem, x)
        eta_final = float(-np.dot(f_final, lmbda))

        objective_value = none_throws(problem.obj).evaluate(x)
        # Lagrangian dual via the barrier method's mini-Newton evaluator -- a
        # guaranteed lower bound on inf_x L (B&V Eq 9.42). Reuses the exact
        # structured ``_BarrierAdapter.evaluate_dual`` the BarrierSolver uses,
        # rather than the analytic per-component decomposition. The analytic
        # path's plug-in fallback (``L`` at the current ``x``) is only an UPPER
        # bound on inf_x L, so a ``dual_value > 0`` infeasibility test built on
        # it could falsely certify a feasible problem infeasible. ``nu`` is
        # empty (no equality constraints in this solver).
        dual_value = (
            adapter.evaluate_dual(lmbda, np.zeros(0), x)
            if direction is None
            else float("-inf")
        )
        # A positive dual value certifies infeasibility only for a Phase I problem,
        # whose objective is the slack (marked by a feasibility threshold); for any
        # other problem it is just a positive lower bound on the optimal value.
        if feasibility_threshold is not None and dual_value > 0.0:
            trace.on_pd_done(
                solver=solver_name,
                nits=nit + 1,
                status=2,
                elapsed_ms=1000.0 * (time.time() - overall_start_time),
                reason="certifiably infeasible (dual value > 0)",
            )
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    f"Problem certifiably infeasible: dual value was {dual_value} > 0.0"
                ),
                result=replace(
                    _pd_solveresult(
                        solution=x,
                        objective_value=objective_value,
                        dual_value=dual_value,
                        equality_multipliers=np.zeros(0),
                        inequality_multipliers=lmbda,
                        suboptimality=eta_final,
                        nits=nit + 1,
                        status=2,
                        message="Infeasibility certified.",
                        surrogate_gaps=[],
                        step_sizes=[],
                        primal_residual_norms=[],
                        dual_residual_norms=[],
                    ),
                    status=SolveStatus.INFEASIBLE,
                ),
            )

        if direction is None:
            gap_scale, _, _ = _pd_scales(settings.outer, problem, x)
            status, message = _certify_pd_status(
                status,
                message,
                objective_value,
                dual_value,
                float(np.max(f_final, initial=0.0)),
                settings.outer,
                gap_scale=gap_scale,
            )

        trace.on_pd_done(
            solver=solver_name,
            nits=nit + 1,
            status=status,
            elapsed_ms=1000.0 * (time.time() - overall_start_time),
            reason=message,
            duality_gap=float(objective_value - dual_value),
        )
        return _pd_solveresult(
            solution=x,
            objective_value=float(objective_value),
            dual_value=float(dual_value),
            equality_multipliers=np.zeros(0),
            inequality_multipliers=lmbda,
            suboptimality=eta_final if direction is None else float("inf"),
            direction=direction,
            nits=nit + 1,
            status=status,
            message=message,
            surrogate_gaps=surrogate_gaps,
            step_sizes=step_sizes,
            primal_residual_norms=primal_residual_norms,
            dual_residual_norms=dual_residual_norms,
        )

    def _compute_r_dual(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the dual residual r_dual = ∇f0(x) + G^T λ.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.

        Returns
        -------
         r_dual : npt.NDArray[np.float64]
            The dual residual, of length M.

        """
        return none_throws(problem.obj).gradient(
            x
        ) + _PrimalDualKernel.gradient_residual(problem, x, lmbda)

    def _compute_pd_step(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        damping: float = 0.0,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the modified KKT system for the primal and dual steps (Δx, Δλ).

        After eliminating Δλ, the system is
           (H_pd + damping I) Δx = -r_dual - G^T (r_cent / f),
           H_pd = ∇^2 f0 + ∑ λ_i ∇^2 f_i - G^T diag(λ / f) G,
        which has the same shape as the barrier method's Newton system. The modified
        Hessian is assembled by ``Problem.build_hessian_system`` and solved through the
        resulting structured system. Δλ is then recovered from the same Δx, so the pair
        solves the damped system consistently.

        The iteration itself uses ``_mpc_solve_step``; this single-solve form is the
        plain primal-dual Newton step.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, of length M.
         r_cent : npt.NDArray[np.float64]
            Centering residual -λ ∘ f - 1 / t, of length m.
         damping : float
            Levenberg term added to the modified Hessian as ``damping`` I; 0 gives the
            exact Newton step, and a positive value bounds ‖Δx‖ when H_pd is nearly
            singular.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            Primal step, of length M.
         delta_lmbda : npt.NDArray[np.float64]
            Dual step, of length m.

        Raises
        ------
         NewtonStepError
            Propagated from ``_PrimalDualKernel.validate_pd_iterate``.

        """
        wrapped = self._build_pd_solve(problem, x, lmbda, f, damping)
        return self._direction_from_rcent(problem, x, lmbda, f, r_dual, r_cent, wrapped)

    @staticmethod
    def _build_pd_solve(
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        damping: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Factor the (optionally damped) modified Hessian into a solve closure.

        Assembles
           H_pd + damping I = ∇^2 f0 + ∑ λ_i ∇^2 f_i - G^T diag(λ / f) G + damping I
        with ``Problem.build_hessian_system`` at the dual weights. The closure is reused
        for the predictor and corrector solves of one Mehrotra iteration (one
        factorization, two solves).

        Each solve takes one step of iterative refinement, Δx += H^{-1} (b - H Δx),
        reusing the factorization at the cost of one multiply and one more solve. Late
        in a solve the duals span many orders of magnitude, and when ∇^2 f0 is singular
        (a QP whose Q leaves some variables uncurved) the only curvature along its
        nullspace comes from the smallest of them, so H is nearly singular there. An
        unrefined step then carries enough error that the dual residual, which an
        exact step can only shrink, grows from one iteration to the next and the
        solve stalls short of its tolerance.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         damping : float
            Levenberg damping added to the diagonal; 0 for none.

        Returns
        -------
         solve : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            Maps a right-hand side of length M to the Δx solving the damped system.

        Raises
        ------
         NewtonStepError
            Propagated from ``_PrimalDualKernel.validate_pd_iterate``.

        """
        _PrimalDualKernel.validate_pd_iterate(lmbda, f)
        # H_pd = nabla^2 f0 + sum_i lambda_i C_i, assembled from the structured
        # contributions at the dual weights lmbda (objective_scale = 1). This is
        # exactly nabla^2 f0 + sum_i lambda_i nabla^2 f_i + G^T diag(lambda / -f) G.
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, lmbda
        )
        # Levenberg-Marquardt: add ``damping * I`` to the modified Hessian.
        if damping > 0.0:
            system = system.plus(DiagonalSystem(np.full(x.shape[0], damping)))

        def refined_solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            step = system.solve(rhs)
            return step + system.solve(rhs - system.multiply(step))

        return refined_solve

    @staticmethod
    def _direction_from_rcent(
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        wrapped: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Recover the steps (Δx, Δλ) for a given centering residual.

        The bottom KKT row,
           -diag(λ) G Δx - diag(f) Δλ = -r_cent,
        gives
           Δλ = (r_cent - λ ∘ (G Δx)) / f.
        Substituting into the top row,
           H Δx + G^T Δλ = -r_dual,   H = ∇^2 f0 + ∑ λ_i ∇^2 f_i,
        gives the reduced system that ``wrapped`` solves:
           (H - G^T diag(λ / f) G) Δx = -r_dual - G^T (r_cent / f).
        Since f < 0, the weights -λ / f are nonnegative, so the Jacobian term is
        positive semidefinite.

        Parameters
        ----------
         problem : Problem
            The problem whose layers supply G.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, of length M.
         r_cent : npt.NDArray[np.float64]
            Centering residual, of length m: the standard -λ ∘ f - 1 / t, or the
            Mehrotra predictor or corrector variant.
         wrapped : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            Solve closure for the reduced system, from ``_build_pd_solve``.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            Primal step, of length M.
         delta_lmbda : npt.NDArray[np.float64]
            Dual step, of length m.

        """
        rhs = -r_dual - _PrimalDualKernel.g_transpose_times(problem, x, r_cent / f)
        delta_x = wrapped(rhs)
        g_delta_x = _PrimalDualKernel.g_times(problem, x, delta_x)
        delta_lmbda = (r_cent - lmbda * g_delta_x) / f
        return delta_x, delta_lmbda

    def _residual_norm(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        t: float,
    ) -> float:
        """Compute the KKT residual norm ‖(r_dual, r_cent)‖ at a fixed t.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         t : float
            Barrier parameter defining r_cent = -λ ∘ f - 1 / t.

        Returns
        -------
         r_norm : float
            The Euclidean norm of the stacked residual.

        """
        f = _PrimalDualKernel.concat_constraints(problem, x)
        r_dual = self._compute_r_dual(problem, x, lmbda)
        r_cent = -lmbda * f - 1.0 / t
        return float(np.sqrt(np.dot(r_dual, r_dual) + np.dot(r_cent, r_cent)))

    @staticmethod
    def _relax_damping(
        damping: float, rho: float, x_scale: float, lm: LevenbergSettings
    ) -> float:
        """Shrink the Levenberg damping after an accepted step, snapping to zero.

        A ratio above ``lm.rho_good`` (the damped step behaved like the full Newton
        step) shrinks the damping by ``lm.shrink_good``; a ratio above ``lm.rho_ok``
        shrinks it by ``lm.shrink_ok``; a poorer ratio holds it. Damping below
        ``lm.drop_to_zero`` × ``x_scale`` becomes exactly zero, restoring the undamped
        step.

        Parameters
        ----------
         damping : float
            Damping of the accepted step.
         rho : float
            Predicted-versus-actual residual-reduction ratio of the step.
         x_scale : float
            Scale max(1, ‖x‖) for the snap-to-zero threshold.
         lm : LevenbergSettings
            The damping schedule.

        Returns
        -------
         damping : float
            Damping for the next iteration; 0 when the input was not positive or fell
            below the threshold.

        """
        if damping <= 0.0:
            return 0.0
        if rho > lm.rho_good:
            damping *= lm.shrink_good
        elif rho > lm.rho_ok:
            damping *= lm.shrink_ok
        if damping < lm.drop_to_zero * x_scale:
            return 0.0
        return damping

    def _mpc_direction(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        eta: float,
        wrapped: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64]
    ]:
        """Compute the Mehrotra predictor-corrector direction from a prebuilt solve.

        With slacks s = -f and μ = η / m:
        1. The affine predictor solves with r_cent = -λ ∘ f (zero complementarity
           target), giving (Δx_aff, Δλ_aff) and Δs_aff = -G Δx_aff.
        2. The fraction-to-boundary steps α_pri and α_dual on s and λ give
              μ_aff = (s + α_pri Δs_aff)^T (λ + α_dual Δλ_aff) / m,
           and the centering parameter σ = (μ_aff / μ)^3, clipped to [0, 1].
        3. The corrector solves with
              r_cent = -λ ∘ f - σ μ + Δλ_aff ∘ Δs_aff,
           which recenters toward σ μ and carries the second-order term.
        This is the predictor-corrector scheme of (Mehrotra, 1992). Both solves reuse
        ``wrapped`` (one factorization).

        Parameters
        ----------
         problem : Problem
            The problem whose layers supply G.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, of length M.
         eta : float
            Surrogate duality gap -f^T λ.
         wrapped : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            Solve closure for the reduced system, from ``_build_pd_solve``.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            Corrector primal step, of length M.
         delta_lmbda : npt.NDArray[np.float64]
            Corrector dual step, of length m.
         sigma : float
            The centering parameter σ.
         r_cent_cor : npt.NDArray[np.float64]
            Corrector centering residual, including the affine cross term.

        References
        ----------
        Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method,
        SIAM Journal on Optimization 2, 1992.

        """
        m = max(lmbda.size, 1)
        s = -f
        mu = eta / m
        dx_aff, dlam_aff = self._direction_from_rcent(
            problem, x, lmbda, f, r_dual, -lmbda * f, wrapped
        )
        ds_aff = -_PrimalDualKernel.g_times(problem, x, dx_aff)
        a_pri = fraction_to_boundary(s, ds_aff)
        a_dual = fraction_to_boundary(lmbda, dlam_aff)
        mu_aff = float(np.dot(s + a_pri * ds_aff, lmbda + a_dual * dlam_aff)) / m
        sigma = min(max((mu_aff / mu) ** 3 if mu > 0.0 else 0.0, 0.0), 1.0)
        r_cent_cor = -lmbda * f - sigma * mu + dlam_aff * ds_aff
        delta_x, delta_lmbda = self._direction_from_rcent(
            problem, x, lmbda, f, r_dual, r_cent_cor, wrapped
        )
        return delta_x, delta_lmbda, sigma, r_cent_cor

    @staticmethod
    def _feasible_step(
        problem: Problem,
        x: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        dlam: npt.NDArray[np.float64],
        gamma: float,
        beta: float,
        min_step: float,
    ) -> float | None:
        """Choose a fraction-to-boundary step that keeps every constraint negative.

        Starts from
           γ min(α_s, α_λ),
        where α_s is the fraction-to-boundary step on the slacks s = -f along their
        direction -G dx, α_λ the one on λ along ``dlam``, and γ is the Mehrotra boundary
        factor max(τ_min, 1 - η / m), which relaxes toward a full step as η / m → 0
        (Mehrotra, 1992; Wright, 1997). The step is then shrunk by ``beta`` until every
        constraint is strictly negative at the trial point, which the slack cap alone
        does not guarantee for curved constraints.

        The returned step may be smaller than ``min_step``: ``min_step`` bounds the
        depth of the shrink search, not the returned value. When the boundary cap itself
        is tiny (near a singular start, where a huge direction puts the boundary close),
        that tiny step is returned if feasible; a small feasible step lets the iteration
        continue without forcing damping.

        Parameters
        ----------
         problem : Problem
            The problem whose layers supply f and G.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         dx : npt.NDArray[np.float64]
            Primal direction, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         dlam : npt.NDArray[np.float64]
            Dual direction, of length m.
         gamma : float
            Mehrotra boundary factor γ.
         beta : float
            Shrink factor of the search.
         min_step : float
            Step below which the search stops shrinking.

        Returns
        -------
         step : float | None
            The largest step found with f(x + step dx) < 0: the γ-capped step when
            feasible, otherwise a ``beta``-shrunk one. ``None`` when even the final step
            is infeasible, which tells the caller to grow the damping.

        References
        ----------
        Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method,
        SIAM Journal on Optimization 2, 1992.

        Wright, Stephen J., Primal-Dual Interior-Point Methods, SIAM, 1997.

        """
        ds = -_PrimalDualKernel.g_times(problem, x, dx)  # change in slacks s = -f
        step = gamma * min(
            fraction_to_boundary(-f, ds),
            fraction_to_boundary(lmbda, dlam),
        )
        while step > min_step:
            if np.all(_PrimalDualKernel.concat_constraints(problem, x + step * dx) < 0):
                return step
            step *= beta
        # Final check at the floored step (loop-exhausted) or the sub-min_step
        # gamma cap (loop skipped): accept it if feasible, else give up (None).
        if np.all(_PrimalDualKernel.concat_constraints(problem, x + step * dx) < 0):
            return step
        return None

    @staticmethod
    def _grow_damping(
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        damping: float,
        x_scale: float,
        lm: LevenbergSettings,
    ) -> float:
        """Grow the Levenberg damping after a step was rejected.

        Multiplies positive damping by ``lm.grow``. From zero, initializes it to
           ‖rhs‖ / x_scale,   rhs = -r_dual - G^T (r_cent / f),
        the scale of the reduced right-hand side relative to the iterate.

        Parameters
        ----------
         problem : Problem
            The problem whose layers supply G.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m; unused by the computation.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, of length M.
         r_cent : npt.NDArray[np.float64]
            Standard centering residual -λ ∘ f - 1 / t, of length m.
         damping : float
            Damping of the rejected attempt.
         x_scale : float
            Scale max(1, ‖x‖).
         lm : LevenbergSettings
            The damping schedule.

        Returns
        -------
         damping : float
            The grown damping.

        """
        if damping > 0.0:
            return damping * lm.grow
        rhs = -r_dual - _PrimalDualKernel.g_transpose_times(problem, x, r_cent / f)
        return float(np.linalg.norm(rhs)) / x_scale

    def _mpc_solve_step(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        *,
        eta: float,
        t: float,
        beta: float,
        min_step: float,
        tau_min: float,
        damping: float,
        lm: LevenbergSettings,
    ) -> _PdStepInfo:
        """Take one Mehrotra predictor-corrector step with adaptive Levenberg damping.

        Tries the step at the incoming ``damping`` (0 on a healthy iterate, giving the
        undamped step). When no feasible step exists (``_feasible_step`` returns
        ``None``), it grows the damping and re-solves, up to ``lm.max_attempts`` times.
        After a damped success, the ratio
           ρ = (‖r_t‖ - ‖r_t(trial)‖) / (‖r_t‖ - ‖r_t(model)‖),
        with r_t = (r_dual, r_cent) at the fixed t, decides how far ``_relax_damping``
        relaxes the damping for the next iteration. The linear model uses the
        residual change induced by the damped predictor-corrector direction.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, of length M.
         r_cent : npt.NDArray[np.float64]
            Standard centering residual -λ ∘ f - 1 / t, of length m; used only for the
            ratio test and the damping initialization (the Mehrotra direction builds its
            own centering residuals).
         eta : float
            Surrogate duality gap.
         t : float
            Barrier parameter of the ratio test.
         beta : float
            Shrink factor of the feasibility search.
         min_step : float
            Depth bound of the feasibility search.
         tau_min : float
            Floor τ_min of the Mehrotra boundary factor.
         damping : float
            Damping carried from the previous iteration.
         lm : LevenbergSettings
            The damping schedule.

        Returns
        -------
         info : _PdStepInfo
            The accepted direction and step, with its trace diagnostics.

        Raises
        ------
         NewtonStepError
            When no feasible step is found after ``lm.max_attempts`` attempts; also
            propagated from ``_build_pd_solve``.

        """
        gamma = mehrotra_boundary_factor(eta, lmbda.size, tau_min)
        x_scale = max(1.0, float(np.linalg.norm(x)))
        r_norm0 = float(np.sqrt(np.dot(r_dual, r_dual) + np.dot(r_cent, r_cent)))
        for attempt in range(lm.max_attempts):
            wrapped = self._build_pd_solve(problem, x, lmbda, f, damping)
            dx, dlam, sigma, r_cent_cor = self._mpc_direction(
                problem, x, lmbda, f, r_dual, eta, wrapped
            )
            step = self._feasible_step(
                problem, x, f, dx, lmbda, dlam, gamma, beta, min_step
            )
            if step is None:
                damping = self._grow_damping(
                    problem, x, lmbda, f, r_dual, r_cent, damping, x_scale, lm
                )
                continue
            rho: float | None = None
            next_damping = 0.0
            if damping > 0.0:
                r_norm_new = self._residual_norm(
                    problem, x + step * dx, lmbda + step * dlam, t
                )
                # The damped KKT equations give J_dual d = -r_dual - damping dx
                # and J_cent d = -r_cent_cor. The merit residual still uses 1 / t
                # as its target, so its baseline centering block is r_cent.
                r_dual_model = (1.0 - step) * r_dual - (step * damping) * dx
                r_cent_model = r_cent - step * r_cent_cor
                r_norm_model = math.sqrt(
                    float(
                        np.dot(r_dual_model, r_dual_model)
                        + np.dot(r_cent_model, r_cent_model)
                    )
                )
                predicted = r_norm0 - r_norm_model
                rho = (r_norm0 - r_norm_new) / predicted if predicted > 0.0 else 0.0
                next_damping = self._relax_damping(damping, rho, x_scale, lm)
            return _PdStepInfo(
                delta_x=dx,
                delta_lmbda=dlam,
                step=step,
                next_damping=next_damping,
                sigma=sigma,
                rho=rho,
                damping_used=damping,
                attempts=attempt,
            )

        raise NewtonStepError(
            f"MPC failed to find a feasible step after {lm.max_attempts} "
            f"attempts (final damping={damping:.3e}); the modified Hessian "
            "could not be regularized into a usable step."
        )

    # ------------------------------------------------------------------
    # Equality path: infeasible-start PD for Problems carrying ``A w = b``.
    #
    # The infeasible-start counterpart of the no-equality MPC path above: it
    # carries the primal residual ``r_pri = A x - b`` and takes a Mehrotra
    # predictor-corrector step, routing the saddle solve through
    # ``problem.eq.saddle_system(system).solve_refined(g_x, g_nu)`` (non-zero
    # lower RHS). ``solve`` dispatches here when ``problem.eq is not None``.
    # The KKT Newton system after Schur elimination of ``Delta lambda`` is
    #
    #     | H_pd  A^T | | Dx  |   | -r_dual - G^T (r_cent / f) |
    #     |  A     0  | | Dnu | = |          -r_pri             |
    #
    # with ``H_pd = H + G^T diag(lambda / -f) G`` (same modified Hessian as the
    # no-equality variant). The equality block has a non-zero RHS when ``x`` is
    # equality-infeasible; the solve is routed through the equality's
    # ``saddle_system`` seam (the saddle's ``b_bottom``).
    # ------------------------------------------------------------------

    @staticmethod
    def _central_path_duals(
        problem: Problem, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Choose starting duals: λ = 1, rescaled when that is the wrong scale.

        The unit start λ = 1 is kept unless its constraint force ‖G^T 1‖ differs from
        the objective's ‖∇f0‖ by more than a factor ``_DUAL_SCALE_RANGE``. Then λ = 1
        is the wrong scale outright (on an objective scaled by 1e3 the method spent
        most of its iterations escaping it, and at 1e6 it failed), and the start moves
        to the barrier's centering direction λ̂ = -1 / f, with G^T λ̂ = ∇φ, at the scale
        that matches magnitudes,
           λ = (‖∇f0‖ / ‖∇φ‖) λ̂,   so that ‖G^T λ‖ = ‖∇f0‖.
        Its products λ_i (-f_i) are all equal: the start is perfectly centered.

        Two alternatives were tried and rejected:
        - Rescaling whenever it helps, not only at a wrong scale: matching magnitudes
          where ∇φ points along ∇f0 doubles the dual residual, and on a
          one-dimensional curved test problem (ratio 0.04, inside the range) that
          drove λ to zero before stationarity and stalled the solve.
        - The least-squares scale κ* = -⟨∇f0, ∇φ⟩ / ‖∇φ‖^2, which minimizes the
          starting dual residual: it is the magnitude scale times the cosine between
          -∇f0 and ∇φ, and that cosine can be tiny (1e-5 on RadOptLIB rt-qp-0062),
          where it collapses λ and the complementarity with it and the solve
          stalls, while reducing the residual only by a factor sqrt(1 - cos^2).

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Strictly inequality-feasible start, of length M.

        Returns
        -------
         lmbda : npt.NDArray[np.float64]
            Positive starting duals, of length m.

        """
        f = _PrimalDualKernel.concat_constraints(problem, x)
        ones = np.ones(f.size)
        norm_f0 = float(np.linalg.norm(none_throws(problem.obj).gradient(x)))
        norm_ones = float(
            np.linalg.norm(_PrimalDualKernel.g_transpose_times(problem, x, ones))
        )
        well_scaled = (
            norm_f0 / _DUAL_SCALE_RANGE <= norm_ones <= norm_f0 * _DUAL_SCALE_RANGE
        )
        if well_scaled or not (norm_f0 > 0.0 and math.isfinite(norm_f0)):
            return ones
        direction = -1.0 / f
        norm_phi = float(
            np.linalg.norm(_PrimalDualKernel.g_transpose_times(problem, x, direction))
        )
        if not (norm_phi > 0.0 and math.isfinite(norm_phi)):
            return ones
        lmbda: npt.NDArray[np.float64] = (norm_f0 / norm_phi) * direction
        return lmbda

    def _run_equality(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64],
    ) -> SolveResult:
        """Run the infeasible-start primal-dual iteration for a problem with equality.

        Starts from ν = 0, zero damping, and λ = 1, moved to the central path's
        direction at the objective's scale when λ = 1 is the wrong scale
        (``_central_path_duals``; λ = 1 in Phase I), and at each
        iteration evaluates
           r_dual = ∇f0(x) + G^T λ + A^T ν,   r_pri = A x - b,   η = -f^T λ,
        sets t = μ m / η, and takes one Mehrotra predictor-corrector step
        (``_solve_damped_step_equality``). After eliminating Δλ, each of its two solves
        is
            _           _  _    _     _                            _
           |  H_pd  A^T  ||  Δx  |   |  -r_dual - G^T (r_cent / f)  |
           |_  A     0  _||_ Δν _| = |_          -r_pri            _|
        with the same modified Hessian H_pd as the no-equality path; the lower
        right-hand side is nonzero while x is off the equality manifold.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective and equality constraints.
         x0 : npt.NDArray[np.float64]
            Strictly inequality-feasible starting point, of length M; it need not lie on
            the equality manifold.

        Returns
        -------
         result : SolveResult
            The solution, multipliers, dual value and per-iteration diagnostics.

        Raises
        ------
         RuntimeError
            When ``problem.eq`` is ``None`` (``solve`` never dispatches such a problem
            here).
         NewtonStepError
            Propagated from ``_solve_damped_step_equality`` and
            ``_clamp_dual_after_update``, unless the best iterate seen is already within
            the soft tolerance, which is then returned (as in ``_run``).

        Notes
        -----
        The stopping rules mirror ``_run``, with ‖r_pri‖ < tol added to the convergence
        test and to the stall metric. The dual value comes from the equality-aware
        ``_BarrierAdapter.evaluate_dual``, a lower bound on inf_x L(x, λ, ν). Unlike
        ``_run``, this path does not raise on a positive dual value.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        settings = (
            self.settings if self.settings is not None else OptimizationSettings()
        )
        m = problem.num_inequality_constraints()
        # solve() dispatches on problem.eq, but `assert` is stripped under
        # `python -O`; keep an explicit runtime guard so a direct call fails
        # loudly instead of an opaque AttributeError on problem.eq.A below.
        if problem.eq is None:
            raise RuntimeError(
                "_run_equality requires problem.eq is not None; this should be "
                "enforced by solve() before _run_equality is called."
            )
        A_eq = problem.eq.A
        b_eq = problem.eq.b
        p = A_eq.shape[0]
        nu = np.zeros(p)
        mu = settings.outer.barrier_multiplier
        outer_tol = settings.outer.tolerance
        max_iter = settings.inner.max_iterations
        alpha = settings.armijo.alpha
        beta = settings.armijo.beta
        min_step = settings.armijo.min_step
        tau_min = settings.pd_fraction_to_boundary_min
        lm = settings.primal_dual.damping

        x = x0.copy()
        # Start the duals at the objective's scale, except in Phase I: its objective is
        # the slack, unit-scaled by construction, where λ = 1 already matches it and a
        # gradient-balanced t (huge, against a unit ∇f0) would make every weight
        # λ / -f, and so an LP's H_pd, vanish.
        lmbda = (
            np.ones(m)
            if none_throws(problem.obj).feasibility_threshold is not None
            else self._central_path_duals(problem, x)
        )
        # Levenberg damping, carried across iterations; 0 = exact Newton step.
        damping = 0.0

        surrogate_gaps: list[float] = []
        step_sizes: list[float] = []
        dual_residual_norms: list[float] = []
        primal_residual_norms: list[float] = []
        status: Literal[0, 1, 2] = 0
        message = "Equality primal-dual method converged."
        stall = _StallWatch(
            patience=settings.inner.stall_patience,
            rel_improvement=settings.inner.stall_rel_improvement,
            soft_tolerance=settings.outer.tolerance_soft,
        )

        trace = settings.trace
        label = settings.observability.label
        solver_name = label if label is not None else self.__class__.__name__
        overall_start_time = time.time()
        trace.on_pd_start(solver_name)

        # Feasibility-objective early-stop threshold (see PrimalDualSolver._run);
        # ``None`` for an ordinary objective.
        feasibility_threshold = none_throws(problem.obj).feasibility_threshold

        adapter = _BarrierAdapter(problem, settings)
        check_ray = feasibility_threshold is None
        direction = None
        nit = 0
        for nit in range(max_iter):
            iter_start = time.time()
            if (
                feasibility_threshold is not None
                and none_throws(problem.obj).feasibility_residual(x)
                < feasibility_threshold
            ):
                status = 2
                message = (
                    "Feasibility objective crossed its threshold; "
                    "the problem is feasible"
                )
                # Mirror the barrier loop's early-stop trace event (see
                # PrimalDualSolver._run) for cross-path observability parity.
                trace.on_early_stop(nit + 1)
                break
            f = _PrimalDualKernel.concat_constraints(problem, x)
            grad_residual_inequality = _PrimalDualKernel.gradient_residual(
                problem, x, lmbda
            )
            grad_residual_equality = A_eq.T @ nu
            r_dual = (
                none_throws(problem.obj).gradient(x)
                + grad_residual_inequality
                + grad_residual_equality
            )
            r_pri = A_eq @ x - b_eq

            eta = float(-np.dot(f, lmbda))
            dual_norm = float(np.linalg.norm(r_dual))
            pri_norm = float(np.linalg.norm(r_pri))
            residual_norm = float(
                np.sqrt(dual_norm * dual_norm + eta * eta + pri_norm * pri_norm)
            )
            surrogate_gaps.append(eta)
            dual_residual_norms.append(dual_norm)
            primal_residual_norms.append(pri_norm)
            gap_scale, dual_scale, pri_scale = _pd_scales(
                settings.outer, problem, x, A_eq, b_eq
            )

            if eta <= 0.0:
                status = (
                    0
                    if dual_norm < outer_tol * dual_scale
                    and pri_norm < outer_tol * pri_scale
                    else 1
                )
                message = (
                    f"Iteration stopped at non-positive surrogate gap ({eta:.3e})."
                )
                break

            if (
                eta < outer_tol * gap_scale
                and dual_norm < outer_tol * dual_scale
                and pri_norm < outer_tol * pri_scale
            ):
                status = 0
                break

            # Same stall guard as the inequality-only path (see ``_StallWatch``),
            # over the three quantities this loop must drive down together.
            if stall.observe(
                max(eta / gap_scale, dual_norm / dual_scale, pri_norm / pri_scale),
                nit,
                x,
                lmbda,
                nu,
            ):
                status = 1
                message = (
                    f"stalled: no relative improvement in "
                    f"max(gap, ||r_dual||, ||r_pri||) over "
                    f"{settings.inner.stall_patience} iterations; returning "
                    f"best iterate from iteration {stall.best_nit} "
                    f"(max(gap, ||r_dual||, ||r_pri||)={stall.best_metric:.3e})"
                )
                x = none_throws(stall.best_x)
                lmbda = none_throws(stall.best_lmbda)
                nu = none_throws(stall.best_nu)
                trace.on_early_stop(nit + 1)
                break

            t = mu * m / eta
            r_cent = -lmbda * f - 1.0 / t

            try:
                delta_x, delta_lmbda, delta_nu, step, damping = (
                    self._solve_damped_step_equality(
                        problem,
                        x,
                        lmbda,
                        nu,
                        f,
                        r_dual,
                        r_cent,
                        r_pri,
                        A_eq,
                        b_eq,
                        eta=eta,
                        t=t,
                        alpha=alpha,
                        beta=beta,
                        min_step=min_step,
                        tau_min=tau_min,
                        damping=damping,
                        lm=lm,
                    )
                )
            except NewtonStepError as e:
                # Same fallback as the inequality-only path (see ``_run``).
                if not stall.within_soft_tolerance:
                    raise
                status = 1
                message = (
                    f"the Newton system broke down ({e}); returning best iterate "
                    f"from iteration {stall.best_nit} "
                    f"(max(gap, ||r_dual||, ||r_pri||)={stall.best_metric:.3e})"
                )
                x = none_throws(stall.best_x)
                lmbda = none_throws(stall.best_lmbda)
                nu = none_throws(stall.best_nu)
                trace.on_early_stop(nit + 1)
                break
            if check_ray:
                direction = _verify_recession_direction(
                    adapter,
                    x,
                    delta_x,
                    max(float(np.max(f, initial=0.0)), pri_norm),
                    outer_tol,
                )
                if direction is not None:
                    status = 2
                    message = (
                        "Verified a feasible recession direction with negative slope."
                    )
                    break
            step_sizes.append(step)
            x = x + step * delta_x
            lmbda = _clamp_dual_after_update(lmbda + step * delta_lmbda)
            nu = nu + step * delta_nu
            trace.on_pd_iteration(
                nit=nit,
                eta=eta,
                dual_norm=dual_norm,
                pri_norm=pri_norm,
                residual_norm=residual_norm,
                t=t,
                step=step,
                elapsed_ms=1000.0 * (time.time() - iter_start),
            )
        else:
            status = 1
            message = (
                f"Equality primal-dual reached iteration cap ({max_iter}) without "
                "convergence."
            )

        # Feasibility early-stop short-circuit (see PrimalDualSolver._run): the
        # loop broke before ``lmbda``/``nu`` were refined past their seeds, so the
        # dual-side fields would only echo arbitrary multipliers. Report the primal
        # certificate and zero the dual fields rather than evaluate them.
        if status == 2 and direction is None:
            objective_value = none_throws(problem.obj).evaluate(x)
            trace.on_pd_done(
                solver=solver_name,
                nits=nit + 1,
                status=status,
                elapsed_ms=1000.0 * (time.time() - overall_start_time),
            )
            # The objective crossed its feasibility threshold: report the point as
            # FEASIBLE, not by the integer code 2, which maps to MAX_ITERATIONS
            # (issue #52).
            return replace(
                _pd_solveresult(
                    solution=x,
                    objective_value=float(objective_value),
                    dual_value=0.0,
                    equality_multipliers=nu,
                    inequality_multipliers=lmbda,
                    suboptimality=0.0,
                    nits=nit + 1,
                    status=status,
                    message=message,
                    surrogate_gaps=surrogate_gaps,
                    step_sizes=step_sizes,
                    primal_residual_norms=primal_residual_norms,
                    dual_residual_norms=dual_residual_norms,
                ),
                status=SolveStatus.FEASIBLE,
            )

        # Surrogate-gap proxy at the solution (r_dual and r_primal not re-evaluated).
        f_final = _PrimalDualKernel.concat_constraints(problem, x)
        eta_final = float(-np.dot(f_final, lmbda))

        objective_value = none_throws(problem.obj).evaluate(x)
        # Lagrangian dual via the equality barrier's mini-Newton evaluator -- a
        # guaranteed lower bound on inf_x L(x, lambda, nu) (B&V Eq 9.42), with
        # the ``A^T nu`` / ``-nu^T b`` equality terms threaded through. Reuses
        # the equality-aware ``_BarrierAdapter.evaluate_dual`` (it branches on
        # ``problem.eq``) instead of the analytic per-component decomposition,
        # whose plug-in fallback (``L`` at the current ``x``) is only an UPPER
        # bound on inf_x L.
        dual_value = (
            adapter.evaluate_dual(lmbda, nu, x) if direction is None else float("-inf")
        )

        if direction is None:
            gap_scale, _, pri_scale = _pd_scales(settings.outer, problem, x, A_eq, b_eq)
            status, message = _certify_pd_status(
                status,
                message,
                objective_value,
                dual_value,
                max(
                    float(np.max(f_final, initial=0.0)),
                    float(np.linalg.norm(A_eq @ x - b_eq)),
                ),
                settings.outer,
                gap_scale=gap_scale,
                feasibility_scale=pri_scale,
            )

        trace.on_pd_done(
            solver=solver_name,
            nits=nit + 1,
            status=status,
            elapsed_ms=1000.0 * (time.time() - overall_start_time),
        )
        return _pd_solveresult(
            solution=x,
            objective_value=float(objective_value),
            dual_value=float(dual_value),
            equality_multipliers=nu,
            inequality_multipliers=lmbda,
            suboptimality=eta_final if direction is None else float("inf"),
            direction=direction,
            nits=nit + 1,
            status=status,
            message=message,
            surrogate_gaps=surrogate_gaps,
            step_sizes=step_sizes,
            primal_residual_norms=primal_residual_norms,
            dual_residual_norms=dual_residual_norms,
        )

    @staticmethod
    def _build_pd_solve_equality(
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        damping: float,
    ) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
        tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ],
    ]:
        """Factor the equality-path KKT system into a refined solve closure.

        The closure solves the full (unreduced) primal-dual system
           H Δx + G^T Δλ + A^T Δν = e_dual,
           -λ ∘ (G Δx) - f ∘ Δλ = e_cent,
           A Δx = e_pri,
        with H = ∇^2 f0 + ∑ λ_i ∇^2 f_i (+ damping I), by eliminating
           Δλ = -(e_cent + λ ∘ (G Δx)) / f
        into the saddle system
            _           _  _    _     _                        _
           |  H_pd  A^T  ||  Δx  |   |  e_dual + G^T (e_cent / f)  |
           |_  A     0  _||_ Δν _| = |_          e_pri            _|,
        H_pd = H + G^T diag(λ / -f) G, solved through the equality layer's
        ``saddle_system`` (refined, since it grows ill-conditioned near the optimum,
        issue #22).

        Each solve then takes one step of iterative refinement on the full system, kept
        only when it reduces the full residual (it can fail to on a badly scaled
        system, where the unrefined step is the better one). The
        elimination divides by f, which is tiny on near-active rows late in a solve, so
        a reduced-system error that is negligible in Δx is amplified in Δλ and from
        there into the dual residual G^T Δλ. Refining the reduced system alone does not
        see that error; refining the full system does. Without it, a Mehrotra step,
        which has no merit line search to reject a bad direction, can push the dual
        residual up by orders of magnitude in one step near the optimum. The predictor
        and corrector share the factorization.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective and equality constraints.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         damping : float
            Levenberg damping added to H; 0 for none.

        Returns
        -------
         solve : Callable
            Maps (e_dual, e_cent, e_pri) to (Δx, Δλ, Δν).

        Raises
        ------
         NewtonStepError
            Propagated from ``_PrimalDualKernel.validate_pd_iterate``.

        """
        _PrimalDualKernel.validate_pd_iterate(lmbda, f)
        eq = none_throws(problem.eq)
        # H_pd = nabla^2 f0 + sum_i lambda_i C_i at the dual weights (scale = 1);
        # see the no-equality _build_pd_solve.
        system = problem.build_hessian_system(
            x, none_throws(problem.obj).at(x), 1.0, lmbda
        )
        if damping > 0.0:
            system = system.plus(DiagonalSystem(np.full(x.shape[0], damping)))
        saddle = eq.saddle_system(system)
        A = eq.A

        def reduced(
            e_dual: npt.NDArray[np.float64],
            e_cent: npt.NDArray[np.float64],
            e_pri: npt.NDArray[np.float64],
        ) -> tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ]:
            g_x = e_dual + _PrimalDualKernel.g_transpose_times(problem, x, e_cent / f)
            delta_x, delta_nu = saddle.solve_refined(g_x, e_pri)
            g_dx = _PrimalDualKernel.g_times(problem, x, delta_x)
            delta_lmbda = -(e_cent + lmbda * g_dx) / f
            return delta_x, delta_lmbda, delta_nu

        def residual(
            e_dual: npt.NDArray[np.float64],
            e_cent: npt.NDArray[np.float64],
            e_pri: npt.NDArray[np.float64],
            delta_x: npt.NDArray[np.float64],
            delta_lmbda: npt.NDArray[np.float64],
            delta_nu: npt.NDArray[np.float64],
        ) -> tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ]:
            g_dx = _PrimalDualKernel.g_times(problem, x, delta_x)
            # ``system`` is the reduced H_pd, so H Δx = H_pd Δx + G^T (λ ∘ G Δx / f).
            h_dx = system.multiply(delta_x) + _PrimalDualKernel.g_transpose_times(
                problem, x, lmbda * g_dx / f
            )
            res_dual = e_dual - (
                h_dx
                + _PrimalDualKernel.g_transpose_times(problem, x, delta_lmbda)
                + A.T @ delta_nu
            )
            res_cent = e_cent - (-lmbda * g_dx - f * delta_lmbda)
            res_pri = e_pri - A @ delta_x
            return res_dual, res_cent, res_pri

        def size(parts: tuple[npt.NDArray[np.float64], ...]) -> float:
            # A plain float sum: overflow should read as inf, not raise.
            total = sum(float(np.dot(v, v)) for v in parts)
            return math.sqrt(total) if math.isfinite(total) else math.inf

        def solve(
            e_dual: npt.NDArray[np.float64],
            e_cent: npt.NDArray[np.float64],
            e_pri: npt.NDArray[np.float64],
        ) -> tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ]:
            step = reduced(e_dual, e_cent, e_pri)
            with np.errstate(all="ignore"):
                res = residual(e_dual, e_cent, e_pri, *step)
                res_size = size(res)
                if not math.isfinite(res_size) or res_size == 0.0:
                    return step
                try:
                    correction = reduced(*res)
                except (NewtonStepError, ValueError, np.linalg.LinAlgError):
                    return step
                refined = (
                    step[0] + correction[0],
                    step[1] + correction[1],
                    step[2] + correction[2],
                )
                # Keep the refinement only when it actually improves the full
                # residual; on a badly scaled system the correction can be worse
                # (or non-finite), and the unrefined step is then the better one.
                if size(residual(e_dual, e_cent, e_pri, *refined)) < res_size:
                    return refined
            return step

        return solve

    def _compute_pd_step_equality(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        r_pri: npt.NDArray[np.float64],
        damping: float = 0.0,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Solve the equality-augmented modified KKT system for (Δx, Δλ, Δν).

        Solves
           (H_pd + damping I) Δx + A^T Δν = -r_dual - G^T (r_cent / f),
           A Δx = -r_pri
        and recovers
           Δλ = (r_cent - λ ∘ (G Δx)) / f,
        through ``_build_pd_solve_equality`` (one factorization, refined on the full
        system).

        Parameters
        ----------
         problem : Problem
            The problem, with an objective and equality constraints.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, including A^T ν, of length M.
         r_cent : npt.NDArray[np.float64]
            Centering residual, of length m.
         r_pri : npt.NDArray[np.float64]
            Primal residual A x - b, of length p.
         damping : float
            Levenberg term added to the modified Hessian as ``damping`` I; 0 gives the
            exact Newton step.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            Primal step, of length M.
         delta_lmbda : npt.NDArray[np.float64]
            Inequality-dual step, of length m.
         delta_nu : npt.NDArray[np.float64]
            Equality-multiplier step, of length p.

        Raises
        ------
         NewtonStepError
            Propagated from ``_build_pd_solve_equality``.

        """
        solve = self._build_pd_solve_equality(problem, x, lmbda, f, damping)
        return solve(-r_dual, -r_cent, -r_pri)

    def _backtracking_step_equality(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        delta_lmbda: npt.NDArray[np.float64],
        delta_nu: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        r_pri: npt.NDArray[np.float64],
        A_eq: npt.NDArray[np.float64],
        b_eq: npt.NDArray[np.float64],
        eta: float,
        t: float,
        alpha: float,
        beta: float,
        min_step: float,
        tau_min: float,
    ) -> float:
        """Choose the equality-path step by a residual line search that includes r_pri.

        Three stages:
        1. Cap the step at the Mehrotra fraction-to-boundary bound on dual positivity,
              s = max(τ_min, 1 - η / m) s_max,
           where s_max is the largest step keeping λ + s Δλ > 0 (``mehrotra_step_cap``).
        2. Shrink by ``beta`` until x + s Δx is strictly feasible. If the step reaches
           ``min_step`` first, return it when feasible (skipping stage 3) and raise
           otherwise.
        3. Shrink by ``beta`` until the KKT residual satisfies
              ‖r(x + s Δx, λ + s Δλ, ν + s Δν)‖ ≤ (1 - α s) ‖r(x, λ, ν)‖,
           with r = (r_dual, r_cent, r_pri) at the fixed t. ``ResidualNormMerit``
           supplies the test; this loop returns the ``min_step``-floored step silently
           rather than raising as ``LineSearch`` would.
        Stage 3 is the search of (Boyd and Vandenberghe, 2004, section 11.7.3).

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         nu : npt.NDArray[np.float64]
            Equality multipliers ν, of length p.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m; unused by the computation.
         delta_x : npt.NDArray[np.float64]
            Primal step, of length M.
         delta_lmbda : npt.NDArray[np.float64]
            Inequality-dual step, of length m.
         delta_nu : npt.NDArray[np.float64]
            Equality-multiplier step, of length p.
         r_dual : npt.NDArray[np.float64]
            Dual residual at the iterate, of length M.
         r_cent : npt.NDArray[np.float64]
            Centering residual at the iterate, of length m.
         r_pri : npt.NDArray[np.float64]
            Primal residual at the iterate, of length p.
         A_eq : npt.NDArray[np.float64]
            Equality matrix A, of shape (p, M).
         b_eq : npt.NDArray[np.float64]
            Equality right-hand side b, of length p.
         eta : float
            Surrogate duality gap.
         t : float
            Barrier parameter defining the trial centering residual.
         alpha : float
            Sufficient-decrease coefficient α.
         beta : float
            Shrink factor.
         min_step : float
            Step below which the search stops shrinking.
         tau_min : float
            Floor τ_min of the Mehrotra boundary factor.

        Returns
        -------
         step : float
            The accepted step; possibly at or below ``min_step`` when stage 3 found no
            sufficient decrease.

        Raises
        ------
         NewtonStepError
            When stage 2 reaches ``min_step`` without a strictly feasible trial point.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        # Stage 1: Mehrotra fraction-to-boundary cap on dual positivity.
        s = mehrotra_step_cap(lmbda, delta_lmbda, eta, tau_min)

        # Stage 2: primal-feasibility shrink on inequality.
        while s > min_step:
            f_trial = _PrimalDualKernel.concat_constraints(problem, x + s * delta_x)
            if np.all(f_trial < 0):
                break
            s *= beta
        if s <= min_step:
            # Mirror of the no-equality feasible-step check: verify the trial
            # iterate is actually feasible before returning, so the silent-NaN
            # propagation through ``pd_weights = lmbda / (-f)`` is caught at
            # the cause site rather than far downstream.
            f_trial = _PrimalDualKernel.concat_constraints(problem, x + s * delta_x)
            if not np.all(f_trial < 0):
                raise NewtonStepError(
                    f"BTLS could not maintain primal feasibility at "
                    f"s={s:.3e} (min_step={min_step:.3e}); the trial iterate "
                    "has non-negative slack on at least one inequality. The "
                    "Newton direction may be poorly conditioned, or the "
                    "dual-positivity cap drove the step below min_step."
                )
            return s

        # Stage 3: residual-Armijo shrink (incorporating r_pri).
        # ``ResidualNormMerit`` provides the acceptance threshold and trial-
        # residual evaluation; PD owns the contraction loop because its
        # silent-return-at-``min_step`` semantics differ from
        # ``LineSearch.find_step``'s ``SevereCurvatureError``.
        r_norm = float(
            np.sqrt(
                np.dot(r_dual, r_dual) + np.dot(r_cent, r_cent) + np.dot(r_pri, r_pri)
            )
        )

        def compute_trial_residual_norm(s_trial: float) -> float:
            """Compute the KKT residual norm at a trial step.

            Parameters
            ----------
             s_trial : float
                Trial step length along (Δx, Δλ, Δν).

            Returns
            -------
             r_norm : float
                The residual norm ‖(r_dual, r_cent, r_pri)‖ at the trial point, at the
                fixed barrier parameter; inf when the trial point is not strictly
                feasible.

            """
            x_trial = x + s_trial * delta_x
            lmbda_trial = lmbda + s_trial * delta_lmbda
            nu_trial = nu + s_trial * delta_nu
            f_trial = _PrimalDualKernel.concat_constraints(problem, x_trial)
            # Stage 2 found a strictly feasible step, and by convexity so is every
            # shorter one -- but not in floating point, at an iterate within rounding
            # of a bound (x_j ~ 1e-16 against x_j >= 0). Reject such a trial here.
            if not np.all(f_trial < 0):
                return math.inf
            r_dual_trial = (
                none_throws(problem.obj).gradient(x_trial)
                + _PrimalDualKernel.gradient_residual(problem, x_trial, lmbda_trial)
                + A_eq.T @ nu_trial
            )
            r_cent_trial = -lmbda_trial * f_trial - 1.0 / t
            r_pri_trial = A_eq @ x_trial - b_eq
            return float(
                np.sqrt(
                    np.dot(r_dual_trial, r_dual_trial)
                    + np.dot(r_cent_trial, r_cent_trial)
                    + np.dot(r_pri_trial, r_pri_trial)
                )
            )

        merit = ResidualNormMerit(
            compute_trial_residual_norm=compute_trial_residual_norm,
            r_norm0=r_norm,
            alpha=alpha,
        )
        while s > min_step:
            if merit.evaluate(x, s) <= merit.accept_threshold(s):
                return s
            s *= beta
        if not np.all(
            _PrimalDualKernel.concat_constraints(problem, x + s * delta_x) < 0
        ):
            raise NewtonStepError(
                f"BTLS could not maintain primal feasibility at s={s:.3e}; the "
                "residual search shrank the step onto a bound within rounding."
            )
        return s

    def _residual_norm_equality(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        a_eq: npt.NDArray[np.float64],
        b_eq: npt.NDArray[np.float64],
        t: float,
    ) -> float:
        """Compute the KKT residual norm ‖(r_dual, r_cent, r_pri)‖ at a fixed t.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         nu : npt.NDArray[np.float64]
            Equality multipliers ν, of length p.
         a_eq : npt.NDArray[np.float64]
            Equality matrix A, of shape (p, M).
         b_eq : npt.NDArray[np.float64]
            Equality right-hand side b, of length p.
         t : float
            Barrier parameter defining r_cent = -λ ∘ f - 1 / t.

        Returns
        -------
         r_norm : float
            The Euclidean norm of the stacked residual.

        """
        f = _PrimalDualKernel.concat_constraints(problem, x)
        r_dual = (
            none_throws(problem.obj).gradient(x)
            + _PrimalDualKernel.gradient_residual(problem, x, lmbda)
            + a_eq.T @ nu
        )
        r_cent = -lmbda * f - 1.0 / t
        r_pri = a_eq @ x - b_eq
        return float(
            np.sqrt(
                np.dot(r_dual, r_dual) + np.dot(r_cent, r_cent) + np.dot(r_pri, r_pri)
            )
        )

    def _solve_damped_step_equality(
        self,
        problem: Problem,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        r_dual: npt.NDArray[np.float64],
        r_cent: npt.NDArray[np.float64],
        r_pri: npt.NDArray[np.float64],
        a_eq: npt.NDArray[np.float64],
        b_eq: npt.NDArray[np.float64],
        *,
        eta: float,
        t: float,
        alpha: float,
        beta: float,
        min_step: float,
        tau_min: float,
        damping: float,
        lm: LevenbergSettings,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
        float,
    ]:
        """Take one equality-path Mehrotra predictor-corrector step, with damping.

        The infeasible-start counterpart of ``_mpc_solve_step``. With slacks s = -f and
        μ = η / m, on one factorization (``_build_pd_solve_equality``):
        1. The affine predictor solves with zero complementarity target,
           r_cent = -λ ∘ f, and the full primal residual -r_pri.
        2. The fraction-to-boundary steps α_pri on s and α_dual on λ give
              μ_aff = (s + α_pri Δs_aff)^T (λ + α_dual Δλ_aff) / m,
           and σ = (μ_aff / μ)^3, clipped to [0, 1].
        3. The corrector solves with
              r_cent = -λ ∘ f - σ μ + Δλ_aff ∘ Δs_aff.
        The step is the Mehrotra fraction-to-boundary step of ``_feasible_step``. The
        primal residual A x - b then shrinks by exactly 1 - step, as in any
        infeasible-start method. One safeguard remains: when the step would raise the
        surrogate gap η, or no feasible step exists, the iteration falls back to the
        centering step at t with the residual line search of
        ``_backtracking_step_equality`` (Boyd and Vandenberghe, 2004, section 11.7.3),
        the method this one replaced. That method alone took about three times as many
        iterations (103 against 26 on RadOptLIB rt-qp-0062, and 20 against 7 on the
        test suite's equality problems).

        When neither step is feasible, or the factorization fails (a numerically
        singular H), the damping grows (from zero to ‖rhs‖ / x_scale,
        rhs = -r_dual - G^T (r_cent / f); otherwise by ``lm.grow``) and the step is
        re-solved, up to ``lm.max_attempts`` times. After a damped step, the gain ratio
           ρ = (‖r_t‖ - ‖r_t(trial)‖) / (‖r_t‖ - ‖r_t(model)‖),
        with r_t = (r_dual, r_cent, r_pri) at the fixed t, sets the next damping.

        Parameters
        ----------
         problem : Problem
            The problem, with an objective and equality constraints.
         x : npt.NDArray[np.float64]
            Primal iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Inequality duals λ, of length m.
         nu : npt.NDArray[np.float64]
            Equality multipliers ν, of length p.
         f : npt.NDArray[np.float64]
            Constraint values at ``x``, of length m.
         r_dual : npt.NDArray[np.float64]
            Dual residual, of length M.
         r_cent : npt.NDArray[np.float64]
            Standard centering residual -λ ∘ f - 1 / t, of length m; used for the
            ratio test and the damping initialization.
         r_pri : npt.NDArray[np.float64]
            Primal residual A x - b, of length p.
         a_eq : npt.NDArray[np.float64]
            Equality matrix A, of shape (p, M).
         b_eq : npt.NDArray[np.float64]
            Equality right-hand side b, of length p.
         eta : float
            Surrogate duality gap.
         t : float
            Barrier parameter of the ratio test.
         beta : float
            Shrink factor of the feasibility search.
         min_step : float
            Depth bound of the feasibility search.
         tau_min : float
            Floor τ_min of the Mehrotra boundary factor.
         damping : float
            Damping carried from the previous iteration.
         lm : LevenbergSettings
            The damping schedule.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            Primal step, of length M.
         delta_lmbda : npt.NDArray[np.float64]
            Inequality-dual step, of length m.
         delta_nu : npt.NDArray[np.float64]
            Equality-multiplier step, of length p.
         step : float
            Accepted step length.
         next_damping : float
            Damping for the next iteration; 0 when the step was undamped.

        Raises
        ------
         NewtonStepError
            When no feasible step is found after ``lm.max_attempts`` attempts; also
            propagated from ``_build_pd_solve_equality``.

        References
        ----------
        Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method,
        SIAM Journal on Optimization 2, 1992.

        """
        gamma = mehrotra_boundary_factor(eta, lmbda.size, tau_min)
        x_scale = max(1.0, float(np.linalg.norm(x)))
        r_norm0 = math.sqrt(
            float(
                np.dot(r_dual, r_dual) + np.dot(r_cent, r_cent) + np.dot(r_pri, r_pri)
            )
        )
        m = max(lmbda.size, 1)
        s = -f
        mu = eta / m

        def grown(current: float) -> float:
            if current > 0.0:
                return current * lm.grow
            rhs = -r_dual - _PrimalDualKernel.g_transpose_times(problem, x, r_cent / f)
            return float(np.linalg.norm(rhs)) / x_scale

        _PrimalDualKernel.validate_pd_iterate(lmbda, f)
        for _attempt in range(lm.max_attempts):
            try:
                solve = self._build_pd_solve_equality(problem, x, lmbda, f, damping)
                dx_aff, dlam_aff, _ = solve(-r_dual, lmbda * f, -r_pri)
            except NewtonStepError:
                # A numerically singular H (an LP far from the central path, where
                # every weight λ / -f is small) is what the damping is for.
                damping = grown(damping)
                continue
            ds_aff = -_PrimalDualKernel.g_times(problem, x, dx_aff)
            a_pri = fraction_to_boundary(s, ds_aff)
            a_dual = fraction_to_boundary(lmbda, dlam_aff)
            mu_aff = float(np.dot(s + a_pri * ds_aff, lmbda + a_dual * dlam_aff)) / m
            sigma = min(max((mu_aff / mu) ** 3 if mu > 0.0 else 0.0, 0.0), 1.0)
            r_cent_cor = -lmbda * f - sigma * mu + dlam_aff * ds_aff
            delta_x, delta_lmbda, delta_nu = solve(-r_dual, -r_cent_cor, -r_pri)
            step = self._feasible_step(
                problem, x, f, delta_x, lmbda, delta_lmbda, gamma, beta, min_step
            )
            eta_new = (
                math.inf
                if step is None
                else float(
                    -np.dot(
                        _PrimalDualKernel.concat_constraints(
                            problem, x + step * delta_x
                        ),
                        lmbda + step * delta_lmbda,
                    )
                )
            )
            if step is None or not eta_new <= eta:
                # Safeguard: a Mehrotra step exists to cut complementarity (its
                # predicted surrogate gap is about (1 - step (1 - σ)) η), so a trial
                # gap above η, or no feasible step at all, marks a step gone wrong.
                # Take the centering step at t instead, with the residual line search
                # of (Boyd and Vandenberghe, 2004, section 11.7.3). On the c x 1e6
                # scaled LPs of the test suite this cuts the iterations from 25 / 69 /
                # 21 to 17 / 20 / 16. It is no substitute for a well-scaled start:
                # from λ = 1 on those LPs the method still fails, which is why the
                # duals start at the objective's scale (``_central_path_duals``).
                # The fixed-t KKT residual is deliberately not the test: the
                # corrector targets σ μ rather than 1 / t, so a good Mehrotra step
                # routinely raises it (on RadOptLIB rt-qp-0062 that test rejected
                # most steps and doubled the iteration count).
                delta_x, delta_lmbda, delta_nu = solve(-r_dual, -r_cent, -r_pri)
                r_cent_cor = r_cent
                try:
                    step = self._backtracking_step_equality(
                        problem,
                        x,
                        lmbda,
                        nu,
                        f,
                        delta_x,
                        delta_lmbda,
                        delta_nu,
                        r_dual,
                        r_cent,
                        r_pri,
                        a_eq,
                        b_eq,
                        eta=eta,
                        t=t,
                        alpha=alpha,
                        beta=beta,
                        min_step=min_step,
                        tau_min=tau_min,
                    )
                except NewtonStepError:
                    step = None
                if step is None:
                    damping = grown(damping)
                    continue
            assert step is not None
            if damping <= 0.0:
                return delta_x, delta_lmbda, delta_nu, step, 0.0
            r_norm_new = self._residual_norm_equality(
                problem,
                x + step * delta_x,
                lmbda + step * delta_lmbda,
                nu + step * delta_nu,
                a_eq,
                b_eq,
                t,
            )
            # Gain ratio against the *damped* linear model (issue #23): the damped
            # step leaves (1 - s) r_dual - s δ Δx in the dual block, the step's own
            # centering target in the centering block (r_cent - s r_cent_cor; for the
            # centering fallback, (1 - s) r_cent), and (1 - s) r_pri in the primal
            # block.
            r_dual_model = (1.0 - step) * r_dual - (step * damping) * delta_x
            r_cent_model = r_cent - step * r_cent_cor
            r_norm_model = math.sqrt(
                float(
                    np.dot(r_dual_model, r_dual_model)
                    + np.dot(r_cent_model, r_cent_model)
                )
                + (1.0 - step) ** 2 * float(np.dot(r_pri, r_pri))
            )
            predicted = r_norm0 - r_norm_model
            rho = (r_norm0 - r_norm_new) / predicted if predicted > 0.0 else 0.0
            next_damping = self._relax_damping(damping, rho, x_scale, lm)
            return delta_x, delta_lmbda, delta_nu, step, next_damping
        raise NewtonStepError(
            f"MPC (equality) failed to find a feasible step after {lm.max_attempts} "
            f"attempts (final damping={damping:.3e}); the modified Hessian could not be "
            "regularized into a usable step."
        )
