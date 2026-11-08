"""Infeasible-start Newton barrier solver for a two-block ``StationarityProblem``.

``StationaritySolver`` finds a joint stationary point (a saddle point, when the
objective is concave in x and convex in y) of a ``StationarityProblem``. The outer loop
raises the barrier parameter t geometrically. At each t, an inner centering loop takes
infeasible-start Newton steps on the barrier-smoothed residual, solving the coupled
saddle system the problem assembles, and accepts each step by a residual-norm
backtracking line search. A start that is not strictly interior is first made so by a
per-block Phase I.

At each outer step the solver also brackets the saddle value V in two ways: a
closed-form bracket that is exact on the central path, and a certified bracket from two
per-block dual solves (``_BlockDual``) that holds off it too. ``_BarrierTrace``
accumulates both, with the per-step convergence trajectories, for the result's
diagnostics.

"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import numpy.typing as npt

from ..exceptions import (
    ConstraintBoundaryError,
    InvalidDescentDirectionError,
    NewtonStepError,
    ProblemInfeasibleError,
    SevereCurvatureError,
)
from ..linalg.systems import (
    CoupledSaddleSystem,
    EqualityCoupledSaddleSystem,
    MatrixSystem,
)
from ..problems.problem import Problem
from ..problems.result import (
    Duals,
    ObjectiveOutcome,
    SaddleBarrierDiagnostics,
    SolveStatus,
)
from ..saddle.stationarity import (
    StationarityConvergenceError,
    StationarityProblem,
    TwoBlockSolveResult,
    two_block_result_message,
)
from ..solvers.barrier_solver import (
    _DUAL_REGION_DECREMENT_SQ_MAX,
    BarrierSolver,
)
from ..solvers.line_search import (
    ArmijoBarrierMerit,
    LineSearch,
    ResidualNormMerit,
)
from ..solvers.settings import (
    OptimizationSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..problems.constraint_layers import (
        ConstraintLayer,
    )
    from ..problems.linear_constraints import (
        EqualityConstraintLayer,
    )

logger: logging.Logger = logging.getLogger("cvxium.stationarity_solver")

# Initial inverse barrier weight `t_0`; the effective start is
# ``max(t_0, settings.outer.min_initial_barrier_parameter)``.
_INITIAL_BARRIER_PARAMETER: float = 1.0


@dataclass
class _BarrierTrace:
    """Per-outer-step convergence trajectories accumulated across the barrier loop.

    A small mutable accumulator that the outer loop appends to once per outer
    (barrier-parameter) step and freezes into a ``SaddleBarrierDiagnostics`` at
    finalization. Entry i of every field describes outer step i. The inner-step fields
    are tuples per outer step, the shape
    ``SaddleBarrierDiagnostics.plot_newton_convergence`` indexes and iterates.

    Fields
    ------
     duality_gaps : list[float]
        The surrogate duality gap n_barriers / t at each centered iterate.
     inner_nits : list[int]
        The number of inner Newton steps taken at each outer step.
     inner_suboptimalities : list[tuple[float, ...]]
        The start-of-step residual norm ‖r‖ of each accepted Newton step, so tuple i has
        length ``inner_nits[i]``.
     inner_btls_steps : list[tuple[float, ...]]
        The line-search step accepted for each inner Newton step, parallel to
        ``inner_suboptimalities``.
     upper_bounds : list[float]
        Upper end of the closed-form central-path bracket on the saddle value (see
        ``StationaritySolver._saddle_value_bounds``).
     lower_bounds : list[float]
        Lower end of the closed-form central-path bracket.
     certified_upper_bounds : list[float]
        Upper end of the certified per-block dual bracket, which holds off the central
        path too (see ``StationaritySolver._certified_saddle_value_bounds``); inf when
        that outer step's dual solve deferred.
     certified_lower_bounds : list[float]
        Lower end of the certified bracket; -inf when that outer step's dual solve
        deferred.

    """

    duality_gaps: list[float] = field(default_factory=list)
    inner_nits: list[int] = field(default_factory=list)
    inner_suboptimalities: list[tuple[float, ...]] = field(default_factory=list)
    inner_btls_steps: list[tuple[float, ...]] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    certified_upper_bounds: list[float] = field(default_factory=list)
    certified_lower_bounds: list[float] = field(default_factory=list)

    def record(
        self,
        gap: float,
        inner_residuals: Sequence[float],
        inner_steps: Sequence[float],
        upper_bound: float,
        lower_bound: float,
        certified_upper_bound: float,
        certified_lower_bound: float,
    ) -> None:
        """Append one outer step's gap, inner trajectories, and both brackets.

        The step trajectory is recorded because it shows failure modes that leave the
        residual trajectory looking healthy: a step pinned at ``min_step`` (the line
        search failing outright), or a step capped at β where the full step should have
        been accepted.

        Parameters
        ----------
         gap : float
            The surrogate duality gap at the centered iterate.
         inner_residuals : Sequence[float]
            The start-of-step residual norm of each accepted inner Newton step; its
            length becomes the step's ``inner_nits`` entry.
         inner_steps : Sequence[float]
            The line-search step accepted for each inner Newton step, parallel to
            ``inner_residuals``.
         upper_bound : float
            Upper end of the closed-form central-path bracket.
         lower_bound : float
            Lower end of the closed-form central-path bracket.
         certified_upper_bound : float
            Upper end of the certified bracket; inf on a deferral.
         certified_lower_bound : float
            Lower end of the certified bracket; -inf on a deferral.

        """
        self.duality_gaps.append(gap)
        self.inner_nits.append(len(inner_residuals))
        self.inner_suboptimalities.append(tuple(inner_residuals))
        self.inner_btls_steps.append(tuple(inner_steps))
        self.upper_bounds.append(upper_bound)
        self.lower_bounds.append(lower_bound)
        self.certified_upper_bounds.append(certified_upper_bound)
        self.certified_lower_bounds.append(certified_lower_bound)

    def to_diagnostics(self) -> SaddleBarrierDiagnostics:
        """Freeze the accumulated trajectories into a ``SaddleBarrierDiagnostics``.

        ``inner_quadratic_from`` is left unset: the barrier method flags its quadratic
        phase by comparing the Newton decrement with a self-concordance threshold, but
        this solver's inner metric is a residual norm with no such guarantee.

        Returns
        -------
         diagnostics : SaddleBarrierDiagnostics
            The trajectories, as tuples.

        """
        return SaddleBarrierDiagnostics(
            duality_gaps=tuple(self.duality_gaps),
            inner_nits=tuple(self.inner_nits),
            inner_suboptimalities=tuple(self.inner_suboptimalities),
            inner_btls_steps=tuple(self.inner_btls_steps),
            # ``inner_quadratic_from`` is deliberately left unset. The barrier
            # IPM flags its quadratic phase off the Newton decrement against a
            # self-concordance threshold; this solver's inner metric is a
            # residual norm, which carries no equivalent guarantee, so there is
            # no honest index to report. Consequently
            # ``steps_past_quadratic_convergence()`` stays empty here and
            # consumers assert on the step trajectory directly instead -- that
            # it stays off ``min_step`` and reaches a substantial step.
            upper_bounds=tuple(self.upper_bounds),
            lower_bounds=tuple(self.lower_bounds),
            certified_upper_bounds=tuple(self.certified_upper_bounds),
            certified_lower_bounds=tuple(self.certified_lower_bounds),
        )


# Smallest interior margin used when recovering barrier multipliers for the
# certified bracket: constraint values are clamped below this (strictly negative)
# so lambda_i = -1/(t f_i) stays finite and nonnegative even for a non-interior
# iterate, keeping the certificate rigorous (see ``_block_inequality_duals``).
_INTERIOR_CLAMP: float = 1e-12


@dataclass(frozen=True)
class _BlockDual:
    """Warm-started per-block dual evaluator for the certified saddle-value bracket.

    Certifies one side of the saddle value V by minimizing a convex block Lagrangian
    over one block, with the other block frozen at the centered iterate. The closed-form
    central-path bracket (``StationaritySolver._saddle_value_bounds``) is exact only on
    the central path; this bound is a guaranteed one-sided certificate that holds off it
    too. It is the two-block analogue of ``BarrierSolver.evaluate_dual``, applied once
    per block.

    The objective is concave in x and convex in y (``neg_hessian_xx`` is positive
    definite), so
       V = max_x min_y ζ(x, y).
    Both sides of the bracket reduce to the same convex subproblem:
    - Lower bound. Freeze x at x_bar and minimize the y-block Lagrangian
         L_Y(y) = ζ(x_bar, y) + λ_y · f_y(y).
      By weak duality,
         inf_y L_Y ≤ min_y ζ(x_bar, y) = g(x_bar) ≤ V.
    - Upper bound. Freeze y at y_bar and minimize
         L~(x) = -ζ(x, y_bar) + λ_x · f_x(x) + ν · (A_eq x - b_eq).
      By weak duality,
         inf_x L~ ≤ min_x (-ζ(x, y_bar)) = -h(y_bar) ≤ -V,
      so -inf_x L~ ≥ V; the caller negates.
    Here g(x) = min_y ζ(x, y) and h(y) = max_x ζ(x, y), with the minima and maxima taken
    over each block's feasible set. Both minimands are convex (positive definite block
    curvature plus convex constraint curvature), so each is a well-posed unconstrained
    convex minimization, warm-started from the centered iterate, where the block
    Lagrangian is already approximately stationary.

    The multipliers are the per-block barrier duals λ_i = -1 / (t f_i) ≥ 0, and, for the
    x-block, ν = -``lam`` (the solver's equality multiplier, negated so that L~ is
    stationary at x_bar under the residual's +A_eq^T ``lam`` convention). The block
    objective ±ζ is defined on the whole block (the ``TwoBlockObjective`` contract), so,
    unlike the single-block evaluator, which caps steps at the objective's domain, the
    mini-Newton runs uncapped. The block Lagrangian carries no log barrier, only the
    finite penalty λ · f, so the inequality domain must not restrict the infimum; that
    would break weak duality.

    ``evaluate`` returns the mini-Newton value corrected by -δ^2, where δ is the Newton
    decrement. For a self-concordant function, that is a guaranteed lower bound on the
    infimum inside the quadratic-convergence region
       δ ≤ 0.68
    (Boyd and Vandenberghe, 2004, Eq. 9.50). Outside the region, ``evaluate`` returns
    -inf, a conservative deferral that never falsely certifies; a later, better-centered
    outer step resolves it.

    Fields
    ------
     objective_value : Callable[[npt.NDArray[np.float64]], float]
        The block objective ±ζ, with the other block frozen.
     objective_gradient : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        Its gradient in the block variable.
     objective_curvature : Callable[[npt.NDArray[np.float64]], MatrixSystem]
        Its positive definite curvature (``hessian_yy`` for y, ``neg_hessian_xx`` for
        x).
     domain : Sequence[ConstraintLayer]
        The block's inequality layers; may be empty.
     duals : npt.NDArray[np.float64]
        The barrier multipliers λ_i = -1 / (t f_i) for ``domain``, stacked in layer
        order; empty when the block is unconstrained.
     settings : OptimizationSettings
        Armijo and trace settings for the mini-Newton, shared with the solver.
     atnu : npt.NDArray[np.float64] | None
        Precomputed A_eq^T ν, the x-block's equality term in the gradient; ``None`` for
        the y-block, which never has an equality, or when there is no equality.
     equality_value : Callable[[npt.NDArray[np.float64]], float] | None
        The x-block's equality term ν · (A_eq w - b_eq) in the Lagrangian value;
        ``None`` omits it.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    objective_value: Callable[[npt.NDArray[np.float64]], float]
    objective_gradient: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    objective_curvature: Callable[[npt.NDArray[np.float64]], MatrixSystem]
    domain: Sequence[ConstraintLayer]
    duals: npt.NDArray[np.float64]
    settings: OptimizationSettings
    atnu: npt.NDArray[np.float64] | None = None
    equality_value: Callable[[npt.NDArray[np.float64]], float] | None = None

    def evaluate(self, w0: npt.NDArray[np.float64]) -> float:
        """Compute a certified lower bound on the block Lagrangian's infimum.

        Runs the warm-started mini-Newton; when it cannot step, falls back to w0 and a
        standalone Newton decrement. It then applies the quadratic-region guard and
        returns L(w_min) - δ^2, a guaranteed lower bound on the infimum inside the
        region.

        Parameters
        ----------
         w0 : npt.NDArray[np.float64]
            Warm start, the block's centered iterate.

        Returns
        -------
         bound : float
            The lower bound, or -inf when δ^2 is outside the region, unavailable or NaN,
            or when the bound is not finite.

        """
        w_min, delta2 = self._minimize(w0)
        if w_min is None:
            w_min = w0
            delta2 = self._newton_decrement_sq(w0)
        # ``not (delta2 <= MAX)`` also defers on NaN (every NaN comparison is
        # False), so a NaN decrement can never fall through and falsely certify.
        if delta2 is None or not (delta2 <= _DUAL_REGION_DECREMENT_SQ_MAX):
            return float("-inf")
        bound = self._lagrangian_value(w_min) - delta2
        # A non-finite Lagrangian value (e.g. from a degenerate dual) must defer,
        # not be reported as a certificate.
        return bound if math.isfinite(bound) else float("-inf")

    def _lagrangian_value(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the block Lagrangian L(w).

        Computes
           L(w) = f0(w) + λ · f(w)  [+ ν · (A_eq w - b_eq)],
        where f0 is ``objective_value`` and the bracketed term is present when
        ``equality_value`` is set. Each layer's slice of ``duals`` has length
        ``num_constraints``, the same convention ``_lagrangian_gradient`` and
        ``_hessian_solve`` use.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            The block point.

        Returns
        -------
         L : float
            The Lagrangian value.

        """
        result = self.objective_value(w)
        offset = 0
        for layer in self.domain:
            # Slice ``duals`` by ``num_constraints`` -- the same convention
            # ``_lagrangian_gradient`` and ``_hessian_solve`` use -- so all three
            # read identical per-layer slices of the dual vector.
            n_c = layer.num_constraints(w.size)
            f = np.asarray(layer.constraints(w), dtype=np.float64)
            result += float(np.dot(self.duals[offset : offset + n_c], f))
            offset += n_c
        if self.equality_value is not None:
            result += self.equality_value(w)
        return float(result)

    def _lagrangian_gradient(
        self, w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the block Lagrangian gradient.

        Computes
           ∇L(w) = ∇f0(w) + ∑ λ_i ∇f_i(w)  [+ A_eq^T ν],
        where the bracketed term is present when ``atnu`` is set.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            The block point.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient, of the same length as w.

        """
        g = np.asarray(self.objective_gradient(w), dtype=np.float64).copy()
        offset = 0
        for layer in self.domain:
            n_c = layer.num_constraints(w.size)
            g = g + layer.grad_constraints_transpose_multiply(
                w, self.duals[offset : offset + n_c]
            )
            offset += n_c
        if self.atnu is not None:
            g = g + self.atnu
        return g

    def _hessian_solve(
        self, w: npt.NDArray[np.float64]
    ) -> Callable[..., npt.NDArray[np.float64]]:
        """Return a solve operator for the block Lagrangian Hessian at w.

        Assembles the positive definite block curvature (``objective_curvature``) plus
        each layer's ∑ λ_i ∇^2 f_i (``ConstraintLayer.hessian_curvature``: curvature
        only, with no Jacobian outer-product term, and a ``ZeroSystem`` for affine
        layers). The equality term is linear in w and adds no curvature.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            The block point.

        Returns
        -------
         solve : Callable[..., npt.NDArray[np.float64]]
            The assembled ``MatrixSystem``'s ``solve`` method.

        """
        system = self.objective_curvature(w)
        offset = 0
        for layer in self.domain:
            n_c = layer.num_constraints(w.size)
            system = system.plus(
                layer.hessian_curvature(w, self.duals[offset : offset + n_c])
            )
            offset += n_c
        return system.solve

    def _minimize(
        self,
        w0: npt.NDArray[np.float64],
        max_iter: int = 5,
        tol: float = 1e-10,
    ) -> tuple[npt.NDArray[np.float64] | None, float | None]:
        """Minimize the block Lagrangian by a warm-started Armijo mini-Newton.

        The block objective is defined on the whole block, so the line search is
        uncapped (its step-cap finder returns inf), and the positive definite block
        Hessian makes every Newton step a descent direction. Each accepted step fires
        the trace's ``on_dual_newton_step``.

        Parameters
        ----------
         w0 : npt.NDArray[np.float64]
            Warm start; not modified.
         max_iter : int
            Maximum number of Newton steps.
         tol : float
            Stop when the squared Newton decrement is at most ``tol``.

        Returns
        -------
         w_min : npt.NDArray[np.float64] | None
            The last iterate, or ``None`` when no step could be taken (a failed or
            non-finite Hessian solve, or a failed line search).
         delta2 : float | None
            The squared Newton decrement at ``w_min`` (zero when the gradient norm is
            below 1e-10, and possibly ``None`` after ``max_iter`` steps when the final
            decrement cannot be computed), or ``None`` together with ``w_min``.

        """
        w = w0.copy()
        armijo = self.settings.armijo
        for nit in range(max_iter):
            step_start = time.time()
            g = self._lagrangian_gradient(w)
            grad_norm = float(np.linalg.norm(g))
            if grad_norm < 1e-10:
                return w, 0.0
            try:
                delta_w = self._hessian_solve(w)(-g)
            except (NewtonStepError, np.linalg.LinAlgError):
                return None, None
            if not np.all(np.isfinite(delta_w)):
                return None, None
            # delta_w = -H_L^{-1} g, so delta^2 = g^T H_L^{-1} g = -g^T delta_w >= 0.
            delta2 = max(0.0, -float(np.dot(g, delta_w)))
            if delta2 <= tol:
                return w, delta2
            merit = ArmijoBarrierMerit(
                evaluate_barrier_objective=self._lagrangian_value,
                ft0=self._lagrangian_value(w),
                neg_alpha_grad=armijo.alpha * delta2,
                grad_ft_dot_delta_x=-delta2,
            )
            line_search = LineSearch(
                merit=merit,
                max_step_finder=lambda ww, dd: math.inf,
                beta=armijo.beta,
                min_step=armijo.min_step,
            )
            try:
                s = line_search.find_step(w, delta_w, step_cap=1.0)
            except (
                ConstraintBoundaryError,
                SevereCurvatureError,
                InvalidDescentDirectionError,
            ):
                return None, None
            w = w + s * delta_w
            self.settings.trace.on_dual_newton_step(
                inner_nit=nit + 1,
                elapsed_ms=1000 * (time.time() - step_start),
                gradient_norm=grad_norm,
            )
        # max_iter exhausted: the last w is post-step, so recompute delta^2 at it.
        return w, self._newton_decrement_sq(w)

    def _newton_decrement_sq(self, w: npt.NDArray[np.float64]) -> float | None:
        """Compute the squared Newton decrement of the block Lagrangian at w.

        Computes
           δ^2 = ∇L^T (∇^2 L)^{-1} ∇L.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            The block point.

        Returns
        -------
         delta2 : float | None
            The squared decrement, clipped at zero; 0.0 when ‖∇L‖ < 1e-10 (the solve is
            skipped); ``None`` when the Hessian solve fails or is not finite, so the
            caller defers.

        """
        g = self._lagrangian_gradient(w)
        if float(np.linalg.norm(g)) < 1e-10:
            return 0.0
        try:
            z = self._hessian_solve(w)(g)
        except (NewtonStepError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(z)):
            return None
        return max(0.0, float(np.dot(g, z)))


class StationaritySolver:
    """Infeasible-start Newton barrier solver for a two-block stationarity problem.

    Drives a ``StationarityProblem`` (a smooth two-block objective ζ(x, y) with
    log-barrier domains) to a joint stationary point. The outer loop raises the barrier
    parameter t geometrically. Each inner centering solves the coupled Newton system
       [[-M_xx, B], [B^T, H_yy]]
    and takes an infeasible-start step that reduces ‖r‖: a cap that keeps both blocks
    inside their domains, then a residual-norm Armijo backtrack. The solve converges
    when the barrier gap n_barriers / t falls within
       max(``settings.outer.tolerance``, ``settings.outer.tolerance_rel`` |ζ|),
    an absolute floor, optionally relaxed to a fraction of the objective's scale (see
    ``_gap_tolerance``).

    The class stands apart from ``Solver``: a two-block ``StationarityProblem`` with a
    start (x0, y0) does not fit ``Solver.solve``'s single-``Problem``, single-``x0``
    signature, and the ‖r‖ centering has no single scalar objective to backtrack on (the
    same reasoning that keeps ``CoupledSaddleSystem`` off the ``SaddleSystem`` base).

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses ``OptimizationSettings()``.
     check_inertia : bool
        Whether to check the coupled saddle's block curvature signs once, at the first
        Newton step of the first outer iteration (see ``_check_inertia``).

    """

    def __init__(
        self,
        settings: OptimizationSettings | None = None,
        check_inertia: bool = True,
    ) -> None:
        self.settings: OptimizationSettings = (
            settings if settings is not None else OptimizationSettings()
        )
        self.check_inertia: bool = check_inertia
        # Trace ``solver`` label: honor ``observability.label`` when set (so a
        # caller can name this engine in its trace), else the concrete class.
        self._name: str = (
            self.settings.observability.label
            if self.settings.observability.label is not None
            else type(self).__name__
        )

    def solve(
        self,
        problem: StationarityProblem,
        x0: npt.NDArray[np.float64] | None = None,
        y0: npt.NDArray[np.float64] | None = None,
    ) -> TwoBlockSolveResult:
        """Solve a two-block problem from a start (x0, y0), feasibilizing as needed.

        A block's start need not be strictly interior. A start already strictly inside
        its domain is used as given (the infeasible-start Newton loop drives any
        ``x_equality`` residual to zero), while a ``None`` or non-interior start
        triggers a per-block Phase I that produces a strictly interior point, respecting
        ``x_equality`` for the x-block (see ``_resolve_block``). With no inequality
        constraints at all, the solver centers once and reports the exact saddle value.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x0 : npt.NDArray[np.float64] | None
            x-block start. ``None``, or a non-interior guess, is feasibilized by Phase
            I, which a supplied guess seeds; when ``x_domain`` is empty, ``None``
            becomes zeros.
         y0 : npt.NDArray[np.float64] | None
            y-block start, with the same semantics as ``x0`` (the y-block has no
            equality).

        Returns
        -------
         result : TwoBlockSolveResult
            The converged iterate, work counts, per-block duals, saddle value with its
            bracket, diagnostics and status (``OPTIMAL`` or ``ACCEPTABLE``).

        Raises
        ------
         StationarityConvergenceError
            When the barrier loop fails to converge: centering stalls, or the outer
            budget runs out with the gap above tolerance. The best-effort iterate is on
            the exception's ``result``, with status ``MAX_ITERATIONS``.
         ProblemInfeasibleError
            When Phase I finds that a block's domain has no strictly interior point.
         ValueError
            When the inertia check fails, or when ``x0`` or ``y0`` does not match its
            block's dimension.

        """
        x, y = self._resolve_start(problem, x0, y0)
        # ``on_ipm_start`` fires after Phase I feasibilization (mirroring the
        # barrier IPM), so the reported ``elapsed_ms`` covers the barrier loop
        # only; an infeasibility raised by Phase I short-circuits before the loop.
        trace = self.settings.trace
        trace.on_ipm_start(self._name)
        ipm_start = time.time()
        outer = self.settings.outer
        t = max(_INITIAL_BARRIER_PARAMETER, outer.min_initial_barrier_parameter)
        n_barriers = self._count_barriers(problem)
        if n_barriers == 0:
            trace.on_centering_start(1, t)
            center_start = time.time()
            x, y, lam, steps, converged, hard, inner_residuals, inner_steps = (
                self._center(problem, x, y, t, first_outer=True, outer_nit=1)
            )
            # No barriers, so the surrogate duality gap is identically zero and
            # the value bracket collapses to the exact saddle value.
            trace.on_centering_done(1, (time.time() - center_start) * 1000.0, 0.0)
            trace_data = _BarrierTrace()
            value = float(problem.objective.evaluate(x, y))
            cert_lower, cert_upper = self._certified_saddle_value_bounds(
                problem, x, y, t, lam
            )
            trace_data.record(
                0.0, inner_residuals, inner_steps, value, value, cert_upper, cert_lower
            )
            if converged:
                status = SolveStatus.OPTIMAL if hard else SolveStatus.ACCEPTABLE
                result = self._finalize(
                    problem, x, y, lam, t, steps, trace_data, status
                )
                trace.on_ipm_done(self._name, (time.time() - ipm_start) * 1000.0)
                return result
            trace.on_ipm_done(self._name, (time.time() - ipm_start) * 1000.0)
            return self._accept_or_raise(
                problem, x, y, lam, t, n_barriers, steps, trace_data
            )
        return self._run(problem, x, y, t, n_barriers, ipm_start)

    def _run(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
        n_barriers: int,
        ipm_start: float,
    ) -> TwoBlockSolveResult:
        """Run the outer barrier loop: center, test the gap, then raise t.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            Strictly interior x-block start.
         y : npt.NDArray[np.float64]
            Strictly interior y-block start.
         t : float
            Initial inverse barrier weight.
         n_barriers : int
            Total number of inequality constraints across both domains, at least 1.
         ipm_start : float
            The ``time.time()`` stamp that ``solve`` took just before ``on_ipm_start``;
            it anchors the elapsed time ``on_ipm_done`` reports, so that time spans the
            whole barrier loop.

        Returns
        -------
         result : TwoBlockSolveResult
            The result, with status ``OPTIMAL`` or ``ACCEPTABLE``.

        Raises
        ------
         StationarityConvergenceError
            When centering stops short of the soft tolerance (through
            ``_accept_or_raise``), or the outer budget runs out with the gap above
            tolerance.
         ValueError
            Propagated from ``_check_inertia``.

        """
        trace = self.settings.trace
        outer = self.settings.outer
        mu = outer.barrier_multiplier
        max_outer_iterations = outer.max_outer_iterations
        newton_steps = 0
        trace_data = _BarrierTrace()
        # Per-block inequality counts are fixed across the loop; they set the
        # closed-form central-path bracket offsets recorded at each outer step.
        m_x, m_y = self._block_barrier_counts(problem)
        # Bind ``gap`` on every path before the loop: it is read by ``_fail``
        # below, but is otherwise only assigned inside the loop body, so mypy
        # cannot see that ``max_outer_iterations >= 1`` guarantees at least one
        # iteration. The loop's first iteration overwrites this with the gap at
        # the centered iterate's own ``t``, so this initial value (the gap at the
        # starting ``t``) never reaches the reported suboptimality.
        gap = n_barriers / t
        # Bind ``lam`` on every path before the loop for the same reason as
        # ``gap`` above: it is read by ``_fail`` below but only assigned inside
        # the loop body (by ``_center``). ``None`` is the correct pre-loop value
        # -- it is the equality multiplier for a problem with no ``x_equality``.
        lam: npt.NDArray[np.float64] | None = None
        for outer_step in range(1, max_outer_iterations + 1):
            trace.on_centering_start(outer_step, t)
            center_start = time.time()
            x, y, lam, steps, converged, hard, inner_residuals, inner_steps = (
                self._center(
                    problem,
                    x,
                    y,
                    t,
                    first_outer=(outer_step == 1),
                    outer_nit=outer_step,
                )
            )
            newton_steps += steps
            # ``n_barriers / t`` is the barrier method's surrogate duality gap at
            # this centered point -- exactly what ``on_centering_done`` reports.
            gap = n_barriers / t
            trace.on_centering_done(
                outer_step, (time.time() - center_start) * 1000.0, gap
            )
            _, lower, upper = self._saddle_value_bounds(problem, x, y, t, m_x, m_y)
            cert_lower, cert_upper = self._certified_saddle_value_bounds(
                problem, x, y, t, lam
            )
            trace_data.record(
                gap, inner_residuals, inner_steps, upper, lower, cert_upper, cert_lower
            )
            if not converged:
                trace.on_ipm_done(self._name, (time.time() - ipm_start) * 1000.0)
                return self._accept_or_raise(
                    problem, x, y, lam, t, n_barriers, newton_steps, trace_data
                )
            if gap <= self._gap_tolerance(problem, x, y):
                trace.on_converged(outer_step, gap)
                trace.on_ipm_done(self._name, (time.time() - ipm_start) * 1000.0)
                return self._finalize(
                    problem,
                    x,
                    y,
                    lam,
                    t,
                    newton_steps,
                    trace_data,
                    SolveStatus.OPTIMAL if hard else SolveStatus.ACCEPTABLE,
                )
            t *= mu
        # Report the duality gap at the returned iterate's own barrier
        # parameter: the last ``gap`` computed inside the loop, before the
        # final ``t *= mu``. Recomputing ``n_barriers / t`` here (after the
        # multiply) would divide by the *next* iterate's larger ``t`` and
        # understate the reported suboptimality by ~mu.
        trace.on_ipm_done(self._name, (time.time() - ipm_start) * 1000.0)
        # The returned iterate was centered at ``t / mu`` -- the loop multiplied
        # ``t`` by ``mu`` after the last centering -- so pass that barrier
        # parameter for duals consistent with the reported (pre-multiply) ``gap``.
        self._fail(
            problem,
            x,
            y,
            lam,
            t / mu,
            newton_steps,
            trace_data,
            gap,
            f"exhausted the {max_outer_iterations}-outer-iteration barrier "
            f"budget with gap {gap:.03g} still above tolerance",
        )

    def _gap_tolerance(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> float:
        """Return the barrier-gap acceptance threshold.

        The threshold is
           max(tol_abs, tol_rel |ζ(x, y)|),
        with tol_abs = ``outer.tolerance`` and tol_rel = ``outer.tolerance_rel``. The
        absolute tolerance is a mandatory floor: it keeps the test meaningful when the
        objective is near zero (for example, a bilinear matrix game with optimal value
        zero), where a purely relative test would never terminate. The relative term
        relaxes the threshold for objectives in large units, which would otherwise chase
        absurd absolute precision. ``tolerance_rel = None`` leaves the test purely
        absolute.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The x-block iterate.
         y : npt.NDArray[np.float64]
            The y-block iterate.

        Returns
        -------
         tolerance : float
            The threshold the gap n_barriers / t must not exceed.

        """
        outer = self.settings.outer
        abs_tol = outer.tolerance
        if outer.tolerance_rel is None:
            return abs_tol
        value = abs(problem.objective.evaluate(x, y))
        return max(abs_tol, outer.tolerance_rel * value)

    def _center(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
        first_outer: bool,
        outer_nit: int,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
        int,
        bool,
        bool,
        list[float],
        list[float],
    ]:
        """Center by inner Newton steps at a fixed t.

        Each step solves the coupled Newton system for the barrier-smoothed residual and
        takes the step ``_residual_step`` accepts. Centering stops when ‖r‖ falls below
        ``inner.tolerance``, when the Newton solve or the step fails, or when
        ``inner.max_iterations`` steps are spent.

        With an ``x_equality``, the multiplier ``lam`` starts at zero (an
        infeasible-start Newton method: x need not satisfy A_eq x = b_eq), and each
        accepted step moves ``lam`` by the same step length as the primal blocks. It is
        unconstrained, so only x and y enter the domain cap.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            Strictly interior x-block start.
         y : npt.NDArray[np.float64]
            Strictly interior y-block start.
         t : float
            The inverse barrier weight to center at.
         first_outer : bool
            True on the first outer iteration; with ``check_inertia``, the first step
            then runs the inertia check, even when the start already meets
            ``inner.tolerance``.
         outer_nit : int
            The 1-indexed outer iteration, used only to tag each accepted step's
            ``on_saddle_newton_step`` trace event.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The centered x-block iterate.
         y : npt.NDArray[np.float64]
            The centered y-block iterate.
         lam : npt.NDArray[np.float64] | None
            The equality multiplier iterate; ``None`` when the problem has no
            ``x_equality``.
         steps : int
            The number of accepted Newton steps.
         converged : bool
            True when ‖r‖ is within ``inner.tolerance_soft``. A failed Newton solve or
            step returns False whatever the residual.
         hard : bool
            True when ‖r‖ also meets the tight ``inner.tolerance`` (a genuinely centered
            point), so the outer loop can report ``ACCEPTABLE`` instead of ``OPTIMAL``
            for a point that met only the soft tolerance.
         inner_residuals : list[float]
            The start-of-step residual norm of each accepted step, of length ``steps``.
         inner_steps : list[float]
            The step length accepted for each step, of length ``steps``.

        Raises
        ------
         ValueError
            Propagated from ``_check_inertia``.

        """
        inner = self.settings.inner
        trace = self.settings.trace
        n = problem.dimension_x
        m = problem.dimension_y
        lam = self._initial_multiplier(problem)
        steps = 0
        inner_residuals: list[float] = []
        inner_steps: list[float] = []
        for it in range(inner.max_iterations):
            step_start = time.time()
            r = problem.residual(x, y, t, lam)
            r_norm = float(np.linalg.norm(r))
            # The inertia check is once-only insurance (first Newton step of the
            # first outer iteration); run it even when the residual already
            # satisfies ``tolerance``, so an unusually good warm start cannot
            # silently skip the check.
            check_now = first_outer and it == 0 and self.check_inertia
            if r_norm < inner.tolerance:
                if check_now:
                    self._check_inertia(problem.build_coupled_system(x, y, t))
                return x, y, lam, steps, True, True, inner_residuals, inner_steps
            system = problem.build_coupled_system(x, y, t)
            if check_now:
                self._check_inertia(system)
            try:
                delta_x, delta_y, delta_lambda = self._newton_solve(system, r, n, m)
            except NewtonStepError as e:
                # A non-PD/singular/non-finite saddle has no Newton step; stop
                # centering and let the caller map the residual to a status.
                logger.debug("coupled saddle solve failed at t=%g: %s", t, e)
                return x, y, lam, steps, False, False, inner_residuals, inner_steps
            step = self._residual_step(
                problem, x, y, lam, t, r_norm, delta_x, delta_y, delta_lambda
            )
            if step is None:
                return x, y, lam, steps, False, False, inner_residuals, inner_steps
            x = x + step * delta_x
            y = y + step * delta_y
            if lam is not None and delta_lambda is not None:
                lam = lam + step * delta_lambda
            steps += 1
            # ``r_norm`` is the start-of-step residual (the metric the step drives
            # toward zero); ``steps`` is the 1-indexed inner iteration count. Record
            # it once per accepted step so ``len(inner_residuals) == steps``.
            inner_residuals.append(r_norm)
            inner_steps.append(step)
            lambda_norm = 0.0 if lam is None else float(np.linalg.norm(lam))
            trace.on_saddle_newton_step(
                outer_nit,
                steps,
                (time.time() - step_start) * 1000.0,
                r_norm,
                lambda_norm,
            )
        residual_norm = float(np.linalg.norm(problem.residual(x, y, t, lam)))
        return (
            x,
            y,
            lam,
            steps,
            residual_norm < inner.tolerance_soft,
            residual_norm < inner.tolerance,
            inner_residuals,
            inner_steps,
        )

    @staticmethod
    def _initial_multiplier(
        problem: StationarityProblem,
    ) -> npt.NDArray[np.float64] | None:
        """Return the zero starting equality multiplier, or None without an equality.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.

        Returns
        -------
         lam : npt.NDArray[np.float64] | None
            Zeros of length p, or ``None`` when the problem has no ``x_equality``.

        """
        p = problem.num_equalities
        if p == 0:
            return None
        return np.zeros(p, dtype=np.float64)

    @staticmethod
    def _newton_solve(
        system: CoupledSaddleSystem | EqualityCoupledSaddleSystem,
        r: npt.NDArray[np.float64],
        n: int,
        m: int,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | None,
    ]:
        """Solve the coupled Newton system against -r, dispatching on the border.

        Parameters
        ----------
         system : CoupledSaddleSystem | EqualityCoupledSaddleSystem
            The assembled coupled saddle.
         r : npt.NDArray[np.float64]
            The residual [r_x; r_y] or [r_x; r_y; r_eq].
         n : int
            Dimension of the x-block.
         m : int
            Dimension of the y-block.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The x-block step, of length n.
         delta_y : npt.NDArray[np.float64]
            The y-block step, of length m.
         delta_lambda : npt.NDArray[np.float64] | None
            The equality-multiplier step, or ``None`` for an unbordered saddle.

        Raises
        ------
         NewtonStepError
            Propagated from the system's ``solve`` when M_xx or the reduced complement
            cannot be solved.

        """
        if isinstance(system, EqualityCoupledSaddleSystem):
            return system.solve(-r[:n], -r[n : n + m], -r[n + m :])
        delta_x, delta_y = system.solve(-r[:n], -r[n:])
        return delta_x, delta_y, None

    def _residual_step(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam: npt.NDArray[np.float64] | None,
        t: float,
        r_norm0: float,
        delta_x: npt.NDArray[np.float64],
        delta_y: npt.NDArray[np.float64],
        delta_lambda: npt.NDArray[np.float64] | None,
    ) -> float | None:
        """Find the step length: a domain cap, then a backtrack on ‖r‖.

        Stage 1 caps the step at the domain boundary (``_domain_step_cap``). Stage 2
        backtracks on the residual norm with an Armijo test (``ResidualNormMerit``). The
        line search backs off from the cap by the factor β, so every trial point stays
        strictly interior. The equality multiplier ``lam`` is unconstrained: it does not
        enter the cap, but each trial residual moves it by the same step as the primal
        blocks.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            Current x-block iterate, strictly interior.
         y : npt.NDArray[np.float64]
            Current y-block iterate, strictly interior.
         lam : npt.NDArray[np.float64] | None
            Current equality multiplier, or ``None``.
         t : float
            The inverse barrier weight.
         r_norm0 : float
            ‖r‖ at the current iterate, already computed by ``_center``.
         delta_x : npt.NDArray[np.float64]
            The x-block Newton step.
         delta_y : npt.NDArray[np.float64]
            The y-block Newton step.
         delta_lambda : npt.NDArray[np.float64] | None
            The equality-multiplier step, or ``None``.

        Returns
        -------
         step : float | None
            The accepted step length, or ``None`` when the domain cap is below
            ``settings.armijo.min_step`` or the line search fails.

        """
        armijo = self.settings.armijo
        alpha_max = self._domain_step_cap(problem, x, y, delta_x, delta_y)
        if alpha_max < armijo.min_step:
            return None

        def compute_trial_residual_norm(s: float) -> float:
            """Compute ‖r‖ at the trial point for step length s.

            Parameters
            ----------
             s : float
                Trial step length.

            Returns
            -------
             r_norm : float
                The residual norm at (x + s Δx, y + s Δy), with the multiplier moved by
                the same step when there is one.

            """
            if lam is None or delta_lambda is None:
                trial_lam = None
            else:
                trial_lam = lam + s * delta_lambda
            trial = problem.residual(x + s * delta_x, y + s * delta_y, t, trial_lam)
            return float(np.linalg.norm(trial))

        merit = ResidualNormMerit(
            compute_trial_residual_norm=compute_trial_residual_norm,
            r_norm0=r_norm0,
            alpha=armijo.alpha,
        )
        # Stage 1 (``alpha_max``) is the raw fraction-to-*boundary*: the closed-form
        # ``btls_max_step`` overrides return the exact step at which a constraint hits
        # zero (``c_i = 0``), not a strictly-interior step. Feeding it as
        # ``max_step_finder`` lets ``LineSearch`` apply the fraction-to-boundary
        # back-off ``s_max = beta * alpha_max``, the same margin the barrier IPM uses,
        # so the first trial (and every accepted step) stays strictly interior. A no-op
        # ``inf`` finder with ``step_cap = alpha_max`` would instead put the first trial
        # exactly on the boundary, where the barrier gradient divides by zero and
        # iterates jam against the face. When nothing binds ``alpha_max`` is ``inf`` and
        # ``step_cap = 1.0`` recovers the full Newton step.
        line_search = LineSearch(
            merit=merit,
            max_step_finder=lambda w, dw: alpha_max,
            beta=armijo.beta,
            min_step=armijo.min_step,
        )
        w = np.concatenate([x, y])
        delta_w = np.concatenate([delta_x, delta_y])
        try:
            step = line_search.find_step(w, delta_w, step_cap=1.0)
        except (ConstraintBoundaryError, SevereCurvatureError) as e:
            logger.debug("residual-norm line search failed to find a step: %s", e)
            return None
        return float(step)

    def _domain_step_cap(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        delta_y: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step to the boundary of either block's domain.

        Each layer's cap comes from ``ConstraintLayer.btls_max_step``; the joint cap is
        their minimum. Layers with a closed-form bound report the step at which a
        constraint reaches zero, so the cap is the step to the boundary itself, not a
        strictly interior step; ``_residual_step``'s line search applies the back-off.
        The cap is not limited to 1, and with no domain on either block it is inf.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            Current x-block iterate, strictly interior.
         y : npt.NDArray[np.float64]
            Current y-block iterate, strictly interior.
         delta_x : npt.NDArray[np.float64]
            The x-block step direction.
         delta_y : npt.NDArray[np.float64]
            The y-block step direction.

        Returns
        -------
         alpha_max : float
            The smallest layer cap over both domains; inf when there are no layers.

        """
        line_search = self.settings.line_search
        caps: list[float] = []
        for layer in problem.x_domain:
            caps.append(layer.btls_max_step(x, delta_x, line_search=line_search))
        for layer in problem.y_domain:
            caps.append(layer.btls_max_step(y, delta_y, line_search=line_search))
        if not caps:
            return math.inf
        return min(caps)

    @staticmethod
    def _check_inertia(
        system: CoupledSaddleSystem | EqualityCoupledSaddleSystem,
    ) -> None:
        """Check once that the coupled saddle's curvature blocks have the right signs.

        For the symmetric saddle [[-M_xx, B], [B^T, H_yy]], positive definite M_xx and
        H_yy are sufficient for the correct inertia (n negative and m positive
        eigenvalues): with M_xx positive definite, the Haynsworth inertia formula
        reduces the question to the Schur complement
           S = H_yy + B^T M_xx^{-1} B,
        which is then positive definite. The condition is sufficient, not necessary, but
        it is the contract ``TwoBlockObjective`` states. With an equality border the
        same two-block test is the relevant curvature-sign check (the reduced complement
        is also positive definite when A_eq has full row rank, a structural precondition
        of the equality layer). Each block answers through its own structure-aware
        ``MatrixSystem.is_positive_definite``. The check runs once, not per step, so it
        is cheap insurance against a caller that supplies curvature with the wrong
        signs.

        Parameters
        ----------
         system : CoupledSaddleSystem | EqualityCoupledSaddleSystem
            The assembled coupled saddle.

        Raises
        ------
         ValueError
            When M_xx (``neg_hessian_xx`` plus barrier) or H_yy (``hessian_yy`` plus
            barrier) is not positive definite.

        """
        for name, block in (
            ("neg_hessian_xx", system.xx_system),
            ("hessian_yy", system.yy_system),
        ):
            if not block.is_positive_definite():
                raise ValueError(
                    f"coupled saddle lacks the expected inertia -- {name} must be "
                    "positive definite; check curvature signs"
                )

    def _resolve_start(
        self,
        problem: StationarityProblem,
        x0: npt.NDArray[np.float64] | None,
        y0: npt.NDArray[np.float64] | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Resolve the start for both blocks, feasibilizing as needed.

        The x-block carries any ``x_equality``; the y-block never has one. See
        ``_resolve_block``.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x0 : npt.NDArray[np.float64] | None
            The caller's x-block start.
         y0 : npt.NDArray[np.float64] | None
            The caller's y-block start.

        Returns
        -------
         x : npt.NDArray[np.float64]
            Strictly interior x-block start.
         y : npt.NDArray[np.float64]
            Strictly interior y-block start.

        Raises
        ------
         ValueError
            Propagated from ``_resolve_block`` when a start has the wrong shape.
         ProblemInfeasibleError
            Propagated from ``_phase1_point``.

        """
        x = self._resolve_block(
            x0, problem.dimension_x, problem.x_domain, problem.x_equality, "x"
        )
        y = self._resolve_block(y0, problem.dimension_y, problem.y_domain, None, "y")
        return x, y

    def _resolve_block(
        self,
        w0: npt.NDArray[np.float64] | None,
        dimension: int,
        domain: Sequence[ConstraintLayer],
        equality: EqualityConstraintLayer | None,
        name: str,
    ) -> npt.NDArray[np.float64]:
        """Return a strictly interior start for one block, running Phase I if needed.

        A supplied ``w0`` already strictly inside ``domain`` is returned as given: the
        infeasible-start Newton loop drives any ``equality`` residual to zero, so an
        interior start off the equality manifold is fine. When ``w0`` is ``None`` or not
        strictly interior, a Phase I solve over ``domain`` (subject to ``equality`` for
        the x-block), seeded by ``w0`` when one was supplied, produces a strictly
        interior point.

        Parameters
        ----------
         w0 : npt.NDArray[np.float64] | None
            The caller's start for this block.
         dimension : int
            The block's dimension.
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         equality : EqualityConstraintLayer | None
            The block's equality constraint (only ever set for the x-block).
         name : str
            The block's name, "x" or "y", used in error messages.

        Returns
        -------
         w : npt.NDArray[np.float64]
            A strictly interior start, of length ``dimension``.

        Raises
        ------
         ValueError
            When ``w0`` does not have length ``dimension``.
         ProblemInfeasibleError
            Propagated from ``_phase1_point``.

        """
        seed = None if w0 is None else np.asarray(w0, dtype=np.float64)
        if seed is not None and seed.shape != (dimension,):
            raise ValueError(
                f"{name}0 has shape {seed.shape}, expected ({dimension},)."
            )
        if seed is not None and self._strictly_interior(seed, domain):
            return seed
        return self._phase1_point(dimension, domain, equality, seed, name)

    @staticmethod
    def _strictly_interior(
        w: npt.NDArray[np.float64],
        domain: Sequence[ConstraintLayer],
    ) -> bool:
        """Test whether w satisfies every layer of a domain strictly.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            The block point.
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; an empty domain is satisfied vacuously.

        Returns
        -------
         interior : bool
            True when every constraint value is negative.

        """
        return all(bool(np.all(layer.constraints(w) < 0)) for layer in domain)

    def _phase1_point(
        self,
        dimension: int,
        domain: Sequence[ConstraintLayer],
        equality: EqualityConstraintLayer | None,
        seed: npt.NDArray[np.float64] | None,
        name: str,
    ) -> npt.NDArray[np.float64]:
        """Compute a strictly interior point of a domain, on the equality if one is set.

        With no inequality domain there is nothing to make feasible. ``_resolve_block``
        only reaches here with a ``None`` seed in that case, since a supplied seed is
        vacuously interior to an empty domain and returned as given, so the result is
        zeros, and any ``equality`` residual is left to the Newton loop. Otherwise a
        ``Problem`` with no objective goes through Cvxium's Phase I (``BarrierSolver``),
        which returns a point strictly inside ``domain`` and, when ``equality`` is set,
        on the equality manifold.

        Parameters
        ----------
         dimension : int
            The block's dimension.
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         equality : EqualityConstraintLayer | None
            The block's equality constraint, or ``None``.
         seed : npt.NDArray[np.float64] | None
            Starting guess for Phase I; must be ``None`` when ``domain`` is empty.
         name : str
            The block's name, "x" or "y", used in the error message.

        Returns
        -------
         w : npt.NDArray[np.float64]
            A strictly interior point, of length ``dimension``.

        Raises
        ------
         ProblemInfeasibleError
            When Phase I finds the domain infeasible; re-raised with a message naming
            the block.

        """
        if not domain:
            assert (
                seed is None
            ), "_resolve_block already returns a seed vacuously interior to an empty domain"
            return np.zeros(dimension, dtype=np.float64)
        problem = Problem(obj=None, eq=equality, ineq=list(domain))
        try:
            result = problem.solve(BarrierSolver(self.settings), x0=seed)
        except ProblemInfeasibleError as exc:
            raise ProblemInfeasibleError(
                f"{name}-block Phase I found no strictly-interior point; the "
                f"{name}_domain (with its equality, if any) appears infeasible."
            ) from exc
        return np.asarray(result.solution, dtype=np.float64)

    @staticmethod
    def _count_barriers(problem: StationarityProblem) -> int:
        """Count the inequality constraints across both block domains.

        This is the m in the surrogate duality gap m / t.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.

        Returns
        -------
         n_barriers : int
            Total number of inequality constraints.

        """
        total = 0
        for layer in problem.x_domain:
            total += layer.num_constraints(problem.dimension_x)
        for layer in problem.y_domain:
            total += layer.num_constraints(problem.dimension_y)
        return total

    @staticmethod
    def _block_barrier_counts(problem: StationarityProblem) -> tuple[int, int]:
        """Count each block's inequality constraints.

        These set the offsets in the closed-form central-path bracket: the x-block (the
        outer maximization) sets the upper-bound offset m_x / t, and the y-block (the
        inner minimization) the lower-bound offset m_y / t. Their sum is
        ``_count_barriers``.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.

        Returns
        -------
         m_x : int
            Number of x-domain constraints.
         m_y : int
            Number of y-domain constraints.

        """
        m_x = sum(
            layer.num_constraints(problem.dimension_x) for layer in problem.x_domain
        )
        m_y = sum(
            layer.num_constraints(problem.dimension_y) for layer in problem.y_domain
        )
        return m_x, m_y

    @staticmethod
    def _saddle_value_bounds(
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
        m_x: int,
        m_y: int,
    ) -> tuple[float, float, float]:
        """Compute the closed-form central-path bracket on the saddle value V.

        The objective is concave in x and convex in y (``neg_hessian_xx`` is positive
        definite), so a joint stationary point solves
           V = max_x min_y ζ(x, y).
        On the central path, centering drives the x-block residual ∇_x ζ - (1/t) ∇φ_x to
        zero (the centering condition for maximizing ζ(x, y_bar) over x) and the y-block
        residual ∇_y ζ + (1/t) ∇φ_y to zero (for minimizing ζ(x_bar, y) over y).
        Central-path duality, applied to each block (Boyd and Vandenberghe, 2004), then
        gives the two frozen-block bounds in closed form. With value = ζ(x, y),
           lower = value - m_y / t ≤ min_y ζ(x, y) ≤ V,
           upper = value + m_x / t ≥ max_x ζ(x, y) ≥ V,
        so upper - lower = (m_x + m_y) / t, the barrier's surrogate gap. The bracket is
        exact on the central path, and approximate to the centering tolerance off it,
        the same caveat the surrogate gap carries.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The centered x-block iterate.
         y : npt.NDArray[np.float64]
            The centered y-block iterate.
         t : float
            The inverse barrier weight at which (x, y) was centered.
         m_x : int
            Number of x-domain constraints.
         m_y : int
            Number of y-domain constraints.

        Returns
        -------
         value : float
            The objective value ζ(x, y).
         lower : float
            value - m_y / t.
         upper : float
            value + m_x / t.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        value = float(problem.objective.evaluate(x, y))
        return value, value - m_y / t, value + m_x / t

    def _certified_saddle_value_bounds(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
        lam: npt.NDArray[np.float64] | None,
    ) -> tuple[float, float]:
        """Compute the certified per-block dual bracket on the saddle value V.

        Unlike the closed-form bracket of ``_saddle_value_bounds``, which is exact only
        on the central path, this bracket holds off it too, at the cost of two
        warm-started mini-Newton solves of block Lagrangians on every call. Each side is
        a ``_BlockDual`` evaluated at the centered iterate. Either side may defer, to
        -inf (lower) or inf (upper), when its dual solve lands outside the
        quadratic-convergence region, so callers must treat an infinite bound as "not
        certified this step".

        The y-block (the inner minimization) minimizes
           L_Y(w) = ζ(x, w) + λ_y · f_y(w)
        for a lower bound on min_y ζ(x, y), and so on V. The x-block (the outer
        maximization) minimizes
           L~(w) = -ζ(w, y) + λ_x · f_x(w) + ν · (A_eq w - b_eq)
        for a lower bound on -max_x ζ(x, y), which negates into an upper bound on V. The
        multipliers are the per-block barrier duals
           λ_i = -1 / (t f_i),
        and ν is the negated equality multiplier, -``lam``. See ``_BlockDual`` for the
        weak-duality argument.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The centered x-block iterate.
         y : npt.NDArray[np.float64]
            The centered y-block iterate.
         t : float
            The inverse barrier weight at which (x, y) was centered.
         lam : npt.NDArray[np.float64] | None
            The equality multiplier, or ``None`` when the problem has no ``x_equality``.

        Returns
        -------
         certified_lower : float
            The certified lower bound on V, or -inf on a deferral.
         certified_upper : float
            The certified upper bound on V, or inf on a deferral.

        """
        obj = problem.objective
        lam_y = self._block_inequality_duals(problem.y_domain, y, t)
        y_dual = _BlockDual(
            objective_value=lambda w: float(obj.evaluate(x, w)),
            objective_gradient=lambda w: np.asarray(
                obj.gradient_y(x, w), dtype=np.float64
            ),
            objective_curvature=lambda w: obj.hessian_yy(x, w),
            domain=problem.y_domain,
            duals=lam_y,
            settings=self.settings,
        )
        certified_lower = y_dual.evaluate(y)

        lam_x = self._block_inequality_duals(problem.x_domain, x, t)
        atnu: npt.NDArray[np.float64] | None = None
        equality_value: Callable[[npt.NDArray[np.float64]], float] | None = None
        if problem.x_equality is not None and lam is not None:
            a_eq = problem.x_equality.A
            b_eq = np.asarray(problem.x_equality.b, dtype=np.float64)
            nu = -np.asarray(lam, dtype=np.float64)
            atnu = np.asarray(a_eq.T @ nu, dtype=np.float64).ravel()

            def _equality_value(w: npt.NDArray[np.float64]) -> float:
                """Evaluate the x-block equality term ν · (A_eq w - b_eq).

                Parameters
                ----------
                 w : npt.NDArray[np.float64]
                    The x-block point.

                Returns
                -------
                 value : float
                    The equality term.

                """
                return float(
                    np.dot(nu, np.asarray(a_eq @ w, dtype=np.float64).ravel() - b_eq)
                )

            equality_value = _equality_value
        x_dual = _BlockDual(
            objective_value=lambda w: -float(obj.evaluate(w, y)),
            objective_gradient=lambda w: -np.asarray(
                obj.gradient_x(w, y), dtype=np.float64
            ),
            objective_curvature=lambda w: obj.neg_hessian_xx(w, y),
            domain=problem.x_domain,
            duals=lam_x,
            settings=self.settings,
            atnu=atnu,
            equality_value=equality_value,
        )
        # ``_BlockDual`` bounds ``inf_x L~`` from below; negating turns that into
        # the certified upper bound on ``V``. A ``-inf`` deferral becomes ``+inf``.
        certified_upper = -x_dual.evaluate(x)
        return certified_lower, certified_upper

    def _accept_or_raise(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam: npt.NDArray[np.float64] | None,
        t: float,
        n_barriers: int,
        newton_steps: int,
        trace_data: _BarrierTrace,
    ) -> TwoBlockSolveResult:
        """Accept an early-stopped iterate as ACCEPTABLE if it meets the soft spec.

        Called when centering stopped short: the inner budget ran out above the soft
        tolerance, or a Newton solve or line search could make no further progress. Such
        an iterate is a valid ``ACCEPTABLE`` solution only when it is near-centered (‖r‖
        within ``inner.tolerance_soft``, so the gap estimate n_barriers / t is
        meaningful) and that gap is within ``_gap_tolerance``.

        The gap check is essential: without it, a small residual at a still-small t
        would pass as a solution while the iterate is still far from the saddle point,
        the barrier term still dominating. When either check fails the solve has not
        reached the user's tolerance, so this raises rather than returning a status a
        caller might overlook.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The x-block iterate.
         y : npt.NDArray[np.float64]
            The y-block iterate.
         lam : npt.NDArray[np.float64] | None
            The equality multiplier, or ``None``.
         t : float
            The inverse barrier weight at which centering stopped.
         n_barriers : int
            Total number of inequality constraints; zero gives a zero gap.
         newton_steps : int
            Total inner Newton steps so far.
         trace_data : _BarrierTrace
            The accumulated trajectories.

        Returns
        -------
         result : TwoBlockSolveResult
            The result, with status ``ACCEPTABLE``.

        Raises
        ------
         StationarityConvergenceError
            When the residual or the gap misses its tolerance, through ``_fail``.

        """
        residual_norm = float(np.linalg.norm(problem.residual(x, y, t, lam)))
        gap = n_barriers / t if n_barriers else 0.0
        if (
            residual_norm < self.settings.inner.tolerance_soft
            and gap <= self._gap_tolerance(problem, x, y)
        ):
            return self._finalize(
                problem, x, y, lam, t, newton_steps, trace_data, SolveStatus.ACCEPTABLE
            )
        self._fail(
            problem,
            x,
            y,
            lam,
            t,
            newton_steps,
            trace_data,
            max(gap, residual_norm),
            "the centering loop stopped short of tolerance "
            f"(||r|| = {residual_norm:.03g}, gap = {gap:.03g})",
        )

    def _fail(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam: npt.NDArray[np.float64] | None,
        t: float,
        newton_steps: int,
        trace_data: _BarrierTrace,
        suboptimality: float,
        cause: str,
    ) -> NoReturn:
        """Raise ``StationarityConvergenceError`` carrying the best-effort iterate.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The x-block iterate.
         y : npt.NDArray[np.float64]
            The y-block iterate.
         lam : npt.NDArray[np.float64] | None
            The equality multiplier, or ``None``.
         t : float
            The inverse barrier weight at which (x, y) was centered.
         newton_steps : int
            Total inner Newton steps.
         trace_data : _BarrierTrace
            The accumulated trajectories.
         suboptimality : float
            The failure metric reported on the exception.
         cause : str
            Which break fired, as a clause.

        Raises
        ------
         StationarityConvergenceError
            Always, with the best-effort result attached at status ``MAX_ITERATIONS``.

        Notes
        -----
        A non-convergence outcome (a failed Newton or line-search step, an exhausted
        inner budget, or an exhausted outer loop) is a failure, not a conclusion of
        infeasibility, and this engine raises rather than returning a status the caller
        might overlook, as the barrier method does with ``InteriorPointMethodError``.
        Infeasibility is never reported here; a genuine certificate surfaces as
        ``ProblemInfeasibleError`` from Phase I. The best-effort iterate is attached so
        a caller that catches the error can inspect the last point without re-solving.

        ``cause`` reaches the reader twice and must say the same thing both times: in
        the exception's text, and in the ``message`` of the attached result, which a
        caller inspecting ``exc.result`` reads instead. Passing it through ``_finalize``
        keeps the two consistent; on its own, the shared formatter has only
        ``MAX_ITERATIONS`` to go on, a status all three breaks share, and would report
        each as a spent iteration budget.

        """
        result = self._finalize(
            problem,
            x,
            y,
            lam,
            t,
            newton_steps,
            trace_data,
            SolveStatus.MAX_ITERATIONS,
            cause=cause,
        )
        message = f"Stationarity solve did not converge: {cause}."
        raise StationarityConvergenceError(message, suboptimality, result)

    @staticmethod
    def _reported_bracket(
        cert_lower: float,
        cert_upper: float,
        belt_lower: float,
        belt_upper: float,
    ) -> tuple[float, float]:
        """Choose the reported dual value and bracket width from the two brackets.

        ``dual_value`` is the certified lower bound when it is finite, and otherwise the
        closed-form central-path lower bound value - m_y / t. The width, however, must
        come from a single self-consistent bracket: off the central path, a certified
        bound on one side can cross the closed-form bound on the other (a certified
        lower bound above the closed-form upper bound), so mixing sides could report a
        negative width. The certified width is therefore reported only when both
        certified sides are finite, where cert_lower ≤ V ≤ cert_upper keeps it
        nonnegative; otherwise the width is the closed-form (m_x + m_y) / t, which is
        always nonnegative.

        Parameters
        ----------
         cert_lower : float
            The certified lower bound, or -inf on a deferral.
         cert_upper : float
            The certified upper bound, or inf on a deferral.
         belt_lower : float
            The closed-form lower bound.
         belt_upper : float
            The closed-form upper bound.

        Returns
        -------
         dual_value : float
            The reported lower bound on V.
         suboptimality : float
            The reported bracket width.

        """
        both_certified = math.isfinite(cert_lower) and math.isfinite(cert_upper)
        dual_value = cert_lower if math.isfinite(cert_lower) else belt_lower
        suboptimality = (
            cert_upper - cert_lower if both_certified else belt_upper - belt_lower
        )
        return dual_value, suboptimality

    def _finalize(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam: npt.NDArray[np.float64] | None,
        t: float,
        newton_steps: int,
        trace_data: _BarrierTrace,
        status: SolveStatus,
        cause: str | None = None,
    ) -> TwoBlockSolveResult:
        """Assemble the ``TwoBlockSolveResult`` with duals, bracket and diagnostics.

        ``n_iterations`` is the total inner Newton step count. The composed ``duals``
        carry the equality multiplier ``lam`` and the barrier inequality multipliers
        recovered at the returned iterate (see ``_inequality_duals``); the composed
        ``diagnostics`` freeze the per-step trajectories, including both brackets.

        The composed ``objective`` reports the saddle value ζ(x, y) with a certificate,
        preferring the certified per-block dual bracket over the closed-form
        central-path bracket whenever the final outer step's dual solves stayed in
        region (see ``_reported_bracket``). The upper ends of both brackets are in the
        diagnostics' ``upper_bounds`` and ``certified_upper_bounds``.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The returned x-block iterate.
         y : npt.NDArray[np.float64]
            The returned y-block iterate.
         lam : npt.NDArray[np.float64] | None
            The equality multiplier, or ``None``.
         t : float
            The inverse barrier weight at which (x, y) was centered.
         newton_steps : int
            Total inner Newton steps.
         trace_data : _BarrierTrace
            The accumulated trajectories; its last entry was recorded at this iterate.
         status : SolveStatus
            The result status.
         cause : str | None
            The ``_fail`` path's account of which break fired, forwarded to the message
            so the best-effort result names the real termination rather than the
            exhausted budget the shared formatter infers from a bare ``MAX_ITERATIONS``.
            ``None`` on the converged paths, whose status says it all.

        Returns
        -------
         result : TwoBlockSolveResult
            The assembled result.

        """
        m_x, m_y = self._block_barrier_counts(problem)
        value, lower, upper = self._saddle_value_bounds(problem, x, y, t, m_x, m_y)
        # Prefer the rigorous per-block dual certificate when this iterate's
        # solve stayed in region; otherwise fall back to the closed-form belt.
        # The final trace entry was recorded at this same iterate on every path
        # reaching ``_finalize`` (the outer loop records before converging/
        # accepting/failing, and the no-barrier path records before finalizing).
        cert_lower = (
            trace_data.certified_lower_bounds[-1]
            if trace_data.certified_lower_bounds
            else lower
        )
        cert_upper = (
            trace_data.certified_upper_bounds[-1]
            if trace_data.certified_upper_bounds
            else upper
        )
        dual_value, suboptimality = self._reported_bracket(
            cert_lower, cert_upper, lower, upper
        )
        objective = ObjectiveOutcome(
            value=value, suboptimality=suboptimality, dual_value=dual_value
        )
        ineq = self._inequality_duals(problem, x, y, t)
        duals = (
            None
            if lam is None and ineq is None
            else Duals(equality_multipliers=lam, inequality_multipliers=ineq)
        )
        return TwoBlockSolveResult.from_blocks(
            x=x,
            y=y,
            status=status,
            message=two_block_result_message(
                status, self._name, suboptimality, cause=cause
            ),
            solver=self._name,
            n_iterations=newton_steps,
            objective=objective,
            duals=duals,
            diagnostics=trace_data.to_diagnostics(),
        )

    @staticmethod
    def _inequality_duals(
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
    ) -> npt.NDArray[np.float64] | None:
        """Recover the barrier inequality multipliers λ_i = -1 / (t f_i).

        This is the standard log-barrier dual estimate at the returned iterate: for each
        strictly satisfied constraint f_i(w) < 0, -1 / (t f_i) is the corresponding
        nonnegative Lagrange multiplier. The x-block constraints are evaluated at x and
        the y-block's at y, then concatenated in that order, as in ``_count_barriers``.

        Only a strictly interior constraint has a valid multiplier. A best-effort
        iterate returned at ``MAX_ITERATIONS`` may sit on or past the boundary (a
        nonnegative f_i), where -1 / (t f_i) is infinite or negative; such entries are
        NaN rather than a misleading value, so callers can mask them.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem.
         x : npt.NDArray[np.float64]
            The x-block iterate.
         y : npt.NDArray[np.float64]
            The y-block iterate.
         t : float
            The inverse barrier weight.

        Returns
        -------
         duals : npt.NDArray[np.float64] | None
            The multipliers, x-block first; ``None`` when neither block has an
            inequality domain.

        """
        duals: list[float] = []
        for layer in problem.x_domain:
            f = np.asarray(layer.constraints(x), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(f < 0.0, -1.0 / (t * f), np.nan)
            duals.extend(lam.tolist())
        for layer in problem.y_domain:
            f = np.asarray(layer.constraints(y), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(f < 0.0, -1.0 / (t * f), np.nan)
            duals.extend(lam.tolist())
        if not duals:
            return None
        return np.asarray(duals, dtype=np.float64)

    @staticmethod
    def _block_inequality_duals(
        domain: Sequence[ConstraintLayer],
        w: npt.NDArray[np.float64],
        t: float,
    ) -> npt.NDArray[np.float64]:
        """Recover the barrier multipliers λ_i = -1 / (t f_i) for one block's domain.

        The per-block counterpart of ``_inequality_duals``, stacked in layer order to
        weight the block's constraints in its ``_BlockDual`` Lagrangian. Weak duality
        needs only λ_i ≥ 0, so a constraint value that is not strictly negative is
        clamped to -``_INTERIOR_CLAMP`` first. The multiplier stays finite and
        nonnegative, and the bound stays rigorous, just looser; a true boundary hit
        gives a huge multiplier, on which the region guard defers.

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         w : npt.NDArray[np.float64]
            The block iterate.
         t : float
            The inverse barrier weight.

        Returns
        -------
         duals : npt.NDArray[np.float64]
            The multipliers, stacked in layer order; empty for an unconstrained block.

        """
        duals: list[float] = []
        for layer in domain:
            f = np.asarray(layer.constraints(w), dtype=np.float64)
            # Weak duality only needs lambda_i >= 0, which the reciprocal gives
            # for a strictly-interior f_i < 0. Clamp non-interior entries to a
            # tiny negative so lambda stays finite and nonnegative: the bound
            # remains rigorous (any lambda >= 0 is valid), just looser, and a true
            # boundary hit yields a huge multiplier the region guard defers on --
            # never inf/nan silently poisoning the certificate.
            f = np.minimum(f, -_INTERIOR_CLAMP)
            duals.extend((-1.0 / (t * f)).tolist())
        return np.asarray(duals, dtype=np.float64)
