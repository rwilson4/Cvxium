"""Interior-point (barrier) solver over a Cvxium ``Problem``, auto-dispatching.

Defines ``BarrierSolver``, the public interior-point method for inequality-constrained
problems, and its private ``_BarrierAdapter``, which bridges a ``Problem`` to Cvxium's
shared ``InteriorPointMethodSolver`` loop. The solver auto-dispatches on whether the
``Problem`` carries an equality layer: problems with only inequalities use the plain
barrier method, while problems with equality constraints use the KKT-system barrier
method. This module also holds the standalone barrier-parameter initialization helpers
shared by the inequality-only and equality paths.

"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._typing import none_throws
from ..exceptions import (
    ConstraintBoundaryError,
    InvalidDescentDirectionError,
    NewtonStepError,
    SevereCurvatureError,
)
from ..problems.constraint_layers import (
    DirectConstraintLayer,
)
from ..problems.result import (
    SolveResult,
    SolveStatus,
)
from ..solvers.line_search import (
    ArmijoBarrierMerit,
    LineSearch,
)
from ..solvers.optimization import (
    BaseInteriorPointMethodSolver,
    InteriorPointMethodSolver,
    ProblemCertifiablyInfeasibleError,
    TrivialFeasibilitySolver,
    _barrier_solveresult,
)
from ..solvers.phase1 import _SlackActiveLayer
from ..solvers.settings import (
    BarrierInitStrategy,
    OptimizationSettings,
)
from ..solvers.solver import (
    EqualitySolver,
    Solver,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..problems.objective import ObjectivePoint
    from ..problems.problem import (
        OnInfeasible,
        Problem,
    )

# Largest squared Newton decrement for which the dual correction
# ``L(x) - inf_x L <= delta^2`` is a valid bound. The bound holds only inside the
# self-concordant quadratic-convergence region, delta <= ~0.68 (Boyd and
# Vandenberghe, 2004, Eq 9.50). Outside it we cannot certify a lower bound, so
# ``evaluate_dual`` returns -inf rather than a possibly-invalid bound.
_DUAL_REGION_DECREMENT_SQ_MAX: float = 0.68**2


def _gradient_balance_t0(
    grad_f0: npt.NDArray[np.float64],
    grad_phi: npt.NDArray[np.float64],
    floor: float,
) -> float:
    r"""Initialize the barrier parameter by the gradient balance strategy.

    Balances the objective gradient against the log-barrier gradient at the starting
    point, so that neither dominates the first centering step:
       t0 = max(||∇φ|| / ||∇f0||, floor).

    Parameters
    ----------
     grad_f0 : npt.NDArray[np.float64]
        Objective gradient at x0, ∇f0.
     grad_phi : npt.NDArray[np.float64]
        Log-barrier gradient at x0, ∇φ.
     floor : float
        Strictly positive lower bound on t0
        (``OuterSettings.min_initial_barrier_parameter``).

    Returns
    -------
     t0 : float
        The initial barrier parameter, or ``floor`` when ||∇f0|| = 0.

    Notes
    -----
    Shared across the adapter's no-equality and equality init paths (the equality path
    uses this when its ``barrier_init`` is ``GRADIENT_BALANCE`` rather than the
    equality-aware ``EQUALITY_SCHUR``). The floor keeps t0 strictly positive: the ratio
    is exactly zero when ||∇f0|| = 0 (the unconstrained minimizer of f0 coincides with
    x0) or ||∇φ|| = 0 (a strict box center where the bound forces cancel, common in
    LP-style problems). A t0 of zero poisons the IPM's ``num_steps`` calculation, since
    log(t0 ε) = -inf for outer tolerance ε. Keep the floor small: a too-large t0 starts
    the first centering far off the central path, where Newton diverges and the
    backtracking line search collapses.

    """
    grad_f0_norm = float(np.linalg.norm(grad_f0))
    grad_phi_norm = float(np.linalg.norm(grad_phi))
    if grad_f0_norm == 0.0:
        return floor
    return max(grad_phi_norm / grad_f0_norm, floor)


def _objective_value_gap_t0(
    f0: float,
    p_star: float | None,
    num_ineq: int,
    barrier_multiplier: float,
    outer_tolerance: float,
) -> float:
    r"""Initialize the barrier parameter by the objective value gap strategy.

    Chooses t0 so that the duality gap bound after the first barrier increase,
       m / (μ t0),
    equals the gap between the starting objective value and an under-estimate of the
    optimum.

    Parameters
    ----------
     f0 : float
        Objective value at x0.
     p_star : float | None
        Under-estimate of the optimal objective value
        (``OuterSettings.objective_value_estimate``); None disables the heuristic and
        returns 1.0.
     num_ineq : int
        Number of inequality constraints, m.
     barrier_multiplier : float
        Per-step barrier increase factor, μ.
     outer_tolerance : float
        Outer duality-gap tolerance, ε, used to compute the cap t_max.

    Returns
    -------
     t0 : float
        The initial barrier parameter, floored at 1.0 and then capped at t_max (so t_max
        wins when t_max < 1); 1.0 when ``p_star`` is None.

    Notes
    -----
    Following (Boyd and Vandenberghe, 2004, p. 570), initialize the barrier parameter as
       t0 = m / (μ (f0 - p_star)),
    where p_star is an under-estimate of the optimal objective value, m is the
    inequality count, and μ is the barrier multiplier. The result is capped at
       t_max = m / (μ ε),
    one step from the end of the barrier schedule, and floored at 1.0 (at least one
    centering step). When x0 is already near optimal (f0 <= p_star), t0 is set near
    t_max (t_max / μ^3) so the outer loop takes only a couple of steps. Shared across
    the adapter's no-equality and equality init paths.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """
    if p_star is None:
        return 1.0
    t_max = num_ineq / (barrier_multiplier * outer_tolerance)
    if f0 <= p_star + 1e-12:
        # Near-optimal: skip to near the end of the barrier schedule.
        t0 = t_max / (barrier_multiplier**3)
    else:
        t0 = num_ineq / (barrier_multiplier * (f0 - p_star))
    return min(max(t0, 1.0), t_max)


def _raise_to_objective_hint(
    t0: float,
    problem: Problem,
    x0: npt.NDArray[np.float64],
    num_ineq: int,
    outer_tolerance: float,
) -> float:
    r"""Raise ``t0`` to a direct-form active layer's ``barrier_parameter_hint``.

    In a direct-form Phase I, ``Problem.make_phase1`` hoists the active
    ``DirectConstraintLayer`` to the objective. Such a layer may expose a problem-scaled
    lower bound on the barrier parameter, which the structured barrier path would
    otherwise ignore, under-progressing the centering on large problems.

    Parameters
    ----------
     t0 : float
        Candidate initial barrier parameter to raise.
     problem : Problem
        The problem being solved; its objective is inspected for a hint.
     x0 : npt.NDArray[np.float64]
        Starting iterate passed through to the layer's hint computation.
     num_ineq : int
        Number of inequality constraints, passed through to the hint computation.
     outer_tolerance : float
        Outer duality-gap tolerance, passed through to the hint computation.

    Returns
    -------
     t0 : float
        max(t0, hint) when the objective is a ``DirectConstraintLayer`` that offers a
        hint; otherwise ``t0`` unchanged (the Phase II case, or a layer whose hint is
        the ``None`` default).

    """
    objective = problem.obj
    if not isinstance(objective, DirectConstraintLayer):
        return t0
    hint = objective.barrier_parameter_hint(
        x0, num_ineq=num_ineq, outer_tolerance=outer_tolerance
    )
    if hint is None:
        return t0
    return max(t0, hint)


class _BarrierAdapter(InteriorPointMethodSolver):
    """Bridge from a ``Problem`` to ``InteriorPointMethodSolver``, auto-dispatching.

    Implements the abstract methods of ``InteriorPointMethodSolver`` by composing across
    ``problem.ineq`` and forwarding to ``problem.obj``. Private; users instantiate
    ``BarrierSolver`` instead.

    Parameters
    ----------
     problem : Problem
        The problem to solve; its objective, inequality layers, and optional equality
        layer drive every adapter method.
     settings : OptimizationSettings | None
        Solver settings; ``None`` uses the defaults.
     equality : EqualitySolver | None
        Optional pre-built equality collaborator (sharing the equality layer's cached
        SVD). Pass ``None`` to have the adapter build one from ``problem.eq``. Must be
        ``None`` when ``problem.eq`` is ``None``; the constructor raises ``ValueError``
        on that inconsistent configuration.

    Notes
    -----
    The ``phase1_solver`` is a ``TrivialFeasibilitySolver`` because feasibility is
    established by the outer ``BarrierSolver`` before the adapter is constructed.

    Handles both the no-equality and equality cases in a single class: when
    ``problem.eq`` is set, it builds an ``EqualitySolver`` collaborator and the
    equality-aware methods thread the equality matrix and Lagrange multiplier through;
    when ``problem.eq`` is ``None`` they collapse to the plain inequality-only path.
    Newton steps in the equality case route through the equality's ``saddle_system``
    seam (the KKT saddle), composed with the structured H_ft^{-1} assembled from the
    objective and the constraint layers.

    """

    def __init__(
        self,
        problem: Problem,
        settings: OptimizationSettings | None = None,
        equality: EqualitySolver | None = None,
    ) -> None:
        super().__init__(
            phase1_solver=TrivialFeasibilitySolver(settings=settings),
            settings=settings,
        )
        self._problem = problem
        # An `equality` collaborator only makes sense when the Problem actually carries
        # an equality layer; otherwise the equality-aware methods would thread a real A
        # while ``problem.eq`` reads as absent -- a latent inconsistency that would
        # later trip deep in ``calculate_newton_step``. Reject it at construction
        # instead of failing opaquely.
        if equality is not None and problem.eq is None:
            raise ValueError(
                "_BarrierAdapter received an `equality` collaborator but the "
                "Problem has no equality layer (problem.eq is None); pass "
                "eq=... on the Problem, or omit `equality`."
            )
        # The equality projection / KKT collaborator over the Problem's equality layer
        # (sharing its cached SVD) or ``None`` for an inequality-only problem. On the
        # objective path ``BarrierSolver.solve`` builds one for the x0 projection and
        # passes it here, so that projection and this adapter's KKT solves share a
        # single instance. (The feasibility (obj=None) path builds no adapter: it
        # projects x0 with that collaborator but then hands the solve to
        # ``_solve_feasibility``, which builds its own from ``problem.eq``.) Callers
        # without one (e.g. the primal-dual adapter) leave it ``None`` and we build it
        # from ``problem.eq`` with this solver's (normalized) ``self.settings``.
        self._equality: EqualitySolver | None
        if equality is not None:
            self._equality = equality
        elif problem.eq is not None:
            self._equality = EqualitySolver(
                constraint=problem.eq, settings=self.settings
            )
        else:
            self._equality = None
        # Per-x cached ObjectivePoint so evaluate / gradient / hessian_contribution at
        # the same x share one forward pass. Keyed on array *contents*, not identity:
        # ``centering_step`` advances the iterate in place (``x += delta_x``), so the
        # array object is reused across iterations while its contents change. An
        # identity check (``is``) would freeze the cache on the first iterate and hand
        # back a stale point for every subsequent one, silently degrading Newton's
        # method to a frozen-Jacobian iteration. A content snapshot (``x.copy()``) is
        # stored as the key so a later in-place mutation of ``x`` cannot retroactively
        # alter it.
        self._cached_x: npt.NDArray[np.float64] | None = None
        self._cached_point: ObjectivePoint | None = None

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """De-augment a solution: drop a slack-form Phase I's trailing slack.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The (possibly slack-augmented) solution vector to de-augment.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The first ``base_dimension`` components when the active layer is a
            ``_SlackActiveLayer``; otherwise ``x`` unchanged.

        Notes
        -----
        Certain feasibility problems augment the primal variable with a slack variable.
        This method removes the slack variable (when present). For other problems, there
        is no slack variable and this method simply returns x.

        This fires at every barrier result-construction site, so the IPM's infeasibility
        certificate and ``on_infeasible="continue"`` best-effort results (raised from
        inside the loop, bypassing ``_solve_phase1``'s own de-augmentation of the return
        path) carry the original dimension. Callers reading ``error.result`` then need
        not slice off the slack themselves.

        """
        active = self._problem.ineq[-1] if self._problem.ineq else None
        if isinstance(active, _SlackActiveLayer):
            return np.asarray(x)[: active.base_dimension]
        return x

    def _point(self, x: npt.NDArray[np.float64]) -> ObjectivePoint:
        """Return the cached ``ObjectivePoint`` at ``x``, recomputing on a miss.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the objective is evaluated.

        Returns
        -------
         point : ObjectivePoint
            The objective's forward pass at ``x``, shared by the value, gradient, and
            Hessian contribution at the same point.

        """
        if (
            self._cached_point is None
            or self._cached_x is None
            or not np.array_equal(self._cached_x, x)
        ):
            self._cached_point = none_throws(self._problem.obj).at(x)
            self._cached_x = x.copy()
        return self._cached_point

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Return the equality matrix A.

        Returns
        -------
         A : npt.NDArray[np.float64]
            The equality matrix, of shape (p, n); shape (0, n) when the Problem has no
            equality layer.

        """
        if self._problem.eq is None:
            return np.zeros((0, self._problem.dimension))
        return self._problem.eq.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Return the equality right-hand side b.

        Returns
        -------
         b : npt.NDArray[np.float64]
            The equality right-hand side, of length p; empty when the Problem has no
            equality layer.

        """
        if self._problem.eq is None:
            return np.zeros(0)
        return self._problem.eq.b

    @property
    def num_eq_constraints(self) -> int:
        """Count the equality constraints.

        Returns
        -------
         p : int
            Number of rows in the Problem's equality layer; 0 when it has none.

        """
        if self._problem.eq is None:
            return 0
        return self._problem.eq.num_eq_constraints

    @property
    def num_ineq_constraints(self) -> int:
        """Count the inequality constraints across all layers.

        Returns
        -------
         m : int
            Sum of the constraint counts of the Problem's inequality layers.

        """
        return self._problem.num_inequality_constraints()

    def _feasibility_threshold(self) -> float | None:
        """Return the objective's feasibility threshold.

        Returns
        -------
         threshold : float | None
            The threshold below which a feasibility (Phase I) solve stops; ``None`` for
            an ordinary optimization solve.

        """
        return none_throws(self._problem.obj).feasibility_threshold

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective f0 at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the objective is evaluated.

        Returns
        -------
         f0 : float
            The objective value, from the cached ``ObjectivePoint``.

        """
        return self._point(x).evaluate()

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the objective gradient ∇f0 at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the gradient is evaluated.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The objective gradient, of length n, from the cached ``ObjectivePoint``.

        """
        return self._point(x).gradient()

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Stack the constraint values fi(x) across all inequality layers.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the constraints are evaluated.

        Returns
        -------
         f : npt.NDArray[np.float64]
            The constraint values, of length m, in layer order; empty when there are no
            inequality layers.

        """
        if not self._problem.ineq:
            return np.zeros(0)
        return np.concatenate([layer.constraints(x) for layer in self._problem.ineq])

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Stack the constraint Jacobians across all inequality layers.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated.

        Returns
        -------
         J : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (m, n), whose row i is ∇fi(x).

        """
        if not self._problem.ineq:
            return np.zeros((0, self._problem.dimension))
        return np.vstack([layer.grad_constraints(x) for layer in self._problem.ineq])

    def grad_constraints_multiply(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the stacked constraint Jacobian, layer by layer.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated.
         y : npt.NDArray[np.float64]
            Vector of length n to multiply.

        Returns
        -------
         Jy : npt.NDArray[np.float64]
            The product J y, of length m, computed without forming J.

        """
        if not self._problem.ineq:
            return np.zeros(0)
        return np.concatenate(
            [layer.grad_constraints_multiply(x, y) for layer in self._problem.ineq]
        )

    def grad_constraints_transpose_multiply(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian, layer by layer.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated.
         y : npt.NDArray[np.float64]
            Vector of length m to multiply, sliced into one block per layer.

        Returns
        -------
         JTy : npt.NDArray[np.float64]
            The product J^T y, of length n, accumulated across layers without forming J.

        """
        result = np.zeros(self._problem.dimension)
        offset = 0
        for layer in self._problem.ineq:
            n_c = layer.num_constraints(self._problem.dimension)
            u_slice = y[offset : offset + n_c]
            result = result + layer.grad_constraints_transpose_multiply(x, u_slice)
            offset += n_c
        return result

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        r"""Test whether x is feasible enough to stop a feasibility (Phase I) solve.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Candidate iterate to test.

        Returns
        -------
         feasible : bool
            True when the feasibility-residual objective is negative at x, False
            otherwise.

        Notes
        -----
        A feasibility solver's objective is its feasibility residual. For a slack-form
        Phase I problem the objective is the slack s. For a direct-form Phase I problem
        it is the active constraint value. The original problem is strictly feasible
        once that residual drops below zero (the barrier keeps every constraint strictly
        inside its domain, so a negative objective means a strictly interior iterate).
        Only consulted in feasibility mode; an optimization solve never calls it.

        """
        return bool(none_throws(self._problem.obj).evaluate(x) < 0.0)

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SolveResult:
        """Run the barrier loop and normalize the result.

        The barrier loop's feasibility (Phase I) mode is determined by the objective's
        feasibility threshold: when it is set, the loop stops as soon as ``is_feasible``
        holds, so the adapter serves as a feasibility solver; otherwise it solves to the
        duality-gap tolerance, serving as an optimization solver.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64] | None
            Strictly feasible starting iterate (see Notes); ``None`` defers seeding to
            the shared loop.
         **kwargs : Any
            Extra keyword arguments forwarded to the shared barrier loop (e.g.
            ``on_infeasible``).

        Returns
        -------
         res : SolveResult
            The barrier result, normalized to the barrier-style shape: a full barrier
            result on every path except the pre-loop feasibility early-exit, which is
            wrapped into a zero-iteration feasible result.

        Raises
        ------
         ProblemCertifiablyInfeasibleError
            Propagated from the barrier loop when the dual certifies infeasibility.
         ProblemMarginallyFeasibleError
            Propagated from the barrier loop when a feasibility solve ends without a
            strictly feasible point or a certificate of infeasibility.
         InteriorPointMethodError
            Propagated from the barrier loop when it cannot reach the duality-gap
            tolerance.

        Notes
        -----
        This method requires a strictly feasible x0. In practice ``BarrierSolver.solve``
        finds one before constructing the adapter.

        """
        res = BaseInteriorPointMethodSolver.solve(self, x0, **kwargs)
        # The shared loop returns a full barrier result on every path except the
        # pre-loop feasibility early-exit (feasibility mode, x0 already feasible), where
        # it returns the nested phase-1 solver's bare feasibility result with no
        # objective. Wrap that into a zero-iteration feasible barrier result so callers
        # always get the documented shape. The branch below is reached on that path; it
        # is not dead code.
        if res.objective is not None:
            return res
        x = np.asarray(res.solution)
        return replace(
            _barrier_solveresult(
                solution=x,
                objective_value=self.evaluate_objective(x),
                dual_value=float("-inf"),
                equality_multipliers=np.zeros(self.num_eq_constraints),
                inequality_multipliers=np.zeros(self.num_ineq_constraints),
                suboptimality=float("inf"),
                duality_gaps=[],
                nits=0,
                inner_nits=[],
                inner_suboptimalities=[],
                inner_btls_steps=[],
                inner_quadratic_from=[],
                status=2,
                message="Starting iterate is already feasible; no centering needed.",
                phase1=res,
            ),
            status=SolveStatus.FEASIBLE,
        )

    def hessian_multiply(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the barrier-weighted Hessian H_ft.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the barrier-weighted Hessian is assembled.
         t : float
            Current barrier parameter.
         y : npt.NDArray[np.float64]
            Vector to multiply by the Hessian.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product H_ft y, computed through the structured system assembled by
            ``Problem.build_hessian_system`` without forming H_ft.

        """
        lmbda = -1.0 / self.constraints(x)
        system = self._problem.build_hessian_system(x, self._point(x), t, lmbda)
        return system.multiply(y)

    def btls_keep_feasible(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
    ) -> float:
        r"""Find the largest step keeping every constraint layer strictly feasible.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current iterate (assumed strictly feasible).
         delta_x : npt.NDArray[np.float64]
            Proposed step direction, Δx.

        Returns
        -------
         s : float
            The largest s such that x + s Δx stays strictly feasible across every layer;
            inf when there are no inequality layers.

        """
        return min(
            (
                layer.btls_max_step(x, delta_x, line_search=self.settings.line_search)
                for layer in self._problem.ineq
            ),
            default=np.inf,
        )

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        r"""Initialize the IPM barrier parameter.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Strictly feasible starting iterate at which gradients are evaluated.

        Returns
        -------
         t0 : float
            The initial barrier parameter.

        Raises
        ------
         ValueError
            When ``barrier_init`` is ``EQUALITY_SCHUR`` but the problem has no equality
            constraints.

        Notes
        -----
        There are various initialization strategies controlled by
        ``outer.barrier_init``, with a default strategy for problems with and without
        equality constraints. See the docstring in ``BarrierInitStrategy`` for details.

        For every adaptive strategy, when the problem is a direct-form Phase I (its
        objective is a ``DirectConstraintLayer`` exposing a ``barrier_parameter_hint``),
        the resulting t0 is raised to that hint. ``UNIT`` is the literal escape hatch
        and is left unraised.

        """
        barrier_init = self.settings.outer.barrier_init
        equality = self._equality
        default_strategy = (
            BarrierInitStrategy.EQUALITY_SCHUR
            if equality is not None
            else BarrierInitStrategy.GRADIENT_BALANCE
        )
        strategy = barrier_init if barrier_init is not None else default_strategy
        if strategy is BarrierInitStrategy.UNIT:
            return super().initialize_barrier_parameter(x0)
        if strategy is BarrierInitStrategy.EQUALITY_SCHUR and equality is None:
            raise ValueError(
                "barrier_init=EQUALITY_SCHUR requires equality constraints, but "
                "the Problem has none; use UNIT or GRADIENT_BALANCE, or attach "
                "an equality layer (eq=...) to the Problem."
            )
        if strategy is BarrierInitStrategy.OBJECTIVE_VALUE_GAP:
            t0 = _objective_value_gap_t0(
                float(self.evaluate_objective(x0)),
                self.settings.outer.objective_value_estimate,
                self.num_ineq_constraints,
                self.settings.outer.barrier_multiplier,
                self.settings.outer.tolerance,
            )
            return _raise_to_objective_hint(
                t0,
                self._problem,
                x0,
                self.num_ineq_constraints,
                self.settings.outer.tolerance,
            )
        grad_f0 = self.gradient(x0)
        constraints_x0 = self.constraints(x0)
        # grad_phi = -sum_i (1 / c_i) * grad_c_i = -(grad_c)^T * (1 / c).
        grad_phi = -self.grad_constraints_transpose_multiply(x0, 1.0 / constraints_x0)
        if strategy is BarrierInitStrategy.GRADIENT_BALANCE:
            t0 = _gradient_balance_t0(
                grad_f0, grad_phi, self.settings.outer.min_initial_barrier_parameter
            )
        else:
            # EQUALITY_SCHUR with an equality collaborator: fold A into the gradient
            # balance via the EqualitySolver's Schur initializer. EQUALITY_SCHUR is
            # reachable only with an equality layer (the no-equality strategy rejects it
            # above), so this is non-None.
            assert equality is not None
            t0 = equality.initialize_barrier_parameter(
                gradient=grad_f0,
                grad_phi=grad_phi,
                num_ineq=self.num_ineq_constraints,
                outer_tol=self.settings.outer.tolerance,
                super_init=super().initialize_barrier_parameter(x0),
            )
        # A direct-form Phase I hoists its active layer to the objective; honor its
        # barrier_parameter_hint as a lower bound.
        return _raise_to_objective_hint(
            t0,
            self._problem,
            x0,
            self.num_ineq_constraints,
            self.settings.outer.tolerance,
        )

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Compute the Newton step, with or without equality constraints.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current iterate.
         t : float
            Current barrier parameter, weighting the objective against the barrier.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The primal Newton step.
         nu : npt.NDArray[np.float64]
            Lagrange multipliers for the equality constraints (empty when there are
            none).

        Raises
        ------
         NewtonStepError
            Propagated from the structured solve when the assembled Hessian (or KKT
            system) is not positive definite, is singular, or yields a non-finite step.

        Notes
        -----
        Solves either the Hessian system (no equality constraints),
           H_ft Δx = -∇ft,
        or the equality-constrained KKT system,
            _         _   _      _     _        _
           | H_ft  A^T | |   Δx   |   |   -∇ft   |
           |_ A     0 _| |_  ν   _| = |_    0   _|
        Both branches use structured assembly: every component declares its Hessian as a
        ``MatrixSystem``, and the Problem assembles the weighted Hessian via
        ``build_hessian_system`` at the central-path barrier weights λ = -1/f (stacked
        in the same layer order as ``self.constraints``).

        With no equality layer, the KKT system collapses to a plain Hessian solve and
        the returned ``nu`` is empty. With one, the weighted Hessian is fed into the
        equality KKT saddle via the Problem equality's ``saddle_system`` seam. The
        iterate is already equality-feasible, so the saddle's lower right-hand side is
        zero (the default ``b_bottom=None``).

        """
        lmbda = -1.0 / self.constraints(x)
        system = self._problem.build_hessian_system(x, self._point(x), t, lmbda)
        if self._equality is None:
            delta_x = system.solve(-self.gradient_barrier(x, t))
            # No equality constraints => empty nu vector.
            return delta_x, np.zeros(0)
        assert self._problem.eq is not None
        return self._problem.eq.saddle_system(system).solve(
            -self.gradient_barrier(x, t)
        )

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        r"""Evaluate the Lagrangian dual via a warm-started mini-Newton.

        Computes a lower bound on the dual function
           g(λ, ν) = inf_x L(x, λ, ν),
        starting the inner minimization from the centering iterate x_star.

        Parameters
        ----------
         lmbda : npt.NDArray[np.float64]
            Inequality multipliers, λ (the central-path values -1/f).
         nu : npt.NDArray[np.float64]
            Equality multipliers, ν; ignored when the problem has no equality layer.
         x_star : npt.NDArray[np.float64]
            Centering iterate to warm-start the mini-Newton from.

        Returns
        -------
         g : float
            A lower bound on the dual function value, or -inf when an all-linear
            Lagrangian has a nonzero gradient residual, or when the corrected bound
            cannot be certified (the mini-Newton lands outside the self-concordant
            quadratic region, or δ^2 is unavailable).

        Notes
        -----
        The Lagrangian is assembled from the same structured components as the barrier
        Hessian, so its minimization is an unconstrained Newton problem. When the
        multipliers are close to optimal, x_star is a good starting point, and typically
        only a few Newton steps are needed, so in practice this method is fast.

        The mini-Newton (``_minimize_lagrangian``) returns the minimizing point, x_min,
        and the squared Newton decrement
           δ^2 = ∇L^T (∇^2 L)^{-1} ∇L.
        We return the corrected bound L(x_min, λ, ν) - δ^2. For self-concordant
        functions, this correction is a guaranteed lower bound on inf_x L only inside
        the region of quadratic convergence, where δ < 0.68 (Boyd and Vandenberghe,
        2004, Eq. 9.50). The region guard returns -inf when δ^2 exceeds
        ``_DUAL_REGION_DECREMENT_SQ_MAX`` (out of region) or when δ^2 is unavailable.
        This conservative deferral never falsely certifies convergence or infeasibility;
        a later, better-warm-started outer iteration lands in-region and reports the
        finite certificate. For quadratic L, the mini-Newton lands at the exact
        minimizer and δ^2 = 0, so the correction is a no-op at convergence.

        When the problem is linear, the dual has a simple analytic form. We detect the
        all-linear case and use the analytic form in that case.

        When the mini-Newton cannot take a step (indefinite H_L, non-finite step, and so
        on) it returns ``(None, None)``; we then fall back to L(x_star, λ, ν) with a
        standalone ``_newton_decrement_sq`` correction at x_star, still subject to the
        region guard. ``_minimize_lagrangian`` can instead return ``(x, None)`` when
        ``max_iter`` is exhausted and the final decrement solve fails; there x_min is
        not None, so the x_star fallback is skipped and the region guard converts the
        missing δ^2 to -inf.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        atnu = self.A.T @ nu if self._equality is not None else None
        dual_nu = nu if self._equality is not None else None
        if self._all_linear():
            r = self._lagrangian_gradient(x_star, lmbda, atnu)
            # Use a loose atol=1e-3: the barrier central-path multipliers are only
            # ~1e-3-stationary, so a 1e-10 gate returns -inf at every iterate and never
            # certifies an infeasible all-linear (slack-form) Phase I. Accepting the
            # near-stationary multipliers reports the finite certificate.
            if not np.allclose(r, 0.0, atol=1e-3):
                return float("-inf")
            return self._evaluate_lagrangian(x_star, lmbda, dual_nu)
        x_min, delta2 = self._minimize_lagrangian(x_star, lmbda, atnu=atnu, nu=dual_nu)
        if x_min is None:
            # mini-Newton could not even take a step; fall back to x_star + a
            # standalone decrement.
            x_min = x_star
            delta2 = self._newton_decrement_sq(x_min, lmbda, atnu=atnu)
        # Region guard: the bound L - inf L <= delta^2 only holds in the
        # self-concordant quadratic region (B&V 9.50, delta <= ~0.68). Outside it
        # (or if delta^2 is unavailable) we cannot certify a valid lower bound, so
        # return -inf conservatively rather than a possibly-invalid bound. -inf
        # simply defers certification to a later, better-warm-started outer
        # iteration; it never falsely certifies convergence or infeasibility.
        if delta2 is None or delta2 > _DUAL_REGION_DECREMENT_SQ_MAX:
            return float("-inf")
        return self._evaluate_lagrangian(x_min, lmbda, dual_nu) - delta2

    def _all_linear(self) -> bool:
        r"""Determine whether every term contributing to L is affine in x.

        Drives the analytical LP-dual dispatch in ``evaluate_dual``: for all-linear L,
        inf_x L = -inf if the gradient residual is nonzero (any nonzero coefficient on x
        drives L to -inf along that direction), else L(x) is the exact dual at every x.
        Sidesteps the mini-Newton path and its Woodbury-anchor degeneracy in the LP
        case.

        Returns
        -------
         linear : bool
            True when the objective and every inequality layer are linear, otherwise
            False.

        """
        return none_throws(self._problem.obj).is_linear and all(
            layer.is_linear for layer in self._problem.ineq
        )

    def _evaluate_lagrangian(
        self,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64] | None = None,
    ) -> float:
        r"""Evaluate the Lagrangian at the specified primal/dual values.

        Evaluates the Lagrangian directly,
           L(x, λ, ν) = f0 + λ^T f + ν^T (A x - b).
        The ν^T (A x - b) equality term is included only when ``nu`` is supplied (an
        equality problem); it drops out for the inequality-only case.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the Lagrangian is evaluated.
         lmbda : npt.NDArray[np.float64]
            Inequality multipliers, λ.
         nu : npt.NDArray[np.float64] | None
            Equality multipliers, ν; ``None`` omits the equality term.

        Returns
        -------
         L : float
            The Lagrangian value.

        """
        f0 = self._point(x).evaluate()
        result = f0 + np.dot(lmbda, self.constraints(x))
        if nu is not None:
            result += np.dot(nu, self.A @ x - self.b)
        return float(result)

    def _lagrangian_gradient(
        self,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        atnu: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        r"""Compute the Lagrangian gradient with respect to x.

        Computes
           ∇L = ∇f0 + ∑ λi ∇fi + A^T ν.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the gradient is evaluated.
         lmbda : npt.NDArray[np.float64]
            Inequality multipliers, λ.
         atnu : npt.NDArray[np.float64] | None
            Precomputed A^T ν for the equality term; ``None`` omits it.

        Returns
        -------
         dL : npt.NDArray[np.float64]
            The Lagrangian gradient at x.

        """
        g = self.gradient(x) + self.grad_constraints_transpose_multiply(x, lmbda)
        if atnu is not None:
            g = g + atnu
        return g

    def _lagrangian_hessian_solve(
        self,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> Callable[..., npt.NDArray[np.float64]]:
        r"""Build the solve operator for the Lagrangian Hessian H_L at x.

        Assembles the Lagrangian Hessian ∇^2 L = ∇^2 f0 + ∑ λi ∇^2 fi from structured
        curvatures: the objective's ∇^2 f0 (``ObjectivePoint.hessian_contribution``)
        plus each layer's ∑ λi ∇^2 fi (``ConstraintLayer.hessian_curvature``). This is
        the curvature-only Hessian: it omits the -(1/fi) ∇fi ∇fi^T Jacobian-outer term
        that the barrier/primal-dual ``hessian_contribution`` carries. Affine layers
        contribute a ``ZeroSystem`` (no curvature), so a pure-affine problem reduces to
        the objective Hessian. The equality multiplier term is linear in x and
        contributes no curvature.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the Hessian curvatures are assembled.
         lmbda : npt.NDArray[np.float64]
            Inequality multipliers weighting each layer's curvature.

        Returns
        -------
         solve : Callable[..., npt.NDArray[np.float64]]
            The assembled system's ``solve`` method, mapping a right-hand side r to
            H_L^{-1} r.

        """
        n = self._problem.dimension
        system = self._point(x).hessian_contribution()
        offset = 0
        for layer in self._problem.ineq:
            n_c = layer.num_constraints(n)
            system = system.plus(
                layer.hessian_curvature(x, lmbda[offset : offset + n_c])
            )
            offset += n_c
        return system.solve

    def _minimize_lagrangian(
        self,
        x0: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        atnu: npt.NDArray[np.float64] | None = None,
        nu: npt.NDArray[np.float64] | None = None,
        max_iter: int = 5,
        tol: float = 1e-10,
    ) -> tuple[npt.NDArray[np.float64] | None, float | None]:
        r"""Minimize the Lagrangian over x by a warm-started Armijo mini-Newton.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Warm-start iterate (copied; not mutated in place).
         lmbda : npt.NDArray[np.float64]
            Inequality multipliers, λ, defining the Lagrangian.
         atnu : npt.NDArray[np.float64] | None
            Precomputed A^T ν for the equality gradient term; ``None`` omits it.
         nu : npt.NDArray[np.float64] | None
            Equality multipliers, ν, for the Lagrangian value the line search evaluates;
            ``None`` omits the equality term. Must be consistent with ``atnu``.
         max_iter : int
            Maximum number of mini-Newton steps.
         tol : float
            Convergence tolerance on the squared Newton decrement.

        Returns
        -------
         x_min : npt.NDArray[np.float64] | None
            The recovered (approximate) minimizer, or ``None`` on total failure.
         delta2 : float | None
            The squared Newton decrement at ``x_min``, or ``None`` when it could not be
            computed.

        Notes
        -----
        The squared Newton decrement
           δ^2 = ∇L^T (∇^2 L)^{-1} ∇L
        is evaluated at the returned x_min. Convergence is on δ^2 < tol, which is
        affine-invariant, rather than on the gradient norm, which is not. Each Newton
        step
           Δx = -(∇^2 L)^{-1} ∇L
        yields δ^2 = -Δx^T ∇L at no extra cost. A gradient norm below 1e-10
        short-circuits to δ^2 = 0 without a Hessian solve. The step is accepted by an
        Armijo backtracking line search (the same sufficient-decrease rule the barrier
        centering step uses), so each iterate is a guaranteed descent on L, driving the
        iterate into the self-concordant quadratic region where the δ^2 correction is a
        valid dual bound (Boyd and Vandenberghe, 2004, Eq. 9.50).

        Total failure means the Newton step cannot be computed (indefinite H_L, or a
        non-finite step) or the line search cannot take a step. Both return values are
        then None, and the caller falls back to L(x0, λ, ν) with a standalone decrement.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        x = x0.copy()
        alpha = self.settings.armijo.alpha
        beta = self.settings.armijo.beta
        min_step = self.settings.armijo.min_step
        obj = none_throws(self._problem.obj)
        for nit in range(max_iter):
            step_start = time.time()
            g = self._lagrangian_gradient(x, lmbda, atnu)
            grad_norm = float(np.linalg.norm(g))
            # Cheap pre-check: at the centering optimum grad_x L ~= 0 already, so skip
            # the Hessian solve and report delta^2 = 0 (same floor _newton_decrement_sq
            # uses).
            if grad_norm < 1e-10:
                return x, 0.0
            try:
                h_solve = self._lagrangian_hessian_solve(x, lmbda)
                delta_x = h_solve(-g)
            except (NewtonStepError, np.linalg.LinAlgError):
                return None, None
            if not np.all(np.isfinite(delta_x)):
                return None, None

            # delta_x = -H_L^{-1} g, so delta^2 = g^T H_L^{-1} g = -g^T delta_x >= 0.
            # Clamp roundoff negatives to 0.
            delta2 = max(0.0, -float(np.dot(g, delta_x)))
            if delta2 <= tol:
                # delta^2 is evaluated AT x (do not step).
                return x, delta2

            # Armijo backtracking line search on L (mirrors the barrier centering
            # step). ``ArmijoBarrierMerit`` is generic Armijo-on-a-scalar-objective;
            # here the scalar objective is L(., lambda, nu). neg_alpha_grad = alpha *
            # delta^2 (the per-unit-step required decrease) and the directional
            # derivative is grad L . delta_x = -delta^2 < 0 (a descent direction).
            L0 = self._evaluate_lagrangian(x, lmbda, nu)
            merit = ArmijoBarrierMerit(
                evaluate_barrier_objective=lambda xt: self._evaluate_lagrangian(
                    xt, lmbda, nu
                ),
                ft0=L0,
                neg_alpha_grad=alpha * delta2,
                grad_ft_dot_delta_x=-delta2,
            )
            line_search = LineSearch(
                merit=merit,
                max_step_finder=lambda xx, dd: obj.btls_keep_in_domain(xx, dd),
                beta=beta,
                min_step=min_step,
            )
            try:
                s = line_search.find_step(x, delta_x, step_cap=1.0)
            except (
                ConstraintBoundaryError,
                SevereCurvatureError,
                InvalidDescentDirectionError,
            ):
                return None, None

            x = x + s * delta_x
            self.settings.trace.on_dual_newton_step(
                inner_nit=nit + 1,
                elapsed_ms=1000 * (time.time() - step_start),
                gradient_norm=grad_norm,
            )

        # max_iter exhausted: the last x is post-step, so its delta^2 is stale. Compute
        # it AT the final x with one more gradient + Hessian solve.
        return x, self._newton_decrement_sq(x, lmbda, atnu)

    def _newton_decrement_sq(
        self,
        x: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        atnu: npt.NDArray[np.float64] | None = None,
    ) -> float | None:
        r"""Compute the squared Newton decrement of the Lagrangian.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the decrement is evaluated.
         lmbda : npt.NDArray[np.float64]
            Inequality multipliers defining the Lagrangian.
         atnu : npt.NDArray[np.float64] | None
            Precomputed A^T ν for the equality term; ``None`` omits it.

        Returns
        -------
         delta2 : float | None
            The squared Newton decrement (roundoff negatives clamped to 0.0), 0.0 near
            stationarity, or ``None`` when the Hessian solve fails.

        Notes
        -----
        The Newton decrement is
           δ^2(x) = ∇L^T (∇^2 L)^{-1} ∇L.
        (Boyd and Vandenberghe, 2004 use λ^2, but that overloads its use as a Lagrange
        multiplier in this context, so we use δ.) For a strongly convex quadratic L,
           L(x) - inf_x L = (1/2) δ^2(x);
        for self-concordant L in the quadratic-convergence region,
           L(x) - inf_x L <= δ^2(x)
        (Boyd and Vandenberghe, 2004, Eq. 9.50). Returning δ^2 lets the caller subtract
        a correction guaranteed to leave a lower bound on inf_x L (looser by (1/2) δ^2
        for pure-quadratic L, but always valid).

        Short-circuits to 0.0 when ||∇L|| < 1e-10: at that gradient norm δ^2 <= ||∇L||^2
        divided by the minimum eigenvalue of ∇^2 L and is bounded by roundoff. In that
        case, we avoid touching the Hessian solve at all (this matters for a linear
        program with a quadratic constraint, where ∇^2 L is analytically positive
        definite but the Woodbury anchor is degenerate). Returns ``None`` when the
        Hessian solve fails away from the optimum; ``evaluate_dual`` then returns -inf,
        deferring certification to a later outer iteration.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        g = self._lagrangian_gradient(x, lmbda, atnu)
        if float(np.linalg.norm(g)) < 1e-10:
            return 0.0
        try:
            h_solve = self._lagrangian_hessian_solve(x, lmbda)
            z = h_solve(g)
        except (NewtonStepError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(z)):
            return None
        return max(0.0, float(np.dot(g, z)))

    def check_for_infeasibility(self, result: SolveResult) -> None:
        """Raise if the result certifies that the problem is infeasible.

        Called only in feasibility (Phase I) mode, where the objective is the
        feasibility residual. The dual value is a lower bound on the smallest attainable
        residual, so a strictly positive dual value proves that no feasible point
        exists.

        Parameters
        ----------
         result : SolveResult
            The barrier result whose dual value may certify infeasibility.

        Raises
        ------
         ProblemCertifiablyInfeasibleError
            When the dual value of ``result`` is strictly positive.

        """
        if self._is_infeasible_certified(result):
            raise ProblemCertifiablyInfeasibleError(
                message=(
                    "Problem certifiably infeasible: dual value was "
                    f"{result.dual_value} > 0.0"
                ),
                result=result,
            )


class BarrierSolver(Solver):
    r"""Interior-point (barrier) method for inequality-constrained problems.

    Solves
       minimize    f0(x)
       subject to  A x = b     (when problem.eq is set)
                   fi(x) <= 0  for each layer, i,
    auto-dispatching on ``problem.eq``: a ``Problem`` with no equality layer is solved
    by the plain barrier method, and a ``Problem`` carrying A x = b by the
    equality-constrained barrier method.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm-level settings (tolerances, iteration caps, line-search and
        barrier-init parameters, etc.) inherited from ``Solver``. When ``None``, the
        solver uses its default settings.

    Notes
    -----
    The solve proceeds in two stages:
    1. Establish strict feasibility: if ``x0`` is not strictly feasible, call
       ``problem.make_phase1(x0)`` and recursively solve the Phase I ``Problem`` with
       another ``BarrierSolver``. For an equality problem a supplied ``x0`` is first
       projected onto the equality manifold A x = b.
    2. Run the standard barrier method (centering steps with increasing barrier
       parameter t) over the original ``Problem``. With equality constraints, each
       Newton step solves the KKT saddle
            _         _   _      _     _        _
           | H_ft  A^T | |   Δx   |   |   -∇ft   |
           |_ A     0 _| |_  ν   _| = |_    0   _|
       via the Problem equality's ``saddle_system`` seam.

    The Phase I recursion bottoms out because slack-form Phase I produces an augmented
    ``Problem`` whose constructed starting point is strictly feasible by design
    (verified at construction time inside ``Problem.make_phase1``).

    Internally delegates to ``InteriorPointMethodSolver`` via a private adapter that
    composes the Hessian solve from the objective's and each layer's structured
    ``hessian_contribution``.

    """

    def solve(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Solve ``problem`` via the barrier method, auto-dispatching on equality.

        Parameters
        ----------
         problem : Problem
            The optimization problem; must carry at least one inequality layer
            (otherwise the barrier method is inapplicable).
         x0 : npt.NDArray[np.float64] | None
            Optional starting iterate. May be ``None`` or inequality-infeasible; for an
            equality problem an off-manifold ``x0`` is projected first.
         on_infeasible : OnInfeasible
            Policy when the problem is detected infeasible (e.g. ``"raise"`` or
            ``"continue"``), forwarded to the feasibility solve and barrier loop.

        Returns
        -------
         result : SolveResult
            The solve result: an optimization result, or a feasibility result when the
            problem has no objective.

        Raises
        ------
         ValueError
            When ``problem.ineq`` is empty (use ``NewtonSolver`` instead).
         ProblemInfeasibleError
            Propagated from the equality projection when b is not in the range of A.
         ProblemCertifiablyInfeasibleError
            Propagated from the feasibility solve or barrier loop when the problem is
            certifiably infeasible and ``on_infeasible="raise"``.
         ProblemMarginallyFeasibleError
            Propagated from the feasibility solve when Phase I breaks down without a
            certificate and ``on_infeasible="raise"``.
         InteriorPointMethodError
            Propagated from the barrier loop when it cannot reach the duality-gap
            tolerance.

        Notes
        -----
        ``x0`` may be omitted (``None``) or inequality-infeasible; either way
        ``_solve_feasibility`` constructs a strictly feasible starting iterate from the
        problem's own constraint layers (ignoring the objective). For an equality
        problem a supplied ``x0`` need not lie on the equality manifold. It is projected
        onto it (least-norm toward ``x0``) before the solve, since the feasible-start
        barrier maintains A x = b but never corrects an off-manifold iterate.

        The barrier loop drives to the duality-gap tolerance (an optimization solve)
        unless the objective sets a feasibility threshold, in which case it returns as
        soon as a strictly feasible point is found (objective < 0), so the same solver
        doubles as a feasibility solver, e.g. for a Phase I problem, where iterating to
        the optimum is both wasteful and (for the slack form) numerically degenerate as
        the slack is driven to its minimum.

        """
        if not problem.ineq:
            raise ValueError(
                "BarrierSolver requires at least one inequality constraint layer; "
                "use NewtonSolver for unconstrained problems."
            )

        # The feasible-start barrier never corrects an off-manifold iterate (its KKT
        # step carries a zero equality RHS), so project a supplied ``x0`` onto the
        # equality manifold first, consistently for both the feasibility (obj=None) and
        # objective paths, so neither needlessly re-runs Phase I on an x0 that was
        # merely off-manifold. ``EqualitySolver.solve`` returns the least-norm point
        # toward ``x0`` (a no-op when ``x0`` is already on the manifold); an omitted
        # ``x0`` is seeded and projected inside ``_solve_feasibility``. Build the
        # equality collaborator once (sharing problem.eq's cached SVD) and reuse it for
        # the projection here and the adapter's KKT solves.
        equality = (
            EqualitySolver(constraint=problem.eq, settings=self.settings)
            if problem.eq is not None
            else None
        )
        if equality is not None and x0 is not None:
            x0 = np.asarray(equality.solve(x0=x0).solution)

        # A feasibility problem (no objective) is solved directly: find any point
        # satisfying the constraints. _solve_feasibility runs Phase I and returns a rich
        # SolveResult (carrying an objective), except when x0 is already strictly
        # feasible, where it returns a bare feasibility SolveResult (no objective) that
        # the fallback below lifts.
        if problem.obj is None:
            result = self._solve_feasibility(problem, x0, on_infeasible=on_infeasible)
            if result.objective is not None:
                return result
            return self._feasible_result(problem, np.asarray(result.solution))

        x0_feasible = self._solve_feasibility(
            replace(problem, obj=None), x0, on_infeasible=on_infeasible
        ).solution
        adapter = _BarrierAdapter(
            problem=problem, settings=self.settings, equality=equality
        )
        return adapter.solve(x0=np.asarray(x0_feasible), on_infeasible=on_infeasible)
