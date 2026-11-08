"""Pluggable sinks for the structured events that Cvxium solvers emit.

A ``Trace`` receives the events a solver reports while it runs: the equality solver's
feasible-point computation, the barrier method's Phase I outcome, centering steps,
Newton steps, dual evaluations and line searches, the two-block saddle solver's Newton
steps, and the iterations of the homogeneous self-dual and primal-dual loops. Each event
is one ``on_*`` method, a no-op in the base class, so a subclass overrides only the
events it cares about. Solvers read the trace from ``OptimizationSettings.trace``.

Three implementations are provided:
- ``LoggingTrace``, the default, writes each event as a log record on the
  ``cvxium.optimization``, ``cvxium.linear_programs``, ``cvxium.hsd_solver`` or
  ``cvxium.pd_solver`` logger, with structured ``cvx_`` fields that
  ``IterationHandler`` collects.
- ``ListTrace`` collects every event as a dict in memory, with no logging
  configuration.
- ``DataFrameTrace`` adds conversion of those dicts to a pandas DataFrame.

Two ``want_*`` methods let a trace tell the solver whether to compute diagnostics that
cost extra work and that only a consumer of the corresponding event needs.

"""

import logging
import time
from typing import Any

_optimization_logger: logging.Logger = logging.getLogger("cvxium.optimization")
_linear_programs_logger: logging.Logger = logging.getLogger("cvxium.linear_programs")
_hsd_solver_logger: logging.Logger = logging.getLogger("cvxium.hsd_solver")
_pd_solver_logger: logging.Logger = logging.getLogger("cvxium.pd_solver")


class Trace:
    """Base class for solver event traces.

    Each ``on_*`` method reports one solver event and does nothing by default; subclass
    and override the events to collect. The two ``want_*`` methods return False by
    default, so the solver skips the extra work behind ``on_btls`` and the ``rt_new``
    field of ``on_pd_iteration``.

    ``LoggingTrace``, the default value of ``OptimizationSettings.trace``, writes each
    event as a structured record on the appropriate ``cvxium.*`` logger, following the
    logging contract described in Cvxium's USAGE.md (section 7).

    """

    def on_feasible_initial_guess(self) -> None:
        """Report that the initial guess already satisfies the equality constraints.

        Emitted by ``EqualitySolver`` when the supplied x0 satisfies A x = b to within
        1e-10 in every component, so it is returned unchanged.

        """

    def on_svd_done(self, elapsed_ms: float) -> None:
        """Report the time taken to compute an equality-feasible point.

        Emitted by ``EqualitySolver`` after it computes a point satisfying A x = b. For
        a dense A the time covers the SVD of A and the projection that uses it; for a
        sparse A, which has no SVD path, the event is reused for the saddle-point
        projection.

        Parameters
        ----------
         elapsed_ms : float
            Wall-clock time of the computation, in milliseconds: the SVD of A and the
            projection that follows it, or, for a sparse A, the saddle-point projection,
            which forms no SVD.

        """

    def on_phase1_feasible(self, solver: str) -> None:
        """Report that Phase I returned a point already feasible for this solver.

        Emitted by a barrier solver running in feasibility mode when the point from its
        nested Phase I solver already passes its own feasibility test, so its
        interior-point loop is skipped.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """

    def on_ipm_start(self, solver: str) -> None:
        """Report that the outer barrier loop is about to begin.

        Emitted after Phase I, by the barrier method and by ``StationaritySolver``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """

    def on_centering_start(self, outer_nit: int, t: float) -> None:
        """Report that a centering step is starting.

        Emitted by the barrier method and by ``StationaritySolver``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         t : float
            Barrier parameter for this centering step.

        """

    def on_centering_done(
        self, outer_nit: int, elapsed_ms: float, duality_gap: float
    ) -> None:
        """Report that a centering step has finished.

        Emitted by the barrier method and by ``StationaritySolver``, which report
        different gaps (see ``duality_gap``). The barrier method's per-step gap is not
        monotone, and it inflates when t is large even while the best-bounds certificate
        holds; the solve's convergence test and its recorded duality gaps use that
        certificate instead.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the centering step, in milliseconds.
         duality_gap : float
            Duality gap at the centered iterate. The barrier method reports the per-step
            gap objective - dual at the returned iterate; ``StationaritySolver`` reports
            the surrogate gap n_b / t, where n_b counts its barrier terms (0.0 when
            there are none).

        """

    def on_dual_eval_start(self, outer_nit: int) -> None:
        """Report that the per-centering Lagrange dual evaluation is starting.

        The barrier method evaluates the dual once per centering step, itself a small
        Newton minimization of the Lagrangian (see ``on_dual_newton_step``); this event
        and ``on_dual_eval_done`` bracket it, so its cost can be compared with the
        Newton and line-search timings.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.

        """

    def on_dual_eval_done(self, outer_nit: int, elapsed_ms: float) -> None:
        """Report that the per-centering Lagrange dual evaluation has finished.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the dual evaluation, in milliseconds.

        """

    def on_early_stop(self, outer_nit: int) -> None:
        """Report that a solve is stopping before its outer loop runs to completion.

        Emitted in three situations: a barrier Phase I solve finds a strictly feasible
        point; a primal-dual solver minimizing an objective with a feasibility threshold
        sees the value cross it; or a primal-dual solver's stall detector stops and
        returns the best iterate seen. Distinct from ``on_converged``.

        Parameters
        ----------
         outer_nit : int
            Iteration at which the solve stopped, 1-indexed: the outer (centering)
            iteration of a barrier solve, or the iteration count of a primal-dual solve.

        """

    def on_converged(self, outer_nit: int, duality_gap: float) -> None:
        """Report that the outer loop met its gap tolerance and is stopping.

        In the barrier method (optimization only, not Phase I), emitted when the
        best-bounds certificate, the best primal objective minus the best dual value
        seen so far, is within ``outer.tolerance`` before the planned number of
        centering steps is exhausted. Stopping there keeps t out of the ill-conditioned
        large-t regime. ``StationaritySolver`` emits it when its surrogate gap meets its
        gap tolerance.

        Parameters
        ----------
         outer_nit : int
            Outer iteration at which the tolerance was met, 1-indexed.
         duality_gap : float
            The gap that met the tolerance: the best-bounds certificate in the barrier
            method, the surrogate gap in ``StationaritySolver``.

        """

    def on_ipm_done(self, solver: str, elapsed_ms: float) -> None:
        """Report that the outer barrier loop has finished.

        Emitted by the barrier method and by ``StationaritySolver``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         elapsed_ms : float
            Wall-clock time of the outer loop, in milliseconds.

        """

    def on_newton_step(
        self,
        outer_nit: int,
        inner_nit: int,
        elapsed_ms: float,
        suboptimality: float,
        lambda_norm: float,
        quadratic_convergence_threshold: bool,
    ) -> None:
        """Report a completed Newton step of a barrier centering step.

        Emitted once per inner iteration, after the Newton step is computed and before
        the line search.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Newton iteration within the centering step, 1-indexed.
         elapsed_ms : float
            Wall-clock time to compute the Newton step, in milliseconds.
         suboptimality : float
            Estimated suboptimality λ^2 / 2 of the iterate for the centering problem,
            where λ is the Newton decrement.
         lambda_norm : float
            The Newton decrement λ at the start of the step.
         quadratic_convergence_threshold : bool
            True only on the first inner step of the centering step whose Newton
            decrement satisfies λ ≤ η, where η = (1 - 2α) / 4 and α is the Armijo
            parameter: the step at which Newton's method enters its quadratically
            convergent phase.

        """

    def on_saddle_newton_step(
        self,
        outer_nit: int,
        inner_nit: int,
        elapsed_ms: float,
        residual_norm: float,
        lambda_norm: float,
    ) -> None:
        """Report a completed Newton step of a two-block saddle centering step.

        ``StationaritySolver`` centers a coupled two-block saddle problem with an
        infeasible-start Newton step driven by the residual norm: a cap that keeps the
        iterate in the domain, then an Armijo backtrack on ‖r‖. It computes no Newton
        decrement, so this is a separate event from ``on_newton_step`` rather than a
        reuse with an unused field. Emitted once per accepted step.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Accepted inner step within the centering, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the step, including its line search, in milliseconds.
         residual_norm : float
            ‖r(x, y, t, λ)‖_2, the norm of the residual the step drives toward zero,
            measured at the start of the step.
         lambda_norm : float
            ‖λ‖_2, the norm of the equality multiplier after the step; 0.0 when the
            problem has no ``x_equality``.

        """

    def on_dual_newton_step(
        self,
        inner_nit: int,
        elapsed_ms: float,
        gradient_norm: float,
    ) -> None:
        """Report a completed Newton step of a Lagrange dual minimization.

        Evaluating the dual means minimizing the Lagrangian over x, which the barrier
        method does with a few damped Newton steps. During the per-centering dual
        evaluation these steps fall between ``on_dual_eval_start`` and
        ``on_dual_eval_done``; other dual evaluations emit them too, as does
        ``StationaritySolver`` when it minimizes a block Lagrangian to certify a bound
        on the saddle value.

        Parameters
        ----------
         inner_nit : int
            Step within the dual minimization, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the step, including its line search, in milliseconds.
         gradient_norm : float
            ‖∇_x L‖, the norm of the Lagrangian gradient at the start of the step, the
            quantity the minimization drives toward zero.

        """

    def on_btls(
        self,
        outer_nit: int,
        inner_nit: int,
        btls_s: float,
        improvement: float,
        expected_improvement: float,
    ) -> None:
        """Report a completed backtracking line search.

        Called by the barrier method after each line search, but only when
        ``want_btls_details`` returns True, since filling in ``improvement`` costs extra
        barrier-objective evaluations.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Newton iteration within the centering step, 1-indexed.
         btls_s : float
            The accepted step size s.
         improvement : float
            Actual decrease of the barrier objective, f_t(x) - f_t(x + s Δx).
         expected_improvement : float
            Reference decrease α β λ^2 / (1 + λ), where λ is the Newton decrement and α
            and β are the Armijo parameters.

        """

    def on_hsd_start(self, solver: str) -> None:
        """Report that the homogeneous self-dual loop is about to begin.

        Emitted by ``HsdSolver`` before its long-step path-following loop.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """

    def on_hsd_iteration(
        self,
        nit: int,
        mu: float,
        rel_pres: float,
        rel_dres: float,
        tau: float,
        kappa: float,
        elapsed_ms: float,
    ) -> None:
        """Report a completed homogeneous self-dual iteration.

        Emitted by ``HsdSolver`` after each iteration's Newton step. The residual
        metrics are those measured at the top of the iteration, before the step, while
        ``tau`` and ``kappa`` are post-step. An iteration that converges, or whose step
        fails at the cone boundary, ends the loop without emitting this event.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         mu : float
            Complementarity gap μ at the top of the iteration, before its Newton step.
         rel_pres : float
            Relative primal residual ‖r_p‖ / (1 + ‖b‖) at the top of the iteration.
         rel_dres : float
            Relative dual residual ‖r_d‖ / (1 + ‖c‖) at the top of the iteration.
         tau : float
            Homogenizing variable τ after the step.
         kappa : float
            Homogenizing variable κ after the step.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.

        """

    def on_hsd_done(
        self,
        solver: str,
        nits: int,
        status: int,
        elapsed_ms: float,
    ) -> None:
        """Report that the homogeneous self-dual loop has terminated.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         nits : int
            Number of iterations completed, each with one Newton step.
         status : int
            Integer status code: 0 converged to the outer tolerance, 1 reached only the
            soft tolerance, 2 did not converge.
         elapsed_ms : float
            Wall-clock time of the whole loop, in milliseconds.

        """

    def on_pd_start(
        self,
        solver: str,
        num_vars: int | None = None,
        num_ineq: int | None = None,
        tol: float | None = None,
    ) -> None:
        """Report that a primal-dual loop is about to begin.

        ``PrimalDualSolver``'s inequality-only (Mehrotra predictor-corrector) path and
        both paths of ``PrimalDualStationaritySolver`` supply the problem dimensions and
        tolerance; ``PrimalDualSolver``'s equality path does not.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         num_vars : int | None
            Number of decision variables, when supplied.
         num_ineq : int | None
            Number of inequality constraints, when supplied.
         tol : float | None
            Outer convergence tolerance, when supplied.

        """

    def on_pd_iteration(
        self,
        nit: int,
        eta: float,
        dual_norm: float,
        pri_norm: float,
        residual_norm: float,
        t: float,
        step: float,
        elapsed_ms: float,
        sigma: float | None = None,
        rho: float | None = None,
        damping: float = 0.0,
        cent_norm: float | None = None,
        rt_norm: float | None = None,
        dir_ms: float | None = None,
        rt_new: float | None = None,
    ) -> None:
        """Report a completed primal-dual iteration.

        Emitted after each accepted step, with the residual metrics measured at the top
        of the iteration. An iteration that converges or stops breaks before taking a
        step and emits nothing, so the number of these events equals the number of
        accepted steps.

        Which optional fields are populated depends on the path:
        - ``PrimalDualSolver``'s inequality-only (Mehrotra predictor-corrector) path
          supplies all of them, ``rt_new`` only when ``want_pd_step_details`` returns
          True.
        - ``PrimalDualStationaritySolver`` supplies ``sigma`` only.
        - ``PrimalDualSolver``'s equality path supplies none, leaving each at its
          default.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         eta : float
            Surrogate duality gap η = -f^T λ at the top of the iteration.
         dual_norm : float
            ‖r_dual‖_2, the dual (stationarity) residual norm at the top of the
            iteration.
         pri_norm : float
            ‖A x - b‖_2, the equality residual norm; 0.0 when the problem has no
            equality constraints.
         residual_norm : float
            The proxy sqrt(‖r_dual‖^2 + ‖A x - b‖^2 + η^2), kept for the result history.
         t : float
            Surrogate barrier parameter μ m / η, where m counts the inequality
            constraints and μ is the barrier multiplier. The two-block solver omits μ
            and reports 0.0 when η ≤ 0.
         step : float
            Step length actually taken.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.
         sigma : float | None
            Mehrotra centering parameter (in the two-block solver, the mean of its two
            blocks' values).
         rho : float | None
            Ratio of the actual to the predicted residual reduction; ``None`` when the
            step was undamped and the ratio test was skipped.
         damping : float
            Levenberg damping used for the accepted step.
         cent_norm : float | None
            ‖r_cent‖_2, the centrality residual norm.
         rt_norm : float | None
            ‖[r_dual; r_cent]‖_2, the KKT residual at barrier parameter t, before the
            step.
         dir_ms : float | None
            Time to compute the search direction, in milliseconds.
         rt_new : float | None
            The same KKT residual after the step, at the same t; computed only when
            ``want_pd_step_details`` returns True, otherwise ``None``.

        """

    def want_pd_step_details(self) -> bool:
        """Return whether primal-dual solvers should compute the post-step KKT residual.

        Computing ``rt_new`` for ``on_pd_iteration`` costs one extra residual evaluation
        per iteration, so ``PrimalDualSolver`` skips it unless this returns True.
        ``LoggingTrace`` returns True exactly when the ``cvxium.pd_solver`` logger is
        enabled at DEBUG; collectors that want the field, such as ``ListTrace``, return
        True.

        Returns
        -------
         want : bool
            False here.

        """
        return False

    def on_pd_damping_engaged(self, nit: int, damping: float, retries: int) -> None:
        """Report that Levenberg damping had to grow to find a feasible step.

        Emitted by ``PrimalDualSolver``'s inequality-only path, only on iterations where
        the damping had to be grown to recover a feasible step: the diagnostic signature
        of a nearly singular modified Hessian.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed, as in ``on_pd_iteration``.
         damping : float
            Levenberg damping of the accepted step.
         retries : int
            Number of times the damping was grown before a feasible step was found.

        """

    def on_pd_done(
        self,
        solver: str,
        nits: int,
        status: int,
        elapsed_ms: float,
        reason: str = "",
        duality_gap: float | None = None,
    ) -> None:
        """Report that a primal-dual loop has terminated.

        The per-iteration gap is the cheap surrogate η; the true dual is evaluated only
        once, at termination, and only on the path that reports ``duality_gap``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         nits : int
            Number of iterations started, including a final one that stopped without
            taking a step.
         status : int
            Integer status code. 0 means converged; the meaning of 1 and 2 depends on
            the solver. For ``PrimalDualSolver``, 1 is a stall or the iteration cap, and
            2 a feasibility stop or certified infeasibility. For
            ``PrimalDualStationaritySolver``, 1 is the iteration cap or another
            non-converged break, and 2 acceptance at the soft tolerance.
         elapsed_ms : float
            Wall-clock time of the whole loop, in milliseconds.
         reason : str
            Human-readable termination cause; empty when the solver supplies none (as on
            ``PrimalDualSolver``'s equality path).
         duality_gap : float | None
            True duality gap objective - dual, evaluated once at termination. Supplied
            only by ``PrimalDualSolver``'s inequality-only path on a normal exit;
            ``None`` otherwise.

        """

    def want_btls_details(self) -> bool:
        """Return whether barrier solvers should call ``on_btls``.

        Populating ``improvement`` costs two extra ``evaluate_barrier_objective`` calls
        per inner step, so the solver skips them, and the ``on_btls`` call, when this
        returns False. ``LoggingTrace`` returns True exactly when the
        ``cvxium.optimization`` logger is enabled at DEBUG.

        Returns
        -------
         want : bool
            False here.

        """
        return False


class LoggingTrace(Trace):
    """Default trace, writing each event as a structured ``cvxium.*`` log record.

    Each record carries a human-readable message and, in ``extra``, structured fields
    named with a ``cvx_`` prefix (always including ``cvx_event``), the schema that
    ``IterationHandler`` and the logging contract in USAGE.md rely on. The per-step
    events of the barrier and saddle solvers (``on_newton_step``,
    ``on_saddle_newton_step``, ``on_dual_newton_step`` and ``on_btls``) log at DEBUG on
    ``cvxium.optimization``; every other event logs at INFO. The ``want_*`` methods
    return True exactly when the logger that would show the extra detail is enabled at
    DEBUG.

    """

    def on_feasible_initial_guess(self) -> None:
        """Log that the initial guess already satisfies the equality constraints.

        Writes an INFO record on ``cvxium.linear_programs``.

        """
        _linear_programs_logger.info(
            "  Initial guess was feasible.",
            extra={"cvx_event": "feasible_initial_guess"},
        )

    def on_svd_done(self, elapsed_ms: float) -> None:
        """Log the time taken to compute an equality-feasible point.

        Writes an INFO record on ``cvxium.linear_programs``.

        Parameters
        ----------
         elapsed_ms : float
            Wall-clock time of the computation, in milliseconds: the SVD of A and the
            projection that follows it, or, for a sparse A, the saddle-point projection,
            which forms no SVD.

        """
        _linear_programs_logger.info(
            "  SVD calculated in %.3f ms",
            elapsed_ms,
            extra={"cvx_event": "svd_done", "cvx_elapsed_ms": elapsed_ms},
        )

    def on_phase1_feasible(self, solver: str) -> None:
        """Log that Phase I returned a point already feasible for this solver.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """
        _optimization_logger.info(
            "  Phase I solution was feasible so we're done (%s)",
            solver,
            extra={"cvx_event": "phase1_feasible", "cvx_solver": solver},
        )

    def on_ipm_start(self, solver: str) -> None:
        """Log that the outer barrier loop is about to begin.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """
        _optimization_logger.info(
            "  Starting IPM (%s)",
            solver,
            extra={"cvx_event": "ipm_start", "cvx_solver": solver},
        )

    def on_centering_start(self, outer_nit: int, t: float) -> None:
        """Log that a centering step is starting.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         t : float
            Barrier parameter for this centering step.

        """
        _optimization_logger.info(
            "  %02d Beginning centering step with t=%s",
            outer_nit,
            t,
            extra={
                "cvx_event": "centering_start",
                "cvx_outer_nit": outer_nit,
                "cvx_t": t,
            },
        )

    def on_centering_done(
        self, outer_nit: int, elapsed_ms: float, duality_gap: float
    ) -> None:
        """Log that a centering step has finished, with its duality gap.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the centering step, in milliseconds.
         duality_gap : float
            Duality gap at the centered iterate. The barrier method reports the per-step
            gap objective - dual at the returned iterate; ``StationaritySolver`` reports
            the surrogate gap n_b / t, where n_b counts its barrier terms (0.0 when
            there are none).

        """
        _optimization_logger.info(
            "  %02d Centering step completed in %.3f ms (duality gap = %.6g)",
            outer_nit,
            elapsed_ms,
            duality_gap,
            extra={
                "cvx_event": "centering_done",
                "cvx_outer_nit": outer_nit,
                "cvx_elapsed_ms": elapsed_ms,
                "cvx_duality_gap": duality_gap,
            },
        )

    def on_dual_eval_start(self, outer_nit: int) -> None:
        """Log that the per-centering Lagrange dual evaluation is starting.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.

        """
        _optimization_logger.info(
            "  %02d Evaluating Lagrange dual function",
            outer_nit,
            extra={"cvx_event": "dual_eval_start", "cvx_outer_nit": outer_nit},
        )

    def on_dual_eval_done(self, outer_nit: int, elapsed_ms: float) -> None:
        """Log that the per-centering Lagrange dual evaluation has finished.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the dual evaluation, in milliseconds.

        """
        _optimization_logger.info(
            "  %02d Lagrangian dual function evaluated in %.3f ms",
            outer_nit,
            elapsed_ms,
            extra={
                "cvx_event": "dual_eval_done",
                "cvx_outer_nit": outer_nit,
                "cvx_elapsed_ms": elapsed_ms,
            },
        )

    def on_early_stop(self, outer_nit: int) -> None:
        """Log that a solve is stopping before its outer loop runs to completion.

        Writes an INFO record on ``cvxium.optimization``, whichever solver emitted the
        event.

        Parameters
        ----------
         outer_nit : int
            Iteration at which the solve stopped, 1-indexed: the outer (centering)
            iteration of a barrier solve, or the iteration count of a primal-dual solve.

        """
        _optimization_logger.info(
            "  %02d Result was strictly feasible so we're early-stopping.",
            outer_nit,
            extra={"cvx_event": "early_stop", "cvx_outer_nit": outer_nit},
        )

    def on_converged(self, outer_nit: int, duality_gap: float) -> None:
        """Log that the outer loop met its gap tolerance and is stopping.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer iteration at which the tolerance was met, 1-indexed.
         duality_gap : float
            The gap that met the tolerance: the best-bounds certificate in the barrier
            method, the surrogate gap in ``StationaritySolver``.

        """
        _optimization_logger.info(
            "  %02d Best-bounds duality gap %.6g within tolerance; early-stopping.",
            outer_nit,
            duality_gap,
            extra={
                "cvx_event": "converged",
                "cvx_outer_nit": outer_nit,
                "cvx_duality_gap": duality_gap,
            },
        )

    def on_ipm_done(self, solver: str, elapsed_ms: float) -> None:
        """Log that the outer barrier loop has finished.

        Writes an INFO record on ``cvxium.optimization``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         elapsed_ms : float
            Wall-clock time of the outer loop, in milliseconds.

        """
        _optimization_logger.info(
            "  IPM completed in %.3f ms (%s)",
            elapsed_ms,
            solver,
            extra={
                "cvx_event": "ipm_done",
                "cvx_solver": solver,
                "cvx_elapsed_ms": elapsed_ms,
            },
        )

    def on_newton_step(
        self,
        outer_nit: int,
        inner_nit: int,
        elapsed_ms: float,
        suboptimality: float,
        lambda_norm: float,
        quadratic_convergence_threshold: bool,
    ) -> None:
        """Log a completed Newton step of a barrier centering step.

        Writes a DEBUG record on ``cvxium.optimization``. The message shows the
        decrement and flags the quadratic-convergence step; ``lambda_norm`` and the flag
        are not among the structured fields.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Newton iteration within the centering step, 1-indexed.
         elapsed_ms : float
            Wall-clock time to compute the Newton step, in milliseconds.
         suboptimality : float
            Estimated suboptimality λ^2 / 2 of the iterate for the centering problem,
            where λ is the Newton decrement.
         lambda_norm : float
            The Newton decrement λ at the start of the step.
         quadratic_convergence_threshold : bool
            True only on the first inner step of the centering step whose Newton
            decrement satisfies λ ≤ η, where η = (1 - 2α) / 4 and α is the Armijo
            parameter: the step at which Newton's method enters its quadratically
            convergent phase.

        """
        suffix = (
            " (quadratic convergence threshold)"
            if quadratic_convergence_threshold
            else ""
        )
        _optimization_logger.debug(
            "    %02d Newton step in %.3f ms; sub-opt %.6g = 0.5 * %.6g^2%s",
            inner_nit,
            elapsed_ms,
            suboptimality,
            lambda_norm,
            suffix,
            extra={
                "cvx_event": "newton_step",
                "cvx_outer_nit": outer_nit,
                "cvx_inner_nit": inner_nit,
                "cvx_elapsed_ms": elapsed_ms,
                "cvx_suboptimality": suboptimality,
            },
        )

    def on_saddle_newton_step(
        self,
        outer_nit: int,
        inner_nit: int,
        elapsed_ms: float,
        residual_norm: float,
        lambda_norm: float,
    ) -> None:
        """Log a completed Newton step of a two-block saddle centering step.

        Writes a DEBUG record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Accepted inner step within the centering, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the step, including its line search, in milliseconds.
         residual_norm : float
            ‖r(x, y, t, λ)‖_2, the norm of the residual the step drives toward zero,
            measured at the start of the step.
         lambda_norm : float
            ‖λ‖_2, the norm of the equality multiplier after the step; 0.0 when the
            problem has no ``x_equality``.

        """
        _optimization_logger.debug(
            "    %02d Saddle Newton step in %.3f ms; ||r||=%.6g, ||lambda||=%.6g",
            inner_nit,
            elapsed_ms,
            residual_norm,
            lambda_norm,
            extra={
                "cvx_event": "saddle_newton_step",
                "cvx_outer_nit": outer_nit,
                "cvx_inner_nit": inner_nit,
                "cvx_elapsed_ms": elapsed_ms,
                "cvx_residual_norm": residual_norm,
                "cvx_lambda_norm": lambda_norm,
            },
        )

    def on_dual_newton_step(
        self,
        inner_nit: int,
        elapsed_ms: float,
        gradient_norm: float,
    ) -> None:
        """Log a completed Newton step of a Lagrange dual minimization.

        Writes a DEBUG record on ``cvxium.optimization``.

        Parameters
        ----------
         inner_nit : int
            Step within the dual minimization, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the step, including its line search, in milliseconds.
         gradient_norm : float
            ‖∇_x L‖, the norm of the Lagrangian gradient at the start of the step, the
            quantity the minimization drives toward zero.

        """
        _optimization_logger.debug(
            "      %02d Dual Newton step in %.3f ms; ||grad L||=%.6g",
            inner_nit,
            elapsed_ms,
            gradient_norm,
            extra={
                "cvx_event": "dual_newton_step",
                "cvx_inner_nit": inner_nit,
                "cvx_elapsed_ms": elapsed_ms,
                "cvx_gradient_norm": gradient_norm,
            },
        )

    def on_btls(
        self,
        outer_nit: int,
        inner_nit: int,
        btls_s: float,
        improvement: float,
        expected_improvement: float,
    ) -> None:
        """Log a completed backtracking line search.

        Writes a DEBUG record on ``cvxium.optimization``.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Newton iteration within the centering step, 1-indexed.
         btls_s : float
            The accepted step size s.
         improvement : float
            Actual decrease of the barrier objective, f_t(x) - f_t(x + s Δx).
         expected_improvement : float
            Reference decrease α β λ^2 / (1 + λ), where λ is the Newton decrement and α
            and β are the Armijo parameters.

        """
        _optimization_logger.debug(
            "    %02d btls_s=%.4f, improvement=%.6g, expected improvement=%.6g",
            inner_nit,
            btls_s,
            improvement,
            expected_improvement,
            extra={
                "cvx_event": "btls",
                "cvx_outer_nit": outer_nit,
                "cvx_inner_nit": inner_nit,
                "cvx_btls_s": btls_s,
                "cvx_improvement": improvement,
                "cvx_expected_improvement": expected_improvement,
            },
        )

    def on_hsd_start(self, solver: str) -> None:
        """Log that the homogeneous self-dual loop is about to begin.

        Writes an INFO record on ``cvxium.hsd_solver``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """
        _hsd_solver_logger.info(
            "  Starting HSD (%s)",
            solver,
            extra={"cvx_event": "hsd_start", "cvx_solver": solver},
        )

    def on_hsd_iteration(
        self,
        nit: int,
        mu: float,
        rel_pres: float,
        rel_dres: float,
        tau: float,
        kappa: float,
        elapsed_ms: float,
    ) -> None:
        """Log a completed homogeneous self-dual iteration.

        Writes an INFO record on ``cvxium.hsd_solver``. ``tau`` and ``kappa`` appear
        only in the structured fields, not the message.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         mu : float
            Complementarity gap μ at the top of the iteration, before its Newton step.
         rel_pres : float
            Relative primal residual ‖r_p‖ / (1 + ‖b‖) at the top of the iteration.
         rel_dres : float
            Relative dual residual ‖r_d‖ / (1 + ‖c‖) at the top of the iteration.
         tau : float
            Homogenizing variable τ after the step.
         kappa : float
            Homogenizing variable κ after the step.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.

        """
        _hsd_solver_logger.info(
            "  %02d HSD iter: mu=%.3e, pres=%.2e, dres=%.2e (%.3f ms)",
            nit,
            mu,
            rel_pres,
            rel_dres,
            elapsed_ms,
            extra={
                "cvx_event": "hsd_iteration",
                "cvx_nit": nit,
                "cvx_mu": mu,
                "cvx_rel_pres": rel_pres,
                "cvx_rel_dres": rel_dres,
                "cvx_tau": tau,
                "cvx_kappa": kappa,
                "cvx_elapsed_ms": elapsed_ms,
            },
        )

    def on_hsd_done(
        self,
        solver: str,
        nits: int,
        status: int,
        elapsed_ms: float,
    ) -> None:
        """Log that the homogeneous self-dual loop has terminated.

        Writes an INFO record on ``cvxium.hsd_solver``.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         nits : int
            Number of iterations completed, each with one Newton step.
         status : int
            Integer status code: 0 converged to the outer tolerance, 1 reached only the
            soft tolerance, 2 did not converge.
         elapsed_ms : float
            Wall-clock time of the whole loop, in milliseconds.

        """
        _hsd_solver_logger.info(
            "  HSD completed in %.3f ms after %d iters, status=%d (%s)",
            elapsed_ms,
            nits,
            status,
            solver,
            extra={
                "cvx_event": "hsd_done",
                "cvx_solver": solver,
                "cvx_nits": nits,
                "cvx_status": status,
                "cvx_elapsed_ms": elapsed_ms,
            },
        )

    def on_pd_start(
        self,
        solver: str,
        num_vars: int | None = None,
        num_ineq: int | None = None,
        tol: float | None = None,
    ) -> None:
        """Log that a primal-dual loop is about to begin.

        Writes an INFO record on ``cvxium.pd_solver``; the message shows the dimensions
        and tolerance when they are supplied.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         num_vars : int | None
            Number of decision variables, when supplied.
         num_ineq : int | None
            Number of inequality constraints, when supplied.
         tol : float | None
            Outer convergence tolerance, when supplied.

        """
        ctx = ""
        if num_vars is not None and num_ineq is not None:
            ctx = f" [{num_vars} vars, {num_ineq} ineq]"
        if tol is not None:
            ctx += f" tol={tol:.1e}"
        _pd_solver_logger.info(
            "  Starting PD (%s)%s",
            solver,
            ctx,
            extra={
                "cvx_event": "pd_start",
                "cvx_solver": solver,
                "cvx_num_vars": num_vars,
                "cvx_num_ineq": num_ineq,
                "cvx_tol": tol,
            },
        )

    def on_pd_iteration(
        self,
        nit: int,
        eta: float,
        dual_norm: float,
        pri_norm: float,
        residual_norm: float,
        t: float,
        step: float,
        elapsed_ms: float,
        sigma: float | None = None,
        rho: float | None = None,
        damping: float = 0.0,
        cent_norm: float | None = None,
        rt_norm: float | None = None,
        dir_ms: float | None = None,
        rt_new: float | None = None,
    ) -> None:
        """Log a completed primal-dual iteration in one of three formats.

        Every format writes INFO records on ``cvxium.pd_solver``, and the first record
        of each carries all the arguments as structured fields. The format is chosen by
        what the caller supplied and by the logger's level:
        - No ``rt_norm`` or ``cent_norm``: one compact line
          (``_log_pd_iter_equality``). This covers ``PrimalDualSolver``'s equality path
          and also ``PrimalDualStationaritySolver``, which supplies ``sigma`` but no KKT
          residual breakdown.
        - ``rt_norm`` supplied, logger not enabled at DEBUG: one summary line
          (``_log_pd_iter_mpc_compact``).
        - ``rt_norm`` supplied, logger enabled at DEBUG: a header line and one line per
          quantity (``_log_pd_iter_mpc_verbose``).
        The step length is printed in fixed point when it is at least 1e-3, and in
        scientific notation otherwise.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         eta : float
            Surrogate duality gap η = -f^T λ at the top of the iteration.
         dual_norm : float
            ‖r_dual‖_2, the dual (stationarity) residual norm at the top of the
            iteration.
         pri_norm : float
            ‖A x - b‖_2, the equality residual norm; 0.0 when the problem has no
            equality constraints.
         residual_norm : float
            The proxy sqrt(‖r_dual‖^2 + ‖A x - b‖^2 + η^2), kept for the result history.
         t : float
            Surrogate barrier parameter μ m / η, where m counts the inequality
            constraints and μ is the barrier multiplier. The two-block solver omits μ
            and reports 0.0 when η ≤ 0.
         step : float
            Step length actually taken.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.
         sigma : float | None
            Mehrotra centering parameter (in the two-block solver, the mean of its two
            blocks' values).
         rho : float | None
            Ratio of the actual to the predicted residual reduction; ``None`` when the
            step was undamped and the ratio test was skipped.
         damping : float
            Levenberg damping used for the accepted step.
         cent_norm : float | None
            ‖r_cent‖_2, the centrality residual norm.
         rt_norm : float | None
            ‖[r_dual; r_cent]‖_2, the KKT residual at barrier parameter t, before the
            step.
         dir_ms : float | None
            Time to compute the search direction, in milliseconds.
         rt_new : float | None
            The same KKT residual after the step, at the same t; computed only when
            ``want_pd_step_details`` returns True, otherwise ``None``.

        """
        extra = {
            "cvx_event": "pd_iteration",
            "cvx_outer_nit": nit,
            "cvx_eta": eta,
            "cvx_dual_norm": dual_norm,
            "cvx_pri_norm": pri_norm,
            "cvx_residual_norm": residual_norm,
            "cvx_t": t,
            "cvx_step": step,
            "cvx_elapsed_ms": elapsed_ms,
            "cvx_sigma": sigma,
            "cvx_rho": rho,
            "cvx_damping": damping,
            "cvx_cent_norm": cent_norm,
            "cvx_rt_norm": rt_norm,
            "cvx_dir_ms": dir_ms,
            "cvx_rt_new": rt_new,
        }
        step_str = f"{step:.4f}" if step >= 1e-3 else f"{step:.2e}"
        if rt_norm is None or cent_norm is None:
            # No KKT residual breakdown (the equality path, and the two-block
            # stationarity solver): compact single line.
            self._log_pd_iter_equality(
                nit, eta, dual_norm, pri_norm, step_str, damping, elapsed_ms, extra
            )
        elif not _pd_solver_logger.isEnabledFor(logging.DEBUG):
            # Predictor-corrector at INFO: one scannable summary line.
            self._log_pd_iter_mpc_compact(
                nit, eta, t, rt_norm, step_str, sigma, damping, elapsed_ms, extra
            )
        else:
            # Predictor-corrector at DEBUG: one timestamped line per quantity.
            self._log_pd_iter_mpc_verbose(
                nit,
                eta,
                t,
                dual_norm,
                pri_norm,
                cent_norm,
                rt_norm,
                rt_new,
                dir_ms,
                step_str,
                sigma,
                rho,
                damping,
                elapsed_ms,
                extra,
            )

    @staticmethod
    def _log_pd_iter_equality(
        nit: int,
        eta: float,
        dual_norm: float,
        pri_norm: float,
        step_str: str,
        damping: float,
        elapsed_ms: float,
        extra: dict[str, Any],
    ) -> None:
        """Log a primal-dual iteration as one compact line.

        Used when no KKT residual breakdown is supplied. The line shows the gap, the
        dual residual, the equality residual when it is positive, the step, and the
        damping when it is positive.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         eta : float
            Surrogate duality gap η = -f^T λ at the top of the iteration.
         dual_norm : float
            ‖r_dual‖_2, the dual (stationarity) residual norm at the top of the
            iteration.
         pri_norm : float
            ‖A x - b‖_2, the equality residual norm; 0.0 when the problem has no
            equality constraints.
         step_str : str
            Step length, preformatted by ``on_pd_iteration``.
         damping : float
            Levenberg damping used for the accepted step.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.
         extra : dict[str, Any]
            Structured ``cvx_`` fields attached to the record.

        """
        parts = [f"gap={eta:9.3e}", f"||r_dual||={dual_norm:8.2e}"]
        if pri_norm > 0.0:
            parts.append(f"||r_pri||={pri_norm:8.2e}")
        parts.append(f"step={step_str}")
        if damping > 0.0:
            parts.append(f"damp={damping:8.2e}")
        _pd_solver_logger.info(
            "  %02d PD iter: %s  (%.3f ms)",
            nit,
            "  ".join(parts),
            elapsed_ms,
            extra=extra,
        )

    @staticmethod
    def _log_pd_iter_mpc_compact(
        nit: int,
        eta: float,
        t: float,
        rt_norm: float,
        step_str: str,
        sigma: float | None,
        damping: float,
        elapsed_ms: float,
        extra: dict[str, Any],
    ) -> None:
        """Log a predictor-corrector iteration as one summary line.

        Used for the predictor-corrector path when the logger is not enabled at DEBUG.
        The line shows the gap, t, the KKT residual, the step, σ when supplied, and the
        damping (or off).

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         eta : float
            Surrogate duality gap η = -f^T λ at the top of the iteration.
         t : float
            Surrogate barrier parameter μ m / η, where m counts the inequality
            constraints and μ is the barrier multiplier. The two-block solver omits μ
            and reports 0.0 when η ≤ 0.
         rt_norm : float
            ‖[r_dual; r_cent]‖_2, the KKT residual at barrier parameter t, before the
            step.
         step_str : str
            Step length, preformatted by ``on_pd_iteration``.
         sigma : float | None
            Mehrotra centering parameter (in the two-block solver, the mean of its two
            blocks' values).
         damping : float
            Levenberg damping used for the accepted step.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.
         extra : dict[str, Any]
            Structured ``cvx_`` fields attached to the record.

        """
        seg = [
            f"gap={eta:.3e}",
            f"t={t:.3e}",
            f"||r_t||={rt_norm:.3e}",
            f"step={step_str}",
        ]
        if sigma is not None:
            seg.append(f"sigma={sigma:.2f}")
        seg.append(f"damp={'off' if damping <= 0.0 else f'{damping:.2e}'}")
        _pd_solver_logger.info(
            "  %02d PD iter: %s  (%.3f ms)",
            nit,
            "  ".join(seg),
            elapsed_ms,
            extra=extra,
        )

    @staticmethod
    def _log_pd_iter_mpc_verbose(
        nit: int,
        eta: float,
        t: float,
        dual_norm: float,
        pri_norm: float,
        cent_norm: float,
        rt_norm: float,
        rt_new: float | None,
        dir_ms: float | None,
        step_str: str,
        sigma: float | None,
        rho: float | None,
        damping: float,
        elapsed_ms: float,
        extra: dict[str, Any],
    ) -> None:
        """Log a predictor-corrector iteration as one line per quantity.

        Used for the predictor-corrector path when the logger is enabled at DEBUG. The
        header line carries the full structured record in ``extra``; the detail lines
        that follow carry only ``cvx_event`` (``pd_iteration_detail``) and
        ``cvx_outer_nit``, so a collector counting ``pd_iteration`` events sees one per
        iteration.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         eta : float
            Surrogate duality gap η = -f^T λ at the top of the iteration.
         t : float
            Surrogate barrier parameter μ m / η, where m counts the inequality
            constraints and μ is the barrier multiplier. The two-block solver omits μ
            and reports 0.0 when η ≤ 0.
         dual_norm : float
            ‖r_dual‖_2, the dual (stationarity) residual norm at the top of the
            iteration.
         pri_norm : float
            ‖A x - b‖_2, the equality residual norm; 0.0 when the problem has no
            equality constraints.
         cent_norm : float
            ‖r_cent‖_2, the centrality residual norm.
         rt_norm : float
            ‖[r_dual; r_cent]‖_2, the KKT residual at barrier parameter t, before the
            step.
         rt_new : float | None
            The KKT residual after the step, at the same t; ``None`` omits the
            before-and-after comparison.
         dir_ms : float | None
            Time to compute the search direction, in milliseconds; ``None`` omits that
            line.
         step_str : str
            Step length, preformatted by ``on_pd_iteration``.
         sigma : float | None
            Mehrotra centering parameter (in the two-block solver, the mean of its two
            blocks' values).
         rho : float | None
            Ratio of the actual to the predicted residual reduction; ``None`` when the
            step was undamped and the ratio test was skipped.
         damping : float
            Levenberg damping used for the accepted step.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.
         extra : dict[str, Any]
            Structured ``cvx_`` fields attached to the record.

        """
        detail = {"cvx_event": "pd_iteration_detail", "cvx_outer_nit": nit}
        breakdown = f"dual={dual_norm:.2e} cent={cent_norm:.2e}"
        if pri_norm > 0.0:
            breakdown += f" pri={pri_norm:.2e}"
        step_seg = [f"step={step_str}"]
        if sigma is not None:
            step_seg.append(f"sigma={sigma:.2f}")
        if rt_new is not None:
            factor = rt_new / rt_norm if rt_norm > 0.0 else float("nan")
            step_seg.append(f"||r_t||: {rt_norm:.2e} -> {rt_new:.2e} (x{factor:.3f})")
        if rho is not None:
            step_seg.append(f"rho={rho:+.2f}")
        step_seg.append(f"damp={'off' if damping <= 0.0 else f'{damping:.2e}'}")

        _pd_solver_logger.info("  %02d PD iter:", nit, extra=extra)
        _pd_solver_logger.info(
            "  %02d    surrogate duality gap=%.3e  t=%.3e", nit, eta, t, extra=detail
        )
        _pd_solver_logger.info(
            "  %02d    ||r_t||=%.3e  (%s)", nit, rt_norm, breakdown, extra=detail
        )
        if dir_ms is not None:
            _pd_solver_logger.info(
                "  %02d    PD direction calculated in %.3f ms",
                nit,
                dir_ms,
                extra=detail,
            )
        _pd_solver_logger.info("  %02d    %s", nit, "  ".join(step_seg), extra=detail)
        _pd_solver_logger.info(
            "  %02d    PD iteration complete in %.3f ms", nit, elapsed_ms, extra=detail
        )

    def want_pd_step_details(self) -> bool:
        """Return whether the post-step KKT residual would be logged.

        Returns
        -------
         want : bool
            True exactly when the ``cvxium.pd_solver`` logger is enabled at DEBUG, the
            level at which ``on_pd_iteration`` shows ``rt_new``.

        """
        return _pd_solver_logger.isEnabledFor(logging.DEBUG)

    def on_pd_damping_engaged(self, nit: int, damping: float, retries: int) -> None:
        """Log that Levenberg damping had to grow to find a feasible step.

        Writes an INFO record on ``cvxium.pd_solver``.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed, as in ``on_pd_iteration``.
         damping : float
            Levenberg damping of the accepted step.
         retries : int
            Number of times the damping was grown before a feasible step was found.

        """
        _pd_solver_logger.info(
            "  %02d ! Levenberg damping engaged: %.3e (%d %s) -- near-singular Hessian",
            nit,
            damping,
            retries,
            "retry" if retries == 1 else "retries",
            extra={
                "cvx_event": "pd_damping_engaged",
                "cvx_outer_nit": nit,
                "cvx_damping": damping,
                "cvx_retries": retries,
            },
        )

    def on_pd_done(
        self,
        solver: str,
        nits: int,
        status: int,
        elapsed_ms: float,
        reason: str = "",
        duality_gap: float | None = None,
    ) -> None:
        """Log that a primal-dual loop has terminated.

        Writes an INFO record on ``cvxium.pd_solver``; the message appends the true
        duality gap and the reason when they are supplied.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         nits : int
            Number of iterations started, including a final one that stopped without
            taking a step.
         status : int
            Integer status code. 0 means converged; the meaning of 1 and 2 depends on
            the solver. For ``PrimalDualSolver``, 1 is a stall or the iteration cap, and
            2 a feasibility stop or certified infeasibility. For
            ``PrimalDualStationaritySolver``, 1 is the iteration cap or another
            non-converged break, and 2 acceptance at the soft tolerance.
         elapsed_ms : float
            Wall-clock time of the whole loop, in milliseconds.
         reason : str
            Human-readable termination cause; empty when the solver supplies none (as on
            ``PrimalDualSolver``'s equality path).
         duality_gap : float | None
            True duality gap objective - dual, evaluated once at termination. Supplied
            only by ``PrimalDualSolver``'s inequality-only path on a normal exit;
            ``None`` otherwise.

        """
        gap_str = (
            "" if duality_gap is None else f" [true duality gap {duality_gap:.6g}]"
        )
        _pd_solver_logger.info(
            "  PD completed in %.3f ms after %d iters, status=%d (%s)%s%s",
            elapsed_ms,
            nits,
            status,
            solver,
            gap_str,
            f": {reason}" if reason else "",
            extra={
                "cvx_event": "pd_done",
                "cvx_solver": solver,
                "cvx_nits": nits,
                "cvx_status": status,
                "cvx_elapsed_ms": elapsed_ms,
                "cvx_reason": reason,
                "cvx_duality_gap": duality_gap,
            },
        )

    def want_btls_details(self) -> bool:
        """Return whether line-search records would be logged.

        Returns
        -------
         want : bool
            True exactly when the ``cvxium.optimization`` logger is enabled at DEBUG,
            the level at which ``on_btls`` logs.

        """
        return _optimization_logger.isEnabledFor(logging.DEBUG)


class ListTrace(Trace):
    """In-memory trace that collects each event as a dict.

    Needs no logging configuration, and records every argument of every hook, a superset
    of the structured fields that ``IterationHandler`` extracts from ``LoggingTrace``
    records. Both ``want_*`` methods return True, so the solver computes the extra
    diagnostics on every step: two additional ``evaluate_barrier_objective`` calls per
    barrier Newton step (to populate ``on_btls``) and one additional KKT residual
    evaluation per primal-dual iteration (to populate ``rt_new``). That is not free in
    tight loops; prefer ``LoggingTrace`` on performance-critical paths.

    Each row is a dict with an ``event`` key naming the hook (its name without the
    ``on_`` prefix), a wall-clock ``time`` in seconds since the epoch, and one key per
    argument of the hook, named as in the hook's signature:
    - ``feasible_initial_guess``: no further keys.
    - ``svd_done``: ``elapsed_ms``.
    - ``phase1_feasible``, ``ipm_start``, ``hsd_start``: ``solver``.
    - ``centering_start``: ``outer_nit``, ``t``.
    - ``centering_done``: ``outer_nit``, ``elapsed_ms``, ``duality_gap``.
    - ``dual_eval_start``, ``early_stop``: ``outer_nit``.
    - ``dual_eval_done``: ``outer_nit``, ``elapsed_ms``.
    - ``converged``: ``outer_nit``, ``duality_gap``.
    - ``ipm_done``: ``solver``, ``elapsed_ms``.
    - ``newton_step``: ``outer_nit``, ``inner_nit``, ``elapsed_ms``, ``suboptimality``,
      ``lambda_norm``, ``quadratic_convergence_threshold``.
    - ``saddle_newton_step``: ``outer_nit``, ``inner_nit``, ``elapsed_ms``,
      ``residual_norm``, ``lambda_norm``.
    - ``dual_newton_step``: ``inner_nit``, ``elapsed_ms``, ``gradient_norm``.
    - ``btls``: ``outer_nit``, ``inner_nit``, ``btls_s``, ``improvement``,
      ``expected_improvement``.
    - ``hsd_iteration``: ``nit``, ``mu``, ``rel_pres``, ``rel_dres``, ``tau``,
      ``kappa``, ``elapsed_ms``.
    - ``hsd_done``: ``solver``, ``nits``, ``status``, ``elapsed_ms``.
    - ``pd_start``: ``solver``, ``num_vars``, ``num_ineq``, ``tol``.
    - ``pd_iteration``: ``nit``, ``eta``, ``dual_norm``, ``pri_norm``,
      ``residual_norm``, ``t``, ``step``, ``elapsed_ms``, ``sigma``, ``rho``,
      ``damping``, ``cent_norm``, ``rt_norm``, ``dir_ms``, ``rt_new``.
    - ``pd_damping_engaged``: ``nit``, ``damping``, ``retries``.
    - ``pd_done``: ``solver``, ``nits``, ``status``, ``elapsed_ms``, ``reason``,
      ``duality_gap``.
    Optional arguments the solver did not supply are recorded with their defaults
    (usually ``None``).

    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def records(self) -> list[dict[str, Any]]:
        """Return the accumulated records as a list of dicts.

        The outer list is a fresh copy, so appending to or reordering it does not affect
        the trace. The dicts inside are the objects the trace holds, so mutating one
        (for example, ``trace.records()[0]["event"] = "X"``) changes the trace's record
        too. Treat them as read-only, or ``copy.deepcopy`` the result before mutating
        it.

        Returns
        -------
         records : list[dict[str, Any]]
            One dict per event, in the order the events were emitted.

        """
        return list(self._records)

    def clear(self) -> None:
        """Discard every accumulated record."""
        self._records.clear()

    def want_btls_details(self) -> bool:
        """Return True, so that the solver calls ``on_btls`` on every inner step.

        Returns
        -------
         want : bool
            Always True.

        """
        return True

    def _append(self, event: str, **fields: Any) -> None:
        """Append one event row, stamped with the current wall-clock time.

        Parameters
        ----------
         event : str
            Event name, stored under ``event``.
         **fields : Any
            The event's arguments, stored under their own names.

        """
        row: dict[str, Any] = {"time": time.time(), "event": event}
        row.update(fields)
        self._records.append(row)

    def on_feasible_initial_guess(self) -> None:
        """Record that the initial guess already satisfies A x = b.

        Appends a ``feasible_initial_guess`` row with no further keys.

        """
        self._append("feasible_initial_guess")

    def on_svd_done(self, elapsed_ms: float) -> None:
        """Record the time taken to compute an equality-feasible point.

        Appends a ``svd_done`` row carrying each argument under its own name.

        Parameters
        ----------
         elapsed_ms : float
            Wall-clock time of the computation, in milliseconds: the SVD of A and the
            projection that follows it, or, for a sparse A, the saddle-point projection,
            which forms no SVD.

        """
        self._append("svd_done", elapsed_ms=elapsed_ms)

    def on_phase1_feasible(self, solver: str) -> None:
        """Record that Phase I returned a point feasible for this solver.

        Appends a ``phase1_feasible`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """
        self._append("phase1_feasible", solver=solver)

    def on_ipm_start(self, solver: str) -> None:
        """Record that the outer barrier loop is about to begin.

        Appends a ``ipm_start`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """
        self._append("ipm_start", solver=solver)

    def on_centering_start(self, outer_nit: int, t: float) -> None:
        """Record that a centering step is starting.

        Appends a ``centering_start`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         t : float
            Barrier parameter for this centering step.

        """
        self._append("centering_start", outer_nit=outer_nit, t=t)

    def on_centering_done(
        self, outer_nit: int, elapsed_ms: float, duality_gap: float
    ) -> None:
        """Record that a centering step has finished.

        Appends a ``centering_done`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the centering step, in milliseconds.
         duality_gap : float
            Duality gap at the centered iterate. The barrier method reports the per-step
            gap objective - dual at the returned iterate; ``StationaritySolver`` reports
            the surrogate gap n_b / t, where n_b counts its barrier terms (0.0 when
            there are none).

        """
        self._append(
            "centering_done",
            outer_nit=outer_nit,
            elapsed_ms=elapsed_ms,
            duality_gap=duality_gap,
        )

    def on_dual_eval_start(self, outer_nit: int) -> None:
        """Record that the per-centering dual evaluation is starting.

        Appends a ``dual_eval_start`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.

        """
        self._append("dual_eval_start", outer_nit=outer_nit)

    def on_dual_eval_done(self, outer_nit: int, elapsed_ms: float) -> None:
        """Record that the per-centering dual evaluation has finished.

        Appends a ``dual_eval_done`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the dual evaluation, in milliseconds.

        """
        self._append("dual_eval_done", outer_nit=outer_nit, elapsed_ms=elapsed_ms)

    def on_early_stop(self, outer_nit: int) -> None:
        """Record that a solve is stopping before its outer loop completes.

        Appends a ``early_stop`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Iteration at which the solve stopped, 1-indexed: the outer (centering)
            iteration of a barrier solve, or the iteration count of a primal-dual solve.

        """
        self._append("early_stop", outer_nit=outer_nit)

    def on_converged(self, outer_nit: int, duality_gap: float) -> None:
        """Record that the outer loop met its gap tolerance.

        Appends a ``converged`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer iteration at which the tolerance was met, 1-indexed.
         duality_gap : float
            The gap that met the tolerance: the best-bounds certificate in the barrier
            method, the surrogate gap in ``StationaritySolver``.

        """
        self._append("converged", outer_nit=outer_nit, duality_gap=duality_gap)

    def on_ipm_done(self, solver: str, elapsed_ms: float) -> None:
        """Record that the outer barrier loop has finished.

        Appends a ``ipm_done`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         elapsed_ms : float
            Wall-clock time of the outer loop, in milliseconds.

        """
        self._append("ipm_done", solver=solver, elapsed_ms=elapsed_ms)

    def on_newton_step(
        self,
        outer_nit: int,
        inner_nit: int,
        elapsed_ms: float,
        suboptimality: float,
        lambda_norm: float,
        quadratic_convergence_threshold: bool,
    ) -> None:
        """Record a completed Newton step of a barrier centering step.

        Appends a ``newton_step`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Newton iteration within the centering step, 1-indexed.
         elapsed_ms : float
            Wall-clock time to compute the Newton step, in milliseconds.
         suboptimality : float
            Estimated suboptimality λ^2 / 2 of the iterate for the centering problem,
            where λ is the Newton decrement.
         lambda_norm : float
            The Newton decrement λ at the start of the step.
         quadratic_convergence_threshold : bool
            True only on the first inner step of the centering step whose Newton
            decrement satisfies λ ≤ η, where η = (1 - 2α) / 4 and α is the Armijo
            parameter: the step at which Newton's method enters its quadratically
            convergent phase.

        """
        self._append(
            "newton_step",
            outer_nit=outer_nit,
            inner_nit=inner_nit,
            elapsed_ms=elapsed_ms,
            suboptimality=suboptimality,
            lambda_norm=lambda_norm,
            quadratic_convergence_threshold=quadratic_convergence_threshold,
        )

    def on_saddle_newton_step(
        self,
        outer_nit: int,
        inner_nit: int,
        elapsed_ms: float,
        residual_norm: float,
        lambda_norm: float,
    ) -> None:
        """Record a completed Newton step of a saddle centering step.

        Appends a ``saddle_newton_step`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Accepted inner step within the centering, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the step, including its line search, in milliseconds.
         residual_norm : float
            ‖r(x, y, t, λ)‖_2, the norm of the residual the step drives toward zero,
            measured at the start of the step.
         lambda_norm : float
            ‖λ‖_2, the norm of the equality multiplier after the step; 0.0 when the
            problem has no ``x_equality``.

        """
        self._append(
            "saddle_newton_step",
            outer_nit=outer_nit,
            inner_nit=inner_nit,
            elapsed_ms=elapsed_ms,
            residual_norm=residual_norm,
            lambda_norm=lambda_norm,
        )

    def on_dual_newton_step(
        self,
        inner_nit: int,
        elapsed_ms: float,
        gradient_norm: float,
    ) -> None:
        """Record a completed Newton step of a dual minimization.

        Appends a ``dual_newton_step`` row carrying each argument under its own name.

        Parameters
        ----------
         inner_nit : int
            Step within the dual minimization, 1-indexed.
         elapsed_ms : float
            Wall-clock time of the step, including its line search, in milliseconds.
         gradient_norm : float
            ‖∇_x L‖, the norm of the Lagrangian gradient at the start of the step, the
            quantity the minimization drives toward zero.

        """
        self._append(
            "dual_newton_step",
            inner_nit=inner_nit,
            elapsed_ms=elapsed_ms,
            gradient_norm=gradient_norm,
        )

    def on_btls(
        self,
        outer_nit: int,
        inner_nit: int,
        btls_s: float,
        improvement: float,
        expected_improvement: float,
    ) -> None:
        """Record a completed backtracking line search.

        Appends a ``btls`` row carrying each argument under its own name.

        Parameters
        ----------
         outer_nit : int
            Outer (centering) iteration, 1-indexed.
         inner_nit : int
            Newton iteration within the centering step, 1-indexed.
         btls_s : float
            The accepted step size s.
         improvement : float
            Actual decrease of the barrier objective, f_t(x) - f_t(x + s Δx).
         expected_improvement : float
            Reference decrease α β λ^2 / (1 + λ), where λ is the Newton decrement and α
            and β are the Armijo parameters.

        """
        self._append(
            "btls",
            outer_nit=outer_nit,
            inner_nit=inner_nit,
            btls_s=btls_s,
            improvement=improvement,
            expected_improvement=expected_improvement,
        )

    def on_hsd_start(self, solver: str) -> None:
        """Record that the homogeneous self-dual loop is about to begin.

        Appends a ``hsd_start`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.

        """
        self._append("hsd_start", solver=solver)

    def on_hsd_iteration(
        self,
        nit: int,
        mu: float,
        rel_pres: float,
        rel_dres: float,
        tau: float,
        kappa: float,
        elapsed_ms: float,
    ) -> None:
        """Record a completed homogeneous self-dual iteration.

        Appends a ``hsd_iteration`` row carrying each argument under its own name.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         mu : float
            Complementarity gap μ at the top of the iteration, before its Newton step.
         rel_pres : float
            Relative primal residual ‖r_p‖ / (1 + ‖b‖) at the top of the iteration.
         rel_dres : float
            Relative dual residual ‖r_d‖ / (1 + ‖c‖) at the top of the iteration.
         tau : float
            Homogenizing variable τ after the step.
         kappa : float
            Homogenizing variable κ after the step.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.

        """
        self._append(
            "hsd_iteration",
            nit=nit,
            mu=mu,
            rel_pres=rel_pres,
            rel_dres=rel_dres,
            tau=tau,
            kappa=kappa,
            elapsed_ms=elapsed_ms,
        )

    def on_hsd_done(
        self,
        solver: str,
        nits: int,
        status: int,
        elapsed_ms: float,
    ) -> None:
        """Record that the homogeneous self-dual loop has terminated.

        Appends a ``hsd_done`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         nits : int
            Number of iterations completed, each with one Newton step.
         status : int
            Integer status code: 0 converged to the outer tolerance, 1 reached only the
            soft tolerance, 2 did not converge.
         elapsed_ms : float
            Wall-clock time of the whole loop, in milliseconds.

        """
        self._append(
            "hsd_done",
            solver=solver,
            nits=nits,
            status=status,
            elapsed_ms=elapsed_ms,
        )

    def on_pd_start(
        self,
        solver: str,
        num_vars: int | None = None,
        num_ineq: int | None = None,
        tol: float | None = None,
    ) -> None:
        """Record that a primal-dual loop is about to begin.

        Appends a ``pd_start`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         num_vars : int | None
            Number of decision variables, when supplied.
         num_ineq : int | None
            Number of inequality constraints, when supplied.
         tol : float | None
            Outer convergence tolerance, when supplied.

        """
        self._append(
            "pd_start",
            solver=solver,
            num_vars=num_vars,
            num_ineq=num_ineq,
            tol=tol,
        )

    def on_pd_iteration(
        self,
        nit: int,
        eta: float,
        dual_norm: float,
        pri_norm: float,
        residual_norm: float,
        t: float,
        step: float,
        elapsed_ms: float,
        sigma: float | None = None,
        rho: float | None = None,
        damping: float = 0.0,
        cent_norm: float | None = None,
        rt_norm: float | None = None,
        dir_ms: float | None = None,
        rt_new: float | None = None,
    ) -> None:
        """Record a completed primal-dual iteration.

        Appends a ``pd_iteration`` row carrying each argument under its own name.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed within the loop.
         eta : float
            Surrogate duality gap η = -f^T λ at the top of the iteration.
         dual_norm : float
            ‖r_dual‖_2, the dual (stationarity) residual norm at the top of the
            iteration.
         pri_norm : float
            ‖A x - b‖_2, the equality residual norm; 0.0 when the problem has no
            equality constraints.
         residual_norm : float
            The proxy sqrt(‖r_dual‖^2 + ‖A x - b‖^2 + η^2), kept for the result history.
         t : float
            Surrogate barrier parameter μ m / η, where m counts the inequality
            constraints and μ is the barrier multiplier. The two-block solver omits μ
            and reports 0.0 when η ≤ 0.
         step : float
            Step length actually taken.
         elapsed_ms : float
            Wall-clock time of the iteration, in milliseconds.
         sigma : float | None
            Mehrotra centering parameter (in the two-block solver, the mean of its two
            blocks' values).
         rho : float | None
            Ratio of the actual to the predicted residual reduction; ``None`` when the
            step was undamped and the ratio test was skipped.
         damping : float
            Levenberg damping used for the accepted step.
         cent_norm : float | None
            ‖r_cent‖_2, the centrality residual norm.
         rt_norm : float | None
            ‖[r_dual; r_cent]‖_2, the KKT residual at barrier parameter t, before the
            step.
         dir_ms : float | None
            Time to compute the search direction, in milliseconds.
         rt_new : float | None
            The same KKT residual after the step, at the same t; computed only when
            ``want_pd_step_details`` returns True, otherwise ``None``.

        """
        self._append(
            "pd_iteration",
            nit=nit,
            eta=eta,
            dual_norm=dual_norm,
            pri_norm=pri_norm,
            residual_norm=residual_norm,
            t=t,
            step=step,
            elapsed_ms=elapsed_ms,
            sigma=sigma,
            rho=rho,
            damping=damping,
            cent_norm=cent_norm,
            rt_norm=rt_norm,
            dir_ms=dir_ms,
            rt_new=rt_new,
        )

    def want_pd_step_details(self) -> bool:
        # Collectors want the post-step KKT residual populated for analysis.
        """Return True, so that primal-dual solvers compute the post-step residual.

        Returns
        -------
         want : bool
            Always True.

        """
        return True

    def on_pd_damping_engaged(self, nit: int, damping: float, retries: int) -> None:
        """Record that Levenberg damping had to grow.

        Appends a ``pd_damping_engaged`` row carrying each argument under its own name.

        Parameters
        ----------
         nit : int
            Iteration, 0-indexed, as in ``on_pd_iteration``.
         damping : float
            Levenberg damping of the accepted step.
         retries : int
            Number of times the damping was grown before a feasible step was found.

        """
        self._append(
            "pd_damping_engaged",
            nit=nit,
            damping=damping,
            retries=retries,
        )

    def on_pd_done(
        self,
        solver: str,
        nits: int,
        status: int,
        elapsed_ms: float,
        reason: str = "",
        duality_gap: float | None = None,
    ) -> None:
        """Record that a primal-dual loop has terminated.

        Appends a ``pd_done`` row carrying each argument under its own name.

        Parameters
        ----------
         solver : str
            Human-readable solver name: the label configured in the solver's settings,
            or by default its class name.
         nits : int
            Number of iterations started, including a final one that stopped without
            taking a step.
         status : int
            Integer status code. 0 means converged; the meaning of 1 and 2 depends on
            the solver. For ``PrimalDualSolver``, 1 is a stall or the iteration cap, and
            2 a feasibility stop or certified infeasibility. For
            ``PrimalDualStationaritySolver``, 1 is the iteration cap or another
            non-converged break, and 2 acceptance at the soft tolerance.
         elapsed_ms : float
            Wall-clock time of the whole loop, in milliseconds.
         reason : str
            Human-readable termination cause; empty when the solver supplies none (as on
            ``PrimalDualSolver``'s equality path).
         duality_gap : float | None
            True duality gap objective - dual, evaluated once at termination. Supplied
            only by ``PrimalDualSolver``'s inequality-only path on a normal exit;
            ``None`` otherwise.

        """
        self._append(
            "pd_done",
            solver=solver,
            nits=nits,
            status=status,
            elapsed_ms=elapsed_ms,
            reason=reason,
            duality_gap=duality_gap,
        )


class DataFrameTrace(ListTrace):
    """List trace that converts its records to a pandas DataFrame.

    Adds ``to_dataframe``, which gives one row per event with sparse columns (``NaN``
    where an event does not carry a field), like ``IterationHandler.to_dataframe``.
    pandas is needed only when ``to_dataframe`` is called, not at construction.

    """

    def to_dataframe(self) -> Any:
        """Convert the accumulated records to a pandas DataFrame.

        Requires pandas, which Cvxium does not otherwise depend on; it is imported only
        here.

        Returns
        -------
         df : pd.DataFrame
            One row per recorded event. Columns are the union of the keys across
            records: ``time`` and ``event`` always, plus the event fields listed in
            ``ListTrace``, such as ``outer_nit``, ``inner_nit``, ``elapsed_ms``,
            ``suboptimality``, ``t``, ``btls_s`` and ``solver``. A column with no data
            for a given event holds ``NaN`` in that row.

        Raises
        ------
         ImportError
            When pandas is not installed.

        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required to call to_dataframe(). "
                "Install it with: pip install pandas"
            ) from e
        return pd.DataFrame(self._records)
