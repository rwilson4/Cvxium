"""Homogeneous self-dual interior-point solver for Cvxium's linear conic programs.

Defines ``HsdSolver``, which solves the linear conic program
   minimize c^T x   subject to   A x = b,   x ∈ K,
over a product of symmetric cones K = K_1 × ... × K_q (see ``cone``), and the private
``_HsdEmbedding``, which holds the embedded iterate and computes the residuals, Newton
steps, step lengths, and infeasibility certificates that the solver's loop needs.

Following (Andersen, Roos, and Terlaky, 2003), the solver embeds the program in a
homogeneous model, adding a scalar τ that homogenizes the equations and a scalar κ
complementary to it. The starting point x = s = e_K, y = 0, τ = κ = 1 is built from the
cones' interior points, so no Phase I solve and no strictly feasible x0 are needed. When
the iterates converge with τ > 0, x / τ solves the program; when τ collapses while κ
stays positive, the iterate yields a certificate of primal infeasibility (b^T y > 0) or
of dual infeasibility, that is, unboundedness (c^T x < 0).

Each iteration takes one damped Newton step toward the central path, with a fixed
centering parameter and the Nesterov-Todd scaling W of each cone block. The step reduces
to a (p + 1) × (p + 1) linear system, where p is the number of equality constraints;
each cone assembles its share of the coefficients from its structured W^2 through
``Cone.accumulate_normal_equations``, so W^2 is never formed densely. The embedding also
provides the pieces of a Mehrotra predictor-corrector (``compute_centering_rhs`` with
affine deltas, and ``predicted_mu``), which the solver loop does not call.

The problem's objective must be a ``LinearObjective``, and its equality constraints
supply (A, b). The cones, not inequality-constraint layers, define K: the problem's
inequality layers are not read. ``second_order_cone_programs`` maps second-order cone
programs into this form and subclasses ``_HsdEmbedding`` with a structured kernel.

References
----------
Andersen, Erling D., Roos, Cornelis and Terlaky, Tamás, On Implementing a Primal-Dual
Interior-Point Method for Conic Quadratic Optimization, Mathematical Programming 95,
2003.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from ..exceptions import ConeBoundaryError
from ..problems.objective import LinearObjective
from ..problems.result import (
    HsdDiagnostics,
    InfeasibilityCertificate,
    ObjectiveOutcome,
    SolveResult,
    SolveStatus,
)
from ..solvers.optimization import (
    _duals_from,
    _legacy_solve_status,
)
from ..solvers.settings import (
    OptimizationSettings,
)
from ..solvers.solver import Solver

if TYPE_CHECKING:
    from ..diagnostics.trace import Trace
    from ..problems.cone import Cone
    from ..problems.problem import (
        OnInfeasible,
        Problem,
    )


class HsdSolver(Solver):
    """Homogeneous self-dual interior-point solver for linear conic programs.

    Solves
       minimize c^T x   subject to   A x = b,   x ∈ K,
    where the ``Problem`` supplies c (through its ``LinearObjective``) and (A, b)
    (through its equality constraints, if any), and the solver's cones define K. Builds
    the homogeneous embedding (``_HsdEmbedding``), runs a path-following loop on it, and
    returns either the recovered solution or an infeasibility or unboundedness
    certificate. Throughout, n = ``problem.dimension`` and p is the number of equality
    constraints.

    Unlike ``BarrierSolver`` and ``PrimalDualSolver``, ``HsdSolver`` needs no strictly
    feasible starting point: the embedding starts from the cones' canonical interior
    points, so ``solve`` ignores ``x0``.

    Parameters
    ----------
     cones : list[Cone]
        The cone blocks of K, in the order their coordinates appear in x. Their
        dimensions must sum to n, or ``solve`` raises ``ValueError``.
     settings : OptimizationSettings | None
        Solver settings; ``None`` uses the defaults. The solver reads
        ``outer.tolerance``, ``outer.tolerance_soft``, ``trace`` and
        ``observability.label``. The iteration cap (100) and the centering parameter
        (0.1) are fixed in ``solve``.
     embedding_class : type[_HsdEmbedding] | None
        The embedding class to instantiate; ``None`` uses ``_HsdEmbedding``. A subclass
        can override ``residuals`` and ``kkt_solve`` with a structure-exploiting kernel,
        as ``second_order_cone_programs`` does.

    """

    def __init__(
        self,
        cones: list[Cone],
        settings: OptimizationSettings | None = None,
        embedding_class: type[_HsdEmbedding] | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._cones: list[Cone] = list(cones)
        self._embedding_class: type[_HsdEmbedding] = (
            embedding_class if embedding_class is not None else _HsdEmbedding
        )

    def solve(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Solve a linear conic program through the homogeneous self-dual embedding.

        Builds the embedding, starts it at its canonical interior point, and iterates
        until the stopping test passes, a Newton step fails, or 100 iterations elapse.
        Convergence requires three measures, all on the embedded iterate, to fall below
        ``outer.tolerance``:
        - the relative primal residual ‖r_p‖ / (1 + ‖b‖),
        - the relative dual residual ‖r_d‖ / (1 + ‖c‖),
        - the complementarity gap μ (see ``_HsdEmbedding.complementarity_gap``).

        Dividing by τ recovers the solution and the equality and inequality multipliers,
           x / τ,   y / τ,   s / τ,
        where s is the dual slack. When the loop reaches neither tolerance, the
        embedding is checked for an infeasibility or unboundedness certificate.

        Parameters
        ----------
         problem : Problem
            The program to solve. Its objective must be a ``LinearObjective``; its
            equality constraints, if any, give (A, b). Its inequality layers are not
            read.
         x0 : npt.NDArray[np.float64] | None
            Accepted for uniformity with the ``Solver`` interface and ignored.
         on_infeasible : OnInfeasible
            Accepted for uniformity with the ``Solver`` interface. ``"continue"`` is not
            supported. The default ``"raise"`` has no effect: an infeasible program
            yields a result carrying a certificate, not an exception.

        Returns
        -------
         result : SolveResult
            The recovered solution, with ``solver`` set to ``"hsd"`` and
            ``HsdDiagnostics`` holding the τ, κ and μ histories. The status is
            ``INFEASIBLE`` or ``UNBOUNDED`` when a certificate was found, and otherwise
            ``OPTIMAL`` (``outer.tolerance`` reached), ``ACCEPTABLE`` (only
            ``outer.tolerance_soft`` reached), or ``MAX_ITERATIONS`` (neither reached,
            including a stop after a failed Newton step).

        Raises
        ------
         NotImplementedError
            When ``on_infeasible`` is ``"continue"``.
         ValueError
            Propagated from ``_HsdEmbedding`` when the cone dimensions do not sum to
            ``problem.dimension`` or the objective is not a ``LinearObjective``, and
            from the cone and matrix operations on a shape mismatch.

        Notes
        -----
        A Newton step fails when the iterate reaches a cone boundary, where the
        Nesterov-Todd scaling is undefined (``ConeBoundaryError``), or when the reduced
        system is singular (``np.linalg.LinAlgError``). The loop catches both and stops
        with ``ACCEPTABLE`` when the iterate before the step met
        ``outer.tolerance_soft``, and with ``MAX_ITERATIONS`` otherwise.

        """
        if on_infeasible == "continue":
            raise NotImplementedError(
                "HsdSolver does not support on_infeasible='continue'; use "
                "BarrierSolver (the IPM engine) for the continue policy."
            )
        del x0  # HSD constructs its own start; see class docstring.
        embedding = self._embedding_class(problem=problem, cones=self._cones)
        embedding.initial_iterate()

        # OptimizationSettings has no outer-iteration cap; use a generous
        # hardcoded ceiling. Mehrotra adaptive sigma would tighten this
        # further; the long-step driver here typically needs 30-50 iters.
        max_iters = 100
        settings = (
            self.settings if self.settings is not None else OptimizationSettings()
        )
        sigma = 0.1  # Fixed centering parameter (Mehrotra adaptation deferred).

        trace = settings.trace
        label = settings.observability.label
        solver_name = label if label is not None else self.__class__.__name__
        overall_start_time = time.time()
        trace.on_hsd_start(solver_name)

        (
            mu,
            status,
            message,
            tau_history,
            kappa_history,
            gap_history,
        ) = self._run_iteration_loop(
            embedding=embedding,
            max_iters=max_iters,
            sigma=sigma,
            outer_tol=settings.outer.tolerance,
            soft_tol=settings.outer.tolerance_soft,
            trace=trace,
        )
        trace.on_hsd_done(
            solver=solver_name,
            nits=len(tau_history) - 1,
            status=status,
            elapsed_ms=1000.0 * (time.time() - overall_start_time),
        )
        return self._build_result(
            embedding=embedding,
            mu=mu,
            status=status,
            message=message,
            tau_history=tau_history,
            kappa_history=kappa_history,
            gap_history=gap_history,
        )

    def _run_iteration_loop(
        self,
        embedding: _HsdEmbedding,
        max_iters: int,
        sigma: float,
        outer_tol: float,
        soft_tol: float,
        trace: Trace,
    ) -> tuple[
        float,
        Literal[0, 1, 2],
        str,
        list[float],
        list[float],
        list[float],
    ]:
        """Run the path-following loop until it converges, fails, or hits the cap.

        At each iteration, computes μ and the relative primal and dual residuals at the
        current iterate, stops with status 0 when all three are below ``outer_tol``, and
        otherwise takes one Newton step (``_take_newton_step``). A ``ConeBoundaryError``
        or ``np.linalg.LinAlgError`` from the step ends the loop with status 1 when the
        pre-step iterate met ``soft_tol``, and with status 2 otherwise; the iterate is
        left at its pre-step value. ``_promote_terminal_status`` then settles the final
        status.

        Parameters
        ----------
         embedding : _HsdEmbedding
            The embedding, already at its initial iterate; advanced in place.
         max_iters : int
            Maximum number of Newton steps.
         sigma : float
            Centering parameter σ, fixed across iterations.
         outer_tol : float
            Hard tolerance on μ and the relative residuals.
         soft_tol : float
            Soft tolerance on the same measures.
         trace : Trace
            Receives one event per completed iteration.

        Returns
        -------
         mu : float
            Complementarity gap at the iterate the final status describes.
         status : Literal[0, 1, 2]
            0 when ``outer_tol`` was reached, 1 when only ``soft_tol`` was reached, and
            2 otherwise.
         message : str
            Human-readable description of how the loop ended.
         tau_history : list[float]
            τ at the initial iterate and after each completed step.
         kappa_history : list[float]
            κ at the same iterates.
         gap_history : list[float]
            μ at the same iterates.

        Raises
        ------
         ValueError
            Propagated from the cone methods on a shape mismatch; deliberately not
            caught.

        """
        # Reference scales for relative residual checks.
        b_scale = 1.0 + float(np.linalg.norm(embedding._b))
        c_scale = 1.0 + float(np.linalg.norm(embedding._c))

        tau_history: list[float] = [embedding._tau or 0.0]
        kappa_history: list[float] = [embedding._kappa or 0.0]
        gap_history: list[float] = [embedding.complementarity_gap()]

        status: Literal[0, 1, 2] = (
            2  # ``ran out of iterations`` until proven otherwise.
        )
        message: str = (
            f"HSD long-step driver hit max_iters={max_iters} without converging."
        )
        mu = gap_history[0]
        rel_pres = float("inf")
        rel_dres = float("inf")
        converged_soft = False
        # ``True`` only when the loop body exhausted ``max_iters`` cleanly
        # (no ``converged_hard`` break, no ``LinAlgError``). Gates the
        # post-loop recompute on the advanced iterate.
        exhausted_iters = True

        for nit in range(max_iters):
            iter_start = time.time()
            mu = embedding.complementarity_gap()
            r_p, r_d, r_g, _r_xs, _r_tk = embedding.residuals()
            rel_pres = float(np.linalg.norm(r_p)) / b_scale
            rel_dres = float(np.linalg.norm(r_d)) / c_scale

            converged_hard = (
                mu < outer_tol and rel_pres < outer_tol and rel_dres < outer_tol
            )
            converged_soft = (
                mu < soft_tol and rel_pres < soft_tol and rel_dres < soft_tol
            )
            if converged_hard:
                status = 0
                message = (
                    f"HSD converged to outer.tolerance={outer_tol:.1e} "
                    f"(mu={mu:.2e}, pres={rel_pres:.2e}, dres={rel_dres:.2e})."
                )
                exhausted_iters = False
                break
            try:
                self._take_newton_step(
                    embedding=embedding,
                    sigma=sigma,
                    mu=mu,
                    r_p=r_p,
                    r_d=r_d,
                    r_g=r_g,
                )
            except (ConeBoundaryError, np.linalg.LinAlgError) as exc:
                # Two boundary-degeneracy modes terminate cleanly here:
                # (a) NT scaling raises ``ConeBoundaryError`` -- requires
                #     strictly-interior x and s; on tight boundary-optimum
                #     problems the iterate creeps onto the cone boundary
                #     and NT becomes undefined. (We deliberately do NOT
                #     catch bare ``ValueError`` so shape-mismatch / API
                #     misuse still propagates as a programming error.)
                # (b) ``np.linalg.solve`` raises ``LinAlgError`` -- the
                #     reduced K matrix becomes structurally singular when
                #     ``outer.tolerance`` is set sub-progress-noise.
                # In both cases, terminate with whatever soft-tolerance
                # progress we made; a future Mehrotra adaptation (with
                # boundary-aware sigma) would push further. The
                # ``converged_soft`` check uses the loop-top values of
                # ``mu`` / ``rel_pres`` / ``rel_dres``, which already
                # reflect the pre-step iterate; skip the post-loop
                # recompute (a re-call would just re-raise).
                exhausted_iters = False
                if converged_soft:
                    status = 1
                    message = (
                        f"HSD reached outer.tolerance_soft={soft_tol:.1e} "
                        f"before the inner Newton step failed at the "
                        f"iterate boundary (mu={mu:.2e}, "
                        f"pres={rel_pres:.2e}, dres={rel_dres:.2e}). "
                        f"Underlying error: {exc}"
                    )
                else:
                    status = 2
                    message = (
                        "HSD long-step driver hit a cone-boundary inner "
                        f"failure before reaching outer.tolerance_soft="
                        f"{soft_tol:.1e} (mu={mu:.2e}). Underlying error: {exc}"
                    )
                break
            tau_history.append(embedding._tau or 0.0)
            kappa_history.append(embedding._kappa or 0.0)
            gap_history.append(embedding.complementarity_gap())
            trace.on_hsd_iteration(
                nit=nit,
                mu=mu,
                rel_pres=rel_pres,
                rel_dres=rel_dres,
                tau=embedding._tau or 0.0,
                kappa=embedding._kappa or 0.0,
                elapsed_ms=1000.0 * (time.time() - iter_start),
            )

        status, message, mu, rel_pres, rel_dres, converged_soft = (
            self._promote_terminal_status(
                embedding=embedding,
                status=status,
                message=message,
                mu=mu,
                rel_pres=rel_pres,
                rel_dres=rel_dres,
                converged_soft=converged_soft,
                exhausted_iters=exhausted_iters,
                outer_tol=outer_tol,
                soft_tol=soft_tol,
                b_scale=b_scale,
                c_scale=c_scale,
            )
        )
        return mu, status, message, tau_history, kappa_history, gap_history

    @staticmethod
    def _take_newton_step(
        embedding: _HsdEmbedding,
        sigma: float,
        mu: float,
        r_p: npt.NDArray[np.float64],
        r_d: npt.NDArray[np.float64],
        r_g: float,
    ) -> None:
        """Take one damped Newton step toward the central path.

        Builds the centering right-hand sides for the target σμ
        (``compute_centering_rhs`` without affine deltas), solves the embedded Newton
        system with the current residuals as the linear right-hand sides
        (``kkt_solve``), and advances the embedding by the step length from
        ``compute_step_length``. The residuals pass through unchanged because their
        signs make a full step zero them: for example, r_p = b τ - A x, and the step
        solves
           A Δx - b Δτ = r_p.

        Parameters
        ----------
         embedding : _HsdEmbedding
            The embedding, advanced in place.
         sigma : float
            Centering parameter σ.
         mu : float
            Complementarity gap at the current iterate.
         r_p : npt.NDArray[np.float64]
            Primal residual, of length p.
         r_d : npt.NDArray[np.float64]
            Dual residual, of length n.
         r_g : float
            Gap residual.

        Raises
        ------
         ConeBoundaryError
            Propagated from the Nesterov-Todd scaling when x or s is not strictly inside
            its cone.
         np.linalg.LinAlgError
            Propagated from ``kkt_solve`` when the reduced system is singular.

        """
        rhs_xs_sym, rhs_tk_sym = embedding.compute_centering_rhs(sigma=sigma, mu=mu)
        # The residuals are signed so that ``r = 0`` at feasibility
        # (e.g. ``r_p = -A x + b tau = -(A x - b tau)``). The
        # ``kkt_solve`` then expects ``A Delta_x - b Delta_tau = rhs_p``,
        # which is the Newton step that drives the next iterate to
        # ``A(x + Delta_x) - b(tau + Delta_tau) = 0``. Substituting:
        # ``rhs_p = -(A x - b tau) = r_p``. Pass the residuals directly.
        delta_x, delta_y, delta_tau, delta_s, delta_kappa = embedding.kkt_solve(
            rhs_p=r_p,
            rhs_d=r_d,
            rhs_g=r_g,
            rhs_xs=rhs_xs_sym,
            rhs_tk=rhs_tk_sym,
        )
        alpha = embedding.compute_step_length(delta_x, delta_s, delta_tau, delta_kappa)
        embedding.advance_iterate(
            alpha, delta_x, delta_y, delta_tau, delta_s, delta_kappa
        )

    @staticmethod
    def _promote_terminal_status(
        embedding: _HsdEmbedding,
        status: Literal[0, 1, 2],
        message: str,
        mu: float,
        rel_pres: float,
        rel_dres: float,
        converged_soft: bool,
        exhausted_iters: bool,
        outer_tol: float,
        soft_tol: float,
        b_scale: float,
        c_scale: float,
    ) -> tuple[Literal[0, 1, 2], str, float, float, float, bool]:
        """Recompute the final metrics after a full run and settle the status.

        When the loop used every iteration (``exhausted_iters``), the μ and residuals it
        holds describe the iterate before the last step. This recomputes them on the
        final iterate, so that the reported metrics match the returned solution, and
        promotes the status to 0 when that last step met ``outer_tol``. A loop that
        stopped early already holds consistent metrics and skips the recompute. In
        either case, a status of 2 becomes 1 when the soft test passed.

        Parameters
        ----------
         embedding : _HsdEmbedding
            The embedding at its final iterate.
         status : Literal[0, 1, 2]
            Status reported by the loop.
         message : str
            Message reported by the loop.
         mu : float
            Complementarity gap the loop last computed.
         rel_pres : float
            Relative primal residual the loop last computed.
         rel_dres : float
            Relative dual residual the loop last computed.
         converged_soft : bool
            Whether the loop's last soft test passed.
         exhausted_iters : bool
            True when the loop ran all ``max_iters`` steps without stopping early.
         outer_tol : float
            Hard tolerance.
         soft_tol : float
            Soft tolerance.
         b_scale : float
            Primal residual scale 1 + ‖b‖.
         c_scale : float
            Dual residual scale 1 + ‖c‖.

        Returns
        -------
         status : Literal[0, 1, 2]
            The settled status.
         message : str
            The message matching the settled status.
         mu : float
            Complementarity gap at the final iterate.
         rel_pres : float
            Relative primal residual at the final iterate.
         rel_dres : float
            Relative dual residual at the final iterate.
         converged_soft : bool
            Whether the final iterate meets ``soft_tol``.

        """
        if exhausted_iters:
            mu = embedding.complementarity_gap()
            r_p_final, r_d_final, _, _, _ = embedding.residuals()
            rel_pres = float(np.linalg.norm(r_p_final)) / b_scale
            rel_dres = float(np.linalg.norm(r_d_final)) / c_scale
            converged_hard = (
                mu < outer_tol and rel_pres < outer_tol and rel_dres < outer_tol
            )
            converged_soft = (
                mu < soft_tol and rel_pres < soft_tol and rel_dres < soft_tol
            )
            # Promote to status=0 if the final ``advance_iterate`` crossed
            # the hard tolerance: the loop body checks ``converged_hard``
            # BEFORE the step, so the very last step's success only
            # surfaces here.
            if converged_hard:
                status = 0
                message = (
                    f"HSD converged to outer.tolerance={outer_tol:.1e} on "
                    f"the final iterate (mu={mu:.2e}, pres={rel_pres:.2e}, "
                    f"dres={rel_dres:.2e})."
                )

        if status == 2 and converged_soft:
            status = 1
            message = (
                f"HSD reached outer.tolerance_soft={soft_tol:.1e} "
                f"(mu={mu:.2e}, pres={rel_pres:.2e}, dres={rel_dres:.2e}) "
                f"but did not reach outer.tolerance={outer_tol:.1e}."
            )
        return status, message, mu, rel_pres, rel_dres, converged_soft

    @staticmethod
    def _build_result(
        embedding: _HsdEmbedding,
        mu: float,
        status: Literal[0, 1, 2],
        message: str,
        tau_history: list[float],
        kappa_history: list[float],
        gap_history: list[float],
    ) -> SolveResult:
        """Recover the original-problem solution and package it as a ``SolveResult``.

        When the loop ended with status 2, first asks ``detect_infeasibility`` for a
        certificate. A converged run is not tested, since κ / (τ + κ) can cross the
        threshold transiently when τ and κ shrink together. A primal-infeasibility
        certificate y becomes the ``dual_witness`` of an ``InfeasibilityCertificate`` of
        kind ``"infeasible"``, and a dual-infeasibility certificate x the
        ``primal_witness`` of one of kind ``"unbounded"``. The certificate's message
        replaces the loop's only when a certificate was found.

        The solution, equality multipliers and inequality multipliers are x / τ, y / τ
        and s / τ when τ > 0, and the unscaled embedded vectors otherwise, so the result
        always has the right shapes. The objective value is c^T x / τ, the dual value is
        b^T y / τ, and the suboptimality is the embedded gap μ.

        Parameters
        ----------
         embedding : _HsdEmbedding
            The embedding at its final iterate.
         mu : float
            Complementarity gap at the final iterate.
         status : Literal[0, 1, 2]
            Settled loop status: 0 hard tolerance, 1 soft tolerance, 2 neither.
         message : str
            Settled loop message.
         tau_history : list[float]
            τ at each iterate; its length minus one is the iteration count.
         kappa_history : list[float]
            κ at each iterate.
         gap_history : list[float]
            μ at each iterate.

        Returns
        -------
         result : SolveResult
            The packaged result. Its status is ``INFEASIBLE`` or ``UNBOUNDED`` when a
            certificate was found, and otherwise ``status`` mapped to ``OPTIMAL``,
            ``ACCEPTABLE`` or ``MAX_ITERATIONS``.

        """
        # Infeasibility / unboundedness check. Only meaningful
        # when the loop did *not* already report optimality; on a
        # converged-feasible run kappa/(tau+kappa) may still cross the
        # threshold transiently (joint-collapse pattern) without
        # indicating infeasibility.
        primal_infeas_cert: npt.NDArray[np.float64] | None = None
        dual_infeas_cert: npt.NDArray[np.float64] | None = None
        if status not in (0, 1):
            primal_infeas_cert, dual_infeas_cert, cert_message = (
                embedding.detect_infeasibility()
            )
            # Only replace the loop's original message when a certificate
            # was actually populated. Otherwise (marginal-infeasibility
            # branch returns a diagnostic-only message with no cert) we'd
            # clobber the loop's "max iterations reached" status=2 text.
            if cert_message and (
                primal_infeas_cert is not None or dual_infeas_cert is not None
            ):
                message = cert_message

        tau_final = embedding._tau or 0.0
        x_emb = embedding._x
        y_emb = embedding._y
        s_emb = embedding._s
        assert x_emb is not None and y_emb is not None and s_emb is not None
        c_vec = embedding._c
        b_vec = embedding._b

        # Recover original-problem solution. When infeasibility was detected
        # the recovery is meaningless but we still populate ``solution`` (with
        # the un-rescaled embedded x) so callers that ignore status get a
        # sensible array shape.
        if tau_final > 0.0:
            solution = x_emb / tau_final
            equality_mults = y_emb / tau_final
            inequality_mults = s_emb / tau_final
        else:
            solution = x_emb.copy()
            equality_mults = y_emb.copy()
            inequality_mults = s_emb.copy()
        objective_value = float(c_vec @ solution)
        # ``b^T y`` is the LP-style dual objective; under tau-collapse the
        # ratio ``y_emb / tau`` is numerically noisy. Stable dual recovery
        # in the converged-feasible case requires the kappa/(tau+kappa) ~ 0
        # check above to have passed; otherwise treat as informational.
        dual_value = float(b_vec @ equality_mults)
        suboptimality = mu  # Embedded gap; the quantity the loop checked.

        nits = len(tau_history) - 1
        # Lift the HSD infeasibility certificates onto the unified `certificate`
        # field: a primal certificate proves infeasibility; a dual certificate
        # proves unboundedness.
        certificate: InfeasibilityCertificate | None = None
        if primal_infeas_cert is not None:
            certificate = InfeasibilityCertificate(
                kind="infeasible",
                dual_witness=primal_infeas_cert,
            )
        elif dual_infeas_cert is not None:
            certificate = InfeasibilityCertificate(
                kind="unbounded",
                primal_witness=dual_infeas_cert,
            )
        if certificate is not None:
            result_status = (
                SolveStatus.INFEASIBLE
                if certificate.kind == "infeasible"
                else SolveStatus.UNBOUNDED
            )
        else:
            result_status = _legacy_solve_status(status)
        return SolveResult(
            solution=solution,
            status=result_status,
            message=message,
            solver="hsd",
            n_iterations=nits,
            objective=ObjectiveOutcome(
                value=objective_value,
                suboptimality=suboptimality,
                dual_value=dual_value,
            ),
            duals=_duals_from(equality_mults, inequality_mults),
            diagnostics=HsdDiagnostics(
                duality_gaps=tuple(gap_history),
                tau_history=tuple(tau_history),
                kappa_history=tuple(kappa_history),
            ),
            certificate=certificate,
        )


class _HsdEmbedding:
    """Homogeneous self-dual embedding of a linear conic program.

    Holds the embedded iterate (x, y, τ, s, κ) for the program
       minimize c^T x   subject to   A x = b,   x ∈ K,
    and computes what the ``HsdSolver`` loop needs at each iterate: residuals, the
    complementarity gap, the centering right-hand sides, the Newton step, the step
    length, and infeasibility certificates.

    Following (Andersen, Roos, and Terlaky, 2003), the embedding seeks (x, y, τ, s, κ)
    with
       A x - b τ = 0,
       -A^T y + c τ - s = 0,
       b^T y - c^T x - κ = 0,
       x ∈ K,   s ∈ K,   τ ≥ 0,   κ ≥ 0,
    where s lies in K because each symmetric cone is self-dual. The system is
    homogeneous, and the interior-point method seeks a solution with τ κ = 0. When τ is
    positive there,
       x / τ   and   (y / τ, s / τ)
    are primal and dual optimal; when τ is zero and κ positive, y or x is a certificate
    of infeasibility or unboundedness.

    The starting point (``initial_iterate``) is strictly inside the cones but does not
    in general satisfy the linear equations; a Newton step of length α scales their
    residuals by 1 - α.

    Private to the package; ``second_order_cone_programs`` subclasses it with a
    structured kernel.

    Parameters
    ----------
     problem : Problem
        The program. Its objective must be a ``LinearObjective``, whose gradient is c,
        or the constructor raises ``ValueError``. Its equality constraints, if any, give
        (A, b), with p rows; without them p = 0. Its inequality layers are not read.
     cones : list[Cone]
        The cone blocks of K, in the order their coordinates appear in x and s. Their
        dimensions must sum to n = ``problem.dimension``, or the constructor raises
        ``ValueError``.

    References
    ----------
    Andersen, Erling D., Roos, Cornelis and Terlaky, Tamás, On Implementing a
    Primal-Dual Interior-Point Method for Conic Quadratic Optimization, Mathematical
    Programming 95, 2003.

    """

    def __init__(
        self,
        problem: Problem,
        cones: list[Cone],
    ) -> None:
        self._problem: Problem = problem
        self._cones: list[Cone] = cones
        # Validate cone dimensions sum to the problem dimension.
        cone_dim_sum = sum(c.dimension() for c in cones)
        if cone_dim_sum != problem.dimension:
            raise ValueError(
                f"Sum of cone dimensions ({cone_dim_sum}) must equal "
                f"problem.dimension ({problem.dimension})."
            )
        # Cache slice indices for per-cone block access.
        self._cone_slices: list[slice] = []
        offset = 0
        for cone in cones:
            d = cone.dimension()
            self._cone_slices.append(slice(offset, offset + d))
            offset += d
        # Cache the linear objective coefficient ``c``. HSD's canonical
        # form is ``min c^T x``; for non-linear ``Objective`` subclasses
        # this would need reformulation, deferred until a use case
        # arises. Using ``gradient(zeros)`` works for ``LinearObjective``
        # (returns ``c``) and would silently misuse ``QuadraticObjective``
        # at ``w = 0`` -- explicit subclass check pins the contract.
        if not isinstance(problem.obj, LinearObjective):
            raise ValueError(
                "HSD requires a LinearObjective (canonical form min c^T x); "
                f"got {type(problem.obj).__name__}. Reformulating "
                "QuadraticObjective into SOCP form is a future enhancement."
            )
        self._c: npt.NDArray[np.float64] = np.asarray(
            problem.obj.gradient(np.zeros(problem.dimension)),
            dtype=np.float64,
        )
        # Cache equality system (A, b) or empty placeholders.
        if problem.eq is not None:
            self._A: npt.NDArray[np.float64] = np.asarray(
                problem.eq.A, dtype=np.float64
            )
            self._b: npt.NDArray[np.float64] = np.asarray(
                problem.eq.b, dtype=np.float64
            )
        else:
            self._A = np.zeros((0, problem.dimension), dtype=np.float64)
            self._b = np.zeros(0, dtype=np.float64)
        self._num_eq: int = self._A.shape[0]
        # Embedded state populated by ``initial_iterate``.
        self._x: npt.NDArray[np.float64] | None = None
        self._y: npt.NDArray[np.float64] | None = None
        self._tau: float | None = None
        self._s: npt.NDArray[np.float64] | None = None
        self._kappa: float | None = None

    def initial_iterate(self) -> None:
        """Set the iterate to the canonical strictly interior starting point.

        Sets
           x = s = e_K,   y = 0,   τ = κ = 1,
        where e_K concatenates the cones' interior points and y has length p. Starting s
        at e_K is valid because each symmetric cone is self-dual, so e_K is interior to
        the dual cone too.

        """
        x = np.concatenate([cone.interior_point() for cone in self._cones]).astype(
            np.float64, copy=False
        )
        self._x = x
        self._s = x.copy()
        self._y = np.zeros(self._num_eq, dtype=np.float64)
        self._tau = 1.0
        self._kappa = 1.0

    def residuals(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],  # r_p (primal residual)
        npt.NDArray[np.float64],  # r_d (dual residual)
        float,  # r_g (gap residual)
        npt.NDArray[np.float64],  # r_xs (cone complementarity residual)
        float,  # r_tk (\tau \kappa residual)
    ]:
        """Compute the residuals of the embedded system at the current iterate.

        Uses the signs
           r_p  = b τ - A x,
           r_d  = A^T y + s - c τ,
           r_g  = c^T x - b^T y + κ,
           r_xs = x ∘ s   (per cone block),
           r_tk = τ κ.
        The first three vanish when the linear equations hold, and the last two at a
        complementary solution.

        Returns
        -------
         r_p : npt.NDArray[np.float64]
            Primal residual, of length p.
         r_d : npt.NDArray[np.float64]
            Dual residual, of length n.
         r_g : float
            Gap residual.
         r_xs : npt.NDArray[np.float64]
            Per-block Jordan products x ∘ s, of length n.
         r_tk : float
            Homogenizing complementarity τ κ.

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        # mypy narrowing: re-bind to local names that are non-Optional.
        x = self._x
        s = self._s
        y = self._y
        tau = self._tau
        kappa = self._kappa
        assert s is not None and y is not None and tau is not None and kappa is not None
        r_p = -self._A @ x + self._b * tau
        r_d = self._A.T @ y + s - self._c * tau
        r_g = -float(self._b @ y) + float(self._c @ x) + kappa
        # Per-cone Jordan products, stacked back into the embedding-space layout.
        r_xs = np.empty_like(x)
        for cone, sl in zip(self._cones, self._cone_slices, strict=False):
            r_xs[sl] = cone.jordan_product(x[sl], s[sl])
        r_tk = tau * kappa
        return r_p, r_d, r_g, r_xs, r_tk

    def kkt_solve(
        self,
        rhs_p: npt.NDArray[np.float64],
        rhs_d: npt.NDArray[np.float64],
        rhs_g: float,
        rhs_xs: npt.NDArray[np.float64],
        rhs_tk: float,
    ) -> tuple[
        npt.NDArray[np.float64],  # \Delta x
        npt.NDArray[np.float64],  # \Delta y
        float,  # \Delta \tau
        npt.NDArray[np.float64],  # \Delta s
        float,  # \Delta \kappa
    ]:
        """Solve the embedded Newton system by reduction to normal equations.

        Solves
           (I)    A Δx - b Δτ           = rhs_p,
           (II)   -A^T Δy - Δs + c Δτ   = rhs_d,
           (III)  b^T Δy - c^T Δx - Δκ  = rhs_g,
           (IV)   W^{-2} Δx + Δs        = rhs_xs,
           (V)    κ Δτ + τ Δκ           = rhs_tk,
        where W is the block-diagonal Nesterov-Todd scaling at the current (x, s), with
        W^2 s = x. Equation (IV) is the scaled linearized complementarity condition, and
        ``compute_centering_rhs`` builds its right-hand side. The step is returned, not
        applied.

        Parameters
        ----------
         rhs_p : npt.NDArray[np.float64]
            Right-hand side of (I), of length p.
         rhs_d : npt.NDArray[np.float64]
            Right-hand side of (II), of length n.
         rhs_g : float
            Right-hand side of (III).
         rhs_xs : npt.NDArray[np.float64]
            Right-hand side of (IV), of length n.
         rhs_tk : float
            Right-hand side of (V).

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            Step Δx, of length n.
         delta_y : npt.NDArray[np.float64]
            Step Δy, of length p.
         delta_tau : float
            Step Δτ.
         delta_s : npt.NDArray[np.float64]
            Step Δs, of length n.
         delta_kappa : float
            Step Δκ.

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.
         ConeBoundaryError
            Propagated from ``Cone.prepare_w_squared`` when x or s is not strictly
            inside its cone.
         np.linalg.LinAlgError
            When the reduced system is singular.

        Notes
        -----
        Equations (II), (V) and (IV) give
           Δs = -A^T Δy + c Δτ - rhs_d,
           Δκ = (rhs_tk - κ Δτ) / τ,
           Δx = W^2 (A^T Δy - c Δτ + rhs_d + rhs_xs).
        Substituting into (I) and (III) leaves the (p + 1) × (p + 1) system
           [ M     -u ] [ Δy ]   [ rhs_p - A W^2 (rhs_d + rhs_xs)               ]
           [ v^T    ω ] [ Δτ ] = [ rhs_g + c^T W^2 (rhs_d + rhs_xs) + rhs_tk / τ ]
        with
           M = A W^2 A^T,       u = A W^2 c + b,
           v = b - A W^2 c,     ω = c^T W^2 c + κ / τ
        (ω is ``sigma`` in the code, unrelated to the centering parameter). The
        homogenizing row and column make the system nonsymmetric. Each cone contributes
        its blocks of M, A W^2 c and c^T W^2 c through
        ``Cone.accumulate_normal_equations``, and products W^2 v go through
        ``Cone.apply_w_squared``, so W^2 is never formed densely. Assembling M costs
        O(p^2 n), the dense solve O(p^3), and the remaining products O(p n).

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        x = self._x
        s = self._s
        tau = self._tau
        kappa = self._kappa
        assert s is not None and tau is not None and kappa is not None

        p = self._num_eq
        A = self._A
        b = self._b
        c = self._c

        # Precompute the per-cone NT factors once per kkt_solve. The
        # apply_w_squared closure fires twice (lines below: ``Wp = ...``
        # and ``delta_x = ...``) and accumulate_normal_equations fires
        # once, so this caches the per-cone (scale, y) computation across
        # those three call-sites.
        nt_factors = [
            cone.prepare_w_squared(x[sl], s[sl])
            for cone, sl in zip(self._cones, self._cone_slices, strict=False)
        ]

        def apply_w_squared(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Multiply a vector by the block-diagonal W^2.

            Applies each cone's ``apply_w_squared`` to its block, using the factors
            precomputed for this solve.

            Parameters
            ----------
             v : npt.NDArray[np.float64]
                Vector of length n to multiply.

            Returns
            -------
             W2v : npt.NDArray[np.float64]
                The product W^2 v, of length n.

            """
            out = np.empty_like(v)
            for cone, sl, nt in zip(
                self._cones, self._cone_slices, nt_factors, strict=False
            ):
                out[sl] = cone.apply_w_squared(nt, v[sl])
            return out

        M = np.zeros((p, p), dtype=np.float64)
        AW2c = np.zeros(p, dtype=np.float64)
        cW2c = 0.0
        for cone, sl, nt in zip(
            self._cones, self._cone_slices, nt_factors, strict=False
        ):
            M_b, AW2c_b, cW2c_b = cone.accumulate_normal_equations(nt, A[:, sl], c[sl])
            M += M_b
            AW2c += AW2c_b
            cW2c += cW2c_b

        u = AW2c + b  # (p,)
        v = b - AW2c  # (p,)
        sigma = cW2c + kappa / tau

        p_term = rhs_d + rhs_xs  # (n,)
        Wp = apply_w_squared(p_term)
        rhs_y = rhs_p - A @ Wp  # (p,)
        rhs_tau = rhs_g + float(c @ Wp) + rhs_tk / tau

        # Assemble and solve the (p+1) x (p+1) reduced system.
        K = np.zeros((p + 1, p + 1), dtype=np.float64)
        K[:p, :p] = M
        K[:p, p] = -u
        K[p, :p] = v
        K[p, p] = sigma
        rhs_red = np.empty(p + 1, dtype=np.float64)
        rhs_red[:p] = rhs_y
        rhs_red[p] = rhs_tau
        sol = np.asarray(np.linalg.solve(K, rhs_red), dtype=np.float64)
        delta_y = sol[:p]
        delta_tau = float(sol[p])

        # Back-substitute.
        delta_x = apply_w_squared(A.T @ delta_y - c * delta_tau + p_term)
        delta_s = -A.T @ delta_y + c * delta_tau - rhs_d
        delta_kappa = (rhs_tk - kappa * delta_tau) / tau

        return delta_x, delta_y, delta_tau, delta_s, delta_kappa

    def complementarity_gap(self) -> float:
        """Compute the normalized complementarity gap μ at the current iterate.

        Computes
           μ = (∑_b x_b^T s_b + τ κ) / (n + 1),
        where the sum runs over the cone blocks; the extra 1 in the denominator counts
        the homogenizing pair. μ goes to zero as the iterate approaches a complementary
        solution.

        Returns
        -------
         mu : float
            The normalized gap.

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        x = self._x
        s = self._s
        tau = self._tau
        kappa = self._kappa
        assert s is not None and tau is not None and kappa is not None

        cone_sum = 0.0
        for sl in self._cone_slices:
            cone_sum += float(np.dot(x[sl], s[sl]))
        return (cone_sum + tau * kappa) / (self._problem.dimension + 1)

    def compute_centering_rhs(
        self,
        sigma: float,
        mu: float,
        delta_x_aff: npt.NDArray[np.float64] | None = None,
        delta_s_aff: npt.NDArray[np.float64] | None = None,
        delta_tau_aff: float | None = None,
        delta_kappa_aff: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Build the centering right-hand sides for the complementarity equations.

        Returns rhs_xs for equation (IV) of ``kkt_solve`` and rhs_tk for equation (V).
        For each cone block, with the Nesterov-Todd scaling W at (x_b, s_b) and the
        scaled point
           v = (W^{-1} x_b + W s_b) / 2,
        which equals W s_b = W^{-1} x_b since W^2 s_b = x_b, it forms the target
           t = σ μ e - v ∘ v
        and returns
           rhs_xs = 2 W^{-1} L(v)^{-1} t,
        where e is the block's ``interior_point`` and L(v) its arrow operator, inverted
        by ``solve_jordan_arrow``. For the homogenizing pair it returns
           rhs_tk = σ μ - τ κ.

        Without affine deltas, these are the pure centering right-hand sides that the
        ``HsdSolver`` loop uses. With all four affine deltas, it adds second-order
        corrector terms in the style of Mehrotra: t gains -Δv_aff ∘ Δv_aff, where
           Δv_aff = (W^{-1} Δx_aff + W Δs_aff) / 2,
        and rhs_tk gains -Δτ_aff Δκ_aff. The four must be given together, since a
        partial set would silently drop terms.

        Parameters
        ----------
         sigma : float
            Centering parameter σ.
         mu : float
            Complementarity gap at the current iterate.
         delta_x_aff : npt.NDArray[np.float64] | None
            Affine step Δx_aff, of length n, or ``None`` for pure centering.
         delta_s_aff : npt.NDArray[np.float64] | None
            Affine step Δs_aff, of length n, or ``None``.
         delta_tau_aff : float | None
            Affine step Δτ_aff, or ``None``.
         delta_kappa_aff : float | None
            Affine step Δκ_aff, or ``None``.

        Returns
        -------
         rhs_xs : npt.NDArray[np.float64]
            Right-hand side of equation (IV), of length n.
         rhs_tk : float
            Right-hand side of equation (V).

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.
         ValueError
            When some, but not all, of the affine deltas are given.
         ConeBoundaryError
            Propagated from ``Cone.nt_scaling`` when x or s is not strictly inside its
            cone.

        Notes
        -----
        The factor 2 doubles the complementarity right-hand side relative to Newton's
        method. For an orthant block, where W = diag(sqrt(x / s)) and v = sqrt(x s),
        equation (IV) with this right-hand side reads
           s ∘ Δx + x ∘ Δs = 2 (σ μ e - x ∘ s),
        whereas the Newton linearization of x ∘ s = σ μ e has the undoubled right-hand
        side
           σ μ e - x ∘ s,
        as the homogenizing equation (V) has for τ κ.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        x = self._x
        s = self._s
        tau = self._tau
        kappa = self._kappa
        assert s is not None and tau is not None and kappa is not None

        # All four affine deltas must be passed together (Mehrotra corrector)
        # or all left as ``None`` (predictor / pure centering). Mixing produces
        # a silently inconsistent corrector RHS: omitting any one defaults its
        # contribution to ``0``, which is mathematically wrong, not a no-op.
        aff_args = (delta_x_aff, delta_s_aff, delta_tau_aff, delta_kappa_aff)
        provided = [arg is not None for arg in aff_args]
        if any(provided) and not all(provided):
            missing = [
                name
                for name, ok in zip(
                    (
                        "delta_x_aff",
                        "delta_s_aff",
                        "delta_tau_aff",
                        "delta_kappa_aff",
                    ),
                    provided,
                    strict=False,
                )
                if not ok
            ]
            raise ValueError(
                "Mehrotra corrector requires all four affine deltas; "
                f"missing: {missing}."
            )
        use_mehrotra = all(provided)

        rhs_xs = np.empty(self._problem.dimension, dtype=np.float64)
        for cone, sl in zip(self._cones, self._cone_slices, strict=False):
            x_b = x[sl]
            s_b = s[sl]
            scaling = cone.nt_scaling(x_b, s_b)
            v_b = 0.5 * (scaling.apply_W_inv(x_b) + scaling.apply_W(s_b))
            v_jordan_v = cone.jordan_product(v_b, v_b)
            e_b = cone.interior_point()
            target = sigma * mu * e_b - v_jordan_v
            if use_mehrotra:
                assert delta_x_aff is not None and delta_s_aff is not None
                dv_aff = 0.5 * (
                    scaling.apply_W_inv(delta_x_aff[sl])
                    + scaling.apply_W(delta_s_aff[sl])
                )
                target = target - cone.jordan_product(dv_aff, dv_aff)
            inner = cone.solve_jordan_arrow(v_b, target)
            rhs_xs[sl] = 2.0 * scaling.apply_W_inv(inner)
        rhs_tk = sigma * mu - tau * kappa
        if use_mehrotra:
            assert delta_tau_aff is not None and delta_kappa_aff is not None
            rhs_tk = rhs_tk - delta_tau_aff * delta_kappa_aff
        return rhs_xs, rhs_tk

    def predicted_mu(
        self,
        alpha: float,
        delta_x: npt.NDArray[np.float64],
        delta_s: npt.NDArray[np.float64],
        delta_tau: float,
        delta_kappa: float,
    ) -> float:
        """Compute the complementarity gap a step would produce, without taking it.

        Evaluates the formula of ``complementarity_gap`` at
           (x + α Δx, s + α Δs, τ + α Δτ, κ + α Δκ).
        It supports Mehrotra's adaptive centering rule σ = (μ_aff / μ)^3, with μ_aff the
        gap after the affine step; the ``HsdSolver`` loop uses a fixed σ and does not
        call it.

        Parameters
        ----------
         alpha : float
            Step length α.
         delta_x : npt.NDArray[np.float64]
            Step Δx, of length n.
         delta_s : npt.NDArray[np.float64]
            Step Δs, of length n.
         delta_tau : float
            Step Δτ.
         delta_kappa : float
            Step Δκ.

        Returns
        -------
         mu_new : float
            The normalized gap at the stepped iterate.

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        x = self._x
        s = self._s
        tau = self._tau
        kappa = self._kappa
        assert s is not None and tau is not None and kappa is not None

        cone_sum = 0.0
        for sl in self._cone_slices:
            x_new = x[sl] + alpha * delta_x[sl]
            s_new = s[sl] + alpha * delta_s[sl]
            cone_sum += float(np.dot(x_new, s_new))
        tau_new = tau + alpha * delta_tau
        kappa_new = kappa + alpha * delta_kappa
        return (cone_sum + tau_new * kappa_new) / (self._problem.dimension + 1)

    def compute_step_length(
        self,
        delta_x: npt.NDArray[np.float64],
        delta_s: npt.NDArray[np.float64],
        delta_tau: float,
        delta_kappa: float,
    ) -> float:
        """Find a step length that keeps the iterate strictly inside the cones.

        Takes the largest α ≤ 1 that keeps every x and s block in its closed cone
        (``Cone.step_to_boundary``) and keeps τ + α Δτ ≥ 0 and κ + α Δκ ≥ 0, then scales
        it by 0.99 so that the next iterate stays strictly interior.

        Parameters
        ----------
         delta_x : npt.NDArray[np.float64]
            Step Δx, of length n.
         delta_s : npt.NDArray[np.float64]
            Step Δs, of length n.
         delta_tau : float
            Step Δτ.
         delta_kappa : float
            Step Δκ.

        Returns
        -------
         alpha : float
            The damped step length, in [0, 0.99].

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.
         ValueError
            Propagated from ``Cone.step_to_boundary`` on a shape mismatch.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        x = self._x
        s = self._s
        tau = self._tau
        kappa = self._kappa
        assert s is not None and tau is not None and kappa is not None

        alpha = 1.0
        for cone, sl in zip(self._cones, self._cone_slices, strict=False):
            alpha = min(alpha, cone.step_to_boundary(x[sl], delta_x[sl]))
            alpha = min(alpha, cone.step_to_boundary(s[sl], delta_s[sl]))
        if delta_tau < 0.0:
            alpha = min(alpha, -tau / delta_tau)
        if delta_kappa < 0.0:
            alpha = min(alpha, -kappa / delta_kappa)
        return 0.99 * alpha

    def detect_infeasibility(
        self,
        ratio_threshold: float = 0.9,
        cert_eps: float = 1e-8,
    ) -> tuple[
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
        str,
    ]:
        """Look for an infeasibility or unboundedness certificate in the iterate.

        At a solution of the embedding τ κ = 0. When τ > 0 the program has a solution;
        when τ collapses while κ stays positive, the iterate is a certificate. The test
        uses the ratio κ / (τ + κ), near 0 at an optimum and near 1 at a certificate.
        When the ratio exceeds ``ratio_threshold``:
        - b^T y > ``cert_eps`` makes y a certificate of primal infeasibility. As
          τ → 0 the dual equation gives A^T y + s = 0 with s ∈ K, so any feasible x
          would give b^T y = x^T A^T y = -x^T s ≤ 0.
        - Otherwise, c^T x < -``cert_eps`` makes x a certificate of dual
          infeasibility (the program is unbounded if feasible). As τ → 0, A x = 0
          with x ∈ K, so x is a recession direction along which the objective
          decreases.
        Only the ratio and these sign tests are checked; the equations hold only
        asymptotically and are not verified.

        Parameters
        ----------
         ratio_threshold : float
            Threshold on κ / (τ + κ) above which τ is treated as collapsed.
         cert_eps : float
            Margin for the sign tests on b^T y and c^T x.

        Returns
        -------
         primal_cert : npt.NDArray[np.float64] | None
            A copy of y, of length p, when it certifies primal infeasibility; otherwise
            ``None``.
         dual_cert : npt.NDArray[np.float64] | None
            A copy of x, of length n, when it certifies dual infeasibility; otherwise
            ``None``. At most one of the two certificates is returned.
         message : str
            Description of the finding: empty when the ratio is at most
            ``ratio_threshold``, and a diagnostic without a certificate when the ratio
            is high but neither sign test passes.

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        x = self._x
        y = self._y
        tau = self._tau
        kappa = self._kappa
        assert y is not None and tau is not None and kappa is not None

        ratio = kappa / (tau + kappa + 1e-300)
        if ratio <= ratio_threshold:
            return None, None, ""

        b_dot_y = float(self._b @ y)
        c_dot_x = float(self._c @ x)
        if b_dot_y > cert_eps:
            return (
                y.copy(),
                None,
                f"HSD detected primal infeasibility (kappa/(tau+kappa)"
                f"={ratio:.3f} > {ratio_threshold}, b^T y={b_dot_y:.3e} > 0). "
                "y stored in primal_infeasibility_certificate.",
            )
        if c_dot_x < -cert_eps:
            return (
                None,
                x.copy(),
                f"HSD detected dual infeasibility / primal unboundedness "
                f"(kappa/(tau+kappa)={ratio:.3f} > {ratio_threshold}, "
                f"c^T x={c_dot_x:.3e} < 0). x stored in "
                "dual_infeasibility_certificate.",
            )
        return (
            None,
            None,
            f"HSD found tau-collapse direction "
            f"(kappa/(tau+kappa)={ratio:.3f}) but neither b^T y > "
            f"{cert_eps} nor c^T x < -{cert_eps}. Problem likely "
            "marginally infeasible; no certificate populated.",
        )

    def advance_iterate(
        self,
        alpha: float,
        delta_x: npt.NDArray[np.float64],
        delta_y: npt.NDArray[np.float64],
        delta_tau: float,
        delta_s: npt.NDArray[np.float64],
        delta_kappa: float,
    ) -> None:
        """Advance the embedded iterate in place by α times a step.

        Parameters
        ----------
         alpha : float
            Step length α.
         delta_x : npt.NDArray[np.float64]
            Step Δx, of length n.
         delta_y : npt.NDArray[np.float64]
            Step Δy, of length p.
         delta_tau : float
            Step Δτ.
         delta_s : npt.NDArray[np.float64]
            Step Δs, of length n.
         delta_kappa : float
            Step Δκ.

        Raises
        ------
         RuntimeError
            When ``initial_iterate`` has not been called.

        """
        if self._x is None:
            raise RuntimeError(
                "Embedded state not initialized; call initial_iterate() first."
            )
        assert (
            self._s is not None
            and self._y is not None
            and self._tau is not None
            and self._kappa is not None
        )
        self._x += alpha * delta_x
        self._y += alpha * delta_y
        self._tau += alpha * delta_tau
        self._s += alpha * delta_s
        self._kappa += alpha * delta_kappa
