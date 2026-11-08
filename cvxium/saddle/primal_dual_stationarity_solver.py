"""Mehrotra predictor-corrector primal-dual solver for two-block stationarity problems.

Defines ``PrimalDualStationaritySolver``, the primal-dual sibling of the barrier engine
``StationaritySolver``. Both conform to the ``TwoBlockStationaritySolver`` protocol,
``solve(problem, x0, y0) -> TwoBlockSolveResult``, so a downstream driver or frontier
wrapper can hold either without knowing which engine it drives. Where the barrier engine
nests an inner centering inside an outer loop that raises t, this engine follows the
central path directly: it carries explicit per-block inequality duals λ^x and λ^y, and
takes one Mehrotra predictor-corrector step per iteration on the coupled saddle, driving
the surrogate gap
   η = s^x · λ^x + s^y · λ^y
and the stationarity residual to zero together.

One iteration. The slacks s^x = -f^x(x) and s^y = -f^y(y) are implicit (Boyd and
Vandenberghe, 2004, section 11.7). The stationarity residual (r_x, r_y) comes from
``StationarityProblem.primal_dual_residual``, and the coupled saddle
   [[-M_xx, B], [B^T, H_yy]]
from ``StationarityProblem.build_primal_dual_coupled_system``. Eliminating the
complementarity rows S Λ 1 = μ 1 gives the reduced right-hand side
   b_x = -r_x - J_x^T (r_cx / s^x),   b_y = -r_y + J_y^T (r_cy / s^y),
which the saddle solves; the dual and slack steps follow as
   Δλ^x = S_x^{-1} (-r_cx + Λ^x J_x Δx),   Δs^x = -J_x Δx
(mirrored for y). The concave x-block subtracts its constraint force and the convex
y-block adds it, matching the sign convention of ``primal_dual_residual``. An affine
predictor (r_c = s ∘ λ) sets a per-block Mehrotra centering
   σ_x = (μ^x_aff / μ_x)^3,   σ_y = (μ^y_aff / μ_y)^3,
where each block uses its own surrogate gap and σ is floored (``_SIGMA_FLOOR``), so a
block-imbalanced saddle still recenters both blocks. The corrector recenters each block
to its own σ μ and carries the second-order term Δs_aff ∘ Δλ_aff. One factorization
backs both solves.

Equality path. A ``StationarityProblem`` with an ``x_equality`` A_eq x = b_eq takes the
bordered path: the residual gains the primal-feasibility row r_eq = A_eq x - b_eq and an
A_eq^T ν term in r_x, and the coupled saddle borders to ``EqualityCoupledSaddleSystem``,
whose ``solve(b_x, b_y, b_eq)`` returns the equality-multiplier step Δν alongside the
primal steps. The reduced x and y right-hand sides are unchanged, and b_eq = -r_eq. The
multiplier ν is free (unsigned), so it takes the x-block step length with no
fraction-to-boundary cap or clamp.

There is no Levenberg damping (unlike the single-block ``PrimalDualSolver``):
``QuadraticSaddleObjective`` and barrier-domain problems are well conditioned by
construction, and a singular reduced saddle surfaces as a ``NewtonStepError`` from the
coupled solve. The start-resolution and inertia helpers are module-level mirrors of
``StationaritySolver``'s methods.

References
----------
Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University Press,
2004.

Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method, SIAM
Journal on Optimization 2, 1992.

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import numpy.typing as npt

from ..exceptions import (
    BacktrackingLineSearchError,
    NewtonStepError,
    OptimizationError,
    ProblemInfeasibleError,
)
from ..linalg.systems import (
    CoupledSaddleSystem,
    EqualityCoupledSaddleSystem,
)
from ..problems.problem import Problem
from ..problems.result import (
    Duals,
    ObjectiveOutcome,
    PrimalDualDiagnostics,
    SolveStatus,
)
from ..saddle.stationarity import (
    StationarityConvergenceError,
    StationarityProblem,
    TwoBlockSolveResult,
    two_block_result_message,
)
from ..solvers.barrier_solver import (
    BarrierSolver,
)
from ..solvers.line_search import (
    fraction_to_boundary,
    mehrotra_boundary_factor,
    mehrotra_step_cap,
)
from ..solvers.settings import (
    OptimizationSettings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..problems.constraint_layers import (
        ConstraintLayer,
    )
    from ..problems.linear_constraints import (
        EqualityConstraintLayer,
    )

# Strictly-positive floor for the inequality duals. A primal-dual method keeps
# lambda in the interior of the dual cone; the Mehrotra cap guarantees
# ``lam + step * dlam > 0`` in exact arithmetic, but the boundary factor relaxes
# to 1.0 in the convergence tail, so a binding multiplier can undershoot to <= 0
# by a machine epsilon. Clamping to exactly 0.0 zeros the halfspace's rank-one
# Hessian weight ``d = lam / s`` and trips ``_prepare_rank_one_update``'s
# ``d must be nonzero`` guard; flooring here instead keeps d nonzero. Set far
# below any genuine binding dual (duals initialize at 1.0), so it fires only on
# machine-epsilon leaks and does not perturb the optimum.
_DUAL_FLOOR: float = 1e-12

# Lower bound on the per-block Mehrotra centering parameter sigma. A block whose
# affine step nearly zeroes its own complementarity produces a tiny
# ``(mu_aff / mu)**3`` ratio; without a floor its corrector aims at ~0, so the
# block loses its recentering and can stall against the other block's scale.
# Flooring each sigma keeps a minimum recentering pull in every block (a
# per-block analog of the classic Mehrotra sigma safeguard).
_SIGMA_FLOOR: float = 0.2

# Surrogate-gap threshold below which the endgame residual line search engages.
# Above it the method is still path-following -- the raw KKT residual legitimately
# rises during predictor-corrector -- so a residual-descent merit would fight the
# healthy early phase and must stay off. See ``_endgame_step``.
_MERIT_ENDGAME_GAP: float = 1e-3


@dataclass(frozen=True)
class _Iterate:
    """A strictly interior primal-dual iterate and its KKT score.

    The solve loops keep the best-scoring iterate as the fallback for the soft tolerance
    and for a raised non-convergence. A binding constraint can drive its slack to
    machine zero, after which the affine predictor diverges and later iterates are worse
    than an earlier near-solution, so the final iterate is not a safe fallback. The
    constructor copies every array, so a snapshot cannot alias the loop's state.

    Fields
    ------
     score : float
        The KKT score max(η, ‖r_stat‖, ‖r_eq‖): the quantities the strict convergence
        test uses, collapsed to one number (‖r_eq‖ is absent on the no-equality path).
     x : npt.NDArray[np.float64]
        The x-block iterate, of length n.
     y : npt.NDArray[np.float64]
        The y-block iterate, of length m.
     lam_x : npt.NDArray[np.float64]
        The x-domain inequality duals, of length p_x.
     lam_y : npt.NDArray[np.float64]
        The y-domain inequality duals, of length p_y.
     nu : npt.NDArray[np.float64]
        The equality multipliers (empty on the no-equality path).

    """

    score: float
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    lam_x: npt.NDArray[np.float64]
    lam_y: npt.NDArray[np.float64]
    nu: npt.NDArray[np.float64]

    def __init__(
        self,
        score: float,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
    ) -> None:
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "lam_x", lam_x.copy())
        object.__setattr__(self, "lam_y", lam_y.copy())
        object.__setattr__(self, "nu", nu.copy())


def _raise_nonconvergence(
    result: TwoBlockSolveResult,
    best: _Iterate | None,
    iterations: int,
    reason: str,
    cause: NewtonStepError | None = None,
) -> NoReturn:
    """Raise ``StationarityConvergenceError`` carrying the best-effort iterate.

    The loops reach this only when neither the strict test nor the soft-tolerance
    fallback fired: the iteration budget ran out, a binding slack reached its boundary,
    or a Newton step could not be computed or could not stay strictly interior, all
    without reaching the user's tolerance. That is a non-convergence failure, so the
    engine raises rather than returning a ``MAX_ITERATIONS`` status the caller might
    overlook, as the barrier engine (``StationaritySolver._fail``) and
    ``InteriorPointMethodError`` do.

    The message reports the actual termination cause and the actual number of iterations
    run, not a hard-coded cap.

    Parameters
    ----------
     result : TwoBlockSolveResult
        The result at the best iterate seen (not the last, which a binding slack can
        drive strictly worse), with status ``MAX_ITERATIONS``; attached to the
        exception.
     best : _Iterate | None
        The best iterate seen, whose score becomes the exception's suboptimality;
        ``None`` (score inf) when no interior iterate was recorded.
     iterations : int
        Number of iterations run.
     reason : str
        The recorded termination cause.
     cause : NewtonStepError | None
        The ``NewtonStepError`` that stopped the loop, chained as the exception's cause,
        or ``None``.

    Raises
    ------
     StationarityConvergenceError
        Always.

    """
    score = best.score if best is not None else float("inf")
    message = (
        f"Primal-dual stationarity solve did not converge after {iterations} "
        f"iteration(s): {reason} (best KKT score {score:.03g}). The best iterate "
        "seen is attached to this exception's result."
    )
    raise StationarityConvergenceError(message, score, result) from cause


def _made_no_progress(exc: StationarityConvergenceError) -> bool:
    """Check whether a failed solve never improved on its own start point.

    The exception's ``suboptimality`` is the KKT score of the best strictly interior
    iterate seen, and the first entry of each diagnostic trajectory holds the
    ingredients of the start point's score. The run made progress only if its best score
    is strictly below the start's. A loop that never ran (an iteration budget of zero)
    made none.

    Parameters
    ----------
     exc : StationarityConvergenceError
        The non-convergence raised by a solve loop.

    Returns
    -------
     no_progress : bool
        True when the best score is not below the start point's score, or when the loop
        recorded no iterations.

    Notes
    -----
    An iteration count cannot stand in for this test. A seed that hugs a face leaves the
    first dual step on that row as a cancellation of two nearly equal terms divided by a
    slack of order 1e-12, so its sign is rounding noise. On some CPUs the first step is
    capped to nothing and the run stops at iteration 0; on others it takes one long
    step, blows the surrogate gap up by four orders of magnitude, and stops one
    iteration later. The two runs are equally empty (their best score is the start's),
    but an iteration count tells them apart and the score does not.

    """
    diagnostics = exc.result.primal_dual_diagnostics
    if not diagnostics.surrogate_gaps:
        return True
    start = max(
        diagnostics.surrogate_gaps[0],
        diagnostics.primal_residual_norms[0],
        diagnostics.dual_residual_norms[0],
    )
    # Not >=: a nan suboptimality counts as no progress.
    return not exc.suboptimality < start


def _legacy_status(status: SolveStatus) -> int:
    """Map a ``SolveStatus`` to the trace's integer status code.

    ``on_pd_done`` takes the integer convention the single-block ``PrimalDualSolver``
    also reports.

    Parameters
    ----------
     status : SolveStatus
        The solve's final status.

    Returns
    -------
     code : int
        0 for ``OPTIMAL``, 1 for ``MAX_ITERATIONS``, and 2 for any other status (here,
        ``ACCEPTABLE`` from the soft-tolerance fallback).

    """
    if status is SolveStatus.OPTIMAL:
        return 0
    if status is SolveStatus.MAX_ITERATIONS:
        return 1
    return 2


def _pd_reason(status: SolveStatus, stop_reason: str) -> str:
    """Return the human-readable termination cause for the trace's ``on_pd_done``.

    Only ``OPTIMAL`` has a cause the status alone determines. Every other exit is one of
    several breaks (the iteration budget, a slack that reached its boundary, a Newton
    step that could not stay strictly interior), which the loop distinguishes in
    ``stop_reason`` but which all share the ``MAX_ITERATIONS`` status. Reporting the
    status alone would label a start-point failure at iteration 0 "iteration budget
    exhausted" and send a reader off to raise a cap that was never reached, so the
    recorded reason is passed through instead. ``ACCEPTABLE`` keeps its own text and
    appends the stop cause: what the soft fallback rescued is worth knowing alongside
    why the strict test was never met.

    Parameters
    ----------
     status : SolveStatus
        The solve's final status.
     stop_reason : str
        The loop's recorded termination cause.

    Returns
    -------
     reason : str
        A fixed message for ``OPTIMAL``; the soft-tolerance message with ``stop_reason``
        appended for ``ACCEPTABLE``; ``stop_reason`` otherwise.

    """
    if status is SolveStatus.OPTIMAL:
        return "surrogate gap and residual within tolerance"
    if status is SolveStatus.ACCEPTABLE:
        return f"best iterate within soft tolerance; {stop_reason}"
    return stop_reason


def _pd_cause(status: SolveStatus, stop_reason: str) -> str | None:
    """Return the stop cause for ``two_block_result_message``, or None if implied.

    The shared formatter derives its wording from the status, which works for the two
    exits a status determines: ``OPTIMAL`` and the soft-tolerance ``ACCEPTABLE``. It
    cannot work for ``MAX_ITERATIONS``, the status every short exit shares, which the
    formatter would call an exhausted iteration budget even for a start-point failure at
    the first iteration. On that exit the loop's recorded ``stop_reason`` is reported
    instead.

    This is the correction ``_pd_reason`` makes for the trace, applied to the result's
    ``message``, which is what a caller that catches ``StationarityConvergenceError``
    reads from ``exc.result``.

    Parameters
    ----------
     status : SolveStatus
        The solve's final status.
     stop_reason : str
        The loop's recorded termination cause.

    Returns
    -------
     cause : str | None
        ``stop_reason`` for ``MAX_ITERATIONS``; ``None`` otherwise, so the formatter
        keeps its own text.

    """
    return stop_reason if status is SolveStatus.MAX_ITERATIONS else None


def _stacked_constraints(
    domain: Sequence[ConstraintLayer],
    w: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Stack the constraint values f(w) of every layer in a domain.

    Parameters
    ----------
     domain : Sequence[ConstraintLayer]
        The block's constraint layers; may be empty.
     w : npt.NDArray[np.float64]
        The block's point.

    Returns
    -------
     f : npt.NDArray[np.float64]
        The constraint values in layer order; empty for an empty domain.

    """
    if not domain:
        return np.zeros(0)
    return np.concatenate([layer.constraints(w) for layer in domain])


def _stacked_jac_vec(
    domain: Sequence[ConstraintLayer],
    w: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Multiply a vector by the stacked constraint Jacobian J of a domain.

    Row i of J is ∇f_i(w)^T, so the product is the G v of a single-block solver.

    Parameters
    ----------
     domain : Sequence[ConstraintLayer]
        The block's constraint layers; may be empty.
     w : npt.NDArray[np.float64]
        The block's point, at which J is evaluated.
     v : npt.NDArray[np.float64]
        Vector to multiply, of the block's dimension.

    Returns
    -------
     Jv : npt.NDArray[np.float64]
        The product J v in layer order; empty for an empty domain.

    """
    if not domain:
        return np.zeros(0)
    return np.concatenate([layer.grad_constraints_multiply(w, v) for layer in domain])


def _stacked_jac_transpose(
    domain: Sequence[ConstraintLayer],
    w: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Multiply a vector by the transposed stacked constraint Jacobian of a domain.

    Computes
       J^T u = ∑ u_i ∇f_i(w).

    Parameters
    ----------
     domain : Sequence[ConstraintLayer]
        The block's constraint layers; may be empty.
     w : npt.NDArray[np.float64]
        The block's point, at which J is evaluated.
     u : npt.NDArray[np.float64]
        Vector to multiply, one entry per constraint in layer order.

    Returns
    -------
     JTu : npt.NDArray[np.float64]
        The product J^T u, of the block's dimension; zero for an empty domain.

    """
    total = np.zeros_like(w)
    offset = 0
    for layer in domain:
        count = layer.num_constraints(w.shape[0])
        total = total + layer.grad_constraints_transpose_multiply(
            w, u[offset : offset + count]
        )
        offset += count
    return total


def _num_constraints(
    domain: Sequence[ConstraintLayer],
    dimension: int,
) -> int:
    """Count the inequality constraints across a domain.

    Parameters
    ----------
     domain : Sequence[ConstraintLayer]
        The block's constraint layers.
     dimension : int
        The block's dimension.

    Returns
    -------
     count : int
        The total constraint count, which is the length of the block's dual vector.

    """
    return sum(layer.num_constraints(dimension) for layer in domain)


def _strictly_interior(
    w: npt.NDArray[np.float64],
    domain: Sequence[ConstraintLayer],
) -> bool:
    """Test whether a point satisfies every layer of a domain strictly.

    Parameters
    ----------
     w : npt.NDArray[np.float64]
        The point to test.
     domain : Sequence[ConstraintLayer]
        The block's constraint layers.

    Returns
    -------
     interior : bool
        True when every constraint value is negative; vacuously True for an empty
        domain.

    """
    return all(bool(np.all(layer.constraints(w) < 0)) for layer in domain)


def _phase1_point(
    dimension: int,
    domain: Sequence[ConstraintLayer],
    equality: EqualityConstraintLayer | None,
    seed: npt.NDArray[np.float64] | None,
    name: str,
    settings: OptimizationSettings,
) -> npt.NDArray[np.float64]:
    """Find a point strictly interior to a domain, on the equality when one is set.

    Mirrors ``StationaritySolver._phase1_point``. An empty domain returns zeros.
    Otherwise an objective-free ``Problem`` (the domain, plus the equality when given)
    is solved with ``BarrierSolver``, which runs Cvxium's Phase I.

    Parameters
    ----------
     dimension : int
        The block's dimension.
     domain : Sequence[ConstraintLayer]
        The block's constraint layers.
     equality : EqualityConstraintLayer | None
        The block's equality constraints, or ``None``.
     seed : npt.NDArray[np.float64] | None
        Starting point for Phase I, or ``None``. Must be ``None`` for an empty domain
        (``_resolve_block`` returns any seed verbatim there).
     name : str
        Block name ("x" or "y") for the error message.
     settings : OptimizationSettings
        Settings for the Phase I ``BarrierSolver``.

    Returns
    -------
     w : npt.NDArray[np.float64]
        A strictly interior point, of length ``dimension``.

    Raises
    ------
     ProblemInfeasibleError
        When Phase I certifies the domain (with its equality) infeasible; re-raised with
        the block named, chaining the original.

    """
    if not domain:
        assert (
            seed is None
        ), "_resolve_block already returns a seed vacuously interior to an empty domain"
        return np.zeros(dimension, dtype=np.float64)
    problem = Problem(obj=None, eq=equality, ineq=list(domain))
    try:
        result = problem.solve(BarrierSolver(settings), x0=seed)
    except ProblemInfeasibleError as exc:
        raise ProblemInfeasibleError(
            f"{name}-block Phase I found no strictly-interior point; the "
            f"{name}_domain (with its equality, if any) appears infeasible."
        ) from exc
    return np.asarray(result.solution, dtype=np.float64)


def _resolve_block(
    w0: npt.NDArray[np.float64] | None,
    dimension: int,
    domain: Sequence[ConstraintLayer],
    equality: EqualityConstraintLayer | None,
    name: str,
    settings: OptimizationSettings,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Return a strictly interior start for one block and whether it is the seed.

    A supplied ``w0`` already strictly interior to ``domain`` is returned verbatim.
    Otherwise (``None``, or not interior) a Phase I solve produces one, seeded by ``w0``
    when supplied. Mirrors ``StationaritySolver._resolve_block``.

    Strict interiority is a weak guarantee: a seed can satisfy every f < 0 while sitting
    so close to one face that no first Newton step fits between it and the boundary.
    ``solve`` uses the flag to retry from a Phase I point only when the start was the
    caller's own.

    Parameters
    ----------
     w0 : npt.NDArray[np.float64] | None
        The caller's warm start for the block, or ``None``.
     dimension : int
        The block's dimension.
     domain : Sequence[ConstraintLayer]
        The block's constraint layers.
     equality : EqualityConstraintLayer | None
        The block's equality constraints, used by Phase I only; ``None`` for none.
     name : str
        Block name ("x" or "y") for error messages.
     settings : OptimizationSettings
        Settings for the Phase I solve.

    Returns
    -------
     w : npt.NDArray[np.float64]
        A strictly interior start, of length ``dimension``.
     verbatim : bool
        True when ``w`` is the caller's ``w0`` unchanged.

    Raises
    ------
     ValueError
        When ``w0`` does not have shape (``dimension``,).
     ProblemInfeasibleError
        Propagated from ``_phase1_point``.

    """
    seed = None if w0 is None else np.asarray(w0, dtype=np.float64)
    if seed is not None and seed.shape != (dimension,):
        raise ValueError(f"{name}0 has shape {seed.shape}, expected ({dimension},).")
    if seed is not None and _strictly_interior(seed, domain):
        return seed, True
    return _phase1_point(dimension, domain, equality, seed, name, settings), False


def _check_inertia(
    system: CoupledSaddleSystem | EqualityCoupledSaddleSystem,
) -> None:
    """Check the curvature signs of the coupled saddle once.

    The saddle [[-M_xx, B], [B^T, H_yy]] has the expected inertia if and only if both
    M_xx and H_yy are positive definite. Each block answers with its own structure-aware
    ``is_positive_definite`` test. Mirrors ``StationaritySolver._check_inertia``.

    Parameters
    ----------
     system : CoupledSaddleSystem | EqualityCoupledSaddleSystem
        The coupled saddle at the first iterate.

    Raises
    ------
     ValueError
        When M_xx or H_yy is not positive definite.

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


class PrimalDualStationaritySolver:
    """Mehrotra predictor-corrector primal-dual method for a two-block saddle.

    Drives a ``StationarityProblem`` (a smooth two-block objective ζ(x, y) with
    log-barrier domains, and optionally an equality on x) to a joint stationary point by
    following the central path with explicit per-block inequality duals. Conforms to the
    ``TwoBlockStationaritySolver`` protocol; the module docstring describes one
    iteration.

    Standalone rather than a ``Solver`` subclass, for the same reason as
    ``StationaritySolver``: the two-block problem and its (x0, y0) start do not fit the
    single-``Problem``, single-``x0`` signature of ``Solver.solve``.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings; ``None`` uses ``OptimizationSettings()``. The solver reads
        ``inner.max_iterations`` (iteration cap), ``outer.tolerance`` (strict tolerance
        on the surrogate gap and the residuals), ``outer.tolerance_soft``
        (soft-tolerance fallback), ``pd_fraction_to_boundary_min`` (floor of the
        Mehrotra boundary factor), ``armijo.beta`` and ``armijo.min_step`` (endgame
        backtracking and step floor), ``trace``, and ``observability.label`` (trace
        name). The settings also configure the Phase I ``BarrierSolver``.
     check_inertia : bool
        Whether to check the coupled saddle's inertia at the first iteration (see
        ``_check_inertia``).

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
        # Trace label: an explicit observability label wins, else the class name
        # (mirrors StationaritySolver and BaseInteriorPointMethodSolver).
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
        """Solve a two-block stationarity problem from optional per-block warm starts.

        A start already strictly interior to its domain is used as given; otherwise a
        per-block Phase I finds one (the x-block Phase I also lands on A_eq x = b_eq
        when an ``x_equality`` is set). The bordered ``_run_equality`` loop runs when an
        ``x_equality`` is present, and the plain ``_run`` loop otherwise.

        Strict interiority does not imply a usable start. A warm start can satisfy every
        f < 0 and still hug one face so closely that the first Mehrotra step cannot fit
        between it and the boundary, so the solve fails without improving on its start.
        When a run fails that way (``_made_no_progress``) and some block's start was the
        caller's own, ``solve`` replaces those blocks' starts with Phase I points
        (``_centered_start``) and runs once more. The retry is best-effort: if it fails
        in any way, the original failure is raised.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem, optionally with an ``x_equality``.
         x0 : npt.NDArray[np.float64] | None
            Warm start for the x-block, of length n; Phase I runs when it is ``None`` or
            not strictly interior.
         y0 : npt.NDArray[np.float64] | None
            Warm start for the y-block, of length m, with the same semantics.

        Returns
        -------
         result : TwoBlockSolveResult
            The iterate, iteration count, per-block duals, saddle value, surrogate-gap
            suboptimality and primal-dual diagnostics. Returned only with status
            ``OPTIMAL``, or ``ACCEPTABLE`` when the best iterate met the soft tolerance.

        Raises
        ------
         StationarityConvergenceError
            When the iteration fails to converge to the soft tolerance: an exhausted
            budget, a slack reaching its boundary, or a coupled system admitting no
            usable Newton step. The best iterate rides on the exception's ``result``
            (status ``MAX_ITERATIONS``), and a ``NewtonStepError`` that triggered the
            stop is chained as the cause.
         ProblemInfeasibleError
            When a block's domain admits no strictly interior point.
         ValueError
            When ``x0`` or ``y0`` has the wrong shape, or when the inertia check fails
            on the first run.

        """
        settings = self.settings
        x, x_verbatim = _resolve_block(
            x0, problem.dimension_x, problem.x_domain, problem.x_equality, "x", settings
        )
        y, y_verbatim = _resolve_block(
            y0, problem.dimension_y, problem.y_domain, None, "y", settings
        )
        try:
            return self._dispatch(problem, x, y)
        except StationarityConvergenceError as exc:
            # A run that never improved on its start point produced nothing, so
            # nothing is lost by trying again -- but only a start point we
            # accepted verbatim can be improved on. A Phase I point is already
            # centered, so re-deriving it would just repeat the same failure.
            if not (x_verbatim or y_verbatim):
                raise
            if not _made_no_progress(exc):
                raise
            try:
                cx, cy = self._centered_start(problem, x, y, x_verbatim, y_verbatim)
                return self._dispatch(problem, cx, cy)
            except (
                OptimizationError,
                ProblemInfeasibleError,
                NewtonStepError,
                BacktrackingLineSearchError,
                np.linalg.LinAlgError,
                ValueError,
            ) as retry_error:
                # The retry is strictly best-effort: it may turn a failure into a
                # success, never into a *different* failure. Phase I can decline
                # a domain the main loop accepted (an unattainable analytic
                # center, or a layer its BarrierSolver does not support), and the
                # centered run can fail on its own terms. Either way the caller
                # gets the diagnosis of the start point they actually supplied,
                # with the retry's failure chained as its cause.
                #
                # The tuple spans every failure mode reachable on this path, not
                # just the ``OptimizationError`` family: Phase I raises the
                # standalone ``ProblemInfeasibleError``, and a centered dispatch
                # can surface ``NewtonStepError``, a line-search error, or a
                # singular factorization. Any of those leaking out would break
                # the "never a different failure" contract this branch exists to
                # uphold.
                raise exc from retry_error

    def _centered_start(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_verbatim: bool,
        y_verbatim: bool,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Replace each caller-supplied block start with its Phase I point.

        Only the blocks flagged verbatim are replaced; a block whose start already came
        from Phase I is passed through unchanged. Phase I runs from no seed, so the
        caller's warm start is deliberately given up; the alternative on this path is a
        solve that has already failed.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block problem.
         x : npt.NDArray[np.float64]
            The x-block start of the failed run.
         y : npt.NDArray[np.float64]
            The y-block start of the failed run.
         x_verbatim : bool
            Whether ``x`` is the caller's own start.
         y_verbatim : bool
            Whether ``y`` is the caller's own start.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The x-block start for the retry.
         y : npt.NDArray[np.float64]
            The y-block start for the retry.

        Raises
        ------
         ProblemInfeasibleError
            Propagated from ``_phase1_point``.

        """
        settings = self.settings
        if x_verbatim:
            x = _phase1_point(
                problem.dimension_x,
                problem.x_domain,
                problem.x_equality,
                None,
                "x",
                settings,
            )
        if y_verbatim:
            y = _phase1_point(
                problem.dimension_y, problem.y_domain, None, None, "y", settings
            )
        return x, y

    def _dispatch(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> TwoBlockSolveResult:
        """Run the bordered or the plain iteration loop, as the problem requires.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block problem; ``_run_equality`` runs when it has an ``x_equality``.
         x : npt.NDArray[np.float64]
            Strictly interior x-block start.
         y : npt.NDArray[np.float64]
            Strictly interior y-block start.

        Returns
        -------
         result : TwoBlockSolveResult
            The result of the loop that ran.

        Raises
        ------
         StationarityConvergenceError
            Propagated from ``_run`` or ``_run_equality``.
         ValueError
            Propagated from the first-iteration inertia check.

        """
        if problem.x_equality is None:
            return self._run(problem, x, y)
        return self._run_equality(problem, x, y)

    def _run(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> TwoBlockSolveResult:
        """Run the Mehrotra predictor-corrector loop for a problem without equality.

        Starts from λ^x = 1 and λ^y = 1. Each iteration takes one predictor-corrector
        step on the coupled saddle (see the module docstring), with per-block step
        lengths and, in the endgame, a residual line search (``_endgame_step``).

        Parameters
        ----------
         problem : StationarityProblem
            The two-block problem, without an ``x_equality``.
         x : npt.NDArray[np.float64]
            Strictly interior x-block start, of length n.
         y : npt.NDArray[np.float64]
            Strictly interior y-block start, of length m.

        Returns
        -------
         result : TwoBlockSolveResult
            The result at the returned iterate, with status ``OPTIMAL`` or
            ``ACCEPTABLE``.

        Raises
        ------
         StationarityConvergenceError
            Through ``_raise_nonconvergence``, when no exit met the strict or soft
            tolerance.
         ValueError
            Propagated from ``_check_inertia`` at the first iteration.

        Notes
        -----
        Iteration details beyond the module docstring:
        - The y-block corrector target is floored at
             score / max(m_tot, n + m),
          where score = max(η, ‖r‖) and m_tot is the total constraint count. Without
          the floor, a curved y-block drives its direction onto the boundary, where the
          feasible step shrinks toward zero and the block freezes.
        - Each block's dual step is capped by ``mehrotra_step_cap`` with its own block
          gap, and its primal step by γ times the exact distance to its own boundary
          (``_block_max_step``), with the shared factor
             γ = max(τ_min, 1 - η / m_tot).
          The blocks' constraints are separable, so a throttled block does not clamp
          the other.
        The loop stops at the first of: a non-interior iterate; η ≤ tol and ‖r‖ ≤ tol
        (``OPTIMAL``); a ``NewtonStepError`` from the coupled solve; a block step at or
        below ``armijo.min_step``; or the iteration cap. After a non-optimal stop, the
        best interior iterate is returned as ``ACCEPTABLE`` when its score is within
        ``outer.tolerance_soft``, and otherwise attached to a raised
        ``StationarityConvergenceError``. The reported suboptimality is the surrogate
        gap recomputed at the returned iterate, floored at zero.

        References
        ----------
        Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method,
        SIAM Journal on Optimization 2, 1992.

        """
        settings = self.settings
        n = problem.dimension_x
        x_domain = problem.x_domain
        y_domain = problem.y_domain
        p_x = _num_constraints(x_domain, n)
        p_y = _num_constraints(y_domain, problem.dimension_y)
        # Total complementarity pairs; floored at 1 so the empty-domain (pure
        # Newton) case divides cleanly with mu == 0.
        m_tot = max(p_x + p_y, 1)
        # Divisor for the y-block central-path floor (see the ``r_cy`` comment in
        # the loop). ``score`` is a norm over the whole KKT residual -- stationarity
        # over the ``n + dimension_y`` primal DOF plus complementarity over
        # ``m_tot`` pairs -- so dividing it by the pair count alone over-weights the
        # floor when constraints are few relative to the primal variables. The
        # degenerate case is an unconstrained x-block: ``m_tot`` collapses toward 1
        # while the primal residual still spans ``n + dimension_y`` dimensions, so
        # the floor inflates, injects a spurious y-dual, and stalls. Flooring the
        # divisor at the primal DOF keeps it a per-DOF residual scale; whichever of
        # the two dimensions the residual actually spans dominates.
        score_dof = max(m_tot, n + problem.dimension_y)
        # Per-block pair counts (each floored at 1) for the per-block surrogate
        # gaps mu_x, mu_y used in centering below.
        m_x = max(p_x, 1)
        m_y = max(p_y, 1)

        max_iter = settings.inner.max_iterations
        tol = settings.outer.tolerance
        soft = settings.outer.tolerance_soft
        tau_min = settings.pd_fraction_to_boundary_min
        beta = settings.armijo.beta
        min_step = settings.armijo.min_step

        lam_x = np.ones(p_x)
        lam_y = np.ones(p_y)
        no_eq = np.zeros(0)

        # Per-iteration convergence trajectories, one entry per loop-body entry
        # (recorded at the top of the loop, so the final OPTIMAL/break iteration
        # is captured and ``len == nit + 1 == n_iterations``).
        surrogate_gaps: list[float] = []
        primal_residual_norms: list[float] = []
        dual_residual_norms: list[float] = []
        # The accepted step length, recorded at the BOTTOM of the loop body and
        # so one shorter than the three above whenever the loop breaks before
        # stepping. That is the honest length: an iteration that rejected every
        # step took none, and padding it with a zero would read as a step at the
        # line-search floor. ``late_step_sizes()`` slices this alone and never
        # zips it against the others, so the offset costs nothing.
        step_sizes: list[float] = []

        trace = settings.trace
        run_start = time.time()
        trace.on_pd_start(
            self._name,
            num_vars=n + problem.dimension_y,
            num_ineq=p_x + p_y,
            tol=tol,
        )

        status = SolveStatus.MAX_ITERATIONS
        # Termination-cause tracker for the honest non-convergence message. It
        # defaults to the budget-exhaustion case (the loop ran every iteration
        # without breaking) and is overwritten by whichever break fires.
        stop_reason = (
            f"exhausted the {max_iter}-iteration budget before the surrogate "
            "gap and residual reached tolerance"
        )
        # Best-effort iterate + its KKT score, for the soft-tolerance fallback
        # (see _Iterate and the fallback after the loop).
        best: _Iterate | None = None
        # The NewtonStepError that stopped the loop, if one did, chained onto the
        # raised StationarityConvergenceError so the traceback still reaches the
        # step that actually gave up.
        newton_failure: NewtonStepError | None = None
        # Bind nit to -1 (not 0) so that if max_iter == 0 the loop never runs
        # yet nit stays defined -- avoiding a NameError in the nit + 1 reads
        # below -- and the reported step count (nit + 1) is a correct 0 rather
        # than an off-by-one 1. When the loop runs, nit is overwritten.
        nit = -1
        for nit in range(max_iter):
            iter_start = time.time()
            f_x = _stacked_constraints(x_domain, x)
            f_y = _stacked_constraints(y_domain, y)
            s_x = -f_x
            s_y = -f_y

            # A binding constraint's slack can underflow to the boundary (s <= 0)
            # in the convergence tail. The primal-dual Hessian requires a strict
            # interior (every f < 0), so no further Newton step exists from a
            # boundary iterate. Computed BEFORE the best-iterate/OPTIMAL checks
            # below (not after): a boundary point must never be recorded as
            # `best` or accepted as OPTIMAL -- the soft-tolerance fallback must
            # only ever hand back a strictly-interior point, since
            # build_primal_dual_coupled_system requires one.
            interior = (not s_x.size or np.min(s_x) > 0.0) and (
                not s_y.size or np.min(s_y) > 0.0
            )

            residual = problem.primal_dual_residual(x, y, lam_x, lam_y)
            r_x = residual[:n]
            r_y = residual[n:]
            dual_norm = float(np.linalg.norm(residual))
            eta = float(s_x @ lam_x + s_y @ lam_y)

            # Record this iteration's top-of-loop measures (no equality here, so
            # the primal-feasibility residual is identically zero).
            surrogate_gaps.append(eta)
            primal_residual_norms.append(0.0)
            dual_residual_norms.append(dual_norm)

            score = max(eta, dual_norm)
            if interior and (best is None or score < best.score):
                best = _Iterate(score, x, y, lam_x, lam_y, no_eq)

            if interior and eta <= tol and dual_norm <= tol:
                status = SolveStatus.OPTIMAL
                break

            # Stop and let the soft-tolerance fallback return the best
            # interior iterate seen (recorded above, when interior held).
            if not interior:
                stop_reason = (
                    "a constraint slack reached its boundary, leaving no "
                    "strictly-feasible interior step"
                )
                break

            try:
                system = problem.build_primal_dual_coupled_system(x, y, lam_x, lam_y)
                if self.check_inertia and nit == 0:
                    _check_inertia(system)

                # Per-block surrogate gaps. A single scalar mu = eta / m_tot is
                # swamped when one block operates at a far larger scale (e.g.
                # slacks ~1e6 in one block vs O(1) in the other), mis-centering both;
                # each block gets its own mu so its corrector aims at its own scale.
                mu_x = float(s_x @ lam_x) / m_x
                mu_y = float(s_y @ lam_y) / m_y
                # Affine predictor: target zero complementarity (r_c = s (.) lambda).
                aff = self._solve_direction(
                    problem,
                    system,
                    x,
                    y,
                    lam_x,
                    lam_y,
                    s_x,
                    s_y,
                    r_x,
                    r_y,
                    s_x * lam_x,
                    s_y * lam_y,
                )
                dx_a, dy_a, dlx_a, dly_a, dsx_a, dsy_a, _ = aff
                a_pri = min(
                    fraction_to_boundary(s_x, dsx_a), fraction_to_boundary(s_y, dsy_a)
                )
                a_dual = min(
                    fraction_to_boundary(lam_x, dlx_a),
                    fraction_to_boundary(lam_y, dly_a),
                )
                # Per-block affine complementarity at the shared affine step length,
                # each giving its own Mehrotra centering sigma, floored at
                # _SIGMA_FLOOR.
                mu_aff_x = float((s_x + a_pri * dsx_a) @ (lam_x + a_dual * dlx_a)) / m_x
                mu_aff_y = float((s_y + a_pri * dsy_a) @ (lam_y + a_dual * dly_a)) / m_y
                sigma_x = min(
                    max((mu_aff_x / mu_x) ** 3 if mu_x > 0.0 else 0.0, 0.0), 1.0
                )
                sigma_y = min(
                    max((mu_aff_y / mu_y) ** 3 if mu_y > 0.0 else 0.0, 0.0), 1.0
                )
                sigma_x = max(sigma_x, _SIGMA_FLOOR)
                sigma_y = max(sigma_y, _SIGMA_FLOOR)

                # Corrector: recenter each block to its own sigma*mu and carry the
                # second-order cross-term.
                r_cx = s_x * lam_x - sigma_x * mu_x + dsx_a * dlx_a
                # Floor the y-block central-path target at the problem-wide
                # optimality measure ``score / score_dof`` (score = max(gap,
                # stationarity residual); ``score_dof`` defined above the loop).
                # A curved y-block (e.g. an ellipsoid) otherwise drives its Newton
                # direction straight onto the boundary, where the true-feasible step
                # is curvature-throttled toward zero and the block freezes. Holding
                # the target interior keeps the path off the exact boundary so the
                # block's direction can bend along the manifold. The floor is tied to
                # ``score`` rather than ``eta`` because the gap can freeze while the
                # stationarity residual is still open; an eta-only floor would switch
                # off too early and leave a tail limit-cycle, whereas ``score`` stays
                # engaged until both measures converge and then relaxes to zero.
                mu_y_target = max(sigma_y * mu_y, score / score_dof)
                r_cy = s_y * lam_y - mu_y_target + dsy_a * dly_a
                dx, dy, dlx, dly, dsx, dsy, _ = self._solve_direction(
                    problem, system, x, y, lam_x, lam_y, s_x, s_y, r_x, r_y, r_cx, r_cy
                )
            except NewtonStepError as exc:
                # A singular, indefinite or non-finite coupled system admits no
                # Newton step, so the iteration cannot continue -- but that is a
                # non-convergence, not a different species of failure. Stop here and
                # let the exits below hand back the best interior iterate seen: the
                # soft-tolerance fallback accepts it outright when it is close
                # enough, and otherwise _raise_nonconvergence attaches it to a
                # StationarityConvergenceError with this error chained as the cause.
                # Unguarded, the bare NewtonStepError would escape every downstream
                # handler that keys off the wrapped types and reach the caller
                # carrying no iterate at all. ``best`` is necessarily set by the time
                # this region runs: the top of the loop records it whenever
                # ``interior`` holds, and a non-interior iterate has already broken
                # out above.
                newton_failure = exc
                stop_reason = f"the coupled Newton step could not be computed ({exc})"
                break

            gamma = mehrotra_boundary_factor(eta, p_x + p_y, tau_min)
            # Both primal slacks share the same Mehrotra boundary relaxation factor
            # gamma, but the resulting step lengths are split per block below (see
            # step_x/step_y). Each dual takes mehrotra_step_cap with its OWN block
            # gap (eta_x, eta_y) rather than the shared eta, so a large-scale
            # block's gap does not relax the other block's dual cap.
            # mu_x/mu_y already carry (s_x @ lam_x)/(s_y @ lam_y); avoid
            # recomputing the same dot products.
            eta_x = mu_x * m_x
            eta_y = mu_y * m_y
            cap_lx = mehrotra_step_cap(lam_x, dlx, eta_x, tau_min)
            cap_ly = mehrotra_step_cap(lam_y, dly, eta_y, tau_min)
            # Per-block primal+dual step lengths. The two blocks have separable
            # constraints (x_domain sees only x, y_domain only y; the coupling is
            # solely through the objective cross-Hessian), so a curvature-throttled
            # y-block (e.g. a nonlinear ellipsoid) must not clamp a still-open
            # linear x-block. Each block's primal step is the EXACT distance to its
            # own constraint boundary via ``_block_max_step`` (a closed-form
            # quadratic root for a curved layer such as the ellipsoid, vectorized
            # closed-form for the linear box), relaxed by the shared Mehrotra
            # factor gamma. This supersedes a linearized fraction-to-boundary that
            # overshoots the curved manifold and then backtracks to a vanishing
            # corrective step.
            step_x = min(cap_lx, gamma * self._block_max_step(x_domain, x, dx))
            step_y = min(cap_ly, gamma * self._block_max_step(y_domain, y, dy))
            if min(step_x, step_y) <= min_step:
                # The exact step to some block's boundary underflowed: the Newton
                # direction runs essentially along a constraint and is no longer
                # usable. Stop and let the soft-tolerance fallback return the best
                # interior iterate seen (recorded above).
                stop_reason = (
                    "the coupled Newton step could not maintain a "
                    "strictly-feasible interior"
                )
                break

            # Apply the per-block step. In the endgame (eta <= _MERIT_ENDGAME_GAP)
            # the shared advance is additionally shrunk by rho so the stationarity
            # residual does not increase (see _endgame_step); above the threshold
            # rho == 1.0 and the full per-block step is taken. The duals are
            # floored strictly positive (see _DUAL_FLOOR): the Mehrotra cap keeps
            # them positive in exact arithmetic, but a binding multiplier can leak
            # to <= 0 by a machine epsilon in the tail.
            x, y, lam_x, lam_y, rho = self._endgame_step(
                problem,
                x,
                y,
                lam_x,
                lam_y,
                dx,
                dy,
                dlx,
                dly,
                step_x,
                step_y,
                eta,
                dual_norm,
                tol,
                beta,
                min_step,
            )
            step = min(step_x, step_y) * rho
            step_sizes.append(step)

            # Post-step iteration event. eta/dual_norm are the top-of-iteration
            # measures; t = m/eta is the Boyd-Vandenberghe surrogate barrier
            # parameter; sigma is the Mehrotra centering just applied. There is no
            # equality layer here, so pri_norm is 0.0. (rho/damping and the KKT
            # residual breakdown stay at their defaults: Levenberg damping is
            # deferred in this engine.)
            trace.on_pd_iteration(
                nit=nit,
                eta=eta,
                dual_norm=dual_norm,
                pri_norm=0.0,
                residual_norm=float(np.sqrt(dual_norm * dual_norm + eta * eta)),
                t=m_tot / eta if eta > 0.0 else 0.0,
                step=step,
                elapsed_ms=1000.0 * (time.time() - iter_start),
                sigma=0.5 * (sigma_x + sigma_y),
            )

        # Soft-tolerance fallback: when a binding constraint's slack underflows,
        # the strict tolerance is unreachable and later iterates diverge. Return
        # the best iterate seen if it is within tolerance_soft -- mirroring the
        # barrier engine (stationarity_solver: OPTIMAL if hard else ACCEPTABLE).
        if (
            status is not SolveStatus.OPTIMAL
            and best is not None
            and best.score <= soft
        ):
            x, y, lam_x, lam_y = best.x, best.y, best.lam_x, best.lam_y
            status = SolveStatus.ACCEPTABLE

        # On a raised non-convergence, carry the BEST iterate seen (a binding
        # slack can drive the last iterate strictly worse). Mirrors the
        # soft-fallback restore above so the attached result is the closest point.
        if status is SolveStatus.MAX_ITERATIONS and best is not None:
            x, y, lam_x, lam_y = best.x, best.y, best.lam_x, best.lam_y

        # Surface the saddle value and a suboptimality estimate on the returned
        # result, mirroring the barrier engine. The gap is the surrogate duality
        # gap s^T lambda recomputed at the RETURNED iterate -- not the loop's
        # last surrogate_gaps entry, which the best-iterate restore above can
        # leave pointing at a different (later, worse) point. Floored at zero to
        # absorb the rare best-effort boundary iterate whose slacks underflow.
        value = float(problem.objective.evaluate(x, y))
        gap = max(
            float(
                -_stacked_constraints(x_domain, x) @ lam_x
                - _stacked_constraints(y_domain, y) @ lam_y
            ),
            0.0,
        )
        result = TwoBlockSolveResult.from_blocks(
            x=x,
            y=y,
            status=status,
            message=two_block_result_message(
                status, self._name, gap, cause=_pd_cause(status, stop_reason)
            ),
            solver=self._name,
            n_iterations=nit + 1,
            objective=ObjectiveOutcome(value=value, suboptimality=gap),
            duals=self._pd_duals(lam_x, lam_y, no_eq),
            diagnostics=PrimalDualDiagnostics(
                surrogate_gaps=tuple(surrogate_gaps),
                primal_residual_norms=tuple(primal_residual_norms),
                dual_residual_norms=tuple(dual_residual_norms),
                step_sizes=tuple(step_sizes),
            ),
        )
        trace.on_pd_done(
            solver=self._name,
            nits=nit + 1,
            status=_legacy_status(status),
            elapsed_ms=1000.0 * (time.time() - run_start),
            reason=_pd_reason(status, stop_reason),
        )
        if status is SolveStatus.MAX_ITERATIONS:
            _raise_nonconvergence(
                result, best, nit + 1, stop_reason, cause=newton_failure
            )
        return result

    def _run_equality(
        self,
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> TwoBlockSolveResult:
        """Run the Mehrotra loop on the bordered path, carrying the multiplier ν.

        Mirrors ``_run`` but threads ν: the residual gains the primal-feasibility row
        r_eq = A_eq x - b_eq, the reduced saddle borders to
        ``EqualityCoupledSaddleSystem``, and each step also recovers Δν, which takes the
        x-block step length. Convergence also requires ‖r_eq‖ ≤ tol, and r_eq enters the
        KKT score. Unlike ``_run``, this loop applies the full per-block step with no
        endgame residual line search, so ``armijo.beta`` is unused.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block problem, with an ``x_equality``.
         x : npt.NDArray[np.float64]
            Strictly interior x-block start, of length n; need not satisfy the equality.
         y : npt.NDArray[np.float64]
            Strictly interior y-block start, of length m.

        Returns
        -------
         result : TwoBlockSolveResult
            The result at the returned iterate, with status ``OPTIMAL`` or
            ``ACCEPTABLE``.

        Raises
        ------
         StationarityConvergenceError
            Through ``_raise_nonconvergence``, when no exit met the strict or soft
            tolerance.
         ValueError
            Propagated from ``_check_inertia`` at the first iteration.

        """
        settings = self.settings
        n = problem.dimension_x
        m = problem.dimension_y
        x_domain = problem.x_domain
        y_domain = problem.y_domain
        p_x = _num_constraints(x_domain, n)
        p_y = _num_constraints(y_domain, m)
        m_tot = max(p_x + p_y, 1)
        # Divisor for the y-block central-path floor (see the ``r_cy`` comment in
        # the loop). ``score`` is a norm over the whole KKT residual -- stationarity
        # over the ``n + dimension_y`` primal DOF plus complementarity over
        # ``m_tot`` pairs -- so dividing it by the pair count alone over-weights the
        # floor when constraints are few relative to the primal variables. The
        # degenerate case is an unconstrained x-block: ``m_tot`` collapses toward 1
        # while the primal residual still spans ``n + dimension_y`` dimensions, so
        # the floor inflates, injects a spurious y-dual, and stalls. Flooring the
        # divisor at the primal DOF keeps it a per-DOF residual scale; whichever of
        # the two dimensions the residual actually spans dominates.
        score_dof = max(m_tot, n + problem.dimension_y)
        # Per-block pair counts (each floored at 1) for the per-block surrogate
        # gaps mu_x, mu_y used in centering below.
        m_x = max(p_x, 1)
        m_y = max(p_y, 1)

        max_iter = settings.inner.max_iterations
        tol = settings.outer.tolerance
        soft = settings.outer.tolerance_soft
        tau_min = settings.pd_fraction_to_boundary_min
        # NB: the equality loop has no endgame residual merit (see the apply block
        # below), so armijo.beta is unused here; only min_step gates the step.
        min_step = settings.armijo.min_step

        lam_x = np.ones(p_x)
        lam_y = np.ones(p_y)
        nu = np.zeros(problem.num_equalities)

        # Per-iteration convergence trajectories, one entry per loop-body entry
        # (recorded at the top of the loop, so the final OPTIMAL/break iteration
        # is captured and ``len == nit + 1 == n_iterations``).
        surrogate_gaps: list[float] = []
        primal_residual_norms: list[float] = []
        dual_residual_norms: list[float] = []
        # Accepted step lengths; see the inequality-only loop for why this one is
        # recorded at the bottom of the body and may be one entry shorter.
        step_sizes: list[float] = []

        trace = settings.trace
        run_start = time.time()
        trace.on_pd_start(
            self._name,
            num_vars=n + m,
            num_ineq=p_x + p_y,
            tol=tol,
        )

        status = SolveStatus.MAX_ITERATIONS
        # Termination-cause tracker for the honest non-convergence message. It
        # defaults to the budget-exhaustion case (the loop ran every iteration
        # without breaking) and is overwritten by whichever break fires.
        stop_reason = (
            f"exhausted the {max_iter}-iteration budget before the surrogate "
            "gap and residual reached tolerance"
        )
        # Best-effort iterate + its KKT score, for the soft-tolerance fallback.
        # See _Iterate for why the two-block PD engine needs this.
        best: _Iterate | None = None
        # The NewtonStepError that stopped the loop, if one did, chained onto the
        # raised StationarityConvergenceError so the traceback still reaches the
        # step that actually gave up.
        newton_failure: NewtonStepError | None = None
        # Bind nit to -1 (not 0) so that if max_iter == 0 the loop never runs
        # yet nit stays defined -- avoiding a NameError in the nit + 1 reads
        # below -- and the reported step count (nit + 1) is a correct 0 rather
        # than an off-by-one 1. When the loop runs, nit is overwritten.
        nit = -1
        for nit in range(max_iter):
            iter_start = time.time()
            f_x = _stacked_constraints(x_domain, x)
            f_y = _stacked_constraints(y_domain, y)
            s_x = -f_x
            s_y = -f_y

            # A binding constraint's slack can underflow to the boundary (s <= 0)
            # in the convergence tail. The primal-dual Hessian requires a strict
            # interior (every f < 0), so no further Newton step exists from a
            # boundary iterate. Computed BEFORE the best-iterate/OPTIMAL checks
            # below (not after): a boundary point must never be recorded as
            # `best` or accepted as OPTIMAL -- the soft-tolerance fallback must
            # only ever hand back a strictly-interior point, since
            # build_primal_dual_coupled_system requires one.
            interior = (not s_x.size or np.min(s_x) > 0.0) and (
                not s_y.size or np.min(s_y) > 0.0
            )

            residual = problem.primal_dual_residual(x, y, lam_x, lam_y, lam_eq=nu)
            r_x = residual[:n]
            r_y = residual[n : n + m]
            r_eq = residual[n + m :]
            # Split the stationarity (dual-feasibility) block from the equality
            # primal-feasibility block; both must vanish at the solution.
            stat_norm = float(np.linalg.norm(residual[: n + m]))
            pri_norm = float(np.linalg.norm(r_eq))
            eta = float(s_x @ lam_x + s_y @ lam_y)

            # Record this iteration's top-of-loop measures: the surrogate gap, the
            # equality primal-feasibility residual, and the stationarity (dual)
            # residual (mirroring the ``on_pd_iteration`` event below).
            surrogate_gaps.append(eta)
            primal_residual_norms.append(pri_norm)
            dual_residual_norms.append(stat_norm)

            score = max(eta, stat_norm, pri_norm)
            if interior and (best is None or score < best.score):
                best = _Iterate(score, x, y, lam_x, lam_y, nu)

            if interior and eta <= tol and stat_norm <= tol and pri_norm <= tol:
                status = SolveStatus.OPTIMAL
                break

            # Stop and let the soft-tolerance fallback return the best
            # interior iterate seen (recorded above, when interior held).
            if not interior:
                stop_reason = (
                    "a constraint slack reached its boundary, leaving no "
                    "strictly-feasible interior step"
                )
                break

            try:
                system = problem.build_primal_dual_coupled_system(x, y, lam_x, lam_y)
                if self.check_inertia and nit == 0:
                    _check_inertia(system)

                # Per-block surrogate gaps. A single scalar mu = eta / m_tot is
                # swamped when one block operates at a far larger scale (e.g.
                # slacks ~1e6 in one block vs O(1) in the other), mis-centering both;
                # each block gets its own mu so its corrector aims at its own scale.
                mu_x = float(s_x @ lam_x) / m_x
                mu_y = float(s_y @ lam_y) / m_y
                # Affine predictor: target zero complementarity (r_c = s (.) lambda).
                aff = self._solve_direction(
                    problem,
                    system,
                    x,
                    y,
                    lam_x,
                    lam_y,
                    s_x,
                    s_y,
                    r_x,
                    r_y,
                    s_x * lam_x,
                    s_y * lam_y,
                    r_eq,
                )
                dx_a, dy_a, dlx_a, dly_a, dsx_a, dsy_a, _ = aff
                a_pri = min(
                    fraction_to_boundary(s_x, dsx_a), fraction_to_boundary(s_y, dsy_a)
                )
                a_dual = min(
                    fraction_to_boundary(lam_x, dlx_a),
                    fraction_to_boundary(lam_y, dly_a),
                )
                # Per-block affine complementarity at the shared affine step length,
                # each giving its own Mehrotra centering sigma, floored at
                # _SIGMA_FLOOR.
                mu_aff_x = float((s_x + a_pri * dsx_a) @ (lam_x + a_dual * dlx_a)) / m_x
                mu_aff_y = float((s_y + a_pri * dsy_a) @ (lam_y + a_dual * dly_a)) / m_y
                sigma_x = min(
                    max((mu_aff_x / mu_x) ** 3 if mu_x > 0.0 else 0.0, 0.0), 1.0
                )
                sigma_y = min(
                    max((mu_aff_y / mu_y) ** 3 if mu_y > 0.0 else 0.0, 0.0), 1.0
                )
                sigma_x = max(sigma_x, _SIGMA_FLOOR)
                sigma_y = max(sigma_y, _SIGMA_FLOOR)

                # Corrector: recenter each block to its own sigma*mu and carry the
                # second-order cross-term.
                r_cx = s_x * lam_x - sigma_x * mu_x + dsx_a * dlx_a
                # Floor the y-block central-path target at the problem-wide
                # optimality measure ``score / score_dof`` (score = max(gap,
                # stationarity residual); ``score_dof`` defined above the loop).
                # A curved y-block (e.g. an ellipsoid) otherwise drives its Newton
                # direction straight onto the boundary, where the true-feasible step
                # is curvature-throttled toward zero and the block freezes. Holding
                # the target interior keeps the path off the exact boundary so the
                # block's direction can bend along the manifold. The floor is tied to
                # ``score`` rather than ``eta`` because the gap can freeze while the
                # stationarity residual is still open; an eta-only floor would switch
                # off too early and leave a tail limit-cycle, whereas ``score`` stays
                # engaged until both measures converge and then relaxes to zero.
                mu_y_target = max(sigma_y * mu_y, score / score_dof)
                r_cy = s_y * lam_y - mu_y_target + dsy_a * dly_a
                dx, dy, dlx, dly, dsx, dsy, dnu = self._solve_direction(
                    problem,
                    system,
                    x,
                    y,
                    lam_x,
                    lam_y,
                    s_x,
                    s_y,
                    r_x,
                    r_y,
                    r_cx,
                    r_cy,
                    r_eq,
                )
            except NewtonStepError as exc:
                # Same guard as the no-equality loop above: a coupled system with
                # no Newton step is a non-convergence, so stop and let the best
                # interior iterate ride out on the soft-tolerance fallback or on
                # _raise_nonconvergence's StationarityConvergenceError, with this
                # error chained as the cause. ``best`` is necessarily set here --
                # the top of the loop records it whenever ``interior`` holds, and a
                # non-interior iterate has already broken out above.
                newton_failure = exc
                stop_reason = f"the coupled Newton step could not be computed ({exc})"
                break

            gamma = mehrotra_boundary_factor(eta, p_x + p_y, tau_min)
            # Both primal slacks share the same Mehrotra boundary relaxation factor
            # gamma, but the resulting step lengths are split per block below (see
            # step_x/step_y). Each dual takes mehrotra_step_cap with its OWN block
            # gap (eta_x, eta_y) rather than the shared eta, so a large-scale
            # block's gap does not relax the other block's dual cap.
            # mu_x/mu_y already carry (s_x @ lam_x)/(s_y @ lam_y); avoid
            # recomputing the same dot products.
            eta_x = mu_x * m_x
            eta_y = mu_y * m_y
            cap_lx = mehrotra_step_cap(lam_x, dlx, eta_x, tau_min)
            cap_ly = mehrotra_step_cap(lam_y, dly, eta_y, tau_min)
            # Per-block primal+dual step lengths. The two blocks have separable
            # constraints (x_domain sees only x, y_domain only y; the coupling is
            # solely through the objective cross-Hessian), so a curvature-throttled
            # y-block (e.g. a nonlinear ellipsoid) must not clamp a still-open
            # linear x-block. Each block's primal step is the EXACT distance to its
            # own constraint boundary via ``_block_max_step`` (a closed-form
            # quadratic root for a curved layer such as the ellipsoid, vectorized
            # closed-form for the linear box), relaxed by the shared Mehrotra
            # factor gamma. This supersedes a linearized fraction-to-boundary that
            # overshoots the curved manifold and then backtracks to a vanishing
            # corrective step.
            step_x = min(cap_lx, gamma * self._block_max_step(x_domain, x, dx))
            step_y = min(cap_ly, gamma * self._block_max_step(y_domain, y, dy))
            if min(step_x, step_y) <= min_step:
                # The exact step to some block's boundary underflowed: the Newton
                # direction runs essentially along a constraint and is no longer
                # usable. Stop and let the soft-tolerance fallback return the best
                # interior iterate seen (recorded above).
                stop_reason = (
                    "the coupled Newton step could not maintain a "
                    "strictly-feasible interior"
                )
                break

            # Apply the per-block step. Unlike the inequality-only loop this branch
            # does NOT run the endgame residual merit: the equality primal residual
            # couples the blocks through nu, so a shared residual-descent shrink is
            # not yet validated here. Duals are floored strictly positive (see
            # _DUAL_FLOOR): the Mehrotra cap keeps them positive in exact
            # arithmetic, but a binding multiplier can leak to <= 0 by a machine
            # epsilon in the tail.
            x = x + step_x * dx
            y = y + step_y * dy
            lam_x = np.maximum(lam_x + step_x * dlx, _DUAL_FLOOR)
            lam_y = np.maximum(lam_y + step_y * dly, _DUAL_FLOOR)
            # The equality multiplier is free (unsigned): it shares the x-block
            # primal step fraction and takes no fraction-to-boundary cap or clamp.
            nu = nu + step_x * dnu
            step = min(step_x, step_y)
            step_sizes.append(step)

            # Post-step iteration event. dual_norm carries the stationarity
            # (dual-feasibility) block; pri_norm carries the equality primal
            # residual ||A x - b||; both must vanish at the solution. t = m/eta is
            # the Boyd-Vandenberghe surrogate barrier parameter, sigma the Mehrotra
            # centering just applied.
            trace.on_pd_iteration(
                nit=nit,
                eta=eta,
                dual_norm=stat_norm,
                pri_norm=pri_norm,
                residual_norm=float(
                    np.sqrt(stat_norm * stat_norm + pri_norm * pri_norm + eta * eta)
                ),
                t=m_tot / eta if eta > 0.0 else 0.0,
                step=step,
                elapsed_ms=1000.0 * (time.time() - iter_start),
                sigma=0.5 * (sigma_x + sigma_y),
            )

        # Soft-tolerance fallback: when a binding constraint's slack underflows,
        # the strict tolerance is unreachable and later iterates diverge. Return
        # the best iterate seen if it is within tolerance_soft -- mirroring the
        # barrier engine (stationarity_solver: OPTIMAL if hard else ACCEPTABLE).
        if (
            status is not SolveStatus.OPTIMAL
            and best is not None
            and best.score <= soft
        ):
            x, y, lam_x, lam_y, nu = best.x, best.y, best.lam_x, best.lam_y, best.nu
            status = SolveStatus.ACCEPTABLE

        # On a raised non-convergence, carry the BEST iterate seen (a binding
        # slack can drive the last iterate strictly worse). Mirrors the
        # soft-fallback restore above so the attached result is the closest point.
        if status is SolveStatus.MAX_ITERATIONS and best is not None:
            x, y, lam_x, lam_y, nu = best.x, best.y, best.lam_x, best.lam_y, best.nu

        # Surface the saddle value and a suboptimality estimate on the returned
        # result, mirroring the barrier engine. The gap is the surrogate duality
        # gap s^T lambda over the INEQUALITY slacks (equalities enter via nu, not
        # the complementarity gap), recomputed at the RETURNED iterate -- not the
        # loop's last surrogate_gaps entry, which the best-iterate restore above
        # can leave pointing at a different point. Floored at zero to absorb the
        # rare best-effort boundary iterate whose slacks underflow.
        value = float(problem.objective.evaluate(x, y))
        gap = max(
            float(
                -_stacked_constraints(x_domain, x) @ lam_x
                - _stacked_constraints(y_domain, y) @ lam_y
            ),
            0.0,
        )
        result = TwoBlockSolveResult.from_blocks(
            x=x,
            y=y,
            status=status,
            message=two_block_result_message(
                status, self._name, gap, cause=_pd_cause(status, stop_reason)
            ),
            solver=self._name,
            n_iterations=nit + 1,
            objective=ObjectiveOutcome(value=value, suboptimality=gap),
            duals=self._pd_duals(lam_x, lam_y, nu),
            diagnostics=PrimalDualDiagnostics(
                surrogate_gaps=tuple(surrogate_gaps),
                primal_residual_norms=tuple(primal_residual_norms),
                dual_residual_norms=tuple(dual_residual_norms),
                step_sizes=tuple(step_sizes),
            ),
        )
        trace.on_pd_done(
            solver=self._name,
            nits=nit + 1,
            status=_legacy_status(status),
            elapsed_ms=1000.0 * (time.time() - run_start),
            reason=_pd_reason(status, stop_reason),
        )
        if status is SolveStatus.MAX_ITERATIONS:
            _raise_nonconvergence(
                result, best, nit + 1, stop_reason, cause=newton_failure
            )
        return result

    @staticmethod
    def _pd_duals(
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
    ) -> Duals | None:
        """Pack the per-block duals and the equality multipliers into ``Duals``.

        The inequality multipliers are [λ^x; λ^y], in x-then-y block order; the equality
        multipliers are ν. Each side becomes ``None`` when it is empty, and the whole
        result is ``None`` when the problem has neither inequalities nor equalities.

        Parameters
        ----------
         lam_x : npt.NDArray[np.float64]
            The x-domain inequality duals.
         lam_y : npt.NDArray[np.float64]
            The y-domain inequality duals.
         nu : npt.NDArray[np.float64]
            The equality multipliers (empty on the no-equality path).

        Returns
        -------
         duals : Duals | None
            The packed multipliers, or ``None`` when there are none.

        """
        ineq = np.concatenate([lam_x, lam_y])
        ineq_multipliers = ineq if ineq.size else None
        eq_multipliers = nu if nu.size else None
        if ineq_multipliers is None and eq_multipliers is None:
            return None
        return Duals(
            equality_multipliers=eq_multipliers,
            inequality_multipliers=ineq_multipliers,
        )

    @staticmethod
    def _solve_direction(
        problem: StationarityProblem,
        system: CoupledSaddleSystem | EqualityCoupledSaddleSystem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
        s_x: npt.NDArray[np.float64],
        s_y: npt.NDArray[np.float64],
        r_x: npt.NDArray[np.float64],
        r_y: npt.NDArray[np.float64],
        r_cx: npt.NDArray[np.float64],
        r_cy: npt.NDArray[np.float64],
        r_eq: npt.NDArray[np.float64] | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Solve the reduced saddle for one centering residual; recover dual steps.

        Eliminating the complementarity rows
           Λ Δs + S Δλ = -r_c,   Δs = -J Δw,
        leaves the reduced right-hand side
           b_x = -r_x - J_x^T (r_cx / s_x),   b_y = -r_y + J_y^T (r_cy / s_y),
        which the coupled saddle solves for (Δx, Δy). The dual and slack steps follow as
           Δλ = S^{-1} (-r_c + Λ J Δw),   Δs = -J Δw,
        per block. With an equality, the bordered solve also enforces A_eq Δx = -r_eq
        and returns Δν; r_eq does not enter b_x or b_y.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block problem, which supplies the domains.
         system : CoupledSaddleSystem | EqualityCoupledSaddleSystem
            The factored coupled saddle; an ``EqualityCoupledSaddleSystem`` exactly when
            ``r_eq`` is given.
         x : npt.NDArray[np.float64]
            The x-block iterate, of length n.
         y : npt.NDArray[np.float64]
            The y-block iterate, of length m.
         lam_x : npt.NDArray[np.float64]
            The x-domain duals, of length p_x.
         lam_y : npt.NDArray[np.float64]
            The y-domain duals, of length p_y.
         s_x : npt.NDArray[np.float64]
            The x-domain slacks -f^x(x), of length p_x.
         s_y : npt.NDArray[np.float64]
            The y-domain slacks -f^y(y), of length p_y.
         r_x : npt.NDArray[np.float64]
            The x-block stationarity residual, of length n.
         r_y : npt.NDArray[np.float64]
            The y-block stationarity residual, of length m.
         r_cx : npt.NDArray[np.float64]
            The x-domain centering residual (predictor or corrector), of length p_x.
         r_cy : npt.NDArray[np.float64]
            The y-domain centering residual, of length p_y.
         r_eq : npt.NDArray[np.float64] | None
            The equality residual A_eq x - b_eq, or ``None`` without an equality.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The x-block step, of length n.
         delta_y : npt.NDArray[np.float64]
            The y-block step, of length m.
         delta_lam_x : npt.NDArray[np.float64]
            The x-domain dual step, of length p_x.
         delta_lam_y : npt.NDArray[np.float64]
            The y-domain dual step, of length p_y.
         delta_s_x : npt.NDArray[np.float64]
            The x-domain slack step, of length p_x.
         delta_s_y : npt.NDArray[np.float64]
            The y-domain slack step, of length p_y.
         delta_nu : npt.NDArray[np.float64]
            The equality-multiplier step; empty without an equality.

        Raises
        ------
         NewtonStepError
            Propagated from the coupled saddle's ``solve`` when the system is singular,
            indefinite, or non-finite.

        """
        x_domain = problem.x_domain
        y_domain = problem.y_domain
        # Empty-domain slices are length-0, so the divisions and products below are
        # vacuous (no complementarity rows to eliminate for an unconstrained block).
        b_x = -r_x - _stacked_jac_transpose(x_domain, x, r_cx / s_x)
        b_y = -r_y + _stacked_jac_transpose(y_domain, y, r_cy / s_y)
        if r_eq is None:
            assert isinstance(system, CoupledSaddleSystem)
            delta_x, delta_y = system.solve(b_x, b_y)
            delta_nu = np.zeros(0)
        else:
            assert isinstance(system, EqualityCoupledSaddleSystem)
            delta_x, delta_y, delta_nu = system.solve(b_x, b_y, -r_eq)

        jx_dx = _stacked_jac_vec(x_domain, x, delta_x)
        jy_dy = _stacked_jac_vec(y_domain, y, delta_y)
        delta_s_x = -jx_dx
        delta_s_y = -jy_dy
        delta_lam_x = (-r_cx + lam_x * jx_dx) / s_x
        delta_lam_y = (-r_cy + lam_y * jy_dy) / s_y
        return (
            delta_x,
            delta_y,
            delta_lam_x,
            delta_lam_y,
            delta_s_x,
            delta_s_y,
            delta_nu,
        )

    @staticmethod
    def _block_max_step(
        domain: Sequence[ConstraintLayer],
        v: npt.NDArray[np.float64],
        dv: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step to one block's constraint boundary.

        The blocks have separable constraints (each domain reads only its own
        variables), so their boundary distances are found independently, and a
        curvature-throttled block does not clamp a still-open one. Each layer's
        ``btls_max_step`` gives its bound: closed form for the layers that override it
        (such as ``BoundsConstraintLayer`` and the norm layers), and the base class's
        backtracking search, capped at 1 / β, otherwise.

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's constraint layers; may be empty.
         v : npt.NDArray[np.float64]
            The block's current point, strictly interior.
         dv : npt.NDArray[np.float64]
            The block's step direction.

        Returns
        -------
         alpha : float
            The minimum over the layers' bounds; inf for an empty domain, leaving only
            the dual cap to limit the block's step.

        """
        return min(
            (layer.btls_max_step(v, dv) for layer in domain),
            default=float("inf"),
        )

    @staticmethod
    def _endgame_step(
        problem: StationarityProblem,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
        dy: npt.NDArray[np.float64],
        dlx: npt.NDArray[np.float64],
        dly: npt.NDArray[np.float64],
        step_x: float,
        step_y: float,
        eta: float,
        dual_norm: float,
        tol: float,
        beta: float,
        min_step: float,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Apply the per-block step, with a residual line search in the endgame.

        While η > ``_MERIT_ENDGAME_GAP`` the method is still path-following: the raw KKT
        residual legitimately rises during predictor-corrector, so the full per-block
        step is taken (ρ = 1). Once η ≤ ``_MERIT_ENDGAME_GAP``, complementarity is
        essentially met and what remains is stationarity descent, so a shared shrink ρ
        backtracks by ``beta`` until the residual at the trial point satisfies
           ‖r_t‖ ≤ (1 - 10^-4 ρ) ‖r‖   or   ‖r_t‖ ≤ tol.
        Both blocks scale by the same ρ, so their relative step proportions are kept.
        The trial duals are floored at ``_DUAL_FLOOR``.

        When no ρ above ``min_step`` satisfies the test, the search returns the first
        scale at or below ``min_step`` rather than declining to move. That is
        deliberate: the obvious alternative, falling back to the full step, was measured
        on three production problems and diverged, because ρ = 1 is the worst point on
        the backtracking grid at these iterates (a stationarity residual a median of 300
        times the incumbent, and up to 8000 times). A tiny step that fails the test
        costs one iteration; the full step undoes many. The outer loop's convergence
        test is the safety net, and the returned ρ shows which case occurred.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block problem, which evaluates the residual.
         x : npt.NDArray[np.float64]
            The x-block iterate, of length n.
         y : npt.NDArray[np.float64]
            The y-block iterate, of length m.
         lam_x : npt.NDArray[np.float64]
            The x-domain duals, of length p_x.
         lam_y : npt.NDArray[np.float64]
            The y-domain duals, of length p_y.
         dx : npt.NDArray[np.float64]
            The x-block direction.
         dy : npt.NDArray[np.float64]
            The y-block direction.
         dlx : npt.NDArray[np.float64]
            The x-domain dual direction.
         dly : npt.NDArray[np.float64]
            The y-domain dual direction.
         step_x : float
            The x-block step length (primal and dual).
         step_y : float
            The y-block step length (primal and dual).
         eta : float
            The surrogate gap at the iterate, which selects the endgame.
         dual_norm : float
            The stationarity residual norm ‖r‖ at the iterate.
         tol : float
            Residual norm below which any trial point is accepted.
         beta : float
            Shrink factor for ρ.
         min_step : float
            Scale at which the search stops.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The new x-block iterate.
         y : npt.NDArray[np.float64]
            The new y-block iterate.
         lam_x : npt.NDArray[np.float64]
            The new x-domain duals, floored at ``_DUAL_FLOOR``.
         lam_y : npt.NDArray[np.float64]
            The new y-domain duals, floored at ``_DUAL_FLOOR``.
         rho : float
            The accepted shrink: 1 outside the endgame, and at most ``min_step`` when
            the search was exhausted.

        """
        rho = 1.0
        x_t = x + rho * step_x * dx
        y_t = y + rho * step_y * dy
        lx_t = np.maximum(lam_x + rho * step_x * dlx, _DUAL_FLOOR)
        ly_t = np.maximum(lam_y + rho * step_y * dly, _DUAL_FLOOR)
        if eta > _MERIT_ENDGAME_GAP:
            return x_t, y_t, lx_t, ly_t, rho
        while rho > min_step:
            r_t = float(
                np.linalg.norm(problem.primal_dual_residual(x_t, y_t, lx_t, ly_t))
            )
            if r_t <= (1.0 - 1e-4 * rho) * dual_norm or r_t <= tol:
                break
            rho *= beta
            x_t = x + rho * step_x * dx
            y_t = y + rho * step_y * dy
            lx_t = np.maximum(lam_x + rho * step_x * dlx, _DUAL_FLOOR)
            ly_t = np.maximum(lam_y + rho * step_y * dly, _DUAL_FLOOR)
        # Falls through here when the budget is exhausted, returning the smallest
        # scale tried. Deliberate -- see the docstring for the measurements.
        return x_t, y_t, lx_t, ly_t, rho
