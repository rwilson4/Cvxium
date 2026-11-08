"""Unified solver result for Cvxium: a ``SolveResult`` built from concern-scoped parts.

A solver's result carries fields whose applicability depends on two independent things:
the problem (a feasibility problem has no objective value; an unconstrained problem has
no Lagrange multipliers) and the solver (the barrier method has inner centering steps
that the primal-dual method lacks). Rather than encode that two-dimensional fact in a
one-dimensional inheritance chain, which forces sentinel values into the data and
``isinstance`` checks into every caller, ``SolveResult`` is a single concrete type
assembled by composition:
- the always-present spine (``solution``, ``status``, ``message``, ``solver``,
  ``n_iterations``);
- problem-shape parts (``objective``, ``duals``), present exactly when the problem has
  that structure, so their absence itself carries the information a sentinel would;
- a single solver-method part (``diagnostics``), whose concrete type names the solver
  that ran;
- optional ``certificate`` and ``phase1`` parts.
Convenience accessors (``objective_value``, ``equality_multipliers``, and so on)
delegate into the parts and raise a specific error when the concern is absent, so the
common path stays flat. ``TwoBlockSolveResult`` adds x- and y-block views for
saddle-point solves.

The module depends only on NumPy, matplotlib, and the standard library; it imports no
other Cvxium module, so it can be used anywhere without risking an import cycle.
(matplotlib backs the diagnostics plotting methods and, under lazy imports, is not
loaded until a plot method is first called.) Every container is frozen (a result is a
value, not a workspace) with ``eq=False``, because the fields include NumPy arrays and
nested parts with no clean value equality or hash; compare fields explicitly instead.
Each container supports a ``to_dict``/``from_dict`` round trip, and ``SolveResult`` adds
``serialize``/``deserialize`` for JSON.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _encode_array(a: npt.NDArray[np.float64] | None) -> list[float] | None:
    """Encode an optional float array as a JSON-ready list.

    Parameters
    ----------
     a : npt.NDArray[np.float64] | None
        The array to encode, or ``None``.

    Returns
    -------
     v : list[float] | None
        The entries as Python floats, or ``None`` when ``a`` is ``None``.

    """
    if a is None:
        return None
    return [float(x) for x in a]


def _decode_array(v: list[float] | None) -> npt.NDArray[np.float64] | None:
    """Decode an optional list back to a float array.

    Parameters
    ----------
     v : list[float] | None
        The list to decode, or ``None``.

    Returns
    -------
     a : npt.NDArray[np.float64] | None
        The float64 array, or ``None`` when ``v`` is ``None``.

    """
    if v is None:
        return None
    return np.asarray(v, dtype=np.float64)


class NoObjectiveError(AttributeError):
    """Error raised when objective fields are read from a feasibility result."""


class NoDualValueError(AttributeError):
    """Error raised when ``dual_value`` is read but the solver did not compute one."""


class NoDualsError(AttributeError):
    """Error raised when reading multipliers from a result that has none."""


class WrongDiagnosticsError(TypeError):
    """Error raised when diagnostics are narrowed to the wrong solver engine."""


class SolveStatus(Enum):
    """Outcome of a solve.

    The members are:
    - ``OPTIMAL``: the duality gap is within tolerance;
    - ``ACCEPTABLE``: the gap is within the soft tolerance;
    - ``FEASIBLE``: a feasibility solve found a feasible point;
    - ``MAX_ITERATIONS``: an optimization stopped at a best-effort iterate;
    - ``INFEASIBLE``: certified infeasible, or no feasible point was found
      (``ProblemMarginallyFeasibleError`` carries this status without claiming a
      certificate);
    - ``UNBOUNDED``: certified unbounded.
    The centering step still reports the integer codes 0, 1, and 2 internally;
    ``_legacy_solve_status`` in ``optimization`` maps them to ``OPTIMAL``,
    ``ACCEPTABLE``, and ``MAX_ITERATIONS``.

    """

    OPTIMAL = "optimal"  # gap within tolerance
    ACCEPTABLE = "acceptable"  # gap within soft tolerance
    FEASIBLE = "feasible"  # feasibility solve found a feasible point
    MAX_ITERATIONS = "max_iterations"  # optimization stopped at a best-effort iterate
    # Certified infeasible, or the solver could not find a feasible point.
    # ``ProblemMarginallyFeasibleError`` carries this status deliberately
    # without claiming a certificate.
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"  # certified unbounded


@dataclass(frozen=True, eq=False)
class SolverDiagnostics:
    """Solver-method part of a result; the concrete subclass names the solver.

    The base carries no fields; each solver's subclass exposes its own per-iteration
    trajectories as typed tuples. The subclasses are ``BarrierDiagnostics``,
    ``SaddleBarrierDiagnostics``, ``PrimalDualDiagnostics`` and ``HsdDiagnostics``.

    """

    def iteration_count(self) -> int:
        """Return the number of iterations performed, comparable within one solver.

        The unit differs by method (Newton steps for the barrier method, outer
        iterations for the primal-dual and homogeneous self-dual methods), but within
        one solver it is an honest measure of work, which is what a performance guard
        needs.

        Returns
        -------
         n : int
            The iteration count; 0 in the base, meaning the solver does not report one.

        """
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict tagged with the concrete type.

        Returns
        -------
         d : dict[str, Any]
            A dict whose ``"__type__"`` entry names the diagnostics class.

        """
        return {"__type__": "SolverDiagnostics"}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SolverDiagnostics:
        """Reconstruct the tagged diagnostics subclass from ``to_dict`` output.

        Dispatches on the ``"__type__"`` tag. The barrier step trajectories
        (``inner_btls_steps``, ``inner_quadratic_from``) and the primal-dual
        ``step_sizes`` are optional in the payload and default to empty.

        Parameters
        ----------
         d : dict[str, Any]
            A dict produced by some diagnostics class's ``to_dict``.

        Returns
        -------
         diagnostics : SolverDiagnostics
            An instance of the tagged class.

        Raises
        ------
         ValueError
            When the tag names no known diagnostics class.
         KeyError
            When a required entry is missing.

        """
        kind = d["__type__"]
        if kind == "SolverDiagnostics":
            return SolverDiagnostics()
        elif kind == "BarrierDiagnostics":
            return BarrierDiagnostics(
                duality_gaps=tuple(d["duality_gaps"]),
                inner_nits=tuple(d["inner_nits"]),
                inner_suboptimalities=tuple(
                    tuple(s) for s in d["inner_suboptimalities"]
                ),
                # Defaulted: a payload written before these existed still
                # round-trips, it just reports no step trajectory.
                inner_btls_steps=tuple(tuple(s) for s in d.get("inner_btls_steps", [])),
                inner_quadratic_from=tuple(d.get("inner_quadratic_from", [])),
            )
        elif kind == "SaddleBarrierDiagnostics":
            return SaddleBarrierDiagnostics(
                duality_gaps=tuple(d["duality_gaps"]),
                inner_nits=tuple(d["inner_nits"]),
                inner_suboptimalities=tuple(
                    tuple(s) for s in d["inner_suboptimalities"]
                ),
                inner_btls_steps=tuple(tuple(s) for s in d.get("inner_btls_steps", [])),
                inner_quadratic_from=tuple(d.get("inner_quadratic_from", [])),
                upper_bounds=tuple(d["upper_bounds"]),
                lower_bounds=tuple(d["lower_bounds"]),
                certified_upper_bounds=tuple(d["certified_upper_bounds"]),
                certified_lower_bounds=tuple(d["certified_lower_bounds"]),
            )
        elif kind == "PrimalDualDiagnostics":
            return PrimalDualDiagnostics(
                surrogate_gaps=tuple(d["surrogate_gaps"]),
                primal_residual_norms=tuple(d["primal_residual_norms"]),
                dual_residual_norms=tuple(d["dual_residual_norms"]),
                step_sizes=tuple(d.get("step_sizes", [])),
            )
        elif kind == "HsdDiagnostics":
            return HsdDiagnostics(
                duality_gaps=tuple(d["duality_gaps"]),
                tau_history=tuple(d["tau_history"]),
                kappa_history=tuple(d["kappa_history"]),
            )
        else:
            raise ValueError(f"unknown diagnostics type {kind!r}")


@dataclass(frozen=True, eq=False)
class BarrierDiagnostics(SolverDiagnostics):
    """Barrier-method trajectories, one entry per outer (barrier-parameter) step.

    Entry i of ``inner_nits``, ``inner_suboptimalities``, ``inner_btls_steps`` and
    ``inner_quadratic_from`` describes the inner Newton centering performed during outer
    step i.

    Together, ``inner_btls_steps`` and ``inner_quadratic_from`` answer a question the
    other trajectories cannot: whether the solver takes full Newton steps once it is
    entitled to. Past the quadratic-convergence threshold, backtracking accepts the unit
    step (Boyd and Vandenberghe, 2004, section 9.5.3), so a step below 1 there means a
    step cap is binding when it should not be, a defect that costs iterations silently
    and otherwise shows up only as a slow solver. ``steps_past_quadratic_convergence``
    extracts those steps.

    Fields
    ------
     duality_gaps : tuple[float, ...]
        Duality gap after each outer step.
     inner_nits : tuple[int, ...]
        Number of Newton iterations in each centering step.
     inner_suboptimalities : tuple[tuple[float, ...], ...]
        Per centering step, the suboptimality estimate λ^2 / 2 (λ the Newton decrement)
        at each Newton iteration.
     inner_btls_steps : tuple[tuple[float, ...], ...]
        Per centering step, the accepted backtracking step length of each Newton
        iteration.
     inner_quadratic_from : tuple[int | None, ...]
        Per centering step, the 0-based index into that step's ``inner_btls_steps``
        entry of the first step taken after the suboptimality fell below the
        quadratic-convergence threshold; ``None`` when it never did.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    duality_gaps: tuple[float, ...] = ()
    inner_nits: tuple[int, ...] = ()
    inner_suboptimalities: tuple[tuple[float, ...], ...] = ()
    inner_btls_steps: tuple[tuple[float, ...], ...] = ()
    inner_quadratic_from: tuple[int | None, ...] = ()

    def iteration_count(self) -> int:
        """Return the total number of Newton iterations across all centering steps.

        Returns
        -------
         n : int
            The sum of ``inner_nits``.

        """
        return sum(self.inner_nits)

    def steps_past_quadratic_convergence(self) -> tuple[float, ...]:
        """Return every accepted step length taken in the quadratic phase.

        The hook for regression tests, which assert that these are all 1: a solver in
        the quadratic phase that is not taking unit steps has a cap binding somewhere,
        and the iterations it costs are invisible in any aggregate the result otherwise
        exposes.

        Returns
        -------
         steps : tuple[float, ...]
            The steps from each centering step's ``inner_quadratic_from`` index onward,
            concatenated over the outer steps. Empty when no outer step reached the
            quadratic phase, which callers should treat as no evidence rather than as a
            pass.

        """
        out: list[float] = []
        for steps, start in zip(
            self.inner_btls_steps, self.inner_quadratic_from, strict=False
        ):
            if start is None:
                continue
            out.extend(steps[start:])
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the barrier trajectories to a JSON-ready dict.

        Returns
        -------
         d : dict[str, Any]
            The fields as lists, tagged ``"BarrierDiagnostics"``.

        """
        return {
            "__type__": "BarrierDiagnostics",
            "duality_gaps": list(self.duality_gaps),
            "inner_nits": list(self.inner_nits),
            "inner_suboptimalities": [list(s) for s in self.inner_suboptimalities],
            "inner_btls_steps": [list(s) for s in self.inner_btls_steps],
            "inner_quadratic_from": list(self.inner_quadratic_from),
        }

    def plot_convergence(self, ax: Axes | None = None) -> Axes:
        """Plot the duality gap per outer step on a log scale.

        Draws a step function (``stairs``), since the barrier duality gap is a discrete
        value sampled once per outer centering step. ``PrimalDualDiagnostics`` and
        ``HsdDiagnostics`` track per-iteration trajectories and draw ordinary line plots
        instead.

        Parameters
        ----------
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        """
        if ax is None:
            _, ax = plt.subplots()
        n = len(self.duality_gaps)
        ax.stairs(
            values=list(self.duality_gaps),
            edges=list(range(n + 1)),
            baseline=None,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Centering Step")
        ax.set_ylabel("Duality Gap")
        return ax

    def plot_newton_convergence(self, it: int, ax: Axes | None = None) -> Axes:
        """Plot the inner Newton suboptimality for one outer (centering) step.

        Parameters
        ----------
         it : int
            The outer step to plot, 1-indexed.
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        Raises
        ------
         ValueError
            When ``inner_nits`` and ``inner_suboptimalities`` differ in length, when no
            trajectories were recorded, or when ``it`` is outside 1 to the number of
            outer steps.

        """
        if len(self.inner_nits) != len(self.inner_suboptimalities):
            raise ValueError(
                "inner_nits and inner_suboptimalities must have the same length"
            )
        n_steps = len(self.inner_nits)
        if n_steps == 0:
            raise ValueError("no inner-Newton trajectories recorded")
        if it < 1 or it > n_steps:
            raise ValueError(f"`it` must be between 1 and {n_steps}, inclusive")
        if ax is None:
            _, ax = plt.subplots()
        suboptimalities = self.inner_suboptimalities[it - 1]
        ax.plot(
            [ii + 1 for ii in range(len(suboptimalities))], suboptimalities, marker="o"
        )
        ax.set_yscale("log")
        ax.set_title(f"Convergence of Centering Step {it}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sub-Optimality")
        return ax


@dataclass(frozen=True, eq=False)
class SaddleBarrierDiagnostics(BarrierDiagnostics):
    """Barrier trajectories for a two-block saddle solve, with saddle-value brackets.

    Extends ``BarrierDiagnostics`` with a closed-form bracket on the saddle value V,
    recorded once per outer step:
       lower_bounds[i] ≤ V ≤ upper_bounds[i].
    The two-block objective ζ is concave in the x-block and convex in the y-block (the
    negated x-block Hessian is positive definite), so a joint stationary point solves
       max_x min_y ζ(x, y).
    On the barrier central path, the two frozen-block best-response duals have a closed
    form (central-path duality applied per block). With the barrier parameter t, the
    per-block inequality counts m_x (the x-block, a maximization) and m_y (the y-block,
    a minimization), and the achieved saddle value
       v = ζ(x, y),
    the bounds are
       L = v - m_y / t ≤ V,   U = v + m_x / t ≥ V.
    The y-block, the inner minimization (min_y ζ(x_bar, y) ≤ V), sets the lower bound,
    and the x-block, the outer maximization (max_x ζ(x, y_bar) ≥ V), sets the upper
    bound. So upper_bounds[i] - lower_bounds[i] equals duality_gaps[i], the surrogate
    gap (m_x + m_y) / t. The bracket is exact on the central path and approximate, to
    the inner centering tolerance, off it, the same caveat the surrogate gap carries.

    The optional certified bounds give a guaranteed bracket that holds off the central
    path too. Each is obtained by minimizing the frozen-block Lagrangian again with a
    warm-started mini-Newton solve and subtracting δ^2, the squared Newton decrement.
    Inside the self-concordant region (Boyd and Vandenberghe, 2004, Eq. 9.50) this
    correction yields a guaranteed bound, so
       certified_lower_bounds[i] ≤ V ≤ certified_upper_bounds[i]
    holds unconditionally. When a block's minimization leaves the self-concordant region
    where the correction is valid, the entry is -inf (lower) or inf (upper), a
    conservative deferral rather than an unsound bound.

    The fields of ``BarrierDiagnostics`` are inherited; the new fields are below.

    Fields
    ------
     upper_bounds : tuple[float, ...]
        Closed-form upper bound U on the saddle value after each outer step.
     lower_bounds : tuple[float, ...]
        Closed-form lower bound L on the saddle value after each outer step.
     certified_upper_bounds : tuple[float, ...]
        Guaranteed upper bound after each outer step, inf where deferred; empty when the
        certified dual was not evaluated.
     certified_lower_bounds : tuple[float, ...]
        Guaranteed lower bound after each outer step, -inf where deferred; empty when
        the certified dual was not evaluated.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    upper_bounds: tuple[float, ...] = ()
    lower_bounds: tuple[float, ...] = ()
    certified_upper_bounds: tuple[float, ...] = ()
    certified_lower_bounds: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the saddle trajectories and value brackets to a JSON-ready dict.

        The certified bounds may contain ±inf deferrals. ``json.dumps`` encodes them as
        ``Infinity`` and ``-Infinity`` under its default ``allow_nan=True``, and
        ``json.loads`` reads them back.

        Returns
        -------
         d : dict[str, Any]
            The fields as lists, tagged ``"SaddleBarrierDiagnostics"``.

        """
        return {
            "__type__": "SaddleBarrierDiagnostics",
            "duality_gaps": list(self.duality_gaps),
            "inner_nits": list(self.inner_nits),
            "inner_btls_steps": [list(s) for s in self.inner_btls_steps],
            "inner_quadratic_from": list(self.inner_quadratic_from),
            "inner_suboptimalities": [list(s) for s in self.inner_suboptimalities],
            "upper_bounds": list(self.upper_bounds),
            "lower_bounds": list(self.lower_bounds),
            "certified_upper_bounds": list(self.certified_upper_bounds),
            "certified_lower_bounds": list(self.certified_lower_bounds),
        }

    def plot_bounds(self, ax: Axes | None = None) -> Axes:
        """Plot the upper and lower bounds on the saddle value against the outer step.

        Each bound is sampled once per outer centering step, so the two curves are drawn
        as step functions closing on the saddle value, with the bracket between them
        shaded. ``plot_convergence`` shows the same information as a single gap curve on
        a log scale. When present, the certified bracket is overlaid as dashed curves,
        with the ±inf deferrals masked out so that only steps with a guaranteed bound
        are drawn.

        Parameters
        ----------
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        Raises
        ------
         ValueError
            When ``upper_bounds`` and ``lower_bounds`` differ in length, when no bounds
            were recorded, or when certified bounds are present but do not match that
            length.

        """
        if len(self.upper_bounds) != len(self.lower_bounds):
            raise ValueError("upper_bounds and lower_bounds must have the same length")
        n = len(self.upper_bounds)
        if n == 0:
            raise ValueError("no value bounds recorded")
        if ax is None:
            _, ax = plt.subplots()
        xs = list(range(n))
        ax.fill_between(
            xs,
            list(self.lower_bounds),
            list(self.upper_bounds),
            step="post",
            alpha=0.2,
            label="bracket",
        )
        ax.step(xs, list(self.upper_bounds), where="post", label="upper bound")
        ax.step(xs, list(self.lower_bounds), where="post", label="lower bound")
        if self.certified_upper_bounds or self.certified_lower_bounds:
            if (
                len(self.certified_upper_bounds) != n
                or len(self.certified_lower_bounds) != n
            ):
                raise ValueError(
                    "certified bounds, when present, must match the belt length"
                )
            # Mask deferrals (+/-inf) with NaN; matplotlib skips NaN so the
            # dashed curves show only the steps with a guaranteed bound.
            cu = [
                b if np.isfinite(b) else float("nan")
                for b in self.certified_upper_bounds
            ]
            cl = [
                b if np.isfinite(b) else float("nan")
                for b in self.certified_lower_bounds
            ]
            ax.step(xs, cu, where="post", linestyle="--", label="certified upper bound")
            ax.step(xs, cl, where="post", linestyle="--", label="certified lower bound")
        ax.set_xlabel("Centering Step")
        ax.set_ylabel("Saddle Value")
        ax.legend()
        return ax


@dataclass(frozen=True, eq=False)
class PrimalDualDiagnostics(SolverDiagnostics):
    """Primal-dual trajectories, one entry per outer iteration.

    Fields
    ------
     surrogate_gaps : tuple[float, ...]
        Surrogate duality gap η̂ = -f^T λ at each iteration.
     primal_residual_norms : tuple[float, ...]
        Euclidean norm of the primal residual r_pri at each iteration.
     dual_residual_norms : tuple[float, ...]
        Euclidean norm of the dual residual r_dual at each iteration.
     step_sizes : tuple[float, ...]
        Step length taken at each iteration.

    """

    surrogate_gaps: tuple[float, ...] = ()
    primal_residual_norms: tuple[float, ...] = ()
    dual_residual_norms: tuple[float, ...] = ()
    step_sizes: tuple[float, ...] = ()

    def iteration_count(self) -> int:
        """Return the number of outer primal-dual iterations.

        Returns
        -------
         n : int
            The length of ``surrogate_gaps``.

        """
        return len(self.surrogate_gaps)

    def late_step_sizes(self, fraction: float = 0.25) -> tuple[float, ...]:
        """Return the step lengths over the last fraction of the solve.

        The primal-dual analogue of the barrier method's quadratic phase. There is no
        threshold to cross here; the health signal is that the step length approaches 1
        as the iterates converge, since a method still taking short steps at the end is
        being held back by something. Asserting on the tail rather than on a phase flag
        keeps the claim to what the method actually guarantees.

        Parameters
        ----------
         fraction : float
            Share of the trajectory to return, in (0, 1]. For n recorded steps the tail
            has
               max(1, floor(n × fraction))
            entries. Out-of-range values are rejected rather than clamped, since each
            has a plausible but wrong answer: 0 or a negative value would floor to a
            one-element tail, and a value above 1 (such as 25 meant as 25%) would return
            the whole trajectory.

        Returns
        -------
         steps : tuple[float, ...]
            The last entries of ``step_sizes``; empty when no steps were recorded.

        Raises
        ------
         ValueError
            When ``fraction`` is outside (0, 1].

        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"fraction must be in (0, 1], got {fraction}. It is the share "
                "of the trajectory to look at, not a count of iterations."
            )
        if not self.step_sizes:
            return ()
        n = max(1, int(len(self.step_sizes) * fraction))
        return self.step_sizes[-n:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the primal-dual trajectories to a JSON-ready dict.

        Returns
        -------
         d : dict[str, Any]
            The fields as lists, tagged ``"PrimalDualDiagnostics"``.

        """
        return {
            "__type__": "PrimalDualDiagnostics",
            "surrogate_gaps": list(self.surrogate_gaps),
            "step_sizes": list(self.step_sizes),
            "primal_residual_norms": list(self.primal_residual_norms),
            "dual_residual_norms": list(self.dual_residual_norms),
        }

    def plot_convergence(self, ax: Axes | None = None) -> Axes:
        """Plot the surrogate duality gap η̂ = -f^T λ against the iteration.

        The primal-dual method does not evaluate a true duality gap at each iteration;
        it tracks the cheap surrogate gap, and the axis is labeled accordingly. Pass
        ``ax`` to combine with ``plot_feasibility`` on a shared figure.

        Parameters
        ----------
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(range(len(self.surrogate_gaps)), self.surrogate_gaps, marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Surrogate Duality Gap")
        return ax

    def plot_feasibility(self, ax: Axes | None = None) -> Axes:
        """Plot the feasibility residual against the iteration.

        At each iteration the residual is
           r_feas = sqrt(‖r_pri‖^2 + ‖r_dual‖^2),
        the primal and dual parts of the KKT residual, excluding the centrality term
        (Boyd and Vandenberghe, 2004, section 11.7). Pass ``ax`` to combine with
        ``plot_convergence`` on a shared figure.

        Parameters
        ----------
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        Raises
        ------
         ValueError
            When ``primal_residual_norms`` and ``dual_residual_norms`` differ in length
            (raised by ``zip(..., strict=True)``).

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        if ax is None:
            _, ax = plt.subplots()
        r_feas = [
            float(np.hypot(d, p))
            for d, p in zip(
                self.dual_residual_norms, self.primal_residual_norms, strict=True
            )
        ]
        ax.plot(range(len(r_feas)), r_feas, marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feasibility Residual")
        return ax


@dataclass(frozen=True, eq=False)
class HsdDiagnostics(SolverDiagnostics):
    """Homogeneous self-dual trajectories, one entry per outer iteration.

    There is no step-length trajectory: the step length is chosen inside the inner cone
    driver and is not surfaced here. An always-empty field would read as "the solver
    took no steps" rather than "this solver does not report them".

    Fields
    ------
     duality_gaps : tuple[float, ...]
        Normalized complementarity gap μ of the embedding at each iteration.
     tau_history : tuple[float, ...]
        Homogenizing variable τ at each iteration.
     kappa_history : tuple[float, ...]
        Homogenizing variable κ at each iteration.

    """

    duality_gaps: tuple[float, ...] = ()
    tau_history: tuple[float, ...] = ()
    kappa_history: tuple[float, ...] = ()

    def iteration_count(self) -> int:
        """Return the number of outer homogeneous self-dual iterations.

        Returns
        -------
         n : int
            The length of ``duality_gaps``.

        """
        return len(self.duality_gaps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the homogeneous self-dual trajectories to a JSON-ready dict.

        Returns
        -------
         d : dict[str, Any]
            The fields as lists, tagged ``"HsdDiagnostics"``.

        """
        return {
            "__type__": "HsdDiagnostics",
            "duality_gaps": list(self.duality_gaps),
            "tau_history": list(self.tau_history),
            "kappa_history": list(self.kappa_history),
        }

    def plot_convergence(self, ax: Axes | None = None) -> Axes:
        """Plot the embedding's duality gap μ per iteration on a log scale.

        Parameters
        ----------
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(range(len(self.duality_gaps)), self.duality_gaps, marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Duality Gap")
        return ax

    def plot_homogeneity(self, ax: Axes | None = None) -> Axes:
        """Plot the homogenizing variables τ and κ per iteration on a log scale.

        In the homogeneous self-dual embedding, τ > 0 at convergence recovers the
        original optimum, while τ → 0 with κ bounded away from zero certifies
        infeasibility or unboundedness; τ κ → 0 is the embedding's complementarity
        condition. Plotting both makes the termination regime visible at a glance.

        The log scale silently omits nonpositive values. The solver terminates before τ
        or κ underflows to zero, so in practice every stored value is positive and the
        infeasibility regime stays visible.

        Parameters
        ----------
         ax : Axes | None
            Axes to draw on; ``None`` creates a new figure.

        Returns
        -------
         ax : Axes
            The axes drawn on.

        Raises
        ------
         ValueError
            When ``tau_history`` and ``kappa_history`` differ in length.

        """
        if len(self.tau_history) != len(self.kappa_history):
            raise ValueError("tau_history and kappa_history must have the same length")
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(
            range(len(self.tau_history)),
            self.tau_history,
            marker="o",
            label=r"$\tau$",
        )
        ax.plot(
            range(len(self.kappa_history)),
            self.kappa_history,
            marker="s",
            label=r"$\kappa$",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Homogenizing Variable")
        ax.legend()
        return ax


@dataclass(frozen=True, eq=False)
class ObjectiveOutcome:
    """Objective part of a result, present exactly when the problem had an objective.

    Fields
    ------
     value : float
        Objective value at the solution.
     suboptimality : float
        Bound on the achieved suboptimality, a nonnegative magnitude.
     dual_value : float | None
        Value of the Lagrangian dual at the solution, or ``None`` when the solver did
        not compute one.

    """

    value: float
    suboptimality: float
    dual_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict.

        Returns
        -------
         d : dict[str, Any]
            The three fields.

        """
        return {
            "value": self.value,
            "suboptimality": self.suboptimality,
            "dual_value": self.dual_value,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ObjectiveOutcome:
        """Reconstruct an outcome from a dict produced by ``to_dict``.

        Parameters
        ----------
         d : dict[str, Any]
            The serialized outcome.

        Returns
        -------
         outcome : ObjectiveOutcome
            The reconstructed outcome.

        Raises
        ------
         KeyError
            When an entry is missing.

        """
        return ObjectiveOutcome(
            value=d["value"],
            suboptimality=d["suboptimality"],
            dual_value=d["dual_value"],
        )

    @property
    def duality_gap(self) -> float | None:
        """Return the gap between the objective value and its dual bound, if known.

        Reported as a magnitude so that it reads the same in both optimization senses:
        for a minimization ``value`` lies above the lower bound ``dual_value``, while
        for a maximization (see ``negated``) it lies below the upper bound. For a
        minimization, where weak duality forces value ≥ dual_value, the gap equals
        ``value - dual_value``.

        Returns
        -------
         gap : float | None
            |value - dual_value|, or ``None`` when ``dual_value`` is ``None``.

        """
        if self.dual_value is None:
            return None
        return abs(self.value - self.dual_value)

    def negated(self) -> ObjectiveOutcome:
        """Return this outcome viewed from the opposite optimization sense.

        The solvers only minimize; ``Problem.maximize(g)`` has them minimize -g, so
        ``value`` and ``dual_value`` describe that minimization. Negating both recovers
        the maximization view: ``value`` becomes g(x_star) and ``dual_value`` the
        corresponding upper bound on g. ``suboptimality`` is a nonnegative magnitude and
        does not depend on the sense, so it carries over unchanged, as does
        ``duality_gap``.

        Returns
        -------
         outcome : ObjectiveOutcome
            The outcome with ``value`` and ``dual_value`` negated.

        """
        return ObjectiveOutcome(
            value=-self.value,
            suboptimality=self.suboptimality,
            dual_value=None if self.dual_value is None else -self.dual_value,
        )


@dataclass(frozen=True, eq=False)
class Duals:
    """Lagrange multipliers, each present exactly when the problem had such constraints.

    Fields
    ------
     equality_multipliers : npt.NDArray[np.float64] | None
        Multipliers ν of the equality constraints A x = b, or ``None``.
     inequality_multipliers : npt.NDArray[np.float64] | None
        Multipliers λ of the inequality constraints, or ``None``.

    """

    equality_multipliers: npt.NDArray[np.float64] | None = None
    inequality_multipliers: npt.NDArray[np.float64] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the multipliers to a JSON-ready dict.

        Returns
        -------
         d : dict[str, Any]
            The multipliers as lists, or ``None`` where absent.

        """
        return {
            "equality_multipliers": _encode_array(self.equality_multipliers),
            "inequality_multipliers": _encode_array(self.inequality_multipliers),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Duals:
        """Reconstruct the multipliers from a dict produced by ``to_dict``.

        Parameters
        ----------
         d : dict[str, Any]
            The serialized multipliers.

        Returns
        -------
         duals : Duals
            The reconstructed multipliers.

        Raises
        ------
         KeyError
            When an entry is missing.

        """
        return Duals(
            equality_multipliers=_decode_array(d["equality_multipliers"]),
            inequality_multipliers=_decode_array(d["inequality_multipliers"]),
        )


@dataclass(frozen=True, eq=False)
class InfeasibilityCertificate:
    """Witness that a problem is infeasible or unbounded.

    Fields
    ------
     kind : Literal["infeasible", "unbounded"]
        What the certificate proves.
     primal_witness : npt.NDArray[np.float64] | None
        Primal vector supporting the certificate, or ``None``.
     dual_witness : npt.NDArray[np.float64] | None
        Dual vector supporting the certificate, or ``None``.

    """

    kind: Literal["infeasible", "unbounded"]
    primal_witness: npt.NDArray[np.float64] | None = None
    dual_witness: npt.NDArray[np.float64] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the certificate to a JSON-ready dict.

        Returns
        -------
         d : dict[str, Any]
            The kind and the witnesses as lists, or ``None`` where absent.

        """
        return {
            "kind": self.kind,
            "primal_witness": _encode_array(self.primal_witness),
            "dual_witness": _encode_array(self.dual_witness),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> InfeasibilityCertificate:
        """Reconstruct a certificate from a dict produced by ``to_dict``.

        Parameters
        ----------
         d : dict[str, Any]
            The serialized certificate.

        Returns
        -------
         certificate : InfeasibilityCertificate
            The reconstructed certificate.

        Raises
        ------
         ValueError
            When ``kind`` is neither ``"infeasible"`` nor ``"unbounded"``.
         KeyError
            When an entry is missing.

        """
        kind = d["kind"]
        if kind not in ("infeasible", "unbounded"):
            raise ValueError(f"unknown certificate kind {kind!r}")
        return InfeasibilityCertificate(
            kind=kind,
            primal_witness=_decode_array(d["primal_witness"]),
            dual_witness=_decode_array(d["dual_witness"]),
        )


@dataclass(frozen=True, eq=False)
class SolveResult:
    """Result of a solve.

    The spine fields are always present. ``objective`` and ``duals`` are present
    according to the problem's shape, ``diagnostics`` according to the solver,
    ``certificate`` only when the solve certified infeasibility or unboundedness, and
    ``phase1`` only when a Phase I feasibility solve ran. The accessors
    (``objective_value``, ``dual_value``, ``equality_multipliers``, the typed
    diagnostics views, and so on) raise a specific error when the part they read is
    absent.

    Fields
    ------
     solution : npt.NDArray[np.float64]
        The final iterate.
     status : SolveStatus
        Outcome of the solve.
     message : str
        Human-readable description of the outcome.
     solver : str
        Name of the solver that ran.
     n_iterations : int
        Iteration count reported by the solver.
     objective : ObjectiveOutcome | None
        Objective part; ``None`` for a feasibility problem.
     duals : Duals | None
        Lagrange multipliers; ``None`` when the problem had no constraints or the solver
        did not report them.
     diagnostics : SolverDiagnostics | None
        Solver-method trajectories, whose type names the solver.
     certificate : InfeasibilityCertificate | None
        Certificate of infeasibility or unboundedness, when one was found.
     phase1 : SolveResult | None
        Result of the Phase I feasibility solve, when one ran.

    """

    solution: npt.NDArray[np.float64]
    status: SolveStatus
    message: str
    solver: str
    n_iterations: int
    objective: ObjectiveOutcome | None = None
    duals: Duals | None = None
    diagnostics: SolverDiagnostics | None = None
    certificate: InfeasibilityCertificate | None = None
    phase1: SolveResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict tagged with the concrete type.

        The parts (``objective``, ``duals``, ``diagnostics``, ``certificate``) and the
        nested ``phase1`` result are serialized in place. Each absent part is kept as
        ``None``, so its absence, itself information, survives the round trip.

        Returns
        -------
         d : dict[str, Any]
            The serialized result, tagged ``"SolveResult"``.

        """
        return {
            "__type__": "SolveResult",
            "solution": _encode_array(self.solution),
            "status": self.status.value,
            "message": self.message,
            "solver": self.solver,
            "n_iterations": self.n_iterations,
            "objective": None if self.objective is None else self.objective.to_dict(),
            "duals": None if self.duals is None else self.duals.to_dict(),
            "diagnostics": (
                None if self.diagnostics is None else self.diagnostics.to_dict()
            ),
            "certificate": (
                None if self.certificate is None else self.certificate.to_dict()
            ),
            "phase1": None if self.phase1 is None else self.phase1.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SolveResult:
        """Reconstruct the tagged ``SolveResult`` class from ``to_dict`` output.

        Dispatches on the ``"__type__"`` tag to ``SolveResult`` or
        ``TwoBlockSolveResult``, reconstructing each part and the nested ``phase1``
        result.

        Parameters
        ----------
         d : dict[str, Any]
            A dict produced by ``to_dict``.

        Returns
        -------
         result : SolveResult
            The reconstructed result, of the tagged class.

        Raises
        ------
         ValueError
            When the tag names no known result class or ``status`` is not a
            ``SolveStatus`` value. Also propagated from the parts' ``from_dict`` methods
            and from ``TwoBlockSolveResult`` validation.
         KeyError
            When a required entry is missing.

        """
        kind = d["__type__"]
        objective = d["objective"]
        duals = d["duals"]
        diagnostics = d["diagnostics"]
        certificate = d["certificate"]
        phase1 = d["phase1"]
        common = {
            "solution": np.asarray(d["solution"], dtype=np.float64),
            "status": SolveStatus(d["status"]),
            "message": d["message"],
            "solver": d["solver"],
            "n_iterations": d["n_iterations"],
            "objective": (
                None if objective is None else ObjectiveOutcome.from_dict(objective)
            ),
            "duals": None if duals is None else Duals.from_dict(duals),
            "diagnostics": (
                None
                if diagnostics is None
                else SolverDiagnostics.from_dict(diagnostics)
            ),
            "certificate": (
                None
                if certificate is None
                else InfeasibilityCertificate.from_dict(certificate)
            ),
            "phase1": None if phase1 is None else SolveResult.from_dict(phase1),
        }
        if kind == "SolveResult":
            return SolveResult(**common)
        elif kind == "TwoBlockSolveResult":
            return TwoBlockSolveResult(**common, x_dimension=d["x_dimension"])
        else:
            raise ValueError(f"unknown SolveResult type {kind!r}")

    def serialize(self) -> str:
        """Serialize the result to a JSON string.

        ±inf deferrals in the diagnostics are encoded under ``json``'s default
        ``allow_nan=True`` and round-trip through ``deserialize``.

        Returns
        -------
         s : str
            The JSON text, indented by 2.

        """
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def deserialize(s: str) -> SolveResult:
        """Reconstruct a result from ``serialize`` output.

        Parameters
        ----------
         s : str
            JSON text produced by ``serialize``.

        Returns
        -------
         result : SolveResult
            The reconstructed result, of the tagged class.

        Raises
        ------
         ValueError
            Propagated from ``from_dict``; also raised by ``json.loads`` for malformed
            text.
         KeyError
            Propagated from ``from_dict``.

        """
        return SolveResult.from_dict(json.loads(s))

    @property
    def succeeded(self) -> bool:
        """Check whether the solve reached a usable optimal or feasible point.

        Returns
        -------
         ok : bool
            True when ``status`` is ``OPTIMAL``, ``ACCEPTABLE`` or ``FEASIBLE``.

        """
        return self.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.ACCEPTABLE,
            SolveStatus.FEASIBLE,
        )

    @property
    def barrier_diagnostics(self) -> BarrierDiagnostics:
        """Return the diagnostics, narrowed to the barrier engine.

        A typed view of ``diagnostics`` for the caller that reads the barrier
        convergence trajectories. It raises rather than returning a wrongly typed value,
        so misuse fails loudly even under ``python -O``. A saddle barrier solve's
        diagnostics also qualify.

        Returns
        -------
         diagnostics : BarrierDiagnostics
            The diagnostics part.

        Raises
        ------
         WrongDiagnosticsError
            When ``diagnostics`` is not a ``BarrierDiagnostics``.

        """
        diagnostics = self.diagnostics
        if not isinstance(diagnostics, BarrierDiagnostics):
            raise WrongDiagnosticsError(
                f"not a barrier solve; diagnostics are {type(diagnostics).__name__}"
            )
        return diagnostics

    @property
    def saddle_barrier_diagnostics(self) -> SaddleBarrierDiagnostics:
        """Return the diagnostics, narrowed to the two-block saddle barrier engine.

        A typed view of ``diagnostics`` for the caller that reads the saddle value
        bracket (``upper_bounds`` and ``lower_bounds``). It raises rather than returning
        a wrongly typed value, so misuse fails loudly.

        Returns
        -------
         diagnostics : SaddleBarrierDiagnostics
            The diagnostics part.

        Raises
        ------
         WrongDiagnosticsError
            When ``diagnostics`` is not a ``SaddleBarrierDiagnostics``.

        """
        diagnostics = self.diagnostics
        if not isinstance(diagnostics, SaddleBarrierDiagnostics):
            raise WrongDiagnosticsError(
                f"not a saddle barrier solve; diagnostics are "
                f"{type(diagnostics).__name__}"
            )
        return diagnostics

    @property
    def primal_dual_diagnostics(self) -> PrimalDualDiagnostics:
        """Return the diagnostics, narrowed to the primal-dual engine.

        Returns
        -------
         diagnostics : PrimalDualDiagnostics
            The diagnostics part.

        Raises
        ------
         WrongDiagnosticsError
            When ``diagnostics`` is not a ``PrimalDualDiagnostics``.

        """
        diagnostics = self.diagnostics
        if not isinstance(diagnostics, PrimalDualDiagnostics):
            raise WrongDiagnosticsError(
                f"not a primal-dual solve; diagnostics are {type(diagnostics).__name__}"
            )
        return diagnostics

    @property
    def hsd_diagnostics(self) -> HsdDiagnostics:
        """Return the diagnostics, narrowed to the homogeneous self-dual engine.

        Returns
        -------
         diagnostics : HsdDiagnostics
            The diagnostics part.

        Raises
        ------
         WrongDiagnosticsError
            When ``diagnostics`` is not an ``HsdDiagnostics``.

        """
        diagnostics = self.diagnostics
        if not isinstance(diagnostics, HsdDiagnostics):
            raise WrongDiagnosticsError(
                f"not an HSD solve; diagnostics are {type(diagnostics).__name__}"
            )
        return diagnostics

    @property
    def objective_value(self) -> float:
        """Return the objective value at the solution.

        Returns
        -------
         value : float
            ``objective.value``.

        Raises
        ------
         NoObjectiveError
            When the result has no objective part (a feasibility solve).

        """
        if self.objective is None:
            raise NoObjectiveError(
                "this was a feasibility solve; it has no objective value"
            )
        return self.objective.value

    @property
    def suboptimality(self) -> float:
        """Return the achieved suboptimality bound at the solution.

        Returns
        -------
         suboptimality : float
            ``objective.suboptimality``.

        Raises
        ------
         NoObjectiveError
            When the result has no objective part (a feasibility solve).

        """
        if self.objective is None:
            raise NoObjectiveError(
                "this was a feasibility solve; it has no suboptimality"
            )
        return self.objective.suboptimality

    @property
    def dual_value(self) -> float:
        """Return the value of the Lagrangian dual at the solution.

        Returns
        -------
         dual_value : float
            ``objective.dual_value``.

        Raises
        ------
         NoObjectiveError
            When the result has no objective part (a feasibility solve).
         NoDualValueError
            When the solver did not compute a dual value.

        """
        if self.objective is None:
            raise NoObjectiveError("this was a feasibility solve; it has no dual value")
        if self.objective.dual_value is None:
            raise NoDualValueError("this solve did not compute a dual value")
        return self.objective.dual_value

    @property
    def equality_multipliers(self) -> npt.NDArray[np.float64]:
        """Return the Lagrange multipliers of the equality constraints.

        Returns
        -------
         nu : npt.NDArray[np.float64]
            ``duals.equality_multipliers``.

        Raises
        ------
         NoDualsError
            When the result has no duals part, or the problem had no equality
            constraints.

        """
        if self.duals is None:
            raise NoDualsError("this solve had no dual variables")
        if self.duals.equality_multipliers is None:
            raise NoDualsError("this solve had no equality constraints")
        return self.duals.equality_multipliers

    @property
    def inequality_multipliers(self) -> npt.NDArray[np.float64]:
        """Return the Lagrange multipliers of the inequality constraints.

        Returns
        -------
         lmbda : npt.NDArray[np.float64]
            ``duals.inequality_multipliers``.

        Raises
        ------
         NoDualsError
            When the result has no duals part, or the problem had no inequality
            constraints.

        """
        if self.duals is None:
            raise NoDualsError("this solve had no dual variables")
        if self.duals.inequality_multipliers is None:
            raise NoDualsError("this solve had no inequality constraints")
        return self.duals.inequality_multipliers


@dataclass(frozen=True, eq=False)
class TwoBlockSolveResult(SolveResult):
    """Result of a solve whose primal splits into an x-block and a y-block.

    A saddle-point (two-block stationarity) solve produces two coupled variables, an
    x-block and a y-block, rather than a single primal vector. This class keeps the
    spine's ``solution`` as the flat concatenation (x, y), so every generic
    ``SolveResult`` consumer works unchanged, and adds ``x`` and ``y`` views that split
    it at ``x_dimension``. Both saddle engines (``stationarity_solver`` and
    ``primal_dual_stationarity_solver``) return this type; which one ran is recorded in
    ``diagnostics``, as for ordinary solves. Build one with ``from_blocks``. The fields
    of ``SolveResult`` are inherited; the new field is below.

    Fields
    ------
     x_dimension : int
        Length of the x-block. Must lie in [0, len(solution)], or construction raises
        ``ValueError``.

    """

    x_dimension: int = 0

    def __post_init__(self) -> None:
        """Check that the block split lies within the solution vector.

        Raises
        ------
         ValueError
            When ``x_dimension`` is outside [0, len(solution)].

        """
        M = len(self.solution)
        if not 0 <= self.x_dimension <= M:
            raise ValueError(f"x_dimension {self.x_dimension} outside [0, {M}]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict, adding the block split point.

        Reuses ``SolveResult.to_dict`` and retags the concrete type, so that
        ``SolveResult.from_dict`` routes back to this class.

        Returns
        -------
         d : dict[str, Any]
            The serialized result, tagged ``"TwoBlockSolveResult"``, with
            ``x_dimension``.

        """
        d = super().to_dict()
        d["__type__"] = "TwoBlockSolveResult"
        d["x_dimension"] = self.x_dimension
        return d

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """Return the x-block of the solution.

        Returns
        -------
         x : npt.NDArray[np.float64]
            The first ``x_dimension`` entries of ``solution`` (a view, not a copy).

        """
        return self.solution[: self.x_dimension]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """Return the y-block of the solution.

        Returns
        -------
         y : npt.NDArray[np.float64]
            The entries of ``solution`` after the first ``x_dimension`` (a view, not a
            copy).

        """
        return self.solution[self.x_dimension :]

    @classmethod
    def from_blocks(
        cls,
        *,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        status: SolveStatus,
        message: str,
        solver: str,
        n_iterations: int,
        objective: ObjectiveOutcome | None = None,
        duals: Duals | None = None,
        diagnostics: SolverDiagnostics | None = None,
    ) -> TwoBlockSolveResult:
        """Build a two-block result from its x- and y-blocks.

        Concatenates the blocks into the flat ``solution`` and records the split point
        in ``x_dimension``, so that ``x`` and ``y`` recover them. The ``certificate``
        and ``phase1`` parts are left ``None``.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, converted to float64.
         y : npt.NDArray[np.float64]
            The y-block, converted to float64.
         status : SolveStatus
            Outcome of the solve.
         message : str
            Human-readable description of the outcome.
         solver : str
            Name of the solver that ran.
         n_iterations : int
            Iteration count reported by the solver.
         objective : ObjectiveOutcome | None
            Objective part, or ``None``.
         duals : Duals | None
            Lagrange multipliers, or ``None``.
         diagnostics : SolverDiagnostics | None
            Solver-method trajectories, or ``None``.

        Returns
        -------
         result : TwoBlockSolveResult
            The assembled result.

        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return cls(
            solution=np.concatenate([x, y]),
            status=status,
            message=message,
            solver=solver,
            n_iterations=n_iterations,
            objective=objective,
            duals=duals,
            diagnostics=diagnostics,
            x_dimension=x.shape[0],
        )
