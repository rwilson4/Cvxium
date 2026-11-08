"""Backtracking line searches and step-length caps for Cvxium's Newton solvers.

A backtracking line search starts from a feasible trial step and shrinks it until a
sufficient-decrease rule holds. This module separates the two concerns. A
``MeritFunction`` states the rule: ``ArmijoBarrierMerit`` is Armijo's condition on the
barrier objective, used by the barrier method, and ``ResidualNormMerit`` is the
residual-norm condition of the primal-dual method. ``LineSearch`` runs the loop: a
descent check, a cap from a caller-supplied maximum-step finder, then contraction until
the merit accepts.

The module also provides the fraction-to-boundary helpers that cap primal-dual steps so
that multipliers and slacks stay strictly positive: ``fraction_to_boundary``,
``mehrotra_boundary_factor`` and ``mehrotra_step_cap``.

"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ..exceptions import (
    ConstraintBoundaryError,
    InvalidDescentDirectionError,
    SevereCurvatureError,
)


class MeritFunction(ABC):
    """Abstract scalar merit deciding acceptance in a backtracking line search.

    A merit is constructed at the current iterate x, so the base-point merit value, the
    directional derivative and any cached intermediates are computed once. A line search
    then calls only ``evaluate`` at each trial step s and compares the result with
    ``accept_threshold(s)``, which encodes the sufficient-decrease rule.

    ``evaluate`` receives the step s alongside the trial point so that a merit for a
    solver stepping jointly in (x, λ), such as ``ResidualNormMerit``, can recompose the
    trial dual λ + s Δλ itself. A merit that depends only on x, such as
    ``ArmijoBarrierMerit``, ignores s.

    Each concrete merit is paired with a solver: ``ArmijoBarrierMerit`` with the barrier
    method, driven by ``LineSearch``; and ``ResidualNormMerit`` with the primal-dual
    method, which runs its own contraction loop.

    """

    @abstractmethod
    def evaluate(self, x_trial: npt.NDArray[np.float64], s: float) -> float:
        """Evaluate the merit at a trial point.

        Parameters
        ----------
         x_trial : npt.NDArray[np.float64]
            The trial point x + s Δx.
         s : float
            The trial step, for merits that also step other variables.

        Returns
        -------
         value : float
            The merit value at the trial step.

        """

    @abstractmethod
    def accept_threshold(self, s: float) -> float:
        """Return the largest merit value accepted at a trial step.

        Parameters
        ----------
         s : float
            The trial step.

        Returns
        -------
         threshold : float
            The step is accepted when ``evaluate`` returns a value no larger than this.

        """

    @abstractmethod
    def directional_derivative_sign_check(self) -> None:
        """Check that the search direction is a descent direction for the merit.

        ``LineSearch.find_step`` calls it once, before the contraction loop. It guards
        against numerical pathologies that produce an ascent direction, such as a nearly
        singular Hessian.

        Raises
        ------
         InvalidDescentDirectionError
            Implementations raise when the direction is not a descent direction.

        """

    @abstractmethod
    def required_improvement(self, s: float) -> float:
        """Return the merit decrease required to accept a trial step.

        A diagnostic, reported when the line search fails.

        Parameters
        ----------
         s : float
            The trial step.

        Returns
        -------
         required : float
            The decrease the sufficient-decrease rule demands at step s.

        """

    @abstractmethod
    def actual_improvement(self, trial_value: float) -> float:
        """Return the merit decrease a trial step achieved.

        A diagnostic, reported when the line search fails.

        Parameters
        ----------
         trial_value : float
            The merit value at the trial step, as returned by ``evaluate``.

        Returns
        -------
         actual : float
            The base-point merit value minus ``trial_value``.

        """


class ArmijoBarrierMerit(MeritFunction):
    """Armijo sufficient-decrease rule on the barrier objective f_t.

    A trial step s is accepted when
       f_t(x + s Δx) ≤ f_t(x) - α s λ^2,
    where λ^2 is the squared Newton decrement,
       λ^2 = Δx^T ∇^2 f_t(x) Δx,
    which equals -∇f_t(x)^T Δx at the exact Newton step and is positive for any descent
    direction. The caller supplies α λ^2 directly, because it already computes λ^2 for
    its inner stopping test.

    ``BaseInteriorPointMethodSolver.backtracking_line_search`` uses this merit with f_t
    the barrier objective. ``BarrierSolver`` also uses it, with a Lagrangian in place of
    f_t, when it evaluates the dual.

    Parameters
    ----------
     evaluate_barrier_objective : Callable[[npt.NDArray[np.float64]], float]
        Maps a trial point to f_t(x_trial). The barrier parameter t is captured by the
        closure, so the merit's interface stays independent of it.
     ft0 : float
        The value f_t(x) at the base point.
     neg_alpha_grad : float
        The product α λ^2, the required decrease per unit step.
     grad_ft_dot_delta_x : float
        The directional derivative ∇f_t(x)^T Δx, used only by the descent check.

    """

    def __init__(
        self,
        evaluate_barrier_objective: Callable[[npt.NDArray[np.float64]], float],
        ft0: float,
        neg_alpha_grad: float,
        grad_ft_dot_delta_x: float,
    ) -> None:
        self._evaluate_barrier_objective = evaluate_barrier_objective
        self._ft0 = ft0
        self._neg_alpha_grad = neg_alpha_grad
        self._grad_ft_dot_delta_x = grad_ft_dot_delta_x

    def evaluate(self, x_trial: npt.NDArray[np.float64], s: float) -> float:
        """Evaluate the barrier objective at a trial point.

        Parameters
        ----------
         x_trial : npt.NDArray[np.float64]
            The trial point x + s Δx.
         s : float
            The trial step; unused.

        Returns
        -------
         ft : float
            The barrier objective f_t(x_trial).

        """
        del s
        return self._evaluate_barrier_objective(x_trial)

    def accept_threshold(self, s: float) -> float:
        """Return the largest barrier value accepted at a trial step.

        Parameters
        ----------
         s : float
            The trial step.

        Returns
        -------
         threshold : float
            The bound f_t(x) - α s λ^2.

        """
        return self._ft0 - s * self._neg_alpha_grad

    def directional_derivative_sign_check(self) -> None:
        """Reject the search direction if it is an ascent direction.

        Raises
        ------
         InvalidDescentDirectionError
            When ∇f_t(x)^T Δx > 0. A zero directional derivative passes.

        """
        if self._grad_ft_dot_delta_x > 0:
            raise InvalidDescentDirectionError(
                message="Newton step was not a descent direction.",
                grad_ft_dot_delta_x=self._grad_ft_dot_delta_x,
            )

    def required_improvement(self, s: float) -> float:
        """Return the barrier decrease required at a trial step.

        Parameters
        ----------
         s : float
            The trial step.

        Returns
        -------
         required : float
            The decrease α s λ^2.

        """
        return s * self._neg_alpha_grad

    def actual_improvement(self, trial_value: float) -> float:
        """Return the barrier decrease a trial step achieved.

        Parameters
        ----------
         trial_value : float
            The barrier objective at the trial step.

        Returns
        -------
         actual : float
            The decrease f_t(x) - ``trial_value``.

        """
        return self._ft0 - trial_value


class ResidualNormMerit(MeritFunction):
    """Sufficient-decrease rule on the norm of the primal-dual residual.

    A trial step s is accepted when
       ‖r(x + s Δx, λ + s Δλ)‖ ≤ (1 - α s) ‖r(x, λ)‖,
    the line-search condition of the primal-dual interior-point method (Boyd and
    Vandenberghe, 2004, section 11.7.3), where r is the KKT residual at the current
    barrier parameter and ‖·‖ is the Euclidean norm.

    The trial residual depends jointly on x and λ (and, with equality constraints, on
    the equality multiplier ν), and the dual step is not encoded in the trial point. So
    the caller supplies ``compute_trial_residual_norm``, a closure that recomposes the
    full trial state from s and returns the residual norm, and ``evaluate`` ignores its
    trial point and calls that closure.

    The descent check is a no-op. At a point where the Newton system is nonsingular, the
    primal-dual direction is a descent direction for the residual norm by construction
    (the directional derivative is -‖r‖), so there is no scalar to test as
    ``ArmijoBarrierMerit`` does.

    Parameters
    ----------
     compute_trial_residual_norm : Callable[[float], float]
        Maps a trial step s to the norm of the KKT residual at
           (x + s Δx, λ + s Δλ),
        or at (x + s Δx, λ + s Δλ, ν + s Δν) with equality constraints. The caller owns
        the trial recomposition and the residual assembly.
     r_norm0 : float
        The residual norm at the base point.
     alpha : float
        The sufficient-decrease coefficient α, in (0, 1).

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    def __init__(
        self,
        compute_trial_residual_norm: Callable[[float], float],
        r_norm0: float,
        alpha: float,
    ) -> None:
        self._compute_trial_residual_norm = compute_trial_residual_norm
        self._r_norm0 = r_norm0
        self._alpha = alpha

    def evaluate(self, x_trial: npt.NDArray[np.float64], s: float) -> float:
        """Evaluate the trial residual norm at a trial step.

        Parameters
        ----------
         x_trial : npt.NDArray[np.float64]
            The trial point; unused, as the closure rebuilds the trial state.
         s : float
            The trial step.

        Returns
        -------
         r_norm : float
            The norm of the KKT residual at the trial step.

        """
        del x_trial
        return self._compute_trial_residual_norm(s)

    def accept_threshold(self, s: float) -> float:
        """Return the largest residual norm accepted at a trial step.

        Parameters
        ----------
         s : float
            The trial step.

        Returns
        -------
         threshold : float
            The bound (1 - α s) ‖r(x, λ)‖.

        """
        return (1.0 - self._alpha * s) * self._r_norm0

    def directional_derivative_sign_check(self) -> None:
        """Skip the descent check, which the Newton system guarantees.

        Never raises; see the class description.

        """
        return None

    def required_improvement(self, s: float) -> float:
        """Return the residual-norm decrease required at a trial step.

        Parameters
        ----------
         s : float
            The trial step.

        Returns
        -------
         required : float
            The decrease α s ‖r(x, λ)‖.

        """
        return self._alpha * s * self._r_norm0

    def actual_improvement(self, trial_value: float) -> float:
        """Return the residual-norm decrease a trial step achieved.

        Parameters
        ----------
         trial_value : float
            The residual norm at the trial step.

        Returns
        -------
         actual : float
            The decrease ‖r(x, λ)‖ - ``trial_value``.

        """
        return self._r_norm0 - trial_value


def fraction_to_boundary(
    v: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
) -> float:
    """Compute the largest step in (0, 1] that keeps a positive vector nonnegative.

    Assumes v > 0 componentwise. Returns 1 when no component decreases (no dv_i < 0);
    otherwise
       min(1, min over dv_i < 0 of -v_i / dv_i),
    the step at which the first decreasing component reaches zero, clamped to a full
    step. That step lands exactly on the boundary, so callers scale it by a factor below
    1 (see ``mehrotra_boundary_factor``) to stay strictly inside. The primal-dual
    solvers apply it to both the multipliers ``lmbda`` and the primal slacks -f.

    Parameters
    ----------
     v : npt.NDArray[np.float64]
        Current vector, strictly positive.
     dv : npt.NDArray[np.float64]
        Step direction, of the same length.

    Returns
    -------
     a : float
        The largest a in (0, 1] with v + a dv ≥ 0 componentwise.

    """
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def mehrotra_boundary_factor(eta: float, num_constraints: int, tau_min: float) -> float:
    """Compute the adaptive fraction-to-boundary factor γ for a primal-dual step.

    Returns
       γ = max(τ_min, 1 - μ),   μ = η / m,   m = max(``num_constraints``, 1),
    where η is the surrogate duality gap and μ the gap per multiplier. Far from the
    solution (μ ≥ 1 - τ_min) this is τ_min. Near the solution (μ → 0) γ approaches 1,
    allowing a nearly full step to the boundary, which keeps the step from stalling as
    the iterates converge. The rule follows (Mehrotra, 1992) and (Wright, 1997).

    Parameters
    ----------
     eta : float
        The surrogate duality gap η.
     num_constraints : int
        Number of inequality constraints m; zero is treated as one.
     tau_min : float
        The smallest factor τ_min, in (0, 1).

    Returns
    -------
     gamma : float
        The factor γ, in [τ_min, 1].

    References
    ----------
    Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method, SIAM
    Journal on Optimization 2, 1992.

    Wright, Stephen J., Primal-Dual Interior-Point Methods, SIAM, 1997.

    """
    m = max(num_constraints, 1)
    return max(tau_min, 1.0 - eta / m)


def mehrotra_step_cap(
    lmbda: npt.NDArray[np.float64],
    delta_lmbda: npt.NDArray[np.float64],
    eta: float,
    tau_min: float,
) -> float:
    """Cap a primal-dual step so that the multipliers stay strictly positive.

    Returns
       s = γ s_max,
    where γ is ``mehrotra_boundary_factor`` with m = ``lmbda.size``, and s_max is
    ``fraction_to_boundary(lmbda, delta_lmbda)``, the largest step in (0, 1] with
       λ + s_max Δλ ≥ 0.
    Far from the solution this is τ_min s_max. Near the solution (μ → 0) the cap
    approaches s_max, which keeps the step from pinning at the minimum step when an
    active multiplier is close to zero while Δλ keeps pulling it negative.

    Parameters
    ----------
     lmbda : npt.NDArray[np.float64]
        Current multipliers λ, strictly positive.
     delta_lmbda : npt.NDArray[np.float64]
        Multiplier step Δλ, of the same length.
     eta : float
        The surrogate duality gap η.
     tau_min : float
        The smallest factor τ_min, in (0, 1).

    Returns
    -------
     s : float
        The capped step, in (0, 1].

    References
    ----------
    Mehrotra, Sanjay, On the Implementation of a Primal-Dual Interior Point Method, SIAM
    Journal on Optimization 2, 1992.

    Wright, Stephen J., Primal-Dual Interior-Point Methods, SIAM, 1997.

    """
    return mehrotra_boundary_factor(eta, lmbda.size, tau_min) * fraction_to_boundary(
        lmbda, delta_lmbda
    )


class LineSearch:
    """Backtracking line search with a pluggable merit and maximum-step finder.

    ``find_step`` proceeds in three phases:
    1. Descent check. The merit raises ``InvalidDescentDirectionError`` when Δx is not
       a descent direction.
    2. Feasibility cap. With s_max = β ``max_step_finder(x, delta_x)``, raise
       ``ConstraintBoundaryError`` when s_max is below ``min_step``; otherwise start
       from s = min(``step_cap``, s_max).
    3. Armijo contraction. While the merit rejects s, raise ``SevereCurvatureError``
       if s is already below ``min_step``, and otherwise shrink s by the factor β.

    The caller supplies ``step_cap``, so an optional cap lives with the caller that
    computes it rather than inside the search. An example is the damped Newton step for
    self-concordant functions (Boyd and Vandenberghe, 2004, section 9.6.4),
       s ≤ 1 / (1 + λ).

    Parameters
    ----------
     merit : MeritFunction
        The sufficient-decrease rule.
     max_step_finder : Callable[[npt.NDArray, npt.NDArray], float]
        Maps (x, Δx), both float64 arrays, to the largest step α such that x + α Δx
        stays strictly feasible; inf when nothing binds.
     beta : float
        The shrink factor β, in (0, 1).
     min_step : float
        The smallest step tried before giving up. The accepted step can still be as
        small as β ``min_step``, since a step that shrinks below ``min_step`` is tried
        once.

    References
    ----------
    Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
    Press, 2004.

    """

    def __init__(
        self,
        merit: MeritFunction,
        max_step_finder: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], float
        ],
        beta: float,
        min_step: float,
    ) -> None:
        self._merit = merit
        self._max_step_finder = max_step_finder
        self._beta = beta
        self._min_step = min_step

    def find_step(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        step_cap: float = 1.0,
    ) -> float:
        """Find the first trial step the merit accepts, backtracking from the cap.

        Tries s_0 = min(``step_cap``, β s_feas), where s_feas is the maximum feasible
        step, then β s_0, β^2 s_0, and so on, returning the first step the merit
        accepts.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current iterate.
         delta_x : npt.NDArray[np.float64]
            Search direction, of the same length.
         step_cap : float
            Upper bound on the first trial step; 1 is the full Newton step.

        Returns
        -------
         s : float
            The accepted step.

        Raises
        ------
         InvalidDescentDirectionError
            Propagated from the merit's ``directional_derivative_sign_check``.
         ConstraintBoundaryError
            When β times the maximum feasible step is below ``min_step``.
         SevereCurvatureError
            When a trial step below ``min_step`` is rejected; carries the required and
            actual improvements at that step.

        """
        self._merit.directional_derivative_sign_check()

        s_max = self._beta * self._max_step_finder(x, delta_x)
        if s_max < self._min_step:
            raise ConstraintBoundaryError(
                message="Descent step takes us too close to constraint boundaries.",
            )

        s = min(step_cap, s_max)
        while True:
            x_trial = x + s * delta_x
            trial_value = self._merit.evaluate(x_trial, s)
            if trial_value <= self._merit.accept_threshold(s):
                return s
            if s < self._min_step:
                raise SevereCurvatureError(
                    message="Small step sizes did not adequately decrease objective.",
                    required_improvement=self._merit.required_improvement(s),
                    actual_improvement=self._merit.actual_improvement(trial_value),
                )
            s *= self._beta
