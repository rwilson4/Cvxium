"""Exceptions raised by Cvxium's solvers and their numerical building blocks.

The exceptions fall into four groups:
- ``ProblemInfeasibleError``, for a problem shown or judged to have no feasible point.
  ``optimization`` defines two subclasses, for a certified and for a suspected
  infeasibility.
- ``NewtonStepError``, for a Newton step that cannot be computed, typically because a
  structured solve met a matrix that is not positive definite.
- ``BacktrackingLineSearchError`` and its subclasses, for each way the backtracking
  line search can fail to accept a step.
- ``OptimizationError`` and its subclasses, for a solve that stops before reaching its
  tolerance. These carry the last (or best) iterate so a caller can recover a partial
  solution.

``ConeBoundaryError`` stands apart: a cone operation that needs a strictly interior
point was given one on the boundary.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .problems.result import SolveResult


class ProblemInfeasibleError(Exception):
    """Error raised when a problem has, or appears to have, no feasible point.

    Raised, for example, when the equality constraints A x = b have no solution because
    b is not in the range of A, or when Phase I finds no strictly interior point. The
    subclasses ``ProblemCertifiablyInfeasibleError`` and
    ``ProblemMarginallyFeasibleError`` distinguish a proof of infeasibility from a
    failure to find a feasible point.

    """


class NewtonStepError(Exception):
    """Error raised when a Newton step cannot be computed.

    Raised by the structured linear-system solvers when the purported Hessian (or KKT
    system) is not positive definite, is singular, or yields a non-finite result. A
    solver that catches it can fall back rather than take a corrupt step.

    Parameters
    ----------
     message : str
        Description of the failure.

    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        """Format the error message for display.

        Returns
        -------
         message : str
            The message given at construction.

        """
        return self.message


class BacktrackingLineSearchError(Exception):
    """Base error for a backtracking line search that cannot accept a step.

    The line search raises one of the subclasses, which name the cause:
    ``ConstraintBoundaryError``, ``InvalidDescentDirectionError`` or
    ``SevereCurvatureError``. Catch this class to handle any of them.

    Parameters
    ----------
     message : str
        Description of the failure.

    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        """Format the error message for display.

        Returns
        -------
         message : str
            The message given at construction.

        """
        return self.message


class ConstraintBoundaryError(BacktrackingLineSearchError):
    """Error raised when even tiny steps would leave the feasible region.

    Raised by the line search when the largest step that keeps every constraint strictly
    satisfied is below the minimum step size, so the iterate is pinned against a
    constraint boundary.

    Parameters
    ----------
     message : str
        Description of the failure.

    """


class InvalidDescentDirectionError(BacktrackingLineSearchError):
    """Error raised when the Newton step is not a descent direction.

    A Newton step Δx should satisfy
       ∇f^T Δx ≤ 0.
    When it does not, the Hessian is usually nearly singular and the step was computed
    inaccurately.

    Parameters
    ----------
     message : str
        Description of the failure.
     grad_ft_dot_delta_x : float
        The directional derivative ∇f^T Δx, which was positive.

    """

    def __init__(self, message: str, grad_ft_dot_delta_x: float) -> None:
        super().__init__(message)
        self.grad_ft_dot_delta_x = grad_ft_dot_delta_x

    def __str__(self) -> str:
        """Format the error message for display.

        Returns
        -------
         message : str
            The message followed by the offending directional derivative.

        """
        msg = (
            f"{self.message} (∇f^T Δx = {self.grad_ft_dot_delta_x} > 0, "
            "but should be <= 0)"
        )
        return msg


class SevereCurvatureError(BacktrackingLineSearchError):
    """Error raised when no step size meets the sufficient-decrease condition.

    The line search backtracked to its minimum step size without the objective
    decreasing enough. Usually the linear approximation fails even for small steps, a
    sign of severe curvature.

    Parameters
    ----------
     message : str
        Description of the failure.
     required_improvement : float
        Decrease in the merit function required at the last step tried.
     actual_improvement : float
        Decrease actually achieved at that step.

    """

    def __init__(
        self,
        message: str,
        required_improvement: float,
        actual_improvement: float,
    ) -> None:
        super().__init__(message)
        self.required_improvement = required_improvement
        self.actual_improvement = actual_improvement

    def __str__(self) -> str:
        """Format the error message for display.

        Returns
        -------
         message : str
            The message followed by the required and actual improvement.

        """
        msg = (
            f"{self.message} (required improvement >= {self.required_improvement:.03g}"
            f"; actual improvement = {self.actual_improvement:.03g})"
        )
        return msg


class OptimizationError(Exception):
    """Base error for a solve that stops before reaching its tolerance.

    Carries the last iterate and how far it was from optimal, so a caller can inspect or
    reuse a partial solution.

    Parameters
    ----------
     message : str
        Description of the failure.
     suboptimality : float
        Bound on, or estimate of, the suboptimality of ``last_iterate`` when the solve
        stopped.
     last_iterate : npt.NDArray[np.float64]
        The iterate when the solve stopped.

    """

    def __init__(
        self,
        message: str,
        suboptimality: float,
        last_iterate: npt.NDArray[np.float64],
    ) -> None:
        self.message = message
        self.suboptimality = suboptimality
        self.last_iterate = last_iterate

    def __str__(self) -> str:
        """Format the error message for display.

        Returns
        -------
         message : str
            The message followed by the suboptimality.

        """
        msg = f"{self.message} (suboptimality = {self.suboptimality:.03g})"
        return msg


class CenteringStepError(OptimizationError):
    """Error raised when a centering step of the barrier method fails.

    Raised when the Newton step cannot be computed, the line search fails, or the
    centering Newton iterations do not converge. When available, the error also carries
    the multiplier estimates at the last iterate.

    Parameters
    ----------
     message : str
        Description of the failure.
     suboptimality : float
        Suboptimality bound of the centering subproblem at ``last_iterate``.
     last_iterate : npt.NDArray[np.float64]
        The iterate when the centering step stopped.
     equality_multipliers : npt.NDArray[np.float64] | None
        Equality multiplier estimates ν at ``last_iterate``, when available.
     inequality_multipliers : npt.NDArray[np.float64] | None
        Inequality multiplier estimates λ at ``last_iterate``, when available.

    """

    def __init__(
        self,
        message: str,
        suboptimality: float,
        last_iterate: npt.NDArray[np.float64],
        equality_multipliers: npt.NDArray[np.float64] | None = None,
        inequality_multipliers: npt.NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(message, suboptimality, last_iterate)
        self.equality_multipliers = equality_multipliers
        self.inequality_multipliers = inequality_multipliers


class ConeBoundaryError(ValueError):
    """Error raised when a cone-interior operation receives a boundary point.

    Raised by the cone primitives (for example ``SecondOrderCone.nt_scaling`` and
    ``factored_w_squared``) when x or s lies on the cone boundary, where the
    Nesterov-Todd scaling is undefined. Subclassing ``ValueError`` keeps it catchable by
    callers that catch ``ValueError`` broadly, while letting others tell boundary
    degeneracy apart from malformed input.

    """


class InteriorPointMethodError(OptimizationError):
    """Error raised when the interior-point method cannot reach its tolerance.

    When available, the error carries the best primal-dual result found before stopping.
    That result may not meet the soft tolerance, but it is often good enough; inspect
    its objective and duality gap before re-running with a looser tolerance.

    Parameters
    ----------
     message : str
        Description of the failure.
     remaining_steps : int
        Number of outer barrier steps left in the schedule when the method stopped.
     suboptimality : float
        Bound on the suboptimality of ``last_iterate``.
     last_iterate : npt.NDArray[np.float64]
        The iterate when the method stopped.
     result : SolveResult | None
        The best result found before stopping, when available.

    """

    def __init__(
        self,
        message: str,
        remaining_steps: int,
        suboptimality: float,
        last_iterate: npt.NDArray[np.float64],
        result: SolveResult | None = None,
    ) -> None:
        super().__init__(message, suboptimality, last_iterate)
        self.remaining_steps = remaining_steps
        # Best primal/dual result found before failing, when available. The
        # solver may not have reached ``tolerance_soft``, but the attached result
        # is often more than good enough -- callers are encouraged to inspect it
        # (objective, duality gap) before re-running with a looser tolerance.
        self.result: SolveResult | None = result

    def __str__(self) -> str:
        """Format the error message for display.

        Returns
        -------
         message : str
            The message followed by the remaining steps and the suboptimality.

        """
        msg = (
            f"{self.message} ({self.remaining_steps} step(s) remaining; "
            f"suboptimality = {self.suboptimality:.03g})"
        )
        return msg
