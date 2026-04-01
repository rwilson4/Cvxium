"""Base optimization classes."""

import time
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy import linalg

from .exceptions import (
    BacktrackingLineSearchError,
    CenteringStepError,
    ConstraintBoundaryError,
    InteriorPointMethodError,
    InvalidDescentDirectionError,
    NewtonStepError,
    ProblemInfeasibleError,
    SevereCurvatureError,
)


@dataclass
class OptimizationSettings:
    """Optimization settings.

    Parameters
    ----------
    outer_tolerance : float, default=1e-6
        The tolerance level for convergence of the overall optimization problem. It
        controls how close the solution must be to the actual optimal point in terms of
        the objective function value.
    outer_tolerance_soft : float, default=1e-3
        Interior Point Methods can struggle to solve problems to high accuracy, and a
        medium accuracy solution is often acceptable. Still, if we're able to solve the
        problem to high accuracy, we'll try to do so. `other_tolerance_soft` is the
        "real" threshold, while `outer_tolerance` is the aspirational threshold.
    barrier_multiplier : float, default=10.0
        The factor by which the barrier parameter is multiplied at each outer iteration
        to help guide the optimization process towards feasible solutions, balancing
        between the penalty for constraint violation and descent direction.
    inner_tolerance : float, default=1e-6
        The tolerance for convergence within inner iterations of the optimization
        method. This defines when the inner problem is considered solved to a
        satisfactory level.
    inner_tolerance_soft : float, default=1e-4
        A less strict tolerance for the inner iterations, which can be used as a
        fallback for allowing convergence when high precision is not achievable. This
        tolerance can be employed on steps that do not need strict adherence to
        constraints.
    max_inner_iterations : int, default=100
        The maximum number of iterations allowed for the inner optimization loop. This
        guards against infinite loops and over-computation in the case where convergence
        is slow.
    backtracking_alpha : float, default=0.01
        The coefficient used in backtracking line search to control the sufficient
        decrease condition. This value scales the gradient term to adjust the step size
        based on actual progress.
    backtracking_beta : float, default=0.5
        The factor used to reduce the step size in the backtracking line search. A value
        less than 1 encourages more conservative adjustments to the step size for
        stability.
    backtracking_min_step : float, default=1e-3
        The minimum allowable step size for backtracking line search. This prevents
        excessively small steps that could result in numerical issues or ineffective
        exploration of the optimization landscape.
    verbose : bool, default=False
        If True, print status along with how long it took to execute each step.

    """

    outer_tolerance: float = 1e-6
    outer_tolerance_soft: float = 1e-3
    barrier_multiplier: float = 10.0
    inner_tolerance: float = 1e-6
    inner_tolerance_soft: float = 1e-4
    max_inner_iterations: int = 1_000
    backtracking_alpha: float = 0.01
    backtracking_beta: float = 0.5
    backtracking_min_step: float = 1e-3
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Wrapper for generic optimization result."""

    solution: npt.NDArray[np.float64]


@dataclass
class NewtonResult(OptimizationResult):
    """Wrapper for the results of Newton's method.

    Parameters
    ----------
     solution : vector
        The solution.
     objective_value : float
        Objective value.
     dual_value : float
        Value of Lagrangian dual function.
     equality_multipliers, inequality_multipliers: vectors
        Lagrange multipliers for constraints.
     suboptimalities : List[float]
        Suboptimality of solution at each iteration.
     nits : int
        Number of iterations before convergence.
     status : [0, 1]
        Solution status:
          0 : method completed successfully
          1 : method failed to converge to the desired tolerance, but achieved an
              acceptable tolerance
          2 : feasibility method successfully found a feasible point
     message : str
          Summary of result.

    """

    objective_value: float
    barrier_objective_value: float
    dual_value: float
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimalities: list[float]
    nits: int
    status: Literal[0, 1, 2]
    message: str

    def plot_convergence(self, ax: Axes | None = None) -> Axes:
        """Plot convergence."""
        if ax is None:
            _, ax = plt.subplots()

        ax.plot([ii + 1 for ii in range(self.nits)], self.suboptimalities, marker="o")
        plt.yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sub-Optimality")
        return ax


@dataclass
class InteriorPointMethodResult(OptimizationResult):
    """Wrapper for the results of the Interior Point Method.

    Parameters
    ----------
    solution : vector
        The optimal weights.
    objective_value : float
        Distance between optimal weights and baseline.
    dual_value : float
        Dual function, evaluated at optimal weights and Lagrange multipliers.
    equality_multipliers, inequality_multipliers: vectors
        Optimal Lagrange multipliers for constraints.
    suboptimality : float
        Suboptimality of solution.
    nits : int
        The number of outer iterations performed during the optimization
        process, indicating how many times the centering step was executed.
    inner_nits : List[int]
        Number of inner iterations performed during each centering step.
    status : int
        Solution status:
          0 : method completed successfully
          1 : method failed to converge to the desired tolerance, but achieved an
              acceptable tolerance
          2 : feasibility method successfully found a feasible point
     message : str
          Summary of result.

    """

    objective_value: float
    dual_value: float
    equality_multipliers: npt.NDArray[np.float64]
    inequality_multipliers: npt.NDArray[np.float64]
    suboptimality: float
    duality_gaps: list[float]
    nits: int
    inner_nits: list[int]
    inner_suboptimalities: list[list[float]]
    status: Literal[0, 1, 2]
    message: str
    phase1_res: OptimizationResult | None = None

    def plot_convergence(self, ax: Axes | None = None) -> Axes:
        """Plot convergence."""
        if ax is None:
            _, ax = plt.subplots()

        ax.stairs(
            values=self.duality_gaps,
            edges=list(range(self.nits + 1)),
            baseline=None,
        )
        plt.yscale("log")
        ax.set_xlabel("Newton Iterations")
        ax.set_ylabel("Duality Gap")
        return ax

    def plot_newton_convergence(self, it: int, ax: Axes | None = None) -> Axes:
        """Plot convergence."""
        if ax is None:
            _, ax = plt.subplots()

        if it < 1 or it > len(self.inner_nits):
            raise ValueError(
                f"`it` must be between 1 and {len(self.inner_nits)}, inclusive"
            )

        nits = self.inner_nits[it - 1]
        suboptimalities = self.inner_suboptimalities[it - 1]
        ax.plot([ii + 1 for ii in range(nits)], suboptimalities, marker="o")
        plt.yscale("log")
        ax.set_title(f"Convergence of Centering Step {it}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sub-Optimality")
        return ax


class ProblemCertifiablyInfeasibleError(ProblemInfeasibleError):
    """Raised when problem is certifiably infeasible."""

    def __init__(self, message: str, result: NewtonResult) -> None:
        self.message = message
        self.result = result

    def __str__(self) -> str:
        """Pretty-print error."""
        return self.message


class ProblemMarginallyFeasibleError(ProblemInfeasibleError):
    """Raised when problem may be infeasible, but we can't certify it as such."""

    def __init__(self, message: str, result: NewtonResult) -> None:
        self.message = message
        self.result = result

    def __str__(self) -> str:
        """Pretty-print error."""
        return self.message


class Optimizer(ABC):
    """Base class for an optimizer."""

    def __init__(
        self,
        phase1_solver: "PhaseISolver | None" = None,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimizer."""
        self.phase1_solver = phase1_solver
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

    @abstractproperty
    def num_eq_constraints(self) -> int:
        """Count equality constraints."""

    @abstractproperty
    def num_ineq_constraints(self) -> int:
        """Count inequality constraints."""

    @abstractmethod
    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Solve optimization problem.

        Parameters
        ----------
         x0 : vector, optional
            Initial guess.

        Returns
        -------
         res : OptimizationResult
            The solution.

        """


class PhaseISolver(Optimizer):
    """Base class for a PhaseISolver."""

    @abstractmethod
    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Solve optimization problem.

        Parameters
        ----------
         x0 : vector, optional
            Initial guess.
         fully_optimize : bool, optional
            If True, solve the problem to the full optimal point. Otherwise, return as
            soon as we have a feasible point. Defaults to False.

        Returns
        -------
         res : OptimizationResult
            The solution.

        """


class BaseInteriorPointMethodSolver(Optimizer):
    """Base class for Interior Point Methods."""

    def __init__(
        self,
        phase1_solver: "PhaseISolver | None" = None,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimizer."""
        self.phase1_solver = phase1_solver
        if settings is None:
            self.settings: OptimizationSettings = OptimizationSettings()
        else:
            self.settings = settings

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> OptimizationResult:
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    f0(x)
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.

        Parameters
        ----------
         x0 : vector
            Initial guess, intended to be feasible for some of the constraints, allowing
            the Phase I method to focus on a particular set of constraints. See Notes.
         fully_optimize : bool
            Interpretation differs for InteriorPointMethodSolver and
            PhaseIInteriorPointSolver instances.

        Returns
        -------
         res : InteriorPointMethodResult
            The results are wrapped in a InteriorPointMethodResult class, which includes
            a feasible point and other helpful info.

        """
        if self.phase1_solver is None:
            raise ValueError("PhaseISolver not specified.")

        # The nested Phase I Solver should return x such that A * x = b and fi(x) < 0,
        # i=1, ..., M.
        phase1_res = self.phase1_solver.solve(x0=x0, **kwargs)
        x = self.augment_previous_solution(phase1_res)

        # Applicable only for Phase I methods.
        if not fully_optimize and self.is_feasible(x):
            if self.settings.verbose:
                print(
                    f"  Phase I solution was feasible so we're done ({self.__class__.__name__})"
                )
            return phase1_res

        t = self.initialize_barrier_parameter(x0=x)
        num_steps = (
            int(
                np.ceil(
                    (
                        np.log(self.num_ineq_constraints)
                        - np.log(t * self.settings.outer_tolerance)
                    )
                    / np.log(self.settings.barrier_multiplier)
                )
            )
            + 1
        )
        if self.settings.verbose:
            overall_start_time = time.time()
            print(f"  Starting IPM ({self.__class__.__name__})")

        inner_nits = []
        inner_suboptimalities = []
        duality_gaps = []
        status: Literal[0, 1, 2] = 0
        message = (
            "Interior Point Method completed successfully to the desired tolerance"
        )
        equality_multipliers = np.zeros(1)
        inequality_multipliers = np.zeros(1)
        for nit in range(num_steps):
            if self.settings.verbose:
                print(f"  {nit + 1:02d} Beginning centering step with {t=:}")
                start_time = time.time()

            try:
                result = self.centering_step(
                    x,
                    t,
                    last_step=(nit + 1 == num_steps),
                    fully_optimize=fully_optimize,
                )
            except CenteringStepError as e:
                suboptimality = (
                    self.num_ineq_constraints * self.settings.barrier_multiplier / t
                )
                if nit > 1 and suboptimality < self.settings.outer_tolerance_soft:
                    # Convergence was good enough
                    x = e.last_iterate
                    status = 1
                    message = (
                        "Interior Point Method reached an acceptable precision but "
                        f"then ran into numerical difficulties -- {e.__str__()}"
                    )
                    nu = e.equality_multipliers
                    lmbda = e.inequality_multipliers
                    if nu is not None and lmbda is not None:
                        duality_gaps.append(
                            self.evaluate_objective(x)
                            - self.evaluate_dual(
                                lmbda=lmbda,
                                nu=nu,
                                x_star=x,
                            )
                        )

                    break

                raise InteriorPointMethodError(
                    message="Centering step failed",
                    remaining_steps=num_steps - nit - 1,
                    suboptimality=self.num_ineq_constraints
                    * self.settings.barrier_multiplier
                    / t,
                    last_iterate=e.last_iterate,
                ) from e

            if self.settings.verbose:
                end_time = time.time()
                print(
                    f"  {nit + 1:02d} Centering step completed in "
                    f"{1000 * (end_time - start_time):.03f} ms"
                )

            x = result.solution
            equality_multipliers = result.equality_multipliers  # type: ignore[assignment]
            inequality_multipliers = result.inequality_multipliers  # type: ignore[assignment]
            inner_nits.append(result.nits)
            inner_suboptimalities.append(result.suboptimalities)
            duality_gaps.append(result.objective_value - result.dual_value)

            # Applicable only for Phase I methods.
            if not fully_optimize and self.is_feasible(x):
                status = 2
                message = "Feasibility method successfully found a feasible point"
                if self.settings.verbose:
                    print(
                        f"  {nit + 1:02d} Result was strictly feasible so we're "
                        "early-stopping."
                    )
                break

            # Dual can provide certificate of infeasibility, in which case we can quit
            # faster. Applicably only for Phase I methods.
            if not fully_optimize:
                self.check_for_infeasibility(result)

            t *= self.settings.barrier_multiplier

        if self.settings.verbose:
            overall_end_time = time.time()
            print(
                f"  IPM completed in "
                f"{1000 * (overall_end_time - overall_start_time):.03f} ms"
                f" ({self.__class__.__name__})"
            )

        # Applicable only for Phase I methods.
        if not fully_optimize and not self.is_feasible(x):
            raise ProblemMarginallyFeasibleError(
                message=(
                    "Problem may be infeasible: dual value was "
                    f"{result.dual_value} < 0, but last centering step resulted in "
                    f"a point with value {result.objective_value}"
                ),
                result=result,
            )

        return InteriorPointMethodResult(
            solution=self.finalize_solution(x),
            objective_value=self.evaluate_objective(x),
            dual_value=self.evaluate_dual(
                lmbda=inequality_multipliers,
                nu=equality_multipliers,
                x_star=x,
            ),
            equality_multipliers=equality_multipliers,
            inequality_multipliers=inequality_multipliers,
            suboptimality=self.num_ineq_constraints
            * self.settings.barrier_multiplier
            / t,
            duality_gaps=duality_gaps,
            nits=nit + 1,
            inner_nits=inner_nits,
            inner_suboptimalities=inner_suboptimalities,
            status=status,
            message=message,
            phase1_res=phase1_res,
        )

    def augment_previous_solution(
        self,
        phase1_res: OptimizationResult,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Initialize variable based on Phase I result."""
        return phase1_res.solution

    def finalize_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """De-augment solution."""
        return x

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        """Initialize barrier parameter.

        We can often skip early centering steps by initializing the barrier parameter to
        a higher value. See the discussion in Boyd and Vandenberghe, section 11.3.1.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Initial guess. Must be strictly feasible.

        Returns
        -------
         t : float
            Barrier parameter.

        """
        return 1.0

    def centering_step(
        self,
        x0: npt.NDArray[np.float64],
        t: float,
        last_step: bool,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> NewtonResult:
        r"""Solve centering step.

        The centering step solves:
          minimize   ft(x) := t * f0(x) - \sum_i log(-fi(x))
          subject to A * x = b.

        Parameters
        ----------
         x0 : vector
            Initial guess, [w0; s0]. Must be strictly feasible.
         t : float
            Barrier parameter.
         last_step: bool
             Indicates whether this is the last centering step. See Notes.

        Returns
        -------
         res : NewtonResult
            The results are wrapped in a NewtonResult class, which includes the optimal
            weights and other helpful info.

        Notes
        -----
        Uses Newton's method with a feasible starting point. It isn't always possible to
        solve this to high precision, so we use a "soft" threshold as a fallback. If we
        can solve the problem to high precision, great; otherwise we're content if we
        have solved it to medium precision.

        The exception is on the very last step, where the overall suboptimality of the
        interior point method relies on the last centering step being solved to high
        precision. So this "soft" threshold does not apply in this last step.

        """
        x = x0.copy()
        eta = 0.25 * (1.0 - 2 * self.settings.backtracking_alpha)
        in_quadratic_phase = False
        suboptimality = np.inf
        suboptimalities = []
        for nit in range(self.settings.max_inner_iterations):
            if self.settings.verbose:
                start_time = time.time()

            try:
                delta_x, nu_hat = self.calculate_newton_step(x, t)
            except NewtonStepError as e:
                raise CenteringStepError(
                    message="Failed to calculate Newton step.",
                    suboptimality=suboptimality,
                    last_iterate=x,
                ) from e

            if self.settings.verbose:
                end_time = time.time()

            # Check for convergence
            lambda_squared = self.newton_decrement_squared(x, t, delta_x)
            suboptimality = 0.5 * lambda_squared
            suboptimalities.append(suboptimality)
            if self.settings.verbose:
                if not in_quadratic_phase and np.sqrt(lambda_squared) <= eta:
                    in_quadratic_phase = True
                    quadratic_convergence = " (quadratic convergence threshold)"
                else:
                    quadratic_convergence = ""

                print(
                    f"    {nit + 1:02d} Newton step calculated in "
                    f"{1000 * (end_time - start_time):.03f} ms; "
                    f"Sub-optimality {suboptimality} = 0.5 * {np.sqrt(lambda_squared)}^2"
                    f"{quadratic_convergence}"
                )

            if suboptimality < self.settings.inner_tolerance:
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(x, t, delta_x)
                return NewtonResult(
                    solution=x,
                    objective_value=self.evaluate_objective(x),
                    barrier_objective_value=self.evaluate_barrier_objective(x, t),
                    dual_value=self.evaluate_dual(
                        lmbda=lambda_star, nu=nu_star, x_star=x
                    ),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimalities=suboptimalities,
                    nits=nit + 1,
                    status=0,
                    message=(
                        "Newton's method completed successfully to the desired "
                        "tolerance."
                    ),
                )

            # Update
            try:
                btls_s = self.backtracking_line_search(
                    x=x, delta_x=delta_x, t=t, lambda_squared=lambda_squared
                )
            except BacktrackingLineSearchError as e:
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(
                    x, t, delta_x, centering_step_solved_perfectly=False
                )
                if not last_step and suboptimality < self.settings.inner_tolerance_soft:
                    return NewtonResult(
                        solution=x,
                        objective_value=self.evaluate_objective(x),
                        barrier_objective_value=self.evaluate_barrier_objective(x, t),
                        dual_value=self.evaluate_dual(
                            lmbda=lambda_star, nu=nu_star, x_star=x
                        ),
                        equality_multipliers=nu_star,
                        inequality_multipliers=lambda_star,
                        suboptimalities=suboptimalities,
                        nits=nit + 1,
                        status=1,
                        message=(
                            "Newton's method achieved an acceptable tolerance but then "
                            f"ran into numerical issues -- {e.__str__()}"
                        ),
                    )

                raise CenteringStepError(
                    message=f"Backtracking line search failed -- {e.__str__()}",
                    suboptimality=suboptimality,
                    last_iterate=x,
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                ) from e

            if self.settings.verbose:
                ft = self.evaluate_barrier_objective(x, t)
                ft_new = self.evaluate_barrier_objective(x + btls_s * delta_x, t)
                expected_improvement = (
                    self.settings.backtracking_alpha
                    * self.settings.backtracking_beta
                    * lambda_squared
                    / (1 + np.sqrt(lambda_squared))
                )
                print(
                    f"    {nit + 1:02d} {btls_s=:}, improvement={ft - ft_new}, "
                    f"expected improvement = {expected_improvement}"
                )

            # Applicable only for Phase I methods.
            if not fully_optimize and self.is_feasible(x + btls_s * delta_x):
                nu_star = nu_hat / t
                lambda_star = self.inequality_multipliers(
                    x, t, delta_x, centering_step_solved_perfectly=False
                )
                return NewtonResult(
                    solution=x + btls_s * delta_x,
                    objective_value=self.evaluate_objective(x + btls_s * delta_x),
                    barrier_objective_value=self.evaluate_barrier_objective(
                        x + btls_s * delta_x, t
                    ),
                    dual_value=self.evaluate_dual(
                        lmbda=lambda_star, nu=nu_star, x_star=x
                    ),
                    equality_multipliers=nu_star,
                    inequality_multipliers=lambda_star,
                    suboptimalities=suboptimalities,
                    nits=nit + 1,
                    status=1,
                    message=("Newton's method found a feasible point."),
                )

            x += btls_s * delta_x

        raise CenteringStepError(
            "Centering step did not converge.",
            suboptimality=suboptimality,
            last_iterate=x,
        )

    def backtracking_line_search(
        self,
        x: npt.NDArray[np.float64],
        delta_x: npt.NDArray[np.float64],
        t: float,
        lambda_squared: float,
    ) -> float:
        """Perform backtracking line search.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current estimate.
         delta_x : npt.NDArray[np.float64]
            Descent direction.
         t : float
            Barrier parameter.
         lambda_squared : float
            Square of Newton decrement. As the name suggests, this should be a positive number.

        Returns
        -------
         btls_s : float
            Step modifier.

        """
        if (grad_ft_dot_delta_x := np.dot(delta_x, self.gradient_barrier(x, t))) > 0:
            raise InvalidDescentDirectionError(
                message="Newton step was not a descent direction.",
                grad_ft_dot_delta_x=grad_ft_dot_delta_x,
            )

        alpha = self.settings.backtracking_alpha
        beta = self.settings.backtracking_beta
        min_step = self.settings.backtracking_min_step

        # Find the largest step size we could take while maintaining feasibility.
        # Note: for an exact line search, we'd want to minimize ft(x + btls_s * delta_x)
        # over the interval [0, btls_s_max].
        btls_s_max = beta * self.btls_keep_feasible(x, delta_x)
        if btls_s_max < min_step:
            raise ConstraintBoundaryError(
                message="Descent step takes us too close to constraint boundaries.",
            )

        # For backtracking line search, we ignore btls_s_max > 1.0
        btls_s = min(1.0, btls_s_max)

        # neg_alpha_grad is a positive number when lambda_squared > 0 (which it should
        # always be).
        neg_alpha_grad = alpha * lambda_squared
        ft = self.evaluate_barrier_objective(x, t)

        # btls_s_min = 1.0 / (1.0 + np.sqrt(lambda_squared))
        # ft_new = self.evaluate_barrier_objective(x + btls_s_min * delta_x, t)
        # if ft_new + btls_s_min * neg_alpha_grad > ft:
        #     print("Possible violation of self-concordance assumption detected")
        #     print(ft_new, ft, btls_s_min, neg_alpha_grad)

        while (
            ft_new := self.evaluate_barrier_objective(x + btls_s * delta_x, t)
        ) + btls_s * neg_alpha_grad > ft:
            if btls_s < min_step:
                raise SevereCurvatureError(
                    message="Small step sizes did not adequately decrease objective.",
                    required_improvement=btls_s * neg_alpha_grad,
                    actual_improvement=ft - ft_new,
                )
            btls_s *= beta

        return btls_s

    def newton_decrement_squared(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        delta_x: npt.NDArray[np.float64],
    ) -> float:
        """Calculate Newton decrement.

        The Newton decrement is the square root of:
           delta_x * H * delta_x,
        where H is the Hessian. This also equals the negative dot product of the barrier
        gradient and delta_x. For equality constrained problems, it does *not* equal:
           grad_ft * H^{-1} * grad_ft.

        """
        return max(0.0, np.dot(delta_x, self.hessian_multiply(x, t, delta_x)))

    def inequality_multipliers(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        delta_x: npt.NDArray[np.float64],
        centering_step_solved_perfectly: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Calculate Lagrange multipliers for inequality constraints."""
        lmbda = -1.0 / (t * self.constraints(x))
        if not centering_step_solved_perfectly:
            lmbda *= 1.0 - (
                self.grad_constraints_multiply(x, delta_x)
            ) / self.constraints(x)
        return lmbda

    @abstractmethod
    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate Newton step.

        Calculates Newton step for the "inner" problem:
          minimize   ft(x) := t * f0(x) - \sum_i log(-fi(x))
          subject to A * x = b.

        Parameters
        ----------
         x : vector
            Current estimate.
         t : float
            Barrier parameter.

        Returns
        -------
         delta_x : vector
            Newton step.
         nu : vector
            Lagrange multiplier associated with equality constraints.

        Notes
        -----
        The Newton step, delta_x, is the solution of the system:
           _       _   _       _     _         _
          | H   A^T | | delta_x |   | - grad_ft |
          | A    0  | |   nu    | = |      0    |
           -       -   -       -     -         -
        where H is the Hessian of ft evaluated at x, grad_f is the gradient of ft
        evaluated at x, and nu is the Lagrange multiplier associated with the equality
        constraints.

        """

    @abstractmethod
    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""

    def check_for_infeasibility(self, result: NewtonResult) -> None:
        """Check if infeasible.

        For some problems, the dual function can provide a "certificate of
        infeasibility". For these problems, this function should inspect the result of
        the centering step, and if the problem has proven infeasible, this function
        should raise a ProblemCertifiablyInfeasibleError. For other problems, it's fine
        to just let this function do nothing.

        """
        pass

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """Make sure x + btls_s * delta_x stays strictly feasible.

        This is provided as a convenience function, but it's typically possible to speed
        up this calculation by providing a custom implementation.

        """
        btls_s = 1.0
        while np.any(self.constraints(x + btls_s * delta_x) >= 0):
            btls_s *= self.settings.backtracking_beta
            if btls_s < self.settings.backtracking_min_step:
                raise ConstraintBoundaryError(
                    message="Descent step takes us too close to constraint boundaries.",
                )

        return btls_s

    @abstractmethod
    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate f0 at x."""

    def evaluate_barrier_objective(self, x: npt.NDArray[np.float64], t: float) -> float:
        """Calculate ft at x."""
        return t * self.evaluate_objective(x) - np.sum(np.log(-self.constraints(x)))

    @abstractmethod
    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of f0 at x."""

    def gradient_barrier(
        self, x: npt.NDArray[np.float64], t: float
    ) -> npt.NDArray[np.float64]:
        """Calculate gradient of ft at x."""
        return t * self.gradient(x) - (
            self.grad_constraints_transpose_multiply(x, 1.0 / self.constraints(x))
        )

    @abstractmethod
    def hessian_multiply(
        self, x: npt.NDArray[np.float64], t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Multiply H * y."""

    @abstractmethod
    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the vector of constraints, fi(x) <= 0."""

    @abstractmethod
    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the gradients of constraints, fi(x) <= 0."""

    def grad_constraints_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x) @ y.

        This is provided as a convenience function, but it's typically possible to speed
        up this calculation by providing a custom implementation.

        """
        return self.grad_constraints(x) @ y

    def grad_constraints_transpose_multiply(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate grad_constraints(x).T @ y.

        This is provided as a convenience function, but it's typically possible to speed
        up this calculation by providing a custom implementation.

        """
        return self.grad_constraints(x).T @ y

    @abstractmethod
    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate dual function.

        Parameters
        ----------
         lmbda : vector
            Lagrange multipliers for inequality constraints.
         nu : vector
            Lagrange multipliers for equality constraints.
         x_star : vector
            The solution.

        Returns
        -------
         g : float
            The dual function evaluated at lmbda, nu.

        Notes
        -----
        Ideally the implementation should support evaluating the dual at any points,
        lmbda and nu. However in practice evaluating the dual can be challenging, and we
        only really need to evaluate it at the optimal lambda_star and nu_star, in which
        case the dual is simply the Lagrangian evaluated at x_star, lambda_star, nu_star.

        """


class InteriorPointMethodSolver(BaseInteriorPointMethodSolver):
    """Solve an optimization problem using an Interior Point Method."""

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """Determine whether a feasible point has been found."""
        return False

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> InteriorPointMethodResult:
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    f0(x)
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.

        Parameters
        ----------
         x0 : vector
            Initial guess. If infeasible, a Phase I method will be used to find a
            feasible point.
         fully_optimize : bool
            Placeholder for Phase I methods. Doesn't do anything here.

        Returns
        -------
         res : InteriorPointMethodResult
            The results are wrapped in a InteriorPointMethodResult class, which includes
            the solution and other helpful info.

        """
        res = super().solve(x0, fully_optimize=True, **kwargs)

        # This is just a hack to tell the type checker that res has the desired type. By
        # construction, it always will.
        assert isinstance(res, InteriorPointMethodResult)
        return res


class PhaseIInteriorPointSolver(BaseInteriorPointMethodSolver, PhaseISolver):
    """Base class for a Phase I solver that uses an Interior Point Method.

    The main distinction is that a Phase I solver can terminate as soon as a feasible
    point is found.

    """

    def __init__(
        self,
        phase1_solver: "PhaseISolver | None" = None,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize optimizer."""
        super().__init__(phase1_solver=phase1_solver, settings=settings, **kwargs)

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> OptimizationResult:
        r"""Solve optimization problem.

        Uses an interior point method with a logarithmic barrier penalty to
        solve:
           minimize    0
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.
                       fj(x) <= 0, j=1, ..., N.

        Parameters
        ----------
         x0 : vector
            Initial guess, intended to be feasible for some of the constraints, allowing
            the Phase I method to focus on a particular set of constraints. See Notes.
         fully_optimize : bool
            If True, solve the underlying problem to full precision. Otherwise, return a
            feasible point as soon as we find one. See Notes.

        Returns
        -------
         res : InteriorPointMethodResult
            The results are wrapped in a InteriorPointMethodResult class, which includes
            a feasible point and other helpful info.

        Raises
        ------
         ProblemInfeasibleError: if a feasible point does not exist.

        Notes
        -----
        Given a point x0 satisfying A * x0 = b and fi(x0) < 0 for i=1, ..., M, solves:
           minimize    s
           subject to  A * x = b
                       fi(x) <= 0, i=1, ..., M.
                       fj(x) <= s, j=1, ..., N.

        If the solution, s^\star, is > 0, the problem is infeasible, and the Lagrange
        multipliers provide a certificate of this infeasibility. Otherwise, the solution
        x^\star is feasible for all constraints.

        We do not have to solve this problem to high precision. If we find feasible (x,
        s) with s < 0, we can quit.

        """
        return super().solve(x0, fully_optimize=fully_optimize, **kwargs)


class EqualityConstrainedInteriorPointMethodSolver(BaseInteriorPointMethodSolver):
    """Class for problems with equality constraints, A * x = b."""

    @abstractproperty
    def A(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Matrix representing equality constraints."""

    @abstractproperty
    def b(self) -> npt.NDArray[np.float64]:
        """Right-hand side of equality constraints."""

    def svd_A(  # noqa: N802
        self,
    ) -> tuple[
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
        npt.NDArray[np.float32 | np.float64],
    ]:
        """Calculate and cache SVD of A."""
        return linalg.svd(self.A, full_matrices=False)

    def initialize_barrier_parameter(self, x0: npt.NDArray[np.float64]) -> float:
        r"""Initialize barrier penalty.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Initial guess. Must be strictly feasible.

        Returns
        -------
         t : float
            Barrier parameter.

        Notes
        -----
        Per the discussion in Boyd and Vandenberghe, section 11.3.1, we can choose the
        initial barrier penalty, t, to minimize:
           inf_nu \| t * ∇f0 + ∇phi + A^T nu \|_2,
        where phi is the barrier penalty encoding the inequality constraints, and the
        gradients are evaluated at x0.

        This leads to the system of equations:
               A * A^T * nu_star +       A * ∇f0 * t_star = -A * ∇phi
           (A * ∇f0)^T * nu_star + \| ∇f0 \|_2^2 * t_star = -∇f0^T ∇phi.

        We solve this using the Schur complement, per the discussion in Boyd and
        Vandenberghe A.5.5. In this case,
           S = \| ∇f0 \|_2^2 - (A * ∇f0)^T * (A * A^T)^{-1} * (A * ∇f0),
        which is a scalar. Then:
           t_star = (1 / S) * (-∇f0^T * ∇phi + (A * ∇f0)^T (A * A^T)^{-1} * (A * ∇phi)).

        We do *not* assume that A is full rank, so A * A^T may be singular. Define z_f0
        and z_phi as any solutions of:
           (A * A^T) *  z_f0 = (A * ∇f0)
           (A * A^T) * z_phi = (A * ∇phi).
        We use the Singular Value Decomposition to select the minimum norm solution.

        Then
           S = \| ∇f0 \|_2^2 - (A * ∇f0)^T * z_f0,
        and
           t_star = (1 / S) * (-∇f0^T * ∇phi + (A * ∇f0)^T * z_phi).

        If A is full rank, S >= 0 since the problem is convex (and so the matrix
        corresponding to the system of equations is positive semidefinite). Yet due to
        numerical issues, sometimes S as calculated is slightly less than zero. For this
        reason, we clip S at 1e-16. In practice this seems to happen when x0 is very
        close to optimal.

        We always want to do at least one centering step, so we clip t_star at m / eps.

        """
        delta_f0 = self.gradient(x0)
        delta_phi = -self.grad_constraints_transpose_multiply(
            x0, 1.0 / self.constraints(x0)
        )

        A_delta_f0 = self.A @ delta_f0
        A_delta_phi = self.A @ delta_phi

        # z_phi = (A * A^T)^{-1} * A * delta_phi
        # Singular Value Decomposition: A = U * diag(s) * V^T,
        # where U is p-by-r (r is the rank of A), s is length r, V is M-by-r.
        # Since the columns of U and V are orthonormal, U^T U = V^T V = I_r,
        # but in general U * U^T does not equal I_p and V^T * V does not equal I_M.
        #
        # A * A^T = U * diag(s) * V^T * V * diag(s) * U^T
        #            = U * diag(s^2) * U^T
        # A * A^T * z_phi = A * delta_phi
        # -> U * diag(s^2) * U^T * z_phi = A * delta_phi
        # -> diag(s^2) * U^T * z_phi = U^T * A * delta_phi
        # -> U^T * z_phi = diag(s^{-2}) * U^T * A * delta_phi
        #
        # Now, in general U * U^T does not equal I_p, but if we *define*
        # z_phi = U * diag(s^{-2}) * U^T * A * delta_phi,
        # then U^T * z_phi = U^T * U * diag(s^{-2}) * U^T * A * delta_phi
        #                  = diag(s^{-2}) * U^T * A * delta_phi,
        # so this z_phi satisfies the equation, and is the minimum norm solution.
        #
        # When A has full row rank, z_phi is the *unique* solution to the equation,
        # but this strategy works even when A is rank deficient.
        U, s, Vh = self.svd_A()
        rank = int(np.sum(s > 1e-5))
        U = U[:, 0:rank]
        s = s[0:rank]
        z_phi = U @ ((U.T @ A_delta_phi) / np.square(s))
        z_f0 = U @ ((U.T @ A_delta_f0) / np.square(s))

        schur_complement = np.dot(delta_f0, delta_f0) - np.dot(A_delta_f0, z_f0)
        t_star_s = -np.dot(delta_f0, delta_phi) + np.dot(A_delta_f0, z_phi)

        return min(
            self.num_ineq_constraints / self.settings.outer_tolerance,
            max(
                super().initialize_barrier_parameter(x0),
                t_star_s / max(schur_complement, 1e-16),
            ),
        )


class UnconstrainedNewtonSolver(BaseInteriorPointMethodSolver):
    """Newton's method for unconstrained problems (no equality or inequality constraints).

    The barrier parameter is irrelevant without inequality constraints, so the outer IPM
    loop is skipped entirely. Instead, Newton's method is applied directly to f0(x),
    which is equivalent to a single centering step with t=1.

    Usage
    -----
    Subclass this and implement:
    - newton_step: solve H * delta_x = -grad_f0, return delta_x
    - evaluate_objective: f0(x)
    - gradient: grad f0(x)
    - hessian_vector_product: H(x) @ y

    """

    def __init__(
        self,
        settings: OptimizationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize solver."""
        super().__init__(phase1_solver=None, settings=settings, **kwargs)

    @property
    def num_eq_constraints(self) -> int:
        """No equality constraints."""
        return 0

    @property
    def num_ineq_constraints(self) -> int:
        """No inequality constraints."""
        return 0

    def solve(
        self,
        x0: npt.NDArray[np.float64] | None = None,
        fully_optimize: bool = False,
        **kwargs: Any,
    ) -> NewtonResult:
        """Solve unconstrained problem using Newton's method.

        Skips the outer barrier loop entirely and runs Newton's method once.

        Parameters
        ----------
         x0 : vector
            Initial guess.
         fully_optimize : bool
            Unused; present for API compatibility with base class.

        Returns
        -------
         res : NewtonResult

        """
        if x0 is None:
            raise ValueError("Initial guess x0 is required.")
        return self.centering_step(x0, t=1.0, last_step=True, fully_optimize=True)

    def is_feasible(self, x: npt.NDArray[np.float64]) -> bool:
        """All points are feasible (no constraints)."""
        return True

    def constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """No inequality constraints."""
        return np.zeros(0)

    def grad_constraints(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """No inequality constraints."""
        return np.zeros((0, x.size))

    def btls_keep_feasible(
        self, x: npt.NDArray[np.float64], delta_x: npt.NDArray[np.float64]
    ) -> float:
        """No constraints to maintain feasibility for; any step size is valid."""
        return np.inf

    @abstractmethod
    def newton_step(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Solve H(x) * delta_x = -grad_f0(x) and return delta_x."""

    @abstractmethod
    def hessian_vector_product(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute H(x) @ y."""

    def calculate_newton_step(
        self,
        x: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Newton step; t is unused (always 1, no barrier)."""
        return self.newton_step(x), np.zeros(0)

    def hessian_multiply(
        self,
        x: npt.NDArray[np.float64],
        t: float,
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute H(x) @ y; t is unused (always 1, no barrier)."""
        return self.hessian_vector_product(x, y)

    def evaluate_dual(
        self,
        lmbda: npt.NDArray[np.float64],
        nu: npt.NDArray[np.float64],
        x_star: npt.NDArray[np.float64],
    ) -> float:
        """Dual equals primal for unconstrained problems."""
        return self.evaluate_objective(x_star)
