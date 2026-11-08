"""Newton's method for Cvxium problems with no inequality constraints.

Defines ``NewtonSolver``, the ``Solver`` for a ``Problem`` whose only constraints, if
any, are equalities A w = b. It runs Newton's method with backtracking line search,
taking each step from the objective's structured Hessian, or, with an equality, from the
bordered KKT system that the equality layer builds over that Hessian.

The iteration itself is not reimplemented here. The private ``_ObjectiveNewtonAdapter``
presents the ``Objective`` and the optional ``EqualityConstraintLayer`` through the
interface of ``EqualityConstrainedNewtonSolver`` in ``optimization``, which supplies the
centering loop, the line search, and the ``SolveResult``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._typing import none_throws
from ..solvers.optimization import (
    EqualityConstrainedNewtonSolver,
)
from ..solvers.solver import Solver

if TYPE_CHECKING:
    from ..problems.linear_constraints import (
        EqualityConstraintLayer,
    )
    from ..problems.objective import Objective
    from ..problems.problem import (
        OnInfeasible,
        Problem,
    )
    from ..problems.result import SolveResult
    from ..solvers.settings import (
        OptimizationSettings,
    )


class _ObjectiveNewtonAdapter(EqualityConstrainedNewtonSolver):
    """Adapter from an ``Objective`` and an optional equality layer to the Newton loop.

    Implements the abstract members of ``EqualityConstrainedNewtonSolver`` (``A``,
    ``b``, ``evaluate_objective``, ``gradient``, ``hessian_vector_product`` and
    ``newton_step``) by delegating to the wrapped ``Objective``. ``newton_step``
    branches on whether an ``EqualityConstraintLayer`` is present: without one it solves
    the Hessian system directly; with one it solves the bordered KKT system that the
    layer's ``saddle_system`` builds over the same Hessian. Reusing the base class's
    loop avoids reimplementing centering, backtracking line search and convergence
    bookkeeping.

    The base class reads the equality data only through the ``A`` and ``b`` properties.
    Without an equality they are empty (zero rows), so the base machinery (the equality
    count, the dual value, the Newton decrement) behaves exactly as for a pure
    unconstrained Newton solve, and the returned equality multiplier ``nu`` is empty.

    Private; users instantiate ``NewtonSolver`` instead.

    Parameters
    ----------
     objective : Objective
        The objective to minimize. Its Hessian must be invertible (see ``newton_step``).
     equality : EqualityConstraintLayer | None
        Equality constraints A w = b, or ``None`` for an unconstrained problem.
     settings : OptimizationSettings | None
        Algorithm settings passed to the base class; ``None`` uses the defaults.

    """

    def __init__(
        self,
        objective: Objective,
        equality: EqualityConstraintLayer | None = None,
        settings: OptimizationSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._objective = objective
        self._equality = equality
        # Cache the gradient computed at the start of each Newton iteration so that
        # the subsequent `backtracking_line_search` (which calls `self.gradient(x)`
        # via `gradient_barrier` at the same iterate) reuses the value instead of
        # recomputing. The cache is invalidated at the top of `newton_step` because
        # the centering loop mutates `x` in place between iterations
        # (`x += btls_s * delta_x` in `optimization.py`), so `id(x)` alone cannot
        # distinguish iterations.
        self._cached_gradient: npt.NDArray[np.float64] | None = None

    @property
    def A(self) -> npt.NDArray[np.float64]:
        """Return the equality constraint matrix A.

        Without an equality layer, returns an empty matrix of shape (0, M) rather than
        (0, 0), so that ``A.shape[1]`` and ``A @ x`` stay consistent with the
        decision-variable dimension M.

        Returns
        -------
         A : npt.NDArray[np.float64]
            The equality matrix, of shape (p, M), with p = 0 when there is no equality.

        """
        if self._equality is None:
            # Zero-row equality over the n-dim iterate: shape (0, n), not (0, 0),
            # so ``A.shape[1]`` and ``A @ x`` stay dimensionally consistent with
            # the objective's decision-variable dimension.
            return np.zeros((0, self._objective.dimension))
        return self._equality.A

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Return the equality right-hand side b.

        Returns
        -------
         b : npt.NDArray[np.float64]
            The right-hand side, of length p; empty when there is no equality.

        """
        if self._equality is None:
            return np.zeros(0)
        return self._equality.b

    def evaluate_objective(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective f0 at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which to evaluate, of length M.

        Returns
        -------
         f0 : float
            The objective value.

        """
        return self._objective.evaluate(x)

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the objective gradient ∇f0 at the current iterate, cached.

        The centering loop updates x in place between iterations, so the cache cannot be
        keyed on x. Instead ``newton_step`` clears it at the top of each iteration and
        then fills it; the line search and the Newton decrement, which ask for the
        gradient at the same iterate, reuse the value. A call at any other point between
        two Newton steps would return the stale cached value; within the base class's
        loop, none occurs.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which to evaluate, of length M; read only when the cache is empty.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The objective gradient, of length M.

        """
        if self._cached_gradient is not None:
            return self._cached_gradient
        g = self._objective.gradient(x)
        self._cached_gradient = g
        return g

    def hessian_vector_product(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the objective Hessian ∇^2 f0 at x.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Point at which the Hessian is evaluated, of length M.
         y : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Hy : npt.NDArray[np.float64]
            The product ∇^2 f0(x) y, of length M.

        """
        return self._objective.hessian_contribution(x).multiply(y)

    def newton_step(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve for the Newton step using the objective's structured Hessian.

        With no equality, solves
           H Δx = -∇f0
        through ``hessian_contribution(x).solve``. With an equality, solves the bordered
        KKT system
           | H   A^T | | Δx |   | -∇f0 |
           | A    0  | | ν  | = |   0  |
        through the layer's ``saddle_system``. In both cases H is the objective's
        ``hessian_contribution(x)``. Clears the gradient cache first, since x may have
        changed in place since the previous iteration.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            Current iterate, of length M.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The Newton step, of length M.
         nu : npt.NDArray[np.float64]
            The equality multipliers, of length p; empty when there is no equality.

        Raises
        ------
         NewtonStepError
            Propagated from the Hessian or saddle solve when the system is singular, for
            example a ``LinearObjective`` (whose Hessian is a ``ZeroSystem``) or a dense
            Hessian that is not positive definite.

        """
        self._cached_gradient = None  # x may have mutated since last iteration
        hessian = self._objective.hessian_contribution(x)
        grad = self.gradient(x)
        if self._equality is None:
            return hessian.solve(-grad), np.zeros(0)
        return self._equality.saddle_system(hessian).solve(-grad)


class NewtonSolver(Solver):
    """Newton's method for a ``Problem`` with no inequality constraints.

    Solves either the unconstrained problem
       minimize    f0(w)
    or, when ``problem.eq`` is present, the equality-constrained problem
       minimize    f0(w)
       subject to  A w = b,
    where f0 is ``problem.obj``, using Newton's method with backtracking line search.
    The solver dispatches on ``problem.eq``. With no equality, each step comes from the
    objective's structured Hessian (``hessian_contribution(x).solve``). With an
    equality, it solves the bordered KKT system
       | H   A^T | | Δw |   | -∇f0 |
       | A    0  | | ν  | = |   0  |
    through the ``SaddleSystem`` that ``problem.eq.saddle_system`` builds over the same
    Hessian H.

    The objective needs an invertible Hessian (``QuadraticObjective`` with positive
    definite Q is the canonical example); with a singular one, such as a
    ``LinearObjective`` or a rank-deficient ``QuadraticObjective``, the first Newton
    step fails. With an equality, the initial guess x0 should satisfy A x0 = b: each
    step keeps A w fixed, so the equality residual at x0 is never corrected, and it is
    not checked.

    Internally delegates to ``EqualityConstrainedNewtonSolver`` through a private
    adapter, which provides the centering loop, the line search, and the
    ``SolveResult``.

    Parameters
    ----------
     settings : OptimizationSettings | None
        Algorithm settings (tolerances, iteration limits, line-search parameters);
        ``None`` uses the defaults.

    """

    def solve(
        self,
        problem: Problem,
        x0: npt.NDArray[np.float64] | None = None,
        on_infeasible: OnInfeasible = "raise",
    ) -> SolveResult:
        """Solve a problem from an initial guess with Newton's method.

        ``on_infeasible`` is accepted for uniformity with the ``Solver`` interface. A
        Newton solve has no feasibility stage, so it cannot honor the ``"continue"``
        policy.

        Parameters
        ----------
         problem : Problem
            The problem to solve. It must have an objective (``none_throws`` fails on a
            feasibility-only problem) and no inequality constraint layers.
         x0 : npt.NDArray[np.float64] | None
            Initial guess, of length M. Required despite the ``None`` default, which
            exists only to match the ``Solver`` signature. With an equality it should
            satisfy A x0 = b (not checked).
         on_infeasible : OnInfeasible
            Infeasibility policy. Only ``"raise"`` is supported.

        Returns
        -------
         result : SolveResult
            The solution and the solve's diagnostics.

        Raises
        ------
         NotImplementedError
            When ``on_infeasible`` is ``"continue"``.
         ValueError
            When ``x0`` is ``None``, or when ``problem`` has any inequality constraint
            layers (use ``BarrierSolver`` or ``PrimalDualSolver`` for those).
         CenteringStepError
            Propagated from ``centering_step`` when a Newton step cannot be computed
            (for example, a singular Hessian; the ``NewtonStepError`` is chained as the
            cause), or when the line search fails or the iteration limit is reached
            before the iterate meets the acceptable tolerance.

        """
        if on_infeasible == "continue":
            raise NotImplementedError(
                "NewtonSolver does not support on_infeasible='continue'; use "
                "BarrierSolver (the IPM engine) for the continue policy."
            )
        if x0 is None:
            raise ValueError("NewtonSolver requires an explicit starting point x0.")
        if problem.ineq:
            raise ValueError(
                "NewtonSolver requires a Problem with no inequality constraints; "
                f"got {len(problem.ineq)} constraint layer(s). Use "
                "BarrierSolver or PrimalDualSolver for inequality-constrained "
                "problems."
            )
        adapter = _ObjectiveNewtonAdapter(
            objective=none_throws(problem.obj),
            equality=problem.eq,
            settings=self.settings,
        )
        return adapter.solve(x0=x0)
