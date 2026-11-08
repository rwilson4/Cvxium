"""Composable inequality-constraint layers for Cvxium's interior-point solvers.

Defines ``ConstraintLayer``, the abstract base for a block of inequality constraints
   f_i(w) ≤ 0,   i = 1, ..., m,
that a ``Problem`` stacks to form its inequality constraints. A layer supplies the
constraint values, the Jacobian and its products, the log barrier and its gradient, its
weighted Hessian contributions to barrier and primal-dual Newton steps, and a
step-length bound for the line search.

Every concrete layer subclasses one of two bases, which select how
``Problem.make_phase1`` builds a Phase I problem when that layer is the one to make
feasible: ``SlackConstraintLayer`` (slack form) and ``DirectConstraintLayer`` (direct
form, for a single constraint that can serve as its own Phase I objective). The concrete
layers live in ``linear_constraints``, ``quadratic_constraints`` and ``phase1``.

"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..linalg.systems import (
    MatrixSystem,
    ZeroSystem,
)
from ..problems.objective import (
    Objective,
    ObjectivePoint,
    _DefaultObjectivePoint,
)
from ..solvers.settings import (
    LineSearchSettings,
)

if TYPE_CHECKING:
    # Imported for typing only: ``linear_constraints`` imports this module, so a
    # runtime import would be circular. ``interior_point`` takes the equality
    # manifold as an optional argument, annotated via this guarded name.
    from ..problems.linear_constraints import (
        EqualityConstraintLayer,
    )

logger: logging.Logger = logging.getLogger("cvxium.constraint_layers")


class ConstraintLayer(ABC):
    """Abstract block of inequality constraints for an interior-point solver.

    Encodes m constraints f_i(w) ≤ 0 on a decision variable w of dimension M, and
    supplies what a solver needs to assemble a barrier or primal-dual Newton step:
    constraint values, Jacobian products, Hessian contributions, and a step-length
    bound.

    Most methods accept a ``shift`` that relaxes every constraint to
       f_i(w) - shift ≤ 0.
    The default shift = 0 gives the layer's own constraints. Slack-form Phase I passes
    its slack variable as the shift and minimizes it, driving the constraints back into
    strict feasibility.

    Concrete layers do not subclass ``ConstraintLayer`` directly. They subclass
    ``SlackConstraintLayer`` or ``DirectConstraintLayer``, which select the Phase I
    strategy ``Problem.make_phase1`` uses when the layer is the active one (the layer
    whose constraints Phase I must satisfy):
    - ``SlackConstraintLayer``: slack form. Phase I adds a slack variable s, relaxes the
      layer's constraints to f_i(w) ≤ s, and minimizes s. Works for any layer.
    - ``DirectConstraintLayer``: direct form, for a single constraint. Phase I
      minimizes the constraint value itself, with no slack variable.

    Equality constraints A w = b are not constraint layers; a ``Problem`` carries them
    separately, as an ``EqualityConstraintLayer``.

    """

    is_linear: bool = False
    """Whether every constraint in this layer is affine in w, so that ∇^2 f_i = 0.

    Solvers use this, together with ``Objective.is_linear``, to evaluate the dual in
    closed form rather than by a Newton minimization. The default False is safe: a new
    affine layer that forgets to override it only takes the slower path, never a wrong
    one. ``SlackConstraintLayer`` overrides it to True.

    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension M of the decision variable this layer constrains.

        Every concrete layer fixes its dimension at construction (for example, the
        length of a bounds vector, or the column count of a constraint matrix).
        ``Problem`` checks its declared dimension against each component's.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        ...

    @abstractmethod
    def num_constraints(self, M: int) -> int:
        """Count the inequality constraints in this layer.

        Parameters
        ----------
         M : int
            Dimension of the decision variable.

        Returns
        -------
         m : int
            Number of constraints the layer contributes.

        """
        ...

    @abstractmethod
    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the shifted constraint values f_i(w) - shift.

        With shift = 0 this returns the layer's own constraint values; with shift > 0
        every constraint is loosened by the shift.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraints are evaluated, of length M.
         shift : float
            Relaxation subtracted from every constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The shifted constraint values, of length m; all negative when w strictly
            satisfies the shifted constraints.

        """
        ...

    @abstractmethod
    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint Jacobian G at w.

        Row i of G is ∇f_i(w). Subclasses must implement this. The default
        ``grad_constraints_multiply`` and ``grad_constraints_transpose_multiply`` form G
        and multiply, which gets a new layer running quickly; for speed, override them
        with structure-exploiting versions that never form G.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (m, M).

        """
        ...

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian.

        The default forms G with ``grad_constraints``; override it with a structured
        fast path.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length m.

        """
        return np.asarray(self.grad_constraints(w) @ v)

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian.

        The default forms G with ``grad_constraints``; override it with a structured
        fast path.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M.
         u : npt.NDArray[np.float64]
            Vector of length m to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        return np.asarray(self.grad_constraints(w).T @ u)

    @abstractmethod
    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return this layer's weighted Hessian contribution as a ``MatrixSystem``.

        Each constraint i contributes the curvature atom
           C_i = ∇^2 f_i - (1 / f_i) ∇f_i ∇f_i^T,
        weighted by λ_i, where f_i denotes the shifted constraint value. This single
        contract serves both interior-point methods. The barrier method passes λ_i = -1
        / f_i, turning the atom into the log-barrier curvature
           (1 / f_i^2) ∇f_i ∇f_i^T - (1 / f_i) ∇^2 f_i,
        while the primal-dual method passes its dual variables. The two weightings
        coincide on the central path.

        This is abstract because the right ``MatrixSystem`` structure depends on the
        layer, and there is no safe default. A layer whose Jacobian rows are basis
        vectors (such as ``BoundsConstraintLayer``) is diagonal and returns a
        ``DiagonalSystem``. A layer with few constraints relative to M is genuinely low
        rank and can use ``_low_rank_barrier_contribution``. A layer must not assume G^T
        G is low rank in general, since for a layer with many constraints it is not.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; the atoms depend on f_i(w), ∇f_i(w) and ∇^2
            f_i(w).
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m, in the layer's constraint order.
            Required; the ``None`` default exists only so the signature matches
            ``Objective.hessian_contribution``, which ``DirectConstraintLayer`` also
            inherits.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature ∑ λ_i C_i, which ``Problem.build_hessian_system``
            sums into the Newton-system Hessian.

        Raises
        ------
         ValueError
            Concrete layers raise when ``lmbda`` is ``None``.

        """
        ...

    def _low_rank_barrier_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the weighted Jacobian outer-product term as a low-rank update on zero.

        Returns a ``ZeroSystem`` carrying one low-rank update with factor G^T and
        weights d_i = -λ_i / f_i, that is
           ∑ (-λ_i / f_i) ∇f_i ∇f_i^T.
        This omits the curvature term ∑ λ_i ∇^2 f_i, so it is the whole contribution
        only for an affine layer, where ∇^2 f_i = 0.

        The coalesced low-rank form is correct only when the layer's constraint count is
        small relative to M, so that G^T G really is low rank. A layer whose Jacobian
        rows are basis vectors should return a ``DiagonalSystem`` instead, and a layer
        whose curvature has mixed inertia (a downdate, as in a second-order cone
        barrier) must not use this form.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M, at which f_i and G are evaluated.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m. Required; the ``None`` default
            mirrors ``hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` carrying the single low-rank update.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``.

        """
        if lmbda is None:
            raise ValueError(
                "hessian_contribution requires per-constraint weights `lmbda` "
                "when a layer contributes as a background constraint."
            )
        f = self.constraints(w, shift)
        kappa = self.grad_constraints(w).T
        return ZeroSystem(len(w)).low_rank_update(kappa, -lmbda / f)

    @abstractmethod
    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the weighted constraint curvature ∑ λ_i ∇^2 f_i as a ``MatrixSystem``.

        This is the second-derivative term alone, without the Jacobian outer-product
        term that ``hessian_contribution`` adds. It is the constraint side of the
        Lagrangian Hessian
           ∇^2 L = ∇^2 f_0 + ∑ λ_i ∇^2 f_i,
        which the barrier method minimizes over w when it evaluates the dual.

        Abstract with no default, so every layer states its own curvature: affine layers
        (such as ``BoundsConstraintLayer``, ``ImbalanceConstraintLayer`` and
        ``LinearInequalityConstraintLayer``) return a ``ZeroSystem``, and curved layers
        (such as ``NormConstraintLayer`` and ``SquaredNormConstraintLayer``) return
        their real curvature. Requiring the declaration keeps a curved layer from
        silently inheriting zero curvature and mis-assembling the dual Hessian.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M. The curvature of a quadratic layer does not
            depend on w, but the signature allows curvature that does.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length m.

        Returns
        -------
         system : MatrixSystem
            The weighted curvature ∑ λ_i ∇^2 f_i.

        """
        ...

    def barrier_objective(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> float:
        """Evaluate the log barrier of the shifted constraints.

        Computes
           φ(w) = -∑ log(shift - f_i(w)).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the barrier is evaluated, of length M.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         phi : float
            The barrier value; inf when any shifted constraint is not strictly
            satisfied.

        """
        c = self.constraints(w, shift=shift)
        if np.any(c >= 0):
            return np.inf
        return float(-np.sum(np.log(-c)))

    def barrier_gradient(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Compute the gradient of the log barrier with respect to w.

        With c_i = f_i(w) - shift, negative for strictly feasible w,
           ∇φ(w) = -G^T (1 / c),
        where 1 / c is taken componentwise.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Strictly feasible point at which the gradient is evaluated, of length M.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         grad_phi : npt.NDArray[np.float64]
            The barrier gradient, of length M.

        """
        c = self.constraints(w, shift=shift)
        return -self.grad_constraints_transpose_multiply(w, 1.0 / c)

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps the layer strictly feasible.

        Finds the largest α such that every constraint satisfies
           f_i(w + α dw) - (shift + α ds) < 0.
        The pair (shift, ds) lets the relaxation move with the step: slack-form Phase I,
        which moves its slack s jointly with w, passes shift = s and ds equal to the
        slack's search direction. The defaults shift = 0 and ds = 0 check the layer's
        own constraints.

        The default implementation backtracks from 1 / β by factors of β, evaluating
        ``constraints`` at each trial point. It handles any nonlinearity but is slow,
        and the largest value it can report is 1 / β: enough for the caller's β α to
        reach the full unit step, but not the true maximum. Subclasses with closed-form
        bounds should override it, report the true maximum (inf when no constraint
        binds), and may ignore ``line_search``.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Settings supplying β and the minimum step; ``None`` uses the defaults.

        Returns
        -------
         alpha : float
            The largest feasible trial step, at most 1 / β; a value no larger than the
            minimum step when no trial step is feasible.

        """
        if line_search is None:
            line_search = LineSearchSettings()
        beta = line_search.beta
        min_step = line_search.min_step
        # Start at ``1 / beta``, not at 1. The caller multiplies this by
        # ``beta`` and caps at the unit step, so an initial 1.0 can only ever
        # yield ``beta`` (~0.5) -- a permanent half-step, including in the
        # quadratic phase where Boyd and Vandenberghe 9.5.3 guarantees the
        # unit step is accepted. ``Optimizer.btls_keep_feasible`` already
        # starts at ``1 / beta`` for exactly this reason.
        alpha = 1.0 / beta
        while alpha > min_step:
            shift_alpha = shift + alpha * ds
            if np.all(self.constraints(w + alpha * dw, shift=shift_alpha) < 0):
                return alpha
            alpha *= beta
        return alpha

    def is_satisfied(
        self,
        w: npt.NDArray[np.float64],
    ) -> bool:
        """Test whether every constraint in this layer holds strictly at w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         satisfied : bool
            True when f_i(w) < 0 for every constraint.

        """
        return bool(np.all(self.constraints(w) < 0))


class SlackConstraintLayer(ConstraintLayer):
    """Constraint layer whose Phase I relaxes the constraints with a slack variable.

    When a slack layer is the active layer, ``Problem.make_phase1`` builds a slack-form
    Phase I problem: it augments the decision variable with a slack s, relaxes the
    layer's constraints to
       f_i(w) ≤ s,
    and minimizes s. The layer remains a constraint in that problem, never the
    objective, so it needs no objective face, and slack form works for any layer.

    Every slack layer defined in Cvxium is affine, so ``is_linear`` is True here; a
    lifted layer (``LiftedConstraintLayer``) instead forwards the value of the layer it
    wraps.

    """

    is_linear: bool = True


class DirectConstraintLayer(ConstraintLayer, Objective):
    """Single-constraint layer whose Phase I minimizes the constraint value directly.

    When a direct layer is the active layer, ``Problem.make_phase1`` builds a
    direct-form Phase I problem: the layer becomes the objective, and minimizing its
    value f(w), subject to the remaining (background) layers, drives it below zero
    without a slack variable.

    A direct layer is therefore also an ``Objective``, and this class supplies that
    face: ``num_constraints`` is fixed at 1, so the single-constraint requirement is
    structural; ``evaluate`` and ``gradient`` return the constraint value and gradient
    row; and ``at`` supplies the constraint's curvature as the objective Hessian.
    Subclasses provide the constraint face (``dimension``, ``constraints``,
    ``grad_constraints``, ``hessian_contribution`` and ``hessian_curvature``) and may
    offer ``barrier_parameter_hint`` and ``interior_point``.

    """

    def num_constraints(self, M: int) -> int:
        """Return one, the number of constraints in a direct layer.

        Parameters
        ----------
         M : int
            Dimension of the decision variable; the count does not depend on it.

        Returns
        -------
         m : int
            Always 1.

        """
        return 1

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective face, the layer's constraint value f(w).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraint is evaluated, of length M.

        Returns
        -------
         f : float
            The unshifted constraint value.

        """
        return float(self.constraints(w)[0])

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the objective-face gradient ∇f(w), the constraint's gradient row.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the gradient is evaluated, of length M.

        Returns
        -------
         grad_f : npt.NDArray[np.float64]
            The constraint gradient, of length M.

        """
        return np.asarray(self.grad_constraints(w)[0])

    def at(self, w: npt.NDArray[np.float64]) -> ObjectivePoint:
        """Return the per-point context for using the constraint value as an objective.

        When a direct layer is the objective of a direct-form Phase I problem,
        ``build_hessian_system`` needs the objective's structured Hessian. That is the
        constraint's own curvature ∇^2 f(w), with no Jacobian outer-product term (the
        layer acts as an objective, not as a barrier-wrapped constraint), which is
        ``hessian_curvature`` at unit weight. The inherited ``Objective.at`` would route
        the point's Hessian to the constraint-face ``hessian_contribution``, which
        raises without ``lmbda``; this override supplies the curvature instead.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the objective is evaluated, of length M.

        Returns
        -------
         point : ObjectivePoint
            The objective's value, gradient and structured Hessian at w.

        """
        return _DirectConstraintObjectivePoint(self, w)

    def barrier_parameter_hint(
        self,
        x0: npt.NDArray[np.float64],
        num_ineq: int,
        outer_tolerance: float,
    ) -> float | None:
        """Return a lower bound on the initial barrier parameter in direct-form Phase I.

        When this layer is the objective of a direct-form Phase I problem,
        ``BarrierSolver`` raises its initial barrier parameter t0 to max(t0, hint) under
        every adaptive initialization strategy. Without such a floor, t0 can be too
        small relative to the objective's magnitude: the centering then under-progresses
        on the objective, and the solver may spuriously certify infeasibility.

        A subclass with a closed-form version of the heuristic
           t0 ≈ m / (f(x0) - p_star),
        where p_star is a lower bound on the optimal value (Boyd and Vandenberghe, 2004,
        section 11.3.1), can override this hook to return a problem-aware bound.

        Parameters
        ----------
         x0 : npt.NDArray[np.float64]
            Strictly feasible starting point of the Phase I problem.
         num_ineq : int
            Number of inequality constraints in the Phase I problem (the background
            layers' constraints), the m in the heuristic.
         outer_tolerance : float
            Outer duality-gap tolerance, a floor for denominators that may be tiny when
            x0 is nearly optimal.

        Returns
        -------
         hint : float | None
            Lower bound on t0, or ``None`` (the default) when the layer has none to
            offer.

        References
        ----------
        Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge
        University Press, 2004.

        """
        return None

    def interior_point(
        self, equality: "EqualityConstraintLayer | None" = None
    ) -> npt.NDArray[np.float64] | None:
        """Return a strictly interior point of this layer, or None if none is known.

        A direct layer that knows a point w with f(w) < 0 in closed form can override
        this hook to supply it. When the layer is the only inequality of a feasibility
        problem, ``Solver._solve_feasibility`` takes that point as the answer,
        certifying feasibility with no Phase I solve. This also sidesteps the
        direct-form Phase I in that case, whose objective-only shape (the sole layer
        hoisted to the objective, leaving no inequality layer) the barrier solver cannot
        handle.

        When the feasibility problem also carries equality constraints A w = b, they are
        passed as ``equality``, so a layer that knows the equality-constrained solution
        in closed form (for example, the minimum-residual point of an ellipsoid
        restricted to A w = b) can return a point satisfying both. The caller rechecks
        the strict inequality and the equality residual, and falls back to the iterative
        Phase I if either fails. A layer that cannot satisfy the equality in closed form
        should return ``None`` when ``equality`` is given.

        Parameters
        ----------
         equality : EqualityConstraintLayer | None
            The problem's equality constraints, or ``None`` for an inequality-only
            problem.

        Returns
        -------
         w : npt.NDArray[np.float64] | None
            A point with f(w) < 0 that also satisfies ``equality`` when one is given;
            ``None`` by default, since most layers know no such point.

        """
        return None


class _DirectConstraintObjectivePoint(_DefaultObjectivePoint):
    """Objective point for a direct constraint layer used as a ``Problem`` objective.

    Delegates the value and gradient to the layer's objective face (through
    ``_DefaultObjectivePoint``), and returns the constraint curvature ∇^2 f(w), which is
    ``hessian_curvature`` at unit weight, as the structured Hessian. A direct-form Phase
    I problem therefore assembles through ``build_hessian_system``.

    Parameters
    ----------
     layer : DirectConstraintLayer
        The layer acting as the objective.
     w : npt.NDArray[np.float64]
        Point at which the objective is evaluated, of length M.

    """

    def __init__(
        self,
        layer: "DirectConstraintLayer",
        w: npt.NDArray[np.float64],
    ) -> None:
        super().__init__(layer, w)
        self._layer = layer

    def hessian_contribution(self) -> MatrixSystem:
        """Return the structured objective Hessian, the constraint curvature ∇^2 f(w).

        Returns
        -------
         system : MatrixSystem
            The layer's ``hessian_curvature`` at w with unit weight.

        """
        return self._layer.hessian_curvature(self._w, np.ones(1))
