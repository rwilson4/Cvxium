"""Composable objective functions for Cvxium's convex optimization solvers.

Defines ``Objective``, the abstract base for the scalar function f0(w) that a
``Problem`` minimizes, and ``ObjectivePoint``, the per-point context through which a
solver evaluates the objective's value, gradient, and structured Hessian at one iterate.
An objective reports its Hessian as a ``MatrixSystem``, which
``Problem.build_hessian_system`` sums with the constraint layers' contributions, so
structure in ∇^2 f0 (diagonal, low rank, zero) carries through to the Newton step.

The concrete objectives are ``LinearObjective`` and ``QuadraticObjective``. Two wrappers
build new objectives from existing ones: ``NegatedObjective`` (the negation -g, used by
``Problem.maximize``) and ``LiftedObjective`` (an objective that ignores trailing
auxiliary coordinates, used for epigraph and Phase I formulations).

"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt

from ..linalg.systems import (
    BlockDiagonalSystem,
    DiagonalSystem,
    MatrixSystem,
    ZeroSystem,
)


class ObjectivePoint(ABC):
    """Per-point evaluation context returned by ``Objective.at``.

    A solver acquires one ``ObjectivePoint`` per iterate w (``BarrierSolver`` caches it
    until the iterate changes) and routes the value, gradient, and Hessian calls through
    it. A subclass can therefore compute w-dependent work (forward passes, structured
    Hessian data) once per iterate and share it across those calls, rather than
    recomputing it inside each method.

    The default point returned by ``Objective.at`` is an uncached pass-through that
    delegates back to the ``Objective`` methods. An objective with expensive w-dependent
    work should override ``Objective.at`` to return a cached point.

    """

    @abstractmethod
    def evaluate(self) -> float:
        """Return the objective value f0(w) at the bound point.

        Returns
        -------
         f0 : float
            The objective value.

        """
        ...

    @abstractmethod
    def gradient(self) -> npt.NDArray[np.float64]:
        """Return the gradient ∇f0(w) at the bound point.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The gradient, of length M.

        """
        ...

    @abstractmethod
    def hessian_contribution(self) -> MatrixSystem:
        """Return the structured Hessian ∇^2 f0(w) at the bound point.

        The per-point counterpart of ``Objective.hessian_contribution``, with w already
        bound.

        Returns
        -------
         system : MatrixSystem
            The objective's curvature ∇^2 f0(w).

        """
        ...


class _DefaultObjectivePoint(ObjectivePoint):
    """Uncached pass-through objective point.

    Returned by ``Objective.at`` for objectives that do not override it. Each method
    forwards to the corresponding ``Objective`` method at the bound point, which is
    correct for any objective but forgoes per-point caching.

    Parameters
    ----------
     objective : Objective
        The objective to delegate to.
     w : npt.NDArray[np.float64]
        The bound point, of length M.

    """

    def __init__(self, objective: Objective, w: npt.NDArray[np.float64]) -> None:
        self._objective = objective
        self._w = w

    def evaluate(self) -> float:
        """Return the objective value f0(w) from ``Objective.evaluate``.

        Returns
        -------
         f0 : float
            The objective value at the bound point.

        """
        return self._objective.evaluate(self._w)

    def gradient(self) -> npt.NDArray[np.float64]:
        """Return the gradient ∇f0(w) from ``Objective.gradient``.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The gradient at the bound point, of length M.

        """
        return self._objective.gradient(self._w)

    def hessian_contribution(self) -> MatrixSystem:
        """Return the structured Hessian from ``Objective.hessian_contribution``.

        Returns
        -------
         system : MatrixSystem
            The objective's curvature ∇^2 f0(w) at the bound point.

        """
        return self._objective.hessian_contribution(self._w)


class Objective(ABC):
    """Abstract objective of a convex optimization problem.

    An ``Objective`` supplies the scalar function f0(w) and its derivatives in the form
    the solvers consume, mirroring the ``ConstraintLayer`` interface:
    - ``evaluate`` returns f0(w);
    - ``gradient`` returns ∇f0(w);
    - ``hessian_contribution`` returns ∇^2 f0(w) as a ``MatrixSystem``, which
      ``Problem.build_hessian_system`` sums with the constraint layers' contributions;
    - ``at`` returns an ``ObjectivePoint`` bundling the three at one iterate;
    - ``btls_keep_in_domain`` bounds a line-search step by the domain of f0.

    Because the Newton system is assembled from structured contributions, an objective
    with structured curvature (for example, a ``QuadraticObjective`` with diagonal Q, or
    with low rank Q = κ κ^T) keeps that structure through the whole Newton step.

    """

    is_linear: bool = False
    """Whether f0 is affine in w, so that ∇^2 f0 ≡ 0.

    ``BarrierSolver`` checks it, together with each inequality layer's ``is_linear``, to
    evaluate the dual analytically (the LP dual, decided by the gradient residual)
    instead of through the mini-Newton solve. The default False is safe: an affine
    objective that forgets to override it only takes the slower path, never a wrong one.
    It is a plain class attribute rather than a ``ClassVar`` so that wrappers whose
    linearity depends on what they wrap, such as ``NegatedObjective``, can forward the
    wrapped value per instance.

    """

    feasibility_threshold: float | None = None
    """Stop threshold for a feasibility solve, or ``None`` for ordinary minimization.

    When set, a solver minimizing this objective is answering a feasibility question: it
    stops as soon as the value falls below ``feasibility_threshold`` (the feasibility
    certificate), and runs to the optimum only to prove the problem infeasible. ``None``
    disables the early stop. Attach a threshold to a copy with
    ``with_feasibility_threshold``; the objective is otherwise unchanged, so its
    ``is_linear`` and fast dual path are preserved.

    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension M of the decision variable this objective acts on.

        Every concrete objective fixes its dimension at construction (for example, the
        length of a coefficient vector c, or the order of a Hessian system). ``Problem``
        checks its declared dimension against each component's.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        ...

    def with_feasibility_threshold(self, threshold: float) -> Self:
        """Return a copy of this objective that stops a solve at a threshold.

        The copy carries ``feasibility_threshold`` set to ``threshold``, so a solver
        treats the minimization as a feasibility question and stops once the value falls
        below it. ``Solver._solve_feasibility`` attaches a threshold of 0 to the Phase I
        objective.

        The copy is shallow: only ``feasibility_threshold`` is independent, while the
        underlying data (such as a ``LinearObjective``'s ``c``) is shared with the
        original. That keeps the copy cheap and preserves its type, its ``is_linear``,
        and the fast dual path. Attaching the threshold never mutates the shared data,
        so the original is unchanged; callers should treat that data as read-only, since
        in-place changes would show through both.

        Parameters
        ----------
         threshold : float
            Objective value below which a solve stops.

        Returns
        -------
         obj : Self
            A shallow copy of this objective with ``feasibility_threshold = threshold``.

        """
        obj = copy.copy(self)
        obj.feasibility_threshold = threshold
        return obj

    @abstractmethod
    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective value f0(w).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the objective is evaluated, of length M.

        Returns
        -------
         f0 : float
            The objective value.

        """
        ...

    @abstractmethod
    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient ∇f0(w).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the gradient is evaluated, of length M.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The gradient, of length M.

        """
        ...

    def at(self, w: npt.NDArray[np.float64]) -> ObjectivePoint:
        """Return a per-point evaluation context bound to w.

        The default returns an uncached pass-through that forwards each call to the
        corresponding ``Objective`` method. An objective with expensive w-dependent work
        (forward passes, assembly of Hessian data) should override this to return a
        cached point, so a solver pays that cost once per iterate rather than once per
        call to ``evaluate``, ``gradient``, and ``hessian_contribution``.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to bind, of length M.

        Returns
        -------
         point : ObjectivePoint
            The evaluation context at w.

        """
        return _DefaultObjectivePoint(self, w)

    def btls_keep_in_domain(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
    ) -> float:
        """Find the largest step along a direction that stays in the domain of f0.

        Returns the largest α such that w + α dw lies in dom f0. This is the domain
        analogue of ``ConstraintLayer.btls_max_step``: it bounds a line search by the
        geometry of dom f0 rather than of the constraint set. The default domain is all
        of R^M, so the bound is inf, and the line search caps the step at
           min(1, β · inf) = 1,
        the full Newton step.

        An objective with a restricted domain (for example, a logarithm whose argument
        must stay positive) overrides this with the closed-form distance to the domain
        boundary, as a ``ConstraintLayer`` does for its feasible set. ``BarrierSolver``
        uses it in the mini-Newton solve that evaluates the dual: that solve minimizes
        the Lagrangian over unconstrained w, so dom f0 is the only geometry bounding its
        steps.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current point, in dom f0, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.

        Returns
        -------
         alpha : float
            The largest step keeping w + α dw in dom f0; inf by default.

        """
        del w, dw
        return float("inf")

    @abstractmethod
    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the objective's structured Hessian as a ``MatrixSystem``.

        The owning ``Problem`` sums the returned curvature ∇^2 f0(w) with the constraint
        layers' contributions in ``build_hessian_system``.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M, at which the curvature is evaluated.

        Returns
        -------
         system : MatrixSystem
            The objective's curvature ∇^2 f0(w).

        """
        ...


class LinearObjective(Objective):
    """Linear objective f0(w) = c^T w.

    The Hessian is identically zero, so the structured contribution is a ``ZeroSystem``,
    and ``is_linear`` is True.

    Parameters
    ----------
     c : npt.NDArray[np.float64]
        Coefficient vector of length M, converted to float64.

    """

    is_linear: bool = True

    def __init__(self, c: npt.NDArray[np.float64]) -> None:
        self.c: npt.NDArray[np.float64] = np.asarray(c, dtype=np.float64)

    @property
    def dimension(self) -> int:
        """Return the dimension M, the length of ``c``.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self.c.shape[0]

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective value c^T w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the objective is evaluated, of length M.

        Returns
        -------
         f0 : float
            The value c^T w.

        """
        return float(np.dot(self.c, w))

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the zero Hessian of a linear objective.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; ignored, since the Hessian is zero everywhere.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` of order M.

        """
        return ZeroSystem(self.c.shape[0])

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the gradient c, which does not depend on w.

        Returns a fresh copy, so callers may mutate the result without corrupting the
        stored ``c``.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the gradient is evaluated, of length M; ignored.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            A copy of ``c``, of length M.

        """
        return self.c.copy()


class QuadraticObjective(Objective):
    """Quadratic objective f0(w) = (1/2) w^T Q w + c^T w, with Q symmetric PSD.

    The gradient is Q w + c and the Hessian is the constant Q. The caller supplies Q as
    a ``MatrixSystem`` describing its structure: ``DiagonalSystem`` for a diagonal Q,
    ``ZeroSystem(M).low_rank_update(kappa)`` for a low-rank Q = κ κ^T, ``DenseSystem``
    otherwise. That system is the single source of truth: the value and gradient use its
    ``multiply``, and ``hessian_contribution`` returns it unchanged, so Q is never
    materialized or factored here and its structure is exploited end to end.

    Parameters
    ----------
     c : npt.NDArray[np.float64]
        Linear coefficient vector of length M, converted to float64.
     hessian_contribution : MatrixSystem
        The constant Hessian Q. Its ``dimension`` defines M.

    """

    def __init__(
        self,
        c: npt.NDArray[np.float64],
        hessian_contribution: MatrixSystem,
    ) -> None:
        self.c: npt.NDArray[np.float64] = np.asarray(c, dtype=np.float64)
        self._hessian_contribution: MatrixSystem = hessian_contribution
        self._n: int = hessian_contribution.dimension

    @property
    def dimension(self) -> int:
        """Return the dimension M, the order of the Hessian system.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self._n

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the caller-supplied ``MatrixSystem`` for the constant Hessian Q.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; ignored, since Q is constant.

        Returns
        -------
         system : MatrixSystem
            The system given at construction (not a copy).

        """
        return self._hessian_contribution

    def _q_multiply(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by Q through the structured Hessian system.

        Parameters
        ----------
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Qv : npt.NDArray[np.float64]
            The product Q v, of length M.

        """
        return np.asarray(self._hessian_contribution.multiply(v))

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective value (1/2) w^T Q w + c^T w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the objective is evaluated, of length M.

        Returns
        -------
         f0 : float
            The objective value.

        """
        return float(0.5 * w @ self._q_multiply(w) + self.c @ w)

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient Q w + c.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the gradient is evaluated, of length M.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The gradient, of length M.

        """
        return np.asarray(self._q_multiply(w) + self.c)


class _NegatedObjectivePoint(ObjectivePoint):
    """Objective point for a ``NegatedObjective``, negating the wrapped point.

    Wraps the ``ObjectivePoint`` of the objective g bound at w and reports the value,
    gradient, and Hessian of -g, so any caching that ``g.at(w)`` performs carries over
    to the negated calls.

    Parameters
    ----------
     inner : ObjectivePoint
        The wrapped objective's point at w.

    """

    def __init__(self, inner: ObjectivePoint) -> None:
        self._inner = inner

    def evaluate(self) -> float:
        """Return the value -g(w) at the bound point.

        Returns
        -------
         f0 : float
            The negated wrapped value.

        """
        return -self._inner.evaluate()

    def gradient(self) -> npt.NDArray[np.float64]:
        """Return the gradient -∇g(w) at the bound point.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The negated wrapped gradient, of length M.

        """
        return -self._inner.gradient()

    def hessian_contribution(self) -> MatrixSystem:
        """Return the curvature -∇^2 g(w), the wrapped Hessian negated.

        Returns
        -------
         system : MatrixSystem
            The wrapped point's Hessian system, passed through ``MatrixSystem.negated``.

        """
        return self._inner.hessian_contribution().negated()


class NegatedObjective(Objective):
    """Negation -g of a wrapped objective g.

    Minimizing -g maximizes g. This is the mechanism behind ``Problem.maximize``: a
    concave objective g (with negative semidefinite Hessian) is wrapped so that the
    solver, which only minimizes, sees the convex -g.

    Value and gradient negate directly. The Hessian goes through
    ``MatrixSystem.negated``: when g advertises its curvature as ``H.negated()`` for a
    positive semidefinite H, the second negation collapses back to H itself, so the
    round trip costs no arithmetic and never asks a factorization to handle a
    negative-definite matrix.

    Negation preserves linearity (-c^T w is affine exactly when c^T w is), so the
    wrapper forwards the wrapped objective's ``is_linear``; a negated
    ``LinearObjective`` therefore keeps the analytical LP dual path.

    Parameters
    ----------
     objective : Objective
        The wrapped objective g. For a maximization it should be concave; nothing here
        checks that, just as Cvxium does not check the convexity of a minimized
        objective beyond the assembled system's positive definiteness.

    """

    def __init__(self, objective: Objective) -> None:
        self._objective = objective
        self.is_linear: bool = objective.is_linear

    @property
    def objective(self) -> Objective:
        """Return the wrapped objective g (this instance represents -g).

        Returns
        -------
         objective : Objective
            The objective given at construction.

        """
        return self._objective

    @property
    def dimension(self) -> int:
        """Return the dimension M, the wrapped objective's.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self._objective.dimension

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective value -g(w).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the objective is evaluated, of length M.

        Returns
        -------
         f0 : float
            The negated wrapped value.

        """
        return -self._objective.evaluate(w)

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient -∇g(w).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the gradient is evaluated, of length M.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The negated wrapped gradient, of length M.

        """
        return -self._objective.gradient(w)

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the curvature -∇^2 g(w), the wrapped Hessian negated.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M.

        Returns
        -------
         system : MatrixSystem
            The wrapped Hessian system, passed through ``MatrixSystem.negated``.

        """
        return self._objective.hessian_contribution(w).negated()

    def at(self, w: npt.NDArray[np.float64]) -> ObjectivePoint:
        """Return a per-point context negating the wrapped objective's point.

        Delegates to ``g.at(w)``, so any per-point caching the wrapped objective
        performs is preserved, then negates each reported quantity.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to bind, of length M.

        Returns
        -------
         point : ObjectivePoint
            A ``_NegatedObjectivePoint`` wrapping ``g.at(w)``.

        """
        return _NegatedObjectivePoint(self._objective.at(w))

    def btls_keep_in_domain(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
    ) -> float:
        """Return the wrapped objective's domain bound on the step.

        Negation does not change the domain, dom(-g) = dom g, so the bound is g's.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current point, in dom g, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.

        Returns
        -------
         alpha : float
            The largest step keeping w + α dw in dom g.

        """
        return self._objective.btls_keep_in_domain(w, dw)


class LiftedObjective(Objective):
    """Objective lifted to a larger space by ignoring trailing coordinates.

    The objective counterpart of ``LiftedConstraintLayer``. The wrapped objective sees
    only the leading ``objective.dimension`` components w of the augmented vector, and
    is constant in the ``extra_dimension`` trailing components t. Use it to minimize an
    existing objective over an epigraph formulation, or in slack-form Phase I, where the
    auxiliary t carries constraints of its own but no cost.

    The gradient is zero-padded. A ``DiagonalSystem`` Hessian stays diagonal, with zero
    curvature in the trailing coordinates (a barrier term on t makes the assembled
    system positive definite); any other structure becomes the first block of a
    ``BlockDiagonalSystem`` whose second block is a ``ZeroSystem``. Lifting preserves
    linearity, so the wrapper forwards the wrapped objective's ``is_linear``.

    Parameters
    ----------
     objective : Objective
        The wrapped objective, acting on the first ``objective.dimension`` components.
     extra_dimension : int
        Number of trailing components the objective ignores. Must be at least 1, or the
        constructor raises ``ValueError``.

    """

    def __init__(self, objective: Objective, extra_dimension: int) -> None:
        if extra_dimension < 1:
            raise ValueError(
                f"extra_dimension must be positive (got {extra_dimension})."
            )
        self._objective = objective
        self._base_dim: int = objective.dimension
        self._extra_dim: int = extra_dimension
        self.is_linear: bool = objective.is_linear

    @property
    def objective(self) -> Objective:
        """Return the wrapped objective.

        Returns
        -------
         objective : Objective
            The objective given at construction.

        """
        return self._objective

    @property
    def dimension(self) -> int:
        """Return the augmented dimension, the wrapped dimension plus the trailing one.

        Returns
        -------
         M : int
            ``objective.dimension + extra_dimension``.

        """
        return self._base_dim + self._extra_dim

    def evaluate(self, w: npt.NDArray[np.float64]) -> float:
        """Evaluate the wrapped objective on the leading components of w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point, of length ``dimension``.

        Returns
        -------
         f0 : float
            The wrapped objective's value at the leading ``objective.dimension``
            components.

        """
        return self._objective.evaluate(w[: self._base_dim])

    def gradient(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the wrapped gradient, zero-padded in the trailing components.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point, of length ``dimension``.

        Returns
        -------
         grad_f0 : npt.NDArray[np.float64]
            The gradient, of length ``dimension``.

        """
        g = np.zeros(self.dimension)
        g[: self._base_dim] = self._objective.gradient(w[: self._base_dim])
        return g

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the wrapped Hessian with zero curvature in the trailing components.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate, of length ``dimension``.

        Returns
        -------
         system : MatrixSystem
            A zero-padded ``DiagonalSystem`` when the wrapped Hessian is a
            ``DiagonalSystem``; otherwise a ``BlockDiagonalSystem`` of the wrapped
            Hessian and a ``ZeroSystem``.

        """
        base = self._objective.hessian_contribution(w[: self._base_dim])
        if isinstance(base, DiagonalSystem):
            return DiagonalSystem(np.concatenate([base.eta, np.zeros(self._extra_dim)]))
        return BlockDiagonalSystem([base, ZeroSystem(self._extra_dim)])

    def btls_keep_in_domain(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
    ) -> float:
        """Return the wrapped domain bound; the trailing components are free.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point, of length ``dimension``.
         dw : npt.NDArray[np.float64]
            Augmented step direction, of length ``dimension``.

        Returns
        -------
         alpha : float
            The wrapped objective's bound on the leading components.

        """
        return self._objective.btls_keep_in_domain(
            w[: self._base_dim], dw[: self._base_dim]
        )
