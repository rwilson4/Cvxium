"""Two-block stationarity problems: barrier-smoothed residual and coupled Newton step.

Cvxium's single-block ``Problem`` minimizes one objective over one decision variable.
This module states a different problem: find a joint stationary point of a smooth
objective ζ(x, y) in two variable blocks, a large structured x-block and a small
y-block, each confined to the strict interior of its own inequality domain. With ζ
concave in x and convex in y, the stationary point is a saddle point,
   V = max_x min_y ζ(x, y).

``TwoBlockObjective`` is the abstract objective (value, block gradients, and curvature
blocks), with ``QuadraticSaddleObjective`` as a concrete quadratic example.
``StationarityProblem`` bundles an objective with its block domains and an optional
equality on the x-block, and assembles the residuals and coupled Newton systems for both
the barrier method and the primal-dual method, without running any loop.
``TwoBlockStationaritySolver`` is the protocol the engines (``StationaritySolver`` and
``PrimalDualStationaritySolver``) satisfy, ``StationarityConvergenceError`` is what they
raise on non-convergence, and ``two_block_result_message`` formats their result
messages.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..exceptions import OptimizationError
from ..linalg.systems import (
    CoupledSaddleSystem,
    DenseSystem,
    EqualityCoupledSaddleSystem,
    MatrixSystem,
)
from ..problems.result import (
    SolveStatus,
    TwoBlockSolveResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..problems.constraint_layers import (
        ConstraintLayer,
    )
    from ..problems.linear_constraints import (
        EqualityConstraintLayer,
    )


class TwoBlockObjective(ABC):
    """Abstract smooth objective ζ(x, y) in two variable blocks.

    The decision variables split into a large, typically structured x-block of dimension
    n and a small y-block of dimension m, ordered [x; y]. Subclasses supply the
    objective's value, its two block gradients, and its three second-order blocks. The
    interface carries no minimax vocabulary: a joint stationary point of ζ is a point
    where both block gradients vanish, and whether it is a minimum, maximum or saddle
    depends on the signs of the curvature the subclass returns, not on this interface.

    The one curvature sign the interface fixes is the x-block's: ``neg_hessian_xx``
    returns the negated x-curvature -∇^2_xx ζ, which a consumer assembles as a positive
    definite block. Everything else is returned raw (un-negated), and the coupled Newton
    system applies the remaining signs.

    Notes
    -----
    ``StationarityProblem`` builds the barrier-smoothed joint stationarity residual and
    the coupled Newton step from these pieces. Supplying the x-curvature already negated
    lets the large positive definite block flow straight into a ``CoupledSaddleSystem``
    with no sign flip at the call site.

    """

    @property
    @abstractmethod
    def dimension_x(self) -> int:
        """Return the dimension n of the x-block, the large structured block.

        Returns
        -------
         n : int
            Dimension of the x-block.

        """
        ...

    @property
    @abstractmethod
    def dimension_y(self) -> int:
        """Return the dimension m of the y-block, the small block.

        Returns
        -------
         m : int
            Dimension of the y-block.

        """
        ...

    @abstractmethod
    def evaluate(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate the objective ζ(x, y).

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         value : float
            The objective value ζ(x, y).

        """
        ...

    @abstractmethod
    def gradient_x(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the raw x-gradient ∇_x ζ(x, y).

        Implementations return the plain partial gradient, un-negated; the consumer
        applies any sign it needs.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient ∇_x ζ(x, y), of length n.

        """
        ...

    @abstractmethod
    def gradient_y(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the raw y-gradient ∇_y ζ(x, y).

        Implementations return the plain partial gradient, un-negated; the consumer
        applies any sign it needs.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient ∇_y ζ(x, y), of length m.

        """
        ...

    @abstractmethod
    def neg_hessian_xx(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the negated x-curvature -∇^2_xx ζ(x, y) as a ``MatrixSystem``.

        The consumer relies on this block being positive definite: it is the large M_xx
        block of a ``CoupledSaddleSystem``. A subclass must return -∇^2_xx ζ, not
        +∇^2_xx ζ.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         system : MatrixSystem
            The negated x-curvature -∇^2_xx ζ(x, y), of order n.

        """
        ...

    @abstractmethod
    def hessian_yy(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the raw y-curvature ∇^2_yy ζ(x, y) as a ``MatrixSystem``.

        Returned un-negated. The consumer relies on this block being positive definite:
        it is the small H_yy block of a ``CoupledSaddleSystem``.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         system : MatrixSystem
            The y-curvature ∇^2_yy ζ(x, y), of order m.

        """
        ...

    @abstractmethod
    def mixed_hessian(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the mixed curvature ∇^2_xy ζ(x, y) as a dense matrix.

        The matrix is tall and skinny (n ≫ m). It is dense, but cheap despite its n
        rows: the coupled system's Schur step applies it as m batched right-hand-side
        columns, in one M_xx solve.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         mixed : npt.NDArray[np.float64]
            The mixed curvature ∇^2_xy ζ(x, y), of shape (n, m).

        """
        ...


class QuadraticSaddleObjective(TwoBlockObjective):
    """Quadratic two-block objective, concave in x and convex in y.

    Models
       ζ(x, y) = a^T x + b^T y - (1/2) x^T P x + x^T C y + (1/2) y^T R y,
    with P (n × n) symmetric positive definite, R (m × m) symmetric positive definite,
    and C (n × m) the coupling. The -(1/2) x^T P x term makes ζ concave in x and the
    +(1/2) y^T R y term makes it convex in y, so a joint stationary point is a saddle
    point. The block gradients and curvatures are
       ∇_x ζ = a - P x + C y,      ∇_y ζ = b + C^T x + R y,
       -∇^2_xx ζ = P,              ∇^2_yy ζ = R,
       ∇^2_xy ζ = C.

    Parameters
    ----------
     a : npt.NDArray[np.float64]
        Linear x-coefficient, of length n. Must be one-dimensional, or the constructor
        raises ``ValueError``.
     b : npt.NDArray[np.float64]
        Linear y-coefficient, of length m. Must be one-dimensional, or the constructor
        raises ``ValueError``.
     P : npt.NDArray[np.float64]
        Symmetric positive definite x-curvature, of shape (n, n). ζ carries -P as its
        x-curvature; P itself is what ``neg_hessian_xx`` returns. A wrong shape raises
        ``ValueError``.
     C : npt.NDArray[np.float64]
        Coupling, of shape (n, m). A wrong shape raises ``ValueError``.
     R : npt.NDArray[np.float64]
        Symmetric positive definite y-curvature, of shape (m, m). A wrong shape raises
        ``ValueError``.

    Notes
    -----
    P and R are symmetrized, as (P + P^T) / 2, so a slightly asymmetric input still
    gives a symmetric curvature. Definiteness is not checked. All arrays are copied at
    construction, so later changes to the caller's arrays do not affect the objective.

    """

    def __init__(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        P: npt.NDArray[np.float64],
        C: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
    ) -> None:
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        P = np.array(P, dtype=np.float64)
        C = np.array(C, dtype=np.float64)
        R = np.array(R, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(f"`a` must be a 1D vector, got shape {a.shape}.")
        if b.ndim != 1:
            raise ValueError(f"`b` must be a 1D vector, got shape {b.shape}.")
        n = a.shape[0]
        m = b.shape[0]
        if P.shape != (n, n):
            raise ValueError(
                f"`P` must be square of shape ({n}, {n}) to match `a`, got {P.shape}."
            )
        if R.shape != (m, m):
            raise ValueError(
                f"`R` must be square of shape ({m}, {m}) to match `b`, got {R.shape}."
            )
        if C.shape != (n, m):
            raise ValueError(
                f"`C` must have shape ({n}, {m}) to match `a` and `b`, got {C.shape}."
            )
        self._a: npt.NDArray[np.float64] = a
        self._b: npt.NDArray[np.float64] = b
        # Symmetrize defensively; the curvature blocks must be symmetric PD.
        self._P: npt.NDArray[np.float64] = 0.5 * (P + P.T)
        self._C: npt.NDArray[np.float64] = C
        self._R: npt.NDArray[np.float64] = 0.5 * (R + R.T)
        self._n: int = n
        self._m: int = m

    @property
    def dimension_x(self) -> int:
        """Return the dimension n of the x-block, the length of ``a``.

        Returns
        -------
         n : int
            Dimension of the x-block.

        """
        return self._n

    @property
    def dimension_y(self) -> int:
        """Return the dimension m of the y-block, the length of ``b``.

        Returns
        -------
         m : int
            Dimension of the y-block.

        """
        return self._m

    def evaluate(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate the quadratic objective ζ(x, y).

        Computes
           ζ(x, y) = a^T x + b^T y - (1/2) x^T P x + x^T C y + (1/2) y^T R y.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         value : float
            The objective value.

        """
        P = self._P
        R = self._R
        C = self._C
        quad = -0.5 * x @ (P @ x) + x @ (C @ y) + 0.5 * y @ (R @ y)
        return float(self._a @ x + self._b @ y + quad)

    def gradient_x(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the raw x-gradient a - P x + C y.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient ∇_x ζ, of length n.

        """
        return np.asarray(self._a - self._P @ x + self._C @ y)

    def gradient_y(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the raw y-gradient b + C^T x + R y.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The gradient ∇_y ζ, of length m.

        """
        return np.asarray(self._b + self._C.T @ x + self._R @ y)

    def neg_hessian_xx(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the negated x-curvature -∇^2_xx ζ = P, which is constant.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n; unused, since the curvature is constant.
         y : npt.NDArray[np.float64]
            The y-block, of length m; unused.

        Returns
        -------
         system : MatrixSystem
            A ``DenseSystem`` holding the symmetrized P.

        """
        return DenseSystem(self._P)

    def hessian_yy(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return the y-curvature ∇^2_yy ζ = R, which is constant.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n; unused, since the curvature is constant.
         y : npt.NDArray[np.float64]
            The y-block, of length m; unused.

        Returns
        -------
         system : MatrixSystem
            A ``DenseSystem`` holding the symmetrized R.

        """
        return DenseSystem(self._R)

    def mixed_hessian(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the mixed curvature ∇^2_xy ζ = C, which is constant.

        Returns a copy, so a caller that mutates the result cannot corrupt the stored C
        and later evaluations.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n; unused, since the curvature is constant.
         y : npt.NDArray[np.float64]
            The y-block, of length m; unused.

        Returns
        -------
         mixed : npt.NDArray[np.float64]
            A copy of C, of shape (n, m).

        """
        return self._C.copy()


@dataclass(frozen=True)
class StationarityProblem:
    """Two-block stationarity problem with log-barrier domains.

    Bundles a smooth ``TwoBlockObjective`` ζ(x, y) with the inequality
    ``ConstraintLayer`` lists whose strict interiors are the domains of the two blocks:
    ``x_domain`` constrains x, and ``y_domain`` constrains y. Each domain enters through
    its log barrier
       φ(w) = -∑ log(-f_i(w)),
    weighted by 1 / t, where the caller supplies t on each call (the outer loop that
    drives t up lives in the solvers).

    Like ``Problem``, this is a passive container: it states the barrier-smoothed joint
    stationarity residual and assembles the coupled Newton system, but runs no outer
    loop, line search, tolerance test or Phase I. The barrier-smoothed residual is
       r = [r_x; r_y],
    with
       r_x = ∇_x ζ - (1/t) g_X(x),
       r_y = ∇_y ζ + (1/t) g_Y(y),
    where g_X, the sum of the x-domain layers' ``barrier_gradient``, is the gradient of
    the x-domain barrier, and g_Y likewise for the y-domain. The concave x-block
    subtracts its barrier gradient; the convex y-block adds it. The Jacobian of r is
        _              _
       |  -M_xx   B     |
       |_ B^T     H_yy _|
    with
       M_xx = -∇^2_xx ζ + (1/t) H_X,   H_yy = ∇^2_yy ζ + (1/t) H_Y,   B = ∇^2_xy ζ,
    where H_X and H_Y are the barrier Hessians of the two domains, and M_xx and H_yy are
    positive definite. A Newton step solves this coupled saddle against (-r_x, -r_y).

    The primal-dual counterparts (``primal_dual_residual`` and
    ``build_primal_dual_coupled_system``) replace the barrier terms with explicit
    inequality duals and reduce to the same saddle structure.

    Fields
    ------
     objective : TwoBlockObjective
        The smooth two-block objective ζ(x, y).
     x_domain : Sequence[ConstraintLayer]
        Inequality layers whose strict interior is the domain of x. May be empty (an
        unconstrained x-block). Stored as a tuple after construction, so a caller cannot
        mutate it and bypass the dimension validation.
     y_domain : Sequence[ConstraintLayer]
        Inequality layers for the y-block. May be empty. Also stored as a tuple.
     x_equality : EqualityConstraintLayer | None
        An optional linear equality A_eq x = b_eq on the x-block; the y-block has no
        equality. When present, the residual gains a multiplier term and a
        primal-feasibility row, and the Newton system is bordered (see Notes). When
        ``None``, the problem is the plain two-block saddle. The multiplier λ changes
        from one Newton step to the next and belongs to the solver, not to this passive
        container, so the methods take it as an argument.

    Notes
    -----
    With an equality A_eq x = b_eq and multiplier λ, the residual extends to
       r = [r_x; r_y; r_eq],
    with
       r_x = ∇_x ζ - (1/t) g_X + A_eq^T λ,
       r_y = ∇_y ζ + (1/t) g_Y,
       r_eq = A_eq x - b_eq,
    and the Newton Jacobian becomes the equality-bordered coupled saddle
        _                       _
       |  -M_xx   B      A_eq^T  |
       |  B^T     H_yy   0       |
       |_ A_eq    0      0      _|
    an ``EqualityCoupledSaddleSystem`` rather than a ``CoupledSaddleSystem``. An
    infeasible-start Newton method handles the equality naturally: x need not satisfy
    A_eq x = b_eq initially (the r_eq row drives the residual to zero), and λ starts at
    zero.

    """

    objective: TwoBlockObjective
    x_domain: Sequence[ConstraintLayer] = field(default_factory=tuple)
    y_domain: Sequence[ConstraintLayer] = field(default_factory=tuple)
    x_equality: EqualityConstraintLayer | None = None

    def __post_init__(self) -> None:
        """Validate the domain dimensions and freeze the domains to tuples.

        Each ``ConstraintLayer`` fixes its decision-variable dimension
        (``ConstraintLayer.dimension``). A layer whose dimension disagrees with its
        block would otherwise surface as an opaque shape error deep inside the barrier
        assembly rather than at construction. The optional ``x_equality`` is checked
        against the x-block in the same way.

        ``x_domain`` and ``y_domain`` are then converted to tuples, through
        ``object.__setattr__`` since the dataclass is frozen. Without this, a caller
        holding the mutable list passed to the constructor could change it after
        construction and silently defeat the validation.

        Raises
        ------
         ValueError
            When a domain layer's dimension does not match its block, or the equality's
            dimension does not match the x-block.

        """
        n = self.dimension_x
        for layer in self.x_domain:
            if layer.dimension != n:
                raise ValueError(
                    "x_domain layer dimension does not match the objective's "
                    f"x-block: layer pins {layer.dimension}, expected {n}."
                )
        m = self.dimension_y
        for layer in self.y_domain:
            if layer.dimension != m:
                raise ValueError(
                    "y_domain layer dimension does not match the objective's "
                    f"y-block: layer pins {layer.dimension}, expected {m}."
                )
        if self.x_equality is not None and self.x_equality.dimension != n:
            raise ValueError(
                "x_equality dimension does not match the objective's x-block: "
                f"layer pins {self.x_equality.dimension}, expected {n}."
            )
        object.__setattr__(self, "x_domain", tuple(self.x_domain))
        object.__setattr__(self, "y_domain", tuple(self.y_domain))

    def solve(
        self,
        solver: TwoBlockStationaritySolver,
        x0: npt.NDArray[np.float64] | None = None,
        y0: npt.NDArray[np.float64] | None = None,
    ) -> TwoBlockSolveResult:
        """Solve the problem with a two-block engine; shorthand for ``solver.solve``.

        Like ``Problem.solve``: a caller states the two-block problem once and picks the
        algorithm by passing a ``TwoBlockStationaritySolver``, the barrier
        ``StationaritySolver`` or the Mehrotra ``PrimalDualStationaritySolver``.

        Parameters
        ----------
         solver : TwoBlockStationaritySolver
            The two-block engine to run.
         x0 : npt.NDArray[np.float64] | None
            x-block warm start, forwarded to the engine.
         y0 : npt.NDArray[np.float64] | None
            y-block warm start, forwarded to the engine.

        Returns
        -------
         result : TwoBlockSolveResult
            The engine's result.

        Raises
        ------
         StationarityConvergenceError
            Propagated from ``solver.solve`` when the solve does not converge.
         ProblemInfeasibleError
            Propagated from ``solver.solve`` when a block's domain has no strictly
            interior point.

        """
        return solver.solve(self, x0=x0, y0=y0)

    @property
    def num_equalities(self) -> int:
        """Count the equality rows p on the x-block.

        Returns
        -------
         p : int
            Number of rows of ``x_equality``; zero when it is ``None``.

        """
        if self.x_equality is None:
            return 0
        return self.x_equality.num_eq_constraints

    @property
    def dimension_x(self) -> int:
        """Return the dimension n of the x-block, taken from the objective.

        Returns
        -------
         n : int
            The objective's ``dimension_x``.

        """
        return self.objective.dimension_x

    @property
    def dimension_y(self) -> int:
        """Return the dimension m of the y-block, taken from the objective.

        Returns
        -------
         m : int
            The objective's ``dimension_y``.

        """
        return self.objective.dimension_y

    @staticmethod
    def _barrier_gradient(
        domain: Sequence[ConstraintLayer],
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Sum the layers' barrier gradients over a block's domain.

        Unlike ``_barrier_hessian``, this does not check that w is strictly interior, so
        ``residual`` can be evaluated without that check. Each layer's gradient is
           -G^T (1 / f),
        so on the boundary (some f_i = 0) the result contains inf, and outside the
        domain (some f_i > 0) it is finite but meaningless. Callers must pass a strictly
        interior w.

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         w : npt.NDArray[np.float64]
            The block's point, strictly inside ``domain``.

        Returns
        -------
         grad : npt.NDArray[np.float64]
            The summed barrier gradient, of the same length as w; zeros for an empty
            domain.

        """
        total = np.zeros_like(w)
        for layer in domain:
            total = total + layer.barrier_gradient(w)
        return total

    @staticmethod
    def _barrier_hessian(
        domain: Sequence[ConstraintLayer],
        w: npt.NDArray[np.float64],
        objective_curvature: MatrixSystem,
        t: float,
    ) -> MatrixSystem:
        """Add a block's weighted barrier Hessian to its objective curvature.

        Each layer's log-barrier Hessian comes from ``hessian_contribution`` with the
        barrier weights λ_i = -1 / f_i. The layers' terms are summed with
        ``MatrixSystem.plus``, scaled by 1 / t, and added to the objective curvature:
           objective_curvature + (1/t) ∑ H_i.

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         w : npt.NDArray[np.float64]
            The block's point, strictly inside ``domain``.
         objective_curvature : MatrixSystem
            The block's objective curvature (-∇^2_xx ζ for x, ∇^2_yy ζ for y).
         t : float
            Inverse barrier weight; the barrier Hessian enters with weight 1 / t.

        Returns
        -------
         system : MatrixSystem
            The assembled block curvature; ``objective_curvature`` itself for an empty
            domain.

        Raises
        ------
         ValueError
            When w is not strictly inside some layer (a constraint value is not
            negative).

        """
        if not domain:
            return objective_curvature
        atoms: list[MatrixSystem] = []
        for layer in domain:
            f = layer.constraints(w)
            if not np.all(f < 0.0):
                raise ValueError(
                    "point is not strictly interior to the domain; the log-barrier "
                    f"Hessian requires all constraint values < 0, got {f!r}."
                )
            # Barrier weight lambda_i = -1 / f_i (> 0 for strictly interior f_i).
            lmbda = -1.0 / f
            atoms.append(layer.hessian_contribution(w, lmbda=lmbda))
        barrier_sum = reduce(lambda left, right: left.plus(right), atoms)
        return barrier_sum.scaled(1.0 / t).plus(objective_curvature)

    @staticmethod
    def _num_block_constraints(
        domain: Sequence[ConstraintLayer],
        dimension: int,
    ) -> int:
        """Count the inequality constraints across a block's domain.

        This is the length the block's stacked primal-dual dual vector must have.

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         dimension : int
            The block's dimension, passed to each layer's ``num_constraints``.

        Returns
        -------
         count : int
            Total number of constraints; zero for an empty domain.

        """
        return sum(layer.num_constraints(dimension) for layer in domain)

    @staticmethod
    def _jacobian_transpose_dual(
        domain: Sequence[ConstraintLayer],
        w: npt.NDArray[np.float64],
        duals: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint force G^T λ = ∑ λ_i ∇f_i(w) over a block's domain.

        The dual vector is stacked in the domain's layer order; each layer takes its own
        slice of length ``num_constraints(len(w))`` and contributes through
        ``grad_constraints_transpose_multiply``. This primal-dual constraint force takes
        the place of the barrier term (1/t) g = -(1/t) G^T (1 / f) in the stationarity
        rows; the two coincide on the central path, where λ_i = -1 / (t f_i).

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         w : npt.NDArray[np.float64]
            The block's point.
         duals : npt.NDArray[np.float64]
            The block's inequality duals, stacked in layer order.

        Returns
        -------
         force : npt.NDArray[np.float64]
            G^T λ, of the same length as w; zeros for an empty domain.

        """
        total = np.zeros_like(w)
        offset = 0
        for layer in domain:
            count = layer.num_constraints(w.shape[0])
            lam_i = duals[offset : offset + count]
            offset += count
            total = total + layer.grad_constraints_transpose_multiply(w, lam_i)
        return total

    @staticmethod
    def _primal_dual_hessian(
        domain: Sequence[ConstraintLayer],
        w: npt.NDArray[np.float64],
        objective_curvature: MatrixSystem,
        duals: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Add a block's dual-weighted constraint curvature to its objective curvature.

        The primal-dual counterpart of ``_barrier_hessian``: each layer's curvature atom
           C_i = ∇^2 f_i - (1 / f_i) ∇f_i ∇f_i^T
        is weighted by the actual dual λ_i (through ``hessian_contribution``) rather
        than the barrier weight -1 / f_i, and the sum, with no 1 / t scaling, is added
        to the objective curvature:
           objective_curvature + ∑ λ_i C_i.
        The two assemblies coincide on the central path, where λ_i = -1 / (t f_i).

        Parameters
        ----------
         domain : Sequence[ConstraintLayer]
            The block's inequality layers; may be empty.
         w : npt.NDArray[np.float64]
            The block's point, strictly inside ``domain``; the atom's 1 / f_i term
            diverges on the boundary.
         objective_curvature : MatrixSystem
            The block's objective curvature (-∇^2_xx ζ for x, ∇^2_yy ζ for y).
         duals : npt.NDArray[np.float64]
            The block's inequality duals, stacked in layer order.

        Returns
        -------
         system : MatrixSystem
            The assembled block curvature; ``objective_curvature`` itself for an empty
            domain.

        Raises
        ------
         ValueError
            When w is not strictly inside some layer (a constraint value is not
            negative).

        """
        if not domain:
            return objective_curvature
        atoms: list[MatrixSystem] = []
        offset = 0
        for layer in domain:
            f = layer.constraints(w)
            if not np.all(f < 0.0):
                raise ValueError(
                    "point is not strictly interior to the domain; the primal-dual "
                    f"Hessian requires all constraint values < 0, got {f!r}."
                )
            count = f.shape[0]
            lam_i = duals[offset : offset + count]
            offset += count
            atoms.append(layer.hessian_contribution(w, lmbda=lam_i))
        pd_sum = reduce(lambda left, right: left.plus(right), atoms)
        return pd_sum.plus(objective_curvature)

    def residual(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
        lam: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute the barrier-smoothed joint stationarity residual.

        Without an ``x_equality`` this is [r_x; r_y], with
           r_x = ∇_x ζ - (1/t) g_X,   r_y = ∇_y ζ + (1/t) g_Y.
        With one, it extends to [r_x; r_y; r_eq]: r_x gains the multiplier term A_eq^T
        λ, and the equality row is r_eq = A_eq x - b_eq.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n, strictly inside ``x_domain`` (not checked; see
            ``_barrier_gradient``).
         y : npt.NDArray[np.float64]
            The y-block, of length m, strictly inside ``y_domain`` (not checked).
         t : float
            The inverse barrier weight, positive; the barrier enters with weight 1 / t.
         lam : npt.NDArray[np.float64] | None
            The equality multiplier λ, of length p. Required when ``x_equality`` is set,
            and not allowed otherwise.

        Returns
        -------
         residual : npt.NDArray[np.float64]
            [r_x; r_y], of length n + m, or [r_x; r_y; r_eq], of length n + m + p, when
            ``x_equality`` is set.

        Raises
        ------
         ValueError
            When t ≤ 0, when ``lam`` is missing although ``x_equality`` is set (or
            supplied although it is not), or when ``lam`` has the wrong length.

        """
        if t <= 0.0:
            raise ValueError(f"barrier weight t must be strictly positive, got {t=}.")
        obj = self.objective
        r_x = obj.gradient_x(x, y) - (1.0 / t) * self._barrier_gradient(
            self.x_domain, x
        )
        r_y = obj.gradient_y(x, y) + (1.0 / t) * self._barrier_gradient(
            self.y_domain, y
        )
        if self.x_equality is None:
            if lam is not None:
                raise ValueError("lam was supplied but the problem has no x_equality.")
            return np.concatenate([r_x, r_y])
        if lam is None:
            raise ValueError("x_equality is set; the multiplier lam is required.")
        p = self.x_equality.num_eq_constraints
        lam = np.asarray(lam, dtype=np.float64)
        if lam.shape != (p,):
            raise ValueError(
                f"lam must have shape ({p},) to match x_equality, got {lam.shape}."
            )
        A_eq = self.x_equality.A
        b_eq = np.asarray(self.x_equality.b, dtype=np.float64)
        r_x = r_x + np.asarray(A_eq.T @ lam, dtype=np.float64).ravel()
        r_eq = np.asarray(A_eq @ x, dtype=np.float64).ravel() - b_eq
        return np.concatenate([r_x, r_y, r_eq])

    def build_coupled_system(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
    ) -> CoupledSaddleSystem | EqualityCoupledSaddleSystem:
        """Assemble the barrier coupled Newton system for one step.

        Without an ``x_equality`` this is the plain saddle
           [[-M_xx, B], [B^T, H_yy]],
        a ``CoupledSaddleSystem``. With one it is the equality-bordered saddle
           [[-M_xx, B, A_eq^T], [B^T, H_yy, 0], [A_eq, 0, 0]],
        an ``EqualityCoupledSaddleSystem``.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n, strictly inside ``x_domain``.
         y : npt.NDArray[np.float64]
            The y-block, of length m, strictly inside ``y_domain``.
         t : float
            The inverse barrier weight, positive; the barrier Hessians enter with weight
            1 / t.

        Returns
        -------
         system : CoupledSaddleSystem | EqualityCoupledSaddleSystem
            The coupled saddle with
               M_xx = -∇^2_xx ζ + (1/t) H_X,
               H_yy = ∇^2_yy ζ + (1/t) H_Y,
               B = ∇^2_xy ζ,
            bordered by A_eq when ``x_equality`` is set.

        Raises
        ------
         ValueError
            When t ≤ 0, or when x or y is not strictly inside its domain (propagated
            from ``_barrier_hessian``).

        """
        if t <= 0.0:
            raise ValueError(f"barrier weight t must be strictly positive, got {t=}.")
        obj = self.objective
        m_xx = self._barrier_hessian(self.x_domain, x, obj.neg_hessian_xx(x, y), t)
        h_yy = self._barrier_hessian(self.y_domain, y, obj.hessian_yy(x, y), t)
        coupling = obj.mixed_hessian(x, y)
        if self.x_equality is None:
            return CoupledSaddleSystem(m_xx, coupling, h_yy)
        return EqualityCoupledSaddleSystem(m_xx, coupling, h_yy, self.x_equality.A)

    def newton_step(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute one Newton step (Δx, Δy) for the barrier-smoothed residual.

        Solves the coupled saddle against (-r_x, -r_y), the update that drives the
        residual to zero to first order.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n, strictly inside ``x_domain``.
         y : npt.NDArray[np.float64]
            The y-block, of length m, strictly inside ``y_domain``.
         t : float
            The inverse barrier weight, positive.

        Returns
        -------
         delta_x : npt.NDArray[np.float64]
            The x-block step, of length n.
         delta_y : npt.NDArray[np.float64]
            The y-block step, of length m.

        Raises
        ------
         ValueError
            When the problem has an ``x_equality``: the equality multiplier carries over
            from step to step, so a stateless single step is ill-defined. Drive such a
            problem through ``StationaritySolver``, which owns the multiplier, or
            assemble the step from ``residual`` and ``build_coupled_system``. Also
            propagated from those two methods when t ≤ 0 or a block is not strictly
            inside its domain.
         NewtonStepError
            Propagated from ``CoupledSaddleSystem.solve`` when M_xx or the Schur
            complement cannot be solved.

        """
        if self.x_equality is not None:
            raise ValueError(
                "newton_step does not support x_equality (the multiplier is "
                "stateful); use StationaritySolver, or residual + "
                "build_coupled_system directly."
            )
        n = self.dimension_x
        r = self.residual(x, y, t)
        r_x = r[:n]
        r_y = r[n:]
        system = self.build_coupled_system(x, y, t)
        # x_equality is None here (guarded above), so the assembled system is the
        # plain coupled saddle; assert to narrow the union for the type checker.
        assert isinstance(system, CoupledSaddleSystem)
        return system.solve(-r_x, -r_y)

    def primal_dual_residual(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
        lam_eq: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute the primal-dual joint stationarity residual, with explicit duals.

        The primal-dual counterpart of ``residual``: the barrier's smoothed gradient
        terms are replaced by the explicit constraint forces G^T λ built from the
        per-block inequality duals. Without an ``x_equality`` this is [r_x; r_y]; with
        one it extends to [r_x; r_y; r_eq], where r_x gains A_eq^T λ_eq and the equality
        row is r_eq = A_eq x - b_eq:
           r_x = ∇_x ζ - G_X^T λ_x  (+ A_eq^T λ_eq),
           r_y = ∇_y ζ + G_Y^T λ_y,
        where G_X and G_Y are the x-domain and y-domain constraint Jacobians. The
        concave x-block subtracts its constraint force and the convex y-block adds it,
        matching the sign convention of ``residual``. These are only the
        dual-feasibility (stationarity) rows; the complementarity rows S Λ 1 = μ 1
        belong to the solver, which carries the slacks, duals and centering.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n.
         y : npt.NDArray[np.float64]
            The y-block, of length m.
         lam_x : npt.NDArray[np.float64]
            The x-domain inequality duals, one per x-domain constraint, stacked in layer
            order; empty when ``x_domain`` is empty.
         lam_y : npt.NDArray[np.float64]
            The y-domain inequality duals, in the same convention.
         lam_eq : npt.NDArray[np.float64] | None
            The equality multiplier λ_eq, of length p. Required when ``x_equality`` is
            set, and not allowed otherwise.

        Returns
        -------
         residual : npt.NDArray[np.float64]
            [r_x; r_y], of length n + m, or [r_x; r_y; r_eq], of length n + m + p, when
            ``x_equality`` is set.

        Raises
        ------
         ValueError
            When ``lam_x`` or ``lam_y`` has the wrong length, or when ``lam_eq`` is
            missing although ``x_equality`` is set (or supplied although it is not), or
            has the wrong length.

        """
        n = self.dimension_x
        m = self.dimension_y
        lam_x = np.asarray(lam_x, dtype=np.float64)
        lam_y = np.asarray(lam_y, dtype=np.float64)
        p_x = self._num_block_constraints(self.x_domain, n)
        p_y = self._num_block_constraints(self.y_domain, m)
        if lam_x.shape != (p_x,):
            raise ValueError(
                f"lam_x must have shape ({p_x},) to match x_domain, got {lam_x.shape}."
            )
        if lam_y.shape != (p_y,):
            raise ValueError(
                f"lam_y must have shape ({p_y},) to match y_domain, got {lam_y.shape}."
            )
        obj = self.objective
        r_x = obj.gradient_x(x, y) - self._jacobian_transpose_dual(
            self.x_domain, x, lam_x
        )
        r_y = obj.gradient_y(x, y) + self._jacobian_transpose_dual(
            self.y_domain, y, lam_y
        )
        if self.x_equality is None:
            if lam_eq is not None:
                raise ValueError(
                    "lam_eq was supplied but the problem has no x_equality."
                )
            return np.concatenate([r_x, r_y])
        if lam_eq is None:
            raise ValueError("x_equality is set; the multiplier lam_eq is required.")
        p = self.x_equality.num_eq_constraints
        lam_eq = np.asarray(lam_eq, dtype=np.float64)
        if lam_eq.shape != (p,):
            raise ValueError(
                f"lam_eq must have shape ({p},) to match x_equality, got {lam_eq.shape}."
            )
        A_eq = self.x_equality.A
        b_eq = np.asarray(self.x_equality.b, dtype=np.float64)
        r_x = r_x + np.asarray(A_eq.T @ lam_eq, dtype=np.float64).ravel()
        r_eq = np.asarray(A_eq @ x, dtype=np.float64).ravel() - b_eq
        return np.concatenate([r_x, r_y, r_eq])

    def build_primal_dual_coupled_system(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        lam_x: npt.NDArray[np.float64],
        lam_y: npt.NDArray[np.float64],
    ) -> CoupledSaddleSystem | EqualityCoupledSaddleSystem:
        """Assemble the primal-dual coupled Newton system, with explicit duals.

        The primal-dual counterpart of ``build_coupled_system``: the weights the barrier
        method derives from t are replaced by the explicit duals in ``lam_x`` and
        ``lam_y``, so the reduced Newton system has the same two-block saddle structure
           [[-M_xx, B], [B^T, H_yy]]
        (or its equality-bordered form), with
           M_xx = -∇^2_xx ζ + ∑ λ_x,i C_x,i,
           H_yy = ∇^2_yy ζ + ∑ λ_y,j C_y,j,
           B = ∇^2_xy ζ,
        where
           C_i = ∇^2 f_i - (1 / f_i) ∇f_i ∇f_i^T
        is the shared curvature atom. On the central path (λ_i = -1 / (t f_i)) this
        reproduces ``build_coupled_system``.

        The equality multiplier enters only the residual, not the Newton matrix (the
        border A_eq is structural), so this method, like ``build_coupled_system``, takes
        no equality multiplier.

        Parameters
        ----------
         x : npt.NDArray[np.float64]
            The x-block, of length n, strictly inside ``x_domain`` (the atom's 1 / f_i
            term requires it).
         y : npt.NDArray[np.float64]
            The y-block, of length m, strictly inside ``y_domain``.
         lam_x : npt.NDArray[np.float64]
            The x-domain inequality duals, stacked in layer order as in
            ``primal_dual_residual``; strictly positive for a genuine primal-dual
            iterate.
         lam_y : npt.NDArray[np.float64]
            The y-domain inequality duals, in the same convention.

        Returns
        -------
         system : CoupledSaddleSystem | EqualityCoupledSaddleSystem
            The coupled saddle with M_xx, H_yy and B as above, bordered by A_eq when
            ``x_equality`` is set.

        Raises
        ------
         ValueError
            When ``lam_x`` or ``lam_y`` has the wrong length, or when a block point is
            not strictly inside its domain.

        """
        n = self.dimension_x
        m = self.dimension_y
        lam_x = np.asarray(lam_x, dtype=np.float64)
        lam_y = np.asarray(lam_y, dtype=np.float64)
        p_x = self._num_block_constraints(self.x_domain, n)
        p_y = self._num_block_constraints(self.y_domain, m)
        if lam_x.shape != (p_x,):
            raise ValueError(
                f"lam_x must have shape ({p_x},) to match x_domain, got {lam_x.shape}."
            )
        if lam_y.shape != (p_y,):
            raise ValueError(
                f"lam_y must have shape ({p_y},) to match y_domain, got {lam_y.shape}."
            )
        obj = self.objective
        m_xx = self._primal_dual_hessian(
            self.x_domain, x, obj.neg_hessian_xx(x, y), lam_x
        )
        h_yy = self._primal_dual_hessian(self.y_domain, y, obj.hessian_yy(x, y), lam_y)
        coupling = obj.mixed_hessian(x, y)
        if self.x_equality is None:
            return CoupledSaddleSystem(m_xx, coupling, h_yy)
        return EqualityCoupledSaddleSystem(m_xx, coupling, h_yy, self.x_equality.A)


def two_block_result_message(
    status: SolveStatus,
    solver: str,
    gap: float | None = None,
    cause: str | None = None,
) -> str:
    """Build a human-readable message for a two-block stationarity result.

    The two-block engines share this helper, so the ``message`` of the
    ``TwoBlockSolveResult`` they return (or attach to a raised
    ``StationarityConvergenceError``) reads the same whichever engine produced it.

    Parameters
    ----------
     status : SolveStatus
        The solve outcome to describe. ``OPTIMAL`` and ``ACCEPTABLE`` are the returned
        outcomes; ``MAX_ITERATIONS`` is the status of the best-effort result attached to
        a raised ``StationarityConvergenceError``. Any other status is described by
        name.
     solver : str
        The label of the engine that produced the result (for example,
        ``"stationarity-barrier"`` or ``"stationarity-primal-dual"``).
     gap : float | None
        The certified saddle-value gap at the returned iterate, the width of the
        reported lower and upper bracket on the saddle value. When given, it is appended
        to the message, as the single-block solvers append their duality gaps, so a
        reader can see how tightly the saddle value is pinned whatever the status (a
        wide gap on a ``MAX_ITERATIONS`` result flags a non-converged solve). ``None``
        omits it.
     cause : str | None
        The engine's own account of why the solve stopped, as a clause. When given, it
        replaces the description derived from ``status``. ``MAX_ITERATIONS`` is the
        shared status of every early exit (a spent iteration budget, a slack that
        reached its boundary, a Newton step that could not stay strictly interior), so
        the status alone cannot name the cause, and calling it an exhausted budget would
        send a reader off to raise a limit that was never reached. ``None`` on the exits
        the status does determine.

    Returns
    -------
     message : str
        A one-line description of the outcome for ``solver``.

    """
    descriptions = {
        SolveStatus.OPTIMAL: "converged to the requested tolerance",
        SolveStatus.ACCEPTABLE: "converged to the soft (acceptable) tolerance",
        SolveStatus.MAX_ITERATIONS: (
            "exhausted its iteration budget before reaching tolerance"
        ),
    }
    gap_note = f" (saddle gap {gap:.3g})" if gap is not None else ""
    if cause is not None:
        return f"Two-block stationarity solver '{solver}' stopped: {cause}{gap_note}."
    detail = descriptions.get(status, f"terminated with status {status.name}")
    return f"Two-block stationarity solver '{solver}' {detail}{gap_note}."


class StationarityConvergenceError(OptimizationError):
    """Error raised when a two-block stationarity solve fails to converge.

    The stationarity engines (``StationaritySolver`` and
    ``PrimalDualStationaritySolver``) return a ``TwoBlockSolveResult`` only when the
    joint stationarity problem is solved to the user's tolerance: an ``OPTIMAL`` result,
    or an ``ACCEPTABLE`` one at the soft tolerance. Every other termination (an
    exhausted outer or inner iteration budget, or a centering or Newton step that can
    make no further progress) is a non-convergence failure, and raises this exception
    rather than returning a status a caller might overlook.

    This mirrors the barrier method's ``InteriorPointMethodError``: running out of
    iterations is never proof that the problem is infeasible, so it is never reported as
    an ``INFEASIBLE`` status. Infeasibility needs a genuine certificate, raised as
    ``ProblemInfeasibleError`` from Phase I.

    Parameters
    ----------
     message : str
        Description of the failure.
     suboptimality : float
        The failure metric at the point of giving up: the barrier gap on outer
        exhaustion, or the stationarity residual or KKT score on a stalled step.
     result : TwoBlockSolveResult
        The best-effort iterate reached before the solve gave up, with status
        ``SolveStatus.MAX_ITERATIONS``. It did not reach the user's tolerance but is
        often close, so a caller that repackages a terminal status, or continues a sweep
        past a hard subproblem, can catch the error and inspect ``result`` instead of
        re-running with a looser tolerance. Stored as ``result``; its x-block also
        becomes the base class's ``last_iterate``.

    """

    def __init__(
        self,
        message: str,
        suboptimality: float,
        result: TwoBlockSolveResult,
    ) -> None:
        # ``last_iterate`` (the ``OptimizationError`` base contract) is the
        # primary decision block; the full two-block iterate rides on ``result``.
        super().__init__(message, suboptimality, result.x)
        self.result: TwoBlockSolveResult = result


@runtime_checkable
class TwoBlockStationaritySolver(Protocol):
    """Structural interface for a two-block stationarity solver.

    A two-block ``StationarityProblem`` with a start (x0, y0) does not fit the
    single-``Problem``, single-``x0`` signature of Cvxium's ``Solver``, so the two-block
    engines, the barrier ``StationaritySolver`` and the primal-dual
    ``PrimalDualStationaritySolver``, sit outside that hierarchy. This protocol names
    the shape they share, so a caller can hold either without knowing which engine it
    drives.

    An implementer solves the joint stationarity problem from optional per-block warm
    starts and returns a ``TwoBlockSolveResult``. With structural typing, an engine
    conforms by shape alone, with no base class and no registration.
    ``runtime_checkable`` also lets a caller test conformance with ``isinstance``; that
    checks only that the method exists, not its signature, which is left to the static
    type checker.

    """

    def solve(
        self,
        problem: StationarityProblem,
        x0: npt.NDArray[np.float64] | None = None,
        y0: npt.NDArray[np.float64] | None = None,
    ) -> TwoBlockSolveResult:
        """Solve a two-block stationarity problem from optional warm starts.

        Parameters
        ----------
         problem : StationarityProblem
            The two-block stationarity problem to solve.
         x0 : npt.NDArray[np.float64] | None
            x-block warm start; an implementer may feasibilize it, or choose a default
            when it is ``None``.
         y0 : npt.NDArray[np.float64] | None
            y-block warm start, with the same semantics as ``x0``.

        Returns
        -------
         result : TwoBlockSolveResult
            The converged iterate, work counts, per-block duals, convergence diagnostics
            and status, returned only when the solve reaches the user's (soft)
            tolerance. An engine that computes the saddle value reports it, with its
            bracket, in the result's ``objective``.

        Raises
        ------
         StationarityConvergenceError
            When the solve fails to converge (an exhausted budget or a stalled step);
            the best-effort iterate is on the exception's ``result``.
         ProblemInfeasibleError
            When a block's domain has no strictly interior point.

        """
        ...
