"""Dimension-lifting adapters that express slack-form Phase I as a ``Problem``.

When the active layer of a ``Problem`` (its last inequality layer, the one whose
feasibility Phase I seeks) is a ``SlackConstraintLayer``, ``Problem.make_phase1`` builds
a slack-form Phase I problem: an ordinary ``Problem`` over the augmented variable
   w_aug = [w; s]
of dimension M + 1, which minimizes the slack s. This module supplies the layers of that
problem:
- ``LiftedConstraintLayer`` lifts each background layer (already satisfied at the
  starting point) from dimension M to the augmented dimension; the layer sees only w.
- ``_SlackActiveLayer`` wraps the active layer, relaxing its constraints to f_i(w) ≤ s.
- ``_SlackUpperBoundLayer`` bounds the slack, s ≤ s_max, so the barrier subproblem has a
  unique solution.

``LiftedConstraintLayer`` is public: a caller formulating its own problem over an
augmented variable (an epigraph [w; t], say) lifts its existing layers with it, and its
objective with ``LiftedObjective``. The other two classes are construction details of
``Problem.make_phase1`` and not part of the public API.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..linalg.systems import (
    ArrowSystem,
    BlockDiagonalSystem,
    DiagonalSystem,
    LowRankUpdatedSystem,
    MatrixSystem,
    ZeroSystem,
)
from ..problems.constraint_layers import (
    ConstraintLayer,
    SlackConstraintLayer,
)

if TYPE_CHECKING:
    from ..solvers.settings import (
        LineSearchSettings,
    )


class LiftedConstraintLayer(SlackConstraintLayer):
    """Constraint layer lifted to act on a larger decision variable.

    The wrapped layer sees only the first ``base_dimension`` components of the augmented
    vector and ignores the trailing ones. The augmented Jacobian gets zero columns for
    the extra components, and the augmented Hessian contributions get a zero block.

    ``Problem.make_phase1`` uses it to embed the background layers, which act on w of
    dimension M, into a slack-form Phase I problem over w_aug = [w; s]. Callers that
    augment w with auxiliary variables of their own use it the same way, with
    ``extra_dimension`` set to the number of auxiliaries.

    The class subclasses ``SlackConstraintLayer``, so when a lifted layer is the last
    inequality layer of a caller-built problem, ``Problem.make_phase1`` treats it like
    any other slack layer and builds a slack-form Phase I problem around it.

    Parameters
    ----------
     base : ConstraintLayer
        The layer being lifted. It sees ``w_aug[:base_dimension]`` only.
     base_dimension : int
        Dimension M the base layer was constructed for.
     extra_dimension : int
        Number of trailing components the base layer ignores. The default, 1, is the
        single slack of a slack-form Phase I. Must be at least 1, or the constructor
        raises ``ValueError``.

    Notes
    -----
    ``is_linear`` is forwarded from the base layer, so a solver choosing its dual path
    does not mistake a lifted squared-norm layer for an affine one.

    """

    def __init__(
        self,
        base: ConstraintLayer,
        base_dimension: int,
        extra_dimension: int = 1,
    ) -> None:
        if extra_dimension < 1:
            raise ValueError(
                f"extra_dimension must be positive (got {extra_dimension})."
            )
        self._base: ConstraintLayer = base
        self._base_dim: int = base_dimension
        self._extra_dim: int = extra_dimension
        self.is_linear: bool = base.is_linear

    @property
    def dimension(self) -> int:
        """Return the augmented dimension, M plus the trailing components.

        Returns
        -------
         n : int
            ``base_dimension + extra_dimension``.

        """
        return self._base_dim + self._extra_dim

    def num_constraints(self, M: int) -> int:
        """Count the constraints, which are the base layer's.

        Parameters
        ----------
         M : int
            Augmented dimension of the decision variable; ignored, since the base layer
            is queried at ``base_dimension``.

        Returns
        -------
         m : int
            Number of constraints in the base layer.

        """
        return self._base.num_constraints(self._base_dim)

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the base layer's shifted constraints on the leading components.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point; only ``w[:base_dimension]`` is read.
         shift : float
            Relaxation subtracted from every constraint value, as in
            ``ConstraintLayer.constraints``.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The base layer's shifted constraint values, of length m.

        """
        return self._base.constraints(w[: self._base_dim], shift=shift)

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest feasible step by delegating to the base layer.

        The trailing components do not enter this layer's constraints, so the augmented
        step is feasible exactly when its truncation is. Delegating, rather than
        inheriting the default backtracking search, lets a base layer with a closed-form
        bound keep it under lifting.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current augmented iterate, strictly feasible; only ``w[:base_dimension]`` is
            read.
         dw : npt.NDArray[np.float64]
            Augmented step direction; only ``dw[:base_dimension]`` is read.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Settings passed through to the base layer; ``None`` uses the defaults.

        Returns
        -------
         alpha : float
            The base layer's ``btls_max_step`` on the truncated point and direction.

        """
        return self._base.btls_max_step(
            w[: self._base_dim],
            dw[: self._base_dim],
            shift=shift,
            ds=ds,
            line_search=line_search,
        )

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the augmented Jacobian: base columns, then zero columns.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point; only ``w[:base_dimension]`` is read.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The base Jacobian padded with ``len(w) - base_dimension`` zero columns, of
            shape (m, len(w)).

        """
        G = self._base.grad_constraints(w[: self._base_dim])
        extra = len(w) - self._base_dim
        if extra == 0:
            return G
        return np.asarray(np.hstack([G, np.zeros((G.shape[0], extra))]))

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the augmented Jacobian.

        The extra columns are zero, so
           G v = G_base v[:base_dimension].

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point; only ``w[:base_dimension]`` is read.
         v : npt.NDArray[np.float64]
            Augmented vector to multiply; only ``v[:base_dimension]`` is read.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length m.

        """
        return self._base.grad_constraints_multiply(
            w[: self._base_dim], v[: self._base_dim]
        )

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed augmented Jacobian.

        Computes G_base^T u and pads it with zeros in the extra components.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point; only ``w[:base_dimension]`` is read.
         u : npt.NDArray[np.float64]
            Vector of length m to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length ``len(w)``.

        """
        result_base = self._base.grad_constraints_transpose_multiply(
            w[: self._base_dim], u
        )
        extra = len(w) - self._base_dim
        if extra == 0:
            return result_base
        result = np.zeros(len(w))
        result[: self._base_dim] = result_base
        return result

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Lift the base layer's Hessian contribution to the augmented dimension.

        The base layer has curvature only in its own ``base_dimension`` coordinates, so
        its ``MatrixSystem`` becomes the first block of a ``BlockDiagonalSystem``, with
        a ``ZeroSystem`` over the trailing coordinates. ``Problem.build_hessian_system``
        sums this with the other layers' full-dimension contributions, so the embedded
        base composes correctly whatever its structure (diagonal, low rank, and so on).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate; only ``w[:base_dimension]`` is read.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m, passed to the base layer. Required;
            the ``None`` default mirrors ``ConstraintLayer.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            The block-diagonal lifted contribution, or the base contribution itself when
            ``len(w)`` equals ``base_dimension``.

        Raises
        ------
         ValueError
            Propagated from the base layer's ``hessian_contribution`` when ``lmbda`` is
            ``None``.

        """
        base_contribution = self._base.hessian_contribution(
            w[: self._base_dim], lmbda, shift=shift
        )
        extra = len(w) - self._base_dim
        if extra == 0:
            return base_contribution
        return BlockDiagonalSystem([base_contribution, ZeroSystem(extra)])

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Lift the base layer's weighted curvature to the augmented dimension.

        The lifted constraint f_i(w_aug[:M]) has curvature only in the base coordinates,
        so the base ``hessian_curvature`` becomes the first block of a
        ``BlockDiagonalSystem``, with a ``ZeroSystem`` over the trailing coordinates, as
        in ``hessian_contribution``.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate; only ``w[:base_dimension]`` is read.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length m.

        Returns
        -------
         system : MatrixSystem
            The block-diagonal lifted curvature, or the base curvature itself when
            ``len(w)`` equals ``base_dimension``.

        """
        base_curvature = self._base.hessian_curvature(w[: self._base_dim], lmbda)
        extra = len(w) - self._base_dim
        if extra == 0:
            return base_curvature
        return BlockDiagonalSystem([base_curvature, ZeroSystem(extra)])


class _SlackActiveLayer(SlackConstraintLayer):
    """Slack-relaxed active layer of a slack-form Phase I problem.

    Wraps the active ``SlackConstraintLayer`` and presents it on the augmented variable
    w_aug = [w; s]:
    - The constraints become f_i(w) - s ≤ 0, computed through the base layer's ``shift``
      argument: ``base.constraints(w, shift=s)`` returns f_i(w) - s.
    - The Jacobian gains a slack column of -1s.

    With c_i = f_i(w) - s < 0 and barrier weights λ_i = -1 / c_i, the weighted Hessian
    contribution is the 2 × 2 block matrix
                _                                                      _
       H_aug = |  G^T diag(1 / c^2) G + ∑ λ_i ∇^2 f_i   -G^T (1 / c^2)  |
               |_ -(G^T (1 / c^2))^T                    ∑ 1 / c_i^2    _|
    where 1 / c^2 is taken componentwise.

    ``hessian_contribution`` adapts its return type to the base layer's structure: a
    diagonal w-block (as from ``BoundsConstraintLayer``) gives an ``ArrowSystem``, while
    a low-rank w-block (as from ``ImbalanceConstraintLayer`` or
    ``LinearInequalityConstraintLayer``) folds the whole matrix into a single low-rank
    update, so the solve stays stable as c → 0.

    Parameters
    ----------
     base : SlackConstraintLayer
        The active layer, whose feasibility Phase I seeks.
     base_dimension : int
        Dimension M of the primal variable w; the slack sits at index M.
     s_max : float | None
        The slack upper bound that ``_SlackUpperBoundLayer`` enforces. Stored, but no
        method reads it.

    """

    def __init__(
        self,
        base: SlackConstraintLayer,
        base_dimension: int,
        s_max: float | None = None,
    ) -> None:
        self._base: SlackConstraintLayer = base
        self._base_dim: int = base_dimension
        # The slack upper bound that _SlackUpperBoundLayer enforces. Stored for
        # reference; no method reads it (that layer supplies the slack-slot term).
        self._s_max: float | None = s_max

    @property
    def dimension(self) -> int:
        """Return the augmented dimension M + 1, the primal variable plus the slack.

        Returns
        -------
         n : int
            ``base_dimension + 1``.

        """
        return self._base_dim + 1

    @property
    def base_dimension(self) -> int:
        """Return the primal dimension M, to which a solution de-augments.

        The barrier solver's ``finalize_solution`` reads it to drop the trailing slack
        from a slack-form Phase I solution.

        Returns
        -------
         M : int
            Dimension of the primal variable w.

        """
        return self._base_dim

    def num_constraints(self, M: int) -> int:
        """Count the constraints, which are the base layer's.

        Parameters
        ----------
         M : int
            Augmented dimension of the decision variable; ignored, since the slack does
            not change the count.

        Returns
        -------
         m : int
            Number of constraints in the base layer.

        """
        return self._base.num_constraints(self._base_dim)

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the slack-relaxed constraints f_i(w) - s - shift.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1.
         shift : float
            Further relaxation subtracted from every constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The relaxed constraint values, of length m.

        """
        s = float(w[self._base_dim])
        return self._base.constraints(w[: self._base_dim], shift=s + shift)

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest feasible step, folding the slack into the base shift.

        At step α this layer's constraint reads
           f_i(w + α dw) - (s + α ds_s) - (shift + α ds) < 0,
        where ds_s is the slack component of the direction. That is the base layer's own
        relaxation with shift replaced by s + shift and ds by ds_s + ds, so the slack
        coupling costs only an argument fold, and a base layer with a closed-form bound
        keeps it.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current augmented iterate [w; s], strictly feasible, of length M + 1.
         dw : npt.NDArray[np.float64]
            Augmented step direction, of length M + 1.
         shift : float
            Further relaxation at α = 0.
         ds : float
            Rate at which the further relaxation grows with α.
         line_search : LineSearchSettings | None
            Settings passed through to the base layer; ``None`` uses the defaults.

        Returns
        -------
         alpha : float
            The base layer's ``btls_max_step`` with the folded shift and rate.

        """
        s = float(w[self._base_dim])
        ds_s = float(dw[self._base_dim])
        return self._base.btls_max_step(
            w[: self._base_dim],
            dw[: self._base_dim],
            shift=s + shift,
            ds=ds_s + ds,
            line_search=line_search,
        )

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the augmented Jacobian: base columns, then a slack column of -1s.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1.

        Returns
        -------
         G_aug : npt.NDArray[np.float64]
            The augmented Jacobian [G, -1], of shape (m, M + 1).

        """
        G = self._base.grad_constraints(w[: self._base_dim])
        return np.asarray(np.hstack([G, -np.ones((G.shape[0], 1))]))

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the augmented Jacobian.

        With v = [v_w; v_s],
           G_aug v = G v_w - v_s 1.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1.
         v : npt.NDArray[np.float64]
            Augmented vector [v_w; v_s] to multiply, of length M + 1.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G_aug v, of length m.

        """
        v_w = v[: self._base_dim]
        v_s = float(v[self._base_dim])
        return np.asarray(
            self._base.grad_constraints_multiply(w[: self._base_dim], v_w) - v_s
        )

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed augmented Jacobian.

        The slack row of G_aug^T is all -1s, so
           G_aug^T u = [G^T u; -∑ u_i].

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1.
         u : npt.NDArray[np.float64]
            Vector of length m to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G_aug^T u, of length M + 1.

        """
        result = np.zeros(self._base_dim + 1)
        result[: self._base_dim] = self._base.grad_constraints_transpose_multiply(
            w[: self._base_dim], u
        )
        result[self._base_dim] = -float(np.sum(u))
        return result

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the slack-augmented weighted Hessian contribution.

        With c = ``base.constraints(w, shift=s + shift)`` (negative on the interior) and
        effective weights μ = -λ / c, the augmented contribution is the 2 × 2 block
        matrix
            _                                        _
           |  G^T diag(μ) G + ∑ λ_i ∇^2 f_i   -G^T μ  |
           |_ -(G^T μ)^T                      ∑ μ_i  _|
        since the slack column of the augmented Jacobian is -1. The upper-left w-block
        is the base layer's own ``hessian_contribution``, the border is -G^T μ, and the
        corner is ∑ μ_i. The barrier weights λ = -1 / c give μ = 1 / c^2. When
        ``Problem.build_hessian_system`` sums the contributions, the other Phase I
        layers' slack-slot terms (the slack upper bound) fold into the corner, and the
        background layers' terms into the w-block.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate [w; s], of length M + 1.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m. Required; the ``None`` default exists
            only so the signature matches ``Objective.hessian_contribution``.
         shift : float
            Further relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            A ``LowRankUpdatedSystem`` on a zero base when the base contribution is a
            ``LowRankUpdatedSystem``; otherwise an ``ArrowSystem``.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``, or when some relaxed constraint value is not
            strictly negative.

        Notes
        -----
        A diagonal base (such as ``BoundsConstraintLayer``) gives an ``ArrowSystem``,
        whose solve dispatches to the compensated-summation diagonal-arrow kernel: O(M),
        and stable as c → 0.

        A low-rank base (such as ``ImbalanceConstraintLayer`` or
        ``LinearInequalityConstraintLayer``) needs different handling. An
        ``ArrowSystem`` over a low-rank w-block would solve through
        ``solve_block_plus_one`` and its scalar Schur complement
           S = corner - border^T H_ww^{-1} border,
        which loses all precision as c → 0: both terms grow like 1 / c while their
        difference stays O(1). Instead, the method expresses the whole matrix as a
        single rank-p update on a zero base of dimension M + 1. With
           κ_aug = [G^T; -1^T],
        of shape (M + 1, p),
                                    _                        _
           κ_aug diag(μ) κ_aug^T = |  G^T diag(μ) G   -G^T μ  |
                                   |_ -(G^T μ)^T      ∑ μ_i  _|
        which is the whole augmented contribution, because these affine layers have no
        constraint curvature (see ``ConstraintLayer._low_rank_barrier_contribution``).
        Its solve goes through the equilibrated Woodbury kernel
        (``solve_rank_p_update``), which stays stable as c → 0.

        This branch rebuilds the matrix from G and μ and discards the base contribution,
        so it relies on every base that returns a ``LowRankUpdatedSystem`` having no
        curvature and no base matrix beyond the rank-p term.

        """
        if lmbda is None:
            raise ValueError("_SlackActiveLayer.hessian_contribution requires `lmbda`.")
        M = self._base_dim
        w_inner = w[:M]
        base_shift = float(w[M]) + shift  # base shift = slack s (plus any outer shift)
        c = self._base.constraints(w_inner, shift=base_shift)
        if np.any(c >= 0.0):
            raise ValueError(
                "Barrier argument out of domain: every f_i(w) - s must be strictly "
                "negative for the slack-augmented active layer."
            )
        mu = -lmbda / c  # effective rank weights; positive on the strict interior
        interior = self._base.hessian_contribution(w_inner, lmbda, shift=base_shift)
        if isinstance(interior, LowRankUpdatedSystem):
            # Fold the rank-update w-block and the slack border/corner into one
            # rank-p update so the solve avoids the scalar-Schur cancellation; the
            # background diagonals and slack-upper-bound corner fold into the zero
            # base during Problem.build_hessian_system.
            G = self._base.grad_constraints(w_inner)  # shape (p, M)
            p = G.shape[0]
            kappa_aug = np.vstack([G.T, -np.ones((1, p))])  # shape (M + 1, p)
            return ZeroSystem(M + 1).low_rank_update(kappa_aug, mu)
        # Diagonal (or any non-low-rank) w-block: the generic arrow is correct, and
        # for a diagonal interior its solve is the stable O(M) compensated kernel.
        border = -self._base.grad_constraints_transpose_multiply(w_inner, mu)
        corner = float(np.sum(mu))
        return ArrowSystem(interior, border, corner)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Embed the base layer's curvature; the slack relaxation adds none.

        The relaxed constraint f_i(w) - s is the base constraint plus a linear term, so
        the augmented curvature is the base curvature in the w-block with a
        ``ZeroSystem`` in the slack slot. There is no arrow border or corner: those come
        from the Jacobian outer-product term that ``hessian_contribution`` carries, not
        from curvature.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate [w; s], of length M + 1.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length m.

        Returns
        -------
         system : MatrixSystem
            The block-diagonal curvature, base curvature then a zero slack slot.

        """
        base_curvature = self._base.hessian_curvature(w[: self._base_dim], lmbda)
        extra = len(w) - self._base_dim
        if extra == 0:
            return base_curvature
        return BlockDiagonalSystem([base_curvature, ZeroSystem(extra)])


class _SlackUpperBoundLayer(SlackConstraintLayer):
    """Scalar upper bound s ≤ s_max on the slack of a slack-form Phase I problem.

    ``Problem.make_phase1`` adds it so that the slack stays bounded and the barrier
    subproblem has a unique solution. It acts on the augmented variable w_aug = [w; s]
    and constrains only the slack slot. The single constraint is
       f(w_aug) = w_aug[M] - s_max ≤ 0,
    with gradient e_M (zero in the w slots, 1 in the slack slot), so its barrier Hessian
    contribution is the diagonal entry 1 / (s_max - s)^2 in the slack slot.

    Parameters
    ----------
     s_max : float
        Upper bound on the slack.
     base_dimension : int
        Dimension M of the primal variable w. The slack sits at index M of the augmented
        variable.

    """

    def __init__(self, s_max: float, base_dimension: int) -> None:
        self._s_max: float = s_max
        self._base_dim: int = base_dimension

    @property
    def dimension(self) -> int:
        """Return the augmented dimension M + 1, the primal variable plus the slack.

        Returns
        -------
         n : int
            ``base_dimension + 1``.

        """
        return self._base_dim + 1

    def num_constraints(self, M: int) -> int:
        """Return one, the number of constraints in this layer.

        Parameters
        ----------
         M : int
            Augmented dimension of the decision variable; the count does not depend on
            it.

        Returns
        -------
         m : int
            Always 1.

        """
        return 1

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the shifted slack bound s - s_max - shift.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1.
         shift : float
            Relaxation subtracted from the constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The single shifted constraint value, as an array of length 1.

        """
        return np.asarray([float(w[self._base_dim]) - self._s_max - shift])

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps the slack below its shifted ceiling.

        The constraint at step α reads
           s + α ds_s < s_max + shift + α ds,
        where ds_s is the slack component of the direction. The slack closes on its
        ceiling at rate ds_s - ds, and only a positive rate can bind, so the bound is a
        single division. Closed form; ``line_search`` is ignored.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current augmented iterate [w; s], strictly feasible, of length M + 1.
         dw : npt.NDArray[np.float64]
            Augmented step direction, of length M + 1.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Ignored.

        Returns
        -------
         alpha : float
            (s_max + shift - s) / (ds_s - ds) when the rate is positive; inf otherwise.

        """
        rate = float(dw[self._base_dim]) - ds
        if rate <= 0.0:
            return np.inf
        gap = self._s_max + shift - float(w[self._base_dim])
        return gap / rate

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the Jacobian e_M^T: zero in the w slots, 1 in the slack slot.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1; the Jacobian does not depend on it.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The Jacobian, of shape (1, M + 1).

        """
        g = np.zeros((1, self._base_dim + 1))
        g[0, self._base_dim] = 1.0
        return g

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the Jacobian, which picks out its slack slot.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1; the Jacobian does not depend on it.
         v : npt.NDArray[np.float64]
            Augmented vector to multiply, of length M + 1.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v = v[M], as an array of length 1.

        """
        return np.asarray([float(v[self._base_dim])])

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed Jacobian, u[0] e_M.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented point [w; s], of length M + 1; the Jacobian does not depend on it.
         u : npt.NDArray[np.float64]
            Vector of length 1 to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M + 1, zero except in the slack slot.

        """
        result = np.zeros(self._base_dim + 1)
        result[self._base_dim] = float(u[0])
        return result

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the weighted Hessian contribution, a diagonal in the slack slot.

        The single constraint
           f = s - s_max - shift
        is linear, so its curvature atom -(1 / f) ∇f ∇f^T has ∇f = e_M and is the
        diagonal (-1 / f) e_M e_M^T. Weighted by λ, the slack slot receives
           -λ / f = λ / gap,   gap = s_max + shift - s > 0,
        and every other coordinate is zero. The barrier method's weight λ = -1 / f makes
        the entry 1 / gap^2; the primal-dual method passes its dual variable directly.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate [w; s], of length M + 1.
         lmbda : npt.NDArray[np.float64] | None
            Weight λ, of length 1. Required; the ``None`` default exists only so the
            signature matches ``Objective.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint value, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            A ``DiagonalSystem`` with λ / gap in the slack slot and zeros elsewhere.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``, or when gap ≤ 0 (the slack is not strictly below
            s_max + shift).

        """
        if lmbda is None:
            raise ValueError(
                "_SlackUpperBoundLayer.hessian_contribution requires `lmbda`."
            )
        gap = self._s_max + shift - float(w[self._base_dim])
        if gap <= 0.0:
            raise ValueError(
                "Barrier argument out of domain: slack must be strictly below "
                f"s_max + shift (got gap = {gap})."
            )
        eta = np.zeros(self._base_dim + 1)
        eta[self._base_dim] = float(lmbda[0]) / gap
        return DiagonalSystem(eta)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return zero curvature, since the slack bound is affine.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Augmented iterate, of length M + 1; used only for its length.
         lmbda : npt.NDArray[np.float64]
            Weight λ, of length 1; unused, since the curvature is zero.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` of dimension ``len(w)``.

        """
        return ZeroSystem(len(w))
