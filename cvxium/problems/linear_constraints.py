"""Affine constraint layers for Cvxium's interior-point solvers.

Collects the layers whose feasible regions are defined by affine relations on the
decision variable w:
- ``EqualityConstraintLayer``: equality constraints A w = b. It holds the data, a cached
  SVD of A, and the factory that builds the per-step KKT saddle; ``EqualitySolver`` does
  the solving. It is not a ``ConstraintLayer``: a ``Problem`` carries it separately.
- ``BoundsConstraintLayer``: box bounds lb ≤ w ≤ ub, with a diagonal barrier Hessian.
- ``ImbalanceConstraintLayer``: imbalance constraints |(B w - c)_i| ≤ ψ_i.
- ``LinearInequalityConstraintLayer``: two-sided linear inequalities cl ≤ A w ≤ cu, with
  a low-rank barrier Hessian.
- ``BandedLinearInequalityLayer``: one-sided inequalities A w ≤ cu whose barrier Hessian
  A^T diag(d) A is banded under the given column order, assembled as a
  ``BandedSystem`` and solved by banded Cholesky instead of a width-m Woodbury update.
- ``ArrowLinearInequalityLayer``: like the banded layer, but with a dense K-column
  border on the banded core, assembled as an ``ArrowSystem``.

Every inequality layer here subclasses ``SlackConstraintLayer``, so when one is the
active layer, ``Problem.make_phase1`` builds a slack-form Phase I problem for it.

"""

import logging
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

from ..linalg.numerical_helpers import (
    aspect_ratio_svd,
)
from ..linalg.systems import (
    ArrowSystem,
    BandedSystem,
    DiagonalSystem,
    MatrixSystem,
    SaddleFactory,
    SaddleSystem,
    SchurComplementSaddleFactory,
    ZeroSystem,
)
from ..problems.constraint_layers import (
    SlackConstraintLayer,
)
from ..solvers.settings import (
    LineSearchSettings,
)

logger: logging.Logger = logging.getLogger("cvxium.linear_constraints")


class EqualityConstraintLayer:
    """Linear equality constraints A w = b, with a cached SVD of A.

    Holds the equality data, the lazily computed and cached singular value decomposition
    of A that the equality-handling algorithms consume, and the factory that builds the
    per-step KKT saddle. It does no solving itself: the projection and KKT solves live
    in ``EqualitySolver``, which operates on a layer. Separating the data and its
    factorization from the algorithm lets several problems share one layer, and thus one
    warm SVD, without recomputation.

    Parameters
    ----------
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        Equality constraint matrix, of shape (p, M), dense or sparse. A sparse A is
        useful only together with a ``saddle_factory`` that exploits it (such as
        ``SparseSaddleFactory``); the default factory densifies it, and ``svd_A`` is
        dense-only.
     b : npt.NDArray[np.float64]
        Equality right-hand side; must be one-dimensional of length p (checked by
        ``assert``).
     saddle_factory : SaddleFactory | None
        The helper that builds the per-step KKT saddle from a Hessian block and A.
        Injecting it is how the caller chooses which structure of A to exploit without
        this layer knowing or branching on it. ``None`` uses
        ``SchurComplementSaddleFactory`` (a dense Schur complement).

    """

    def __init__(
        self,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
        b: npt.NDArray[np.float64],
        saddle_factory: SaddleFactory | None = None,
    ) -> None:
        # Typed Any so the type checker need not reason about the dense/sparse union
        # on every consumer of ``A`` (same convention as
        # LinearInequalityConstraintLayer).
        A_any: Any = A
        p, _ = A_any.shape
        assert len(b.shape) == 1
        assert b.shape[0] == p
        self.A: Any = A_any
        self.b = b
        self._saddle_factory: SaddleFactory = (
            saddle_factory
            if saddle_factory is not None
            else SchurComplementSaddleFactory()
        )

    @property
    def num_eq_constraints(self) -> int:
        """Return the number p of equality constraints, the row count of A.

        Returns
        -------
         p : int
            Number of equality constraints.

        """
        return self.A.shape[0]

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M, the column count of A.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self.A.shape[1]

    @property
    def saddle_factory(self) -> SaddleFactory:
        """Return the factory that builds this layer's per-step KKT saddle.

        ``Problem`` passes it on when it builds the augmented equality layer of a
        slack-form Phase I problem, so that layer exploits the same structure.

        Returns
        -------
         saddle_factory : SaddleFactory
            The factory given at construction, or the default
            ``SchurComplementSaddleFactory``.

        """
        return self._saddle_factory

    def saddle_system(self, hessian: MatrixSystem) -> SaddleSystem:
        """Build the per-step KKT saddle for this equality over a Hessian block.

        Pairs the equality data A with the supplied Hessian H into the bordered Newton
        system
                _        _
           K = |  H   A^T |
               |_ A    0 _|
        This parallels ``ConstraintLayer.hessian_contribution``: the equality object
        builds the per-step solvable, just as a constraint layer builds its Hessian
        contribution. The injected ``saddle_factory`` builds the concrete saddle, so
        which structure of A is exploited is the caller's choice, not this layer's.

        Parameters
        ----------
         hessian : MatrixSystem
            The Hessian block H, of order M.

        Returns
        -------
         saddle : SaddleSystem
            The saddle-point system K, ready to solve.

        """
        return self._saddle_factory(hessian, self.A)

    @cache  # noqa: B019
    def svd_A(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Compute and cache the thin SVD of A.

        Delegates to ``aspect_ratio_svd`` in ``numerical_helpers``, which takes the
        direct thin SVD, or, for a wide A (at least 1,000 columns and at most a tenth as
        many rows as columns), a QR factorization of A^T followed by the SVD of the
        small triangular factor. The result is cached on the layer, so every problem
        sharing the layer reuses it. ``EqualitySolver`` uses it for the equality
        projection and the barrier-parameter initialization.

        Returns
        -------
         U : npt.NDArray[np.float64]
            Left singular vectors, of shape (p, k), where k = min(p, M).
         s : npt.NDArray[np.float64]
            Singular values, of length k, in descending order.
         Vh : npt.NDArray[np.float64]
            Right singular vectors, transposed, of shape (k, M), with A = U diag(s) Vh.

        Raises
        ------
         NotImplementedError
            When A is sparse; a sparse A must use the injected saddle path instead.

        """
        if scipy.sparse.issparse(self.A):
            raise NotImplementedError(
                "svd_A is dense-only; the SVD-based equality projection and "
                "barrier initialization do not yet support a sparse A. Pass a "
                "dense A for those paths, or use the sparse saddle path "
                "(SparseSaddleFactory), which does not require the SVD."
            )
        return aspect_ratio_svd(self.A)


class BoundsConstraintLayer(SlackConstraintLayer):
    """Box-bound constraints lb ≤ w ≤ ub.

    Either bound may be omitted (``None``), but not both. The constraint vector stacks
    the active sides, lower block first:
       lower:   f_i(w) = lb_i - w_i ≤ 0,
       upper:   f_i(w) = w_i - ub_i ≤ 0.
    With both sides present and every bound finite, the vector has length 2 M, laid out
    as [lower block, upper block]; with one side present it has length M.

    An infinite entry (-inf in ``lb``, +inf in ``ub``) means that component has no bound
    on that side and contributes no constraint: each block then holds only the
    components whose bound is finite, in index order. Keeping a row w_i - inf ≤ 0 would
    not be harmless: its barrier term -log(inf) makes the barrier objective -inf
    everywhere, which disables the line search, and its multiplier 0 times f_i = -inf
    makes the Lagrangian NaN.

    Every Jacobian row is ±e_i, so the barrier Hessian contribution is diagonal and is
    returned as a ``DiagonalSystem``.

    Parameters
    ----------
     lb : float | list[float] | npt.NDArray[np.float64] | None
        Lower bound on each component of w; a scalar broadcasts to all components, and
        ``None`` disables the lower bound. The constructor raises ``ValueError`` when
        both ``lb`` and ``ub`` are ``None``, when they are not broadcast-compatible, or
        when some component has lb ≥ ub (a degenerate or infeasible box).
     ub : float | list[float] | npt.NDArray[np.float64] | None
        Upper bound on each component of w; a scalar broadcasts to all components, and
        ``None`` disables the upper bound.
     dimension : int | None
        Decision-variable dimension M. Required when neither bound is an array, since a
        scalar bound pins no size on its own. When an array bound pins M, an explicit
        ``dimension`` is only a cross-check. The constructor raises ``ValueError`` when
        no dimension can be resolved, when two array bounds disagree on length, or when
        ``dimension`` disagrees with an array bound.

    """

    def __init__(
        self,
        lb: float | list[float] | npt.NDArray[np.float64] | None = None,
        ub: float | list[float] | npt.NDArray[np.float64] | None = None,
        dimension: int | None = None,
    ) -> None:
        if lb is None and ub is None:
            raise ValueError(
                "BoundsConstraintLayer requires at least one of `lb` or `ub`."
            )
        if lb is not None and ub is not None:
            lb_arr = np.atleast_1d(np.asarray(lb, dtype=np.float64))
            ub_arr = np.atleast_1d(np.asarray(ub, dtype=np.float64))
            try:
                lb_b, ub_b = np.broadcast_arrays(lb_arr, ub_arr)
            except ValueError as e:
                raise ValueError(
                    f"`lb` (shape {lb_arr.shape}) and `ub` (shape {ub_arr.shape}) "
                    "are not broadcast-compatible."
                ) from e
            if not np.all(lb_b < ub_b):
                raise ValueError(
                    "All components must satisfy `lb < ub` strictly; got "
                    f"min(ub - lb) = {float(np.min(ub_b - lb_b))!r}."
                )
        self.lb = lb
        self.ub = ub
        self._dimension: int = self._resolve_dimension(lb, ub, dimension)
        # Whether some entry is infinite, i.e. some component lacks a bound on one
        # side. Only then do the methods below select the bounded components; a
        # layer whose bounds are all finite takes exactly the original path.
        self._partial: bool = (
            lb is not None and not bool(np.all(np.isfinite(np.asarray(lb))))
        ) or (ub is not None and not bool(np.all(np.isfinite(np.asarray(ub)))))

    @staticmethod
    def _resolve_dimension(
        lb: float | list[float] | npt.NDArray[np.float64] | None,
        ub: float | list[float] | npt.NDArray[np.float64] | None,
        dimension: int | None,
    ) -> int:
        """Resolve the dimension M from array-bound lengths or an explicit value.

        An array-like bound pins M through its length; two array bounds must agree. An
        explicit ``dimension`` must match any array-pinned length. With no array bound,
        the explicit ``dimension`` is required.

        Parameters
        ----------
         lb : float | list[float] | npt.NDArray[np.float64] | None
            The lower bound, as given to the constructor.
         ub : float | list[float] | npt.NDArray[np.float64] | None
            The upper bound, as given to the constructor.
         dimension : int | None
            The explicit dimension, or ``None``.

        Returns
        -------
         M : int
            The resolved decision-variable dimension.

        Raises
        ------
         ValueError
            When two array bounds disagree on length, when ``dimension`` disagrees with
            an array bound, or when there is no array bound and no ``dimension``.

        """
        M: int | None = None
        if isinstance(lb, list | tuple | np.ndarray) and np.ndim(lb) > 0:
            M = len(np.atleast_1d(np.asarray(lb)))
        if isinstance(ub, list | tuple | np.ndarray) and np.ndim(ub) > 0:
            M_ub = len(np.atleast_1d(np.asarray(ub)))
            if M is not None and M_ub != M:
                raise ValueError(
                    f"Array `lb` (length {M}) and `ub` (length {M_ub}) must agree "
                    "on length."
                )
            M = M_ub
        if M is not None:
            if dimension is not None and dimension != M:
                raise ValueError(
                    f"Explicit `dimension` ({dimension}) disagrees with the "
                    f"array-bound length ({M})."
                )
            return M
        if dimension is None:
            raise ValueError(
                "BoundsConstraintLayer must specify `dimension` when bounds are scalar."
            )
        return dimension

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M resolved at construction.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self._dimension

    def _lb_array(self, M: int) -> npt.NDArray[np.float64]:
        """Return the lower bound as an array of length M.

        A scalar ``lb`` is broadcast; an array-like one is converted as is. Requires
        ``lb`` to be set (checked by ``assert``).

        Parameters
        ----------
         M : int
            Dimension of the decision variable.

        Returns
        -------
         lb : npt.NDArray[np.float64]
            The lower bound on each component, of length M.

        """
        assert self.lb is not None
        if isinstance(self.lb, list | tuple | np.ndarray):
            return np.asarray(self.lb, dtype=np.float64)
        return np.full(M, self.lb, dtype=np.float64)

    def _ub_array(self, M: int) -> npt.NDArray[np.float64]:
        """Return the upper bound as an array of length M.

        A scalar ``ub`` is broadcast; an array-like one is converted as is. Requires
        ``ub`` to be set (checked by ``assert``).

        Parameters
        ----------
         M : int
            Dimension of the decision variable.

        Returns
        -------
         ub : npt.NDArray[np.float64]
            The upper bound on each component, of length M.

        """
        assert self.ub is not None
        if isinstance(self.ub, list | tuple | np.ndarray):
            return np.asarray(self.ub, dtype=np.float64)
        return np.full(M, self.ub, dtype=np.float64)

    def _bounded(self, M: int) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Return the indices of the components with a finite lower and upper bound.

        Parameters
        ----------
         M : int
            Dimension of the decision variable.

        Returns
        -------
         lo : npt.NDArray[np.intp]
            Indices, in increasing order, of the components with a finite lower bound;
            empty when ``lb`` is ``None``.
         hi : npt.NDArray[np.intp]
            Indices, in increasing order, of the components with a finite upper bound;
            empty when ``ub`` is ``None``.

        """
        none = np.empty(0, dtype=np.intp)
        lo = none if self.lb is None else np.flatnonzero(np.isfinite(self._lb_array(M)))
        hi = none if self.ub is None else np.flatnonzero(np.isfinite(self._ub_array(M)))
        return lo, hi

    def num_constraints(self, M: int) -> int:
        """Count the constraints: M per active side, less the infinite entries.

        Parameters
        ----------
         M : int
            Dimension of the decision variable.

        Returns
        -------
         m : int
            The number of finite bounds; 2 M when both sides are present and finite.

        """
        if self._partial:
            lo, hi = self._bounded(M)
            return len(lo) + len(hi)
        return M * (int(self.lb is not None) + int(self.ub is not None))

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the stacked bound constraints, shifted by ``shift``.

        Returns the lower block followed by the upper block,
           lb_i - w_i - shift,   w_i - ub_i - shift,
        over the active sides and the components with a finite bound.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraints are evaluated, of length M.
         shift : float
            Relaxation subtracted from every constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The shifted constraint values, of length m.

        """
        if self._partial:
            M = len(w)
            lo, hi = self._bounded(M)
            lb_lo = self._lb_array(M)[lo] if self.lb is not None else np.empty(0)
            ub_hi = self._ub_array(M)[hi] if self.ub is not None else np.empty(0)
            return np.concatenate([-w[lo] + lb_lo - shift, w[hi] - ub_hi - shift])
        if self.lb is not None and self.ub is not None:
            return np.concatenate(
                [
                    np.asarray(-w + self.lb - shift),
                    np.asarray(w - self.ub - shift),
                ]
            )
        if self.lb is not None:
            return np.asarray(-w + self.lb - shift)
        assert self.ub is not None  # Satisfy type-checker
        return np.asarray(w - self.ub - shift)

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the dense constraint Jacobian, rows -e_i (lower) and +e_i (upper).

        Forms an M × M identity, so it costs O(M^2) memory; the solvers use the
        structured products and ``hessian_contribution`` instead.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M; only its length is
            used.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (m, M).

        """
        eye = np.eye(len(w))
        if self._partial:
            lo, hi = self._bounded(len(w))
            return np.vstack([-eye[lo], eye[hi]])
        if self.lb is not None and self.ub is not None:
            return np.vstack([-eye, eye])
        if self.lb is not None:
            return -eye
        return eye

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian, stacking -v and v.

        Returns -v_i over the finite lower bounds followed by v_i over the finite upper
        bounds, in O(M) time.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M; only its length is
            used.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length m.

        """
        if self._partial:
            lo, hi = self._bounded(len(w))
            return np.concatenate([-v[lo], v[hi]])
        if self.lb is not None and self.ub is not None:
            return np.concatenate([-v, v])
        if self.lb is not None:
            return -v
        return v

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian.

        Splits u into its lower block u_lo and upper block u_up and returns
           -u_lo + u_up,
        with each block scattered to the components it covers, in O(M) time.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated, of length M; only its length is
            used.
         u : npt.NDArray[np.float64]
            Vector of length m to multiply, in the layer's constraint order.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        if self._partial:
            lo, hi = self._bounded(len(w))
            g = np.zeros(len(w))
            g[lo] -= u[: len(lo)]
            g[hi] += u[len(lo) :]
            return g
        if self.lb is not None and self.ub is not None:
            M = len(w)
            return np.asarray(-u[:M] + u[M:])
        if self.lb is not None:
            return -u
        return u

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the weighted diagonal barrier curvature as a ``DiagonalSystem``.

        Each bound is affine with gradient ±e_i, so its atom -(1 / f_i) ∇f_i ∇f_i^T is
        diagonal. With the slacks
           s_lo = w - lb + shift,   s_up = ub - w + shift,
        which are the negated shifted constraint values, coordinate i receives
           η_i = g_lo,i + g_up,i,
           g_lo,i = λ_lo,i / s_lo,i,   g_up,i = λ_up,i / s_up,i.
        The barrier method's λ = 1 / s recovers 1 / s_lo,i^2 + 1 / s_up,i^2; the
        primal-dual method passes its duals directly.

        The system also carries η^{-1} for the stable assembly path. With one side
        present it is s / λ, an exact reciprocal. With both sides it factors out the
        dominant term,
           1 / η_i = (1 / g_max) / (g_lo,i / g_max + g_up,i / g_max),
           g_max = max(g_lo,i, g_up,i),
        which stays finite even when g_lo,i + g_up,i overflows because a slack is tiny.
        With infinite bound entries, a component bounded on one side gets that side's
        term alone, and one bounded on neither gets η_i = 0 and an infinite reciprocal.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M, strictly inside the shifted box.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m, in the layer's constraint order
            (lower block, then upper block). Required; the ``None`` default exists only
            so the signature matches ``Objective.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : DiagonalSystem
            The diagonal η, with its stably computed reciprocal.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``.

        """
        if lmbda is None:
            raise ValueError(
                "BoundsConstraintLayer.hessian_contribution requires `lmbda`."
            )
        M = len(w)
        if self._partial:
            lo, hi = self._bounded(M)
            g_lo = np.zeros(M)
            g_up = np.zeros(M)
            if len(lo):
                g_lo[lo] = lmbda[: len(lo)] / (w[lo] - self._lb_array(M)[lo] + shift)
            if len(hi):
                g_up[hi] = lmbda[len(lo) :] / (self._ub_array(M)[hi] - w[hi] + shift)
            eta = g_lo + g_up
            dominant = np.maximum(g_lo, g_up)
            eta_inverse = np.full(M, np.inf)
            pos = dominant > 0
            eta_inverse[pos] = (1.0 / dominant[pos]) / (
                g_lo[pos] / dominant[pos] + g_up[pos] / dominant[pos]
            )
            return DiagonalSystem(eta, eta_inverse)
        if self.lb is not None and self.ub is not None:
            s_lo = w - self.lb + shift
            s_up = self.ub - w + shift
            g_lo = lmbda[:M] / s_lo
            g_up = lmbda[M:] / s_up
            eta = g_lo + g_up
            dominant = np.maximum(g_lo, g_up)
            eta_inverse = (1.0 / dominant) / (g_lo / dominant + g_up / dominant)
        elif self.lb is not None:
            s_lo = w - self.lb + shift
            eta = lmbda / s_lo
            eta_inverse = s_lo / lmbda
        else:
            assert self.ub is not None  # Satisfy type-checker
            s_up = self.ub - w + shift
            eta = lmbda / s_up
            eta_inverse = s_up / lmbda
        return DiagonalSystem(np.asarray(eta), np.asarray(eta_inverse))

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return zero curvature, since every bound is affine (∇^2 f_i = 0).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; only its length is used.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length m; unused.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` of order M.

        """
        return ZeroSystem(len(w))

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps the shifted bounds strictly satisfied.

        Finds the largest α such that, on the active sides,
           lb - shift - α ds < w + α dw < ub + shift + α ds.
        The lower side can bind only where dw_i + ds < 0 and the upper side only where
        dw_i - ds > 0; the step is the smallest closed-form ratio over those components,
        so ``line_search`` is ignored. An infinite bound entry gives an infinite ratio
        and never binds.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly inside the shifted box, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Unused; accepted for signature compatibility.

        Returns
        -------
         alpha : float
            The largest feasible step; inf when no bound binds.

        """
        alpha = np.inf
        if self.lb is not None:
            denom = dw + ds  # Coefficient of alpha in the lower violation.
            mask = denom < 0
            if np.any(mask):
                lb_arr = self._lb_array(len(w))
                alpha_lo = float(
                    np.min((w[mask] - lb_arr[mask] + shift) / -denom[mask])
                )
                alpha = min(alpha, alpha_lo)
        if self.ub is not None:
            denom = dw - ds  # Coefficient of alpha in the upper violation.
            mask = denom > 0
            if np.any(mask):
                ub_arr = self._ub_array(len(w))
                alpha_up = float(np.min((ub_arr[mask] - w[mask] + shift) / denom[mask]))
                alpha = min(alpha, alpha_up)
        return alpha

    def is_satisfied(
        self,
        w: npt.NDArray[np.float64],
    ) -> bool:
        """Check whether lb < w < ub holds strictly on the active sides.

        An infinite bound entry always holds.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         satisfied : bool
            True when every finite bound holds strictly.

        """
        if self.lb is not None and not bool(np.all(w > self.lb)):
            return False
        return self.ub is None or bool(np.all(w < self.ub))


class ImbalanceConstraintLayer(SlackConstraintLayer):
    """Imbalance constraints |(B w - c)_i| ≤ ψ_i, the infinity-norm bound ‖B w - c‖ ≤ ψ.

    Encoded as 2 q linear inequalities: a first block bounding each residual from below,
    then a second bounding it from above,
       f_i(w)     = -(B w - c)_i - ψ_i ≤ 0,   i = 1, ..., q,
       f_{q+i}(w) =  (B w - c)_i - ψ_i ≤ 0,   i = 1, ..., q.
    The Jacobian is [-B; B], and the barrier Hessian contribution is a width-2q low-rank
    update, suited to q much smaller than M.

    Parameters
    ----------
     B : npt.NDArray[np.float64]
        Constraint matrix, of shape (q, M).
     c : npt.NDArray[np.float64]
        Constraint target, of length q.
     psi : float | list[float] | npt.NDArray[np.float64]
        Tolerance ψ on each row, of length q; a scalar broadcasts to all rows. Must be
        positive for the constraints to have a strict interior.

    """

    def __init__(
        self,
        B: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        psi: float | list[float] | npt.NDArray[np.float64],
    ) -> None:
        self.B = B
        self.c = c
        q = B.shape[0]
        if isinstance(psi, list | tuple | np.ndarray):
            self._psi: npt.NDArray[np.float64] = np.asarray(psi, dtype=np.float64)
        else:
            self._psi = np.full(q, psi, dtype=np.float64)

    @property
    def q(self) -> int:
        """Return the number q of imbalance rows, the row count of B.

        Returns
        -------
         q : int
            Number of imbalance rows; the layer has 2 q constraints.

        """
        return self.B.shape[0]

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M, the column count of B.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self.B.shape[1]

    @property
    def psi(self) -> npt.NDArray[np.float64]:
        """Return the per-row tolerance ψ as an array.

        Returns
        -------
         psi : npt.NDArray[np.float64]
            The tolerance on each row, of length q.

        """
        return self._psi

    def num_constraints(self, M: int) -> int:
        """Count the constraints, 2 q (one lower and one upper per row).

        Parameters
        ----------
         M : int
            Dimension of the decision variable; the count does not depend on it.

        Returns
        -------
         m : int
            Always 2 q.

        """
        return 2 * self.q

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the stacked imbalance constraints, shifted by ``shift``.

        Returns -(B w - c) - ψ - shift followed by (B w - c) - ψ - shift.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraints are evaluated, of length M.
         shift : float
            Relaxation subtracted from every constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The shifted constraint values, of length 2 q.

        """
        Bw_minus_c = self.B @ w - self.c
        return np.concatenate(
            [
                -Bw_minus_c - self._psi - shift,
                Bw_minus_c - self._psi - shift,
            ]
        )

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the constraint Jacobian G = [-B; B].

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated; unused, since the layer is affine.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (2 q, M).

        """
        return np.vstack([-self.B, self.B])

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian [-B; B].

        Forms B v once and returns [-B v, B v].

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated; unused.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length 2 q.

        """
        Bv = self.B @ v
        return np.concatenate([-Bv, Bv])

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian.

        Computes
           [-B; B]^T u = B^T (u_up - u_lo),
        where u_lo holds the first q entries of u and u_up the last q, with one product
        by B^T.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated; unused.
         u : npt.NDArray[np.float64]
            Vector of length 2 q to multiply.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        q = self.q
        return np.asarray(self.B.T @ (u[q:] - u[:q]))

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the imbalance rows' barrier curvature as a low-rank update.

        The layer stands in for a handful of two-sided rows, far fewer than the
        dimension, so its weighted curvature
           ∑ (-λ_i / f_i) ∇f_i ∇f_i^T
        is a genuine width-2q low-rank update on a zero base (see
        ``_low_rank_barrier_contribution``).

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length 2 q, in the layer's constraint order.
            Required; the ``None`` default mirrors
            ``ConstraintLayer.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` carrying the single low-rank update.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``. Propagated from
            ``_low_rank_barrier_contribution``.

        """
        return self._low_rank_barrier_contribution(w, lmbda, shift)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return zero curvature, since the imbalance rows are affine (∇^2 f_i = 0).

        The low-rank structure in ``hessian_contribution`` is the barrier Jacobian
        outer-product term -(1 / f_i) ∇f_i ∇f_i^T, not constraint curvature.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; only its length is used.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length 2 q; unused.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` of order M.

        """
        return ZeroSystem(len(w))

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps every imbalance row strictly satisfied.

        Finds the largest α such that, componentwise,
           |B (w + α dw) - c| < ψ + shift + α ds.
        With r = B w - c, the upper side requires
           α (B dw - ds) < ψ + shift - r,
        and the lower side requires
           α (-B dw - ds) < ψ + shift + r.
        Each side binds only on rows where the coefficient of α is positive; the step is
        the smallest closed-form ratio over those rows, so ``line_search`` is ignored.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible for the shifted constraints, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Unused; accepted for signature compatibility.

        Returns
        -------
         alpha : float
            The largest feasible step; inf when no row binds.

        """
        Bw_minus_c = self.B @ w - self.c
        B_dw = self.B @ dw

        # Upper side (second block): (Bw - c) + alpha B dw < psi + shift + alpha ds.
        denom_pos = B_dw - ds
        mask_pos = denom_pos > 0
        theta_pos = np.inf
        if np.any(mask_pos):
            theta_pos = float(
                np.min(
                    (self._psi[mask_pos] + shift - Bw_minus_c[mask_pos])
                    / denom_pos[mask_pos]
                )
            )

        # Lower side (first block): -((Bw - c) + alpha B dw) < psi + shift + alpha ds.
        denom_neg = -B_dw - ds
        mask_neg = denom_neg > 0
        theta_neg = np.inf
        if np.any(mask_neg):
            theta_neg = float(
                np.min(
                    (self._psi[mask_neg] + shift + Bw_minus_c[mask_neg])
                    / denom_neg[mask_neg]
                )
            )

        return min(theta_pos, theta_neg)

    def is_satisfied(
        self,
        w: npt.NDArray[np.float64],
    ) -> bool:
        """Check whether |(B w - c)_i| < ψ_i holds strictly on every row.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         satisfied : bool
            True when every row holds strictly.

        """
        return bool(np.all(np.abs(self.B @ w - self.c) < self._psi))


class LinearInequalityConstraintLayer(SlackConstraintLayer):
    """Two-sided linear inequalities cl ≤ A w ≤ cu.

    Either side of a row may be omitted by passing ±inf in the corresponding entry of
    ``cl`` or ``cu``. The constraint vector stacks the active sides, upper rows first:
       upper rows (finite cu_i):   f(w) = (A w)_i - cu_i ≤ 0,
       lower rows (finite cl_i):   f(w) = cl_i - (A w)_i ≤ 0.
    A genuinely two-sided row contributes one constraint to each block, so the layer has
    ``num_cu`` + ``num_cl`` constraints. The barrier Hessian contribution is a low-rank
    update of that width, suited to few rows relative to M.

    Parameters
    ----------
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        Constraint matrix, of shape (m, M), dense or sparse. Must be two-dimensional, or
        the constructor raises ``ValueError``.
     cl : npt.NDArray[np.float64] | None
        Lower bound on each row, of length m; -inf disables a row's lower side, and
        ``None`` disables every lower side. The constructor raises ``ValueError`` when
        ``cl`` or ``cu`` does not have length m, when some row has both sides infinite
        (it would contribute no constraint, so passing ``None`` for both is rejected),
        or when some row has cl_i ≥ cu_i (an empty strict interior).
     cu : npt.NDArray[np.float64] | None
        Upper bound on each row, of length m; +inf disables a row's upper side, and
        ``None`` disables every upper side.

    Notes
    -----
    A may be dense or any scipy sparse matrix. The matrix-vector products in
    ``constraints``, ``grad_constraints_multiply``,
    ``grad_constraints_transpose_multiply``, ``btls_max_step`` and ``is_satisfied``
    apply A with the @ operator, so a sparse A stays sparse there. ``grad_constraints``
    densifies A, and ``hessian_contribution`` calls it (through
    ``_low_rank_barrier_contribution``) to form the dense low-rank factor G^T, of shape
    (M, ``num_cu`` + ``num_cl``), at every Newton step. A sparse A therefore saves work
    in the products but not in the Hessian assembly; ``BandedLinearInequalityLayer`` and
    ``ArrowLinearInequalityLayer`` assemble their Hessians without densifying A.

    """

    def __init__(
        self,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
        cl: npt.NDArray[np.float64] | None = None,
        cu: npt.NDArray[np.float64] | None = None,
    ) -> None:
        A_any: Any = A
        if A_any.ndim != 2:
            raise ValueError(f"`A` must be 2-D; got shape {A_any.shape}.")
        m = A_any.shape[0]
        cl_arr = (
            np.full(m, -np.inf, dtype=np.float64)
            if cl is None
            else np.asarray(cl, dtype=np.float64)
        )
        cu_arr = (
            np.full(m, np.inf, dtype=np.float64)
            if cu is None
            else np.asarray(cu, dtype=np.float64)
        )
        if cl_arr.shape != (m,):
            raise ValueError(
                f"`cl` shape {cl_arr.shape} does not match `A` rows ({m},)."
            )
        if cu_arr.shape != (m,):
            raise ValueError(
                f"`cu` shape {cu_arr.shape} does not match `A` rows ({m},)."
            )

        cu_finite_mask = np.isfinite(cu_arr)
        cl_finite_mask = np.isfinite(cl_arr)
        free_rows = ~(cu_finite_mask | cl_finite_mask)
        if np.any(free_rows):
            idx = int(np.argmax(free_rows))
            raise ValueError(
                f"Row {idx} of `A` has both `cl = -inf` and `cu = +inf`; "
                "it contributes no constraint and must be removed by the caller."
            )
        if not (np.any(cu_finite_mask) or np.any(cl_finite_mask)):
            raise ValueError(
                "LinearInequalityConstraintLayer requires at least one finite bound."
            )

        both_finite = cu_finite_mask & cl_finite_mask
        if np.any(both_finite) and not np.all(
            cl_arr[both_finite] < cu_arr[both_finite]
        ):
            bad = both_finite & ~(cl_arr < cu_arr)
            idx = int(np.argmax(bad))
            raise ValueError(
                f"Row {idx} has cl = {cl_arr[idx]} >= cu = {cu_arr[idx]}; "
                "the strict-feasibility interior is empty."
            )

        self._A: Any = A_any
        self._is_sparse: bool = scipy.sparse.issparse(A_any)
        self._dimension: int = int(A_any.shape[1])
        self._cu_idx: npt.NDArray[np.intp] = np.flatnonzero(cu_finite_mask)
        self._cl_idx: npt.NDArray[np.intp] = np.flatnonzero(cl_finite_mask)
        self._cu_finite: npt.NDArray[np.float64] = cu_arr[self._cu_idx]
        self._cl_finite: npt.NDArray[np.float64] = cl_arr[self._cl_idx]

    @property
    def dimension(self) -> int:
        """Return the decision-variable dimension M, the column count of A.

        Returns
        -------
         M : int
            The decision-variable dimension.

        """
        return self._dimension

    @property
    def num_cu(self) -> int:
        """Return the number of active upper-bound rows (finite entries of ``cu``).

        Returns
        -------
         num_cu : int
            Number of rows with a finite upper bound.

        """
        return self._cu_idx.size

    @property
    def num_cl(self) -> int:
        """Return the number of active lower-bound rows (finite entries of ``cl``).

        Returns
        -------
         num_cl : int
            Number of rows with a finite lower bound.

        """
        return self._cl_idx.size

    def num_constraints(self, M: int) -> int:
        """Count the active constraints across both sides.

        Parameters
        ----------
         M : int
            Dimension of the decision variable; must equal the column count of A.

        Returns
        -------
         m : int
            ``num_cu`` + ``num_cl``.

        Raises
        ------
         ValueError
            When M differs from the layer's dimension.

        """
        if self._dimension != M:
            raise ValueError(
                f"Dimension mismatch: layer expects {self._dimension}, got {M}."
            )
        return self.num_cu + self.num_cl

    def _Ax(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute A w as a dense one-dimensional array, whether A is dense or sparse.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Aw : npt.NDArray[np.float64]
            The product A w, of length m.

        """
        return np.asarray(self._A @ w).ravel()

    def constraints(
        self,
        w: npt.NDArray[np.float64],
        shift: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the stacked row constraints, shifted by ``shift``.

        Returns the upper rows followed by the lower rows,
           (A w)_i - cu_i - shift,   cl_i - (A w)_i - shift,
        over the rows with a finite bound on that side, from one product A w.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the constraints are evaluated, of length M.
         shift : float
            Relaxation subtracted from every constraint value.

        Returns
        -------
         c : npt.NDArray[np.float64]
            The shifted constraint values, of length ``num_cu`` + ``num_cl``.

        """
        Ax = self._Ax(w)
        cu_block = Ax[self._cu_idx] - self._cu_finite - shift
        cl_block = self._cl_finite - Ax[self._cl_idx] - shift
        return np.concatenate([cu_block, cl_block])

    def grad_constraints(
        self,
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Form the dense constraint Jacobian G = [A_cu; -A_cl].

        A_cu holds the rows of A with a finite upper bound and A_cl those with a finite
        lower bound. A sparse A is densified, so avoid this in hot loops;
        ``hessian_contribution`` nevertheless calls it once per Newton step.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated; unused, since the layer is affine.

        Returns
        -------
         G : npt.NDArray[np.float64]
            The constraint Jacobian, of shape (``num_cu`` + ``num_cl``, M).

        """
        if self._is_sparse:
            A_dense = np.asarray(self._A.toarray(), dtype=np.float64)
        else:
            A_dense = np.asarray(self._A, dtype=np.float64)
        return np.vstack([A_dense[self._cu_idx], -A_dense[self._cl_idx]])

    def grad_constraints_multiply(
        self,
        w: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the constraint Jacobian, with one product A v.

        Returns (A v) over the upper rows followed by -(A v) over the lower rows.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated; unused.
         v : npt.NDArray[np.float64]
            Vector of length M to multiply.

        Returns
        -------
         Gv : npt.NDArray[np.float64]
            The product G v, of length ``num_cu`` + ``num_cl``.

        """
        Av = np.asarray(self._A @ v).ravel()
        return np.concatenate([Av[self._cu_idx], -Av[self._cl_idx]])

    def grad_constraints_transpose_multiply(
        self,
        w: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed constraint Jacobian.

        Scatters u to the row space (adding the upper-row entries and subtracting the
        lower-row entries, so a two-sided row receives their difference), then applies
        A^T once.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point at which the Jacobian is evaluated; unused.
         u : npt.NDArray[np.float64]
            Vector of length ``num_cu`` + ``num_cl`` to multiply, in the layer's
            constraint order.

        Returns
        -------
         GTu : npt.NDArray[np.float64]
            The product G^T u, of length M.

        """
        n_cu = self.num_cu
        scattered = np.zeros(self._A.shape[0], dtype=np.float64)
        if n_cu > 0:
            scattered[self._cu_idx] += u[:n_cu]
        if self.num_cl > 0:
            scattered[self._cl_idx] -= u[n_cu:]
        return np.asarray(self._A.T @ scattered).ravel()

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Return the linear-inequality barrier curvature as a low-rank update.

        The layer assumes few rows relative to the dimension, so its weighted curvature
           ∑ (-λ_i / f_i) ∇f_i ∇f_i^T
        is a genuine low-rank update, of width ``num_cu`` + ``num_cl``, on a zero base
        (see ``_low_rank_barrier_contribution``). The low-rank factor is the dense G^T,
        formed by ``grad_constraints`` even when A is sparse.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, in the layer's constraint order. Required; the
            ``None`` default mirrors ``ConstraintLayer.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` carrying the single low-rank update.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``. Propagated from
            ``_low_rank_barrier_contribution``.

        """
        return self._low_rank_barrier_contribution(w, lmbda, shift)

    def hessian_curvature(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
    ) -> MatrixSystem:
        """Return zero curvature, since the rows are affine (∇^2 f_i = 0).

        The low-rank structure in ``hessian_contribution`` is the barrier Jacobian
        outer-product term, not constraint curvature.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M; only its length is used.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ; unused.

        Returns
        -------
         system : MatrixSystem
            A ``ZeroSystem`` of order M.

        """
        return ZeroSystem(len(w))

    def btls_max_step(
        self,
        w: npt.NDArray[np.float64],
        dw: npt.NDArray[np.float64],
        shift: float = 0.0,
        ds: float = 0.0,
        line_search: LineSearchSettings | None = None,
    ) -> float:
        """Find the largest step that keeps every active row strictly satisfied.

        Finds the largest α such that, on the active rows,
           cl - shift - α ds < A (w + α dw) < cu + shift + α ds.
        The upper side binds only where (A dw)_i - ds > 0 and the lower side only where
        -(A dw)_i - ds > 0; the step is the smallest closed-form ratio over those rows,
        so ``line_search`` is ignored.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, strictly feasible for the shifted constraints, of length M.
         dw : npt.NDArray[np.float64]
            Step direction, of length M.
         shift : float
            Relaxation at α = 0.
         ds : float
            Rate at which the relaxation grows with α.
         line_search : LineSearchSettings | None
            Unused; accepted for signature compatibility.

        Returns
        -------
         alpha : float
            The largest feasible step; inf when no row binds.

        """
        Ax = self._Ax(w)
        Adw = self._Ax(dw)
        alpha = np.inf
        if self.num_cu > 0:
            denom_cu = Adw[self._cu_idx] - ds
            mask = denom_cu > 0
            if np.any(mask):
                slack = self._cu_finite[mask] + shift - Ax[self._cu_idx][mask]
                alpha = min(alpha, float(np.min(slack / denom_cu[mask])))
        if self.num_cl > 0:
            denom_cl = -Adw[self._cl_idx] - ds
            mask = denom_cl > 0
            if np.any(mask):
                slack = Ax[self._cl_idx][mask] - self._cl_finite[mask] + shift
                alpha = min(alpha, float(np.min(slack / denom_cl[mask])))
        return alpha

    def is_satisfied(
        self,
        w: npt.NDArray[np.float64],
    ) -> bool:
        """Check whether cl < A w < cu holds strictly on the active rows.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Point to test, of length M.

        Returns
        -------
         satisfied : bool
            True when every finite row bound holds strictly.

        """
        Ax = self._Ax(w)
        if self.num_cu > 0 and not bool(np.all(Ax[self._cu_idx] < self._cu_finite)):
            return False
        return self.num_cl == 0 or bool(np.all(Ax[self._cl_idx] > self._cl_finite))


def _row_pairs(
    A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]:
    """Enumerate the upper-triangular column pairs of each row of A.

    The barrier Hessian A^T diag(d) A has entry (a, b) equal to
       ∑_r d_r A_ra A_rb.
    Every such entry comes from a pair of nonzeros sharing a row. This function
    enumerates those pairs once, at construction: each pair of columns a ≤ b within a
    row, with its product A_ra A_rb. The per-step assembly then only scales these by the
    current row weights and scatters them into a structured system, with no dense A^T D
    A product.

    Parameters
    ----------
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        Constraint matrix, of shape (m, n), dense or sparse; it is converted to CSR.

    Returns
    -------
     rows : npt.NDArray[np.int64]
        Row index r of each pair.
     col_a : npt.NDArray[np.int64]
        Lower column index a of each pair.
     col_b : npt.NDArray[np.int64]
        Upper column index b of each pair, with b ≥ a.
     prod : npt.NDArray[np.float64]
        The product A_ra A_rb of each pair.

    Notes
    -----
    A row with k_r stored entries yields k_r (k_r + 1) / 2 pairs, including its k_r
    diagonal pairs, and the enumeration runs in Python loops, so construction costs
       O(∑ k_r^2).
    That is cheap for short rows (three pairs for a difference row with two nonzeros)
    but slow for dense rows.

    """
    Acsr = scipy.sparse.csr_matrix(A)
    indptr = Acsr.indptr
    indices = Acsr.indices
    data = Acsr.data
    rows: list[int] = []
    col_a: list[int] = []
    col_b: list[int] = []
    prod: list[float] = []
    for r in range(Acsr.shape[0]):
        start, end = int(indptr[r]), int(indptr[r + 1])
        cols = indices[start:end]
        vals = data[start:end]
        order = np.argsort(cols)
        cols = cols[order]
        vals = vals[order]
        for i in range(cols.size):
            for j in range(i, cols.size):
                rows.append(r)
                col_a.append(int(cols[i]))
                col_b.append(int(cols[j]))
                prod.append(float(vals[i] * vals[j]))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(col_a, dtype=np.int64),
        np.asarray(col_b, dtype=np.int64),
        np.asarray(prod, dtype=np.float64),
    )


class BandedLinearInequalityLayer(LinearInequalityConstraintLayer):
    """One-sided linear inequalities A w ≤ cu whose barrier Hessian is banded.

    A drop-in ``LinearInequalityConstraintLayer`` for problems whose columns are ordered
    so that the barrier Hessian
       A^T diag(d) A,   d_r = -λ_r / f_r,
    is banded: for example, difference inequalities w_a - w_b ≤ cu_r with the variables
    sorted so each row's two nonzeros sit close together. The base class assembles that
    term as a width-m low-rank (Woodbury) update, whose m × m capacitance matrix costs
    O(m^3) to factor; this layer instead assembles a lower-banded ``BandedSystem``,
    which banded Cholesky factors in O(M L^2), where L is the bandwidth.

    Only the upper side is supported; pass difference-structured rows so the Hessian is
    genuinely banded. The bandwidth is found from A at construction, as the largest
    column span within a row. Constraint values, gradients and the step bound are
    inherited unchanged.

    Parameters
    ----------
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        Constraint matrix, of shape (m, M), dense or sparse, with columns ordered so
        that A^T diag(d) A is banded.
     cu : npt.NDArray[np.float64]
        Upper bound on each row, of length m. Every entry must be finite, or the
        constructor raises ``ValueError`` (a row with an infinite bound and no lower
        side contributes no constraint).
     ridge_scale : float
        Relative ridge, multiplied by the largest row weight d_r and added to the
        Hessian diagonal. A difference-constraint Hessian is a graph Laplacian, singular
        on the constant vector; when the rest of the problem adds only negligible
        diagonal (a generous box, say), banded Cholesky sees a matrix that is not
        positive definite and fails. The ridge keeps it strictly positive definite.
     ridge_floor : float
        Absolute ridge added on top of the relative one.

    Notes
    -----
    At construction, ``_row_pairs`` enumerates the P column pairs of the rows and maps
    each to a fixed slot in the (L + 1) × M band storage. Each Newton step scales the
    pair products by the row weights and accumulates them with one np.bincount, costing
    O(P + L M), before the O(M L^2) banded factorization.

    """

    def __init__(
        self,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
        cu: npt.NDArray[np.float64],
        *,
        ridge_scale: float = 1e-8,
        ridge_floor: float = 1e-12,
    ) -> None:
        super().__init__(A, cl=None, cu=cu)
        A_any: Any = A
        self._m: int = int(A_any.shape[0])
        self._ridge_scale: float = float(ridge_scale)
        self._ridge_floor: float = float(ridge_floor)
        rows, col_a, col_b, prod = _row_pairs(A)
        self._pair_row: npt.NDArray[np.int64] = rows
        self._pair_lo: npt.NDArray[np.int64] = col_a
        self._pair_offset: npt.NDArray[np.int64] = col_b - col_a
        self._pair_prod: npt.NDArray[np.float64] = prod
        self._bandwidth: int = (
            int(self._pair_offset.max()) if self._pair_offset.size else 0
        )
        # Precompute the flat scatter indices into the (bandwidth+1, n) band
        # storage `ab`; they are structurally constant, so each Newton step only
        # rescales by the weights and accumulates with a buffered np.bincount
        # (an unbuffered np.add.at here was ~40% of the step -- bincount is ~10x).
        self._band_flat: npt.NDArray[np.int64] = (
            self._pair_offset * self._dimension + self._pair_lo
        ).astype(np.int64)

    def _row_weights(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        shift: float,
    ) -> npt.NDArray[np.float64]:
        """Compute the per-row barrier weights d_r = -λ_r / f_r.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length m.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         d : npt.NDArray[np.float64]
            The weight of each row of A, of length m.

        """
        wgt = -lmbda / self.constraints(w, shift)
        row_weight = np.zeros(self._m, dtype=np.float64)
        row_weight[self._cu_idx] = wgt
        return row_weight

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Assemble A^T diag(-λ / f) A, plus a ridge, as a lower-banded system.

        Scales the precomputed pair products by the row weights d_r = -λ_r / f_r and
        scatters them into lower band storage, then adds
           ridge = ``ridge_scale`` max_r d_r + ``ridge_floor``
        to the diagonal. Since the layer is affine, this is the whole contribution apart
        from the ridge.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length M.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m. Required; the ``None`` default
            mirrors ``ConstraintLayer.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            A ``BandedSystem`` of order M and bandwidth L, in lower storage.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``.

        """
        if lmbda is None:
            raise ValueError("BandedLinearInequalityLayer requires `lmbda`.")
        row_weight = self._row_weights(w, lmbda, shift)
        contribution = row_weight[self._pair_row] * self._pair_prod
        ab = (
            np.bincount(
                self._band_flat,
                weights=contribution,
                minlength=(self._bandwidth + 1) * self._dimension,
            )
            .reshape(self._bandwidth + 1, self._dimension)
            .astype(np.float64, copy=False)
        )  # float weights, so a float64 no-op
        ridge = self._ridge_scale * float(np.max(row_weight)) + self._ridge_floor
        ab[0] += ridge
        return BandedSystem(ab, lower=True)


class ArrowLinearInequalityLayer(LinearInequalityConstraintLayer):
    """One-sided linear inequalities with a banded-plus-border barrier Hessian.

    Like ``BandedLinearInequalityLayer``, but the columns of A split in two: the first M
    (``core_dimension``) form a banded core, and the trailing K couple densely to it,
    for n = M + K columns in all. The barrier Hessian A^T diag(d) A then has the
    bordered (arrow) structure
            _         _
       H = |  C     B  |
           |_ B^T   D _|
    with C banded of order M, B a dense M × K border, and D a dense K × K corner. This
    arises when K shared variables enter every row of an otherwise difference-structured
    system. The Hessian is returned as an ``ArrowSystem``, whose Schur-complement solve
    touches the core only through its banded factorization.

    A small relative ridge is added to the core diagonal and the corner diagonal: a
    difference-constraint core is a graph Laplacian, singular on the constant vector,
    and the arrow's K × K Schur complement is factored by Cholesky with no
    positive-definite fallback, so the ridge keeps both blocks numerically positive
    definite.

    Parameters
    ----------
     A : npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray
        Constraint matrix, of shape (m, n), dense or sparse. Its first
        ``core_dimension`` columns are the banded core and its trailing K columns the
        border.
     cu : npt.NDArray[np.float64]
        Upper bound on each row, of length m. Every entry must be finite, or the
        constructor raises ``ValueError``.
     core_dimension : int
        Order M of the banded core. Must satisfy 0 < M < n, or the constructor raises
        ``ValueError``.
     ridge_scale : float
        Relative ridge, multiplied by the largest row weight d_r and added to the core
        and corner diagonals.
     ridge_floor : float
        Absolute ridge added on top of the relative one.

    """

    def __init__(
        self,
        A: npt.NDArray[np.float64] | scipy.sparse.spmatrix | scipy.sparse.sparray,
        cu: npt.NDArray[np.float64],
        *,
        core_dimension: int,
        ridge_scale: float = 1e-8,
        ridge_floor: float = 1e-12,
    ) -> None:
        super().__init__(A, cl=None, cu=cu)
        n = self._dimension
        if not 0 < core_dimension < n:
            raise ValueError(
                f"core_dimension must satisfy 0 < M < n; got M={core_dimension}, n={n}."
            )
        A_any: Any = A
        self._m: int = int(A_any.shape[0])
        self._M: int = int(core_dimension)
        self._K: int = n - int(core_dimension)
        self._ridge_scale: float = float(ridge_scale)
        self._ridge_floor: float = float(ridge_floor)

        rows, col_a, col_b, prod = _row_pairs(A)
        M = self._M
        in_core_a = col_a < M
        in_core_b = col_b < M
        core = in_core_a & in_core_b
        border = in_core_a & ~in_core_b
        corner = ~in_core_a & ~in_core_b

        self._core_row: npt.NDArray[np.int64] = rows[core]
        self._core_lo: npt.NDArray[np.int64] = col_a[core]
        self._core_offset: npt.NDArray[np.int64] = col_b[core] - col_a[core]
        self._core_prod: npt.NDArray[np.float64] = prod[core]
        self._bandwidth: int = (
            int(self._core_offset.max()) if self._core_offset.size else 0
        )
        # Border entries couple a core column to a trailing (border) column.
        self._border_row: npt.NDArray[np.int64] = rows[border]
        self._border_core: npt.NDArray[np.int64] = col_a[border]
        self._border_k: npt.NDArray[np.int64] = col_b[border] - M
        self._border_prod: npt.NDArray[np.float64] = prod[border]
        # Corner entries couple two trailing columns (stored k1 <= k2).
        self._corner_row: npt.NDArray[np.int64] = rows[corner]
        self._corner_k1: npt.NDArray[np.int64] = col_a[corner] - M
        self._corner_k2: npt.NDArray[np.int64] = col_b[corner] - M
        self._corner_prod: npt.NDArray[np.float64] = prod[corner]

        # Precompute the flat scatter indices for each block; they are
        # structurally constant, so each Newton step only rescales by the weights
        # and accumulates with a buffered np.bincount (an unbuffered np.add.at was
        # a large share of the step -- bincount is ~10x on the O(edges) blocks).
        K = self._K
        self._core_flat: npt.NDArray[np.int64] = (
            self._core_offset * M + self._core_lo
        ).astype(np.int64)
        self._border_flat: npt.NDArray[np.int64] = (
            self._border_core * K + self._border_k
        ).astype(np.int64)
        self._corner_flat: npt.NDArray[np.int64] = (
            self._corner_k1 * K + self._corner_k2
        ).astype(np.int64)

    def _row_weights(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64],
        shift: float,
    ) -> npt.NDArray[np.float64]:
        """Compute the per-row barrier weights d_r = -λ_r / f_r.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length n.
         lmbda : npt.NDArray[np.float64]
            Per-constraint weights λ, of length m.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         d : npt.NDArray[np.float64]
            The weight of each row of A, of length m.

        """
        wgt = -lmbda / self.constraints(w, shift)
        row_weight = np.zeros(self._m, dtype=np.float64)
        row_weight[self._cu_idx] = wgt
        return row_weight

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        """Assemble the barrier Hessian as a banded core, dense border, and corner.

        Sorts the precomputed column pairs (a ≤ b) of the barrier Hessian A^T diag(d) A,
        with row weights
           d_r = -λ_r / f_r,
        by where they land: both columns in the core go to the lower-banded core, a core
        column paired with a border column goes to the M × K border, and two border
        columns go to the K × K corner (accumulated in its upper triangle, then
        symmetrized). Then adds
           ridge = ``ridge_scale`` max_r d_r + ``ridge_floor``
        to the core and corner diagonals.

        Parameters
        ----------
         w : npt.NDArray[np.float64]
            Current iterate, of length n.
         lmbda : npt.NDArray[np.float64] | None
            Per-constraint weights λ, of length m. Required; the ``None`` default
            mirrors ``ConstraintLayer.hessian_contribution``.
         shift : float
            Relaxation applied to the constraint values, as in ``constraints``.

        Returns
        -------
         system : MatrixSystem
            An ``ArrowSystem`` of order n, with a ``BandedSystem`` core of order M.

        Raises
        ------
         ValueError
            When ``lmbda`` is ``None``.

        """
        if lmbda is None:
            raise ValueError("ArrowLinearInequalityLayer requires `lmbda`.")
        row_weight = self._row_weights(w, lmbda, shift)
        M, K = self._M, self._K
        ridge = self._ridge_scale * float(np.max(row_weight)) + self._ridge_floor

        ab = (
            np.bincount(
                self._core_flat,
                weights=row_weight[self._core_row] * self._core_prod,
                minlength=(self._bandwidth + 1) * M,
            )
            .reshape(self._bandwidth + 1, M)
            .astype(np.float64, copy=False)
        )  # float weights, so a float64 no-op
        ab[0] += ridge
        core = BandedSystem(ab, lower=True)

        border = (
            np.bincount(
                self._border_flat,
                weights=row_weight[self._border_row] * self._border_prod,
                minlength=M * K,
            )
            .reshape(M, K)
            .astype(np.float64, copy=False)
        )  # float weights, so a float64 no-op

        corner_upper = (
            np.bincount(
                self._corner_flat,
                weights=row_weight[self._corner_row] * self._corner_prod,
                minlength=K * K,
            )
            .reshape(K, K)
            .astype(np.float64, copy=False)
        )  # float weights, so a float64 no-op
        corner = corner_upper + corner_upper.T
        np.fill_diagonal(corner, np.diagonal(corner_upper))  # undo double-count
        corner[np.diag_indices(K)] += ridge
        return ArrowSystem(core, border=border, corner=corner)
