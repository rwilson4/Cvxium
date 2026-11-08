"""Second-order cone programs, solved by Cvxium's homogeneous self-dual solver.

Provides ``solve_second_order_cone_program`` for the problem
   minimize    f^T x
   subject to  ‖A_soc x + b_soc‖ ≤ c_soc^T x + d_soc,
               F x = g_eq,
               x ≥ 0.
``_solve_socp_via_hsd`` rewrites it in canonical conic form over the product cone
Nonneg(n) × SOC(m + 1), adding a slack block (t, u), and dispatches to ``HsdSolver``.
Two specializations keep each iteration cheap. When A_soc = [diag(d) | 0] (detected by
``_A_soc_diagonal_zero_padded``, or supplied as a hint),
``_make_socp_structured_embedding_class`` builds an embedding whose residuals and KKT
solve exploit that structure, at a cost linear in n for a fixed number p of equality
constraints. Otherwise the generic embedding runs, with the slack cone from
``_make_slack_soc_cone``, whose normal-equations contribution has a closed form.

"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..problems.linear_constraints import (
    EqualityConstraintLayer,
)
from ..problems.result import (
    Duals,
    ObjectiveOutcome,
    SolveResult,
)
from ..solvers.hsd_solver import _HsdEmbedding
from ..solvers.settings import (
    OptimizationSettings,
)

if TYPE_CHECKING:
    from ..problems.cone import SecondOrderCone


def solve_second_order_cone_program(
    f: npt.NDArray[np.float64],
    A_soc: npt.NDArray[np.float64],
    b_soc: npt.NDArray[np.float64],
    c_soc: npt.NDArray[np.float64],
    d_soc: float,
    F: npt.NDArray[np.float64],
    g_eq: npt.NDArray[np.float64],
    settings: OptimizationSettings | None = None,
    A_soc_diag: npt.NDArray[np.float64] | None = None,
) -> SolveResult:
    """Solve a second-order cone program with bound and equality constraints.

    Solves
       minimize    f^T x
       subject to  ‖A_soc x + b_soc‖ ≤ c_soc^T x + d_soc,
                   F x = g_eq,
                   x ≥ 0,
    with the homogeneous self-dual solver ``HsdSolver`` (Andersen, Roos, and Terlaky,
    2003). The problem is first written in the canonical conic form
       minimize c^T z   subject to   A z = b,   z ∈ K = Nonneg(n) × SOC(m + 1),
    by adding the slack variables
       t = c_soc^T x + d_soc,   u = A_soc x + b_soc,
    which form the second-order cone block (t, u), and the m + 1 equality constraints
    that define them. ``HsdSolver`` needs no strictly feasible starting point, and it
    reports infeasibility or unboundedness through the result's status and certificate
    rather than by raising.

    Parameters
    ----------
     f : npt.NDArray[np.float64]
        Linear objective vector, of length n.
     A_soc : npt.NDArray[np.float64]
        Matrix in the cone constraint, of shape (m, n).
     b_soc : npt.NDArray[np.float64]
        Vector in the cone constraint, of length m.
     c_soc : npt.NDArray[np.float64]
        Linear coefficient in the cone constraint, of length n.
     d_soc : float
        Scalar in the cone constraint.
     F : npt.NDArray[np.float64]
        Equality constraint matrix, of shape (p, n). An empty array means no equality
        constraints.
     g_eq : npt.NDArray[np.float64]
        Equality constraint right-hand side, of length p.
     settings : OptimizationSettings | None
        Solver settings; ``None`` uses the defaults.
     A_soc_diag : npt.NDArray[np.float64] | None
        The diagonal of A_soc, of length m, given as a hint that A_soc is diagonal in
        its first m columns and zero in the rest, which requires m to be at most n. When
        given, the solver skips the O(m n) structure check
        (``_A_soc_diagonal_zero_padded``) and uses the structured embedding directly,
        reading nothing from ``A_soc`` but its row count. The hint is trusted, not
        checked: a wrong hint gives a wrong answer.

    Returns
    -------
     result : SolveResult
        The ``HsdSolver`` result mapped back to the original variables. ``solution`` is
        the original variable x, with the slack variables stripped, and the objective
        value is f^T x. ``equality_multipliers`` holds the multipliers of the p equality
        constraints, or ``None`` when there are none. ``inequality_multipliers`` holds
        the n multipliers of the bounds on x, followed by the leading entry of the dual
        vector of the cone block. The status, message, iteration count, diagnostics, and
        certificate come from ``HsdSolver`` unchanged, so a certificate's witness lives
        in the embedding's coordinates rather than those of x.

    Raises
    ------
     ValueError
        When f is empty or A_soc has no rows, propagated from the ``Nonneg`` and
        ``SecondOrderCone`` constructors.

    References
    ----------
    Andersen, Erling D., Roos, Cornelis and Terlaky, Tamás, On Implementing a
    Primal-Dual Interior-Point Method for Conic Quadratic Optimization, Mathematical
    Programming 95, 2003.

    """
    if settings is None:
        settings = OptimizationSettings()
    return _solve_socp_via_hsd(
        f=f,
        A_soc=A_soc,
        b_soc=b_soc,
        c_soc=c_soc,
        d_soc=d_soc,
        F=F,
        g_eq=g_eq,
        settings=settings,
        A_soc_diag_hint=A_soc_diag,
    )


def _make_slack_soc_cone(m: int) -> SecondOrderCone:
    """Build the second-order cone for the slack block of the generic embedding.

    Returns an instance of a private ``SecondOrderCone`` subclass whose
    ``accumulate_normal_equations`` exploits the fixed structure of the slack columns.
    In the stacked equality system built by ``_solve_socp_via_hsd``, the columns of the
    (t, u) block are zero in the first p rows and the identity in the trailing m + 1
    rows, so the block's contribution A_b W^2 A_b^T to the normal-equations matrix is
    W^2 itself, placed in the trailing (m + 1) × (m + 1) sub-block. Every other cone
    method is inherited unchanged.

    The override matters only on the generic path, taken when A_soc is not diagonal and
    zero-padded, where ``_HsdEmbedding.kkt_solve`` calls
    ``accumulate_normal_equations``. The structured embedding from
    ``_make_socp_structured_embedding_class`` overrides ``kkt_solve`` and never calls
    it.

    Parameters
    ----------
     m : int
        Number of rows of A_soc; the cone has dimension m + 1.

    Returns
    -------
     cone : SecondOrderCone
        The specialized cone, of dimension m + 1.

    Raises
    ------
     ValueError
        When m < 1, propagated from the ``SecondOrderCone`` constructor.

    Notes
    -----
    Let p_full = p + 1 + m be the number of rows of the stacked equality system. The
    inherited ``SecondOrderCone.accumulate_normal_equations`` costs O(p_full^2 m) for
    this block. The closed form costs O(m^2) to build W^2 and O(p_full^2) to zero-fill
    the dense p_full × p_full block it returns, a saving of order m.

    """
    from ..problems.cone import (
        SecondOrderCone,
        SocNtFactors,
    )

    class _SlackSoc(SecondOrderCone):
        """Slack-block second-order cone with closed-form normal equations.

        Parameters
        ----------
         dim : int
            Dimension of the cone, m + 1.

        """

        def accumulate_normal_equations(
            self,
            precomputed: object,
            A_block: npt.NDArray[np.float64],
            c_block: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
            """Compute the slack block's reduced-system contributions in closed form.

            Assumes ``A_block`` holds the slack block's columns of the stacked equality
            system: zero except for an identity in its trailing m + 1 rows. Only its row
            count p_full is read. With W^2 = scale (2 y y^T - J) the local scaling, the
            contributions are
               A_b W^2 A_b^T = W^2 in the trailing block, zero elsewhere,
               A_b W^2 c_b   = W^2 c_b in the trailing m + 1 entries, zero elsewhere,
               c_b^T W^2 c_b.

            Parameters
            ----------
             precomputed : object
                The ``SocNtFactors`` returned by ``prepare_w_squared``.
             A_block : npt.NDArray[np.float64]
                The block's columns of the stacked constraint matrix, with p_full rows
                and m + 1 columns.
             c_block : npt.NDArray[np.float64]
                The block's entries of the cost vector, of length m + 1 (zero in the
                SOCP embedding).

            Returns
            -------
             M_block : npt.NDArray[np.float64]
                A_b W^2 A_b^T, of shape (p_full, p_full).
             AW2c_block : npt.NDArray[np.float64]
                A_b W^2 c_b, of length p_full.
             cW2c_block : float
                c_b^T W^2 c_b.

            Notes
            -----
            Costs O(m^2) to form W^2 and O(p_full^2) to zero-fill ``M_block``, against
            O(p_full^2 m) for the inherited method.

            """
            assert isinstance(precomputed, SocNtFactors)
            n_b = self.dimension()  # m + 1
            m_local = n_b - 1
            p_full = A_block.shape[0]
            # Slack rows live at the tail of the augmented system: row index
            # `p_orig` is `p_full - 1 - m_local`.
            p_orig = p_full - 1 - m_local

            scale, y = precomputed.scale, precomputed.y

            # Local W^2 = scale * (2 y y^T - J),  J = diag(1, -1, ..., -1).
            local = 2.0 * np.outer(y, y)
            local[0, 0] -= 1.0
            local[1:, 1:] += np.eye(m_local, dtype=np.float64)
            local *= scale

            M_block = np.zeros((p_full, p_full), dtype=np.float64)
            M_block[p_orig : p_orig + n_b, p_orig : p_orig + n_b] = local

            local_AW2c = local @ c_block
            AW2c_block = np.zeros(p_full, dtype=np.float64)
            AW2c_block[p_orig : p_orig + n_b] = local_AW2c
            cW2c_block = float(c_block @ local_AW2c)

            return M_block, AW2c_block, cW2c_block

    return _SlackSoc(dim=m + 1)


def _A_soc_diagonal_zero_padded(
    A_soc: npt.NDArray[np.float64],
    atol: float = 1e-12,
) -> npt.NDArray[np.float64] | None:
    """Return d when A_soc = [diag(d) | 0], else ``None``.

    A structural check that selects the structured embedding in ``_solve_socp_via_hsd``.
    A_soc qualifies when it has at least as many columns as rows, its leading m × m
    block is diagonal, and its remaining columns are zero.

    Parameters
    ----------
     A_soc : npt.NDArray[np.float64]
        Matrix in the cone constraint, of shape (m, n).
     atol : float
        Absolute tolerance against zero for the off-diagonal and padding entries,
        applied through ``np.allclose``. A relative tolerance would be meaningless in a
        comparison with zero.

    Returns
    -------
     d : npt.NDArray[np.float64] | None
        A copy of the diagonal of the leading m × m block, of length m, or ``None`` when
        A_soc does not have the structure.

    Notes
    -----
    Costs O(m n): O(m (n - m)) for the padding check and O(m^2) for the off-diagonal
    check.

    """
    m, n = A_soc.shape
    if n < m:
        return None
    if n > m and not np.allclose(A_soc[:, m:], 0.0, atol=atol):
        return None
    head = A_soc[:, :m]
    diag = np.diagonal(head).copy()
    off_diag_mask = ~np.eye(m, dtype=bool)
    if not np.allclose(head[off_diag_mask], 0.0, atol=atol):
        return None
    return diag.astype(np.float64, copy=False)


def _make_socp_structured_embedding_class(
    n: int,
    m: int,
    p_orig: int,
    F: npt.NDArray[np.float64],
    c_soc: npt.NDArray[np.float64],
    d_soc: float,
    A_soc_diag: npt.NDArray[np.float64],
    b_soc: npt.NDArray[np.float64],
    g_eq: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
) -> type[_HsdEmbedding]:
    """Build an HSD embedding class specialized to a diagonal, zero-padded A_soc.

    With A_soc = [diag(d) | 0], the stacked equality matrix
                 _                         _
       A_full = |   F          0     0      |
                |  -c_soc^T    1     0      |
                |_ -A_soc      0     I_m   _|
    built by ``_solve_socp_via_hsd`` is sparse and structured. The returned subclass of
    ``_HsdEmbedding`` overrides ``residuals`` and ``kkt_solve`` to exploit it. Products
    with A_full and A_full^T cost
       O(p n + n + m)   rather than the dense   O((p + m)(n + m)),
    and the KKT solve never forms the dense normal-equations matrix M or the reduced
    matrix K, each with p + 1 + m or more rows. The class is built inside this function
    so that its methods close over the problem data.

    Parameters
    ----------
     n : int
        Length of the original decision variable x.
     m : int
        Number of rows of A_soc; at most n.
     p_orig : int
        Number p of original equality constraints F x = g_eq.
     F : npt.NDArray[np.float64]
        Equality constraint matrix, of shape (p, n).
     c_soc : npt.NDArray[np.float64]
        Linear coefficient in the cone constraint, of length n.
     d_soc : float
        Scalar in the cone constraint.
     A_soc_diag : npt.NDArray[np.float64]
        The diagonal d of A_soc, of length m.
     b_soc : npt.NDArray[np.float64]
        Vector in the cone constraint, of length m.
     g_eq : npt.NDArray[np.float64]
        Equality constraint right-hand side, of length p.
     f : npt.NDArray[np.float64]
        Linear objective vector, of length n.

    Returns
    -------
     embedding_class : type[_HsdEmbedding]
        The specialized embedding class, for ``HsdSolver``'s ``embedding_class``.

    Notes
    -----
    Write w_sq = x / s for the diagonal of W^2 on the Nonneg(n) block, and (scale, y)
    for the factors of W^2 = scale (2 y y^T - J) on the second-order cone block. The
    reduced system in (Δy, Δτ) splits into a block A of size p + 2 (the entries of Δy
    for the F rows and the cone-scalar row, then Δτ) and a block B of size m (the
    entries of Δy for the u rows). The B diagonal block is diagonal plus rank one,
       K_BB = D + α u u^T,   D = diag(d^2 ∘ w_sq[:m]) + scale I,   α = 2 scale,
    where u = y_{1:}, so Sherman-Morrison applies its inverse in O(m) time. The Schur
    complement
       K_AA - K_AB K_BB^{-1} K_BA,
    of shape (p + 2) × (p + 2), is solved densely; for p = 1 it is 3 × 3.

    Per iteration this costs O(p^2 n) to form F diag(w_sq) F^T, O(p^2 m) for the Schur
    complement, and O(p^3) for its solve, so O(p^2 n + p^3) in all (using m ≤ n). For a
    fixed p the cost is linear in n.

    """
    from ..problems.cone import SecondOrderCone

    F_local = F
    c_soc_local = c_soc
    d_soc_local = float(d_soc)
    d_diag = A_soc_diag  # SOC diagonal entries
    b_soc_local = b_soc
    g_eq_local = g_eq
    f_local = f
    n_outer = p_orig + 2
    n_full = n + 1 + m
    p_full = p_orig + 1 + m

    def _A_matvec(v_full: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by the stacked equality matrix A_full.

        Uses the block structure of A_full, in O(p n + n + m) time.

        Parameters
        ----------
         v_full : npt.NDArray[np.float64]
            Vector (x, t, u) of length n + 1 + m.

        Returns
        -------
         Av : npt.NDArray[np.float64]
            The product A_full v, of length p + 1 + m.

        """
        v_x = v_full[:n]
        v_t = float(v_full[n])
        v_u = v_full[n + 1 :]
        out = np.empty(p_full, dtype=np.float64)
        if p_orig > 0:
            out[:p_orig] = F_local @ v_x
        out[p_orig] = -float(c_soc_local @ v_x) + v_t
        out[p_orig + 1 :] = -d_diag * v_x[:m] + v_u
        return out

    def _AT_matvec(y_full: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Multiply a vector by the transposed stacked equality matrix A_full^T.

        Uses the block structure of A_full, in O(p n + n + m) time.

        Parameters
        ----------
         y_full : npt.NDArray[np.float64]
            Vector of length p + 1 + m, one entry per equality constraint.

        Returns
        -------
         ATy : npt.NDArray[np.float64]
            The product A_full^T y, of length n + 1 + m.

        """
        y_top = y_full[:p_orig]
        y_mid = float(y_full[p_orig])
        y_bot = y_full[p_orig + 1 :]
        out = np.zeros(n_full, dtype=np.float64)
        if p_orig > 0:
            out[:n] = F_local.T @ y_top
        out[:n] -= c_soc_local * y_mid
        out[:m] -= d_diag * y_bot
        out[n] = y_mid
        out[n + 1 :] = y_bot
        return out

    def _apply_W_squared_full(
        v_full: npt.NDArray[np.float64],
        w_sq_nonneg: npt.NDArray[np.float64],
        scale_soc: float,
        y_soc: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Multiply a vector by the block-diagonal Nesterov-Todd matrix W^2.

        The Nonneg(n) block of W^2 is diagonal and the second-order cone block is
        factored, so with v = (v_x, v_c),
           W^2 v = (w_sq ∘ v_x,  scale (2 (y^T v_c) y - J v_c)),
        in O(n + m) time.

        Parameters
        ----------
         v_full : npt.NDArray[np.float64]
            Vector of length n + 1 + m.
         w_sq_nonneg : npt.NDArray[np.float64]
            Diagonal w_sq of W^2 on the Nonneg(n) block, of length n.
         scale_soc : float
            The factor scale of W^2 on the second-order cone block.
         y_soc : npt.NDArray[np.float64]
            The factor y of W^2 on the second-order cone block, of length m + 1.

        Returns
        -------
         W2v : npt.NDArray[np.float64]
            The product W^2 v, of length n + 1 + m.

        """
        out = np.empty(n_full, dtype=np.float64)
        out[:n] = w_sq_nonneg * v_full[:n]
        v_soc = v_full[n:]
        yv = float(y_soc @ v_soc)
        soc_out = 2.0 * yv * y_soc
        soc_out[0] -= v_soc[0]
        soc_out[1:] += v_soc[1:]
        out[n:] = scale_soc * soc_out
        return out

    class _SocpLinearHsdEmbedding(_HsdEmbedding):
        """HSD embedding of the stacked SOCP with structured residuals and KKT solve.

        Overrides ``residuals`` and ``kkt_solve`` of ``_HsdEmbedding`` and inherits
        everything else. Neither override reads the stored equality matrix, so
        ``_solve_socp_via_hsd`` passes a zero placeholder of the right shape for it.

        Parameters
        ----------
         problem : Problem
            The stacked problem: a ``LinearObjective`` and the stacked equality
            constraints.
         cones : list[Cone]
            The cones Nonneg(n) and SOC(m + 1), in that order; ``kkt_solve`` takes the
            second-order cone from position 1. Their dimensions must sum to n + 1 + m,
            or the constructor raises ``ValueError``.

        """

        def residuals(
            self,
        ) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
            npt.NDArray[np.float64],
            float,
        ]:
            """Compute the embedded residuals with structured products by A_full.

            Returns the same residuals as ``_HsdEmbedding.residuals``,
               r_p = -A x + b τ,   r_d = A^T y + s - c τ,   r_g = -b^T y + c^T x + κ,
               r_xs = x ∘ s (per cone block),   r_tk = τ κ,
            with A x and A^T y from the structured products, in O(p n + n + m) time
            rather than the dense O((p + m)(n + m)).

            Returns
            -------
             r_p : npt.NDArray[np.float64]
                Primal residual, of length p + 1 + m.
             r_d : npt.NDArray[np.float64]
                Dual residual, of length n + 1 + m.
             r_g : float
                Gap residual.
             r_xs : npt.NDArray[np.float64]
                Cone complementarity residual, of length n + 1 + m.
             r_tk : float
                Homogenizing complementarity residual τ κ.

            Raises
            ------
             RuntimeError
                When ``initial_iterate`` has not initialized the embedded state.

            """
            if self._x is None:
                raise RuntimeError(
                    "Embedded state not initialized; call initial_iterate() first."
                )
            x = self._x
            s = self._s
            y = self._y
            tau = self._tau
            kappa = self._kappa
            assert (
                s is not None
                and y is not None
                and tau is not None
                and kappa is not None
            )
            r_p = -_A_matvec(x) + self._b * tau
            r_d = _AT_matvec(y) + s - self._c * tau
            r_g = -float(self._b @ y) + float(self._c @ x) + kappa
            r_xs = np.empty_like(x)
            for cone, sl in zip(self._cones, self._cone_slices, strict=False):
                r_xs[sl] = cone.jordan_product(x[sl], s[sl])
            r_tk = tau * kappa
            return r_p, r_d, r_g, r_xs, r_tk

        def kkt_solve(
            self,
            rhs_p: npt.NDArray[np.float64],
            rhs_d: npt.NDArray[np.float64],
            rhs_g: float,
            rhs_xs: npt.NDArray[np.float64],
            rhs_tk: float,
        ) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
            npt.NDArray[np.float64],
            float,
        ]:
            """Solve the embedded Newton system through a structured Schur complement.

            Solves the same system as ``_HsdEmbedding.kkt_solve`` and returns the same
            step, but never forms the dense normal-equations matrix M or the reduced
            matrix K. The reduced system in (Δy, Δτ) splits into a block of size p + 2
            and a diagonal-plus-rank-one block of size m, inverted by Sherman-Morrison
            (see ``_make_socp_structured_embedding_class``). The (p + 2) × (p + 2) Schur
            complement is solved densely, and Δx, Δs, and Δκ follow by back-substitution
            as in the generic solve.

            Parameters
            ----------
             rhs_p : npt.NDArray[np.float64]
                Right-hand side of the primal equation, of length p + 1 + m.
             rhs_d : npt.NDArray[np.float64]
                Right-hand side of the dual equation, of length n + 1 + m.
             rhs_g : float
                Right-hand side of the gap equation.
             rhs_xs : npt.NDArray[np.float64]
                Right-hand side of the cone complementarity equation, in NT-scaled form,
                of length n + 1 + m.
             rhs_tk : float
                Right-hand side of the τ κ complementarity equation.

            Returns
            -------
             delta_x : npt.NDArray[np.float64]
                Step in x, of length n + 1 + m.
             delta_y : npt.NDArray[np.float64]
                Step in y, of length p + 1 + m.
             delta_tau : float
                Step in τ.
             delta_s : npt.NDArray[np.float64]
                Step in s, of length n + 1 + m.
             delta_kappa : float
                Step in κ.

            Raises
            ------
             ConeBoundaryError
                When the cone block of x or s is not strictly inside the second-order
                cone. Propagated from ``SecondOrderCone.factored_w_squared``.
             numpy.linalg.LinAlgError
                When the Schur complement is singular. ``HsdSolver`` catches both
                exceptions and ends its iteration loop.

            """
            x = self._x
            s = self._s
            tau = self._tau
            kappa = self._kappa
            assert (
                x is not None
                and s is not None
                and tau is not None
                and kappa is not None
            )

            # Per-iter NT / structural factors (all O(n + m)).
            w_sq_nonneg = x[:n] / s[:n]
            soc_cone = self._cones[1]
            assert isinstance(soc_cone, SecondOrderCone)
            scale_soc, y_soc = soc_cone.factored_w_squared(x[n:], s[n:])
            u_factor = y_soc[1:]  # (m,)
            alpha = 2.0 * scale_soc
            # K_BB = D + alpha * outer(u_factor, u_factor); D diagonal.
            D_diag = d_diag * d_diag * w_sq_nonneg[:m] + scale_soc  # (m,)
            D_inv_diag = 1.0 / D_diag
            D_inv_u = D_inv_diag * u_factor
            denom = 1.0 + alpha * float(u_factor @ D_inv_u)

            def K_BB_inv_vec(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                """Apply K_BB^{-1} to a vector by the Sherman-Morrison formula.

                With K_BB = D + α u u^T,
                   K_BB^{-1} v = D^{-1} v - γ D^{-1} u,
                   γ = α (u^T D^{-1} v) / (1 + α u^T D^{-1} u),
                in O(m) time.

                Parameters
                ----------
                 v : npt.NDArray[np.float64]
                    Vector of length m.

                Returns
                -------
                 K_BB_inv_v : npt.NDArray[np.float64]
                    The product K_BB^{-1} v, of length m.

                """
                coef = (alpha / denom) * float(D_inv_u @ v)
                return D_inv_diag * v - coef * D_inv_u

            # Compute Wp = W^2 (rhs_d + rhs_xs).
            p_term = rhs_d + rhs_xs
            Wp = _apply_W_squared_full(p_term, w_sq_nonneg, scale_soc, y_soc)

            # rhs_y = rhs_p - A @ Wp (structured).
            A_Wp = _A_matvec(Wp)
            rhs_y = rhs_p - A_Wp
            # rhs_tau = rhs_g + c @ Wp + rhs_tk / tau.
            #   c = f_full = [f, 0, 0]  =>  c @ Wp = f @ Wp[:n].
            rhs_tau = rhs_g + float(f_local @ Wp[:n]) + rhs_tk / tau

            # ---- Build K_AA (n_outer x n_outer) directly from state. ----
            # AW2c[top]   = F @ (w_sq[:n] * f),  shape (p_orig,)
            # AW2c[mid]   = -c_soc @ (w_sq[:n] * f) + 0 (SOC contributes 0 since
            #               c_block_soc = zeros)
            # AW2c[bot]   = -d_diag * (w_sq[:n] * f)[:m] + 0  shape (m,)
            # cW2c        = f @ (w_sq[:n] * f) + 0
            wsq_f = w_sq_nonneg * f_local  # (n,)
            AW2c_top = F_local @ wsq_f if p_orig > 0 else np.zeros(0)  # (p_orig,)
            AW2c_mid = -float(c_soc_local @ wsq_f)
            AW2c_bot = -d_diag * wsq_f[:m]  # (m,)
            cW2c = float(f_local @ wsq_f)

            # u = AW2c + b, v = b - AW2c.
            #   b = b_full = [g_eq, d_soc, b_soc]
            u_top = AW2c_top + g_eq_local if p_orig > 0 else np.zeros(0)
            u_mid = AW2c_mid + d_soc_local
            u_bot = AW2c_bot + b_soc_local  # (m,)
            v_top = (g_eq_local - AW2c_top) if p_orig > 0 else np.zeros(0)
            v_mid = d_soc_local - AW2c_mid
            v_bot = b_soc_local - AW2c_bot  # (m,)
            sigma = cW2c + kappa / tau

            K_AA = np.zeros((n_outer, n_outer), dtype=np.float64)
            if p_orig > 0:
                F_w = F_local * w_sq_nonneg  # (p_orig, n)
                K_AA[:p_orig, :p_orig] = F_w @ F_local.T
                top_mid = -F_w @ c_soc_local  # (p_orig,)
                K_AA[:p_orig, p_orig] = top_mid
                K_AA[p_orig, :p_orig] = top_mid
                K_AA[:p_orig, p_orig + 1] = -u_top
                K_AA[p_orig + 1, :p_orig] = v_top
            # mid-mid: Nonneg sum c_soc^2 * w_sq + SOC scale*(2 y0^2 - 1).
            K_AA[p_orig, p_orig] = float(
                np.dot(c_soc_local * c_soc_local, w_sq_nonneg)
            ) + scale_soc * (2.0 * y_soc[0] * y_soc[0] - 1.0)
            K_AA[p_orig, p_orig + 1] = -u_mid
            K_AA[p_orig + 1, p_orig] = v_mid
            K_AA[p_orig + 1, p_orig + 1] = sigma

            # ---- Build K_AB (n_outer x m) and K_BA (m x n_outer). ----
            # M[top, bot]_{i,k} = -F[i, k] * (d_diag * w_sq)[k]   (Nonneg only)
            # M[mid, bot]_k    = c_soc[k] * (d_diag * w_sq)[k] + 2*scale*y0*y_factor[k]
            # K[bot, tau]      = -u[bot]
            # K[tau, bot]      = v[bot]
            d_w = d_diag * w_sq_nonneg[:m]  # (m,)
            soc_mid_bot = 2.0 * scale_soc * y_soc[0] * u_factor  # (m,)
            mid_bot = c_soc_local[:m] * d_w + soc_mid_bot  # (m,)

            K_AB = np.empty((n_outer, m), dtype=np.float64)
            if p_orig > 0:
                K_AB[:p_orig, :] = -F_local[:, :m] * d_w[None, :]
            K_AB[p_orig, :] = mid_bot
            # K[tau, bot] = v[bot] (mirrors K_AA[p_orig + 1, :p_orig] = v_top).
            K_AB[p_orig + 1, :] = v_bot

            # K_BA cross blocks: M is symmetric on the (top, mid) axes so
            # K_BA[:, top] = M[bot, top] = M[top, bot].T.
            # K[bot, tau] = -u[bot] (mirrors K_AA[:p_orig, p_orig + 1] = -u_top).
            K_BA = np.empty((m, n_outer), dtype=np.float64)
            if p_orig > 0:
                K_BA[:, :p_orig] = (-F_local[:, :m] * d_w[None, :]).T
            K_BA[:, p_orig] = mid_bot
            K_BA[:, p_orig + 1] = -u_bot

            # ---- Schur reduction. ----
            # rhs_A = [rhs_y[top..mid]; rhs_tau];  rhs_B = rhs_y[bot].
            rhs_A = np.empty(n_outer, dtype=np.float64)
            if p_orig > 0:
                rhs_A[:p_orig] = rhs_y[:p_orig]
            rhs_A[p_orig] = rhs_y[p_orig]
            rhs_A[p_orig + 1] = rhs_tau
            rhs_B = rhs_y[p_orig + 1 :]  # (m,)

            # K_BB^{-1} K_BA: apply column-by-column (n_outer cols, each O(m)).
            K_BB_inv_K_BA = np.empty((m, n_outer), dtype=np.float64)
            for j in range(n_outer):
                K_BB_inv_K_BA[:, j] = K_BB_inv_vec(K_BA[:, j])
            schur = K_AA - K_AB @ K_BB_inv_K_BA  # (n_outer, n_outer)
            rhs_A_eff = rhs_A - K_AB @ K_BB_inv_vec(rhs_B)  # (n_outer,)

            sol_A = np.linalg.solve(schur, rhs_A_eff)  # (n_outer,)
            sol_B = K_BB_inv_vec(rhs_B - K_BA @ sol_A)  # (m,)

            # delta_y = [sol_A[top..mid]; sol_B];  delta_tau = sol_A[tau slot].
            delta_y = np.empty(p_full, dtype=np.float64)
            if p_orig > 0:
                delta_y[:p_orig] = sol_A[:p_orig]
            delta_y[p_orig] = sol_A[p_orig]
            delta_y[p_orig + 1 :] = sol_B
            delta_tau = float(sol_A[p_orig + 1])

            # Back-substitute: delta_x = W^2 (A^T delta_y - c delta_tau + p_term).
            AT_dy = _AT_matvec(delta_y)
            v = AT_dy.copy()
            v[:n] -= f_local * delta_tau
            v += p_term
            delta_x = _apply_W_squared_full(v, w_sq_nonneg, scale_soc, y_soc)

            # delta_s = -A^T delta_y + c delta_tau - rhs_d.
            delta_s = -AT_dy
            delta_s[:n] += f_local * delta_tau
            delta_s -= rhs_d

            delta_kappa = (rhs_tk - kappa * delta_tau) / tau
            return delta_x, delta_y, delta_tau, delta_s, delta_kappa

    return _SocpLinearHsdEmbedding


def _solve_socp_via_hsd(
    f: npt.NDArray[np.float64],
    A_soc: npt.NDArray[np.float64],
    b_soc: npt.NDArray[np.float64],
    c_soc: npt.NDArray[np.float64],
    d_soc: float,
    F: npt.NDArray[np.float64],
    g_eq: npt.NDArray[np.float64],
    settings: OptimizationSettings,
    A_soc_diag_hint: npt.NDArray[np.float64] | None = None,
) -> SolveResult:
    """Embed the SOCP in canonical conic form and solve it with ``HsdSolver``.

    Builds the problem
       minimize    f_full^T z
       subject to  A_full z = b_full,   z ∈ Nonneg(n) × SOC(m + 1),
    with z = (x, t, u), f_full = (f, 0, 0), b_full = (g_eq, d_soc, b_soc), and
                 _                         _
       A_full = |   F          0     0      |
                |  -c_soc^T    1     0      |
                |_ -A_soc      0     I_m   _|
    The augmented problem has n + 1 + m variables and p + 1 + m equality constraints:
    the p rows of F x = g_eq, then the m + 1 rows defining the slack variables t and u.

    When A_soc is diagonal and zero-padded (from the hint, or detected by
    ``_A_soc_diagonal_zero_padded``), the solve uses the structured embedding from
    ``_make_socp_structured_embedding_class``, and A_full is a read-only zero
    placeholder of the right shape, never read. Otherwise A_full is formed densely and
    the generic embedding runs. Either way the cones are Nonneg(n) and the slack cone
    from ``_make_slack_soc_cone``. The result is mapped back to the original variables
    as described in ``solve_second_order_cone_program``.

    Parameters
    ----------
     f : npt.NDArray[np.float64]
        Linear objective vector, of length n.
     A_soc : npt.NDArray[np.float64]
        Matrix in the cone constraint, of shape (m, n).
     b_soc : npt.NDArray[np.float64]
        Vector in the cone constraint, of length m.
     c_soc : npt.NDArray[np.float64]
        Linear coefficient in the cone constraint, of length n.
     d_soc : float
        Scalar in the cone constraint.
     F : npt.NDArray[np.float64]
        Equality constraint matrix, of shape (p, n); empty for no equality constraints.
     g_eq : npt.NDArray[np.float64]
        Equality constraint right-hand side, of length p.
     settings : OptimizationSettings
        Solver settings.
     A_soc_diag_hint : npt.NDArray[np.float64] | None
        Trusted diagonal of A_soc, as in ``solve_second_order_cone_program``; ``None``
        detects the structure instead.

    Returns
    -------
     result : SolveResult
        The ``HsdSolver`` result mapped back to the original variables.

    Raises
    ------
     ValueError
        When f is empty or A_soc has no rows, propagated from the ``Nonneg`` and
        ``SecondOrderCone`` constructors.

    """
    from ..problems.cone import Cone, Nonneg
    from ..problems.objective import (
        LinearObjective,
    )
    from ..problems.problem import Problem
    from ..solvers.hsd_solver import HsdSolver

    n = f.shape[0]
    m = A_soc.shape[0]
    p = F.shape[0] if F.size > 0 else 0
    n_full = n + 1 + m

    # Determine path early so we can skip the dense O((p+1+m) x (n+1+m))
    # ``A_full`` allocation when the structured embedding will be used.
    if A_soc_diag_hint is not None:
        A_soc_diag_for_path: npt.NDArray[np.float64] | None = A_soc_diag_hint
    else:
        A_soc_diag_for_path = _A_soc_diagonal_zero_padded(A_soc)
    use_structured = A_soc_diag_for_path is not None

    # b_full is always cheap (size O(n)) and IS used by the structured path.
    b_full = np.concatenate([g_eq, [d_soc], b_soc]).astype(np.float64, copy=False)

    if use_structured:
        # The structured embedding overrides ``residuals`` + ``kkt_solve`` and
        # never reads ``self._A``, so we pass a zero placeholder that just
        # satisfies the ``EqualitySolver`` / ``_HsdEmbedding`` shape checks
        # (must have ``shape[0] = p + 1 + m`` to match ``b_full`` and define
        # ``num_eq``). A broadcast view costs O(1) memory yet reports the honest
        # column count ``n_full`` -- the true logical width of the HSD equality
        # system -- so ``Problem``'s dimension-integrity check sees a consistent
        # ``eq.dimension`` instead of the misleading single-column shape.
        A_full: npt.NDArray[np.float64] = np.broadcast_to(
            np.float64(0.0), (p + 1 + m, n_full)
        )
        # Defensive: this is a read-only, stride-0 view, not an allocated matrix.
        # The structured path must never write it or copy through an in-place op;
        # assert the read-only contract so a future regression (e.g. swapping in
        # a writeable placeholder) fails loudly here instead of silently
        # diverging from the dense path's semantics.
        assert not A_full.flags.writeable
    else:
        # Stacked equality system (dense IPM-comparable path):
        #   [F    0    0  ] [x]   [g_eq]
        #   [-c^T 1    0  ] [t] = [d_soc]
        #   [-A   0   I_m ] [u]   [b_soc]
        A_full = np.zeros((p + 1 + m, n_full), dtype=np.float64)
        if p > 0:
            A_full[:p, :n] = F
        A_full[p, :n] = -c_soc
        A_full[p, n] = 1.0
        A_full[p + 1 :, :n] = -A_soc
        A_full[p + 1 :, n + 1 :] = np.eye(m, dtype=np.float64)

    f_full = np.zeros(n_full, dtype=np.float64)
    f_full[:n] = f

    eq_layer = EqualityConstraintLayer(A=A_full, b=b_full)
    problem = Problem(
        obj=LinearObjective(c=f_full),
        eq=eq_layer,
    )
    if use_structured:
        assert A_soc_diag_for_path is not None
        # Structured inner loop, linear in n for a fixed p. Skips dense M / K
        # assembly, uses structured A matvec, Schur via Sherman-Morrison. The override
        # bypasses every cone's accumulate_normal_equations, so the cones
        # below contribute only their cone-API methods (apply_w_squared,
        # solve_jordan_arrow, nt_scaling, ...) -- inherited unchanged from
        # the base classes.
        embedding_class = _make_socp_structured_embedding_class(
            n=n,
            m=m,
            p_orig=p,
            F=F,
            c_soc=c_soc,
            d_soc=d_soc,
            A_soc_diag=A_soc_diag_for_path,
            b_soc=b_soc,
            g_eq=g_eq,
            f=f,
        )
    else:
        embedding_class = None
    cones: list[Cone] = [Nonneg(dim=n), _make_slack_soc_cone(m=m)]

    hsd_result = HsdSolver(
        cones=cones, settings=settings, embedding_class=embedding_class
    ).solve(problem=problem, x0=np.zeros(n_full))

    # Strip slacks: the original problem only sees the first n components of x.
    x_solution = hsd_result.solution[:n]
    # Project HSD's per-block dual cone variables into the original SOCP's
    # ``inequality_multipliers`` shape: ``n + 1`` scalar multipliers
    # (``n`` bound multipliers for ``x >= 0`` plus one scalar SOC
    # multiplier). HSD's per-block dual splits as
    # ``Nonneg(n)`` -> first ``n`` entries (bound-equivalent) and
    # ``SOC(m+1)`` -> next ``1 + m`` entries (a full Lorentz-cone dual
    # vector ``(t, u)`` with ``t >= ||u||``). The leading scalar ``t``
    # is what aligns with the IPM scalar multiplier under central-path
    # stationarity; the ``u`` block has no IPM analog so we drop it.
    hsd_ineq = hsd_result.inequality_multipliers
    inequality_multipliers = np.concatenate([hsd_ineq[:n], hsd_ineq[n : n + 1]])
    # Re-skin the HSD result onto the original SOCP variables: keep the HSD
    # spine (status, message, n_iterations, diagnostics, certificate) and replace
    # the solution / objective / duals with the stripped-and-projected originals.
    # NOTE: the ``certificate`` field (if present) carries witness vectors in
    # embedding coordinates (a primal witness has length n + 1 + m, a dual
    # witness length p + 1 + m), NOT the original n-dimensional SOCP variables.
    # Callers that need a certificate for an infeasible SOCP should treat the
    # witnesses as embedding-space vectors.
    return replace(
        hsd_result,
        solution=x_solution,
        objective=ObjectiveOutcome(
            value=float(f @ x_solution),
            suboptimality=hsd_result.suboptimality,
            dual_value=hsd_result.dual_value,
        ),
        duals=Duals(
            equality_multipliers=(
                hsd_result.equality_multipliers[:p] if p > 0 else None
            ),
            inequality_multipliers=inequality_multipliers,
        ),
    )
