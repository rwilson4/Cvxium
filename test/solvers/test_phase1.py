"""Test the Phase I helper adapters in phase1.py."""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium.linalg.systems import (
    ArrowSystem,
    DiagonalSystem,
    LowRankUpdatedSystem,
)
from cvxium.problems.constraint_layers import (
    ConstraintLayer,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    ImbalanceConstraintLayer,
)
from cvxium.problems.quadratic_constraints import (
    SquaredNormConstraintLayer,
)
from cvxium.solvers.phase1 import (
    LiftedConstraintLayer,
    _SlackActiveLayer,
    _SlackUpperBoundLayer,
)
from cvxium.solvers.settings import (
    LineSearchSettings,
)


class LiftedLayerTest(unittest.TestCase):
    """Tests for `LiftedConstraintLayer` -- dimension-lifting wrapper for background layers."""

    def test_extra_dimension(self) -> None:
        """Several trailing components: dimension, Jacobian padding and constraints."""
        M, extra = 4, 3
        rng = np.random.default_rng(0)
        base = ImbalanceConstraintLayer(
            B=rng.standard_normal((2, M)), c=np.zeros(2), psi=1.0
        )
        lifted = LiftedConstraintLayer(
            base=base, base_dimension=M, extra_dimension=extra
        )
        w = rng.standard_normal(M + extra)
        self.assertEqual(lifted.dimension, M + extra)
        np.testing.assert_allclose(lifted.constraints(w), base.constraints(w[:M]))
        G = lifted.grad_constraints(w)
        self.assertEqual(G.shape, (4, M + extra))
        np.testing.assert_allclose(G[:, M:], 0.0)
        lmbda = np.ones(lifted.num_constraints(M + extra))
        self.assertEqual(lifted.hessian_contribution(w, lmbda).dimension, M + extra)

    def test_linearity_is_forwarded(self) -> None:
        """A lifted squared norm is not affine, whatever its wrapper's class says."""
        M = 3
        affine = LiftedConstraintLayer(
            base=BoundsConstraintLayer(lb=0.0, dimension=M), base_dimension=M
        )
        curved = LiftedConstraintLayer(
            base=SquaredNormConstraintLayer(phi=1.0, dimension=M), base_dimension=M
        )
        self.assertTrue(affine.is_linear)
        self.assertFalse(curved.is_linear)

    def test_extra_dimension_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            LiftedConstraintLayer(
                base=BoundsConstraintLayer(lb=0.0, dimension=2),
                base_dimension=2,
                extra_dimension=0,
            )

    @parameterized.expand([(5, 1), (10, 1), (20, 3)])
    def test_num_constraints_uses_base_dim(self, M: int, extra: int) -> None:
        """`num_constraints` always reports the base layer's count, ignoring lifted dim."""
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        assert lifted.num_constraints(M + extra) == base.num_constraints(M)

    @parameterized.expand([(901, 5), (902, 20)])
    def test_constraints_uses_base_components(self, seed: int, M: int) -> None:
        """`constraints(w_aug)` evaluates the base layer on ``w_aug[:M]`` only."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(0.5, 2.0, size=M)
        slack = -1.0  # arbitrary; should not affect lower-bound layer
        w_aug = np.concatenate([w_base, [slack]])

        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)

        np.testing.assert_array_equal(
            lifted.constraints(w_aug), base.constraints(w_base)
        )

    @parameterized.expand([(911, 5), (912, 20)])
    def test_grad_constraints_zero_in_extra_columns(self, seed: int, M: int) -> None:
        """The Jacobian gets zero columns for the extra (slack) dimensions."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(0.5, 2.0, size=M)
        w_aug = np.concatenate([w_base, [0.0]])

        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        G_aug = lifted.grad_constraints(w_aug)
        G_base = base.grad_constraints(w_base)

        # First M columns match the base; the extra column is all zeros.
        np.testing.assert_array_equal(G_aug[:, :M], G_base)
        np.testing.assert_array_equal(G_aug[:, M], np.zeros(G_aug.shape[0]))

    @parameterized.expand([(921, 5), (922, 20)])
    def test_grad_constraints_multiply_uses_base_components(
        self, seed: int, M: int
    ) -> None:
        """`G @ v` only depends on `v[:M]` since the extra columns are zero."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(0.5, 2.0, size=M)
        w_aug = np.concatenate([w_base, [0.0]])
        v_base = rng.standard_normal(M)
        v_aug = np.concatenate([v_base, [rng.standard_normal()]])  # arbitrary slack
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)

        np.testing.assert_array_equal(
            lifted.grad_constraints_multiply(w_aug, v_aug),
            base.grad_constraints_multiply(w_base, v_base),
        )

    @parameterized.expand([(931, 5), (932, 20)])
    def test_grad_constraints_transpose_multiply_zero_padded(
        self, seed: int, M: int
    ) -> None:
        """`G^T @ u` is zero in the extra dimensions."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(0.5, 2.0, size=M)
        w_aug = np.concatenate([w_base, [0.0]])
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        u = rng.standard_normal(base.num_constraints(M))

        result = lifted.grad_constraints_transpose_multiply(w_aug, u)
        result_base = base.grad_constraints_transpose_multiply(w_base, u)

        np.testing.assert_array_equal(result[:M], result_base)
        assert result[M] == 0.0

    @parameterized.expand([(2001, 4), (2002, 9)])
    def test_lifted_diagonal_base_composes(self, seed: int, M: int) -> None:
        """A lifted diagonal base embeds and composes with a full diagonal.

        The embedded base curvature must occupy the first ``M`` coordinates, the
        slack slot must stay zero, and summing onto a full-dimension diagonal must
        match the dense ground truth under both solve and multiply.
        """
        rng = np.random.default_rng(seed)
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        w_base = rng.uniform(0.5, 2.0, size=M)
        w_aug = np.concatenate([w_base, [rng.standard_normal()]])
        lmbda = -1.0 / base.constraints(w_base)  # barrier weights, all positive

        base_dense = np.column_stack(
            [base.hessian_contribution(w_base, lmbda).multiply(e) for e in np.eye(M)]
        )
        eta_aug = rng.uniform(1.0, 2.0, size=M + 1)
        H = np.diag(eta_aug)
        H[:M, :M] += base_dense

        system = DiagonalSystem(eta_aug).plus(lifted.hessian_contribution(w_aug, lmbda))
        rhs = rng.standard_normal(M + 1)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    @parameterized.expand([(2011, 5), (2012, 8)])
    def test_lifted_low_rank_base_composes(self, seed: int, M: int) -> None:
        """A lifted low-rank base (Imbalance) embeds and composes with a diagonal.

        Exercises the ``BlockDiagonalSystem`` embed for a base whose contribution
        is a low-rank update rather than a diagonal.
        """
        rng = np.random.default_rng(seed)
        q = 2
        B = rng.standard_normal((q, M))
        c = rng.standard_normal(q)
        psi = np.full(q, 5.0)  # generous band so w_base is strictly feasible
        base = ImbalanceConstraintLayer(B=B, c=c, psi=psi)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        w_base = np.zeros(M)
        assert np.all(np.abs(B @ w_base - c) < psi)
        w_aug = np.concatenate([w_base, [rng.standard_normal()]])
        lmbda = -1.0 / base.constraints(w_base)  # barrier weights, all positive

        base_dense = np.column_stack(
            [base.hessian_contribution(w_base, lmbda).multiply(e) for e in np.eye(M)]
        )
        eta_aug = rng.uniform(1.0, 2.0, size=M + 1)
        H = np.diag(eta_aug)
        H[:M, :M] += base_dense

        system = DiagonalSystem(eta_aug).plus(lifted.hessian_contribution(w_aug, lmbda))
        rhs = rng.standard_normal(M + 1)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)


def _augmented(
    w_base: npt.NDArray[np.float64], slack: float
) -> npt.NDArray[np.float64]:
    """Concatenate ``w`` and a scalar ``slack`` into an augmented vector."""
    return np.concatenate([w_base, [slack]])


class SlackActiveLayerTest(unittest.TestCase):
    """Tests for `_SlackActiveLayer` -- slack-augmented active layer for Phase I."""

    @parameterized.expand([(1001, 5), (1002, 20)])
    def test_constraints_subtract_slack(self, seed: int, M: int) -> None:
        """`constraints(w_aug)` returns `f(w) - s`."""
        rng = np.random.default_rng(seed)
        # Construct w that is feasible for the lower bound (w > 0) but with
        # specific values; we want the slack-shifted constraints.
        w_base = rng.uniform(2.0, 3.0, size=M)
        s = -1.5  # arbitrary slack
        w_aug = _augmented(w_base, s)

        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)

        # base.constraints(w_base) = -w_base (since lb=0, constraint = lb - w = -w)
        # active.constraints(w_aug) should = base.constraints - s = -w_base - s
        np.testing.assert_allclose(
            active.constraints(w_aug),
            -w_base - s,
        )

    @parameterized.expand([(1011, 5), (1012, 20)])
    def test_grad_constraints_appends_neg_one_column(self, seed: int, M: int) -> None:
        """Augmented Jacobian: base columns then a `-1` slack column."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(2.0, 3.0, size=M)
        w_aug = _augmented(w_base, 0.0)

        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        G_aug = active.grad_constraints(w_aug)
        G_base = base.grad_constraints(w_base)

        np.testing.assert_array_equal(G_aug[:, :M], G_base)
        np.testing.assert_array_equal(G_aug[:, M], -np.ones(G_aug.shape[0]))

    @parameterized.expand([(1021, 5), (1022, 20)])
    def test_grad_constraints_multiply_matches_dense(self, seed: int, M: int) -> None:
        """`G_aug v` matches the dense Jacobian times v."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(2.0, 3.0, size=M)
        w_aug = _augmented(w_base, 0.0)
        v = rng.standard_normal(M + 1)

        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)

        np.testing.assert_allclose(
            active.grad_constraints_multiply(w_aug, v),
            active.grad_constraints(w_aug) @ v,
            rtol=1e-12,
        )

    @parameterized.expand([(1031, 5), (1032, 20)])
    def test_grad_constraints_transpose_multiply_matches_dense(
        self, seed: int, M: int
    ) -> None:
        """`G_aug^T u` matches the dense Jacobian-transpose times u."""
        rng = np.random.default_rng(seed)
        w_base = rng.uniform(2.0, 3.0, size=M)
        w_aug = _augmented(w_base, 0.0)
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        u = rng.standard_normal(active.num_constraints(M + 1))

        np.testing.assert_allclose(
            active.grad_constraints_transpose_multiply(w_aug, u),
            active.grad_constraints(w_aug).T @ u,
            rtol=1e-12,
        )


class SlackUpperBoundLayerHessianContributionTest(unittest.TestCase):
    """Structured Hessian contribution for `_SlackUpperBoundLayer`."""

    @parameterized.expand([(3,), (8,)])
    def test_matches_diagonal_background(self, M: int) -> None:
        """At the barrier weight the contribution equals 1/gap^2 in the slack slot.

        The contribution is a pure ``DiagonalSystem``; at ``lambda = -1/f`` it must
        equal zeros except the slack slot, which holds ``1/gap^2``.
        """
        layer = _SlackUpperBoundLayer(s_max=4.0, base_dimension=M)
        w = np.zeros(M + 1)
        w[M] = 1.5  # slack strictly below s_max
        gap = 4.0 - 1.5
        f = -gap  # constraint value s - s_max < 0
        lam = -1.0 / f  # barrier weight
        system = layer.hessian_contribution(w, np.array([lam]))

        expected = np.zeros(M + 1)
        expected[M] = 1.0 / (gap * gap)
        # multiply by basis vectors to materialise the diagonal.
        materialized = np.array(
            [system.multiply(e)[i] for i, e in enumerate(np.eye(M + 1))]
        )
        np.testing.assert_allclose(materialized, expected, rtol=1e-12, atol=1e-14)
        # Off-slack coordinates carry no curvature.
        self.assertAlmostEqual(materialized[M], 1.0 / (gap * gap))
        np.testing.assert_array_equal(materialized[:M], np.zeros(M))

    def test_requires_lmbda(self) -> None:
        """``hessian_contribution`` raises without per-constraint weights."""
        layer = _SlackUpperBoundLayer(s_max=4.0, base_dimension=3)
        with self.assertRaisesRegex(ValueError, "requires `lmbda`"):
            layer.hessian_contribution(np.zeros(4))

    def test_out_of_domain(self) -> None:
        """``hessian_contribution`` raises when the slack is at or above s_max."""
        layer = _SlackUpperBoundLayer(s_max=4.0, base_dimension=3)
        w = np.zeros(4)
        w[3] = 5.0  # slack above s_max => gap < 0
        with self.assertRaisesRegex(ValueError, "out of domain"):
            layer.hessian_contribution(w, np.array([1.0]))

        # Boundary: slack exactly at s_max => gap == 0 (the guard is ``gap <= 0``).
        w_boundary = np.zeros(4)
        w_boundary[3] = 4.0
        with self.assertRaisesRegex(ValueError, "out of domain"):
            layer.hessian_contribution(w_boundary, np.array([1.0]))


class SlackActiveLayerHessianContributionTest(unittest.TestCase):
    """Structured Hessian contribution (arrow) for `_SlackActiveLayer`."""

    @parameterized.expand([(3001, 4), (3002, 9)])
    def test_matches_dense(self, seed: int, M: int) -> None:
        """The arrow, summed with an external diagonal, matches the dense 2x2 block.

        The augmented barrier Hessian of the slack-active layer is the arrow

            [[G^T diag(mu) G, -G^T mu], [-(G^T mu)^T, sum mu]]

        with mu = -lambda / c. Built independently here from the raw Jacobian G
        and verified under both solve and multiply.
        """
        rng = np.random.default_rng(seed)
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_inner = rng.uniform(1.5, 2.5, size=M)
        s = 0.5
        w_aug = _augmented(w_inner, s)
        c = base.constraints(w_inner, shift=s)
        assert np.all(c < 0.0)  # strictly feasible slack-shifted constraints
        lmbda = -1.0 / c  # barrier weights
        mu = -lmbda / c

        G = base.grad_constraints(w_inner)  # (M, M), equals -I for a lower bound
        w_block = G.T @ np.diag(mu) @ G
        border = -G.T @ mu
        corner = float(np.sum(mu))
        eta_aug = rng.uniform(1.0, 2.0, size=M + 1)
        H = np.diag(eta_aug)
        H[:M, :M] += w_block
        H[:M, M] += border
        H[M, :M] += border
        H[M, M] += corner

        system = DiagonalSystem(eta_aug).plus(active.hessian_contribution(w_aug, lmbda))
        rhs = rng.standard_normal(M + 1)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    def test_requires_lmbda(self) -> None:
        """``hessian_contribution`` raises without per-constraint weights."""
        active = _SlackActiveLayer(
            base=BoundsConstraintLayer(lb=0.0, dimension=3), base_dimension=3
        )
        with self.assertRaisesRegex(ValueError, "requires `lmbda`"):
            active.hessian_contribution(np.zeros(4))

    def test_out_of_domain(self) -> None:
        """``hessian_contribution`` raises when a slack-shifted constraint is feasible-violating."""
        active = _SlackActiveLayer(
            base=BoundsConstraintLayer(lb=0.0, dimension=2), base_dimension=2
        )
        # s = -1, w = 0.5 => c = -w + lb - s = -0.5 + 1.0 = 0.5 >= 0 (out of domain).
        w_aug = _augmented(np.array([0.5, 0.5]), -1.0)
        with self.assertRaisesRegex(ValueError, "out of domain"):
            active.hessian_contribution(w_aug, np.array([1.0, 1.0]))


class WrapperHessianCurvatureTest(unittest.TestCase):
    """The slack-form wrappers lift their base layer's curvature into augmented space.

    ``hessian_curvature`` is the constraint second-derivative term ONLY. The slack
    relaxation ``f_i(w) - s`` and the dimension lift add only linear terms, so the
    augmented curvature is the base curvature embedded with a zero slack slot -- with
    no barrier-Jacobian / arrow coupling (that lives in ``hessian_contribution``).
    """

    def test_lifted_layer_lifts_curved_base(self) -> None:
        """`LiftedConstraintLayer` embeds a curved base's curvature; the slack slot is zero."""
        M = 4
        base = SquaredNormConstraintLayer(phi=10.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        rng = np.random.default_rng(4242)
        w_aug = np.concatenate([rng.standard_normal(M), [0.5]])
        lmbda = np.array([0.7])
        v = rng.standard_normal(M + 1)

        base_curv = base.hessian_curvature(w_aug[:M], lmbda).multiply(v[:M])
        # Base curvature must be non-trivial, else the embed test is vacuous.
        assert np.any(base_curv != 0.0)

        result = lifted.hessian_curvature(w_aug, lmbda).multiply(v)
        np.testing.assert_allclose(result[:M], base_curv, atol=1e-12)
        np.testing.assert_allclose(result[M:], 0.0, atol=1e-12)

    def test_slack_active_curvature_excludes_barrier_coupling(self) -> None:
        """`_SlackActiveLayer` curvature drops the barrier arrow coupling.

        With an affine base the augmented curvature is exactly zero, while the barrier
        ``hessian_contribution`` (an ``ArrowSystem``) is non-zero at the same point --
        confirming ``hessian_curvature`` is the curvature-only term, not the
        contribution.
        """
        M = 4
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M, s_max=5.0)
        rng = np.random.default_rng(99)
        w_aug = np.concatenate([rng.uniform(0.5, 2.0, M), [0.0]])
        lmbda = rng.uniform(0.1, 1.0, M)
        v = rng.standard_normal(M + 1)

        curvature = active.hessian_curvature(w_aug, lmbda)
        np.testing.assert_allclose(curvature.multiply(v), np.zeros(M + 1), atol=1e-12)

        contribution = active.hessian_contribution(w_aug, lmbda)
        assert np.any(contribution.multiply(v) != 0.0)

    def test_slack_active_curvature_embeds_curved_base(self) -> None:
        """`_SlackActiveLayer` embeds a curved base's curvature; the slack slot is zero.

        Real slack-form layers are affine, so a synthetic curved base exercises the
        embed: the base curvature lands in the first ``M`` coordinates and the slack
        slot is a ``ZeroSystem``.
        """

        class _CurvedBoundsStub(BoundsConstraintLayer):
            """Bounds layer with a synthetic non-zero curvature (passthrough wrap)."""

            def hessian_curvature(
                self,
                w: npt.NDArray[np.float64],
                lmbda: npt.NDArray[np.float64],
            ) -> DiagonalSystem:
                return DiagonalSystem(1.0 + np.arange(len(w), dtype=np.float64))

        M = 4
        base = _CurvedBoundsStub(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M, s_max=5.0)
        rng = np.random.default_rng(123)
        w_aug = np.concatenate([rng.uniform(0.5, 2.0, M), [0.0]])
        lmbda = rng.uniform(0.1, 1.0, M)
        v = rng.standard_normal(M + 1)

        base_curv = base.hessian_curvature(w_aug[:M], lmbda).multiply(v[:M])
        assert np.any(base_curv != 0.0)  # non-trivial, else the embed test is vacuous

        result = active.hessian_curvature(w_aug, lmbda).multiply(v)
        np.testing.assert_allclose(result[:M], base_curv, atol=1e-12)
        np.testing.assert_allclose(result[M:], 0.0, atol=1e-12)


def _imbalance_near_boundary(
    seed: int,
    M: int,
    q: int,
    min_c: float,
    s: float = 0.5,
    psi_val: float = 1.0,
) -> tuple[
    ImbalanceConstraintLayer,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    r"""Build an `ImbalanceConstraintLayer` whose nearest slack-shifted constraint is `-min_c`.

    The negative-side row ``f^- = (B w - c) - psi - s`` is pinned to ``-min_c`` by
    choosing the target ``c = B w - (psi + s - min_c)``; the positive side is then
    safely negative. Returns ``(base, w_base, c_con, lmbda, mu)`` with barrier weights
    ``lmbda = -1 / c_con`` and ``mu = -lmbda / c_con = 1 / c_con^2``.
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((q, M))
    w_base = rng.standard_normal(M)
    psi = np.full(q, psi_val)
    c_target = B @ w_base - (psi_val + s - min_c)
    base = ImbalanceConstraintLayer(B=B, c=c_target, psi=psi)
    c_con = base.constraints(w_base, shift=s)
    lmbda = -1.0 / c_con
    mu = -lmbda / c_con
    return base, w_base, c_con, lmbda, mu


class SlackActiveLayerLowRankContributionTest(unittest.TestCase):
    """Structured contribution for a rank-update base (`ImbalanceConstraintLayer`).

    A rank-update ``H_{ww}`` is folded into a single rank-``p`` ``LowRankUpdatedSystem``
    rather than an ``ArrowSystem``, so the slack-augmented solve goes through the
    equilibrated Woodbury kernel and stays usable as ``c -> 0``.
    """

    def test_contribution_type_by_base(self) -> None:
        """Diagonal base -> ``ArrowSystem``; rank-update base -> ``LowRankUpdatedSystem``."""
        M, q = 6, 2
        rng = np.random.default_rng(7001)

        bounds = BoundsConstraintLayer(lb=0.0, dimension=M)
        active_diag = _SlackActiveLayer(base=bounds, base_dimension=M)
        w_aug_diag = _augmented(rng.uniform(1.5, 2.5, size=M), 0.5)
        lmbda_diag = -1.0 / bounds.constraints(w_aug_diag[:M], shift=0.5)
        assert isinstance(
            active_diag.hessian_contribution(w_aug_diag, lmbda_diag), ArrowSystem
        )

        base, w_base, _c, lmbda, _mu = _imbalance_near_boundary(
            seed=7002, M=M, q=q, min_c=1.0
        )
        active_lr = _SlackActiveLayer(base=base, base_dimension=M)
        contribution = active_lr.hessian_contribution(_augmented(w_base, 0.5), lmbda)
        assert isinstance(contribution, LowRankUpdatedSystem)

    @parameterized.expand([(7101, 5, 2), (7102, 8, 3)])
    def test_imbalance_matches_dense(self, seed: int, M: int, q: int) -> None:
        """The unified rank-p atom, summed with a diagonal, matches the dense block.

        Mirrors the diagonal-base ``test_matches_dense`` but with a genuinely
        low-rank ``H_{ww}``: the atom must equal

            [[G^T diag(mu) G, -G^T mu], [-(G^T mu)^T, sum mu]]

        under both solve and multiply once a positive diagonal (mimicking the
        background layers and the slack upper bound) folds into its zero base.
        """
        base, w_base, c_con, lmbda, mu = _imbalance_near_boundary(
            seed=seed, M=M, q=q, min_c=0.8
        )
        assert np.all(c_con < 0.0)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_aug = _augmented(w_base, 0.5)

        G = base.grad_constraints(w_base)  # (2q, M)
        w_block = G.T @ np.diag(mu) @ G
        border = -G.T @ mu
        corner = float(np.sum(mu))
        rng = np.random.default_rng(seed + 1)
        eta_aug = rng.uniform(1.0, 2.0, size=M + 1)
        H = np.diag(eta_aug)
        H[:M, :M] += w_block
        H[:M, M] += border
        H[M, :M] += border
        H[M, M] += corner

        system = DiagonalSystem(eta_aug).plus(active.hessian_contribution(w_aug, lmbda))
        rhs = rng.standard_normal(M + 1)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(system.multiply(rhs), H @ rhs, rtol=1e-10)

    @parameterized.expand([(7201, 5, 2), (7202, 8, 3)])
    def test_imbalance_elevated_barrier_matches_dense(
        self, seed: int, M: int, q: int
    ) -> None:
        """At an elevated barrier (``min|c| = 1e-4``) the unified solve still tracks dense.

        The conditioning is ``~1e8`` here, where a dense ``np.linalg.solve`` is still
        a trustworthy oracle; the unified Woodbury path must agree to a loose tolerance.
        """
        base, w_base, c_con, lmbda, mu = _imbalance_near_boundary(
            seed=seed, M=M, q=q, min_c=1e-4
        )
        np.testing.assert_allclose(np.min(np.abs(c_con)), 1e-4, rtol=1e-6)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_aug = _augmented(w_base, 0.5)

        G = base.grad_constraints(w_base)
        H = np.zeros((M + 1, M + 1))
        H[:M, :M] = G.T @ np.diag(mu) @ G
        H[:M, M] = -G.T @ mu
        H[M, :M] = -G.T @ mu
        H[M, M] = float(np.sum(mu))
        rng = np.random.default_rng(seed + 1)
        eta_aug = rng.uniform(1.0, 2.0, size=M + 1)
        H[np.diag_indices(M + 1)] += eta_aug

        system = DiagonalSystem(eta_aug).plus(active.hessian_contribution(w_aug, lmbda))
        rhs = rng.standard_normal(M + 1)
        np.testing.assert_allclose(
            system.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-4, atol=1e-6
        )

    @parameterized.expand(
        [
            # Cross-product of (seed, M, q) x (min_c) -- 2 x 3 = 6 cases.
            (7301, 6, 2, 1e-8),
            (7301, 6, 2, 1e-10),
            (7301, 6, 2, 1e-12),
            (7302, 10, 3, 1e-8),
            (7302, 10, 3, 1e-10),
            (7302, 10, 3, 1e-12),
        ]
    )
    def test_high_barrier_unified_stays_finite_and_bounded(
        self, seed: int, M: int, q: int, min_c: float
    ) -> None:
        """At ``c -> 0`` the unified path returns a finite, bounded Newton step.

        This is the regime that motivated the change. The legacy nested
        ``ArrowSystem`` over a low-rank interior solves the slack slot via
        ``solve_block_plus_one``'s scalar Schur ``corner - border^T H_ww^{-1}
        border``, which subtracts two near-equal ``O(1/c)`` terms; once the
        cancellation noise (``~|H| * eps``) swamps the true ``O(D)`` Schur it can land
        non-positive and falsely reject the step with ``NewtonStepError``. The unified
        rank-``p`` update routes through ``sqrt(|d|)`` equilibration and never forms
        that difference, so it solves without false rejection at barriers far past
        where the scalar Schur degenerates.

        Accuracy of this same path is pinned against a dense oracle at moderate
        conditioning by ``test_imbalance_matches_dense`` and
        ``test_imbalance_elevated_barrier_matches_dense``; here -- where a dense oracle
        is itself untrustworthy -- the contract is finiteness and a bounded step (the
        super-stiff slack must be driven toward zero, not blown up).
        """
        base, w_base, c_con, lmbda, mu = _imbalance_near_boundary(
            seed=seed, M=M, q=q, min_c=min_c
        )
        assert np.all(c_con < 0.0)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_aug = _augmented(w_base, 0.5)
        # An O(1) background diagonal (mimicking the other Phase I layers) keeps the
        # w-block well-conditioned, so only the rank/slack subspace carries the c -> 0
        # stiffness that the solve must absorb gracefully.
        rng = np.random.default_rng(seed + 1)
        eta_aug = rng.uniform(1.0, 2.0, size=M + 1)
        rhs = rng.standard_normal(M + 1)

        system = DiagonalSystem(eta_aug).plus(active.hessian_contribution(w_aug, lmbda))
        x = system.solve(rhs)
        assert np.all(np.isfinite(x))
        # The slack coordinate is super-stiff (corner ~ sum(mu) ~ 1/min_c^2), so its
        # Newton step is driven toward zero; the w-block step stays O(1)-bounded.
        assert abs(x[M]) < 1.0
        assert float(np.linalg.norm(x[:M])) < 1e2


def _largest_feasible_by_bisection(
    layer: ConstraintLayer,
    w: npt.NDArray[np.float64],
    dw: npt.NDArray[np.float64],
    shift: float,
    ds: float,
    hi: float,
) -> float:
    """Brute-force the feasibility boundary along the ray, by bisection on ``constraints``.

    An independent oracle for the closed forms: it consults only ``constraints``,
    so it cannot inherit an error from the analytic bound it checks. Requires the
    ray to be feasible at 0 and infeasible at ``hi``.
    """

    def feasible(alpha: float) -> bool:
        return bool(
            np.all(layer.constraints(w + alpha * dw, shift=shift + alpha * ds) < 0)
        )

    assert feasible(0.0), "ray must start strictly feasible"
    assert not feasible(hi), "hi must bracket the boundary"
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


class Phase1BtlsMaxStepTest(unittest.TestCase):
    r"""The Phase I wrappers report the exact feasibility boundary, not a shrunk search.

    Each wrapper's constraints are the base layer's under a relabelling, so each can
    delegate or divide rather than inherit `ConstraintLayer`'s iterative search. Two
    properties matter, and they are tested separately: the bound is *exact* (agrees
    with an independent bisection on ``constraints``), and it is *uncapped* (may
    exceed the unit step). The second is the one that regressed historically -- a
    bound silently capped at 1 makes the caller's ``min(1, beta * cap)`` return
    ``beta``, a permanent half-step past quadratic convergence.
    """

    def test_lifted_layer_ignores_the_slack_direction(self) -> None:
        """The lifted layer's bound is the base's, whatever the trailing slots do."""
        M = 3
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        lifted = LiftedConstraintLayer(base=base, base_dimension=M)
        w_base = np.array([1.0, 2.0, 3.0])
        dw_base = np.array([-1.0, 0.0, 0.0])
        expected = base.btls_max_step(w_base, dw_base)
        # The trailing slack slot does not enter this layer's constraints, so wildly
        # different slack motions must all give the base layer's answer.
        for slack_dir in (-100.0, 0.0, 100.0):
            alpha = lifted.btls_max_step(
                _augmented(w_base, 0.5), _augmented(dw_base, slack_dir)
            )
            np.testing.assert_allclose(alpha, expected)
        np.testing.assert_allclose(expected, 1.0)

    def test_slack_active_layer_couples_the_slack_to_every_constraint(self) -> None:
        r"""A shrinking slack tightens *all* rows, not just the one ``dw`` moves.

        With `f_i(w) = -w_i` relaxed by `s`, the row-`i` bound is `(w_i + s) /
        -(dw_i + ds_s)`. Here `dw = (-1, 0, 0)` alone would allow `1.5`, but
        `ds_s = -0.5` pulls the ceiling down on every row and row 0 binds at `1.0`.
        """
        M = 3
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_aug = _augmented(np.array([1.0, 2.0, 3.0]), 0.5)
        # Slack held still: only row 0 moves, and it binds at (1.0 + 0.5) / 1.0.
        held = _augmented(np.array([-1.0, 0.0, 0.0]), 0.0)
        np.testing.assert_allclose(active.btls_max_step(w_aug, held), 1.5)
        # Slack shrinking: row 0 binds at (1.0 + 0.5) / 1.5, ahead of rows 1 and 2.
        shrinking = _augmented(np.array([-1.0, 0.0, 0.0]), -0.5)
        np.testing.assert_allclose(active.btls_max_step(w_aug, shrinking), 1.0)

    @parameterized.expand([(0.0, 0.0), (0.25, 0.0), (0.0, 0.1), (0.25, -0.1)])
    def test_slack_active_layer_matches_bisection(
        self, shift: float, ds: float
    ) -> None:
        """The analytic bound is the boundary an independent bisection finds."""
        M = 4
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_aug = _augmented(np.array([1.0, 2.0, 0.75, 3.0]), 0.5)
        dw_aug = _augmented(np.array([-1.0, -0.25, -0.5, 0.0]), -0.3)
        alpha = active.btls_max_step(w_aug, dw_aug, shift=shift, ds=ds)
        expected = _largest_feasible_by_bisection(
            active, w_aug, dw_aug, shift=shift, ds=ds, hi=50.0
        )
        np.testing.assert_allclose(alpha, expected, rtol=1e-9)

    def test_slack_upper_bound_is_one_division(self) -> None:
        """Gap over closing rate, with ``shift`` raising the ceiling and ``ds`` slowing it."""
        M = 2
        layer = _SlackUpperBoundLayer(s_max=2.0, base_dimension=M)
        w_aug = _augmented(np.zeros(M), 0.5)
        rising = _augmented(np.zeros(M), 0.5)
        # gap 1.5 at rate 0.5.
        np.testing.assert_allclose(layer.btls_max_step(w_aug, rising), 3.0)
        # A raised ceiling widens the gap to 1.75.
        np.testing.assert_allclose(layer.btls_max_step(w_aug, rising, shift=0.25), 3.5)
        # The ceiling receding at exactly the slack's rate means it never binds.
        assert layer.btls_max_step(w_aug, rising, ds=0.5) == np.inf

    @parameterized.expand([(-1.0, 0.0), (0.0, 0.0), (0.5, 0.75)])
    def test_slack_upper_bound_is_infinite_when_the_slack_cannot_catch_it(
        self, slack_dir: float, ds: float
    ) -> None:
        """A non-positive closing rate means no step is ever blocked by this layer."""
        M = 2
        layer = _SlackUpperBoundLayer(s_max=2.0, base_dimension=M)
        w_aug = _augmented(np.zeros(M), 0.5)
        dw_aug = _augmented(np.zeros(M), slack_dir)
        assert layer.btls_max_step(w_aug, dw_aug, ds=ds) == np.inf

    def test_none_of_the_wrappers_caps_at_the_unit_step(self) -> None:
        r"""Every wrapper may report a bound above 1 -- the half-step regression guard.

        Callers take ``min(1, beta * cap)``. A wrapper that capped its own answer at
        1 would hand back `\beta` (~0.5) forever, which is exactly the half-stepping
        seen in the field. Each wrapper is given a ray with plenty of room, and each
        must say so.
        """
        M = 3
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        w_base = np.array([10.0, 10.0, 10.0])
        dw_base = np.array([-1.0, 0.0, 0.0])
        w_aug = _augmented(w_base, 0.5)
        dw_aug = _augmented(dw_base, 0.0)
        wrappers: list[tuple[str, float]] = [
            (
                "LiftedConstraintLayer",
                LiftedConstraintLayer(base=base, base_dimension=M).btls_max_step(
                    w_aug, dw_aug
                ),
            ),
            (
                "_SlackActiveLayer",
                _SlackActiveLayer(base=base, base_dimension=M).btls_max_step(
                    w_aug, dw_aug
                ),
            ),
            (
                "_SlackUpperBoundLayer",
                _SlackUpperBoundLayer(s_max=100.0, base_dimension=M).btls_max_step(
                    w_aug, _augmented(dw_base, 1.0)
                ),
            ),
        ]
        for name, alpha in wrappers:
            assert alpha > 1.0, f"{name} capped its bound at the unit step: {alpha}"

    def test_the_closed_form_agrees_with_the_inherited_search(self) -> None:
        r"""Same semantics as the default search, just exact instead of within `\beta`.

        The inherited search returns the largest feasible point of the geometric
        sequence starting at `1/\beta`, so it undershoots the true boundary by less
        than a factor of `\beta`. Bracketing the closed form between the two shows
        the override sharpened the answer rather than changing its meaning.
        """
        M = 4
        base = BoundsConstraintLayer(lb=0.0, dimension=M)
        active = _SlackActiveLayer(base=base, base_dimension=M)
        w_aug = _augmented(np.array([1.0, 2.0, 0.75, 3.0]), 0.5)
        dw_aug = _augmented(np.array([-1.0, -0.25, -0.5, 0.0]), -0.3)
        line_search = LineSearchSettings()
        closed = active.btls_max_step(w_aug, dw_aug, line_search=line_search)
        # Call the unbound default explicitly -- the override is what we are comparing against.
        searched = ConstraintLayer.btls_max_step(
            active, w_aug, dw_aug, line_search=line_search
        )
        # The search never overshoots, and never undershoots by a full factor of beta.
        assert searched <= closed
        assert closed < searched / line_search.beta
