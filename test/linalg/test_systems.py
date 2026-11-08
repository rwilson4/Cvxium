"""Tests for the MatrixSystem ADT."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import scipy.sparse
from parameterized import parameterized
from scipy import linalg

from cvxium.exceptions import NewtonStepError
from cvxium.linalg import systems as systems_module
from cvxium.linalg.numerical_helpers import (
    solve_rank_p_update,
)
from cvxium.linalg.systems import (
    ArrowSaddleFactory,
    ArrowSaddleSystem,
    ArrowSystem,
    BandedSaddleFactory,
    BandedSchurSaddleSystem,
    BandedSystem,
    BlockDiagonalSystem,
    CoupledSaddleSystem,
    Definiteness,
    DenseSystem,
    DiagonalSystem,
    EqualityCoupledSaddleSystem,
    IncompatibleStructureError,
    LowRankUpdatedSystem,
    MatrixSystem,
    PermutedSystem,
    SaddleSystem,
    SchurComplementSaddleFactory,
    SchurComplementSaddleSystem,
    SparseSaddleFactory,
    SparseSaddleSystem,
    ZeroSystem,
)


def _random_spd(M: int, seed: int) -> npt.NDArray[np.float64]:
    """Return a random ``(M, M)`` SPD matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M))
    return A @ A.T + M * np.eye(M)


def _dense_banded(ab: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Densify scipy upper compact banded storage to a symmetric ``(n, n)`` matrix."""
    k_plus_1, n = ab.shape
    bandwidth = k_plus_1 - 1
    H = np.zeros((n, n), dtype=np.float64)
    for offset in range(k_plus_1):
        diag_idx = bandwidth - offset
        for j in range(n):
            i = j - diag_idx
            if 0 <= i < n:
                H[i, j] = ab[offset, j]
                if i != j:
                    H[j, i] = ab[offset, j]
    return H


def _densify(system: MatrixSystem) -> npt.NDArray[np.float64]:
    """Materialize a MatrixSystem as a dense matrix, one column at a time."""
    n = system.dimension
    return np.column_stack([system.multiply(e) for e in np.eye(n)])


def _assert_scaled_matches(system: MatrixSystem, t: float) -> None:
    """``scaled(t)`` multiplies and solves as ``t`` times the original matrix."""
    H = _densify(system)
    scaled = system.scaled(t)
    np.testing.assert_allclose(_densify(scaled), t * H, rtol=1e-10, atol=1e-12)
    b = np.arange(1.0, system.dimension + 1.0)
    np.testing.assert_allclose(
        scaled.solve(b), np.linalg.solve(t * H, b), rtol=1e-8, atol=1e-10
    )


def _arrow_dense(
    eta: npt.NDArray[np.float64],
    border: npt.NDArray[np.float64],
    corner: float,
) -> npt.NDArray[np.float64]:
    """Dense form of the scalar-corner arrow ``[[diag(eta), border], [border^T, corner]]``."""
    return np.block(
        [
            [np.diag(eta), border.reshape(-1, 1)],
            [border.reshape(1, -1), np.array([[corner]])],
        ]
    )


class SystemsTest(unittest.TestCase):
    # ----- DiagonalSystem -----

    def test_diagonal_system_dimension_and_solve(self) -> None:
        eta = np.array([2.0, 3.0, 5.0, 7.0])
        sys = DiagonalSystem(eta)
        assert sys.dimension == 4
        b = np.array([4.0, 9.0, 25.0, 49.0])
        x = sys.solve(b)
        np.testing.assert_allclose(x, np.array([2.0, 3.0, 5.0, 7.0]))

    def test_diagonal_system_multiply(self) -> None:
        eta = np.array([2.0, 3.0, 5.0])
        sys = DiagonalSystem(eta)
        y = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(sys.multiply(y), eta)

    def test_diagonal_system_rejects_non_1d(self) -> None:
        with self.assertRaisesRegex(ValueError, "1D"):
            DiagonalSystem(np.eye(3))

    def test_diagonal_system_solve_matrix_rhs(self) -> None:
        eta = np.array([2.0, 4.0])
        sys = DiagonalSystem(eta)
        B = np.array([[2.0, 4.0], [4.0, 8.0]])
        np.testing.assert_allclose(sys.solve(B), np.array([[1.0, 2.0], [1.0, 2.0]]))

    def test_diagonal_system_accepts_caller_supplied_eta_inverse(self) -> None:
        # Mimic an interior-point barrier near the boundary: tiny slacks s give
        # eta = 1/s**2 (large, lossy when round-tripped) but the caller can supply
        # the more accurate eta_inverse = s**2 directly.
        s = np.array([1e-7, 2e-7, 3e-7])
        eta = 1.0 / (s * s)
        sys = DiagonalSystem(eta, eta_inverse=s * s)
        b = np.array([1.0, 1.0, 1.0])
        # solve uses the supplied eta_inverse exactly, so x == s**2 * b.
        np.testing.assert_array_equal(sys.solve(b), s * s)

    def test_diagonal_system_exposes_eta_inverse(self) -> None:
        # The default reciprocal is 1/eta; a supplied stable reciprocal is returned
        # verbatim so a reorderer can carry it forward.
        plain = DiagonalSystem(np.array([2.0, 4.0]))
        np.testing.assert_allclose(plain.eta_inverse, np.array([0.5, 0.25]))
        s = np.array([1e-7, 2e-7])
        stable = DiagonalSystem(1.0 / (s * s), eta_inverse=s * s)
        np.testing.assert_array_equal(stable.eta_inverse, s * s)

    def test_diagonal_system_rejects_mismatched_eta_inverse_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            DiagonalSystem(np.array([1.0, 2.0]), eta_inverse=np.array([1.0, 0.5, 0.25]))

    def test_diagonal_system_pd_enforced_at_solve_not_construction(self) -> None:
        # A non-PD diagonal may be constructed and composed; only solve() enforces
        # positive-definiteness. A negative eta entry...
        non_pd = DiagonalSystem(np.array([1.0, -2.0]))
        with self.assertRaises(NewtonStepError):
            non_pd.solve(np.array([1.0, 1.0]))
        # ...and a caller-supplied non-positive eta_inverse both defer to solve.
        bad_inverse = DiagonalSystem(
            np.array([1.0, 2.0]), eta_inverse=np.array([1.0, -0.5])
        )
        with self.assertRaises(NewtonStepError):
            bad_inverse.solve(np.array([1.0, 1.0]))

    def test_diagonal_system_zero_eta_composes_into_pd_sum(self) -> None:
        # The robust_opt pattern: a slack coordinate with no objective curvature
        # (zero eta) must construct, and absorbing a strictly-positive diagonal
        # (the barrier term) must yield a solvable PD system.
        zero_eta = DiagonalSystem(np.array([2.0, 0.0]))
        barrier = DiagonalSystem(np.array([1.0, 3.0]))
        combined = zero_eta.plus(barrier)
        # combined = diag([3, 3]); solve([3, 3]) == [1, 1].
        np.testing.assert_allclose(
            combined.solve(np.array([3.0, 3.0])), np.array([1.0, 1.0])
        )

    def test_diagonal_system_negative_plus_positive_composes_into_pd_sum(self) -> None:
        # A negative-curvature entry (documented as constructible) absorbed with a
        # larger positive barrier entry yields a PD diagonal that must solve. The
        # absorbed stable reciprocal must stay positive for the PD sum -- regression
        # for the _absorb formula picking the wrong (negative) reciprocal under mixed
        # signs, which spuriously raised "not PD" on a genuinely-PD diagonal.
        negative = DiagonalSystem(np.array([-1.0, 2.0]))
        barrier = DiagonalSystem(np.array([3.0, 5.0]))
        combined = negative.plus(barrier)
        # combined = diag([2, 7]); solve([2, 7]) == [1, 1].
        np.testing.assert_allclose(
            combined.solve(np.array([2.0, 7.0])), np.array([1.0, 1.0])
        )

    # ----- BandedSystem -----

    def test_banded_system_solve_vs_dense(self) -> None:
        # 3-banded (bandwidth 2): construct upper compact storage for a 5x5 SPD matrix.
        n = 5
        rng = np.random.default_rng(11)
        main = rng.uniform(5.0, 10.0, size=n)
        sup1 = rng.uniform(-1.0, 1.0, size=n)
        sup2 = rng.uniform(-0.5, 0.5, size=n)
        ab = np.vstack([np.r_[0.0, 0.0, sup2[2:]], np.r_[0.0, sup1[1:]], main])
        sys = BandedSystem(ab)
        assert sys.dimension == n
        H = _dense_banded(ab)
        b = rng.standard_normal(n)
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)

    def test_banded_system_multiply_vs_dense(self) -> None:
        main = np.array([4.0, 4.0, 4.0, 4.0])
        sup1 = np.array([0.0, 1.0, 1.0, 1.0])
        ab = np.vstack([np.r_[0.0, sup1[1:]], main])
        sys = BandedSystem(ab)
        H = _dense_banded(ab)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(sys.multiply(y), H @ y)

    def test_banded_system_rejects_non_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D"):
            BandedSystem(np.array([1.0, 2.0]))

    # ----- DenseSystem -----

    def test_dense_system_solve_vs_numpy(self) -> None:
        M = 6
        H = _random_spd(M, seed=42)
        b = np.arange(1.0, M + 1)
        sys = DenseSystem(H)
        assert sys.dimension == M
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)

    def test_dense_system_multiply(self) -> None:
        H = _random_spd(4, seed=7)
        sys = DenseSystem(H)
        y = np.arange(1.0, 5)
        np.testing.assert_allclose(sys.multiply(y), H @ y)

    def test_dense_system_rejects_non_square(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            DenseSystem(np.ones((3, 4)))

    def test_dense_system_non_pd_raises_on_solve(self) -> None:
        H = -np.eye(3)
        sys = DenseSystem(H)  # construction succeeds; Cholesky is lazy
        with self.assertRaises(NewtonStepError):
            sys.solve(np.ones(3))

    def test_dense_system_multiply_without_factorize(self) -> None:
        H = -np.eye(3)  # non-PD: multiply is valid, solve would raise
        sys = DenseSystem(H)
        np.testing.assert_allclose(sys.multiply(np.ones(3)), -np.ones(3))

    # ----- LowRankUpdatedSystem -----

    def test_low_rank_update_returns_low_rank_updated_system(self) -> None:
        base = DiagonalSystem(np.array([2.0, 3.0, 5.0]))
        kappa = np.array([1.0, 0.0, -1.0])
        updated = base.low_rank_update(kappa)
        assert isinstance(updated, LowRankUpdatedSystem)
        assert updated.dimension == 3

    def test_low_rank_update_solve_vs_dense(self) -> None:
        eta = np.array([4.0, 5.0, 6.0, 7.0])
        base = DiagonalSystem(eta)
        rng = np.random.default_rng(99)
        kappa = rng.standard_normal((4, 2))
        d = np.array([1.0, -0.5])
        updated = base.low_rank_update(kappa, d)
        H = np.diag(eta) + kappa @ np.diag(d) @ kappa.T
        b = rng.standard_normal(4)
        np.testing.assert_allclose(updated.solve(b), np.linalg.solve(H, b), rtol=1e-10)

    def test_low_rank_update_multiply_vs_dense(self) -> None:
        eta = np.array([1.0, 2.0, 3.0])
        base = DiagonalSystem(eta)
        kappa = np.array([1.0, 1.0, 1.0])
        updated = base.low_rank_update(kappa)
        H = np.diag(eta) + np.outer(kappa, kappa)
        y = np.array([1.0, 0.0, -1.0])
        np.testing.assert_allclose(updated.multiply(y), H @ y)

    def test_low_rank_update_rank_one_matches_dense_and_rank_p(self) -> None:
        # A single update (p == 1) takes the rank-one Sherman-Morrison path; it must
        # agree with both the dense reference and the general rank-p Woodbury path,
        # across a unit update, a weighted update, and a downdate.
        eta = np.array([4.0, 5.0, 6.0, 7.0])
        base = DiagonalSystem(eta)
        rng = np.random.default_rng(123)
        kappa = rng.standard_normal((4, 1))  # p == 1
        for d in (None, np.array([1.5]), np.array([-0.25])):
            updated = base.low_rank_update(kappa, d)
            dval = 1.0 if d is None else float(d[0])
            H = np.diag(eta) + dval * np.outer(kappa[:, 0], kappa[:, 0])
            b = rng.standard_normal(4)
            np.testing.assert_allclose(
                updated.solve(b), np.linalg.solve(H, b), rtol=1e-10
            )
            np.testing.assert_allclose(updated.multiply(b), H @ b, rtol=1e-10)
            np.testing.assert_allclose(
                updated.solve(b),
                solve_rank_p_update(b, kappa, base.solve, d=d),
                rtol=1e-10,
            )

    def test_low_rank_update_normalizes_scalar_d(self) -> None:
        base = DiagonalSystem(np.array([1.0, 1.0, 1.0]))
        kappa = np.array([1.0, 0.0, 0.0])
        updated = base.low_rank_update(kappa, d=2.0)
        H = np.eye(3) + 2.0 * np.outer(kappa, kappa)
        b = np.array([3.0, 0.0, 0.0])
        np.testing.assert_allclose(updated.solve(b), np.linalg.solve(H, b))

    def test_low_rank_update_rejects_wrong_dimension(self) -> None:
        base = DiagonalSystem(np.ones(3))
        with self.assertRaisesRegex(ValueError, "rows"):
            base.low_rank_update(np.ones(4))

    def test_low_rank_update_coalesces(self) -> None:
        base = DiagonalSystem(np.array([2.0, 3.0, 5.0, 7.0]))
        kappa1 = np.array([[1.0], [2.0], [0.0], [1.0]])
        kappa2 = np.array([[0.0], [1.0], [1.0], [0.0]])
        chained = base.low_rank_update(kappa1).low_rank_update(kappa2)
        assert isinstance(chained, LowRankUpdatedSystem)
        # The chained update's base should be the original DiagonalSystem, not a
        # nested LowRankUpdatedSystem -- single Woodbury, not two.
        # _base is private but checked here for coalescing.
        assert chained._base is base
        # Numerical correctness vs the nested form.
        nested_H = np.diag([2.0, 3.0, 5.0, 7.0]) + kappa1 @ kappa1.T + kappa2 @ kappa2.T
        b = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            chained.solve(b), np.linalg.solve(nested_H, b), rtol=1e-10
        )

    def test_low_rank_coalesce_mixes_unit_and_weighted_d(self) -> None:
        base = DiagonalSystem(np.array([10.0, 10.0, 10.0]))
        kappa1 = np.array([[1.0], [0.0], [0.0]])  # d=None => unit
        kappa2 = np.array([[0.0], [1.0], [0.0]])
        chained = base.low_rank_update(kappa1).low_rank_update(kappa2, d=2.0)
        H = (
            np.diag([10.0, 10.0, 10.0])
            + np.outer(kappa1[:, 0], kappa1[:, 0])
            + 2.0 * np.outer(kappa2[:, 0], kappa2[:, 0])
        )
        b = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(chained.solve(b), np.linalg.solve(H, b), rtol=1e-10)

    def test_low_rank_update_nested_matches_dense(self) -> None:
        r"""``coalesce=False`` nests a fresh Woodbury layer; the result matches dense.

        Exercises the intended mixed-inertia layering: positive-``d`` updates
        coalesced onto the SPD base, then a negative-``d`` downdate nested on top.
        """
        M = 5
        rng = np.random.default_rng(303)
        eta = rng.uniform(3.0, 6.0, size=M)
        kappa_pos = rng.standard_normal((M, 2))
        kappa_neg = rng.standard_normal((M, 1))
        d_pos = np.array([1.0, 0.5])
        d_neg = np.array([-0.1])  # small downdate -- keeps H positive definite

        system = (
            DiagonalSystem(eta)
            .low_rank_update(kappa_pos, d_pos)
            .low_rank_update(kappa_neg, d_neg, coalesce=False)
        )

        H_dense = (
            np.diag(eta)
            + kappa_pos @ np.diag(d_pos) @ kappa_pos.T
            + kappa_neg @ np.diag(d_neg) @ kappa_neg.T
        )
        b = rng.standard_normal(M)
        np.testing.assert_allclose(
            system.solve(b), np.linalg.solve(H_dense, b), rtol=1e-9
        )
        np.testing.assert_allclose(system.multiply(b), H_dense @ b, rtol=1e-9)

    def test_low_rank_update_nested_matches_coalesced(self) -> None:
        r"""Nesting and coalescing represent the same operator (all-positive updates).

        The structural assertions confirm ``coalesce=False`` actually nests a second
        layer rather than merging, while the numerical checks confirm both forms
        solve and multiply identically.
        """
        M = 4
        rng = np.random.default_rng(404)
        eta = rng.uniform(1.0, 3.0, size=M)
        k1 = rng.standard_normal((M, 1))
        k2 = rng.standard_normal((M, 2))
        base = DiagonalSystem(eta)

        coalesced = base.low_rank_update(k1).low_rank_update(k2)
        nested = base.low_rank_update(k1).low_rank_update(k2, coalesce=False)

        # Coalescing merges into one layer over the DiagonalSystem; nesting stacks
        # a second LowRankUpdatedSystem on top of the first.
        assert isinstance(coalesced, LowRankUpdatedSystem)
        assert isinstance(coalesced._base, DiagonalSystem)
        assert isinstance(nested, LowRankUpdatedSystem)
        assert isinstance(nested._base, LowRankUpdatedSystem)

        b = rng.standard_normal(M)
        np.testing.assert_allclose(coalesced.solve(b), nested.solve(b), rtol=1e-9)
        np.testing.assert_allclose(coalesced.multiply(b), nested.multiply(b), rtol=1e-9)

    # ----- BlockDiagonalSystem -----

    def test_block_diagonal_dimension_sums_blocks(self) -> None:
        sys = BlockDiagonalSystem(
            [DiagonalSystem(np.ones(2)), DiagonalSystem(np.ones(3))]
        )
        assert sys.dimension == 5

    def test_block_diagonal_rejects_empty_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            BlockDiagonalSystem([])

    def test_block_diagonal_solve_vs_dense(self) -> None:
        b1 = DiagonalSystem(np.array([2.0, 4.0]))
        b2 = DenseSystem(_random_spd(3, seed=12))
        sys = BlockDiagonalSystem([b1, b2])
        H = linalg.block_diag(np.diag(b1.eta), b2._matrix)
        b = np.arange(1.0, 6.0)
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)

    def test_block_diagonal_mixed_block_types(self) -> None:
        # diagonal + banded + dense in one block-diagonal system.
        d_block = DiagonalSystem(np.array([2.0, 3.0]))
        main = np.array([4.0, 4.0, 4.0])
        sup1 = np.array([0.0, 1.0, 1.0])
        ab = np.vstack([np.r_[0.0, sup1[1:]], main])
        band_block = BandedSystem(ab)
        dense_block = DenseSystem(_random_spd(2, seed=3))
        sys = BlockDiagonalSystem([d_block, band_block, dense_block])

        H = linalg.block_diag(
            np.diag(d_block.eta),
            _dense_banded(ab),
            dense_block._matrix,
        )
        rng = np.random.default_rng(8)
        b = rng.standard_normal(sys.dimension)
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)
        y = rng.standard_normal(sys.dimension)
        np.testing.assert_allclose(sys.multiply(y), H @ y, rtol=1e-10)

    # ----- ArrowSystem -----

    def test_arrow_scalar_diagonal_solve_vs_dense(self) -> None:
        eta = np.array([2.0, 3.0, 5.0])
        border = np.array([1.0, 2.0, -1.0])
        corner = 6.0
        sys = ArrowSystem(DiagonalSystem(eta), border, corner)
        H = np.block(
            [
                [np.diag(eta), border.reshape(-1, 1)],
                [border.reshape(1, -1), np.array([[corner]])],
            ]
        )
        b = np.array([1.0, 0.0, -1.0, 2.0])
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)
        np.testing.assert_allclose(sys.multiply(b), H @ b, rtol=1e-10)

    def test_arrow_general_path_non_diagonal_upper_left(self) -> None:
        upper = DenseSystem(_random_spd(4, seed=22))
        border = np.array([1.0, 0.0, -1.0, 2.0])
        corner = 5.0
        sys = ArrowSystem(upper, border, corner)
        H = np.block(
            [
                [upper._matrix, border.reshape(-1, 1)],
                [border.reshape(1, -1), np.array([[corner]])],
            ]
        )
        rng = np.random.default_rng(2)
        b = rng.standard_normal(5)
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)

    def test_arrow_matrix_corner_path(self) -> None:
        upper = DenseSystem(_random_spd(3, seed=33))
        border = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        corner = np.array([[3.0, 0.5], [0.5, 2.0]])
        sys = ArrowSystem(upper, border, corner)
        assert sys.dimension == 5
        H = np.block([[upper._matrix, border], [border.T, corner]])
        rng = np.random.default_rng(44)
        b = rng.standard_normal(5)
        np.testing.assert_allclose(sys.solve(b), np.linalg.solve(H, b), rtol=1e-10)
        np.testing.assert_allclose(sys.multiply(b), H @ b, rtol=1e-10)

    def test_arrow_rejects_border_wrong_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows"):
            ArrowSystem(DiagonalSystem(np.ones(3)), np.ones(4), corner=1.0)

    def test_arrow_rejects_corner_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "corner"):
            ArrowSystem(
                DiagonalSystem(np.ones(3)),
                border=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                corner=1.0,
            )

    # ----- SaddleSystem -----

    def test_saddle_system_not_a_matrix_system(self) -> None:
        sys = SchurComplementSaddleSystem(
            DiagonalSystem(np.ones(3)), np.array([[1.0, 0.0, 1.0]])
        )
        assert isinstance(sys, SaddleSystem)
        assert not isinstance(sys, MatrixSystem)  # type: ignore[unreachable]

    def test_saddle_solve_returns_tuple_with_kkt_residuals(self) -> None:
        M = 4
        p = 2
        H_dense = _random_spd(M, seed=55)
        rng = np.random.default_rng(66)
        A = rng.standard_normal((p, M))
        grad = rng.standard_normal(M)

        sys = SchurComplementSaddleSystem(DenseSystem(H_dense), A)
        result = sys.solve(grad)
        assert isinstance(result, tuple)
        assert len(result) == 2
        delta_x, nu = result
        assert delta_x.shape == (M,)
        assert nu.shape == (p,)

        # KKT residuals: H @ delta_x + A^T @ nu == grad and A @ delta_x == 0.
        np.testing.assert_allclose(H_dense @ delta_x + A.T @ nu, grad, rtol=1e-9)
        np.testing.assert_allclose(A @ delta_x, np.zeros(p), atol=1e-9)

    def test_saddle_solve_lower_rhs_matches_dense_kkt(self) -> None:
        r"""``solve`` with a non-zero lower RHS matches the dense KKT solve."""
        M = 4
        p = 2
        H_dense = _random_spd(M, seed=77)
        rng = np.random.default_rng(88)
        A = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)  # non-zero equality residual (PD case)

        sys = SchurComplementSaddleSystem(DenseSystem(H_dense), A)
        delta_x, nu = sys.solve(b_top, b_bottom)

        # Dense ground truth: solve the full indefinite KKT directly.
        K = np.block([[H_dense, A.T], [A, np.zeros((p, p))]])
        sol = np.linalg.solve(K, np.concatenate([b_top, b_bottom]))
        np.testing.assert_allclose(delta_x, sol[:M], rtol=1e-8)
        np.testing.assert_allclose(nu, sol[M:], rtol=1e-8)

        # KKT residuals: H delta_x + A^T nu == b_top and A delta_x == b_bottom.
        np.testing.assert_allclose(H_dense @ delta_x + A.T @ nu, b_top, rtol=1e-9)
        np.testing.assert_allclose(A @ delta_x, b_bottom, rtol=1e-9)

    def test_saddle_solve_explicit_zero_lower_rhs_matches_default(self) -> None:
        r"""``solve(b_top, 0)`` agrees with the default ``solve(b_top)``."""
        M = 5
        p = 2
        H_dense = _random_spd(M, seed=99)
        rng = np.random.default_rng(101)
        A = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)

        sys = SchurComplementSaddleSystem(DenseSystem(H_dense), A)
        dx_explicit, nu_explicit = sys.solve(b_top, np.zeros(p))
        dx_default, nu_default = sys.solve(b_top)
        np.testing.assert_allclose(dx_explicit, dx_default, rtol=1e-9)
        np.testing.assert_allclose(nu_explicit, nu_default, rtol=1e-9)

    def test_saddle_solve_lower_rhs_diagonal_plus_low_rank_hessian(self) -> None:
        r"""The H block can be any MatrixSystem -- here diagonal-plus-low-rank.

        Confirms the KKT-from-MatrixSystem seam: a Hessian assembled the way
        ``build_hessian_system`` produces it (a ``DiagonalSystem`` with a low-rank
        update) drops into the saddle and solves against the dense KKT.
        """
        M = 5
        p = 2
        rng = np.random.default_rng(202)
        eta = rng.uniform(1.0, 4.0, size=M)
        kappa = rng.standard_normal((M, 1))
        hessian = DiagonalSystem(eta).low_rank_update(kappa)
        A = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)

        sys = SchurComplementSaddleSystem(hessian, A)
        delta_x, nu = sys.solve(b_top, b_bottom)

        H_dense = np.diag(eta) + kappa @ kappa.T
        K = np.block([[H_dense, A.T], [A, np.zeros((p, p))]])
        sol = np.linalg.solve(K, np.concatenate([b_top, b_bottom]))
        np.testing.assert_allclose(delta_x, sol[:M], rtol=1e-8)
        np.testing.assert_allclose(nu, sol[M:], rtol=1e-8)

    def test_saddle_rejects_a_with_wrong_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "columns"):
            SchurComplementSaddleSystem(
                DiagonalSystem(np.ones(3)), np.array([[1.0, 0.0]])
            )

    def test_saddle_solve_rejects_wrong_size_b_bottom(self) -> None:
        """A ``b_bottom`` whose length != the number of equality rows is rejected."""
        M, p = 4, 2
        sys = SchurComplementSaddleSystem(
            DenseSystem(_random_spd(M, seed=5)), np.ones((p, M))
        )
        b_bottom = np.ones(p + 1)  # wrong: should have one entry per row of A
        with self.assertRaisesRegex(ValueError, "one entry for each row"):
            sys.solve(np.ones(M), b_bottom)

    def test_saddle_rejects_a_not_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D"):
            SchurComplementSaddleSystem(
                DiagonalSystem(np.ones(3)), np.array([1.0, 0.0, 1.0])
            )

    def test_saddle_num_equality_constraints_is_a_row_count(self) -> None:
        """``num_equality_constraints`` reports the number of rows of ``A``."""
        M, p = 4, 3
        A = np.ones((p, M))
        sys = SchurComplementSaddleSystem(DenseSystem(_random_spd(M, seed=11)), A)
        assert sys.num_equality_constraints == p

    # ----- Cross-cutting: composition example -----

    def test_low_rank_update_on_block_diagonal_of_banded_solve(self) -> None:
        """Sanity-check that nontrivial composition (block-diag-of-banded plus
        a rank-1 update) solves correctly versus the dense reference."""
        main1 = np.array([3.0, 3.0])
        ab1 = np.vstack([np.array([0.0, 1.0]), main1])
        main2 = np.array([4.0, 4.0, 4.0])
        ab2 = np.vstack([np.array([0.0, 1.0, 1.0]), main2])
        base = BlockDiagonalSystem([BandedSystem(ab1), BandedSystem(ab2)])
        kappa = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        updated = base.low_rank_update(kappa, d=-0.5)

        H_dense = linalg.block_diag(
            _dense_banded(ab1), _dense_banded(ab2)
        ) - 0.5 * np.outer(kappa, kappa)
        rng = np.random.default_rng(77)
        b = rng.standard_normal(5)
        np.testing.assert_allclose(
            updated.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )

    # ----- factorization caching -----

    def test_banded_solve_caches_factor(self) -> None:
        # The banded Cholesky factor is computed on the first solve and reused on
        # the next -- the three solves of one Newton step pay one factorization.
        ab = np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]])
        sys = BandedSystem(ab)
        H = _densify(sys)
        with patch.object(
            systems_module, "factor_banded", wraps=systems_module.factor_banded
        ) as spy:
            x1 = sys.solve(np.array([1.0, 2.0, 3.0]))
            x2 = sys.solve(np.array([4.0, 5.0, 6.0]))
        spy.assert_called_once()
        np.testing.assert_allclose(
            x1, np.linalg.solve(H, np.array([1.0, 2.0, 3.0])), rtol=1e-10
        )
        np.testing.assert_allclose(
            x2, np.linalg.solve(H, np.array([4.0, 5.0, 6.0])), rtol=1e-10
        )

    def test_dense_solve_caches_factor(self) -> None:
        H = _random_spd(4, seed=3)
        sys = DenseSystem(H)
        with patch.object(linalg, "cho_factor", wraps=linalg.cho_factor) as spy:
            x1 = sys.solve(np.arange(1.0, 5.0))
            x2 = sys.solve(np.arange(5.0, 9.0))
        spy.assert_called_once()
        np.testing.assert_allclose(
            x1, np.linalg.solve(H, np.arange(1.0, 5.0)), rtol=1e-10
        )
        np.testing.assert_allclose(
            x2, np.linalg.solve(H, np.arange(5.0, 9.0)), rtol=1e-10
        )

    def test_block_diagonal_solve_caches_block_factors(self) -> None:
        # Block-diagonal caching falls out of the blocks' own caches: each banded
        # block factors once and is reused across both solves (2 blocks -> 2 calls).
        blocks: list[MatrixSystem] = [
            BandedSystem(np.array([[0.0, -1.0], [3.0, 3.0]])),
            BandedSystem(np.array([[0.0, -1.0, -1.0], [5.0, 5.0, 5.0]])),
        ]
        sys = BlockDiagonalSystem(blocks)
        H = _densify(sys)
        with patch.object(
            systems_module, "factor_banded", wraps=systems_module.factor_banded
        ) as spy:
            x1 = sys.solve(np.arange(1.0, 6.0))
            x2 = sys.solve(np.arange(6.0, 11.0))
        assert spy.call_count == 2
        np.testing.assert_allclose(
            x1, np.linalg.solve(H, np.arange(1.0, 6.0)), rtol=1e-10
        )
        # The second solve reuses the cached factors and must still be correct.
        np.testing.assert_allclose(
            x2, np.linalg.solve(H, np.arange(6.0, 11.0)), rtol=1e-10
        )

    def test_low_rank_p_solve_caches_woodbury_prep(self) -> None:
        # The Woodbury prep (xi + the factored capacitance) is built once and reused
        # across solves of the same instance -- the three solves of one Newton step.
        eta = np.array([4.0, 5.0, 6.0, 7.0])
        base = DiagonalSystem(eta)
        rng = np.random.default_rng(31)
        kappa = rng.standard_normal((4, 2))
        d = np.array([1.0, -0.5])
        sys = base.low_rank_update(kappa, d)
        H = np.diag(eta) + kappa @ np.diag(d) @ kappa.T
        with patch.object(
            systems_module,
            "_prepare_rank_p_update",
            wraps=systems_module._prepare_rank_p_update,
        ) as spy:
            b1 = rng.standard_normal(4)
            b2 = rng.standard_normal(4)
            x1 = sys.solve(b1)
            x2 = sys.solve(b2)
        spy.assert_called_once()
        np.testing.assert_allclose(x1, np.linalg.solve(H, b1), rtol=1e-10)
        np.testing.assert_allclose(x2, np.linalg.solve(H, b2), rtol=1e-10)

    def test_low_rank_one_solve_caches_prep(self) -> None:
        base = DiagonalSystem(np.array([3.0, 4.0, 5.0]))
        kappa = np.array([[1.0], [1.0], [0.0]])
        sys = base.low_rank_update(kappa, d=np.array([2.0]))
        H = np.diag([3.0, 4.0, 5.0]) + 2.0 * (kappa @ kappa.T)
        with patch.object(
            systems_module,
            "_prepare_rank_one_update",
            wraps=systems_module._prepare_rank_one_update,
        ) as spy:
            b1 = np.array([1.0, 2.0, 3.0])
            b2 = np.array([3.0, 2.0, 1.0])
            x1 = sys.solve(b1)
            x2 = sys.solve(b2)
        spy.assert_called_once()
        np.testing.assert_allclose(x1, np.linalg.solve(H, b1), rtol=1e-10)
        np.testing.assert_allclose(x2, np.linalg.solve(H, b2), rtol=1e-10)

    def test_low_rank_on_block_banded_caches_prep_cdm_shape(self) -> None:
        # The CDM Hessian shape: a block-diagonal-of-banded base with a rank-p
        # downdate. One prep (capacitance factor) + the blocks' own factor caches
        # mean the whole structured solve factors once and applies three times.
        ab1 = np.array([[0.0, 1.0], [3.0, 3.0]])
        ab2 = np.array([[0.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
        base = BlockDiagonalSystem([BandedSystem(ab1), BandedSystem(ab2)])
        rng = np.random.default_rng(53)
        kappa = 0.2 * rng.standard_normal((5, 3))
        d = -np.ones(3)
        sys = base.low_rank_update(kappa, d)
        H = linalg.block_diag(_dense_banded(ab1), _dense_banded(ab2)) - kappa @ kappa.T
        with (
            patch.object(
                systems_module,
                "_prepare_rank_p_update",
                wraps=systems_module._prepare_rank_p_update,
            ) as prep_spy,
            patch.object(
                systems_module, "factor_banded", wraps=systems_module.factor_banded
            ) as factor_spy,
        ):
            b1 = rng.standard_normal(5)
            b2 = rng.standard_normal(5)
            x1 = sys.solve(b1)
            x2 = sys.solve(b2)
        prep_spy.assert_called_once()
        assert factor_spy.call_count == 2  # two banded blocks, factored once each
        np.testing.assert_allclose(x1, np.linalg.solve(H, b1), rtol=1e-9)
        np.testing.assert_allclose(x2, np.linalg.solve(H, b2), rtol=1e-9)

    # ----- scaled -----

    def test_scaled_diagonal(self) -> None:
        _assert_scaled_matches(DiagonalSystem(np.array([2.0, 3.0, 5.0, 7.0])), t=2.5)

    def test_scaled_diagonal_carries_stable_eta_inverse(self) -> None:
        # The supplied stable reciprocal scales by 1/t rather than being recomputed.
        s = np.array([1e-7, 2e-7, 3e-7])
        sys = DiagonalSystem(1.0 / (s * s), eta_inverse=s * s)
        scaled = sys.scaled(4.0)
        np.testing.assert_array_equal(scaled.solve(np.ones(3)), (s * s) / 4.0)

    def test_scaled_banded(self) -> None:
        # Tridiagonal SPD in upper compact storage: row 0 superdiagonal, row 1 main.
        ab = np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]])
        _assert_scaled_matches(BandedSystem(ab), t=3.0)

    def test_scaled_dense(self) -> None:
        _assert_scaled_matches(DenseSystem(_random_spd(4, seed=1)), t=1.5)

    def test_scaled_low_rank_updated_unit_weight(self) -> None:
        base = DiagonalSystem(np.array([2.0, 3.0, 4.0]))
        kappa = np.array([[1.0], [0.5], [0.25]])
        _assert_scaled_matches(base.low_rank_update(kappa), t=2.0)

    def test_scaled_low_rank_updated_downdate(self) -> None:
        # A negative-weight (downdate) low-rank term must scale correctly too.
        base = DiagonalSystem(np.array([10.0, 10.0, 10.0]))
        kappa = np.array([[1.0], [1.0], [0.0]])
        _assert_scaled_matches(base.low_rank_update(kappa, d=np.array([-1.0])), t=3.0)

    def test_scaled_block_diagonal(self) -> None:
        blocks: list[MatrixSystem] = [
            DiagonalSystem(np.array([2.0, 3.0])),
            DenseSystem(_random_spd(2, seed=2)),
        ]
        _assert_scaled_matches(BlockDiagonalSystem(blocks), t=2.0)

    def test_scaled_arrow(self) -> None:
        upper = DiagonalSystem(np.array([3.0, 4.0, 5.0]))
        border = np.array([1.0, 1.0, 1.0])
        _assert_scaled_matches(ArrowSystem(upper, border, corner=6.0), t=2.0)

    # ----- plus -----

    def test_plus_arrow_absorbs_diagonal(self) -> None:
        """An arrow folds a full-dimension diagonal into its interior and corner."""
        eta = np.array([2.0, 3.0, 5.0])
        border = np.array([1.0, 2.0, -1.0])
        arrow = ArrowSystem(DiagonalSystem(eta), border, corner=6.0)
        add = np.array([0.5, 1.0, 1.5, 2.0])
        diag = DiagonalSystem(add)
        H = _arrow_dense(eta, border, 6.0) + np.diag(add)
        b = np.array([1.0, 0.0, -1.0, 2.0])
        for combined in (arrow.plus(diag), diag.plus(arrow)):  # commutative
            assert isinstance(combined, ArrowSystem)
            np.testing.assert_allclose(
                combined.solve(b), np.linalg.solve(H, b), rtol=1e-10
            )
            np.testing.assert_allclose(combined.multiply(b), H @ b, rtol=1e-10)

    def test_plus_arrow_absorbs_aligned_block_diagonal(self) -> None:
        """An arrow folds a block-diagonal aligned with its interior/corner split."""
        eta = np.array([2.0, 3.0, 5.0])
        border = np.array([1.0, 2.0, -1.0])
        arrow = ArrowSystem(DiagonalSystem(eta), border, corner=6.0)
        # Interior block (dim 3) + corner block (dim 1); boundary aligned at M = 3.
        block_diag = BlockDiagonalSystem(
            [DiagonalSystem(np.array([0.5, 1.0, 1.5])), DiagonalSystem(np.array([2.0]))]
        )
        H = _arrow_dense(eta, border, 6.0) + np.diag([0.5, 1.0, 1.5, 2.0])
        b = np.array([1.0, 0.0, -1.0, 2.0])
        for combined in (arrow.plus(block_diag), block_diag.plus(arrow)):
            assert isinstance(combined, ArrowSystem)
            np.testing.assert_allclose(
                combined.solve(b), np.linalg.solve(H, b), rtol=1e-10
            )
            np.testing.assert_allclose(combined.multiply(b), H @ b, rtol=1e-10)

    def test_plus_arrow_rejects_misaligned_block_diagonal(self) -> None:
        """A block boundary that crosses the interior/corner split cannot be absorbed."""
        arrow = ArrowSystem(
            DiagonalSystem(np.ones(3)), np.array([1.0, 1.0, 1.0]), corner=2.0
        )
        # Blocks of dim 2 + 2 straddle the M = 3 interior/corner boundary.
        block_diag = BlockDiagonalSystem(
            [DiagonalSystem(np.ones(2)), DiagonalSystem(np.ones(2))]
        )
        with self.assertRaises(IncompatibleStructureError):
            arrow.plus(block_diag)

    def test_plus_arrow_arrow(self) -> None:
        """Two arrows over the same partition add block by block."""
        eta_a, border_a, corner_a = (
            np.array([2.0, 3.0, 5.0]),
            np.array([1.0, 2.0, -1.0]),
            6.0,
        )
        eta_b, border_b, corner_b = (
            np.array([1.0, 1.0, 1.0]),
            np.array([0.5, -0.5, 1.0]),
            2.0,
        )
        a = ArrowSystem(DiagonalSystem(eta_a), border_a, corner=corner_a)
        b = ArrowSystem(DiagonalSystem(eta_b), border_b, corner=corner_b)
        combined = a.plus(b)
        assert isinstance(combined, ArrowSystem)
        H = _arrow_dense(eta_a, border_a, corner_a) + _arrow_dense(
            eta_b, border_b, corner_b
        )
        rhs = np.array([1.0, 0.0, -1.0, 2.0])
        np.testing.assert_allclose(
            combined.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-10
        )
        np.testing.assert_allclose(combined.multiply(rhs), H @ rhs, rtol=1e-10)

    def test_plus_arrow_rejects_dense(self) -> None:
        """An arrow cannot absorb a dense summand (its curvature crosses the border)."""
        arrow = ArrowSystem(
            DiagonalSystem(np.ones(3)), np.array([1.0, 1.0, 1.0]), corner=2.0
        )
        dense = DenseSystem(_random_spd(4, seed=7))
        with self.assertRaises(IncompatibleStructureError):
            arrow.plus(dense)

    def test_plus_diagonal_diagonal(self) -> None:
        a = DiagonalSystem(np.array([2.0, 3.0, 5.0]))
        b = DiagonalSystem(np.array([1.0, 1.0, 1.0]))
        combined = a.plus(b)
        assert isinstance(combined, DiagonalSystem)
        np.testing.assert_allclose(_densify(combined), np.diag([3.0, 4.0, 6.0]))

    def test_plus_diagonal_keeps_stable_reciprocal(self) -> None:
        # Two barrier-style diagonals; the sum keeps a stable 1/(a+b) factored from
        # the dominant term rather than round-tripping through a huge reciprocal.
        s = np.array([1e-8, 1e-8])
        big = DiagonalSystem(1.0 / (s * s), eta_inverse=s * s)
        small = DiagonalSystem(np.array([2.0, 2.0]))
        combined = big.plus(small)
        H = np.diag(1.0 / (s * s) + 2.0)
        np.testing.assert_allclose(
            combined.solve(np.ones(2)), np.linalg.solve(H, np.ones(2)), rtol=1e-10
        )

    def test_plus_with_low_rank_operand(self) -> None:
        a = DiagonalSystem(np.array([4.0, 5.0, 6.0]))
        kappa = np.array([[1.0], [1.0], [0.0]])
        b = DiagonalSystem(np.array([1.0, 1.0, 1.0])).low_rank_update(
            kappa, d=np.array([2.0])
        )
        combined = a.plus(b)
        H = np.diag([5.0, 6.0, 7.0]) + 2.0 * (kappa @ kappa.T)
        rng = np.random.default_rng(3)
        rhs = rng.standard_normal(3)
        np.testing.assert_allclose(
            combined.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-10
        )
        np.testing.assert_allclose(_densify(combined), H, rtol=1e-10, atol=1e-12)

    def test_plus_both_low_rank_nests_mixed_inertia(self) -> None:
        # An up-date (d>0) and a down-date (d<0) must stay separate Woodbury layers.
        a = DiagonalSystem(np.array([10.0, 10.0])).low_rank_update(
            np.array([[1.0], [0.0]]), d=np.array([3.0])
        )
        b = DiagonalSystem(np.array([1.0, 1.0])).low_rank_update(
            np.array([[0.0], [1.0]]), d=np.array([-0.5])
        )
        combined = a.plus(b)
        H = (
            np.diag([11.0, 11.0])
            + 3.0 * np.outer([1.0, 0.0], [1.0, 0.0])
            - 0.5 * np.outer([0.0, 1.0], [0.0, 1.0])
        )
        rng = np.random.default_rng(5)
        rhs = rng.standard_normal(2)
        np.testing.assert_allclose(
            combined.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-10
        )

    def test_plus_banded_banded_same_bandwidth(self) -> None:
        # Two tridiagonals (upper storage) add band-by-band into a tridiagonal.
        a = BandedSystem(np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]]))
        b = BandedSystem(np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]))
        combined = a.plus(b)
        assert isinstance(combined, BandedSystem)
        np.testing.assert_allclose(
            _densify(combined), _densify(a) + _densify(b), rtol=1e-12
        )

    def test_plus_banded_widens_to_max_bandwidth(self) -> None:
        # A pentadiagonal absorbs a tridiagonal; the result keeps the wider band.
        penta = BandedSystem(
            np.array(
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 1.0, 1.0, 1.0],
                    [8.0, 8.0, 8.0, 8.0],
                ]
            )
        )
        tri = BandedSystem(np.array([[0.0, -1.0, -1.0, -1.0], [2.0, 2.0, 2.0, 2.0]]))
        combined = penta.plus(tri)
        assert isinstance(combined, BandedSystem)
        assert combined.dimension == 4
        np.testing.assert_allclose(
            _densify(combined), _densify(penta) + _densify(tri), rtol=1e-12
        )

    def test_plus_banded_absorbs_diagonal_as_width_zero_band(self) -> None:
        banded = BandedSystem(np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]]))
        diag = DiagonalSystem(np.array([1.0, 2.0, 3.0]))
        combined = banded.plus(diag)
        assert isinstance(combined, BandedSystem)
        np.testing.assert_allclose(
            _densify(combined), _densify(banded) + np.diag([1.0, 2.0, 3.0]), rtol=1e-12
        )

    def test_plus_banded_lower_storage(self) -> None:
        # CDM stores blocks lower-triangular; banded+banded and banded+diagonal must
        # both compose in that convention.
        a = BandedSystem(np.array([[4.0, 4.0, 4.0], [-1.0, -1.0, 0.0]]), lower=True)
        b = BandedSystem(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 0.0]]), lower=True)
        combined = a.plus(b)
        assert isinstance(combined, BandedSystem)
        np.testing.assert_allclose(
            _densify(combined), _densify(a) + _densify(b), rtol=1e-12
        )
        with_diag = a.plus(DiagonalSystem(np.ones(3)))
        np.testing.assert_allclose(
            _densify(with_diag), _densify(a) + np.eye(3), rtol=1e-12
        )

    def test_plus_banded_different_triangles_raise(self) -> None:
        upper = BandedSystem(
            np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]]), lower=False
        )
        lower = BandedSystem(np.array([[4.0, 4.0, 4.0], [-1.0, -1.0, 0.0]]), lower=True)
        with self.assertRaisesRegex(IncompatibleStructureError, "triangle"):
            upper.plus(lower)

    def test_plus_banded_rejects_dense(self) -> None:
        banded = BandedSystem(np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]]))
        with self.assertRaises(IncompatibleStructureError):
            banded.plus(DenseSystem(2.0 * np.eye(3)))

    def test_plus_block_diagonal_matching_partition(self) -> None:
        def make(scale: float) -> BlockDiagonalSystem:
            return BlockDiagonalSystem(
                [
                    DiagonalSystem(np.array([2.0, 3.0]) * scale),
                    BandedSystem(
                        np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]]) * scale
                    ),
                ]
            )

        a = make(1.0)
        b = make(0.5)
        combined = a.plus(b)
        assert isinstance(combined, BlockDiagonalSystem)
        np.testing.assert_allclose(
            _densify(combined), _densify(a) + _densify(b), rtol=1e-12
        )

    def test_plus_block_diagonal_of_banded_absorbs_diagonal(self) -> None:
        # The CDM-shaped case: a block-diagonal of banded blocks (the objective
        # curvature) plus a diagonal (a bounds layer) folds the diagonal into each
        # block's main diagonal, preserving block-diagonal-of-banded structure.
        block0 = BandedSystem(np.array([[0.0, -1.0], [3.0, 3.0]]))
        block1 = BandedSystem(np.array([[0.0, -1.0, -1.0], [5.0, 5.0, 5.0]]))
        base = BlockDiagonalSystem([block0, block1])
        diag = DiagonalSystem(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        combined = base.plus(diag)
        assert isinstance(combined, BlockDiagonalSystem)
        np.testing.assert_allclose(
            _densify(combined),
            _densify(base) + np.diag([1.0, 2.0, 3.0, 4.0, 5.0]),
            rtol=1e-12,
        )
        # The blocks stay banded -- the diagonal did not densify them.
        rng = np.random.default_rng(11)
        rhs = rng.standard_normal(5)
        np.testing.assert_allclose(
            combined.solve(rhs), np.linalg.solve(_densify(combined), rhs), rtol=1e-10
        )

    def test_plus_block_diagonal_mismatched_partition_raises(self) -> None:
        a = BlockDiagonalSystem(
            [DiagonalSystem(np.array([2.0, 3.0])), DiagonalSystem(np.array([4.0]))]
        )
        b = BlockDiagonalSystem(
            [DiagonalSystem(np.array([1.0])), DiagonalSystem(np.array([1.0, 1.0]))]
        )
        with self.assertRaisesRegex(IncompatibleStructureError, "partition"):
            a.plus(b)

    def test_plus_block_diagonal_rejects_banded(self) -> None:
        block = BlockDiagonalSystem(
            [DiagonalSystem(np.array([2.0, 3.0])), DiagonalSystem(np.array([4.0, 5.0]))]
        )
        banded = BandedSystem(np.array([[0.0, -1.0, -1.0, -1.0], [4.0, 4.0, 4.0, 4.0]]))
        with self.assertRaises(IncompatibleStructureError):
            block.plus(banded)

    def test_plus_dense_absorbs_diagonal_either_side(self) -> None:
        diag = DiagonalSystem(np.array([2.0, 3.0]))
        dense = DenseSystem(np.array([[2.0, 1.0], [1.0, 2.0]]))
        expected = np.array([[4.0, 1.0], [1.0, 5.0]])
        # A dense matrix folds a diagonal into its main diagonal, from either side
        # (``plus`` tries both directions, so the structured ``_absorb`` is reached).
        v = np.array([1.0, -1.0])
        np.testing.assert_allclose(dense.plus(diag).multiply(v), expected @ v)
        np.testing.assert_allclose(diag.plus(dense).multiply(v), expected @ v)

    def test_plus_is_commutative_across_shapes(self) -> None:
        # ``_absorb`` lives on the structured operand, but ``plus`` tries both
        # directions, so a diagonal may sit on either side of a banded or
        # block-diagonal sum and give the same result.
        diag = DiagonalSystem(np.array([2.0, 3.0, 5.0]))
        banded = BandedSystem(np.array([[0.0, -1.0, -1.0], [4.0, 4.0, 4.0]]))
        forward = banded.plus(diag)
        reverse = diag.plus(banded)
        assert isinstance(forward, BandedSystem)
        assert isinstance(reverse, BandedSystem)
        np.testing.assert_allclose(_densify(forward), _densify(reverse), rtol=1e-12)

        block = BlockDiagonalSystem(
            [
                BandedSystem(np.array([[0.0, -1.0], [3.0, 3.0]])),
                DiagonalSystem(np.array([5.0, 6.0])),
            ]
        )
        diag4 = DiagonalSystem(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(
            _densify(block.plus(diag4)), _densify(diag4.plus(block)), rtol=1e-12
        )

    def test_plus_dimension_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(IncompatibleStructureError, "dimension"):
            DiagonalSystem(np.ones(2)).plus(DiagonalSystem(np.ones(3)))

    # ----- ZeroSystem -----

    def test_zero_system_multiply_is_zero(self) -> None:
        z = ZeroSystem(4)
        assert z.dimension == 4
        np.testing.assert_array_equal(z.multiply(np.arange(1.0, 5.0)), np.zeros(4))

    def test_zero_system_solve_raises(self) -> None:
        with self.assertRaises(NewtonStepError):
            ZeroSystem(3).solve(np.ones(3))

    def test_zero_system_is_plus_identity(self) -> None:
        diag = DiagonalSystem(np.array([2.0, 3.0, 5.0]))
        zero = ZeroSystem(3)
        np.testing.assert_allclose(_densify(diag.plus(zero)), _densify(diag))
        np.testing.assert_allclose(_densify(zero.plus(diag)), _densify(diag))
        assert isinstance(zero.plus(ZeroSystem(3)), ZeroSystem)

    def test_zero_system_carries_pure_low_rank_update(self) -> None:
        # An inequality layer's barrier curvature is a low-rank update on a zero
        # base; summed with a bounds diagonal it gives a solvable system.
        kappa = np.array([[1.0], [1.0], [0.0]])
        pure_low_rank = ZeroSystem(3).low_rank_update(kappa, d=np.array([2.0]))
        diag = DiagonalSystem(np.array([4.0, 5.0, 6.0]))
        combined = diag.plus(pure_low_rank)
        H = np.diag([4.0, 5.0, 6.0]) + 2.0 * (kappa @ kappa.T)
        rng = np.random.default_rng(13)
        rhs = rng.standard_normal(3)
        np.testing.assert_allclose(
            combined.solve(rhs), np.linalg.solve(H, rhs), rtol=1e-10
        )
        np.testing.assert_allclose(_densify(combined), H, rtol=1e-10, atol=1e-12)


def _dense_kkt_solve(
    H: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    b_top: npt.NDArray[np.float64],
    b_bottom: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Ground-truth KKT solve via a dense assembly of ``[[H, A^T], [A, 0]]``."""
    p, M = A.shape
    lower = np.zeros(p) if b_bottom is None else b_bottom
    K = np.block([[H, A.T], [A, np.zeros((p, p))]])
    sol = np.linalg.solve(K, np.concatenate([b_top, lower]))
    return sol[:M], sol[M:]


def _coupled_saddle_matrix(
    M_xx_dense: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    H_yy_dense: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Assemble the full indefinite coupled saddle ``[[-M_xx, B], [B^T, H_yy]]``."""
    return np.block([[-M_xx_dense, B], [B.T, H_yy_dense]])


def _bordered_coupled_saddle_matrix(
    M_xx_dense: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    H_yy_dense: npt.NDArray[np.float64],
    A_eq: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Assemble ``[[-M_xx, B, A_eq^T], [B^T, H_yy, 0], [A_eq, 0, 0]]``."""
    m = H_yy_dense.shape[0]
    p = A_eq.shape[0]
    return np.block(
        [
            [-M_xx_dense, B, A_eq.T],
            [B.T, H_yy_dense, np.zeros((m, p))],
            [A_eq, np.zeros((p, m)), np.zeros((p, p))],
        ]
    )


class ZeroWeightLowRankUpdateTest(unittest.TestCase):
    """Zero entries in ``d`` are dropped rather than rejected.

    ``kappa @ diag(d) @ kappa^T`` is unchanged by a column whose weight is
    zero, but carrying it through puts a zero on the equilibrated Woodbury
    matrix's diagonal, which the rank-p prep refuses. In the interior-point
    loop ``d = lambda / (-f)``, so a multiplier on a deeply slack constraint
    underflowing to zero is routine near the boundary.

    """

    @staticmethod
    def _base_and_kappa() -> tuple[DiagonalSystem, npt.NDArray[np.float64]]:
        """A well-conditioned diagonal base and a rank-4 update."""
        rng = np.random.default_rng(404)
        return DiagonalSystem(np.full(12, 3.0)), rng.standard_normal((12, 4))

    def test_zero_weight_column_is_dropped(self) -> None:
        """A zero weight gives the same system as omitting that column."""
        base, kappa = self._base_and_kappa()
        d = np.array([1.5, 0.0, 2.5, 0.75])

        with_zero = base.low_rank_update(kappa, d)
        without = base.low_rank_update(kappa[:, [0, 2, 3]], d[[0, 2, 3]])

        b = np.arange(12, dtype=np.float64)
        np.testing.assert_allclose(with_zero.solve(b), without.solve(b))
        np.testing.assert_allclose(with_zero.multiply(b), without.multiply(b))

    def test_zero_weight_matches_the_dense_matrix(self) -> None:
        """Ground truth: the dropped column contributes nothing to H."""
        base, kappa = self._base_and_kappa()
        d = np.array([1.5, 0.0, 2.5, 0.75])
        dense = np.diag(np.full(12, 3.0)) + kappa @ np.diag(d) @ kappa.T

        b = np.arange(12, dtype=np.float64)

        np.testing.assert_allclose(
            base.low_rank_update(kappa, d).solve(b), np.linalg.solve(dense, b)
        )

    def test_all_weights_zero_returns_the_base(self) -> None:
        """An update of entirely zero weight is the identity."""
        base, kappa = self._base_and_kappa()

        self.assertIs(base.low_rank_update(kappa, np.zeros(4)), base)

    def test_zero_weight_coalesces(self) -> None:
        """Zero columns are dropped when merging into an existing update too."""
        base, kappa = self._base_and_kappa()
        first = base.low_rank_update(kappa[:, :2], np.array([1.5, 2.0]))

        merged = first.low_rank_update(kappa[:, 2:], np.array([0.0, 0.75]))
        expected = base.low_rank_update(kappa[:, [0, 1, 3]], np.array([1.5, 2.0, 0.75]))

        b = np.arange(12, dtype=np.float64)
        np.testing.assert_allclose(merged.solve(b), expected.solve(b))

    def test_scaling_drops_weights_it_zeroes(self) -> None:
        """``t * d`` can introduce zeros that were not in ``d``."""
        base, kappa = self._base_and_kappa()
        d = np.array([1.5, 1e-320, 2.5, 0.75])
        scaled = base.low_rank_update(kappa, d).scaled(1e-10)

        # 1e-320 * 1e-10 underflows to exactly zero; the other three survive.
        b = np.arange(12, dtype=np.float64)
        expected = base.scaled(1e-10).low_rank_update(
            kappa[:, [0, 2, 3]], 1e-10 * d[[0, 2, 3]]
        )
        np.testing.assert_allclose(scaled.solve(b), expected.solve(b))

    def test_scaling_to_zero_returns_the_scaled_base(self) -> None:
        """Scaling by zero annihilates the update, so no layer remains."""
        base, kappa = self._base_and_kappa()
        scaled = base.low_rank_update(kappa, np.array([1.5, 2.0, 2.5, 0.75])).scaled(
            0.0
        )

        b = np.arange(12, dtype=np.float64)
        np.testing.assert_allclose(scaled.multiply(b), base.scaled(0.0).multiply(b))

    def test_scaling_preserves_deliberate_nesting(self) -> None:
        """Scaling a nested stack must not flatten it into one wide update.

        A downdate is nested rather than coalesced on purpose: folding
        positive- and negative-``d`` updates into one capacitance matrix is
        catastrophic near a barrier boundary. Scaling happens on every IPM
        step, so it must not undo that. The invariant is structural and the
        layering is private, so this asserts on shape rather than on numbers --
        coalescing changes the conditioning, not the matrix.

        """
        base, kappa = self._base_and_kappa()
        nested = base.low_rank_update(
            kappa[:, :2], np.array([1.5, 2.0])
        ).low_rank_update(kappa[:, 2:], np.array([-0.5, -0.25]), coalesce=False)

        scaled = nested.scaled(2.0)

        assert isinstance(scaled, LowRankUpdatedSystem)
        self.assertIsInstance(scaled._base, LowRankUpdatedSystem)

    def test_all_weights_zero_coalesces_to_the_existing_update(self) -> None:
        """Merging an all-zero update leaves the existing system untouched."""
        base, kappa = self._base_and_kappa()
        first = base.low_rank_update(kappa[:, :2], np.array([1.5, 2.0]))

        self.assertIs(first.low_rank_update(kappa[:, 2:], np.zeros(2)), first)


class PermutedSystemTest(unittest.TestCase):
    """Structure in a reordered basis, solved without the caller seeing it."""

    @staticmethod
    def _block_diagonal_under_permutation(
        n_groups: int, group_size: int, seed: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[np.ndarray]]:
        """Build ``H`` that is block diagonal only after gathering strided indices.

        Mirrors the motivating layout: a ``(group_size, n_groups)`` decision
        matrix flattened row-major, so group ``g`` owns the strided positions
        ``g, n_groups + g, 2 * n_groups + g, ...``.
        """
        rng = np.random.default_rng(seed)
        n = n_groups * group_size
        blocks = []
        H = np.zeros((n, n))
        order = np.concatenate(
            [np.arange(g, n, n_groups, dtype=np.int64) for g in range(n_groups)]
        )
        for g in range(n_groups):
            a = rng.standard_normal((group_size, group_size))
            block = a @ a.T + group_size * np.eye(group_size)
            blocks.append(block)
            idx = np.arange(g, n, n_groups)
            H[np.ix_(idx, idx)] = block
        return H, order, blocks

    def test_solve_and_multiply_match_the_dense_matrix(self) -> None:
        """The wrapper reproduces ``H`` exactly, permutation and all."""
        n_groups, group_size = 4, 5
        H, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=7
        )
        system = PermutedSystem(
            BlockDiagonalSystem([DenseSystem(b) for b in blocks]), order
        )
        rng = np.random.default_rng(11)
        b = rng.standard_normal(n_groups * group_size)

        np.testing.assert_allclose(system.multiply(b), H @ b, atol=1e-10)
        np.testing.assert_allclose(system.solve(b), np.linalg.solve(H, b), atol=1e-8)
        # And the two are genuine inverses of one another.
        np.testing.assert_allclose(system.multiply(system.solve(b)), b, atol=1e-8)

    def test_reordering_is_what_makes_the_structure_usable(self) -> None:
        # Guards the index convention: ``inner == H[ix_(order, order)]``. With
        # the permutation dropped, the same blocks describe a different matrix,
        # so a wrong or absent ordering cannot silently pass the test above.
        n_groups, group_size = 3, 4
        H, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=3
        )
        inner = BlockDiagonalSystem([DenseSystem(b) for b in blocks])
        b = np.arange(1.0, n_groups * group_size + 1.0)

        self.assertFalse(
            np.allclose(inner.multiply(b), H @ b),
            "the identity ordering must NOT reproduce H, or this fixture is "
            "not actually testing the permutation",
        )
        np.testing.assert_allclose(
            PermutedSystem(inner, order).multiply(b), H @ b, atol=1e-10
        )

    def test_scaling_and_definiteness_pass_through(self) -> None:
        n_groups, group_size = 3, 4
        H, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=5
        )
        system = PermutedSystem(
            BlockDiagonalSystem([DenseSystem(b) for b in blocks]), order
        )
        b = np.linspace(-1.0, 1.0, n_groups * group_size)

        np.testing.assert_allclose(
            system.scaled(2.5).multiply(b), 2.5 * (H @ b), atol=1e-10
        )
        # A permutation is a similarity transform, so definiteness is invariant.
        self.assertTrue(system.is_positive_definite())
        self.assertEqual(system.dimension, n_groups * group_size)

    def test_absorbs_the_barrier_diagonal(self) -> None:
        """The case an interior-point solve hits on every Newton step.

        A structured Hessian that cannot take the barrier's diagonal is
        unusable regardless of its structure, so this is load-bearing rather
        than a convenience.
        """
        n_groups, group_size = 3, 4
        H, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=13
        )
        n = n_groups * group_size
        system = PermutedSystem(
            BlockDiagonalSystem([DenseSystem(b) for b in blocks]), order
        )
        eta = np.linspace(0.5, 3.0, n)
        combined = system.plus(DiagonalSystem(eta))
        b = np.linspace(-2.0, 2.0, n)

        expected = H + np.diag(eta)
        np.testing.assert_allclose(combined.multiply(b), expected @ b, atol=1e-10)
        np.testing.assert_allclose(
            combined.solve(b), np.linalg.solve(expected, b), atol=1e-8
        )

    def test_absorbs_another_permuted_system_sharing_its_order(self) -> None:
        n_groups, group_size = 3, 4
        H, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=17
        )
        system = PermutedSystem(
            BlockDiagonalSystem([DenseSystem(b) for b in blocks]), order
        )
        b = np.linspace(1.0, 2.0, n_groups * group_size)

        doubled = system.plus(system)

        np.testing.assert_allclose(doubled.multiply(b), 2.0 * (H @ b), atol=1e-10)
        # A different ordering is not silently combined.
        other_order = np.roll(order, 1)
        with self.assertRaisesRegex(IncompatibleStructureError, "PermutedSystem"):
            system.plus(
                PermutedSystem(
                    BlockDiagonalSystem([DenseSystem(b_) for b_ in blocks]),
                    other_order,
                )
            )

    def test_matrix_right_hand_sides(self) -> None:
        """Woodbury solves with a matrix RHS, so the wrapper must accept one."""
        n_groups, group_size = 3, 4
        H, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=21
        )
        n = n_groups * group_size
        system = PermutedSystem(
            BlockDiagonalSystem([DenseSystem(b) for b in blocks]), order
        )
        rng = np.random.default_rng(23)
        B = rng.standard_normal((n, 3))

        np.testing.assert_allclose(system.multiply(B), H @ B, atol=1e-10)
        np.testing.assert_allclose(system.solve(B), np.linalg.solve(H, B), atol=1e-8)
        # A low-rank layer composes on top, which is the path that needs it.
        kappa = rng.standard_normal((n, 2))
        updated = system.low_rank_update(kappa, d=np.ones(2))
        expected = H + kappa @ kappa.T
        b = rng.standard_normal(n)
        np.testing.assert_allclose(
            updated.solve(b), np.linalg.solve(expected, b), atol=1e-8
        )

    def test_derives_the_inverse_permutation_and_rejects_a_wrong_one(self) -> None:
        n_groups, group_size = 3, 4
        _, order, blocks = self._block_diagonal_under_permutation(
            n_groups, group_size, seed=29
        )
        inner = BlockDiagonalSystem([DenseSystem(b) for b in blocks])
        b = np.linspace(0.0, 1.0, n_groups * group_size)

        derived = PermutedSystem(inner, order)
        explicit = PermutedSystem(inner, order, np.argsort(order))
        np.testing.assert_allclose(derived.solve(b), explicit.solve(b), atol=1e-12)

        with self.assertRaisesRegex(ValueError, "inverse permutation"):
            PermutedSystem(inner, order, np.roll(np.argsort(order), 1))

    def test_rejects_a_non_permutation(self) -> None:
        inner = DenseSystem(np.eye(3))
        for bad in ([0, 1], [0, 1, 3], [0, 0, 1]):
            with self.assertRaisesRegex(ValueError, "permutation"):
                PermutedSystem(inner, np.asarray(bad, dtype=np.int64))


class CoupledSaddleSystemTest(unittest.TestCase):
    """Tests for ``CoupledSaddleSystem`` (the convex-concave two-block saddle)."""

    @parameterized.expand(
        [
            ("skinny", 40, 3),
            ("square_ish", 5, 4),
            ("medium", 12, 2),
        ]
    )
    def test_solve_matches_dense_ground_truth(self, _name: str, n: int, m: int) -> None:
        r"""``solve`` recovers a known ``[dx; dy]`` from the dense saddle RHS."""
        rng = np.random.default_rng(hash((n, m)) % (2**32))
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)  # PD
        H_yy_dense = _random_spd(m, seed=n * 100 + m)  # PD
        B = rng.standard_normal((n, m))

        # Assemble the indefinite saddle, pick a true solution, form the RHS.
        K = _coupled_saddle_matrix(M_xx_dense, B, H_yy_dense)
        true_dx = rng.standard_normal(n)
        true_dy = rng.standard_normal(m)
        rhs = K @ np.concatenate([true_dx, true_dy])
        b_x = rhs[:n]
        b_y = rhs[n:]

        sys = CoupledSaddleSystem(DenseSystem(M_xx_dense), B, DenseSystem(H_yy_dense))
        delta_x, delta_y = sys.solve(b_x, b_y)

        self.assertEqual(delta_x.shape, (n,))
        self.assertEqual(delta_y.shape, (m,))
        np.testing.assert_allclose(delta_x, true_dx, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(delta_y, true_dy, rtol=1e-8, atol=1e-9)

    def test_solve_residual_is_tiny(self) -> None:
        r"""``K @ [delta_x; delta_y]`` reproduces the RHS to ~1e-10.

        Exercises the batched ``[B | b_x]`` solve and the ``Minv_B`` /
        ``Minv_bx`` split end-to-end: a correct split is the only way the
        assembled residual stays this small.
        """
        n, m = 30, 4
        rng = np.random.default_rng(4242)
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)
        H_yy_dense = _random_spd(m, seed=7)
        B = rng.standard_normal((n, m))
        K = _coupled_saddle_matrix(M_xx_dense, B, H_yy_dense)
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)

        sys = CoupledSaddleSystem(DenseSystem(M_xx_dense), B, DenseSystem(H_yy_dense))
        delta_x, delta_y = sys.solve(b_x, b_y)

        residual = K @ np.concatenate([delta_x, delta_y]) - np.concatenate([b_x, b_y])
        self.assertLess(float(np.linalg.norm(residual)), 1e-10)

    def test_structured_xx_matches_dense(self) -> None:
        r"""A structured ``M_xx`` (diagonal) gives the same solution as its dense form."""
        n, m = 20, 3
        rng = np.random.default_rng(909)
        eta = rng.uniform(1.0, 5.0, size=n)  # PD diagonal M_xx
        H_yy_dense = _random_spd(m, seed=13)
        B = rng.standard_normal((n, m))
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)

        structured = CoupledSaddleSystem(
            DiagonalSystem(eta), B, DenseSystem(H_yy_dense)
        )
        dense = CoupledSaddleSystem(
            DenseSystem(np.diag(eta)), B, DenseSystem(H_yy_dense)
        )
        dx_structured, dy_structured = structured.solve(b_x, b_y)
        dx_dense, dy_dense = dense.solve(b_x, b_y)

        np.testing.assert_allclose(dx_structured, dx_dense, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(dy_structured, dy_dense, rtol=1e-10, atol=1e-12)

    def test_assembled_saddle_has_expected_inertia(self) -> None:
        r"""The assembled ``K`` has ``n`` negative and ``m`` positive eigenvalues.

        Documents the saddle inertia the solver relies on: negative-definite
        x-block, positive-definite y-block.
        """
        n, m = 15, 4
        rng = np.random.default_rng(2718)
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)
        H_yy_dense = _random_spd(m, seed=21)
        B = rng.standard_normal((n, m))
        K = _coupled_saddle_matrix(M_xx_dense, B, H_yy_dense)

        eigenvalues = np.linalg.eigvalsh(K)
        self.assertEqual(int(np.sum(eigenvalues < 0)), n)
        self.assertEqual(int(np.sum(eigenvalues > 0)), m)

    def test_accessors_and_dimensions(self) -> None:
        r"""The read-only accessors and dimension properties report the blocks."""
        n, m = 8, 2
        xx = DenseSystem(_random_spd(n, seed=1))
        yy = DenseSystem(_random_spd(m, seed=2))
        B = np.ones((n, m))
        sys = CoupledSaddleSystem(xx, B, yy)
        self.assertIs(sys.xx_system, xx)
        self.assertIs(sys.yy_system, yy)
        np.testing.assert_array_equal(sys.coupling, B)
        self.assertEqual(sys.dimension_x, n)
        self.assertEqual(sys.dimension_y, m)

    def test_rejects_coupling_not_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D"):
            CoupledSaddleSystem(
                DiagonalSystem(np.ones(3)), np.ones(3), DiagonalSystem(np.ones(3))
            )

    def test_rejects_coupling_wrong_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows"):
            CoupledSaddleSystem(
                DiagonalSystem(np.ones(3)),
                np.ones((4, 2)),
                DiagonalSystem(np.ones(2)),
            )

    def test_rejects_coupling_wrong_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "columns"):
            CoupledSaddleSystem(
                DiagonalSystem(np.ones(3)),
                np.ones((3, 2)),
                DiagonalSystem(np.ones(5)),
            )

    def test_solve_rejects_wrong_size_b_x(self) -> None:
        n, m = 5, 2
        sys = CoupledSaddleSystem(
            DiagonalSystem(np.ones(n)), np.ones((n, m)), DiagonalSystem(np.ones(m))
        )
        with self.assertRaisesRegex(ValueError, "each row"):
            sys.solve(np.ones(n + 1), np.ones(m))

    def test_solve_rejects_wrong_size_b_y(self) -> None:
        n, m = 5, 2
        sys = CoupledSaddleSystem(
            DiagonalSystem(np.ones(n)), np.ones((n, m)), DiagonalSystem(np.ones(m))
        )
        with self.assertRaisesRegex(ValueError, "each column"):
            sys.solve(np.ones(n), np.ones(m + 1))


class EqualityCoupledSaddleSystemTest(unittest.TestCase):
    """Tests for ``EqualityCoupledSaddleSystem`` (the equality-bordered coupled saddle)."""

    @parameterized.expand(
        [
            ("skinny_one_eq", 40, 3, 1),
            ("square_ish_two_eq", 6, 4, 2),
            ("medium_one_eq", 12, 2, 1),
        ]
    )
    def test_solve_matches_dense_ground_truth(
        self, _name: str, n: int, m: int, p: int
    ) -> None:
        r"""``solve`` recovers a known ``[dx; dy; dlambda]`` from the bordered RHS."""
        rng = np.random.default_rng(hash((n, m, p)) % (2**32))
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)  # PD
        H_yy_dense = _random_spd(m, seed=n * 100 + m)  # PD
        B = rng.standard_normal((n, m))
        A_eq = rng.standard_normal((p, n))  # full row rank a.s.

        K = _bordered_coupled_saddle_matrix(M_xx_dense, B, H_yy_dense, A_eq)
        true_dx = rng.standard_normal(n)
        true_dy = rng.standard_normal(m)
        true_dl = rng.standard_normal(p)
        rhs = K @ np.concatenate([true_dx, true_dy, true_dl])
        b_x = rhs[:n]
        b_y = rhs[n : n + m]
        b_eq = rhs[n + m :]

        sys = EqualityCoupledSaddleSystem(
            DenseSystem(M_xx_dense), B, DenseSystem(H_yy_dense), A_eq
        )
        delta_x, delta_y, delta_lambda = sys.solve(b_x, b_y, b_eq)

        self.assertEqual(delta_x.shape, (n,))
        self.assertEqual(delta_y.shape, (m,))
        self.assertEqual(delta_lambda.shape, (p,))
        np.testing.assert_allclose(delta_x, true_dx, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(delta_y, true_dy, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(delta_lambda, true_dl, rtol=1e-7, atol=1e-9)

    def test_solve_residual_is_tiny(self) -> None:
        r"""``K @ [dx; dy; dlambda]`` reproduces the RHS to ~1e-9.

        Exercises the batched ``[B | A_eq^T | b_x]`` solve and the
        ``Minv_B`` / ``Minv_At`` / ``Minv_bx`` split end-to-end.
        """
        n, m, p = 30, 4, 2
        rng = np.random.default_rng(4242)
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)
        H_yy_dense = _random_spd(m, seed=7)
        B = rng.standard_normal((n, m))
        A_eq = rng.standard_normal((p, n))
        K = _bordered_coupled_saddle_matrix(M_xx_dense, B, H_yy_dense, A_eq)
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)
        b_eq = rng.standard_normal(p)

        sys = EqualityCoupledSaddleSystem(
            DenseSystem(M_xx_dense), B, DenseSystem(H_yy_dense), A_eq
        )
        delta_x, delta_y, delta_lambda = sys.solve(b_x, b_y, b_eq)

        residual = K @ np.concatenate(
            [delta_x, delta_y, delta_lambda]
        ) - np.concatenate([b_x, b_y, b_eq])
        self.assertLess(float(np.linalg.norm(residual)), 1e-9)

    def test_feasibility_row_is_satisfied(self) -> None:
        r"""The recovered ``delta_x`` satisfies the equality row ``A_eq delta_x = b_eq``."""
        n, m, p = 18, 3, 1
        rng = np.random.default_rng(1234)
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)
        H_yy_dense = _random_spd(m, seed=99)
        B = rng.standard_normal((n, m))
        A_eq = rng.standard_normal((p, n))
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)
        b_eq = rng.standard_normal(p)

        sys = EqualityCoupledSaddleSystem(
            DenseSystem(M_xx_dense), B, DenseSystem(H_yy_dense), A_eq
        )
        delta_x, _, _ = sys.solve(b_x, b_y, b_eq)
        np.testing.assert_allclose(A_eq @ delta_x, b_eq, rtol=1e-8, atol=1e-9)

    def test_structured_xx_matches_dense(self) -> None:
        r"""A structured ``M_xx`` (diagonal) gives the same solution as its dense form."""
        n, m, p = 20, 3, 1
        rng = np.random.default_rng(909)
        eta = rng.uniform(1.0, 5.0, size=n)  # PD diagonal M_xx
        H_yy_dense = _random_spd(m, seed=13)
        B = rng.standard_normal((n, m))
        A_eq = rng.standard_normal((p, n))
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)
        b_eq = rng.standard_normal(p)

        structured = EqualityCoupledSaddleSystem(
            DiagonalSystem(eta), B, DenseSystem(H_yy_dense), A_eq
        )
        dense = EqualityCoupledSaddleSystem(
            DenseSystem(np.diag(eta)), B, DenseSystem(H_yy_dense), A_eq
        )
        dx_s, dy_s, dl_s = structured.solve(b_x, b_y, b_eq)
        dx_d, dy_d, dl_d = dense.solve(b_x, b_y, b_eq)

        np.testing.assert_allclose(dx_s, dx_d, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(dy_s, dy_d, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(dl_s, dl_d, rtol=1e-10, atol=1e-12)

    def test_accepts_sparse_equality(self) -> None:
        r"""A scipy-sparse ``A_eq`` is densified and matches the dense-``A_eq`` solve."""
        n, m, p = 15, 3, 2
        rng = np.random.default_rng(321)
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)
        H_yy_dense = _random_spd(m, seed=8)
        B = rng.standard_normal((n, m))
        A_eq_dense = rng.standard_normal((p, n))
        b_x = rng.standard_normal(n)
        b_y = rng.standard_normal(m)
        b_eq = rng.standard_normal(p)

        dense = EqualityCoupledSaddleSystem(
            DenseSystem(M_xx_dense), B, DenseSystem(H_yy_dense), A_eq_dense
        )
        sparse = EqualityCoupledSaddleSystem(
            DenseSystem(M_xx_dense),
            B,
            DenseSystem(H_yy_dense),
            scipy.sparse.csr_matrix(A_eq_dense),
        )
        dx_d, dy_d, dl_d = dense.solve(b_x, b_y, b_eq)
        dx_s, dy_s, dl_s = sparse.solve(b_x, b_y, b_eq)
        np.testing.assert_allclose(dx_s, dx_d, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(dy_s, dy_d, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(dl_s, dl_d, rtol=1e-10, atol=1e-12)

    def test_assembled_saddle_has_expected_inertia(self) -> None:
        r"""The bordered ``K`` has ``n`` negative and ``m + p`` positive eigenvalues.

        The equality border contributes ``p`` positive eigenvalues (the
        reduced ``(y, lambda)`` complement is PD), matching the class's claim.
        """
        n, m, p = 15, 4, 2
        rng = np.random.default_rng(2718)
        A = rng.standard_normal((n, n))
        M_xx_dense = A @ A.T + 1e-2 * np.eye(n)
        H_yy_dense = _random_spd(m, seed=21)
        B = rng.standard_normal((n, m))
        A_eq = rng.standard_normal((p, n))
        K = _bordered_coupled_saddle_matrix(M_xx_dense, B, H_yy_dense, A_eq)

        eigenvalues = np.linalg.eigvalsh(K)
        self.assertEqual(int(np.sum(eigenvalues < 0)), n)
        self.assertEqual(int(np.sum(eigenvalues > 0)), m + p)

    def test_accessors_and_dimensions(self) -> None:
        r"""The read-only accessors and dimension properties report the blocks."""
        n, m, p = 8, 2, 1
        xx = DenseSystem(_random_spd(n, seed=1))
        yy = DenseSystem(_random_spd(m, seed=2))
        B = np.ones((n, m))
        A_eq = np.ones((p, n))
        sys = EqualityCoupledSaddleSystem(xx, B, yy, A_eq)
        self.assertIs(sys.xx_system, xx)
        self.assertIs(sys.yy_system, yy)
        np.testing.assert_array_equal(sys.coupling, B)
        np.testing.assert_array_equal(sys.equality, A_eq)
        self.assertEqual(sys.dimension_x, n)
        self.assertEqual(sys.dimension_y, m)
        self.assertEqual(sys.num_equalities, p)

    def test_rejects_equality_not_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D"):
            EqualityCoupledSaddleSystem(
                DiagonalSystem(np.ones(3)),
                np.ones((3, 2)),
                DiagonalSystem(np.ones(2)),
                np.ones(3),
            )

    def test_rejects_equality_wrong_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "columns"):
            EqualityCoupledSaddleSystem(
                DiagonalSystem(np.ones(3)),
                np.ones((3, 2)),
                DiagonalSystem(np.ones(2)),
                np.ones((1, 4)),
            )

    def test_rejects_equality_zero_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row"):
            EqualityCoupledSaddleSystem(
                DiagonalSystem(np.ones(3)),
                np.ones((3, 2)),
                DiagonalSystem(np.ones(2)),
                np.zeros((0, 3)),
            )

    def test_solve_rejects_wrong_size_b_eq(self) -> None:
        n, m, p = 5, 2, 1
        sys = EqualityCoupledSaddleSystem(
            DiagonalSystem(np.ones(n)),
            np.ones((n, m)),
            DiagonalSystem(np.ones(m)),
            np.ones((p, n)),
        )
        with self.assertRaisesRegex(ValueError, "equality matrix"):
            sys.solve(np.ones(n), np.ones(m), np.ones(p + 1))


class SparseSaddleSystemTest(unittest.TestCase):
    """Tests for ``SparseSaddleSystem``."""

    def test_is_saddle_not_matrix_system(self) -> None:
        sys = SparseSaddleSystem(
            DiagonalSystem(np.ones(3)),
            scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0]])),
        )
        assert isinstance(sys, SaddleSystem)
        assert not isinstance(sys, MatrixSystem)  # type: ignore[unreachable]

    def test_diagonal_hessian_matches_dense_kkt(self) -> None:
        M, p = 6, 3
        rng = np.random.default_rng(11)
        eta = rng.uniform(1.0, 5.0, size=M)
        A_dense = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)

        sys = SparseSaddleSystem(DiagonalSystem(eta), scipy.sparse.csr_matrix(A_dense))
        delta_x, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A_dense, b_top, b_bottom)
        np.testing.assert_allclose(delta_x, dx_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-10)

    def test_matches_dense_schur_complement_saddle_system(self) -> None:
        """Same math as the dense ``SchurComplementSaddleSystem``."""
        M, p = 6, 2
        rng = np.random.default_rng(22)
        eta = rng.uniform(1.0, 5.0, size=M)
        A_dense = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)

        sparse_sys = SparseSaddleSystem(
            DiagonalSystem(eta), scipy.sparse.csr_matrix(A_dense)
        )
        dense_sys = SchurComplementSaddleSystem(DiagonalSystem(eta), A_dense)
        dx_sparse, nu_sparse = sparse_sys.solve(b_top)
        dx_dense, nu_dense = dense_sys.solve(b_top)
        np.testing.assert_allclose(dx_sparse, dx_dense, rtol=1e-8)
        np.testing.assert_allclose(nu_sparse, nu_dense, rtol=1e-8)

    def test_zero_lower_rhs_default_matches_explicit(self) -> None:
        M, p = 5, 2
        rng = np.random.default_rng(33)
        eta = rng.uniform(1.0, 5.0, size=M)
        A = scipy.sparse.csr_matrix(rng.standard_normal((p, M)))
        b_top = rng.standard_normal(M)

        sys = SparseSaddleSystem(DiagonalSystem(eta), A)
        dx_explicit, nu_explicit = sys.solve(b_top, np.zeros(p))
        dx_default, nu_default = sys.solve(b_top)
        np.testing.assert_allclose(dx_explicit, dx_default, rtol=1e-9)
        np.testing.assert_allclose(nu_explicit, nu_default, rtol=1e-9)

    def test_dense_hessian_fallback_matches_dense_kkt(self) -> None:
        """A non-diagonal Hessian uses the generic (densifying) fallback path."""
        M, p = 5, 2
        rng = np.random.default_rng(44)
        H_dense = _random_spd(M, seed=45)
        A_dense = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)

        sys = SparseSaddleSystem(DenseSystem(H_dense), scipy.sparse.csr_matrix(A_dense))
        delta_x, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A_dense, b_top, b_bottom)
        np.testing.assert_allclose(delta_x, dx_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-10)

    def test_truly_sparse_incidence_matrix(self) -> None:
        """A genuinely sparse (bipartite-incidence-like) A, full row rank."""
        A_dense = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
        A = scipy.sparse.csr_matrix(A_dense)
        rng = np.random.default_rng(55)
        eta = rng.uniform(1.0, 5.0, size=4)
        b_top = rng.standard_normal(4)
        b_bottom = rng.standard_normal(3)

        sys = SparseSaddleSystem(DiagonalSystem(eta), A)
        delta_x, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A_dense, b_top, b_bottom)
        np.testing.assert_allclose(delta_x, dx_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-10)
        # KKT residuals hold.
        np.testing.assert_allclose(
            np.diag(eta) @ delta_x + A_dense.T @ nu, b_top, rtol=1e-9
        )
        np.testing.assert_allclose(A_dense @ delta_x, b_bottom, rtol=1e-9)

    def test_matrix_rhs_solves_columnwise(self) -> None:
        M, p, k = 5, 2, 3
        rng = np.random.default_rng(66)
        eta = rng.uniform(1.0, 5.0, size=M)
        A_dense = rng.standard_normal((p, M))
        B_top = rng.standard_normal((M, k))

        sys = SparseSaddleSystem(DiagonalSystem(eta), scipy.sparse.csr_matrix(A_dense))
        delta_x, nu = sys.solve(B_top)
        for j in range(k):
            dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A_dense, B_top[:, j], None)
            np.testing.assert_allclose(delta_x[:, j], dx_ref, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(nu[:, j], nu_ref, rtol=1e-8, atol=1e-10)

    def test_accepts_and_keeps_sparse(self) -> None:
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        sys = SparseSaddleSystem(DiagonalSystem(np.ones(3)), A)
        assert scipy.sparse.issparse(sys.A)

    def test_num_equality_constraints_and_hessian(self) -> None:
        M, p = 4, 3
        hessian = DiagonalSystem(np.ones(M))
        sys = SparseSaddleSystem(hessian, scipy.sparse.csr_matrix(np.ones((p, M))))
        assert sys.num_equality_constraints == p
        assert sys.hessian is hessian

    def test_rejects_a_with_wrong_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "columns"):
            SparseSaddleSystem(
                DiagonalSystem(np.ones(3)),
                scipy.sparse.csr_matrix(np.array([[1.0, 0.0]])),
            )

    def test_rejects_wrong_size_b_bottom(self) -> None:
        M, p = 4, 2
        sys = SparseSaddleSystem(
            DiagonalSystem(np.ones(M)), scipy.sparse.csr_matrix(np.ones((p, M)))
        )
        with self.assertRaisesRegex(ValueError, "one entry for each row"):
            sys.solve(np.ones(M), np.ones(p + 1))

    def test_non_pd_diagonal_raises(self) -> None:
        sys = SparseSaddleSystem(
            DiagonalSystem(np.array([1.0, -1.0, 1.0])),
            scipy.sparse.csr_matrix(np.array([[1.0, 1.0, 1.0]])),
        )
        with self.assertRaises(NewtonStepError):
            sys.solve(np.ones(3))


class SaddleFactoryTest(unittest.TestCase):
    """Tests for the SaddleFactory helpers."""

    def test_schur_factory_builds_dense_saddle(self) -> None:
        factory = SchurComplementSaddleFactory()
        A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        saddle = factory(DiagonalSystem(np.ones(3)), A)
        assert isinstance(saddle, SchurComplementSaddleSystem)

    def test_schur_factory_densifies_sparse_a(self) -> None:
        """The dense factory accepts a sparse A by densifying it."""
        factory = SchurComplementSaddleFactory()
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        saddle = factory(DiagonalSystem(np.ones(3)), A)
        assert isinstance(saddle, SchurComplementSaddleSystem)
        assert not scipy.sparse.issparse(saddle.A)

    def test_sparse_factory_builds_sparse_saddle(self) -> None:
        factory = SparseSaddleFactory()
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        saddle = factory(DiagonalSystem(np.ones(3)), A)
        assert isinstance(saddle, SparseSaddleSystem)
        assert scipy.sparse.issparse(saddle.A)

    def test_factories_produce_the_same_solution(self) -> None:
        """Both factories solve the same KKT system to the same answer."""
        M, p = 6, 2
        rng = np.random.default_rng(303)
        eta = rng.uniform(1.0, 5.0, size=M)
        A_dense = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)

        dense = SchurComplementSaddleFactory()(DiagonalSystem(eta), A_dense)
        sparse = SparseSaddleFactory()(
            DiagonalSystem(eta), scipy.sparse.csr_matrix(A_dense)
        )
        dx_dense, nu_dense = dense.solve(b_top)
        dx_sparse, nu_sparse = sparse.solve(b_top)
        np.testing.assert_allclose(dx_sparse, dx_dense, rtol=1e-8)
        np.testing.assert_allclose(nu_sparse, nu_dense, rtol=1e-8)


class BandedSchurSaddleSystemTest(unittest.TestCase):
    """Tests for ``BandedSchurSaddleSystem``."""

    @staticmethod
    def _banded_a(p: int, w: int, seed: int) -> npt.NDArray[np.float64]:
        """A ``(p, p + w)`` whose Schur complement has bandwidth ``w``.

        Row ``i`` has its nonzeros in columns ``[i, i + w]``, so columns are
        shared only by rows within ``w`` of each other -- the staircase also
        makes ``A`` full row rank.
        """
        rng = np.random.default_rng(seed)
        A = np.zeros((p, p + w))
        for i in range(p):
            A[i, i : i + w + 1] = rng.standard_normal(w + 1)
        return A

    def test_banded_schur_matches_dense_kkt(self) -> None:
        p, w = 12, 2
        A = self._banded_a(p, w, seed=1)
        rng = np.random.default_rng(2)
        eta = rng.uniform(1.0, 5.0, size=int(A.shape[1]))
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(p)

        sys = BandedSchurSaddleSystem(DiagonalSystem(eta), scipy.sparse.csr_matrix(A))
        delta_x, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A, b_top, b_bottom)
        np.testing.assert_allclose(delta_x, dx_ref, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-9)

    def test_full_bandwidth_still_correct(self) -> None:
        """A dense-ish A gives a full-bandwidth Schur; the banded path still solves."""
        M, p = 8, 4
        rng = np.random.default_rng(3)
        eta = rng.uniform(1.0, 5.0, size=M)
        A = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)

        sys = BandedSchurSaddleSystem(DiagonalSystem(eta), scipy.sparse.csr_matrix(A))
        delta_x, nu = sys.solve(b_top)
        dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A, b_top, None)
        np.testing.assert_allclose(delta_x, dx_ref, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-9)

    def test_matches_sparse_saddle_system(self) -> None:
        p, w = 10, 2
        A = scipy.sparse.csr_matrix(self._banded_a(p, w, seed=4))
        rng = np.random.default_rng(5)
        eta = rng.uniform(1.0, 5.0, size=int(A.shape[1]))
        b_top = rng.standard_normal(int(A.shape[1]))

        banded = BandedSchurSaddleSystem(DiagonalSystem(eta), A).solve(b_top)
        sparse = SparseSaddleSystem(DiagonalSystem(eta), A).solve(b_top)
        np.testing.assert_allclose(banded[0], sparse[0], rtol=1e-8)
        np.testing.assert_allclose(banded[1], sparse[1], rtol=1e-8)

    def test_rank_deficient_falls_back_to_dense(self) -> None:
        """A rank-deficient A makes the Schur singular -> dense SVD fallback."""
        base = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        A = np.vstack([base, base[0]])  # row 2 duplicates row 0
        rng = np.random.default_rng(6)
        eta = rng.uniform(1.0, 5.0, size=4)
        b_top = rng.standard_normal(4)
        b_bottom = A @ rng.standard_normal(4)  # consistent RHS (in range of A)

        sys = BandedSchurSaddleSystem(DiagonalSystem(eta), scipy.sparse.csr_matrix(A))
        delta_x, nu = sys.solve(b_top, b_bottom)
        # nu is not unique, but the KKT equations must still hold.
        np.testing.assert_allclose(np.diag(eta) @ delta_x + A.T @ nu, b_top, atol=1e-8)
        np.testing.assert_allclose(A @ delta_x, b_bottom, atol=1e-8)

    def test_dense_hessian_uses_fallback(self) -> None:
        M, p = 5, 2
        rng = np.random.default_rng(7)
        H_dense = _random_spd(M, seed=8)
        A = rng.standard_normal((p, M))
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)

        sys = BandedSchurSaddleSystem(DenseSystem(H_dense), scipy.sparse.csr_matrix(A))
        delta_x, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A, b_top, b_bottom)
        np.testing.assert_allclose(delta_x, dx_ref, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-9)

    def test_factory_builds_banded_saddle(self) -> None:
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        saddle = BandedSaddleFactory()(DiagonalSystem(np.ones(3)), A)
        assert isinstance(saddle, BandedSchurSaddleSystem)
        assert isinstance(saddle, SaddleSystem)

    def test_non_banded_schur_falls_back_and_matches_dense(self) -> None:
        """When S is not narrow-banded the guard routes to the dense solve.

        A dense-ish A gives a Schur complement with bandwidth ~ p; the banded
        storage would be near-dense, so the guard uses the dense path. The
        result must match the dense SparseSaddleSystem exactly.
        """
        rng = np.random.default_rng(9)
        M, p = 9, 6
        eta = rng.uniform(1.0, 5.0, size=M)
        A = scipy.sparse.csr_matrix(rng.standard_normal((p, M)))
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)

        banded = BandedSchurSaddleSystem(DiagonalSystem(eta), A).solve(b_top, b_bottom)
        dense = SparseSaddleSystem(DiagonalSystem(eta), A).solve(b_top, b_bottom)
        np.testing.assert_allclose(banded[0], dense[0], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(banded[1], dense[1], rtol=1e-9, atol=1e-10)


class ArrowSaddleSystemTest(unittest.TestCase):
    """Tests for ``ArrowSaddleSystem`` (banded core + dense border)."""

    @staticmethod
    def _core_plus_border(
        p_core: int, w: int, n_border: int, seed: int
    ) -> npt.NDArray[np.float64]:
        """Banded staircase core (bandwidth ``w``) plus ``n_border`` dense rows.

        Shape is ``(p_core + n_border, p_core + w + n_border)`` -- more columns
        than rows so the matrix is full row rank and the reduced Schur is PD.
        """
        rng = np.random.default_rng(seed)
        n_cols = p_core + w + n_border
        A = np.zeros((p_core + n_border, n_cols))
        for i in range(p_core):
            A[i, i : i + w + 1] = rng.standard_normal(w + 1)
        A[p_core:, :] = rng.standard_normal((n_border, n_cols))
        return A

    def test_matches_dense_kkt_multi_border(self) -> None:
        p_core, w, nb = 12, 2, 3
        A = self._core_plus_border(p_core, w, nb, seed=1)
        rng = np.random.default_rng(2)
        eta = rng.uniform(1.0, 5.0, size=int(A.shape[1]))
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(int(A.shape[0]))
        sys = ArrowSaddleSystem(DiagonalSystem(eta), scipy.sparse.csr_matrix(A), nb)
        dx, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-9)

    def test_single_border_row(self) -> None:
        # n_border == 1 exercises ArrowSystem's scalar-corner (block-plus-one) path.
        p_core, w, nb = 10, 2, 1
        A = self._core_plus_border(p_core, w, nb, seed=3)
        rng = np.random.default_rng(4)
        eta = rng.uniform(1.0, 5.0, size=int(A.shape[1]))
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(int(A.shape[0]))
        dx, nu = ArrowSaddleSystem(
            DiagonalSystem(eta), scipy.sparse.csr_matrix(A), nb
        ).solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(np.diag(eta), A, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-9)

    def test_matches_sparse_saddle_system(self) -> None:
        p_core, w, nb = 14, 3, 4
        A = scipy.sparse.csr_matrix(self._core_plus_border(p_core, w, nb, seed=5))
        rng = np.random.default_rng(6)
        eta = rng.uniform(1.0, 5.0, size=int(A.shape[1]))
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(int(A.shape[0]))
        arrow = ArrowSaddleSystem(DiagonalSystem(eta), A, nb).solve(b_top, b_bottom)
        sparse = SparseSaddleSystem(DiagonalSystem(eta), A).solve(b_top, b_bottom)
        np.testing.assert_allclose(arrow[0], sparse[0], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(arrow[1], sparse[1], rtol=1e-8, atol=1e-9)

    def test_zero_border_matches_banded(self) -> None:
        # n_border == 0 reduces to the plain banded solve.
        p, w = 10, 2
        A = scipy.sparse.csr_matrix(BandedSchurSaddleSystemTest._banded_a(p, w, seed=7))
        rng = np.random.default_rng(8)
        eta = rng.uniform(1.0, 5.0, size=int(A.shape[1]))
        b_top = rng.standard_normal(int(A.shape[1]))
        arrow = ArrowSaddleSystem(DiagonalSystem(eta), A, 0).solve(b_top)
        banded = BandedSchurSaddleSystem(DiagonalSystem(eta), A).solve(b_top)
        np.testing.assert_allclose(arrow[0], banded[0], rtol=1e-10)
        np.testing.assert_allclose(arrow[1], banded[1], rtol=1e-10)

    def test_non_pd_falls_back_to_dense(self) -> None:
        # Rank-deficient A -> singular reduced system -> dense SVD fallback; the
        # KKT equations must still hold.
        base = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        A = np.vstack(
            [base, base[0:1], np.array([[1.0, 1.0, 1.0, 1.0]])]
        )  # row 2 dup; last=border
        rng = np.random.default_rng(9)
        eta = rng.uniform(1.0, 5.0, size=4)
        b_top = rng.standard_normal(4)
        b_bottom = A @ rng.standard_normal(4)  # consistent RHS (in range of A)
        dx, nu = ArrowSaddleSystem(
            DiagonalSystem(eta), scipy.sparse.csr_matrix(A), n_border=1
        ).solve(b_top, b_bottom)
        np.testing.assert_allclose(np.diag(eta) @ dx + A.T @ nu, b_top, atol=1e-8)
        np.testing.assert_allclose(A @ dx, b_bottom, atol=1e-8)

    def test_factory_builds_arrow_saddle(self) -> None:
        A = scipy.sparse.csr_matrix(self._core_plus_border(6, 2, 2, seed=10))
        saddle = ArrowSaddleFactory(n_border=2)(DiagonalSystem(np.ones(A.shape[1])), A)
        assert isinstance(saddle, ArrowSaddleSystem)
        self.assertIsInstance(saddle, SaddleSystem)
        self.assertEqual(saddle.num_border, 2)

    def test_rejects_bad_n_border(self) -> None:
        A = scipy.sparse.csr_matrix(np.ones((3, 5)))
        with self.assertRaises(ValueError):
            ArrowSaddleSystem(DiagonalSystem(np.ones(5)), A, n_border=3)
        with self.assertRaises(ValueError):
            ArrowSaddleSystem(DiagonalSystem(np.ones(5)), A, n_border=-1)

    def test_non_diagonal_hessian_falls_back_to_dense(self) -> None:
        """Non-diagonal Hessian yields no sparse Schur complement → dense fallback."""
        p_core, w, nb = 8, 2, 2
        A = self._core_plus_border(p_core, w, nb, seed=11)
        rng = np.random.default_rng(12)
        H_dense = _random_spd(A.shape[1], seed=13)
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(int(A.shape[0]))
        sys = ArrowSaddleSystem(DenseSystem(H_dense), scipy.sparse.csr_matrix(A), nb)
        dx, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-8, atol=1e-9)

    def test_wide_banded_core_falls_back_to_dense(self) -> None:
        """Dense core rows give wide-banded core Schur → arrow path falls back to dense."""
        rng = np.random.default_rng(14)
        M, p_core, nb = 14, 8, 2
        # Dense random core rows ensure Schur bandwidth ~ p_core-1 >> p_core/2.
        A_core = rng.standard_normal((p_core, M))
        A_border = rng.standard_normal((nb, M))
        A = scipy.sparse.csr_matrix(np.vstack([A_core, A_border]))
        eta = rng.uniform(1.0, 5.0, size=M)
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p_core + nb)
        dx_arrow, nu_arrow = ArrowSaddleSystem(DiagonalSystem(eta), A, nb).solve(
            b_top, b_bottom
        )
        dx_ref, nu_ref = SparseSaddleSystem(DiagonalSystem(eta), A).solve(
            b_top, b_bottom
        )
        np.testing.assert_allclose(dx_arrow, dx_ref, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(nu_arrow, nu_ref, rtol=1e-9, atol=1e-10)


def _arrow_hessian(
    d_core: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    c: float,
) -> tuple[ArrowSystem, npt.NDArray[np.float64]]:
    """Build a positive-definite ``ArrowSystem`` Hessian and its dense form.

    Returns ``(H, H_dense)`` where ``H`` is the structured
    ``[[diag(d_core), u], [u^T, corner]]`` and ``H_dense`` is its dense assembly.
    The corner is enforced to be ``u^T diag(d_core)^{-1} u + c`` (with ``c > 0``)
    so the arrow's Schur complement is positive and the whole matrix is PD --
    the regime the fast path is meant for (a slack-augmented Phase-I Hessian).
    """
    corner = float(u @ (u / d_core)) + c
    H = ArrowSystem(DiagonalSystem(d_core), u, corner)
    M = d_core.shape[0]
    H_dense = np.block(
        [
            [np.diag(d_core), u.reshape(M, 1)],
            [u.reshape(1, M), np.array([[corner]])],
        ]
    )
    return H, H_dense


class DiagonalPlusLowRankSaddleTest(unittest.TestCase):
    """Woodbury fast path for a diagonal-plus-low-rank (ArrowSystem) Hessian."""

    def test_k_dlr_decomposition_reproduces_border(self) -> None:
        r"""``K diag(d_lr) K^T`` equals the off-diagonal border ``[[0, u], [u^T, 0]]``."""
        rng = np.random.default_rng(101)
        M = 5
        d_core = rng.uniform(1.0, 4.0, size=M)
        u = rng.standard_normal(M)
        H, H_dense = _arrow_hessian(d_core, u, 3.0)
        corner = float(H_dense[M, M])
        dplr = H.diagonal_plus_low_rank()
        assert dplr is not None
        d_full, K, d_lr = dplr

        # D_full is the diagonal core with the scalar corner appended.
        np.testing.assert_allclose(d_full.eta, np.concatenate([d_core, [corner]]))
        np.testing.assert_allclose(d_lr, np.array([1.0, -1.0]))
        self.assertEqual(K.shape, (M + 1, 2))

        # K diag(d_lr) K^T reproduces the border block [[0, u], [u^T, 0]].
        border_block = np.zeros((M + 1, M + 1))
        border_block[:M, M] = u
        border_block[M, :M] = u
        np.testing.assert_allclose(K @ np.diag(d_lr) @ K.T, border_block, atol=1e-12)

        # And D_full + that border equals the dense arrow.
        np.testing.assert_allclose(
            np.diag(d_full.eta) + K @ np.diag(d_lr) @ K.T, H_dense, atol=1e-12
        )

    def test_plain_diagonal_is_not_low_rank(self) -> None:
        """A plain diagonal Hessian is not detected as diagonal-plus-low-rank."""
        sys = SparseSaddleSystem(
            DiagonalSystem(np.ones(4)),
            scipy.sparse.csr_matrix(np.eye(4)[:2]),
        )
        self.assertIsNone(sys._diagonal_plus_low_rank())

    @parameterized.expand(
        [
            ("zero_lower", False),
            ("nonzero_lower", True),
        ]
    )
    def test_sparse_saddle_dense_a(self, _name: str, with_bottom: bool) -> None:
        """SparseSaddleSystem: small dense A, arrow Hessian, vs dense KKT oracle."""
        rng = np.random.default_rng(202)
        M, p = 7, 3
        d_core = rng.uniform(1.0, 5.0, size=M)
        u = rng.standard_normal(M)
        c = 4.0
        H, H_dense = _arrow_hessian(d_core, u, c)
        A_dense = rng.standard_normal((p, M + 1))
        b_top = rng.standard_normal(M + 1)
        b_bottom = rng.standard_normal(p) if with_bottom else None

        sys = SparseSaddleSystem(H, scipy.sparse.csr_matrix(A_dense))
        dx, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A_dense, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-7, atol=1e-9)

    @parameterized.expand(
        [
            ("zero_lower", False),
            ("nonzero_lower", True),
        ]
    )
    def test_banded_schur_saddle(self, _name: str, with_bottom: bool) -> None:
        """BandedSchurSaddleSystem: banded-core A, arrow Hessian, vs dense KKT."""
        rng = np.random.default_rng(303)
        p, w = 12, 2
        # A is (p, p + w); the arrow Hessian has dimension p + w = M + 1.
        A = BandedSchurSaddleSystemTest._banded_a(p, w, seed=303)
        M = A.shape[1] - 1
        d_core = rng.uniform(1.0, 5.0, size=M)
        u = rng.standard_normal(M)
        c = 5.0
        H, H_dense = _arrow_hessian(d_core, u, c)
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(p) if with_bottom else None

        sys = BandedSchurSaddleSystem(H, scipy.sparse.csr_matrix(A))
        dx, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-7, atol=1e-9)

    @parameterized.expand(
        [
            ("zero_lower", False),
            ("nonzero_lower", True),
        ]
    )
    def test_arrow_saddle(self, _name: str, with_bottom: bool) -> None:
        """ArrowSaddleSystem: banded-core-plus-border A, arrow Hessian, vs dense KKT."""
        rng = np.random.default_rng(404)
        p_core, w, nb = 12, 2, 3
        A = ArrowSaddleSystemTest._core_plus_border(p_core, w, nb, seed=404)
        M = A.shape[1] - 1
        d_core = rng.uniform(1.0, 5.0, size=M)
        u = rng.standard_normal(M)
        c = 6.0
        H, H_dense = _arrow_hessian(d_core, u, c)
        b_top = rng.standard_normal(int(A.shape[1]))
        b_bottom = rng.standard_normal(int(A.shape[0])) if with_bottom else None

        sys = ArrowSaddleSystem(H, scipy.sparse.csr_matrix(A), nb)
        dx, nu = sys.solve(b_top, b_bottom)
        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-7, atol=1e-9)

    def test_matches_dense_schur_complement_saddle(self) -> None:
        """The fast path agrees with the generic dense SchurComplementSaddleSystem."""
        rng = np.random.default_rng(505)
        M, p = 6, 2
        d_core = rng.uniform(1.0, 5.0, size=M)
        u = rng.standard_normal(M)
        c = 4.0
        H, _ = _arrow_hessian(d_core, u, c)
        A_dense = rng.standard_normal((p, M + 1))
        b_top = rng.standard_normal(M + 1)

        fast = SparseSaddleSystem(H, scipy.sparse.csr_matrix(A_dense)).solve(b_top)
        generic = SchurComplementSaddleSystem(H, A_dense).solve(b_top)
        np.testing.assert_allclose(fast[0], generic[0], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(fast[1], generic[1], rtol=1e-7, atol=1e-9)

    def test_newton_step_error_fallback(self) -> None:
        """The except-NewtonStepError path in _solve_diagonal_plus_low_rank falls back
        to the dense SVD-capable solve and returns the correct KKT solution.

        solve_rank_p_update is patched to raise NewtonStepError (simulating a
        non-PD reduced Schur) so the test directly exercises the fallback branch
        without depending on a specific numerical configuration that might change.
        """
        rng = np.random.default_rng(606)
        M, p = 5, 2
        d_core = rng.uniform(1.0, 5.0, size=M)
        u = rng.standard_normal(M)
        H, H_dense = _arrow_hessian(d_core, u, c=3.0)
        A_dense = rng.standard_normal((p, M + 1))
        b_top = rng.standard_normal(M + 1)
        b_bottom = rng.standard_normal(p)

        with patch(
            "cvxium.linalg.systems.solve_rank_p_update",
            side_effect=NewtonStepError("forced for testing"),
        ):
            sys = SparseSaddleSystem(H, scipy.sparse.csr_matrix(A_dense))
            dx, nu = sys.solve(b_top, b_bottom)

        dx_ref, nu_ref = _dense_kkt_solve(H_dense, A_dense, b_top, b_bottom)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(nu, nu_ref, rtol=1e-7, atol=1e-9)


class IsPositiveDefiniteTest(unittest.TestCase):
    """Tests for ``MatrixSystem.is_positive_definite`` across the hierarchy.

    Each concrete system answers with its own structure-aware test, reusing the
    factorization it already caches; ``ArrowSystem`` exercises the densifying
    default inherited from the ABC. A PD matrix returns ``True``; a wrong-sign
    (non-PD) matrix returns ``False`` rather than raising.
    """

    def test_dense_system(self) -> None:
        """Dense PD certifies via Cholesky; a negative-definite matrix does not."""
        self.assertTrue(DenseSystem(_random_spd(4, seed=1)).is_positive_definite())
        self.assertFalse(DenseSystem(-np.eye(3)).is_positive_definite())

    def test_dense_system_populates_cache(self) -> None:
        """A PD check populates the cached factor, so a later solve pays nothing."""
        system = DenseSystem(_random_spd(4, seed=2))
        self.assertIsNone(system._factor)
        self.assertTrue(system.is_positive_definite())
        self.assertIsNotNone(system._factor)

    def test_diagonal_system(self) -> None:
        """A diagonal is PD iff every entry is strictly positive."""
        self.assertTrue(
            DiagonalSystem(np.array([1.0, 2.0, 3.0])).is_positive_definite()
        )
        self.assertFalse(
            DiagonalSystem(np.array([1.0, -2.0, 3.0])).is_positive_definite()
        )
        self.assertFalse(
            DiagonalSystem(np.array([1.0, 0.0, 3.0])).is_positive_definite()
        )

    def test_banded_system(self) -> None:
        """A banded system certifies via its banded Cholesky, reusing the factor."""
        n = 5
        main = np.full(n, 4.0)
        sup1 = np.full(n, 1.0)
        ab_pd = np.vstack([np.r_[0.0, sup1[1:]], main])
        pd = BandedSystem(ab_pd)
        self.assertTrue(pd.is_positive_definite())
        self.assertIsNotNone(pd._factor)

        ab_non_pd = np.vstack([np.zeros(n), -np.ones(n)])
        self.assertFalse(BandedSystem(ab_non_pd).is_positive_definite())

    def test_low_rank_updated_system_update_stays_pd(self) -> None:
        """An update of a PD base stays PD without densifying the base."""
        base = DenseSystem(_random_spd(4, seed=3))
        kappa = np.array([0.3, -0.2, 0.1, 0.4])
        self.assertTrue(base.low_rank_update(kappa).is_positive_definite())

    def test_low_rank_updated_system_non_pd_base(self) -> None:
        """A non-PD base makes the updated system non-PD (the base solve raises)."""
        base = DenseSystem(-np.eye(3))
        kappa = np.array([1.0, 0.0, 0.0])
        self.assertFalse(base.low_rank_update(kappa).is_positive_definite())

    def test_low_rank_updated_system_downdate_breaks_pd(self) -> None:
        """A downdate large enough to break definiteness is detected as non-PD."""
        base = DenseSystem(np.eye(3))
        kappa = np.array([1.0, 0.0, 0.0])
        # base - 10 * kappa kappa^T = diag(-9, 1, 1): indefinite.
        self.assertFalse(base.low_rank_update(kappa, d=-10.0).is_positive_definite())

    def test_low_rank_updated_system_rank_p(self) -> None:
        """A rank-p update path is certified via the guarded Woodbury prep."""
        base = DenseSystem(_random_spd(5, seed=4))
        kappa = np.random.default_rng(5).standard_normal((5, 2))
        self.assertTrue(base.low_rank_update(kappa).is_positive_definite())

    def test_low_rank_updated_system_rank_deficient_base_lifted_to_pd(self) -> None:
        """A PSD (rank-deficient) base lifted to PD by its update certifies True.

        The base ``diag(0, 2, 3)`` is singular, so the Woodbury base solve
        raises and the fast path cannot certify. The update ``kappa kappa^T``
        with ``kappa = (1, 0, 0)`` restores the missing curvature, making the
        assembled matrix ``diag(1, 2, 3)`` genuinely positive definite. The
        densify + Cholesky fallback resolves the true definiteness. Before the
        fallback was added this returned ``False`` (a false negative).
        """
        base = DiagonalSystem(np.array([0.0, 2.0, 3.0]))
        kappa = np.array([1.0, 0.0, 0.0])
        updated = base.low_rank_update(kappa)
        # Ground truth: assembled matrix is diag(1, 2, 3), all eigenvalues > 0.
        assembled = updated.multiply(np.eye(3))
        self.assertTrue(np.all(np.linalg.eigvalsh(assembled) > 0))
        self.assertTrue(updated.is_positive_definite())

    def test_low_rank_updated_system_zero_base_lifted_to_pd(self) -> None:
        """A ``ZeroSystem`` base lifted to PD by a full-rank update certifies True.

        The zero base is singular (its solve always raises), so only the
        densify + Cholesky fallback can resolve the assembled matrix
        ``kappa diag(d) kappa^T = diag(1, 2, 3)``.
        """
        base = ZeroSystem(3)
        kappa = np.eye(3)
        updated = base.low_rank_update(kappa, d=np.array([1.0, 2.0, 3.0]))
        assembled = updated.multiply(np.eye(3))
        self.assertTrue(np.all(np.linalg.eigvalsh(assembled) > 0))
        self.assertTrue(updated.is_positive_definite())

    def test_low_rank_updated_system_singular_base_still_indefinite(self) -> None:
        """A singular base whose update leaves the sum indefinite stays False.

        The update lifts only one of the two zero directions, so the assembled
        matrix ``diag(1, 0, 3)`` is singular (not PD); the fallback's Cholesky
        fails and the method correctly returns ``False``.
        """
        base = DiagonalSystem(np.array([0.0, 0.0, 3.0]))
        kappa = np.array([1.0, 0.0, 0.0])
        updated = base.low_rank_update(kappa)
        assembled = updated.multiply(np.eye(3))
        self.assertFalse(np.all(np.linalg.eigvalsh(assembled) > 0))
        self.assertFalse(updated.is_positive_definite())

    def test_block_diagonal_system(self) -> None:
        """A block-diagonal is PD iff every block is PD."""
        pd = BlockDiagonalSystem(
            [DiagonalSystem(np.array([1.0, 2.0])), DenseSystem(_random_spd(2, seed=6))]
        )
        self.assertTrue(pd.is_positive_definite())

        non_pd = BlockDiagonalSystem(
            [
                DiagonalSystem(np.array([1.0, 2.0])),
                DiagonalSystem(np.array([-1.0, 2.0])),
            ]
        )
        self.assertFalse(non_pd.is_positive_definite())

    def test_zero_system(self) -> None:
        """The zero matrix is PD only when empty (order 0)."""
        self.assertFalse(ZeroSystem(3).is_positive_definite())
        self.assertTrue(ZeroSystem(0).is_positive_definite())

    def test_arrow_system_uses_dense_default(self) -> None:
        """``ArrowSystem`` inherits the densifying ABC default and answers correctly.

        The Schur corner is ``corner - border^T diag(eta)^-1 border`` (about
        ``corner - 2.03`` here); a corner above that keeps the arrow PD, below it
        indefinite. The densifying default reads the arrow out via ``multiply``
        and Choleskys it.
        """
        eta = np.array([2.0, 3.0, 5.0])
        border = np.array([1.0, 2.0, -1.0])
        self.assertTrue(
            ArrowSystem(DiagonalSystem(eta), border, corner=6.0).is_positive_definite()
        )
        self.assertFalse(
            ArrowSystem(DiagonalSystem(eta), border, corner=1.0).is_positive_definite()
        )

    # ----- definiteness and negation -----

    def test_default_definiteness_is_positive_definite(self) -> None:
        """Every concrete base advertises positive definiteness by default."""
        self.assertIs(
            DiagonalSystem(np.array([1.0, 2.0])).definiteness,
            Definiteness.POSITIVE_DEFINITE,
        )
        self.assertIs(
            DenseSystem(_random_spd(3, seed=7)).definiteness,
            Definiteness.POSITIVE_DEFINITE,
        )

    def test_negated_flips_declared_definiteness(self) -> None:
        """``negated`` reports the flipped definiteness; PD becomes ND."""
        pd = DenseSystem(_random_spd(3, seed=8))
        self.assertIs(pd.definiteness, Definiteness.POSITIVE_DEFINITE)
        self.assertIs(pd.negated().definiteness, Definiteness.NEGATIVE_DEFINITE)

    def test_definiteness_negated_property_is_total(self) -> None:
        """The enum flip is defined for all three members."""
        self.assertIs(
            Definiteness.POSITIVE_DEFINITE.negated, Definiteness.NEGATIVE_DEFINITE
        )
        self.assertIs(
            Definiteness.NEGATIVE_DEFINITE.negated, Definiteness.POSITIVE_DEFINITE
        )
        self.assertIs(Definiteness.INDEFINITE.negated, Definiteness.INDEFINITE)

    def test_negated_solve_and_multiply_carry_the_sign(self) -> None:
        """``-H`` solves and multiplies as the sign-flipped wrapped system."""
        H = DenseSystem(_random_spd(4, seed=9))
        neg = H.negated()
        b = np.array([1.0, -2.0, 0.5, 3.0])
        np.testing.assert_allclose(neg.multiply(b), -H.multiply(b), rtol=1e-12)
        np.testing.assert_allclose(neg.solve(b), -H.solve(b), rtol=1e-12)

    def test_double_negation_collapses_to_original_system(self) -> None:
        """A second negation returns the exact wrapped system, not a nested view.

        This identity is what makes a maximize round-trip numerically free: the
        concave objective advertises ``H.negated()`` and ``Problem.maximize``
        negates it again, recovering the positive-definite ``H`` the solver
        factors -- never a negative-definite matrix.
        """
        H = DenseSystem(_random_spd(3, seed=10))
        self.assertIs(H.negated().negated(), H)

    def test_negated_system_is_not_positive_definite(self) -> None:
        """A negated PD system fails the numerical PD check (it is ND)."""
        H = DenseSystem(_random_spd(3, seed=11))
        self.assertTrue(H.is_positive_definite())
        self.assertFalse(H.negated().is_positive_definite())

    def test_zero_system_negation_stays_zero(self) -> None:
        """``-0 = 0``: the additive identity keeps short-circuiting ``plus``.

        A negated linear objective contributes ``ZeroSystem.negated()``; it must
        stay a ``ZeroSystem`` so it drops out of assembly rather than wrapping
        the zero in a sign-carrier that ``plus`` cannot absorb.
        """
        zero = ZeroSystem(3)
        self.assertIs(zero.negated(), zero)
        pd = DiagonalSystem(np.array([1.0, 2.0, 3.0]))
        self.assertIs(zero.negated().plus(pd), pd)

    def test_negated_scaled_preserves_sign(self) -> None:
        """Scaling ``-H`` by a nonnegative factor stays negated: ``t*(-H)``."""
        H = DenseSystem(_random_spd(3, seed=12))
        scaled_neg = H.negated().scaled(2.0)
        b = np.array([1.0, 2.0, -1.0])
        np.testing.assert_allclose(
            scaled_neg.multiply(b), -2.0 * H.multiply(b), rtol=1e-12
        )
        self.assertIs(scaled_neg.definiteness, Definiteness.NEGATIVE_DEFINITE)
