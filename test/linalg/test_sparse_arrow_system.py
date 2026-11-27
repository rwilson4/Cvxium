"""Tests for SparseArrowSystem, the arrow matrix with a sparse border.

Every check compares against the dense matrix the blocks describe: solves (vector and
matrix right-hand sides) for each core shape the Schur complement handles in closed
form (diagonal, diagonal plus a low-rank update) and for the generic fallback (a dense
core), and multiply, scaled, plus and is_positive_definite.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.sparse
from parameterized import parameterized

from cvxium.exceptions import NewtonStepError
from cvxium.linalg.systems import (
    ArrowSystem,
    DenseSystem,
    DiagonalSystem,
    IncompatibleStructureError,
    LowRankUpdatedSystem,
    MatrixSystem,
    SparseArrowSystem,
    ZeroSystem,
)

if TYPE_CHECKING:
    from collections.abc import Callable

M, P = 12, 4


def _dense(system: MatrixSystem) -> npt.NDArray[np.float64]:
    """Materialize a system through its multiply."""
    return np.asarray(system.multiply(np.eye(system.dimension)))


def _diagonal_core(rng: np.random.Generator) -> MatrixSystem:
    return DiagonalSystem(rng.uniform(1.0, 3.0, M))


def _rank_one_core(rng: np.random.Generator) -> MatrixSystem:
    return DiagonalSystem(rng.uniform(1.0, 3.0, M)).low_rank_update(
        rng.standard_normal(M), 2.0
    )


def _rank_three_core(rng: np.random.Generator) -> MatrixSystem:
    return DiagonalSystem(rng.uniform(1.0, 3.0, M)).low_rank_update(
        rng.standard_normal((M, 3)), rng.uniform(0.5, 2.0, 3)
    )


def _dense_core(rng: np.random.Generator) -> MatrixSystem:
    F = rng.standard_normal((M, M))
    return DenseSystem(F @ F.T + M * np.eye(M))


_CORES: list[tuple[str, Callable[[np.random.Generator], MatrixSystem]]] = [
    ("diagonal", _diagonal_core),
    ("rank_one", _rank_one_core),
    ("rank_three", _rank_three_core),
    ("dense_fallback", _dense_core),
]


def _arrow(
    core: MatrixSystem, rng: np.random.Generator, density: float = 0.3
) -> SparseArrowSystem:
    """A positive-definite sparse arrow around ``core``."""
    border = scipy.sparse.random_array(
        (M, P), density=density, rng=rng, data_sampler=rng.standard_normal
    )
    F = rng.standard_normal((P, P))
    corner = F @ F.T + 10.0 * (1.0 + np.abs(border).sum()) * np.eye(P)
    return SparseArrowSystem(core, border, corner)


class SparseArrowSystemTest(unittest.TestCase):
    """SparseArrowSystem agrees with the dense matrix of its blocks."""

    @parameterized.expand(_CORES)
    def test_solve_matches_dense(
        self, _: str, make_core: Callable[[np.random.Generator], MatrixSystem]
    ) -> None:
        rng = np.random.default_rng(7)
        H = _arrow(make_core(rng), rng)
        H_dense = _dense(H)
        b = rng.standard_normal(M + P)
        np.testing.assert_allclose(H.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10)
        B = rng.standard_normal((M + P, 3))
        np.testing.assert_allclose(H.solve(B), np.linalg.solve(H_dense, B), rtol=1e-10)

    def test_multiply_matches_blocks(self) -> None:
        rng = np.random.default_rng(1)
        core = _rank_three_core(rng)
        H = _arrow(core, rng)
        A12 = H.border.toarray()
        expected = np.block([[_dense(core), A12], [A12.T, H.corner]])
        np.testing.assert_allclose(_dense(H), expected, rtol=1e-12)
        y = rng.standard_normal(M + P)
        np.testing.assert_allclose(H.multiply(y), expected @ y, rtol=1e-12)

    def test_scaled(self) -> None:
        rng = np.random.default_rng(2)
        H = _arrow(_rank_one_core(rng), rng)
        scaled = H.scaled(3.0)
        self.assertIsInstance(scaled, SparseArrowSystem)
        np.testing.assert_allclose(_dense(scaled), 3.0 * _dense(H), rtol=1e-12)

    def test_plus_diagonal_folds_into_core_and_corner(self) -> None:
        rng = np.random.default_rng(3)
        H = _arrow(_rank_one_core(rng), rng)
        D = DiagonalSystem(rng.uniform(0.5, 1.5, M + P))
        for total in (H.plus(D), D.plus(H)):
            self.assertIsInstance(total, SparseArrowSystem)
            np.testing.assert_allclose(_dense(total), _dense(H) + _dense(D), rtol=1e-12)

    def test_plus_arrows_on_same_split(self) -> None:
        rng = np.random.default_rng(4)
        H = _arrow(_diagonal_core(rng), rng)
        G = _arrow(_rank_one_core(rng), rng)
        dense_arrow = ArrowSystem(
            _diagonal_core(rng), rng.standard_normal((M, P)), 5.0 * np.eye(P)
        )
        for other in (G, dense_arrow):
            total = H.plus(other)
            self.assertIsInstance(total, SparseArrowSystem)
            np.testing.assert_allclose(
                _dense(total), _dense(H) + _dense(other), rtol=1e-12
            )

    def test_zero_core_is_filled_by_plus(self) -> None:
        # An objective that curves only the border variables reports a zero core;
        # a layer's diagonal fills it in before any solve.
        rng = np.random.default_rng(5)
        F = rng.standard_normal((P, P))
        objective = SparseArrowSystem(
            ZeroSystem(M), scipy.sparse.csr_array((M, P)), F @ F.T
        )
        layer = _arrow(_rank_one_core(rng), rng)
        bounds = DiagonalSystem(rng.uniform(0.5, 1.5, M + P))
        total = objective.plus(layer).plus(bounds)
        self.assertIsInstance(total, SparseArrowSystem)
        H_dense = _dense(objective) + _dense(layer) + _dense(bounds)
        b = rng.standard_normal(M + P)
        np.testing.assert_allclose(
            total.solve(b), np.linalg.solve(H_dense, b), rtol=1e-10
        )

    def test_plus_low_rank_renests_on_top(self) -> None:
        rng = np.random.default_rng(6)
        H = _arrow(_diagonal_core(rng), rng)
        U = ZeroSystem(M + P).low_rank_update(rng.standard_normal(M + P), 1.5)
        total = H.plus(U)
        self.assertIsInstance(total, LowRankUpdatedSystem)
        b = rng.standard_normal(M + P)
        np.testing.assert_allclose(
            total.solve(b), np.linalg.solve(_dense(H) + _dense(U), b), rtol=1e-10
        )

    def test_incompatible_sums_raise(self) -> None:
        rng = np.random.default_rng(8)
        H = _arrow(_diagonal_core(rng), rng)
        with self.assertRaises(IncompatibleStructureError):
            H.plus(DenseSystem(np.eye(M + P)))
        other = SparseArrowSystem(
            DiagonalSystem(np.ones(M - 1)),
            scipy.sparse.csr_array((M - 1, P + 1)),
            np.eye(P + 1),
        )
        with self.assertRaises(IncompatibleStructureError):
            H.plus(other)

    def test_is_positive_definite(self) -> None:
        rng = np.random.default_rng(9)
        H = _arrow(_rank_one_core(rng), rng)
        self.assertTrue(H.is_positive_definite())
        # A corner too small for the border makes the Schur complement indefinite.
        border = scipy.sparse.csr_array(np.ones((M, P)))
        bad = SparseArrowSystem(DiagonalSystem(np.ones(M)), border, np.eye(P))
        self.assertFalse(bad.is_positive_definite())
        with self.assertRaises(NewtonStepError):
            bad.solve(np.ones(M + P))
        nonpositive_core = SparseArrowSystem(
            DiagonalSystem(np.r_[0.0, np.ones(M - 1)]), border, 100.0 * np.eye(P)
        )
        with self.assertRaises(NewtonStepError):
            nonpositive_core.solve(np.ones(M + P))

    def test_constructor_validates_shapes(self) -> None:
        core = DiagonalSystem(np.ones(M))
        with self.assertRaises(ValueError):
            SparseArrowSystem(core, scipy.sparse.csr_array((M + 1, P)), np.eye(P))
        with self.assertRaises(ValueError):
            SparseArrowSystem(core, scipy.sparse.csr_array((M, P)), np.eye(P + 1))
        H = SparseArrowSystem(core, scipy.sparse.csr_array((M, P)), np.eye(P))
        with self.assertRaises(ValueError):
            H.solve(np.ones(M + P + 1))


if __name__ == "__main__":
    unittest.main()
