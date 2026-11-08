"""Test linear program solvers."""

import unittest
from typing import Any

import numpy as np
import scipy.sparse
from parameterized import parameterized
from scipy import linalg

from cvxium.exceptions import ProblemInfeasibleError
from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    MatrixSystem,
    SaddleSystem,
    SchurComplementSaddleSystem,
    SparseSaddleFactory,
    SparseSaddleSystem,
)
from cvxium.problems.linear_constraints import (
    EqualityConstraintLayer,
)
from cvxium.solvers.solver import EqualitySolver


class EqualitySolverTest(unittest.TestCase):
    """Test EqualitySolver."""

    @parameterized.expand(
        [
            (1101, 100, 20),
            (2101, 200, 30),
            (3101, 50, 5),
            (4101, 500, 100),
            (5101, 13, 3),
        ],
    )
    def test_solver(self, seed: int, M: int, p: int) -> None:
        """Test solver."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        w = np.random.randn(M)
        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @parameterized.expand(
        [
            (1102, 100, 20),
            (2102, 200, 30),
            (3102, 50, 5),
            (4102, 500, 100),
            (5102, 13, 3),
        ],
    )
    def test_solver_rank_deficient(self, seed: int, M: int, p: int) -> None:
        """Test solver."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        w = np.random.randn(M)

        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @parameterized.expand(
        [
            (1103, 100, 20),
            (2103, 200, 30),
            (3103, 50, 5),
            (4103, 500, 100),
            (5103, 13, 3),
        ],
    )
    def test_solver_infeasible(self, seed: int, M: int, p: int) -> None:
        """Test solver."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        w = np.random.randn(M)

        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w + np.random.randn(p)

        solver = EqualitySolver(A=A, b=b)

        with self.assertRaises(ProblemInfeasibleError):
            solver.solve()

    @parameterized.expand(
        [
            (1104, 100, 20),
            (2104, 200, 30),
            (3104, 50, 5),
            (4104, 500, 100),
            (5104, 13, 3),
        ],
    )
    def test_solver_with_mean_constraint(self, seed: int, M: int, p: int) -> None:
        """Test solver with an extra row of all 1s in A."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        # Add a row to A of all ones.
        A = np.vstack((A, np.ones((1, M))))
        w = np.random.randn(M)
        b = A @ w

        solver = EqualitySolver(A=A, b=b)

        w = solver.solve().solution
        np.testing.assert_allclose(A @ w, b)

    @parameterized.expand(
        [
            (1105, 100, 20),
            (2105, 200, 30),
            (3105, 50, 5),
            (4105, 500, 100),
            (5105, 13, 3),
        ],
    )
    def test_solver_with_x0(self, seed: int, M: int, p: int) -> None:
        """Test solver with x0."""
        np.random.seed(seed)
        A = np.random.randn(p, M)
        w = np.random.randn(M)

        # Use a rank deficient A
        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh

        b = A @ w

        solver = EqualitySolver(A=A, b=b)
        # Give a set of weights close to the true w as a starting point.
        w0 = w + 0.01 * np.random.randn(M)
        w = solver.solve(x0=w0).solution

        # Verify w is feasible.
        np.testing.assert_allclose(A @ w, b)

        # Check that w is close to w0
        dist = np.dot(w - w0, w - w0)

        # Solve without passing w0
        w = solver.solve().solution

        # Verify the w we calculated when passing w0, is closer to w0 than the w we
        # calculate when not passing w0.
        assert dist < np.dot(w - w0, w - w0)

    @parameterized.expand(
        [
            (6101, 7, 2),
            (6102, 12, 4),
            (6103, 20, 1),
            (6104, 8, 5),
        ],
    )
    def test_saddle_system_against_dense(self, seed: int, M: int, p: int) -> None:
        """The ``saddle_system`` seam matches a dense KKT factorization (zero lower RHS)."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        # Symmetric PD H so the KKT system is well-posed.
        L = rng.standard_normal((M, M))
        H = L @ L.T + M * np.eye(M)
        g = rng.standard_normal(M)

        layer = EqualityConstraintLayer(A=A, b=b)
        saddle = layer.saddle_system(DenseSystem(H))
        delta_x, nu = saddle.solve(g)

        # Dense oracle: build the (M+p)-by-(M+p) KKT matrix and solve.
        K = np.block([[H, A.T], [A, np.zeros((p, p))]])
        rhs = np.concatenate([g, np.zeros(p)])
        sol = np.linalg.solve(K, rhs)
        np.testing.assert_allclose(delta_x, sol[:M], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(nu, sol[M:], rtol=1e-9, atol=1e-9)

    @parameterized.expand(
        [
            (6201, 7, 2),
            (6202, 12, 4),
            (6203, 20, 1),
            (6204, 8, 5),
        ],
    )
    def test_saddle_system_general_rhs_against_dense(
        self, seed: int, M: int, p: int
    ) -> None:
        """The ``saddle_system`` seam matches a dense KKT factorization for non-zero lower RHS."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        L = rng.standard_normal((M, M))
        H = L @ L.T + M * np.eye(M)
        g_x = rng.standard_normal(M)
        g_nu = rng.standard_normal(p)

        layer = EqualityConstraintLayer(A=A, b=b)
        saddle = layer.saddle_system(DenseSystem(H))
        delta_x, delta_nu = saddle.solve(g_x, g_nu)

        K = np.block([[H, A.T], [A, np.zeros((p, p))]])
        rhs = np.concatenate([g_x, g_nu])
        sol = np.linalg.solve(K, rhs)
        np.testing.assert_allclose(delta_x, sol[:M], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(delta_nu, sol[M:], rtol=1e-9, atol=1e-9)

    def test_saddle_system_zero_lower_matches_default(self) -> None:
        """Passing ``b_bottom=0`` agrees with the default (omitted) zero lower RHS."""
        rng = np.random.default_rng(6301)
        M, p = 10, 3
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        L = rng.standard_normal((M, M))
        H = L @ L.T + M * np.eye(M)
        g = rng.standard_normal(M)

        layer = EqualityConstraintLayer(A=A, b=b)
        saddle = layer.saddle_system(DenseSystem(H))
        delta_x_a, nu_a = saddle.solve(g)
        delta_x_b, delta_nu_b = saddle.solve(g, np.zeros(p))
        np.testing.assert_allclose(delta_x_a, delta_x_b, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(nu_a, delta_nu_b, rtol=1e-9, atol=1e-9)

    def test_initialize_barrier_parameter_matches_ipm_path(self) -> None:
        """``EqualitySolver.initialize_barrier_parameter`` agrees with the IPM-side body.

        Pins the bit-exact equivalence between the new method and the
        existing ``EqualityConstrainedInteriorPointMethodSolver`` body
        (``optimization.py:1411-1457``). Load-bearing for D6 of the P2
        stack (when the IPM-side method is rewired to delegate here).
        """
        from cvxium.linalg.systems import DenseSystem
        from cvxium.problems.linear_constraints import (
            BoundsConstraintLayer,
        )
        from cvxium.problems.objective import (
            QuadraticObjective,
        )
        from cvxium.problems.problem import Problem

        # Local imports to avoid pulling barrier_solver into module scope.
        from cvxium.solvers.barrier_solver import (
            _BarrierAdapter,
        )
        from cvxium.solvers.optimization import (
            BaseInteriorPointMethodSolver,
        )
        from cvxium.solvers.settings import (
            OptimizationSettings,
        )

        rng = np.random.default_rng(6401)
        M = 6
        L = rng.standard_normal((M, M))
        Q = L @ L.T + M * np.eye(M)
        c = -Q @ rng.uniform(1.0, 5.0, size=M)
        A = np.vstack([np.ones(M), np.concatenate([np.ones(M // 2), -np.ones(M // 2)])])
        b = np.array([float(M), 0.0])
        equality = EqualitySolver(A=A, b=b)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=equality.constraint,
        )
        adapter = _BarrierAdapter(problem=problem, settings=OptimizationSettings())
        x0 = np.ones(M)

        t_ipm = adapter.initialize_barrier_parameter(x0)

        gradient = adapter.gradient(x0)
        constraints_x0 = adapter.constraints(x0)
        grad_phi = -adapter.grad_constraints_transpose_multiply(
            x0, 1.0 / constraints_x0
        )
        super_init = BaseInteriorPointMethodSolver.initialize_barrier_parameter(
            adapter, x0
        )
        t_es = equality.initialize_barrier_parameter(
            gradient=gradient,
            grad_phi=grad_phi,
            num_ineq=adapter.num_ineq_constraints,
            outer_tol=adapter.settings.outer.tolerance,
            super_init=super_init,
        )

        np.testing.assert_allclose(t_es, t_ipm, rtol=1e-12, atol=1e-12)


class RelativeRankCutPoorlyScaledATest(unittest.TestCase):
    """STAB-6: SVD rank-cut thresholds in ``EqualitySolver`` must be relative to ``s[0]``.

    Pre-fix the rank cut at the minimum-norm path used absolute ``s > 1e-10``,
    and the one at ``initialize_barrier_parameter`` used absolute ``s > 1e-5``.
    Either threshold spuriously truncates the entire spectrum to rank zero
    when ``A`` is poorly scaled (``s[0]`` itself near or below the
    threshold), even though ``A`` is full row rank.
    """

    def test_solve_handles_poorly_scaled_full_rank_A(self) -> None:
        """`solve()` succeeds on full-rank ``A`` even when ``s[0] < 1e-10``."""
        rng = np.random.default_rng(6601)
        # Full row rank, well-conditioned, then scaled so s[0] ~ 1e-11 (below
        # the old absolute 1e-10 cutoff -- pre-fix this raised
        # ProblemInfeasibleError because rank was computed as 0).
        A = 1e-11 * rng.standard_normal((3, 8))
        w_target = rng.standard_normal(8)
        b = A @ w_target

        solver = EqualitySolver(A=A, b=b)
        x_star = solver.solve().solution
        np.testing.assert_allclose(A @ x_star, b, rtol=1e-6, atol=1e-16)

    def test_initialize_barrier_parameter_uses_full_rank_on_poorly_scaled_A(
        self,
    ) -> None:
        """`initialize_barrier_parameter` keeps full rank on poorly-scaled A.

        Compares the result on ``A`` versus ``alpha * A`` for small ``alpha``;
        under absolute thresholding the scaled call drops to rank zero and
        the equality contribution disappears, producing a different ``t``.
        Under relative thresholding both calls retain full rank and the
        equality contribution scales consistently.
        """
        rng = np.random.default_rng(6602)
        M = 6
        p = 2
        A = rng.standard_normal((p, M))
        gradient = rng.standard_normal(M)
        grad_phi = rng.standard_normal(M)
        kwargs: dict[str, Any] = {"num_ineq": 4, "outer_tol": 1e-6, "super_init": 1.0}

        t_well_scaled = EqualitySolver(A=A, b=np.zeros(p)).initialize_barrier_parameter(
            gradient=gradient, grad_phi=grad_phi, **kwargs
        )
        # Scale A so s[0] ~ 1e-7 (below the old absolute 1e-5 cutoff). The
        # Schur formula's z_phi / z_grad scale as 1/alpha^2 and dot-products
        # with A_grad scale as alpha^2, so schur_complement and t_star_s are
        # invariant under uniform scaling of A; t must match well-scaled t.
        alpha = 1e-7
        t_poorly_scaled = EqualitySolver(
            A=alpha * A, b=np.zeros(p)
        ).initialize_barrier_parameter(gradient=gradient, grad_phi=grad_phi, **kwargs)
        np.testing.assert_allclose(
            t_poorly_scaled, t_well_scaled, rtol=1e-6, atol=1e-12
        )


class InitializeBarrierParameterOverflowGuardTest(unittest.TestCase):
    """STAB-9: ``initialize_barrier_parameter`` falls back when ``t_candidate`` is non-finite.

    When ``A`` is so poorly conditioned that ``1 / s_r^2`` overflows even
    after the relative rank cut, the Schur-formula intermediates land at
    ``+/- inf`` (or ``nan`` from ``inf - inf`` in ``schur_complement``).
    The guard detects this at the cause site and returns ``super_init``
    instead of poisoning the IPM with a non-finite ``t``.
    """

    def test_returns_finite_when_t_candidate_overflows(self) -> None:
        """Extreme down-scaling of ``A`` triggers overflow; result remains finite."""
        rng = np.random.default_rng(6701)
        M = 6
        p = 2
        A_well = rng.standard_normal((p, M))
        # Scale A to s[0] ~ 1e-150 so that 1 / s_r^2 overflows to +inf even
        # after STAB-6's relative rank cut (1e-5 * s[0] = 1e-155, square =
        # 1e-310, reciprocal = 1e310 -> overflow).
        A = 1e-150 * A_well
        gradient = rng.standard_normal(M)
        grad_phi = rng.standard_normal(M)
        super_init = 1.0
        num_ineq = 4
        outer_tol = 1e-6

        t = EqualitySolver(A=A, b=np.zeros(p)).initialize_barrier_parameter(
            gradient=gradient,
            grad_phi=grad_phi,
            num_ineq=num_ineq,
            outer_tol=outer_tol,
            super_init=super_init,
        )
        assert np.isfinite(t)
        # With t_candidate -> super_init under the guard, the outer formula
        # collapses to ``min(num_ineq/outer_tol, super_init)``.
        assert t == min(num_ineq / outer_tol, super_init)


class EqualityConstraintLayerTest(unittest.TestCase):
    """Test EqualityConstraintLayer (equality data plus cached SVD of A)."""

    @parameterized.expand(
        [
            (7101, 20, 100),  # wide A: direct SVD path
            (7102, 5, 50),  # small A: direct SVD path
            (7103, 50, 2000),  # M >= 1000, p <= 0.1 M: QR-then-SVD fast path
        ],
    )
    def test_svd_reconstructs_A(self, seed: int, p: int, M: int) -> None:
        """``svd_A`` returns a valid SVD (`U diag(s) Vh == A`) on both paths."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        layer = EqualityConstraintLayer(A=A, b=b)
        U, s, Vh = layer.svd_A()
        np.testing.assert_allclose(U @ np.diag(s) @ Vh, A, rtol=1e-9, atol=1e-9)

    def test_svd_is_cached(self) -> None:
        """``svd_A`` memoizes: repeated calls return the identical arrays."""
        rng = np.random.default_rng(7201)
        A = rng.standard_normal((4, 12))
        layer = EqualityConstraintLayer(A=A, b=A @ rng.standard_normal(12))
        first = layer.svd_A()
        second = layer.svd_A()
        assert first[0] is second[0]
        assert first[1] is second[1]
        assert first[2] is second[2]

    def test_num_eq_constraints_and_storage(self) -> None:
        """The layer stores ``A``/``b`` and reports the equality-constraint count."""
        rng = np.random.default_rng(7301)
        A = rng.standard_normal((3, 9))
        b = A @ rng.standard_normal(9)
        layer = EqualityConstraintLayer(A=A, b=b)
        assert layer.num_eq_constraints == 3
        np.testing.assert_array_equal(layer.A, A)
        np.testing.assert_array_equal(layer.b, b)

    def test_saddle_system_mints_schur_complement_saddle(self) -> None:
        """``saddle_system(H)`` returns a SchurComplementSaddleSystem over ``A``."""
        rng = np.random.default_rng(7601)
        M, p = 6, 2
        B = rng.standard_normal((M, M))
        H_dense = B @ B.T + M * np.eye(M)  # SPD
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        layer = EqualityConstraintLayer(A=A, b=b)

        sys = layer.saddle_system(DenseSystem(H_dense))
        assert isinstance(sys, SchurComplementSaddleSystem)
        assert sys.num_equality_constraints == p
        np.testing.assert_array_equal(sys.A, A)

    def test_saddle_system_solve_matches_direct_construction(self) -> None:
        """The minted saddle's ``solve`` matches a direct construction / KKT solve."""
        rng = np.random.default_rng(7602)
        M, p = 5, 2
        B = rng.standard_normal((M, M))
        H_dense = B @ B.T + M * np.eye(M)  # SPD
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        b_top = rng.standard_normal(M)
        b_bottom = rng.standard_normal(p)
        layer = EqualityConstraintLayer(A=A, b=b)

        H = DenseSystem(H_dense)
        dx_factory, nu_factory = layer.saddle_system(H).solve(b_top, b_bottom)

        # Check against a truly independent oracle: assemble the dense KKT matrix
        # and solve it with ``np.linalg.solve``. (Comparing to ``solve_kkt_system``
        # -- which the Schur saddle solve bottoms out in -- would be tautological.)
        kkt = np.block([[H_dense, A.T], [A, np.zeros((p, p))]])
        sol = np.linalg.solve(kkt, np.concatenate([b_top, b_bottom]))
        dx_ref, nu_ref = sol[:M], sol[M:]
        np.testing.assert_allclose(dx_factory, dx_ref, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(nu_factory, nu_ref, rtol=1e-8, atol=1e-8)


class EqualityConstraintLayerSaddleFactoryTest(unittest.TestCase):
    """The injected ``saddle_factory`` chooses which saddle the layer mints."""

    def test_injected_factory_selects_sparse_saddle(self) -> None:
        """A ``SparseSaddleFactory`` makes ``saddle_system`` mint a sparse saddle."""
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        layer = EqualityConstraintLayer(
            A=A, b=np.zeros(2), saddle_factory=SparseSaddleFactory()
        )
        sys = layer.saddle_system(DiagonalSystem(np.ones(3)))
        assert isinstance(sys, SparseSaddleSystem)

    def test_sparse_a_saddle_solves_against_dense_kkt(self) -> None:
        """End to end: a sparse ``A`` plus the sparse factory solves the KKT."""
        A_dense = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        A = scipy.sparse.csr_matrix(A_dense)
        rng = np.random.default_rng(9001)
        eta = rng.uniform(1.0, 5.0, size=4)
        b_top = rng.standard_normal(4)
        b_bottom = rng.standard_normal(2)

        layer = EqualityConstraintLayer(
            A=A, b=np.zeros(2), saddle_factory=SparseSaddleFactory()
        )
        dx, nu = layer.saddle_system(DiagonalSystem(eta)).solve(b_top, b_bottom)

        kkt = np.block([[np.diag(eta), A_dense.T], [A_dense, np.zeros((2, 2))]])
        sol = np.linalg.solve(kkt, np.concatenate([b_top, b_bottom]))
        np.testing.assert_allclose(dx, sol[:4], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(nu, sol[4:], rtol=1e-8, atol=1e-8)

    def test_svd_a_raises_for_sparse_a(self) -> None:
        """The dense-only SVD path rejects a sparse ``A`` with a clear error."""
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        layer = EqualityConstraintLayer(A=A, b=np.zeros(2))
        with self.assertRaises(NotImplementedError):
            layer.svd_A()

    def test_custom_factory_is_used(self) -> None:
        """Any callable factory is honored -- composition, not inheritance."""
        calls: list[tuple[MatrixSystem, object]] = []

        class RecordingFactory:
            def __call__(self, hessian: MatrixSystem, A: object) -> SaddleSystem:
                calls.append((hessian, A))
                return SchurComplementSaddleSystem(hessian, np.asarray(A))

        A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        layer = EqualityConstraintLayer(
            A=A, b=np.zeros(2), saddle_factory=RecordingFactory()
        )
        hessian = DiagonalSystem(np.ones(3))
        sys = layer.saddle_system(hessian)
        assert isinstance(sys, SchurComplementSaddleSystem)
        assert len(calls) == 1
        assert calls[0][0] is hessian


class EqualitySolverSparseProjectionTest(unittest.TestCase):
    """``EqualitySolver.solve`` projects a sparse ``A`` without an SVD."""

    @staticmethod
    def _sparse_layer(A_dense: np.ndarray, b: np.ndarray) -> EqualityConstraintLayer:
        return EqualityConstraintLayer(
            A=scipy.sparse.csr_matrix(A_dense),
            b=b,
            saddle_factory=SparseSaddleFactory(),
        )

    def test_min_norm_matches_dense(self) -> None:
        rng = np.random.default_rng(8801)
        p, M = 4, 12
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        x_sparse = EqualitySolver(constraint=self._sparse_layer(A, b)).solve().solution
        x_dense = EqualitySolver(A=A, b=b).solve().solution
        np.testing.assert_allclose(A @ x_sparse, b, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x_sparse, x_dense, rtol=1e-7, atol=1e-7)

    def test_projection_from_x0_matches_dense(self) -> None:
        rng = np.random.default_rng(8802)
        p, M = 3, 10
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        x0 = rng.standard_normal(M)
        x_sparse = (
            EqualitySolver(constraint=self._sparse_layer(A, b)).solve(x0=x0).solution
        )
        x_dense = EqualitySolver(A=A, b=b).solve(x0=x0).solution
        np.testing.assert_allclose(A @ x_sparse, b, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(x_sparse, x_dense, rtol=1e-7, atol=1e-7)

    def test_saddle_projects_near_feasible_x0(self) -> None:
        """``_project_via_saddle`` is reached when x0 is slightly off-feasible."""
        rng = np.random.default_rng(8803)
        p, M = 2, 6
        A = rng.standard_normal((p, M))
        x0 = rng.standard_normal(M)
        b = A @ x0  # x0 is the true feasible point
        # Perturb beyond the 1e-10 shortcut threshold so _project_via_saddle runs.
        x0_in = x0 + 1e-6 * rng.standard_normal(M)
        res = EqualitySolver(constraint=self._sparse_layer(A, b)).solve(x0=x0_in)
        np.testing.assert_allclose(A @ res.solution, b, atol=1e-8)
        np.testing.assert_allclose(res.solution, x0, atol=1e-4)

    def test_default_factory_sparse_a_also_projects(self) -> None:
        """Even without SparseSaddleFactory, a sparse A routes through the saddle."""
        rng = np.random.default_rng(8804)
        p, M = 3, 9
        A = rng.standard_normal((p, M))
        b = A @ rng.standard_normal(M)
        # A sparse-A layer with the default (dense) factory still hits the
        # sparse projection branch; the factory just densifies A for the solve.
        layer = EqualityConstraintLayer(A=scipy.sparse.csr_matrix(A), b=b)
        x = EqualitySolver(constraint=layer).solve().solution
        np.testing.assert_allclose(A @ x, b, rtol=1e-8, atol=1e-8)

    def test_infeasible_b_raises(self) -> None:
        # Two identical rows with inconsistent right-hand sides: no solution.
        A = scipy.sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))
        b = np.array([1.0, 2.0])
        layer = EqualityConstraintLayer(A=A, b=b, saddle_factory=SparseSaddleFactory())
        with self.assertRaises(ProblemInfeasibleError):
            EqualitySolver(constraint=layer).solve()


class EqualitySolverDelegatesToLayerTest(unittest.TestCase):
    """``EqualitySolver`` delegates ``A``/``b``/``svd_A`` to its constraint layer."""

    def test_delegation(self) -> None:
        """The solver holds an ``EqualityConstraintLayer`` and reads through it."""
        rng = np.random.default_rng(7401)
        A = rng.standard_normal((4, 16))
        b = A @ rng.standard_normal(16)
        solver = EqualitySolver(A=A, b=b)
        assert isinstance(solver.constraint, EqualityConstraintLayer)
        np.testing.assert_array_equal(solver.A, solver.constraint.A)
        np.testing.assert_array_equal(solver.b, solver.constraint.b)
        assert solver.num_eq_constraints == solver.constraint.num_eq_constraints
        # ``svd_A`` flows through to the layer's cached SVD (same objects).
        assert solver.svd_A()[0] is solver.constraint.svd_A()[0]

    def test_constructor_accepts_prebuilt_layer(self) -> None:
        """``EqualitySolver(constraint=layer)`` shares the given layer and cache."""
        rng = np.random.default_rng(7501)
        A = rng.standard_normal((4, 16))
        b = A @ rng.standard_normal(16)
        layer = EqualityConstraintLayer(A=A, b=b)
        solver = EqualitySolver(constraint=layer)
        assert solver.constraint is layer
        # The shared layer means a warm SVD is reused, not recomputed.
        assert solver.svd_A()[0] is layer.svd_A()[0]

    def test_constructor_requires_consistent_inputs(self) -> None:
        """Neither ``(A, b)`` nor ``constraint``, partial inputs, and all three raise."""
        rng = np.random.default_rng(7502)
        A = rng.standard_normal((3, 9))
        b = A @ rng.standard_normal(9)
        layer = EqualityConstraintLayer(A=A, b=b)
        with self.assertRaises(ValueError):
            EqualitySolver()
        with self.assertRaises(ValueError):
            EqualitySolver(A=A)
        with self.assertRaises(ValueError):
            EqualitySolver(b=b)
        with self.assertRaises(ValueError):
            EqualitySolver(A=A, constraint=layer)
        with self.assertRaises(ValueError):
            EqualitySolver(b=b, constraint=layer)
        with self.assertRaises(ValueError):
            EqualitySolver(A=A, b=b, constraint=layer)
