"""Tests for quadratic program solvers."""

import time

import numpy as np
import numpy.typing as npt
import pytest
from scipy import linalg, optimize

from cvxium import NewtonResult
from cvxium.quadratic_programs import (
    QuadraticEqualityConstrainedNewtonSolver,
    QuadraticNewtonSolver,
    QuadraticProgramEqualityBoundsSolver,
)


@pytest.mark.parametrize(
    "seed,n",
    [
        (101, 5),
        (201, 10),
        (301, 20),
        (401, 50),
        (501, 100),
    ],
)
def test_unconstrained_newton_quadratic(seed: int, n: int) -> None:
    """UnconstrainedNewtonProblem finds the exact minimizer of a quadratic in one step."""
    rng = np.random.default_rng(seed)

    # Q = A^T A + I ensures strict positive definiteness.
    A = rng.standard_normal((n, n))
    Q = A.T @ A + np.eye(n)
    c = rng.standard_normal(n)
    x0 = rng.standard_normal(n)

    x_star_expected = linalg.solve(Q, -c)

    solver = QuadraticNewtonSolver(Q=Q, c=c)
    result = solver.solve(x0=x0)

    assert isinstance(result, NewtonResult)
    np.testing.assert_allclose(result.solution, x_star_expected, rtol=1e-6, atol=1e-8)
    # Newton's method is exact for quadratics: one step reaches the optimum, then a
    # second iteration detects the near-zero Newton decrement and returns.
    assert result.nits == 2


class TestQuadraticProgramEqualityBoundsSolver:
    r"""Tests for QuadraticProgramEqualityBoundsSolver.

    Solves: minimize  x^T Q x + c^T x
            subject to  A x = b
                        x >= xl

    Uses scipy.optimize.minimize (SLSQP) as ground truth.
    """

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1001, 10, 3),
            (2001, 20, 5),
            (3001, 50, 10),
            (4001, 5, 2),
            (5001, 30, 7),
        ],
    )
    def test_solver_interior_solution(self, seed: int, n: int, p: int) -> None:
        """Solver matches scipy when optimal solution is strictly interior."""
        rng = np.random.default_rng(seed)

        # PD Q: B^T B + I ensures strict positive definiteness.
        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)

        # Choose c so that the unconstrained minimum -(1/2) Q^{-1} c is well inside
        # the feasible region (all positive). Use c = Q @ ones so x_unc = -0.5 * ones
        # which is negative; the equality constraints will force a valid solution.
        c = rng.standard_normal(n)

        # Lower bounds: slightly negative so they are rarely active.
        xl = -0.5 * np.ones(n)

        # Feasible starting point: x0 > xl satisfying A x0 = b.
        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        # --- scipy ground truth (trust-constr handles equality + bounds robustly) ---
        t0 = time.perf_counter()
        scipy_result = optimize.minimize(
            fun=lambda x: float(x @ Q @ x + c @ x),
            x0=x0_feas,
            method="trust-constr",
            jac=lambda x: 2.0 * Q @ x + c,
            hess=lambda x: 2.0 * Q,
            constraints=optimize.LinearConstraint(A, b, b),
            bounds=optimize.Bounds(lb=xl),
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        scipy_ms = 1000.0 * (time.perf_counter() - t0)
        if not scipy_result.success:
            pytest.skip(f"scipy reference solver failed: {scipy_result.message}")

        # --- our solver ---
        t0 = time.perf_counter()
        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()
        cvxium_ms = 1000.0 * (time.perf_counter() - t0)

        print(
            f"\n  n={n:3d}, p={p:2d} | "
            f"scipy: {scipy_ms:7.2f} ms ({scipy_result.nit} iters) | "
            f"Cvxium: {cvxium_ms:7.2f} ms ({result.nits} outer, "
            f"{sum(result.inner_nits)} inner Newton) | "
            f"ratio: {cvxium_ms / scipy_ms:.2f}x"
        )

        np.testing.assert_allclose(
            result.solution,
            scipy_result.x,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Solver solution does not match scipy ground truth.",
        )

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1002, 10, 3),
            (2002, 15, 4),
            (3002, 8, 2),
        ],
    )
    def test_equality_constraints_satisfied(self, seed: int, n: int, p: int) -> None:
        """Solution satisfies equality constraints A x = b to high precision."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        np.testing.assert_allclose(
            A @ result.solution,
            b,
            atol=1e-5,
            err_msg="Equality constraints not satisfied.",
        )

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1003, 10, 3),
            (2003, 15, 4),
            (3003, 8, 2),
        ],
    )
    def test_bound_constraints_satisfied(self, seed: int, n: int, p: int) -> None:
        """Solution satisfies bound constraints x >= xl."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl = -0.5 * np.ones(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        assert np.all(
            result.solution >= xl - 1e-6
        ), "Bound constraints violated: x < xl for some components."

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1004, 10, 3),
            (2004, 20, 5),
        ],
    )
    def test_scalar_lower_bound(self, seed: int, n: int, p: int) -> None:
        """Solver works when xl is a scalar (applied to all components)."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl_scalar = 0.0  # scalar lower bound

        A = rng.standard_normal((p, n))
        x0_feas = np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl_scalar)
        result = solver.solve()

        # Equality constraints satisfied.
        np.testing.assert_allclose(A @ result.solution, b, atol=1e-5)
        # Bound constraints satisfied.
        assert np.all(result.solution >= xl_scalar - 1e-6)

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (1005, 10, 3),
            (2005, 20, 5),
        ],
    )
    def test_dual_is_lower_bound(self, seed: int, n: int, p: int) -> None:
        """Dual value is a lower bound on the primal objective."""
        rng = np.random.default_rng(seed)

        B = rng.standard_normal((n, n))
        Q = B.T @ B + np.eye(n)
        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        # Dual value <= primal objective (weak duality).
        assert (
            result.dual_value <= result.objective_value + 1e-6
        ), f"Dual {result.dual_value} > primal {result.objective_value}: weak duality violated."

    @pytest.mark.parametrize(
        "seed,n,p,rank",
        [
            (1006, 10, 3, 7),
            (2006, 15, 4, 10),
        ],
    )
    def test_psd_q(self, seed: int, n: int, p: int, rank: int) -> None:
        """Solver works when Q is PSD (rank-deficient) rather than PD."""
        rng = np.random.default_rng(seed)

        # Build a rank-deficient PSD Q: Q = B^T B where B is rank x n.
        B = rng.standard_normal((rank, n))
        Q = B.T @ B  # PSD, rank = rank < n

        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        solver = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result = solver.solve()

        np.testing.assert_allclose(A @ result.solution, b, atol=1e-5)
        assert np.all(result.solution >= xl - 1e-6)


class TestStructuredQCallables:
    r"""Timing tests for Q_vector_multiply / Q_solve callables.

    Three variants are compared for each structured Q:
      * scipy trust-constr (ground truth / reference)
      * Cvxium without callables (baseline: O(n²) gradient, O(n³) Newton)
      * Cvxium with callables   (target:   O(n)  gradient, O(n)  Newton)
    """

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (7001, 50, 5),
            (7002, 100, 8),
            (7003, 200, 10),
        ],
    )
    def test_diagonal_q(self, seed: int, n: int, p: int) -> None:
        """Diagonal Q: callables enable O(n) gradient and O(n) Newton step."""
        rng = np.random.default_rng(seed)

        # Q = diag(q) with positive entries.
        q = rng.uniform(0.5, 3.0, n)
        Q = np.diag(q)

        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        # Callables: O(n) multiply and solve for diagonal Q.
        def q_multiply(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            if v.ndim == 1:
                return q * v
            return q[:, np.newaxis] * v

        def q_solve(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            if v.ndim == 1:
                return v / q
            return v / q[:, np.newaxis]

        # --- scipy ground truth ---
        t0 = time.perf_counter()
        scipy_result = optimize.minimize(
            fun=lambda x: float(x @ Q @ x + c @ x),
            x0=x0_feas,
            method="trust-constr",
            jac=lambda x: 2.0 * Q @ x + c,
            hess=lambda x: 2.0 * Q,
            constraints=optimize.LinearConstraint(A, b, b),
            bounds=optimize.Bounds(lb=xl),
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        scipy_ms = 1000.0 * (time.perf_counter() - t0)
        if not scipy_result.success:
            pytest.skip(f"scipy reference solver failed: {scipy_result.message}")

        # --- Cvxium without callables ---
        t0 = time.perf_counter()
        solver_naive = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result_naive = solver_naive.solve()
        cvxium_naive_ms = 1000.0 * (time.perf_counter() - t0)

        # --- Cvxium with callables ---
        t0 = time.perf_counter()
        solver_fast = QuadraticProgramEqualityBoundsSolver(
            Q=Q,
            c=c,
            A=A,
            b=b,
            xl=xl,
            Q_vector_multiply=q_multiply,
            Q_solve=q_solve,
        )
        result_fast = solver_fast.solve()
        cvxium_fast_ms = 1000.0 * (time.perf_counter() - t0)

        speedup = (
            cvxium_naive_ms / cvxium_fast_ms if cvxium_fast_ms > 0 else float("inf")
        )
        print(
            f"\n  diagonal Q | n={n:3d}, p={p:2d} | "
            f"scipy: {scipy_ms:7.2f} ms | "
            f"Cvxium naive: {cvxium_naive_ms:7.2f} ms | "
            f"Cvxium fast: {cvxium_fast_ms:7.2f} ms | "
            f"speedup: {speedup:.1f}x"
        )

        # Both variants must match scipy.
        np.testing.assert_allclose(
            result_naive.solution, scipy_result.x, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            result_fast.solution, scipy_result.x, rtol=1e-3, atol=1e-3
        )
        # Constraints satisfied.
        np.testing.assert_allclose(A @ result_fast.solution, b, atol=1e-5)
        assert np.all(result_fast.solution >= xl - 1e-6)

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (8001, 50, 5),
            (8002, 100, 8),
            (8003, 200, 10),
        ],
    )
    def test_diagonal_plus_rank_one_q(self, seed: int, n: int, p: int) -> None:
        """Q = diag(q) + u u^T: O(n) multiply/solve via Sherman-Morrison."""
        rng = np.random.default_rng(seed)

        q = rng.uniform(0.5, 3.0, n)
        u = rng.standard_normal(n)
        Q = np.diag(q) + np.outer(u, u)

        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        # Sherman-Morrison constant: a = 1 / (1 + u^T Q_diag^{-1} u)
        alpha = 1.0 / (1.0 + float(np.dot(u, u / q)))

        def q_multiply(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Q v = diag(q) v + (u^T v) u."""
            if v.ndim == 1:
                return q * v + np.dot(u, v) * u
            return q[:, np.newaxis] * v + np.outer(u, u @ v)

        def q_solve(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Q^{-1} v via Sherman-Morrison: v/q - a*(u^T(v/q))*u/q."""
            if v.ndim == 1:
                v_over_q = v / q
                return v_over_q - alpha * np.dot(u, v_over_q) * (u / q)
            v_over_q = v / q[:, np.newaxis]
            return v_over_q - alpha * np.outer(u / q, u @ v_over_q)

        # --- scipy ground truth ---
        t0 = time.perf_counter()
        scipy_result = optimize.minimize(
            fun=lambda x: float(x @ Q @ x + c @ x),
            x0=x0_feas,
            method="trust-constr",
            jac=lambda x: 2.0 * Q @ x + c,
            hess=lambda x: 2.0 * Q,
            constraints=optimize.LinearConstraint(A, b, b),
            bounds=optimize.Bounds(lb=xl),
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        scipy_ms = 1000.0 * (time.perf_counter() - t0)
        if not scipy_result.success:
            pytest.skip(f"scipy reference solver failed: {scipy_result.message}")

        # --- Cvxium without callables ---
        t0 = time.perf_counter()
        solver_naive = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result_naive = solver_naive.solve()
        cvxium_naive_ms = 1000.0 * (time.perf_counter() - t0)

        # --- Cvxium with callables ---
        t0 = time.perf_counter()
        solver_fast = QuadraticProgramEqualityBoundsSolver(
            Q=Q,
            c=c,
            A=A,
            b=b,
            xl=xl,
            Q_vector_multiply=q_multiply,
            Q_solve=q_solve,
        )
        result_fast = solver_fast.solve()
        cvxium_fast_ms = 1000.0 * (time.perf_counter() - t0)

        speedup = (
            cvxium_naive_ms / cvxium_fast_ms if cvxium_fast_ms > 0 else float("inf")
        )
        print(
            f"\n  diag+rank-1 Q | n={n:3d}, p={p:2d} | "
            f"scipy: {scipy_ms:7.2f} ms | "
            f"Cvxium naive: {cvxium_naive_ms:7.2f} ms | "
            f"Cvxium fast: {cvxium_fast_ms:7.2f} ms | "
            f"speedup: {speedup:.1f}x"
        )

        np.testing.assert_allclose(
            result_naive.solution, scipy_result.x, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            result_fast.solution, scipy_result.x, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(A @ result_fast.solution, b, atol=1e-5)
        assert np.all(result_fast.solution >= xl - 1e-6)

    @pytest.mark.parametrize(
        "seed,n,p",
        [
            (9001, 50, 5),
            (9002, 100, 8),
            (9003, 200, 10),
        ],
    )
    def test_diagonal_plus_rank_one_q_with_scale_and_diag_add(
        self, seed: int, n: int, p: int
    ) -> None:
        """Q = diag(q) + u u^T: scale+diag_add callables enable O(n) Newton step.

        ``Q_solve(v, scale=s, diag_add=e)`` solves ``(s*Q + diag(e)) x = v`` using
        Sherman-Morrison on ``diag(s*q + e) + sqrt(s)*u (sqrt(s)*u)^T``.
        ``scale > 0`` and ``diag_add >= 0`` are guaranteed by the barrier; the
        implementation may assert these preconditions.
        """
        rng = np.random.default_rng(seed)

        q = rng.uniform(0.5, 3.0, n)
        u = rng.standard_normal(n)
        Q = np.diag(q) + np.outer(u, u)

        c = rng.standard_normal(n)
        xl = np.zeros(n)

        A = rng.standard_normal((p, n))
        x0_feas = xl + np.abs(rng.standard_normal(n)) + 1.0
        b = A @ x0_feas

        def q_multiply(
            v: npt.NDArray[np.float64],
            scale: float = 1.0,
            diag_add: npt.NDArray[np.float64] | None = None,
        ) -> npt.NDArray[np.float64]:
            """(scale * Q + diag(diag_add)) v = (scale*q + diag_add)*v + scale*(u^T v)*u."""
            e = diag_add if diag_add is not None else 0.0
            if v.ndim == 1:
                return (scale * q + e) * v + scale * np.dot(u, v) * u
            return (scale * q + e)[:, np.newaxis] * v + scale * np.outer(u, u @ v)

        def q_solve(
            v: npt.NDArray[np.float64],
            scale: float = 1.0,
            diag_add: npt.NDArray[np.float64] | None = None,
        ) -> npt.NDArray[np.float64]:
            """Solve (scale * Q + diag(diag_add)) x = v via Sherman-Morrison.

            (scale*diag(q) + scale*u u^T + diag(e)) x = (diag(scale*q+e) + w w^T) x = v
            where w = sqrt(scale)*u.  Sherman-Morrison with D_e = diag(scale*q+e):
                x = D_e^{-1} v - scale*(u^T D_e^{-1} v)/(1 + scale*u^T D_e^{-1} u)*D_e^{-1} u

            Preconditions: scale > 0, diag_add >= 0 (entry-wise).
            """
            e = diag_add if diag_add is not None else 0.0
            eff_diag = scale * q + e  # diag(D_e)
            coeff = scale / (1.0 + scale * np.dot(u, u / eff_diag))
            if v.ndim == 1:
                v_over_d = v / eff_diag
                return v_over_d - coeff * np.dot(u, v_over_d) * (u / eff_diag)
            v_over_d = v / eff_diag[:, np.newaxis]
            return v_over_d - coeff * np.outer(u / eff_diag, u @ v_over_d)

        # --- scipy ground truth ---
        t0 = time.perf_counter()
        scipy_result = optimize.minimize(
            fun=lambda x: float(x @ Q @ x + c @ x),
            x0=x0_feas,
            method="trust-constr",
            jac=lambda x: 2.0 * Q @ x + c,
            hess=lambda x: 2.0 * Q,
            constraints=optimize.LinearConstraint(A, b, b),
            bounds=optimize.Bounds(lb=xl),
            options={"gtol": 1e-10, "maxiter": 2000},
        )
        scipy_ms = 1000.0 * (time.perf_counter() - t0)
        if not scipy_result.success:
            pytest.skip(f"scipy reference solver failed: {scipy_result.message}")

        # --- Cvxium without callables ---
        t0 = time.perf_counter()
        solver_naive = QuadraticProgramEqualityBoundsSolver(Q=Q, c=c, A=A, b=b, xl=xl)
        result_naive = solver_naive.solve()
        cvxium_naive_ms = 1000.0 * (time.perf_counter() - t0)

        # --- Cvxium with scale+diag_add callables ---
        t0 = time.perf_counter()
        solver_fast = QuadraticProgramEqualityBoundsSolver(
            Q=Q,
            c=c,
            A=A,
            b=b,
            xl=xl,
            Q_vector_multiply=q_multiply,
            Q_solve=q_solve,
        )
        result_fast = solver_fast.solve()
        cvxium_fast_ms = 1000.0 * (time.perf_counter() - t0)

        speedup = (
            cvxium_naive_ms / cvxium_fast_ms if cvxium_fast_ms > 0 else float("inf")
        )
        print(
            f"\n  diag+rank-1 Q (scale+diag_add) | n={n:3d}, p={p:2d} | "
            f"scipy: {scipy_ms:7.2f} ms | "
            f"Cvxium naive: {cvxium_naive_ms:7.2f} ms | "
            f"Cvxium fast: {cvxium_fast_ms:7.2f} ms | "
            f"speedup: {speedup:.1f}x"
        )

        np.testing.assert_allclose(
            result_naive.solution, scipy_result.x, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            result_fast.solution, scipy_result.x, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(A @ result_fast.solution, b, atol=1e-5)
        assert np.all(result_fast.solution >= xl - 1e-6)


@pytest.mark.parametrize(
    "seed,n,p",
    [
        (701, 5, 2),
        (801, 10, 3),
        (901, 20, 5),
        (1101, 50, 10),
        (1201, 100, 20),
    ],
)
def test_equality_constrained_newton_quadratic(seed: int, n: int, p: int) -> None:
    """EqualityConstrainedNewtonProblem finds the exact KKT point of a quadratic in one step."""
    rng = np.random.default_rng(seed)

    # Q = B^T B + I ensures strict positive definiteness.
    B = rng.standard_normal((n, n))
    Q = B.T @ B + np.eye(n)
    c = rng.standard_normal(n)

    # Build a feasible starting point x0 satisfying A x0 = b.
    A = rng.standard_normal((p, n))
    x0 = rng.standard_normal(n)
    b = A @ x0

    # Expected solution via direct KKT solve:
    #   [ Q    A^T ] [ x*  ]   [ -c ]
    #   [ A     0  ] [ nu* ] = [  b ]
    KKT = np.block([[Q, A.T], [A, np.zeros((p, p))]])
    rhs = np.concatenate([-c, b])
    sol = linalg.solve(KKT, rhs)
    x_star_expected = sol[:n]

    solver = QuadraticEqualityConstrainedNewtonSolver(Q=Q, c=c, A=A, b=b)
    result = solver.solve(x0=x0)

    assert isinstance(result, NewtonResult)
    np.testing.assert_allclose(result.solution, x_star_expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(A @ result.solution, b, atol=1e-8)
    # Newton's method is exact for quadratics: one step reaches the optimum.
    assert result.nits == 2


@pytest.mark.parametrize(
    "seed,n,p",
    [
        (2101, 10, 3),
        (2201, 20, 5),
        (2301, 50, 10),
    ],
)
def test_equality_constrained_newton_quadratic_dual(seed: int, n: int, p: int) -> None:
    """Dual value equals primal at the KKT point (zero duality gap)."""
    rng = np.random.default_rng(seed)

    B = rng.standard_normal((n, n))
    Q = B.T @ B + np.eye(n)
    c = rng.standard_normal(n)

    A = rng.standard_normal((p, n))
    x0 = rng.standard_normal(n)
    b = A @ x0

    solver = QuadraticEqualityConstrainedNewtonSolver(Q=Q, c=c, A=A, b=b)
    result = solver.solve(x0=x0)

    # At the KKT point, strong duality holds: dual = primal.
    assert abs(result.objective_value - result.dual_value) < 1e-6
