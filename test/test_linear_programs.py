"""Test linear program solvers."""

import time

import numpy as np
import pytest
from scipy import linalg, optimize

from cvxium.exceptions import (
    InteriorPointMethodError,
    ProblemInfeasibleError,
)
from cvxium.linear_programs import (
    EqualitySolver,
    EqualityWithBoundsAndImbalanceConstraintSolver,
    EqualityWithBoundsSolver,
)
from cvxium.optimization import (
    InteriorPointMethodResult,
    OptimizationSettings,
    ProblemCertifiablyInfeasibleError,
    ProblemMarginallyFeasibleError,
)


class TestEqualitySolver:
    """Test EqualitySolver."""

    @pytest.mark.parametrize(
        "seed,M,p",
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

    @pytest.mark.parametrize(
        "seed,M,p",
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

    @pytest.mark.parametrize(
        "seed,M,p",
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

        with pytest.raises(ProblemInfeasibleError):
            solver.solve()

    @pytest.mark.parametrize(
        "seed,M,p",
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

    @pytest.mark.parametrize(
        "seed,M,p",
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


class TestEqualityWithBoundsSolver:
    """Test EqualityWithBoundsSolver."""

    @pytest.mark.parametrize(
        "seed,M,p,atol",
        [
            (1201, 100, 20, 1e-8),
            (2201, 200, 30, 1e-9),
            (3201, 50, 5, 1e-7),
            (4201, 500, 100, 1e-9),
            (5201, 13, 3, 1e-7),
        ],
    )
    def test_solver_feasible_bounded_below(
        self, seed: int, M: int, p: int, atol: float
    ) -> None:
        """Test solver when problem is feasible and the augmented problem bounded below.

        The dual function for the augmented problem gives a lower bound for the problem
        value. The dual function is finite if there exists a collection nu (length p),
        lambda1 (length M), lambda2 (scalar), such that:
           lambda1 = A^T nu
           lambda2 = sum(lambda1) - 1
           lambda1 >= 0
           lambda2 >= 0.

        And if there exists w such that A * w = b, then the problem is feasible.

        """
        np.random.seed(seed)

        nu = np.random.randn(p)
        # Construct A such that lambda1 := A^T nu >= 0
        A = np.random.randn(p, M)
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        assert np.all(A.T @ nu >= 0)

        # Scale A such that lambda2 := sum(A^T nu) - 1 > 0
        A *= 1.1 / np.sum(A.T @ nu)
        assert np.sum(A.T @ nu) > 1

        # Construct feasible w
        lb = 0.05
        w = 2 * lb + np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            lb=lb,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b, atol=atol)
        assert np.all(res.solution > lb)

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b, atol=atol)
        assert np.all(res.solution > 0.0)

        # Solve with a vector of lb
        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            lb=np.full((M,), lb),
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b, atol=atol)
        assert np.all(res.solution > lb)

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1202, 100, 20),
            (2202, 200, 30),
            (3202, 50, 5),
            (4202, 500, 100),
            (5202, 13, 3),
        ],
    )
    def test_solver_feasible_unbounded_below(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is feasible and the augmented problem is unbounded below.

        The augmented problem is unbounded below (yet is still feasible) if there does
        not exist a collection nu (length p), lambda1 (length M), lambda2 (scalar), such
        that:
           lambda1 = A^T nu
           lambda2 = sum(lambda1) - 1
           lambda1 >= 0
           lambda2 >= 0.

        One simple example is when A is a 1-by-M matrix with at least one strictly
        positive entry and at least one strictly negative entry. Then nu is just a
        scalar, and if nu is:
           positive: at least one entry in lambda1 is negative
           negative: at least one entry in lambda1 is negative
           0: lambda1 = 0 so lambda2 = -1.

        """
        np.random.seed(seed)
        p = 1

        A = np.random.randn(p, M)
        A[0, 0] = -abs(A[0, 0])
        A[0, 1] = abs(A[0, 1])

        # Generate feasible w
        w = np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

        # See what happens if we try to fully optimize
        with pytest.raises(InteriorPointMethodError):
            solver.solve(fully_optimize=True)

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1203, 100, 20),
            (2203, 200, 30),
            (3203, 50, 5),
            (4203, 500, 100),
            (5203, 13, 3),
        ],
    )
    def test_solver_infeasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is infeasible.

        We seek an x satisfying A * x = b and x > 0. No such x exists if there exists a
        nu satisfying:
            A^T * nu >= 0,
            c^T * nu = 1, where c = A * 1, and
            b^T nu < 0.
        This test constructs such an A, b and verifies the solver detects this
        infeasibility.

        """
        np.random.seed(seed)
        nu = np.random.randn(p)

        # Construct A such that A^T nu >= 0
        A = np.random.randn(p, M)
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        assert np.all(A.T @ nu >= 0)

        # Scale A such that c^T * nu = 1
        A /= np.dot(A @ np.ones((M,)), nu)
        assert np.all(A.T @ nu >= 0)
        assert abs(np.dot(A @ np.ones((M,)), nu) - 1) <= 1e-6

        # Construct b such that b^T nu < 0.
        b = np.random.randn(p)
        alpha = np.dot(nu, b) / np.dot(nu, nu)
        b = b - (alpha + 0.1) * nu
        assert np.all(A.T @ nu >= 0)
        assert abs(np.dot(A @ np.ones((M,)), nu) - 1) <= 1e-6
        assert np.dot(b, nu) < 0

        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            settings=OptimizationSettings(verbose=True),
        )

        with pytest.raises(ProblemCertifiablyInfeasibleError):
            solver.solve()

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        assert isinstance(res, InteriorPointMethodResult)
        assert res.dual_value > 0

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1205, 100, 20),
            (2205, 200, 30),
            (3205, 50, 5),
            (4205, 500, 100),
            (5205, 13, 3),
        ],
    )
    def test_solver_marginally_feasible(self, seed: int, M: int, p: int) -> None:
        """Test solver when problem is marginally feasible.

        Here we test a scenario where there exists a technically feasible w, but where
        strict feasibility does not hold. In this case, the dual function will keep
        returning negative values, so we cannot certify the problem as infeasible, but
        we won't find a strictly feasible w (because it doesn't exist).

        """
        np.random.seed(seed)

        w_star = np.random.rand(M)

        # This is a little hacky. We're basically counting on the method to not certify
        # the problem as infeasible.
        w_star[0] = -1e-8

        # Generate a random orthonormal matrix
        A, _, _ = linalg.svd(np.random.randn(M, M))
        b = A @ w_star
        # w_star is the only solution to A * w = b, but w_star is only marginally feasible.

        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            settings=OptimizationSettings(verbose=True, outer_tolerance=1e-4),
        )

        with pytest.raises(ProblemMarginallyFeasibleError):
            solver.solve()

    @pytest.mark.parametrize(
        "seed,M,p",
        [
            (1206, 100, 20),
            (2206, 200, 30),
            (3206, 50, 5),
            (4206, 500, 100),
            (5206, 13, 3),
        ],
    )
    def test_solver_feasible_mean_constraint(self, seed: int, M: int, p: int) -> None:
        """Test solver with an extra row of all 1s in A."""
        np.random.seed(seed)

        nu = np.random.randn(p + 1)
        # Construct A such that lambda1 := A^T nu[:p] >= 0
        A = np.random.randn(p, M)

        for ic in range(M):
            if np.dot(A[:, ic], nu[:p]) < 0:
                A[:, ic] = -A[:, ic]

        # Scale A such that lambda2 := sum(A^T nu) - 1 > 0
        A *= 1.1 / np.sum(A.T @ nu[:p])

        # Add a row to A of all ones. So long as the last element of nu is positive,
        # we'll still have have [A^T 1] * nu = A^T nu[:p] + nu[p] * 1 >= A^T nu[:p] >= 0
        # and sum([A^T 1] * nu) >= sum(A^T nu[:p]) > 1
        A = np.vstack((A, np.ones((1, M))))
        if nu[p] < 0:
            nu[p] = -nu[p]

        assert np.all(A.T @ nu >= 0)
        assert np.sum(A.T @ nu) > 1

        # Construct feasible w
        w = 0.1 + np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b, atol=1e-10)
        assert np.all(res.solution > 0)

    @pytest.mark.parametrize(
        "seed,M,p,atol",
        [
            (1207, 100, 20, 2e-9),
            (2207, 200, 30, 1e-9),
            (3207, 50, 5, 1e-6),
            (4207, 500, 100, 1e-4),
            (5207, 13, 3, 1e-9),
        ],
    )
    def test_solver_feasible_bounded_below_rank_deficient(
        self, seed: int, M: int, p: int, atol: float
    ) -> None:
        """Test solver when problem is feasible and the augmented problem bounded below.

        The dual function for the augmented problem gives a lower bound for the problem
        value. The dual function is finite if there exists a collection nu (length p),
        lambda1 (length M), lambda2 (scalar), such that:
           lambda1 = A^T nu
           lambda2 = sum(lambda1) - 1
           lambda1 >= 0
           lambda2 >= 0.

        And if there exists w such that A * w = b, then the problem is feasible.

        """
        np.random.seed(seed)

        nu = np.random.randn(p)
        # Construct A such that lambda1 := A^T nu >= 0
        A = np.random.randn(p, M)
        U, s, Vh = linalg.svd(A, full_matrices=False)
        s[-1] = 0.0
        s[-2] = 0.0
        A = U @ np.diag(s) @ Vh
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        assert np.all(A.T @ nu >= 0)

        # Scale A such that lambda2 := sum(A^T nu) - 1 > 0
        A *= 1.1 / np.sum(A.T @ nu)
        assert np.sum(A.T @ nu) > 1

        U, s, Vh = linalg.svd(A, full_matrices=False)
        assert s[-1] < 1e-10
        assert s[-2] < 1e-10

        # Construct feasible w
        w = 0.1 + np.random.rand(M)
        b = A @ w

        solver = EqualityWithBoundsSolver(
            A=A,
            b=b,
            settings=OptimizationSettings(verbose=True),
        )

        # Verify we find a feasible point
        res = solver.solve()
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > 0)

        # Since augmented problem is bounded below, verify we can fully solve it (if for
        # whatever reason we wanted to).
        res = solver.solve(fully_optimize=True)
        np.testing.assert_allclose(A @ res.solution, b, atol=atol)
        assert np.all(res.solution > 0)


class TestEqualityWithBoundsAndImbalanceConstraintSolver:
    """Test EqualityWithBoundsAndImbalanceConstraintSolver."""

    @pytest.mark.parametrize(
        "seed,M,p1,p2",
        [
            (1301, 100, 10, 15),
            (2301, 200, 5, 30),
            (3301, 50, 5, 5),
            (4301, 500, 20, 30),
            (5301, 13, 3, 3),
        ],
    )
    def test_solver_feasible(self, seed: int, M: int, p1: int, p2: int) -> None:
        """Test solver when problem is feasible."""
        np.random.seed(seed)

        A = np.random.randn(p1, M)
        B = np.random.randn(p2, M)

        # To generate population mean, simulate true propensity scores with mean 0.1 and
        # variance 0.0045
        q = 0.1
        sigma2 = 0.05 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        true_propensity = np.random.beta(alpha, beta, size=M)

        # Ideal weights are (M/N) / true_propensity, but to make it simple, do:
        w = 1.0 / true_propensity
        w /= np.mean(w)
        lb = 0.01
        assert np.all(w > lb)

        b = A @ w
        # Add a constraint on the mean weights
        A[0, :] = 1.0 / M
        b[0] = 1.0
        c = B @ w  # + 0.2 * np.random.rand(p2) - 0.1
        psi = 0.05

        # Solve using scipy
        c_obj = np.zeros(M + 1)
        c_obj[-1] = 1

        # Equality constraints: A*x = b
        A_eq = np.hstack([A, np.zeros((A.shape[0], 1))])
        b_eq = b

        # Inequality constraints:
        # Bx - c <= psi + s  -->  Bx - s <= psi + c
        A_ub1 = np.hstack([B, -np.ones((p2, 1))])
        b_ub1 = psi + c
        # -Bx + c <= psi + s  -->  -Bx - s <= psi - c
        A_ub2 = np.hstack([-B, -np.ones((p2, 1))])
        b_ub2 = psi - c
        A_ub = np.vstack([A_ub1, A_ub2])
        b_ub = np.concatenate([b_ub1, b_ub2])

        # Bounds: x >= lb, s free (or s >= 0 if desired)
        bounds = [(lb, None) for i in range(M)] + [(None, None)]  # s is unbounded

        # Solve
        st = time.time()
        scipy_res = optimize.linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        et = time.time()
        scipy_time = et - st

        # Results
        assert scipy_res.success, "Scipy failed to solve the problem"
        s_opt = scipy_res.x[-1]

        solver = EqualityWithBoundsAndImbalanceConstraintSolver(
            B=B,
            c=c,
            psi=psi,
            A=A,
            b=b,
            lb=lb,
            settings=OptimizationSettings(verbose=True, outer_tolerance_soft=0.01),
        )

        st = time.time()
        res = solver.solve()
        et = time.time()
        feasibility_time = et - st

        # Verify we found a feasible point
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.all(np.abs(B @ res.solution - c) < psi)

        # Verify we can fully optimize
        st = time.time()
        res = solver.solve(fully_optimize=True)
        et = time.time()
        print(
            f"Feasible point found with Cvxium in {1000 * (feasibility_time):.03f} ms"
        )
        print(f"Full optimization with Cvxium completed in {1000 * (et - st):.03f} ms")
        print(f"Scipy completed in {1000 * (scipy_time):.03f} ms")

        assert isinstance(res, InteriorPointMethodResult)
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.all(np.abs(B @ res.solution - c) < psi)
        assert abs(res.objective_value - s_opt) < 1e-4

        # Test with vector of psi and lb
        solver = EqualityWithBoundsAndImbalanceConstraintSolver(
            B=B,
            c=c,
            psi=np.full((p2,), psi),
            A=A,
            b=b,
            lb=np.full((M,), lb),
            settings=OptimizationSettings(verbose=True, outer_tolerance_soft=0.01),
        )

        st = time.time()
        res = solver.solve()
        et = time.time()
        feasibility_time = et - st
        print(
            f"Feasible point found with Cvxium in {1000 * (feasibility_time):.03f} ms"
        )

        # Verify we found a feasible point
        np.testing.assert_allclose(A @ res.solution, b)
        assert np.all(res.solution > lb)
        assert np.all(np.abs(B @ res.solution - c) < psi)

    @pytest.mark.parametrize(
        "seed,M,p1,p2",
        [
            (1302, 100, 10, 15),
            (2302, 200, 5, 30),
            (3302, 50, 5, 5),
            (4302, 500, 20, 30),
            (5302, 13, 3, 3),
        ],
    )
    def test_solver_infeasible(self, seed: int, M: int, p1: int, p2: int) -> None:
        """Test solver when problem is infeasible.

        We do this by generating problem data and feasible dual variables that lead to a
        dual function value exceeding the desired threshold.

        """
        np.random.seed(seed)

        nu = np.random.randn(p1 - 1)
        # Construct A such that A^T nu >= 0
        A = np.random.randn(p1 - 1, M)
        for ic in range(M):
            if np.dot(A[:, ic], nu) < 0:
                A[:, ic] = -A[:, ic]

        # Add a constraint on the mean weights
        assert np.all(A.T @ nu >= 0)
        A = np.vstack(
            [
                (1 / M)
                * np.ones(
                    M,
                ),
                A,
            ]
        )
        nu = np.concatenate([[1.0], nu])
        assert np.all(A.T @ nu >= 0)

        # Scale lmbda2 and lmbda3 so that sum(lmbda2) + sum(lbmda3) = 1
        lmbda2 = np.random.rand(p2)
        lmbda3 = np.random.rand(p2)
        theta = np.sum(lmbda2) + np.sum(lmbda3)
        lmbda2 /= theta
        lmbda3 /= theta

        assert abs(np.sum(lmbda2) + np.sum(lmbda3) - 1) <= 1e-6

        # Construct B such that B^T (lmbda3 - lmbda2) >= 0
        B = np.random.randn(p2, M)
        for ic in range(M):
            if np.dot(B[:, ic], lmbda3 - lmbda2) < 0:
                B[:, ic] = -B[:, ic]

        assert np.all(B.T @ (lmbda3 - lmbda2) >= 0)

        lmbda1 = A.T @ nu + B.T @ (lmbda3 - lmbda2)
        assert np.all(lmbda1 >= 0)

        # Construct w >= 0 and b = A @ w
        w = 0.5 + np.random.rand(M)
        # Scale w so that it has mean weight 1
        w /= np.mean(w)
        b = A @ w

        # Construct c so that dual function is > psi
        psi = 1.0
        c = np.random.randn(p2)
        if np.dot(c, lmbda2 - lmbda3) < 0:
            c = -c

        # Scale c so that -b^T nu + c^T (lmbda2 - lmbd3) > psi
        theta = (psi + np.dot(b, nu)) / np.dot(c, lmbda2 - lmbda3)
        c = 1.1 * theta * c

        assert -np.dot(b, nu) + np.dot(c, lmbda2 - lmbda3) > psi

        # Set up solver
        solver = EqualityWithBoundsAndImbalanceConstraintSolver(
            B=B,
            c=c,
            psi=psi,
            A=A,
            b=b,
            settings=OptimizationSettings(verbose=True),
        )
        with pytest.raises(ProblemCertifiablyInfeasibleError):
            solver.solve()
