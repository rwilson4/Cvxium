"""Tests for the HsdSolver skeleton (CVX-HSD-2)."""

import unittest
import unittest.mock

import numpy as np

from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.cone import (
    Cone,
    Nonneg,
    SecondOrderCone,
)
from cvxium.problems.linear_constraints import (
    EqualityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.problems.result import (
    SolveResult,
    SolveStatus,
)
from cvxium.solvers.hsd_solver import (
    HsdSolver,
    _HsdEmbedding,
)
from cvxium.solvers.settings import (
    OptimizationSettings,
    OuterSettings,
)

# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _trivial_problem() -> Problem:
    """A minimal Problem fixture for skeleton-stage tests.

    The Problem's actual mathematical content is irrelevant: every
    HsdSolver entry point raises NotImplementedError until HSD-5+.
    What we're testing here is the construction signature and the
    NotImplementedError plumbing, not the math.
    """
    return Problem(
        obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
    )


def _pin_apex_problem() -> Problem:
    """Apex-pin SOCP: ``min x_0 s.t. e_1^T x = 1, x in SOC^3``."""
    return Problem(
        obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
        eq=EqualityConstraintLayer(A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])),
    )


def _embedding_with_state(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    s: np.ndarray,
    kappa: float,
) -> _HsdEmbedding:
    n = c.shape[0]
    problem = Problem(
        obj=LinearObjective(c=c),
        eq=EqualityConstraintLayer(A=A, b=b) if A.size > 0 else None,
    )
    embedding = _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=n)])
    embedding._x = x.copy()
    embedding._y = y.copy()
    embedding._tau = tau
    embedding._s = s.copy()
    embedding._kappa = kappa
    return embedding


def _solve_tight(problem: Problem, cones: list[Cone]) -> SolveResult:
    """Solve with ``outer.tolerance=1e-6`` and zero starting iterate."""
    return HsdSolver(
        cones=cones,
        settings=OptimizationSettings(outer=OuterSettings(tolerance=1e-6)),
    ).solve(problem=problem, x0=np.zeros(problem.dimension))


def _build_dense_kkt(
    embedding: _HsdEmbedding,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the dense augmented KKT in the embedding's NT-symmetric form.

    Block layout (rows = eq I-V, cols = (Δx, Δy, Δτ, Δs, Δκ)):

        [  A     0    -b      0      0 ]
        [  0   -A^T   c    -I      0 ]
        [-c^T   b^T   0     0     -1 ]
        [W^-2   0     0     I      0 ]
        [  0     0     κ     0     τ  ]
    """
    # Assemble W^{-2} as block diagonal.
    x = embedding._x
    s = embedding._s
    assert x is not None and s is not None
    n = embedding._problem.dimension
    p = embedding._num_eq
    W_inv_sq = np.zeros((n, n))
    for cone, sl in zip(embedding._cones, embedding._cone_slices, strict=False):
        scaling = cone.nt_scaling(x[sl], s[sl])
        block_dim = cone.dimension()
        block = np.column_stack(
            [
                scaling.apply_W_inv(scaling.apply_W_inv(np.eye(block_dim)[:, i]))
                for i in range(block_dim)
            ]
        )
        W_inv_sq[sl, sl] = block

    A = embedding._A
    b = embedding._b
    c = embedding._c
    tau = embedding._tau
    kappa = embedding._kappa

    # Block layout: indices 0..n-1 = Δx, n..n+p-1 = Δy, n+p = Δτ,
    # n+p+1..2n+p = Δs, 2n+p+1 = Δκ.
    N = 2 * n + p + 2
    K = np.zeros((N, N))
    # Eq I (rows 0..p-1): A Δx - b Δτ = r_p (p equations).
    K[:p, 0:n] = A
    K[:p, n + p] = -b
    # Eq II (rows p..p+n-1): -A^T Δy - Δs + c Δτ = r_d (n equations).
    K[p : p + n, n : n + p] = -A.T
    K[p : p + n, n + p] = c
    K[p : p + n, n + p + 1 : 2 * n + p + 1] = -np.eye(n)
    # Eq III (row p+n): b^T Δy - c^T Δx - Δκ = r_g.
    K[p + n, 0:n] = -c
    K[p + n, n : n + p] = b
    K[p + n, 2 * n + p + 1] = -1.0
    # Eq IV (rows p+n+1..p+2n): W^{-2} Δx + Δs = r_xs (n equations).
    K[p + n + 1 : p + 2 * n + 1, 0:n] = W_inv_sq
    K[p + n + 1 : p + 2 * n + 1, n + p + 1 : 2 * n + p + 1] = np.eye(n)
    # Eq V (row p+2n+1): κ Δτ + τ Δκ = r_tk.
    K[p + 2 * n + 1, n + p] = kappa
    K[p + 2 * n + 1, 2 * n + p + 1] = tau

    return K, np.array([n, p, 1, n, 1])  # block sizes in row order


def _run_kkt_oracle_check(
    problem: Problem,
    cones: list[Cone],
    rng_seed: int,
) -> None:
    rng = np.random.default_rng(rng_seed)
    embedding = _HsdEmbedding(problem=problem, cones=cones)
    embedding.initial_iterate()
    # ``initial_iterate`` already places the iterate strictly inside the
    # symmetric cone (canonical e_K + unit slacks), so NT scaling is
    # well-defined here without any further perturbation.
    n = problem.dimension
    # Random RHS values.
    p_eq = embedding._num_eq
    rhs_p = rng.standard_normal(p_eq)
    rhs_d = rng.standard_normal(n)
    rhs_g = float(rng.standard_normal())
    rhs_xs = rng.standard_normal(n)
    rhs_tk = float(rng.standard_normal())

    delta_x, delta_y, delta_tau, delta_s, delta_kappa = embedding.kkt_solve(
        rhs_p, rhs_d, rhs_g, rhs_xs, rhs_tk
    )

    # Dense oracle.
    K, _ = _build_dense_kkt(embedding)
    rhs_dense = np.concatenate([rhs_p, rhs_d, [rhs_g], rhs_xs, [rhs_tk]])
    sol = np.linalg.solve(K, rhs_dense)
    # Slice back the unknowns.
    x_oracle = sol[0:n]
    y_oracle = sol[n : n + p_eq]
    tau_oracle = float(sol[n + p_eq])
    s_oracle = sol[n + p_eq + 1 : 2 * n + p_eq + 1]
    kappa_oracle = float(sol[2 * n + p_eq + 1])

    np.testing.assert_allclose(delta_x, x_oracle, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(delta_y, y_oracle, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(delta_tau, tau_oracle, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(delta_s, s_oracle, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(delta_kappa, kappa_oracle, rtol=1e-9, atol=1e-10)


# --------------------------------------------------------------------------- #
# HsdSolver tests
# --------------------------------------------------------------------------- #


class HsdSolverTest(unittest.TestCase):
    # ----------------------------------------------------------------------- #
    # HsdSolver construction invariants
    # ----------------------------------------------------------------------- #

    def test_construct_with_single_cone(self) -> None:
        solver = HsdSolver(cones=[SecondOrderCone(dim=3)])
        assert isinstance(solver, HsdSolver)

    def test_construct_with_multiple_cones(self) -> None:
        """Product cone: SOC of dim 3 x SOC of dim 5."""
        solver = HsdSolver(cones=[SecondOrderCone(dim=3), SecondOrderCone(dim=5)])
        assert isinstance(solver, HsdSolver)

    def test_continue_policy_raises_not_implemented(self) -> None:
        """HSD rejects ``on_infeasible="continue"`` -- it has no Phase I slack to drive."""
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])
            ),
        )
        solver = HsdSolver(cones=[SecondOrderCone(dim=3)])
        with self.assertRaisesRegex(NotImplementedError, "continue"):
            problem.solve(solver, x0=np.zeros(3), on_infeasible="continue")

    # --------------------------------------------------------------------------- #
    # HsdSolver.solve convergence on a known-optimum SOCP
    # --------------------------------------------------------------------------- #

    def test_pin_apex_socp(self) -> None:
        """``min x_0 s.t. e_1^T x = 1, x in SOC^3`` -> ``x* = (1, 0, 0)``.

        HSD's long-step driver should drive ``mu`` below ``outer.tolerance``
        within 100 iterations and recover ``x_star`` after ``tau``-rescaling.
        """
        c = np.array([1.0, 0.0, 0.0])
        A = np.array([[1.0, 0.0, 0.0]])
        b = np.array([1.0])
        problem = Problem(
            obj=LinearObjective(c=c),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        solver = HsdSolver(
            cones=[SecondOrderCone(dim=3)],
            settings=OptimizationSettings(outer=OuterSettings(tolerance=1e-6)),
        )
        result = solver.solve(problem=problem, x0=np.zeros(3))
        # Recover the original optimum: x_star = embedded_x / tau.
        np.testing.assert_allclose(
            result.solution, np.array([1.0, 0.0, 0.0]), atol=1e-3
        )
        assert result.suboptimality < 1e-6
        # Tau may collapse jointly with x as both approach the embedded
        # apex (preserving the ratio that recovers the original optimum);
        # HSD-9 will turn the tau-collapse pattern into a proper
        # infeasibility / unboundedness check. For HSD-7 we just verify
        # the recovered solution is correct.
        assert result.hsd_diagnostics.tau_history[-1] > 0.0

    # --------------------------------------------------------------------------- #
    # HSD-8 termination semantics: status codes + populated result fields
    # --------------------------------------------------------------------------- #

    def test_status_zero_on_hard_tolerance(self) -> None:
        solver = HsdSolver(
            cones=[SecondOrderCone(dim=3)],
            settings=OptimizationSettings(
                outer=OuterSettings(tolerance=1e-6, tolerance_soft=1e-3),
            ),
        )
        result = solver.solve(problem=_pin_apex_problem(), x0=np.zeros(3))
        assert result.status is SolveStatus.OPTIMAL
        assert "converged to outer.tolerance" in result.message

    def test_status_one_on_soft_tolerance(self) -> None:
        """When the hard tolerance is unreachable but the soft is met, status = 1."""
        # Hard tolerance impossibly tight (1e-30) but the soft to a
        # value the long-step driver actually reaches.
        solver = HsdSolver(
            cones=[SecondOrderCone(dim=3)],
            settings=OptimizationSettings(
                outer=OuterSettings(tolerance=1e-30, tolerance_soft=1e-4),
            ),
        )
        result = solver.solve(problem=_pin_apex_problem(), x0=np.zeros(3))
        assert result.status is SolveStatus.ACCEPTABLE
        assert "outer.tolerance_soft" in result.message

    def test_result_fields_populated(self) -> None:
        solver = HsdSolver(cones=[SecondOrderCone(dim=3)])
        result = solver.solve(problem=_pin_apex_problem(), x0=np.zeros(3))
        # Equality multipliers have one entry (one equality constraint).
        assert result.equality_multipliers.shape == (1,)
        # Inequality multipliers (s recovered) match the cone dimension.
        assert result.inequality_multipliers.shape == (3,)
        # Histories are populated and consistent in length.
        diag = result.hsd_diagnostics
        assert len(diag.tau_history) > 1
        assert len(diag.tau_history) == len(diag.kappa_history)
        assert len(diag.duality_gaps) == len(diag.tau_history)
        # n_iterations matches the number of step transitions.
        assert result.n_iterations == len(diag.tau_history) - 1

    def test_primal_value_matches_at_optimum(self) -> None:
        """At convergence, the recovered primal objective matches the true optimum.

        Strong-duality (b^T y = c^T x) requires reliable dual recovery,
        which depends on the embedded ``tau`` not collapsing -- deferred
        to HSD-9 (certificate machinery + tau-floor handling).
        """
        solver = HsdSolver(cones=[SecondOrderCone(dim=3)])
        result = solver.solve(problem=_pin_apex_problem(), x0=np.zeros(3))
        self.assertAlmostEqual(result.objective_value, 1.0, delta=1e-3)
        assert result.suboptimality < 1e-6

    # --------------------------------------------------------------------------- #
    # _HsdEmbedding.detect_infeasibility unit tests
    # --------------------------------------------------------------------------- #

    def test_no_certificate_when_ratio_below_threshold(self) -> None:
        """Joint-collapse pattern (kappa = tau) returns no certificate."""
        embedding = _embedding_with_state(
            A=np.array([[1.0, 0.0, 0.0]]),
            b=np.array([1.0]),
            c=np.array([1.0, 0.0, 0.0]),
            x=np.array([1e-6, 0.0, 0.0]),
            y=np.array([1e-6]),
            tau=1e-6,
            s=np.array([1e-6, 0.0, 0.0]),
            kappa=1e-6,
        )
        primal_cert, dual_cert, msg = embedding.detect_infeasibility()
        assert primal_cert is None
        assert dual_cert is None
        assert msg == ""

    def test_primal_infeasibility_certificate(self) -> None:
        """``kappa >> tau`` with ``b^T y > 0`` yields a primal-infeas cert."""
        # Hand-pick state mimicking the asymptote of an infeasible problem:
        # tau collapsed, kappa dominant, and the recovered y has b^T y > 0.
        embedding = _embedding_with_state(
            A=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            b=np.array([1.0, 1.0]),
            c=np.array([0.0, 0.0, 0.0]),
            x=np.array([0.5, 0.0, 0.0]),
            y=np.array([0.4, 0.4]),  # b^T y = 0.8 > 0
            tau=1e-8,
            s=np.array([1e-8, 0.0, 0.0]),
            kappa=1.0,
        )
        primal_cert, dual_cert, msg = embedding.detect_infeasibility()
        assert primal_cert is not None
        assert dual_cert is None
        assert "primal infeasibility" in msg
        np.testing.assert_array_equal(primal_cert, np.array([0.4, 0.4]))

    def test_dual_infeasibility_certificate(self) -> None:
        """``kappa >> tau`` with ``c^T x < 0`` yields a dual-infeas cert."""
        embedding = _embedding_with_state(
            A=np.zeros((0, 3)),
            b=np.zeros(0),
            c=np.array([-1.0, 0.0, 0.0]),
            x=np.array([1.0, 0.5, 0.0]),  # c^T x = -1.0 < 0
            y=np.zeros(0),
            tau=1e-8,
            s=np.array([1e-8, 0.0, 0.0]),
            kappa=1.0,
        )
        primal_cert, dual_cert, msg = embedding.detect_infeasibility()
        assert primal_cert is None
        assert dual_cert is not None
        assert "dual infeasibility" in msg
        np.testing.assert_array_equal(dual_cert, np.array([1.0, 0.5, 0.0]))

    def test_marginal_infeasibility_no_cert(self) -> None:
        """``kappa >> tau`` but neither sign condition fires: no cert, message preserved."""
        embedding = _embedding_with_state(
            A=np.array([[1.0, 0.0, 0.0]]),
            b=np.array([1.0]),
            c=np.array([0.0, 0.0, 0.0]),
            x=np.array([1e-8, 0.0, 0.0]),  # c^T x = 0
            y=np.array([0.0]),  # b^T y = 0
            tau=1e-8,
            s=np.array([1e-8, 0.0, 0.0]),
            kappa=1.0,
        )
        primal_cert, dual_cert, msg = embedding.detect_infeasibility()
        assert primal_cert is None
        assert dual_cert is None
        assert "marginally infeasible" in msg

    # --------------------------------------------------------------------------- #
    # End-to-end: HsdSolver.solve wires detect_infeasibility into HsdResult
    # --------------------------------------------------------------------------- #

    def test_feasible_problem_no_cert_through_solve(self) -> None:
        """Apex-pin SOCP is feasible; cert fields stay None through the full driver."""
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])
            ),
        )
        solver = HsdSolver(cones=[SecondOrderCone(dim=3)])
        result = solver.solve(problem=problem, x0=np.zeros(3))
        assert result.certificate is None

    def test_primal_cert_promotes_to_infeasible_in_build_result(self) -> None:
        """Primal cert from detect_infeasibility is lifted to INFEASIBLE status in _build_result."""
        # Use a crafted embedding state; the cert itself is injected via mock so the
        # problem does not need to be actually infeasible.
        embedding = _embedding_with_state(
            A=np.array([[1.0, 0.0, 0.0]]),
            b=np.array([1.0]),
            c=np.array([1.0, 0.0, 0.0]),
            x=np.array([1.0, 0.0, 0.0]),
            y=np.array([1.0]),
            tau=1.0,
            s=np.array([1.0, 0.0, 0.0]),
            kappa=1.0,
        )
        primal_cert = np.array([0.5])
        with unittest.mock.patch.object(
            embedding,
            "detect_infeasibility",
            return_value=(primal_cert, None, "primal infeasibility"),
        ):
            result = HsdSolver._build_result(
                embedding=embedding,
                mu=1.0,
                status=2,
                message="max iters",
                tau_history=[1.0, 1.0],
                kappa_history=[1.0, 2.0],
                gap_history=[1.0, 0.5],
            )
        assert result.status is SolveStatus.INFEASIBLE
        assert result.certificate is not None
        assert result.certificate.kind == "infeasible"

    def test_dual_cert_promotes_to_unbounded_in_build_result(self) -> None:
        """Dual cert from detect_infeasibility is lifted to UNBOUNDED status in _build_result."""
        embedding = _embedding_with_state(
            A=np.array([[1.0, 0.0, 0.0]]),
            b=np.array([1.0]),
            c=np.array([1.0, 0.0, 0.0]),
            x=np.array([1.0, 0.0, 0.0]),
            y=np.array([1.0]),
            tau=1.0,
            s=np.array([1.0, 0.0, 0.0]),
            kappa=1.0,
        )
        dual_cert = np.array([1.0, 0.0, 0.0])
        with unittest.mock.patch.object(
            embedding,
            "detect_infeasibility",
            return_value=(None, dual_cert, "dual infeasibility"),
        ):
            result = HsdSolver._build_result(
                embedding=embedding,
                mu=1.0,
                status=2,
                message="max iters",
                tau_history=[1.0, 1.0],
                kappa_history=[1.0, 2.0],
                gap_history=[1.0, 0.5],
            )
        assert result.status is SolveStatus.UNBOUNDED
        assert result.certificate is not None
        assert result.certificate.kind == "unbounded"

    # --------------------------------------------------------------------------- #
    # HSD-10 validation across structural patterns + graceful boundary handling
    # --------------------------------------------------------------------------- #

    def test_apex_pinned_socp_recovers_optimum(self) -> None:
        """``min x_0 s.t. e_1^T x = 1, x in SOC^3`` -> ``x* = (1, 0, 0)``.

        Apex-direction recovery: tau and x both collapse together,
        preserving the recovery ratio. Status = 0 expected.
        """
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])
            ),
        )
        result = _solve_tight(problem, [SecondOrderCone(dim=3)])
        np.testing.assert_allclose(
            result.solution, np.array([1.0, 0.0, 0.0]), atol=1e-3
        )
        assert result.status is SolveStatus.OPTIMAL

    def test_radial_boundary_terminates_gracefully_or_converges(self) -> None:
        """``min x_1 s.t. x_0 = 3, x in SOC^3`` -> optimum on radial boundary.

        Embedded x converges to the cone boundary (``x^T J x -> 0``)
        even with tau bounded, where NT scaling becomes singular. The
        long-step driver should EITHER converge before hitting that wall
        OR terminate cleanly with status 1/2 and an informative message.
        Mehrotra adaptive sigma is the production-grade fix; this test
        pins the graceful-degradation contract HSD-7 / HSD-10 ship.
        """
        problem = Problem(
            obj=LinearObjective(c=np.array([0.0, 1.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([3.0])
            ),
        )
        result = _solve_tight(problem, [SecondOrderCone(dim=3)])
        assert result.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.ACCEPTABLE,
            SolveStatus.MAX_ITERATIONS,
        )
        assert isinstance(result.message, str)
        assert len(result.message) > 0
        # No crash; solution and history shape are populated.
        assert result.solution.shape == (3,)
        assert len(result.hsd_diagnostics.tau_history) > 1

    def test_product_cone_returns_without_raising(self) -> None:
        """Two-block product cone: ``solve`` returns even if the driver doesn't converge.

        The long-step driver doesn't reliably converge on every product-
        cone fixture (Mehrotra adaptive sigma is the production-grade
        fix), but the API contract is that ``solve`` always returns an
        ``HsdResult`` -- with a status code reflecting what happened --
        rather than raising or hanging. This test pins that contract.
        """
        A = np.array(
            [
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        b = np.array([0.0, 2.0])
        problem = Problem(
            obj=LinearObjective(c=np.zeros(6)),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = _solve_tight(problem, [SecondOrderCone(dim=3), SecondOrderCone(dim=3)])
        assert result.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.ACCEPTABLE,
            SolveStatus.MAX_ITERATIONS,
        )
        assert result.solution.shape == (6,)
        assert isinstance(result.message, str)

    # --------------------------------------------------------------------------- #
    # Public HsdSolver.solve interface sanity
    # --------------------------------------------------------------------------- #

    def test_solve_returns_hsd_result(self) -> None:
        """``solve`` returns a SolveResult instance tagged as the HSD solver."""
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])
            ),
        )
        solver = HsdSolver(cones=[SecondOrderCone(dim=3)])
        result = solver.solve(problem=problem, x0=np.zeros(3))
        assert isinstance(result, SolveResult)
        assert result.solver == "hsd"

    # --------------------------------------------------------------------------- #
    # _HsdEmbedding.__init__ validation
    # --------------------------------------------------------------------------- #

    def test_dimension_sum_mismatch_raises(self) -> None:
        problem = _trivial_problem()  # dimension = 3
        with self.assertRaisesRegex(ValueError, "Sum of cone dimensions"):
            _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=2)])

    def test_non_linear_objective_raises(self) -> None:
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
        )
        with self.assertRaisesRegex(ValueError, "LinearObjective"):
            _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=3)])

    # --------------------------------------------------------------------------- #
    # _HsdEmbedding.initial_iterate: canonical strictly-interior start
    # --------------------------------------------------------------------------- #

    def test_initial_iterate_single_cone_no_equality(self) -> None:
        cone = SecondOrderCone(dim=3)
        embedding = _HsdEmbedding(
            problem=Problem(
                obj=LinearObjective(c=np.zeros(3)),
            ),
            cones=[cone],
        )
        embedding.initial_iterate()
        np.testing.assert_array_equal(embedding._x, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(embedding._s, np.array([1.0, 0.0, 0.0]))
        assert embedding._y is not None
        assert embedding._y.shape == (0,)
        assert embedding._tau == 1.0
        assert embedding._kappa == 1.0

    def test_initial_iterate_multi_cone_product(self) -> None:
        embedding = _HsdEmbedding(
            problem=Problem(
                obj=LinearObjective(c=np.zeros(5)),
            ),
            cones=[SecondOrderCone(dim=2), SecondOrderCone(dim=3)],
        )
        embedding.initial_iterate()
        # Stacked cone identities: (1, 0) ++ (1, 0, 0).
        np.testing.assert_array_equal(embedding._x, np.array([1.0, 0.0, 1.0, 0.0, 0.0]))

    def test_initial_iterate_with_equality_constructs_y_with_correct_length(
        self,
    ) -> None:
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([0.5])
            ),
        )
        embedding = _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=3)])
        embedding.initial_iterate()
        assert embedding._y is not None
        assert embedding._y.shape == (1,)
        np.testing.assert_array_equal(embedding._y, np.array([0.0]))

    # --------------------------------------------------------------------------- #
    # _HsdEmbedding.residuals: Andersen-Roos-Terlaky embedding equations
    # --------------------------------------------------------------------------- #

    def test_residuals_at_canonical_start_no_equality_zero_objective(self) -> None:
        """``A = empty``, ``c = 0``: only ``r_d``, ``r_xs``, ``r_g``, ``r_tk`` are non-trivial."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(3)),
        )
        embedding = _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=3)])
        embedding.initial_iterate()
        r_p, r_d, r_g, r_xs, r_tk = embedding.residuals()
        # r_p = -A x + b tau; both empty -> shape (0,).
        assert r_p.shape == (0,)
        # r_d = A^T y + s - c tau = 0 + (1, 0, 0) - 0 = (1, 0, 0).
        np.testing.assert_array_equal(r_d, np.array([1.0, 0.0, 0.0]))
        # r_g = -b^T y + c^T x + kappa = 0 + 0 + 1 = 1.
        assert r_g == 1.0
        # r_xs = e . e = (1, 0, 0) for SOC.
        np.testing.assert_array_equal(r_xs, np.array([1.0, 0.0, 0.0]))
        # r_tk = tau * kappa = 1.
        assert r_tk == 1.0

    def test_residuals_with_equality_and_nonzero_c(self) -> None:
        """Hand-computed canonical-start residuals on a small fixture."""
        c = np.array([1.0, 0.0, 0.0])
        A = np.array([[1.0, 0.0, 0.0]])
        b = np.array([0.5])
        problem = Problem(
            obj=LinearObjective(c=c),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        embedding = _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=3)])
        embedding.initial_iterate()
        r_p, r_d, r_g, r_xs, r_tk = embedding.residuals()
        # x = (1, 0, 0), tau = 1, y = (0,), kappa = 1, s = (1, 0, 0).
        # r_p = -A x + b tau = -1 + 0.5 = -0.5.
        np.testing.assert_array_equal(r_p, np.array([-0.5]))
        # r_d = A^T y + s - c tau = 0 + (1, 0, 0) - (1, 0, 0) = 0.
        np.testing.assert_array_equal(r_d, np.zeros(3))
        # r_g = -b^T y + c^T x + kappa = 0 + 1 + 1 = 2.
        assert r_g == 2.0
        np.testing.assert_array_equal(r_xs, np.array([1.0, 0.0, 0.0]))
        assert r_tk == 1.0

    def test_residuals_multi_cone_xs_stacked_per_block(self) -> None:
        """Two SOC cones; ``r_xs`` is the concatenation of per-block Jordan products."""
        problem = Problem(
            obj=LinearObjective(c=np.zeros(5)),
        )
        embedding = _HsdEmbedding(
            problem=problem,
            cones=[SecondOrderCone(dim=2), SecondOrderCone(dim=3)],
        )
        embedding.initial_iterate()
        _, _, _, r_xs, _ = embedding.residuals()
        # Each block is e . e = (1, 0, ...) of its cone's dimension.
        np.testing.assert_array_equal(r_xs, np.array([1.0, 0.0, 1.0, 0.0, 0.0]))

    def test_residuals_before_initial_iterate_raises(self) -> None:
        problem = _trivial_problem()
        embedding = _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=3)])
        with self.assertRaisesRegex(RuntimeError, "initial_iterate"):
            embedding.residuals()

    # --------------------------------------------------------------------------- #
    # _HsdEmbedding.kkt_solve: NT-symmetric Newton system against dense oracle
    # --------------------------------------------------------------------------- #

    def test_kkt_solve_single_cone_no_equality(self) -> None:
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
        )
        _run_kkt_oracle_check(problem, [SecondOrderCone(dim=3)], rng_seed=10001)

    def test_kkt_solve_single_cone_with_equality(self) -> None:
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([0.5])
            ),
        )
        _run_kkt_oracle_check(problem, [SecondOrderCone(dim=3)], rng_seed=10002)

    def test_kkt_solve_multi_cone_with_equality(self) -> None:
        # Two SOC blocks, dim 2 and dim 3, with two equality constraints.
        A = np.array([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]])
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.5, 0.0, 0.0])),
            eq=EqualityConstraintLayer(A=A, b=np.array([0.7, 0.2])),
        )
        _run_kkt_oracle_check(
            problem,
            [SecondOrderCone(dim=2), SecondOrderCone(dim=3)],
            rng_seed=10003,
        )

    def test_kkt_solve_mixed_soc_and_nonneg_with_equality(self) -> None:
        """Product cone ``SOC(3) x Nonneg(2)`` matches the dense KKT oracle.

        This is the configuration MarkowitzSolver and other Cvxium callers
        with bounds + SOC will exercise — verifies the polymorphic Cone
        dispatch composes correctly across cone families.
        """
        # Dim 5 = SOC(3) || Nonneg(2). One equality.
        A = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0, 0.5, 0.5])),
            eq=EqualityConstraintLayer(A=A, b=np.array([1.0])),
        )
        _run_kkt_oracle_check(
            problem,
            [SecondOrderCone(dim=3), Nonneg(dim=2)],
            rng_seed=10004,
        )

    def test_kkt_solve_before_initial_iterate_raises(self) -> None:
        problem = _trivial_problem()
        embedding = _HsdEmbedding(problem=problem, cones=[SecondOrderCone(dim=3)])
        with self.assertRaisesRegex(RuntimeError, "initial_iterate"):
            embedding.kkt_solve(
                rhs_p=np.zeros(0),
                rhs_d=np.zeros(3),
                rhs_g=0.0,
                rhs_xs=np.zeros(3),
                rhs_tk=0.0,
            )
