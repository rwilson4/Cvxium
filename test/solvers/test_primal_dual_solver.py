"""Test the PrimalDualSolver, including the equality-constrained path."""

import unittest

import numpy as np
import numpy.typing as npt
from parameterized import parameterized

from cvxium._typing import none_throws
from cvxium.exceptions import NewtonStepError
from cvxium.linalg.systems import (
    DenseSystem,
    DiagonalSystem,
    MatrixSystem,
)
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    ImbalanceConstraintLayer,
    LinearInequalityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.problems.quadratic_constraints import (
    NormConstraintLayer,
    SquaredNormConstraintLayer,
)
from cvxium.problems.result import (
    PrimalDualDiagnostics,
    SolveResult,
    SolveStatus,
)
from cvxium.solvers.barrier_solver import (
    BarrierSolver,
)
from cvxium.solvers.line_search import (
    fraction_to_boundary,
)
from cvxium.solvers.primal_dual_solver import (
    PrimalDualSolver,
    _clamp_dual_after_update,
    _PrimalDualKernel,
    _StallWatch,
)
from cvxium.solvers.settings import (
    LevenbergSettings,
    OptimizationSettings,
    OuterSettings,
    PrimalDualSettings,
)


class _BoundsPlusRankPStub(BoundsConstraintLayer):
    """Test fixture: BoundsConstraintLayer with a synthetic extra rank-p contribution.

    Exercises the structured PD assembly (``Problem.build_hessian_system`` ->
    ``layer.hessian_contribution``) folding a layer's non-trivial curvature into
    the modified Hessian, without depending on any production layer. The synthetic
    extra does not correspond to a real constraint; the test checks composition,
    not the solution of a real problem.

    The full contribution is the bounds diagonal (`I diag(\\lambda/-f) I` for
    ``lb=0``) plus the synthetic rank-`p` extra ``kappa_extra diag(d_extra)
    kappa_extra^T`` carried as a low-rank update on top.
    """

    def __init__(
        self,
        lb: float,
        kappa: npt.NDArray[np.float64],
        d: npt.NDArray[np.float64],
    ) -> None:
        super().__init__(lb=lb, dimension=kappa.shape[0])
        self._kappa_extra = kappa
        self._d_extra = d

    def hessian_contribution(
        self,
        w: npt.NDArray[np.float64],
        lmbda: npt.NDArray[np.float64] | None = None,
        shift: float = 0.0,
    ) -> MatrixSystem:
        # Bounds (lb=0): the diagonal I diag(lambda/-f) I, plus the synthetic
        # rank-p extra carried as a low-rank update on top.
        base = super().hessian_contribution(w, lmbda, shift)
        return base.low_rank_update(self._kappa_extra, self._d_extra)


def _make_psd_matrix(rng: np.random.Generator, M: int) -> npt.NDArray[np.float64]:
    """Sample a random symmetric PD matrix with bounded condition number."""
    A = rng.standard_normal((M, M))
    return np.asarray(A @ A.T + M * np.eye(M))


def _dense_pd_newton_step(
    H_pd: npt.NDArray[np.float64],
    G: npt.NDArray[np.float64],
    lmbda: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    r_dual: npt.NDArray[np.float64],
    r_cent: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Reference Newton step for the PD modified KKT system.

    Solves the Schur-eliminated block

       H_pd \Delta x = -r_dual - G^T (r_cent / f),
       \Delta \lambda = (r_cent - \lambda \odot (G \Delta x)) / f,

    via ``np.linalg.solve``. Used as a snapshot oracle for
    ``PrimalDualSolver._compute_pd_step``: the structured assembly
    (which exploits the objective's and layers' structured
    ``hessian_contribution``) must agree with this brute-force
    reference at high precision. Caller assembles ``H_pd`` (objective
    Hessian plus the PD-weighted layer terms) so the helper stays
    agnostic to the layer mix; new layer support in P3.2-P3.5 just
    extends how callers build ``H_pd``.
    """
    rhs = -r_dual - G.T @ (r_cent / f)
    delta_x = np.asarray(np.linalg.solve(H_pd, rhs), dtype=np.float64)
    delta_lmbda = (r_cent - lmbda * (G @ delta_x)) / f
    return delta_x, delta_lmbda


class PrimalDualSolverTest(unittest.TestCase):
    """Tests for `PrimalDualSolver` (no equality)."""

    def test_solves_problem_with_equality(self) -> None:
        """`solve` auto-dispatches a Problem carrying an equality constraint."""
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(2), hessian_contribution=DiagonalSystem(np.ones(2))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=2)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = PrimalDualSolver().solve(problem=problem, x0=np.array([0.5, 0.5]))
        x = np.asarray(result.solution)
        assert problem.is_feasible(x)
        np.testing.assert_allclose(A @ x, b, atol=1e-6)

    def test_run_equality_raises_runtime_error_when_equality_is_none(self) -> None:
        """_run_equality defends against a missing equality system under `python -O`.

        `solve` already validates `problem.eq is not None`, but `assert` is
        stripped under optimised builds. `_run_equality` uses an explicit
        `if ... is None: raise RuntimeError(...)` so the runtime guard survives.

        """
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(2), hessian_contribution=DiagonalSystem(np.ones(2))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=2)],
        )
        with self.assertRaisesRegex(RuntimeError, "problem.eq is not None"):
            # Bypassing solve()'s pre-check is the point of the test.
            PrimalDualSolver()._run_equality(problem=problem, x0=np.ones(2))

    def test_rejects_problem_without_constraints(self) -> None:
        """`solve` raises when the Problem has no inequality constraints."""
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
        )
        with self.assertRaisesRegex(ValueError, "NewtonSolver"):
            PrimalDualSolver().solve(problem=problem, x0=np.zeros(3))

    @parameterized.expand(
        [
            (1301, 5),
            (1302, 15),
        ]
    )
    def test_solves_bound_constrained_quadratic_unconstrained_optimum(
        self, seed: int, M: int
    ) -> None:
        """When the unconstrained optimum is feasible, PrimalDualSolver finds it."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = rng.uniform(0.5, 1.0, size=M)
        result = PrimalDualSolver().solve(problem=problem, x0=x0)

        assert isinstance(result, SolveResult)
        assert isinstance(result.diagnostics, PrimalDualDiagnostics)
        np.testing.assert_allclose(result.solution, w_target, rtol=1e-4, atol=1e-4)

    def test_returns_pd_specific_diagnostics(self) -> None:
        """`PrimalDualResult` carries primal/dual residuals and surrogate-gap trajectory."""
        rng = np.random.default_rng(1311)
        M = 5
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        result = PrimalDualSolver().solve(
            problem=problem, x0=rng.uniform(0.5, 1.0, size=M)
        )
        diag = result.primal_dual_diagnostics
        # No equality -> primal residual identically zero at every iteration.
        assert all(p == 0.0 for p in diag.primal_residual_norms)
        assert diag.dual_residual_norms[-1] < 1e-4
        assert len(diag.dual_residual_norms) == result.n_iterations
        assert len(diag.surrogate_gaps) == result.n_iterations
        # Surrogate gap should fall over iterations.
        assert diag.surrogate_gaps[-1] < diag.surrogate_gaps[0]

    @parameterized.expand(
        [
            (1321, 5),
            (1322, 10),
        ]
    )
    def test_matches_barrier_solver(self, seed: int, M: int) -> None:
        """PrimalDualSolver and BarrierSolver agree on the same problem to 1e-4."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = rng.uniform(0.5, 1.0, size=M)

        pd_result = PrimalDualSolver().solve(problem=problem, x0=x0)
        barrier_result = BarrierSolver().solve(problem=problem, x0=x0)

        np.testing.assert_allclose(
            pd_result.solution, barrier_result.solution, rtol=1e-4, atol=1e-4
        )

    @parameterized.expand(
        [
            (1321, 5),
            (1322, 10),
        ]
    )
    def test_dual_value_is_sound_and_matches_barrier(self, seed: int, M: int) -> None:
        """PD's dual value is a valid lower bound and agrees with BarrierSolver.

        Both solvers now compute the Lagrangian dual through the barrier's
        mini-Newton ``evaluate_dual`` -- a guaranteed lower bound on
        `\\inf_x L(x, \\lambda)` (B&V Eq 9.42) -- rather than the analytic
        per-component decomposition whose plug-in fallback (``L`` at the current
        ``x``) was only an UPPER bound and could falsely certify a feasible
        problem infeasible. The dual must therefore (a) not exceed the primal
        optimum (weak duality) and (b) match the barrier solver's dual.
        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        x0 = rng.uniform(0.5, 1.0, size=M)

        pd_result = PrimalDualSolver().solve(problem=problem, x0=x0)
        barrier_result = BarrierSolver().solve(problem=problem, x0=x0)

        # Sound lower bound: dual <= primal optimum (non-negative duality gap).
        assert pd_result.dual_value <= pd_result.objective_value + 1e-6
        # Same evaluator on (near-)identical optima -> same dual.
        self.assertAlmostEqual(
            pd_result.dual_value, barrier_result.dual_value, delta=1e-3
        )

    def test_solves_lp_corner(self) -> None:
        """LP `min c^T w  s.t. lb <= w <= ub` finds the lower-bound corner."""
        M = 5
        c = np.ones(M)
        problem = Problem(
            obj=LinearObjective(c=c),
            ineq=[BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M)],
        )
        x0 = 5.0 * np.ones(M)  # interior
        result = PrimalDualSolver().solve(problem=problem, x0=x0)
        np.testing.assert_allclose(result.solution, np.zeros(M), atol=1e-3)

    def test_converges_on_near_singular_hessian_via_damping(self) -> None:
        """Near-singular objective Hessian + far-from-boundary start: the raw PD
        Newton step blows up; adaptive Levenberg damping recovers it.

        With a near-zero eigenvalue in ``Q`` and the box bounds placed far away,
        the constraint/centering right-hand side excites the tiny-eigenvalue
        direction and the undamped step is ``~ ||rhs|| / sigma_min(Q)`` (~1e10
        here). Before damping this is exactly the CDM Cobb-Douglas
        saturation-linear failure in miniature: no feasible step exists for the
        raw direction at iteration 0. With damping the solve must converge
        instead of blowing up. The minimizer itself is *not*
        asserted: the 1e-10 eigenvalue leaves the objective nearly flat along one
        direction, so the optimal ``x`` is poorly determined there (the converged
        objective is well-defined, the point is not). The guard is therefore
        convergence + feasibility + residual decrease.
        """
        rng = np.random.default_rng(2026)
        M = 6
        basis, _ = np.linalg.qr(rng.standard_normal((M, M)))
        eigs = np.array([1.0, 1.0, 1.0, 1e-3, 1e-6, 1e-10])
        Q = (basis * eigs) @ basis.T
        Q = np.asarray(0.5 * (Q + Q.T))  # symmetrize against round-off
        w_target = rng.uniform(2.0, 5.0, size=M)
        c = -Q @ w_target
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=-1e8, ub=1e8, dimension=M)],
        )
        x0 = np.ones(M)  # deep interior, far from the +/-1e8 bounds

        result = PrimalDualSolver(
            settings=OptimizationSettings(outer=OuterSettings(tolerance=1e-4))
        ).solve(problem=problem, x0=x0)

        assert isinstance(result, SolveResult)
        assert result.status is SolveStatus.OPTIMAL  # converged, not a NewtonStepError
        assert np.all(np.isfinite(result.solution))
        assert np.all(np.abs(result.solution) < 1e8)  # strictly within the box
        # Surrogate gap fell from the start -- no 1e16 blow-up.
        diag = result.primal_dual_diagnostics
        assert diag.surrogate_gaps[-1] < diag.surrogate_gaps[0]


class PdLevenbergDampingTest(unittest.TestCase):
    """Unit tests for the adaptive Levenberg damping schedule (`_relax_damping`)."""

    _LM = LevenbergSettings()  # default schedule

    def test_zero_damping_stays_zero(self) -> None:
        """No regularization in play -> nothing to relax."""
        lm = PdLevenbergDampingTest._LM
        assert PrimalDualSolver._relax_damping(0.0, 0.99, 1.0, lm) == 0.0

    def test_excellent_ratio_shrinks_hard(self) -> None:
        lm = PdLevenbergDampingTest._LM
        self.assertAlmostEqual(PrimalDualSolver._relax_damping(1.0, 0.9, 1.0, lm), 0.25)

    def test_acceptable_ratio_shrinks_gently(self) -> None:
        lm = PdLevenbergDampingTest._LM
        self.assertAlmostEqual(PrimalDualSolver._relax_damping(1.0, 0.5, 1.0, lm), 0.5)

    def test_poor_ratio_holds_damping(self) -> None:
        lm = PdLevenbergDampingTest._LM
        self.assertAlmostEqual(PrimalDualSolver._relax_damping(1.0, 0.1, 1.0, lm), 1.0)

    def test_snaps_to_zero_below_scaled_floor(self) -> None:
        """Once damping is negligible vs the iterate scale it drops to exact 0."""
        lm = PdLevenbergDampingTest._LM
        assert PrimalDualSolver._relax_damping(1e-13, 0.5, 1.0, lm) == 0.0

    def test_honors_custom_schedule(self) -> None:
        """The schedule is read from the passed ``LevenbergSettings``, not hard-coded."""
        lm = LevenbergSettings(shrink_good=0.1, rho_good=0.6)
        # rho=0.9 > rho_good(0.6) -> shrink by the custom 0.1, not the default 0.25.
        self.assertAlmostEqual(PrimalDualSolver._relax_damping(1.0, 0.9, 1.0, lm), 0.1)

    def test_settings_default_schedule_values(self) -> None:
        """``OptimizationSettings.primal_dual.damping`` exposes the LM schedule."""
        damping = OptimizationSettings().primal_dual.damping
        assert isinstance(damping, LevenbergSettings)
        assert damping.max_attempts == 40
        assert damping.grow == 4.0
        assert damping.rho_good == 0.75

    def test_grow_damping_honors_custom_grow(self) -> None:
        """``_grow_damping`` reads the multiplier from the passed schedule.

        (An end-to-end ``solve`` test can't easily force this path: MPC's
        fraction-to-boundary step control handles the near-singular start
        without ever engaging damping, so the damping-retry loop is rarely
        entered. The configurability is exercised at the unit level here and on
        ``_relax_damping`` above; the full solve suite covers the defaults.)
        """
        dummy = np.array([1.0, 1.0])
        out = PrimalDualSolver._grow_damping(
            PdInputGuardsTest._problem(),
            dummy,
            dummy,
            -dummy,
            dummy,
            dummy,
            damping=2.0,
            x_scale=1.0,
            lm=LevenbergSettings(grow=10.0),
        )
        # 2.0 * custom grow (10.0)
        np.testing.assert_allclose(out, 20.0, rtol=1e-6, atol=1e-12)

    def test_settings_construct_with_nested_primal_dual(self) -> None:
        """The PrimalDualSettings/LevenbergSettings nesting constructs as expected."""
        settings = OptimizationSettings(
            primal_dual=PrimalDualSettings(damping=LevenbergSettings(grow=8.0))
        )
        assert settings.primal_dual.damping.grow == 8.0


class MpcStepTest(unittest.TestCase):
    """Mehrotra predictor-corrector step primitives (``PrimalDualSolver``)."""

    def test_frac_to_boundary_no_negative_returns_one(self) -> None:
        """No component decreases, so a full step keeps every entry positive."""
        a = fraction_to_boundary(np.array([1.0, 2.0]), np.array([1.0, 0.5]))
        assert a == 1.0

    def test_frac_to_boundary_caps_at_binding_ratio(self) -> None:
        """The most-binding ``-v_i / dv_i`` over decreasing components sets the cap."""
        a = fraction_to_boundary(np.array([1.0, 2.0]), np.array([-4.0, -1.0]))
        self.assertAlmostEqual(a, 0.25)  # min(1, 1/4, 2/1)

    @staticmethod
    def _squared_norm_problem() -> Problem:
        """``min 0`` s.t. ``||x||^2 <= 1`` -- a nonlinear (curved) feasible set."""
        return Problem(
            obj=QuadraticObjective(
                c=np.zeros(2), hessian_contribution=DenseSystem(np.eye(2))
            ),
            ineq=[SquaredNormConstraintLayer(phi=1.0, dimension=2)],
        )

    def test_feasible_step_returns_strictly_feasible_step(self) -> None:
        """The returned step keeps the iterate strictly inside ``||x||^2 < 1``.

        The fraction-to-boundary cap is computed on the *linearized* constraint;
        on the curved squared-norm set the linear step overshoots, so the
        primal-feasibility shrink must pull it back. Whatever step is returned
        MUST be strictly feasible -- the no-silent-infeasible-iterate guarantee
        that the removed ``_backtracking_step`` used to enforce.
        """
        problem = MpcStepTest._squared_norm_problem()
        x = np.array([0.5, 0.5])
        f = np.array([float(np.dot(x, x)) - 1.0])
        dx = np.array([1.0, 1.0])  # outward, toward the boundary
        step = PrimalDualSolver._feasible_step(
            problem,
            x,
            f,
            dx,
            lmbda=np.array([1.0]),
            dlam=np.array([0.0]),
            gamma=1.0,  # max(tau_min=0.99, 1 - mu=0.0) == 1.0
            beta=0.5,
            min_step=1e-3,
        )
        assert step is not None
        x_trial = x + step * dx
        assert float(np.dot(x_trial, x_trial)) < 1.0

    def test_feasible_step_returns_none_when_no_step_above_min_step(self) -> None:
        """When every step >= ``min_step`` overshoots the curved boundary, return None.

        This is the MPC analog of the old BTLS raise: rather than silently take
        an infeasible step, ``_feasible_step`` reports failure so the caller
        grows the Levenberg damping and re-solves. Here the linear cap lands at
        ~0.25 but ``||x + 0.25 dx||^2 = 1.125 > 1``, and ``min_step = 0.5`` blocks
        any smaller (feasible) step.
        """
        problem = MpcStepTest._squared_norm_problem()
        x = np.array([0.5, 0.5])
        f = np.array([float(np.dot(x, x)) - 1.0])
        dx = np.array([1.0, 1.0])
        step = PrimalDualSolver._feasible_step(
            problem,
            x,
            f,
            dx,
            lmbda=np.array([1.0]),
            dlam=np.array([0.0]),
            gamma=1.0,  # max(tau_min=0.99, 1 - mu=0.0) == 1.0
            beta=0.5,
            min_step=0.5,
        )
        assert step is None


class PrimalDualSolverWithEqualityTest(unittest.TestCase):
    """Tests for `PrimalDualSolver` on equality-constrained problems."""

    def test_rejects_problem_without_constraints(self) -> None:
        """`solve` raises when the Problem has no inequality constraints."""
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        problem = Problem(
            obj=QuadraticObjective(
                c=np.zeros(2), hessian_contribution=DiagonalSystem(np.ones(2))
            ),
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        with self.assertRaisesRegex(ValueError, "NewtonSolver"):
            PrimalDualSolver().solve(problem=problem, x0=np.array([0.5, 0.5]))

    @parameterized.expand(
        [
            (1401, 5, "simplex"),
            (1402, 10, "simplex"),
            (1403, 8, "two_row"),
            (1404, 12, "rectangular"),
            (1405, 6, "non_unit_b"),
        ]
    )
    def test_matches_equality_barrier_solver(
        self, seed: int, M: int, A_shape: str
    ) -> None:
        """PrimalDualSolver and BarrierSolver agree across A shapes.

        Tight cross-solver agreement (rtol=1e-6) acts as a snapshot oracle for
        the refactor that routes both solvers through the shared
        ``saddle_system`` seam. Multi-row and rectangular ``A`` cases exercise
        the SVD/Schur paths that the simplex case alone cannot.
        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target

        # Constructions are chosen so x0 = ones is feasible for both equality
        # and inequality (lb=0) constraints.
        if A_shape == "simplex":
            A = np.ones((1, M))
            b = np.array([float(M)])
        elif A_shape == "two_row":
            assert M % 2 == 0, "two_row construction needs even M"
            half = M // 2
            A = np.vstack(
                [
                    np.ones(M),
                    np.concatenate([np.ones(half), -np.ones(half)]),
                ]
            )
            b = np.array([float(M), 0.0])
        elif A_shape == "rectangular":
            assert M % 2 == 0, "rectangular construction needs even M"
            half = M // 2
            alt = np.tile([1.0, -1.0], M // 2)
            A = np.vstack(
                [
                    np.ones(M),
                    np.concatenate([np.ones(half), -np.ones(half)]),
                    alt,
                ]
            )
            b = np.array([float(M), 0.0, 0.0])
        elif A_shape == "non_unit_b":
            A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
            b = A @ np.ones(M)
        else:
            raise ValueError(f"Unknown A_shape: {A_shape}")

        # Sanity: x0 = ones is equality-feasible.
        np.testing.assert_allclose(A @ np.ones(M), b, atol=1e-12)

        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = np.ones(M)

        pd_result = PrimalDualSolver().solve(problem=problem, x0=x0)
        barrier_result = BarrierSolver().solve(problem=problem, x0=x0)

        np.testing.assert_allclose(
            pd_result.solution, barrier_result.solution, rtol=1e-6, atol=1e-6
        )
        # PD should drive equality residual to zero.
        np.testing.assert_allclose(A @ pd_result.solution, b, atol=1e-6)
        # Barrier solver also satisfies equality (initialized via Schur).
        np.testing.assert_allclose(A @ barrier_result.solution, b, atol=1e-6)

    @parameterized.expand([(1401, 5), (1404, 12)])
    def test_dual_value_is_sound_and_matches_barrier(self, seed: int, M: int) -> None:
        """Equality PD's dual is a valid lower bound and agrees with the barrier.

        Mirrors the no-equality case: both solvers route the dual through the
        equality barrier's mini-Newton ``evaluate_dual``, which threads the
        ``A^T nu`` / ``-nu^T b`` equality terms and returns a guaranteed lower
        bound on `\\inf_x L(x, \\lambda, \\nu)`.
        """
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        A = np.ones((1, M))
        b = np.array([float(M)])  # x0 = ones is equality-feasible
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        x0 = np.ones(M)

        pd_result = PrimalDualSolver().solve(problem=problem, x0=x0)
        barrier_result = BarrierSolver().solve(problem=problem, x0=x0)

        # Sound lower bound: dual <= primal optimum (non-negative duality gap).
        assert pd_result.dual_value <= pd_result.objective_value + 1e-6
        # Same evaluator on (near-)identical optima -> same dual.
        self.assertAlmostEqual(
            pd_result.dual_value, barrier_result.dual_value, delta=1e-3
        )

    def test_infeasible_start_drives_equality_to_zero(self) -> None:
        """Caller may pass `x0` with `A x0 != b`; PD converges to `A x = b`."""
        rng = np.random.default_rng(1411)
        M = 5
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        A = np.ones((1, M))
        b = np.array([float(M)])
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        # x0 is inequality-feasible (positive) but equality-infeasible:
        # ones^T x0 = 1.5 * M != M.
        x0 = 1.5 * np.ones(M)
        assert not np.isclose((A @ x0).item(), b.item())

        result = PrimalDualSolver().solve(problem=problem, x0=x0)

        assert isinstance(result, SolveResult)
        assert isinstance(result.diagnostics, PrimalDualDiagnostics)
        np.testing.assert_allclose(A @ result.solution, b, atol=1e-6)
        diag = result.primal_dual_diagnostics
        assert diag.primal_residual_norms[-1] < 1e-6
        assert diag.dual_residual_norms[-1] < 1e-4

    def test_returns_pd_specific_diagnostics(self) -> None:
        """`PrimalDualResult` populated with primal/dual residuals and surrogate-gap trajectory."""
        rng = np.random.default_rng(1421)
        M = 5
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        A = np.ones((1, M))
        b = np.array([float(M)])
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = PrimalDualSolver().solve(problem=problem, x0=np.ones(M))
        diag = result.primal_dual_diagnostics
        assert diag.dual_residual_norms[-1] < 1e-4
        assert diag.primal_residual_norms[-1] < 1e-6
        assert len(diag.dual_residual_norms) == result.n_iterations
        assert len(diag.surrogate_gaps) == result.n_iterations
        assert diag.surrogate_gaps[-1] < diag.surrogate_gaps[0]

    def test_converges_on_near_singular_hessian_via_damping(self) -> None:
        """Equality analog of the no-equality near-singular test: adaptive
        Levenberg damping keeps the equality-PD Newton step usable when the
        objective Hessian is near-singular far from every inequality boundary.

        Without damping the raw step blows up and BTLS raises at iteration 0;
        with it the solve converges and satisfies ``A x = b``. The minimizer is
        not asserted (the 1e-10 eigenvalue leaves the objective nearly flat
        along one direction), only convergence + equality feasibility + residual
        decrease.
        """
        rng = np.random.default_rng(2027)
        M = 6
        basis, _ = np.linalg.qr(rng.standard_normal((M, M)))
        eigs = np.array([1.0, 1.0, 1.0, 1e-3, 1e-6, 1e-10])
        Q = (basis * eigs) @ basis.T
        Q = np.asarray(0.5 * (Q + Q.T))
        w_target = rng.uniform(2.0, 5.0, size=M)
        c = -Q @ w_target
        A = np.ones((1, M))
        b = A @ np.ones(M)  # x0 = ones is equality-feasible
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=-1e8, ub=1e8, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )

        result = PrimalDualSolver(
            settings=OptimizationSettings(outer=OuterSettings(tolerance=1e-4))
        ).solve(problem=problem, x0=np.ones(M))

        assert isinstance(result, SolveResult)
        assert result.status is SolveStatus.OPTIMAL
        assert np.all(np.isfinite(result.solution))
        assert np.all(np.abs(result.solution) < 1e8)
        np.testing.assert_allclose(A @ result.solution, b, atol=1e-5)
        diag = result.primal_dual_diagnostics
        assert diag.surrogate_gaps[-1] < diag.surrogate_gaps[0]


class PdResultDiagnosticsAndPlotsTest(unittest.TestCase):
    """Primal-dual residual histories carried on the result diagnostics."""

    @staticmethod
    def _solve_bounds_qp() -> SolveResult:
        rng = np.random.default_rng(1701)
        M = 5
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
        )
        return PrimalDualSolver().solve(
            problem=problem, x0=rng.uniform(0.5, 1.0, size=M)
        )

    def test_residual_norm_histories_populated_no_equality(self) -> None:
        result = PdResultDiagnosticsAndPlotsTest._solve_bounds_qp()
        diag = result.primal_dual_diagnostics
        assert len(diag.dual_residual_norms) == result.n_iterations
        assert len(diag.primal_residual_norms) == result.n_iterations
        assert all(d >= 0.0 for d in diag.dual_residual_norms)
        # No equality constraints -> primal residual identically zero.
        assert all(p == 0.0 for p in diag.primal_residual_norms)

    def test_equality_residual_histories_populated(self) -> None:
        rng = np.random.default_rng(1702)
        M = 4
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 5.0, size=M)
        c = -Q @ w_target
        A = np.ones((1, M))
        b = A @ np.ones(M)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        result = PrimalDualSolver().solve(problem=problem, x0=np.ones(M))
        diag = result.primal_dual_diagnostics
        assert len(diag.dual_residual_norms) == result.n_iterations
        assert len(diag.primal_residual_norms) == result.n_iterations
        assert all(d >= 0.0 for d in diag.dual_residual_norms)
        assert all(p >= 0.0 for p in diag.primal_residual_norms)


class PdInputGuardsTest(unittest.TestCase):
    """Hardening guards on PD inputs (CVX-STAB-4 / STAB-5)."""

    @staticmethod
    def _problem() -> Problem:
        """A minimal PD problem fixture: w >= 0, two layers, quadratic objective."""
        return Problem(
            obj=QuadraticObjective(
                c=np.zeros(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )

    def test_validate_pd_iterate_rejects_non_negative_f(self) -> None:
        """`f >= 0` (infeasible iterate) raises a diagnostic NewtonStepError.

        STAB-5 / H2: ``pd_weights = lmbda / (-f)`` produces non-finite or
        wrong-sign weights when ``f`` has a non-negative entry. The guard
        catches this at the cause site rather than letting NaN propagate
        through Hessian assembly.
        """
        f_bad = np.array([-1.0, 0.0, -1.0])
        lmbda = np.array([1.0, 1.0, 1.0])
        with self.assertRaisesRegex(NewtonStepError, "not strictly negative"):
            _PrimalDualKernel.validate_pd_iterate(lmbda, f_bad)

    def test_validate_pd_iterate_rejects_non_finite_f(self) -> None:
        """Non-finite ``f`` raises before silently corrupting pd_weights."""
        f_bad = np.array([-1.0, -np.inf, -1.0])
        lmbda = np.array([1.0, 1.0, 1.0])
        with self.assertRaisesRegex(
            NewtonStepError, "not strictly negative or not finite"
        ):
            _PrimalDualKernel.validate_pd_iterate(lmbda, f_bad)

    def test_validate_pd_iterate_rejects_negative_lmbda(self) -> None:
        """Negative ``lmbda`` (left the dual cone) raises a diagnostic NewtonStepError."""
        f = np.array([-1.0, -1.0, -1.0])
        lmbda_bad = np.array([1.0, -0.01, 1.0])
        with self.assertRaisesRegex(NewtonStepError, "negative or not finite"):
            _PrimalDualKernel.validate_pd_iterate(lmbda_bad, f)


class PdNewtonStepDenseOracleTest(unittest.TestCase):
    r"""Snapshot oracle for ``PrimalDualSolver._compute_pd_step``.

    Builds the modified KKT matrix
    `H_{pd} = \nabla^2 f_0(x) + G^T \mathrm{diag}(\lambda / -f) G` densely
    and compares the Newton step to the structured assembly that
    ``_compute_pd_step`` produces via the objective's and each layer's
    structured ``hessian_contribution``. Established here for the
    diagonal-layer case (Bounds) and the rank-update layers (Imbalance,
    SquaredNorm, Norm) through their structured Hessian contributions. The
    oracle is the
    contract: every refactor that touches PD assembly must keep these
    equalities passing at the stated tolerances.
    """

    @parameterized.expand(
        [
            (2001, 5, False),
            (2002, 10, False),
            (2003, 6, True),
            (2004, 12, True),
        ]
    )
    def test_pd_step_matches_dense_oracle_on_bounds_only(
        self, seed: int, M: int, with_upper_bound: bool
    ) -> None:
        """`_compute_pd_step` agrees with the dense oracle on a Bounds-only QP."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        if with_upper_bound:
            bounds_layer = BoundsConstraintLayer(lb=0.0, ub=10.0, dimension=M)
        else:
            bounds_layer = BoundsConstraintLayer(lb=0.0, dimension=M)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[bounds_layer],
        )

        # Strictly interior iterate; in (lb, ub) when both bounds present.
        x = rng.uniform(1.0, 5.0, size=M)
        f = bounds_layer.constraints(x)
        assert np.all(f < 0), "iterate must be strictly interior for the test"
        G = bounds_layer.grad_constraints(x)
        lmbda = rng.uniform(0.1, 1.0, size=f.size)

        # Plausible PD residuals at (x, lmbda).
        r_dual = Q @ x + c + G.T @ lmbda
        # Boyd & Vandenberghe eq 11.55: r_cent = -diag(lambda) f - (1/t) e.
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd

        delta_x_struct, delta_lmbda_struct = (
            # Test reaches into a private method on purpose.
            PrimalDualSolver()._compute_pd_step(
                problem=problem,
                x=x,
                lmbda=lmbda,
                f=f,
                r_dual=r_dual,
                r_cent=r_cent,
            )
        )

        # Dense reference: H_pd = Q + G^T diag(lambda/-f) G (Bounds is diagonal).
        H_pd = Q + G.T @ np.diag(lmbda / -f) @ G
        delta_x_dense, delta_lmbda_dense = _dense_pd_newton_step(
            H_pd=H_pd,
            G=G,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
        )

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )

    @parameterized.expand(
        [
            (2101, 5, 1),
            (2102, 8, 2),
            (2103, 10, 3),
        ]
    )
    def test_pd_step_folds_rank_p_contributions(
        self, seed: int, M: int, p: int
    ) -> None:
        """`_compute_pd_step` agrees with the dense oracle when a layer returns rank-p contribution."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        kappa = rng.standard_normal((M, p))
        d = rng.uniform(0.5, 2.0, size=p)  # Pure update; PD by construction.
        layer = _BoundsPlusRankPStub(lb=0.0, kappa=kappa, d=d)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[layer],
        )

        x = rng.uniform(1.0, 5.0, size=M)
        f = layer.constraints(x)
        assert np.all(f < 0)
        G = layer.grad_constraints(x)
        lmbda = rng.uniform(0.1, 1.0, size=f.size)

        r_dual = Q @ x + c + G.T @ lmbda
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd

        delta_x_struct, delta_lmbda_struct = (
            # Test reaches into a private method on purpose.
            PrimalDualSolver()._compute_pd_step(
                problem=problem,
                x=x,
                lmbda=lmbda,
                f=f,
                r_dual=r_dual,
                r_cent=r_cent,
            )
        )

        # Dense reference: H_pd = Q + G^T diag(lambda/-f) G + kappa diag(d) kappa^T.
        H_pd = Q + G.T @ np.diag(lmbda / -f) @ G + kappa @ np.diag(d) @ kappa.T
        delta_x_dense, delta_lmbda_dense = _dense_pd_newton_step(
            H_pd=H_pd,
            G=G,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
        )

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )

    @parameterized.expand(
        [
            (2201, 6, 1),
            (2202, 10, 2),
        ]
    )
    def test_equality_pd_step_folds_rank_p_contributions(
        self, seed: int, M: int, p: int
    ) -> None:
        """`PrimalDualSolver._compute_pd_step` folds layer rank-p contributions correctly."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        kappa = rng.standard_normal((M, p))
        d = rng.uniform(0.5, 2.0, size=p)
        layer = _BoundsPlusRankPStub(lb=0.0, kappa=kappa, d=d)
        # Single equality row 1^T x = M; satisfied by x = ones at start.
        A = np.ones((1, M))
        b = np.array([float(M)])
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[layer],
            eq=EqualityConstraintLayer(A=A, b=b),
        )

        x = rng.uniform(1.0, 5.0, size=M)
        f = layer.constraints(x)
        assert np.all(f < 0)
        G = layer.grad_constraints(x)
        lmbda = rng.uniform(0.1, 1.0, size=f.size)
        nu = rng.standard_normal(int(A.shape[0]))

        r_dual = Q @ x + c + G.T @ lmbda + A.T @ nu
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd
        r_pri = A @ x - b

        assert problem.eq is not None
        (
            delta_x_struct,
            delta_lmbda_struct,
            delta_nu_struct,
        ) = PrimalDualSolver()._compute_pd_step_equality(
            problem=problem,
            x=x,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
            r_pri=r_pri,
        )

        # Dense KKT reference for the equality path:
        #   [ H_pd   A^T ] [Δx ]   [ -r_dual - G^T (r_cent / f) ]
        #   [  A    0   ] [Δν ] = [ -r_pri                       ]
        # then Δλ = (r_cent - λ * (G Δx)) / f.
        H_pd = Q + G.T @ np.diag(lmbda / -f) @ G + kappa @ np.diag(d) @ kappa.T
        m_eq = A.shape[0]
        kkt = np.block([[H_pd, A.T], [A, np.zeros((m_eq, m_eq))]])
        rhs_top = -r_dual - G.T @ (r_cent / f)
        rhs_bot = -r_pri
        rhs = np.concatenate([rhs_top, rhs_bot])
        sol = np.linalg.solve(kkt, rhs)
        delta_x_dense = sol[:M]
        delta_nu_dense = sol[M:]
        delta_lmbda_dense = (r_cent - lmbda * (G @ delta_x_dense)) / f

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_nu_struct, delta_nu_dense, rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )

    @parameterized.expand(
        [
            (2501, 5, 1),
            (2502, 8, 2),
            (2503, 12, 3),
        ]
    )
    def test_pd_step_matches_dense_oracle_on_norm_plus_bounds(
        self, seed: int, M: int, q: int
    ) -> None:
        """`_compute_pd_step` agrees with the dense oracle on Norm + Bounds QP."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c_obj = rng.standard_normal(M)
        B = rng.standard_normal((q, M))
        c_norm = rng.standard_normal(q)

        # Strictly interior iterate (||Bw - c||^2 < phi^2); pick phi adaptively.
        x = rng.uniform(0.5, 2.0, size=M)
        residual = B @ x - c_norm
        phi = float(2.0 * np.linalg.norm(residual) + 1.0)
        bounds_layer = BoundsConstraintLayer(lb=0.0, dimension=M)
        norm_layer = NormConstraintLayer(B=B, c=c_norm, phi=phi)
        problem = Problem(
            obj=QuadraticObjective(c=c_obj, hessian_contribution=DenseSystem(Q)),
            ineq=[bounds_layer, norm_layer],
        )

        f = np.concatenate([bounds_layer.constraints(x), norm_layer.constraints(x)])
        assert np.all(f < 0), "iterate must be strictly interior"
        G = np.vstack(
            [bounds_layer.grad_constraints(x), norm_layer.grad_constraints(x)]
        )

        lmbda = rng.uniform(0.1, 1.0, size=f.size)
        # Norm has its single multiplier last (after the M bounds multipliers).
        lmbda_norm = float(lmbda[bounds_layer.num_constraints(M)])
        r_dual = Q @ x + c_obj + G.T @ lmbda
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd

        delta_x_struct, delta_lmbda_struct = (
            # Test reaches into a private method on purpose.
            PrimalDualSolver()._compute_pd_step(
                problem=problem,
                x=x,
                lmbda=lmbda,
                f=f,
                r_dual=r_dual,
                r_cent=r_cent,
            )
        )

        # Dense reference: H_pd = Q + Σ λ_i ∇²f_i + Gᵀ diag(λ/-f) G
        # = Q + 2 λ_norm BᵀB + Gᵀ diag(λ/-f) G  (Bounds linear → no diag term).
        H_pd = Q + 2.0 * lmbda_norm * (B.T @ B) + G.T @ np.diag(lmbda / -f) @ G
        delta_x_dense, delta_lmbda_dense = _dense_pd_newton_step(
            H_pd=H_pd,
            G=G,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
        )

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )

    @parameterized.expand(
        [
            (2401, 5),
            (2402, 10),
            (2403, 20),
        ]
    )
    def test_pd_step_matches_dense_oracle_on_squared_norm_plus_bounds(
        self, seed: int, M: int
    ) -> None:
        """`_compute_pd_step` agrees with the dense oracle on SquaredNorm + Bounds QP."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)

        # Strictly interior iterate (||x||^2 < phi); pick phi adaptively.
        x = rng.uniform(0.5, 2.0, size=M)
        phi = float(2.0 * np.dot(x, x))  # generous; ||x||^2 < phi
        bounds_layer = BoundsConstraintLayer(lb=0.0, dimension=M)
        sqnorm_layer = SquaredNormConstraintLayer(phi=phi, dimension=M)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[bounds_layer, sqnorm_layer],
        )

        f = np.concatenate([bounds_layer.constraints(x), sqnorm_layer.constraints(x)])
        assert np.all(f < 0), "iterate must be strictly interior"
        G = np.vstack(
            [bounds_layer.grad_constraints(x), sqnorm_layer.grad_constraints(x)]
        )

        lmbda = rng.uniform(0.1, 1.0, size=f.size)
        # Layer constraint Hessians: bounds linear (zero), sqnorm 2I per
        # constraint times its multiplier.
        lmbda_sqnorm = float(lmbda[bounds_layer.num_constraints(M)])
        # f0 part of r_dual is Q @ x + c; full r_dual adds Σ λ_i ∇f_i = G^T λ.
        r_dual = Q @ x + c + G.T @ lmbda
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd

        delta_x_struct, delta_lmbda_struct = (
            # Test reaches into a private method on purpose.
            PrimalDualSolver()._compute_pd_step(
                problem=problem,
                x=x,
                lmbda=lmbda,
                f=f,
                r_dual=r_dual,
                r_cent=r_cent,
            )
        )

        # Dense reference: H_pd = Q + Σ λ_i ∇²f_i + G^T diag(λ/-f) G
        #                       = Q + 2 λ_sqnorm I + G^T diag(λ/-f) G.
        H_pd = Q + 2.0 * lmbda_sqnorm * np.eye(M) + G.T @ np.diag(lmbda / -f) @ G
        delta_x_dense, delta_lmbda_dense = _dense_pd_newton_step(
            H_pd=H_pd,
            G=G,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
        )

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )

    @parameterized.expand(
        [
            (2301, 5, 1),
            (2302, 8, 2),
            (2303, 12, 3),
        ]
    )
    def test_pd_step_matches_dense_oracle_on_imbalance_plus_bounds(
        self, seed: int, M: int, q: int
    ) -> None:
        """`_compute_pd_step` agrees with the dense oracle on Imbalance + Bounds QP."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        B = rng.standard_normal((q, M))
        target = np.zeros(q)

        # Strictly interior iterate; size psi from x so the imbalance bound is loose.
        x = rng.uniform(1.0, 5.0, size=M)
        residual = B @ x - target
        psi = float(2.0 * np.max(np.abs(residual)) + 1.0)
        bounds_layer = BoundsConstraintLayer(lb=0.0, dimension=M)
        imbalance_layer = ImbalanceConstraintLayer(B=B, c=target, psi=psi)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[bounds_layer, imbalance_layer],
        )

        f = np.concatenate(
            [bounds_layer.constraints(x), imbalance_layer.constraints(x)]
        )
        assert np.all(f < 0), "iterate must be strictly interior"
        G = np.vstack(
            [bounds_layer.grad_constraints(x), imbalance_layer.grad_constraints(x)]
        )

        lmbda = rng.uniform(0.1, 1.0, size=f.size)
        r_dual = Q @ x + c + G.T @ lmbda
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd

        delta_x_struct, delta_lmbda_struct = (
            # Test reaches into a private method on purpose.
            PrimalDualSolver()._compute_pd_step(
                problem=problem,
                x=x,
                lmbda=lmbda,
                f=f,
                r_dual=r_dual,
                r_cent=r_cent,
            )
        )

        # Dense reference: H_pd = Q + Gᵀ diag(lambda/-f) G. For Imbalance,
        # this contributes both diagonal and off-diagonal entries via Bᵀ diag(d) B.
        H_pd = Q + G.T @ np.diag(lmbda / -f) @ G
        delta_x_dense, delta_lmbda_dense = _dense_pd_newton_step(
            H_pd=H_pd,
            G=G,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
        )

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )


class ClampDualAfterUpdateTest(unittest.TestCase):
    """Post-update clamp on inequality multipliers (CVX-STAB-8)."""

    def test_strictly_positive_passthrough(self) -> None:
        """Healthy ``lmbda > 0`` is returned unchanged."""
        lmbda = np.array([0.5, 1.0, 2.5])
        out = _clamp_dual_after_update(lmbda)
        np.testing.assert_array_equal(out, lmbda)

    def test_zero_entries_passthrough(self) -> None:
        """Exact zeros are not negative; pass through unchanged."""
        lmbda = np.array([0.0, 1.0])
        out = _clamp_dual_after_update(lmbda)
        np.testing.assert_array_equal(out, lmbda)

    def test_tiny_negative_drift_is_clamped(self) -> None:
        """Sub-rounding negative entries clamp to zero rather than raise."""
        # Drift of order 1e-16 relative to ||lmbda||_inf = 1.0 is well within
        # the 1e-12 * scale tolerance.
        lmbda = np.array([1.0, -1e-16, 2.0])
        out = _clamp_dual_after_update(lmbda)
        assert out[1] == 0.0
        # Other entries untouched.
        assert out[0] == 1.0
        assert out[2] == 2.0

    def test_significant_negative_raises(self) -> None:
        """Negative beyond 1e-12 * scale indicates a real bug; raise diagnostic."""
        # Drift of 1e-3 relative to ||lmbda||_inf = 1.0 is 9 orders of
        # magnitude past the tolerance.
        lmbda = np.array([1.0, -1e-3, 0.5])
        with self.assertRaisesRegex(NewtonStepError, "leaked negative"):
            _clamp_dual_after_update(lmbda)

    def test_tolerance_scales_with_lmbda_magnitude(self) -> None:
        """The negativity tolerance is relative to ``||lmbda||_inf``.

        Same absolute drift (1e-9) is acceptable for ``||lmbda||_inf = 1e6``
        (relative drift 1e-15) but unacceptable for ``||lmbda||_inf = 1e-3``
        (relative drift 1e-6, well past the 1e-12 tolerance).
        """
        # Large-scale lmbda: 1e-9 drift is fine.
        lmbda_large = np.array([1e6, -1e-9])
        out = _clamp_dual_after_update(lmbda_large)
        assert out[1] == 0.0
        # Small-scale lmbda: 1e-9 drift is real.
        lmbda_small = np.array([1e-3, -1e-9])
        with self.assertRaisesRegex(NewtonStepError, "leaked negative"):
            _clamp_dual_after_update(lmbda_small)

    def test_empty_array(self) -> None:
        """Empty multipliers (degenerate problems) pass through without raising."""
        lmbda = np.array([], dtype=np.float64)
        out = _clamp_dual_after_update(lmbda)
        assert out.size == 0

    def test_nan_entry_raises(self) -> None:
        """NaN entries raise rather than silently passing through.

        Without an explicit ``np.isfinite`` check, ``np.min(lmbda)`` returns
        ``NaN`` and ``NaN < 0.0`` evaluates to ``False`` -- so a NaN
        ``delta_lmbda`` entry would propagate undetected. The guard catches
        this at the cause site.
        """
        lmbda = np.array([1.0, np.nan, 0.5])
        with self.assertRaisesRegex(NewtonStepError, "non-finite"):
            _clamp_dual_after_update(lmbda)

    def test_positive_inf_entry_raises(self) -> None:
        """``+inf`` entries raise too (same code path as NaN)."""
        lmbda = np.array([1.0, np.inf, 0.5])
        with self.assertRaisesRegex(NewtonStepError, "non-finite"):
            _clamp_dual_after_update(lmbda)

    def test_negative_inf_entry_raises(self) -> None:
        """``-inf`` entries raise too (same code path as NaN)."""
        lmbda = np.array([1.0, -np.inf, 0.5])
        with self.assertRaisesRegex(NewtonStepError, "non-finite"):
            _clamp_dual_after_update(lmbda)

    def test_diagnostic_reports_actual_norm_not_floored_scale(self) -> None:
        """Error message reports the *actual* ``||lmbda||_inf``, not ``max(1, ...)``.

        The drift tolerance is computed against ``max(1, ||lmbda||_inf)`` so the
        floor doesn't degenerate near zero, but the diagnostic must report the
        real norm so debugging isn't misled when ``||lmbda||_inf < 1``.
        """
        # ||lmbda||_inf = 1e-3, well below the floor; drift = -1e-9
        # is well past the relative tolerance (1e-12 * 1.0 = 1e-12).
        lmbda = np.array([1e-3, -1e-9])
        with self.assertRaises(NewtonStepError) as ctx:
            _clamp_dual_after_update(lmbda)
        # Error message must include the *actual* norm 1e-3, not the floored 1.0.
        msg = str(ctx.exception)
        assert "||lmbda||_inf = 1.000e-03" in msg, msg

    @parameterized.expand(
        [
            (2401, 5, 2),
            (2402, 8, 3),
            (2403, 12, 5),
        ]
    )
    def test_pd_step_matches_dense_oracle_on_linear_inequality_plus_bounds(
        self, seed: int, M: int, m: int
    ) -> None:
        """``_compute_pd_step`` agrees with the dense oracle on LinearInequality + Bounds QP."""
        rng = np.random.default_rng(seed)
        Q = _make_psd_matrix(rng, M)
        c = rng.standard_normal(M)
        A = rng.standard_normal((m, M))

        # Strictly interior iterate; size cl/cu so every row is two-sided
        # and the interior is non-empty.
        x = rng.uniform(1.0, 5.0, size=M)
        Ax = A @ x
        slack = float(np.max(np.abs(Ax)) + 1.0)
        cl = Ax - slack
        cu = Ax + slack

        bounds_layer = BoundsConstraintLayer(lb=0.0, dimension=M)
        linineq_layer = LinearInequalityConstraintLayer(A=A, cl=cl, cu=cu)
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[bounds_layer, linineq_layer],
        )

        f = np.concatenate([bounds_layer.constraints(x), linineq_layer.constraints(x)])
        assert np.all(f < 0), "iterate must be strictly interior"
        G = np.vstack(
            [bounds_layer.grad_constraints(x), linineq_layer.grad_constraints(x)]
        )

        lmbda = rng.uniform(0.1, 1.0, size=f.size)
        r_dual = Q @ x + c + G.T @ lmbda
        t_pd = 10.0
        r_cent = -lmbda * f - 1.0 / t_pd

        delta_x_struct, delta_lmbda_struct = (
            # Test reaches into a private method on purpose.
            PrimalDualSolver()._compute_pd_step(
                problem=problem,
                x=x,
                lmbda=lmbda,
                f=f,
                r_dual=r_dual,
                r_cent=r_cent,
            )
        )

        # Dense reference: H_pd = Q + Gᵀ diag(lambda/-f) G. For
        # LinearInequalityConstraintLayer, this contributes both diagonal and
        # off-diagonal entries via Aᵀ diag(d) A with row-scatter aggregation.
        H_pd = Q + G.T @ np.diag(lmbda / -f) @ G
        delta_x_dense, delta_lmbda_dense = _dense_pd_newton_step(
            H_pd=H_pd,
            G=G,
            lmbda=lmbda,
            f=f,
            r_dual=r_dual,
            r_cent=r_cent,
        )

        np.testing.assert_allclose(delta_x_struct, delta_x_dense, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            delta_lmbda_struct, delta_lmbda_dense, rtol=1e-9, atol=1e-10
        )

    def test_solve_accepts_linear_inequality_and_matches_barrier(self) -> None:
        """Full ``PrimalDualSolver.solve`` accepts a LinearInequality problem.

        ``LinearInequalityConstraintLayer`` supplies a structured Hessian
        contribution, just like its sibling ``ImbalanceConstraintLayer``, and the
        per-step math (see the dense-oracle test above) already matched. This pins
        that the full solve is accepted and agrees with the barrier solver on the
        optimum.
        """
        rng = np.random.default_rng(7777)
        M = 5
        Q = _make_psd_matrix(rng, M)
        w_target = rng.uniform(1.0, 3.0, size=M)  # interior unconstrained optimum
        c = -Q @ w_target
        A = rng.standard_normal((2, M))
        Aw = A @ w_target
        cl = Aw - 10.0
        cu = Aw + 10.0
        problem = Problem(
            obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
            ineq=[
                BoundsConstraintLayer(lb=0.0, dimension=M),
                LinearInequalityConstraintLayer(A=A, cl=cl, cu=cu),
            ],
        )
        x0 = w_target + 0.5  # strictly feasible: > 0, and A x0 within (cl, cu)

        pd_result = PrimalDualSolver().solve(problem=problem, x0=x0)
        barrier_result = BarrierSolver().solve(problem=problem, x0=x0)

        # Guard the comparison: a non-converged solve would otherwise be
        # silently compared, masking a real regression.
        assert pd_result.status is SolveStatus.OPTIMAL, pd_result.status
        assert barrier_result.status is SolveStatus.OPTIMAL, barrier_result.status

        np.testing.assert_allclose(
            pd_result.solution, barrier_result.solution, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(pd_result.solution, w_target, rtol=1e-3, atol=1e-3)


class PdFractionToBoundarySettingTest(unittest.TestCase):
    """`OptimizationSettings` exposes the PD fraction-to-boundary knob."""

    def test_default_settings_value(self) -> None:
        """OptimizationSettings exposes pd_fraction_to_boundary_min defaulting to 0.99."""
        assert OptimizationSettings().pd_fraction_to_boundary_min == 0.99


class PrimalDualFeasibilityThresholdEarlyStopTest(unittest.TestCase):
    """The primal-dual loop also halts once the objective crosses its threshold."""

    def test_stops_at_threshold_above_the_optimum(self) -> None:
        """A threshold-carrying objective stops the PD solve short of the optimum."""
        bounds = BoundsConstraintLayer(lb=-10.0, dimension=1)  # w >= -10
        x0 = np.array([5.0])  # strictly feasible start
        plain = PrimalDualSolver().solve(
            problem=Problem(obj=LinearObjective(c=np.array([1.0])), ineq=[bounds]),
            x0=x0,
        )
        feasible = PrimalDualSolver().solve(
            problem=Problem(
                obj=LinearObjective(c=np.array([1.0])).with_feasibility_threshold(0.0),
                ineq=[bounds],
            ),
            x0=x0,
        )
        assert feasible.status is SolveStatus.MAX_ITERATIONS
        assert feasible.objective_value < 0.0  # crossed the threshold
        assert feasible.objective_value > plain.objective_value  # short of the optimum
        # The loop broke the moment the objective crossed its threshold, before the
        # multipliers were refined past their seed, so the dual-side fields would
        # otherwise echo an arbitrary initial multiplier; the solver zeroes them
        # rather than surface a misleading primal-dual state.
        assert feasible.dual_value == 0.0
        assert feasible.suboptimality == 0.0
        assert isinstance(feasible.diagnostics, PrimalDualDiagnostics)
        diag = feasible.diagnostics
        # The convergence trajectories are recorded, and the strictly-feasible,
        # equality-free start keeps the primal residual at exactly zero throughout
        # -- the substantive feasibility fact the early-threshold break preserves.
        assert len(diag.primal_residual_norms) > 0
        assert all(p == 0.0 for p in diag.primal_residual_norms)


class StallWatchTest(unittest.TestCase):
    r"""The primal-dual stall guard: bail on a creep, never on real progress.

    The failure it exists for: the Armijo search reaches a regime where no step
    at or above ``min_step`` satisfies sufficient decrease, so it returns the
    floor, and the iterate creeps -- gap inching down while dual feasibility
    drifts back up -- so ``max(eta, ||r_dual||)`` never clears tolerance and the
    loop burns its whole budget. Observed on a PyRake frontier corner at 1000
    iterations; 21 with this guard.
    """

    @staticmethod
    def _watch(patience: int = 5, rel: float = 0.01, soft: float = 1e-3) -> _StallWatch:
        return _StallWatch(patience=patience, rel_improvement=rel, soft_tolerance=soft)

    @staticmethod
    def _iterate(value: float) -> tuple[npt.NDArray[np.float64], ...]:
        return np.array([value]), np.array([value])

    def test_a_sub_threshold_creep_does_not_count_as_progress(self) -> None:
        """The whole point: 0.1%-per-iteration improvement must still stall.

        A bare "did it improve at all" test resets every iteration on exactly
        this trajectory and never fires -- which is how the real solve reached
        its iteration cap.
        """
        watch = self._watch(patience=5, rel=0.01)
        metric = 1e-5
        stalled = False
        for nit in range(20):
            metric *= 0.999  # 0.1% per iteration, well under the 1% floor
            x, lmbda = self._iterate(metric)
            stalled = watch.observe(metric, nit, x, lmbda)
            if stalled:
                break

        assert stalled, "a sub-threshold creep must trip the guard"
        assert nit <= 6, f"should trip promptly after patience, tripped at {nit}"

    def test_real_progress_never_stalls(self) -> None:
        """An order-of-magnitude per step must never be cut off."""
        watch = self._watch(patience=5, rel=0.01)
        metric = 1.0
        for nit in range(50):
            metric *= 0.1
            x, lmbda = self._iterate(metric)

            assert not watch.observe(metric, nit, x, lmbda)

    def test_progress_just_above_the_floor_never_stalls(self) -> None:
        """The boundary case: improvement fractionally better than the floor."""
        watch = self._watch(patience=5, rel=0.01)
        metric = 1e-4
        for nit in range(50):
            metric *= 0.98  # 2% per iteration, comfortably over the 1% floor
            x, lmbda = self._iterate(metric)

            assert not watch.observe(metric, nit, x, lmbda)

    def test_it_holds_on_while_the_best_is_outside_soft_tolerance(self) -> None:
        """Short of a usable answer there is nothing to return, so keep trying.

        Mirrors the barrier detector, which only bails once
        ``best_suboptimality < tolerance_soft``.
        """
        watch = self._watch(patience=3, rel=0.01, soft=1e-6)
        for nit in range(30):
            x, lmbda = self._iterate(1e-3)  # flat, but far worse than soft tol

            assert not watch.observe(1e-3, nit, x, lmbda)

    def test_it_hands_back_the_best_iterate_not_the_last(self) -> None:
        """The returned answer is the reason for the guard, not just the saving.

        On the real stall the last iterate's dual residual was ~46,000x worse
        than one the solver had already passed through.
        """
        watch = self._watch(patience=3, rel=0.01)
        good = np.array([42.0])
        watch.observe(1e-5, 7, good, good)
        for nit in range(8, 20):
            worse = np.array([float(nit)])
            if watch.observe(1e-4, nit, worse, worse):
                break

        assert watch.best_nit == 7
        assert watch.best_metric == 1e-5
        np.testing.assert_array_equal(none_throws(watch.best_x), good)
        np.testing.assert_array_equal(none_throws(watch.best_lmbda), good)

    def test_equality_multipliers_ride_along_when_present(self) -> None:
        """``_run_equality`` restores ``nu`` too; the inequality path has none."""
        watch = self._watch(patience=3, rel=0.01)
        x = np.array([1.0])
        nu = np.array([3.0, 4.0])
        watch.observe(1e-5, 0, x, x, nu)

        np.testing.assert_array_equal(none_throws(watch.best_nu), nu)

        bare = self._watch(patience=3, rel=0.01)
        bare.observe(1e-5, 0, x, x)

        assert bare.best_nu is None
