"""Test Trace abstraction (Trace, LoggingTrace, ListTrace, DataFrameTrace)."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from cvxium.diagnostics.trace import (
    DataFrameTrace,
    ListTrace,
    LoggingTrace,
    Trace,
)
from cvxium.linalg.systems import DenseSystem
from cvxium.problems.cone import SecondOrderCone
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
    EqualityConstraintLayer,
)
from cvxium.problems.objective import (
    LinearObjective,
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.saddle.primal_dual_stationarity_solver import (
    PrimalDualStationaritySolver,
)
from cvxium.saddle.stationarity import (
    QuadraticSaddleObjective,
    StationarityProblem,
)
from cvxium.saddle.stationarity_solver import (
    StationaritySolver,
)
from cvxium.solvers.barrier_solver import (
    BarrierSolver,
)
from cvxium.solvers.hsd_solver import HsdSolver
from cvxium.solvers.primal_dual_solver import (
    PrimalDualSolver,
)
from cvxium.solvers.settings import (
    ObservabilitySettings,
    OptimizationSettings,
    OuterSettings,
)
from cvxium.solvers.solver import EqualitySolver


def _make_psd_matrix(rng: np.random.Generator, M: int) -> npt.NDArray[np.float64]:
    A = rng.standard_normal((M, M))
    return np.asarray(A @ A.T + M * np.eye(M))


def _make_bounded_quadratic_problem(
    seed: int = 1234, M: int = 8
) -> tuple[Problem, npt.NDArray[np.float64]]:
    """Return a small bound-constrained PD-quadratic Problem + feasible x0."""
    rng = np.random.default_rng(seed)
    Q = _make_psd_matrix(rng, M)
    w_target = rng.uniform(1.0, 5.0, size=M)
    c = -Q @ w_target
    problem = Problem(
        obj=QuadraticObjective(c=c, hessian_contribution=DenseSystem(Q)),
        ineq=[BoundsConstraintLayer(lb=0.0, dimension=M)],
    )
    x0 = rng.uniform(0.5, 1.0, size=M)
    return problem, x0


def _solve_with_trace(trace: Trace) -> None:
    problem, x0 = _make_bounded_quadratic_problem()
    settings = OptimizationSettings(trace=trace)
    BarrierSolver(settings=settings).solve(problem=problem, x0=x0)


def _solve_hsd_with_trace(trace: Trace) -> None:
    """Drive ``HsdSolver`` on the apex-pin SOCP fixture used in test_hsd_solver."""
    problem = Problem(
        obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
        eq=EqualityConstraintLayer(A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])),
    )
    settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-6), trace=trace)
    HsdSolver(cones=[SecondOrderCone(dim=3)], settings=settings).solve(
        problem=problem, x0=np.zeros(3)
    )


def _solve_pd_with_trace(trace: Trace) -> None:
    """Drive ``PrimalDualSolver`` on the same bounded-quadratic fixture used elsewhere."""
    problem, x0 = _make_bounded_quadratic_problem()
    settings = OptimizationSettings(trace=trace)
    PrimalDualSolver(settings=settings).solve(problem=problem, x0=x0)


def _solve_equality_pd_with_trace(trace: Trace) -> None:
    """Drive ``PrimalDualSolver`` on a tiny bound-constrained QP with one equality."""
    problem, x0 = _make_bounded_quadratic_problem()
    # Add a single equality A x = b through x0 so the strictly feasible iterate is preserved.
    A = np.ones((1, problem.dimension))
    b = np.array([float(np.sum(x0))])
    problem = Problem(
        obj=problem.obj,
        ineq=problem.ineq,
        eq=EqualityConstraintLayer(A=A, b=b),
    )
    settings = OptimizationSettings(trace=trace)
    PrimalDualSolver(settings=settings).solve(problem=problem, x0=x0)


def _make_bounded_saddle_problem(
    seed: int = 7, n: int = 3, m: int = 3
) -> StationarityProblem:
    """Return a small bound-constrained ``StationarityProblem`` (bounds non-binding).

    The bounds bracket the unconstrained saddle widely so both stationarity engines
    iterate a few times and converge cleanly (OPTIMAL), exercising the full
    ``start -> iterate(s) -> done`` event sequence.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(m)
    A = rng.standard_normal((n, n))
    P = A.T @ A + n * np.eye(n)
    B = rng.standard_normal((m, m))
    R = B.T @ B + m * np.eye(m)
    C = rng.standard_normal((n, m))
    obj = QuadraticSaddleObjective(a, b, P, C, R)
    return StationarityProblem(
        objective=obj,
        x_domain=[BoundsConstraintLayer(lb=-10.0 * np.ones(n), ub=10.0 * np.ones(n))],
        y_domain=[BoundsConstraintLayer(lb=-10.0 * np.ones(m), ub=10.0 * np.ones(m))],
    )


# Interior warm starts (0 is strictly inside the +/-10 bounds and satisfies the
# ``sum(x) = 0`` equality), so ``solve`` uses them verbatim and no Phase I
# feasibilization runs -- keeping the trace free of nested start-resolution events
# so the bracket-first/-last assertions are exact.
_SADDLE_X0: npt.NDArray[np.float64] = np.zeros(3)
_SADDLE_Y0: npt.NDArray[np.float64] = np.zeros(3)


def _solve_barrier_stationarity_with_trace(trace: Trace) -> None:
    """Drive the barrier ``StationaritySolver`` on the bounded-saddle fixture."""
    settings = OptimizationSettings(trace=trace)
    StationaritySolver(settings=settings).solve(
        _make_bounded_saddle_problem(), x0=_SADDLE_X0, y0=_SADDLE_Y0
    )


def _solve_pd_stationarity_with_trace(trace: Trace) -> None:
    """Drive ``PrimalDualStationaritySolver`` on the bounded-saddle fixture."""
    settings = OptimizationSettings(trace=trace)
    PrimalDualStationaritySolver(settings=settings).solve(
        _make_bounded_saddle_problem(), x0=_SADDLE_X0, y0=_SADDLE_Y0
    )


def _solve_pd_stationarity_equality_with_trace(trace: Trace) -> None:
    """Drive ``PrimalDualStationaritySolver`` on a bounded saddle with an x-equality."""
    base = _make_bounded_saddle_problem()
    problem = StationarityProblem(
        objective=base.objective,
        x_domain=base.x_domain,
        y_domain=base.y_domain,
        x_equality=EqualityConstraintLayer(A=np.ones((1, 3)), b=np.array([0.0])),
    )
    settings = OptimizationSettings(trace=trace)
    PrimalDualStationaritySolver(settings=settings).solve(
        problem, x0=_SADDLE_X0, y0=_SADDLE_Y0
    )


class DefaultsTest(unittest.TestCase):
    """Verify the default trace is LoggingTrace and behavior is preserved."""

    def test_default_trace_is_logging_trace(self) -> None:
        settings = OptimizationSettings()
        assert isinstance(settings.trace, LoggingTrace)

    def test_default_trace_each_instance_distinct(self) -> None:
        """default_factory gives each settings its own trace instance."""
        s1 = OptimizationSettings()
        s2 = OptimizationSettings()
        assert s1.trace is not s2.trace


class TraceBaseTest(unittest.TestCase):
    """The Trace base class is all no-ops; collecting zero events should solve."""

    def test_base_trace_collects_nothing_but_solves(self) -> None:
        trace = Trace()
        _solve_with_trace(trace)
        # Base trace has no records attribute; the only check is that the solve
        # ran without raising. ``want_btls_details()`` defaults False so BTLS
        # extras are skipped.
        assert trace.want_btls_details() is False


class ListTraceTest(unittest.TestCase):
    """ListTrace collects all 10 event kinds when applicable."""

    def test_collects_expected_events(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        # ipm_start and ipm_done bracket the run exactly once.
        assert events.count("ipm_start") == 1
        assert events.count("ipm_done") == 1
        assert events[0] == "ipm_start"
        assert events[-1] == "ipm_done"
        # At least one centering pair occurred.
        assert events.count("centering_start") >= 1
        assert events.count("centering_start") == events.count("centering_done")
        # At least one Newton step + BTLS pair (ListTrace requests details).
        assert events.count("newton_step") >= 1
        assert events.count("btls") >= 1

    def test_records_have_time_and_event_keys(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        for row in trace.records():
            assert "time" in row
            assert "event" in row

    def test_newton_step_schema(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        newton_rows = [r for r in trace.records() if r["event"] == "newton_step"]
        assert newton_rows
        for row in newton_rows:
            assert set(row.keys()) >= {
                "time",
                "event",
                "outer_nit",
                "inner_nit",
                "elapsed_ms",
                "suboptimality",
                "lambda_norm",
                "quadratic_convergence_threshold",
            }

    def test_btls_schema(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        btls_rows = [r for r in trace.records() if r["event"] == "btls"]
        assert btls_rows
        for row in btls_rows:
            assert set(row.keys()) >= {
                "time",
                "event",
                "outer_nit",
                "inner_nit",
                "btls_s",
                "improvement",
                "expected_improvement",
            }

    def test_centering_done_carries_duality_gap(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        rows = [r for r in trace.records() if r["event"] == "centering_done"]
        assert rows
        for row in rows:
            assert set(row.keys()) >= {
                "time",
                "event",
                "outer_nit",
                "elapsed_ms",
                "duality_gap",
            }

    def test_ipm_start_and_done_carry_solver_class(self) -> None:
        """ipm_start/ipm_done carry the IPM solver class as ``solver``.

        BarrierSolver delegates the actual IPM loop to an internal
        ``_BarrierAdapter`` (it wraps the Problem into an equivalent IPM-shaped
        solver), so the reported solver class name reflects the adapter, not
        the public entry point. Either name is acceptable.

        """
        trace = ListTrace()
        _solve_with_trace(trace)
        starts = [r for r in trace.records() if r["event"] == "ipm_start"]
        dones = [r for r in trace.records() if r["event"] == "ipm_done"]
        assert starts[0]["solver"] in {"BarrierSolver", "_BarrierAdapter"}
        assert dones[0]["solver"] == starts[0]["solver"]
        assert dones[0]["elapsed_ms"] > 0.0

    def test_clear_resets_records(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        assert trace.records()
        trace.clear()
        assert trace.records() == []

    def test_records_returns_defensive_copy(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        first = trace.records()
        first.clear()
        assert trace.records()  # internal state untouched

    def test_want_btls_details_true(self) -> None:
        assert ListTrace().want_btls_details() is True

    def test_equality_solver_feasible_initial_guess(self) -> None:
        """EqualitySolver with x0 satisfying A x = b emits feasible_initial_guess."""
        trace = ListTrace()
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        settings = OptimizationSettings(trace=trace)
        solver = EqualitySolver(A=A, b=b, settings=settings)
        solver.solve(x0=np.array([0.6, 0.4]))
        events = [r["event"] for r in trace.records()]
        assert "feasible_initial_guess" in events

    def test_equality_solver_svd_done(self) -> None:
        """EqualitySolver without feasible x0 emits svd_done with elapsed_ms."""
        trace = ListTrace()
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        settings = OptimizationSettings(trace=trace)
        solver = EqualitySolver(A=A, b=b, settings=settings)
        solver.solve()
        rows = [r for r in trace.records() if r["event"] == "svd_done"]
        assert len(rows) == 1
        assert rows[0]["elapsed_ms"] >= 0.0


class DualEvalEventsTest(unittest.TestCase):
    """Each centering step emits a timed dual_eval_start / dual_eval_done pair."""

    def test_dual_eval_brackets_each_centering_step(self) -> None:
        """One timed dual evaluation per centering step that returns a result."""
        trace = ListTrace()
        _solve_with_trace(trace)
        starts = [r for r in trace.records() if r["event"] == "dual_eval_start"]
        dones = [r for r in trace.records() if r["event"] == "dual_eval_done"]
        centering_dones = [r for r in trace.records() if r["event"] == "centering_done"]
        assert len(starts) >= 1
        # Each centering step that returns evaluates the dual exactly once.
        assert len(starts) == len(dones) == len(centering_dones)

    def test_dual_eval_schema_and_timing(self) -> None:
        trace = ListTrace()
        _solve_with_trace(trace)
        starts = [r for r in trace.records() if r["event"] == "dual_eval_start"]
        dones = [r for r in trace.records() if r["event"] == "dual_eval_done"]
        for row in starts:
            assert set(row.keys()) >= {"time", "event", "outer_nit"}
        for row in dones:
            assert set(row.keys()) >= {"time", "event", "outer_nit", "elapsed_ms"}
            assert row["elapsed_ms"] >= 0.0

    def test_dual_eval_done_precedes_its_centering_done(self) -> None:
        """The dual eval is logged before the centering step it closes."""
        trace = ListTrace()
        _solve_with_trace(trace)
        seen_dual_done: set[int] = set()
        for r in trace.records():
            if r["event"] == "dual_eval_done":
                seen_dual_done.add(r["outer_nit"])
            elif r["event"] == "centering_done":
                # The dual eval for this step ran inside centering_step, so its
                # done event must already have been recorded.
                assert r["outer_nit"] in seen_dual_done

    def test_dual_newton_steps_logged(self) -> None:
        """The mini-Newton inside the dual eval logs its inner steps."""
        trace = ListTrace()
        _solve_with_trace(trace)
        dual_steps = [r for r in trace.records() if r["event"] == "dual_newton_step"]
        assert dual_steps
        for row in dual_steps:
            assert set(row.keys()) >= {
                "time",
                "event",
                "inner_nit",
                "elapsed_ms",
                "gradient_norm",
            }
            assert row["elapsed_ms"] >= 0.0
            assert row["gradient_norm"] >= 0.0

    def test_dual_newton_steps_nested_in_dual_eval(self) -> None:
        """Every dual_newton_step is emitted between a start and its done."""
        trace = ListTrace()
        _solve_with_trace(trace)
        in_dual_eval = False
        saw_step_inside = False
        for r in trace.records():
            event = r["event"]
            if event == "dual_eval_start":
                in_dual_eval = True
            elif event == "dual_eval_done":
                in_dual_eval = False
            elif event == "dual_newton_step":
                assert in_dual_eval
                saw_step_inside = True
        assert saw_step_inside


class HsdEventsTest(unittest.TestCase):
    """HSD long-step driver emits hsd_start, hsd_iteration, hsd_done."""

    def test_brackets_run_exactly_once(self) -> None:
        trace = ListTrace()
        _solve_hsd_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("hsd_start") == 1
        assert events.count("hsd_done") == 1
        assert events[0] == "hsd_start"
        assert events[-1] == "hsd_done"

    def test_emits_at_least_one_iteration(self) -> None:
        trace = ListTrace()
        _solve_hsd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "hsd_iteration"]
        assert iters

    def test_iteration_schema(self) -> None:
        trace = ListTrace()
        _solve_hsd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "hsd_iteration"]
        for row in iters:
            assert set(row.keys()) >= {
                "time",
                "event",
                "nit",
                "mu",
                "rel_pres",
                "rel_dres",
                "tau",
                "kappa",
                "elapsed_ms",
            }
        # ``nit`` is 0-indexed and contiguous from 0.
        assert [r["nit"] for r in iters] == list(range(len(iters)))

    def test_done_carries_solver_status_nits_elapsed(self) -> None:
        trace = ListTrace()
        _solve_hsd_with_trace(trace)
        done = [r for r in trace.records() if r["event"] == "hsd_done"]
        assert len(done) == 1
        row = done[0]
        assert row["solver"] == "HsdSolver"
        assert row["status"] in (0, 1, 2)
        assert row["nits"] >= 1
        assert row["elapsed_ms"] > 0.0

    def test_start_carries_solver_class(self) -> None:
        trace = ListTrace()
        _solve_hsd_with_trace(trace)
        starts = [r for r in trace.records() if r["event"] == "hsd_start"]
        assert starts[0]["solver"] == "HsdSolver"


class PdEventsTest(unittest.TestCase):
    """PrimalDualSolver (with and without equality constraints) emits pd_start/iteration/done."""

    def test_pd_brackets_run_exactly_once(self) -> None:
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("pd_start") == 1
        assert events.count("pd_done") == 1
        assert events[0] == "pd_start"
        assert events[-1] == "pd_done"

    def test_pd_emits_at_least_one_iteration(self) -> None:
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        assert iters

    def test_pd_iteration_schema(self) -> None:
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        for row in iters:
            assert set(row.keys()) >= {
                "time",
                "event",
                "nit",
                "eta",
                "dual_norm",
                "pri_norm",
                "residual_norm",
                "t",
                "step",
                "elapsed_ms",
            }
        # ``nit`` is 0-indexed and contiguous from 0 (converged iters don't emit,
        # so a missing tail is fine, but the prefix must be tight).
        assert [r["nit"] for r in iters] == list(range(len(iters)))

    def test_pd_no_equality_has_zero_pri_norm(self) -> None:
        """``PrimalDualSolver`` has no equality constraint so ``pri_norm`` is identically zero."""
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        for row in iters:
            assert row["pri_norm"] == 0.0

    def test_pd_done_carries_solver_status_nits_elapsed(self) -> None:
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        done = [r for r in trace.records() if r["event"] == "pd_done"]
        assert len(done) == 1
        row = done[0]
        assert row["solver"] == "PrimalDualSolver"
        assert row["status"] in (0, 1, 2)
        assert row["nits"] >= 1
        assert row["elapsed_ms"] > 0.0

    def test_equality_pd_emits_full_event_set(self) -> None:
        trace = ListTrace()
        _solve_equality_pd_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("pd_start") == 1
        assert events.count("pd_done") == 1
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        assert iters
        done = [r for r in trace.records() if r["event"] == "pd_done"]
        assert done[0]["solver"] == "PrimalDualSolver"

    def test_pd_iteration_carries_mpc_diagnostics(self) -> None:
        """The MPC path records sigma (centering), rho (ratio), and damping."""
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        for row in iters:
            assert {"sigma", "rho", "damping"} <= set(row.keys())
            assert 0.0 <= row["sigma"] <= 1.0  # Mehrotra sigma is clipped to [0, 1]
            assert row["damping"] >= 0.0

    def test_pd_iteration_carries_residual_breakdown_and_timing(self) -> None:
        """The MPC path records the KKT residual breakdown, true ||r_t||, the
        direction-solve time, and (since ListTrace opts in) the post-step ||r_t||."""
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        for row in iters:
            assert {"cent_norm", "rt_norm", "dir_ms", "rt_new"} <= set(row.keys())
            assert row["rt_norm"] > 0.0
            assert row["cent_norm"] >= 0.0
            assert row["dir_ms"] >= 0.0
            # ListTrace.want_pd_step_details() is True, so rt_new is populated.
            assert row["rt_new"] is not None

    def test_pd_start_carries_dims_and_tol(self) -> None:
        """The MPC path supplies problem dimensions and tolerance to pd_start."""
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        start = next(r for r in trace.records() if r["event"] == "pd_start")
        assert start["num_vars"] > 0
        assert start["num_ineq"] > 0
        assert start["tol"] > 0.0

    def test_pd_done_carries_reason(self) -> None:
        """pd_done carries a human-readable termination cause."""
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        done = next(r for r in trace.records() if r["event"] == "pd_done")
        assert isinstance(done["reason"], str)
        assert done["reason"]  # non-empty

    def test_pd_done_carries_true_duality_gap(self) -> None:
        """pd_done carries the one-time true duality gap (objective - dual)."""
        trace = ListTrace()
        _solve_pd_with_trace(trace)
        done = next(r for r in trace.records() if r["event"] == "pd_done")
        assert done["duality_gap"] is not None
        assert isinstance(done["duality_gap"], float)

    def test_pd_damping_engaged_event_recorded(self) -> None:
        """The damping-engaged callout records nit, damping value, and retries."""
        trace = ListTrace()
        trace.on_pd_damping_engaged(nit=3, damping=1.5e3, retries=2)
        rec = [r for r in trace.records() if r["event"] == "pd_damping_engaged"]
        assert len(rec) == 1
        assert rec[0]["nit"] == 3
        assert rec[0]["damping"] == 1.5e3
        assert rec[0]["retries"] == 2


class SaddleStationarityEventsTest(unittest.TestCase):
    """The barrier ``StationaritySolver`` emits ipm/centering/saddle_newton events."""

    def test_brackets_run_exactly_once(self) -> None:
        trace = ListTrace()
        _solve_barrier_stationarity_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("ipm_start") == 1
        assert events.count("ipm_done") == 1
        assert events[0] == "ipm_start"
        assert events[-1] == "ipm_done"

    def test_centering_pairs_balanced(self) -> None:
        trace = ListTrace()
        _solve_barrier_stationarity_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("centering_start") >= 1
        assert events.count("centering_start") == events.count("centering_done")

    def test_emits_saddle_newton_steps(self) -> None:
        trace = ListTrace()
        _solve_barrier_stationarity_with_trace(trace)
        steps = [r for r in trace.records() if r["event"] == "saddle_newton_step"]
        assert steps

    def test_saddle_newton_step_schema(self) -> None:
        trace = ListTrace()
        _solve_barrier_stationarity_with_trace(trace)
        steps = [r for r in trace.records() if r["event"] == "saddle_newton_step"]
        assert steps
        for row in steps:
            assert set(row.keys()) >= {
                "time",
                "event",
                "outer_nit",
                "inner_nit",
                "elapsed_ms",
                "residual_norm",
                "lambda_norm",
            }
            assert row["inner_nit"] >= 1
            assert row["elapsed_ms"] >= 0.0

    def test_converged_event_on_success(self) -> None:
        """A clean solve fires ``converged`` carrying the surrogate duality gap."""
        trace = ListTrace()
        _solve_barrier_stationarity_with_trace(trace)
        rows = [r for r in trace.records() if r["event"] == "converged"]
        assert len(rows) == 1
        assert set(rows[0].keys()) >= {"time", "event", "outer_nit", "duality_gap"}

    def test_ipm_done_carries_solver_name(self) -> None:
        trace = ListTrace()
        _solve_barrier_stationarity_with_trace(trace)
        starts = [r for r in trace.records() if r["event"] == "ipm_start"]
        dones = [r for r in trace.records() if r["event"] == "ipm_done"]
        assert starts[0]["solver"] == "StationaritySolver"
        assert dones[0]["solver"] == "StationaritySolver"
        assert dones[0]["elapsed_ms"] > 0.0


class PdStationarityEventsTest(unittest.TestCase):
    """``PrimalDualStationaritySolver`` reuses the pd_start/iteration/done family."""

    def test_pd_brackets_run_exactly_once(self) -> None:
        trace = ListTrace()
        _solve_pd_stationarity_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("pd_start") == 1
        assert events.count("pd_done") == 1
        assert events[0] == "pd_start"
        assert events[-1] == "pd_done"

    def test_pd_emits_at_least_one_iteration(self) -> None:
        trace = ListTrace()
        _solve_pd_stationarity_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        assert iters

    def test_pd_iteration_schema(self) -> None:
        trace = ListTrace()
        _solve_pd_stationarity_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        assert iters
        for row in iters:
            assert set(row.keys()) >= {
                "time",
                "event",
                "nit",
                "eta",
                "dual_norm",
                "pri_norm",
                "residual_norm",
                "t",
                "step",
                "elapsed_ms",
                "sigma",
            }
            # Mehrotra centering is clipped to [0, 1].
            assert 0.0 <= row["sigma"] <= 1.0
        # ``nit`` is 0-indexed and contiguous (converged iters don't emit).
        assert [r["nit"] for r in iters] == list(range(len(iters)))

    def test_pd_no_equality_has_zero_pri_norm(self) -> None:
        """The plain (no x-equality) path carries an identically-zero ``pri_norm``."""
        trace = ListTrace()
        _solve_pd_stationarity_with_trace(trace)
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        for row in iters:
            assert row["pri_norm"] == 0.0

    def test_pd_start_carries_dims_and_tol(self) -> None:
        trace = ListTrace()
        _solve_pd_stationarity_with_trace(trace)
        start = next(r for r in trace.records() if r["event"] == "pd_start")
        assert start["num_vars"] > 0
        assert start["num_ineq"] > 0
        assert start["tol"] > 0.0

    def test_pd_done_carries_solver_status_nits_reason(self) -> None:
        trace = ListTrace()
        _solve_pd_stationarity_with_trace(trace)
        done = [r for r in trace.records() if r["event"] == "pd_done"]
        assert len(done) == 1
        row = done[0]
        assert row["solver"] == "PrimalDualStationaritySolver"
        assert row["status"] in (0, 1, 2)
        assert row["nits"] >= 1
        assert row["elapsed_ms"] > 0.0
        assert isinstance(row["reason"], str)
        assert row["reason"]  # non-empty

    def test_equality_path_emits_full_event_set(self) -> None:
        """The bordered ``_run_equality`` path emits the same event family."""
        trace = ListTrace()
        _solve_pd_stationarity_equality_with_trace(trace)
        events = [r["event"] for r in trace.records()]
        assert events.count("pd_start") == 1
        assert events.count("pd_done") == 1
        iters = [r for r in trace.records() if r["event"] == "pd_iteration"]
        assert iters
        # The bordered path reports the primal-feasibility residual in ``pri_norm``
        # (a real float; near zero once Phase I has landed on ``A x = b``).
        for row in iters:
            assert isinstance(row["pri_norm"], float)
            assert row["pri_norm"] >= 0.0


class SolverLabelTest(unittest.TestCase):
    """``ObservabilitySettings.label`` sets the ``solver`` field of trace events."""

    def test_label_overrides_barrier_solver_name(self) -> None:
        """A label replaces the internal ``_BarrierAdapter`` name in ipm events."""
        problem, x0 = _make_bounded_quadratic_problem()
        trace = ListTrace()
        settings = OptimizationSettings(
            trace=trace,
            observability=ObservabilitySettings(label="SalesSolver"),
        )
        BarrierSolver(settings=settings).solve(problem=problem, x0=x0)
        starts = [r for r in trace.records() if r["event"] == "ipm_start"]
        dones = [r for r in trace.records() if r["event"] == "ipm_done"]
        assert starts[0]["solver"] == "SalesSolver"
        assert dones[0]["solver"] == "SalesSolver"

    def test_label_overrides_pd_solver_name(self) -> None:
        problem, x0 = _make_bounded_quadratic_problem()
        trace = ListTrace()
        settings = OptimizationSettings(
            trace=trace,
            observability=ObservabilitySettings(label="SalesSolver"),
        )
        PrimalDualSolver(settings=settings).solve(problem=problem, x0=x0)
        starts = [r for r in trace.records() if r["event"] == "pd_start"]
        dones = [r for r in trace.records() if r["event"] == "pd_done"]
        assert starts[0]["solver"] == "SalesSolver"
        assert dones[0]["solver"] == "SalesSolver"

    def test_label_overrides_equality_pd_solver_name(self) -> None:
        problem, x0 = _make_bounded_quadratic_problem()
        A = np.ones((1, problem.dimension))
        b = np.array([float(np.sum(x0))])
        problem = Problem(
            obj=problem.obj,
            ineq=problem.ineq,
            eq=EqualityConstraintLayer(A=A, b=b),
        )
        trace = ListTrace()
        settings = OptimizationSettings(
            trace=trace,
            observability=ObservabilitySettings(label="SalesSolver"),
        )
        PrimalDualSolver(settings=settings).solve(problem=problem, x0=x0)
        starts = [r for r in trace.records() if r["event"] == "pd_start"]
        dones = [r for r in trace.records() if r["event"] == "pd_done"]
        assert starts[0]["solver"] == "SalesSolver"
        assert dones[0]["solver"] == "SalesSolver"

    def test_default_name_when_no_label(self) -> None:
        """Without a label, the solver falls back to the (adapter) class name."""
        problem, x0 = _make_bounded_quadratic_problem()
        trace = ListTrace()
        BarrierSolver(settings=OptimizationSettings(trace=trace)).solve(
            problem=problem, x0=x0
        )
        starts = [r for r in trace.records() if r["event"] == "ipm_start"]
        assert starts[0]["solver"] in {"BarrierSolver", "_BarrierAdapter"}

    def test_label_overrides_hsd_solver_name(self) -> None:
        problem = Problem(
            obj=LinearObjective(c=np.array([1.0, 0.0, 0.0])),
            eq=EqualityConstraintLayer(
                A=np.array([[1.0, 0.0, 0.0]]), b=np.array([1.0])
            ),
        )
        trace = ListTrace()
        settings = OptimizationSettings(
            outer=OuterSettings(tolerance=1e-6),
            trace=trace,
            observability=ObservabilitySettings(label="SalesSolver"),
        )
        HsdSolver(cones=[SecondOrderCone(dim=3)], settings=settings).solve(
            problem=problem, x0=np.zeros(3)
        )
        starts = [r for r in trace.records() if r["event"] == "hsd_start"]
        dones = [r for r in trace.records() if r["event"] == "hsd_done"]
        assert starts[0]["solver"] == "SalesSolver"
        assert dones[0]["solver"] == "SalesSolver"


class DataFrameTraceTest(unittest.TestCase):
    """DataFrameTrace inherits ListTrace and adds to_dataframe()."""

    def test_inherits_list_trace(self) -> None:
        assert issubclass(DataFrameTrace, ListTrace)

    def test_to_dataframe_returns_one_row_per_event(self) -> None:
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("pandas not available")
        trace = DataFrameTrace()
        _solve_with_trace(trace)
        df = trace.to_dataframe()
        assert len(df) == len(trace.records())

    def test_to_dataframe_raises_without_pandas(self) -> None:
        """When pandas isn't installed, to_dataframe() raises a clear ImportError."""
        with patch.dict(sys.modules, {"pandas": None}):
            trace = DataFrameTrace()
            _solve_with_trace(trace)
            with self.assertRaisesRegex(ImportError, "pandas is required"):
                trace.to_dataframe()

    def test_dataframe_has_sparse_columns(self) -> None:
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("pandas not available")
        trace = DataFrameTrace()
        _solve_with_trace(trace)
        df = trace.to_dataframe()
        # Required columns from any event.
        assert "event" in df.columns
        assert "time" in df.columns
        # newton_step-specific column should appear as NaN-padded sparse column.
        assert "lambda_norm" in df.columns
        # btls-specific column.
        assert "btls_s" in df.columns
        # ipm_*-specific column.
        assert "solver" in df.columns
