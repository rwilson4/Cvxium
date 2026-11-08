"""``InteriorPointMethodSolver`` reports the best (tightest valid) duality-gap
certificate, not the last transient per-step estimate.

Every barrier iterate is strictly feasible (its objective is a valid upper bound
on p*) and every barrier multiplier is >= 0 (its dual is a valid lower bound), so
the tightest certificate is ``min(objective) - max(dual)`` across centering steps.
The per-step gap is non-monotone: at high ``t`` the conditioning degrades and the
dual sags even while the primal holds, so the *last* raw gap is not a trustworthy
convergence signal. ``solve`` must therefore record the running-best certificate in
``duality_gaps`` -- monotone non-increasing, ending at the reported
``suboptimality`` -- and never let a sagged final dual become the reported bound.
"""

import unittest
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from cvxium.linalg.systems import DiagonalSystem
from cvxium.problems.linear_constraints import (
    BoundsConstraintLayer,
)
from cvxium.problems.objective import (
    QuadraticObjective,
)
from cvxium.problems.problem import Problem
from cvxium.solvers.barrier_solver import (
    _BarrierAdapter,
)
from cvxium.solvers.optimization import (
    _CenteringOutcome,
)
from cvxium.solvers.settings import (
    OptimizationSettings,
    OuterSettings,
)


def _result(obj: float, dual: float, marker: float) -> _CenteringOutcome:
    """A centering result with a marker so the test can tell steps apart."""
    return _CenteringOutcome(
        solution=np.full(3, marker),
        objective_value=obj,
        barrier_objective_value=obj,
        dual_value=dual,
        equality_multipliers=np.zeros(0),
        inequality_multipliers=np.full(3, marker),
        suboptimalities=[obj - dual],
        nits=2,
        status=0,
        message="completed",
    )


def _pinned_t0(x0: npt.NDArray[np.float64]) -> float:
    """Pin the barrier init so num_steps is comfortably > 2."""
    return 1.0


class OptimizationBestBoundsTest(unittest.TestCase):
    def test_duality_gaps_track_running_best_not_raw_per_step(self) -> None:
        r"""The gap trace is the running-best certificate, not the raw per-step gap.

        A scripted centering sequence makes the dual sag hard on the final (high-``t``)
        step while the primal is unchanged. A raw trace would *spike* there; the
        running-best trace must stay monotone non-increasing, end at the reported
        ``suboptimality``, and keep the earlier tight dual as ``dual_value``.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        # tolerance below the achievable gap so the loop runs every step (no early
        # stop) and reaches the high-t step where the dual sags.
        settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-9))
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if last_step:
                # Dual sags hard at high t while the primal is unchanged.
                return _result(5.0, -100.0, 9.0)
            if outer_nit == 1:
                return _result(10.0, 2.0, 1.0)
            # Best primal (obj 5.0) certified tightly (gap 1e-4 < tolerance_soft).
            return _result(5.0, 4.9999, 2.0)

        # Pin the barrier init so num_steps is comfortably > 2.
        with (
            patch.object(adapter, "initialize_barrier_parameter", new=_pinned_t0),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        gaps = result.barrier_diagnostics.duality_gaps
        assert len(gaps) >= 3  # ran past the first two scripted steps to the sag
        assert all(gaps[i + 1] <= gaps[i] + 1e-9 for i in range(len(gaps) - 1)), gaps
        np.testing.assert_allclose(
            gaps[-1], result.suboptimality, rtol=1e-6, atol=1e-12
        )
        # The sagged final dual must not become the reported certificate.
        np.testing.assert_allclose(result.dual_value, 4.9999, rtol=1e-6, atol=1e-12)

    def test_a_nan_dual_is_no_certificate(self) -> None:
        """A NaN dual must not read as a zero duality gap.

        Comparisons against NaN are all False, so a NaN first dual used to become
        the incumbent for good, and the gap `max(0, min(nan, ...))` read as 0.0:
        the solve stopped after one centering step, reporting a NaN dual.
        """
        problem = Problem(
            obj=QuadraticObjective(
                c=-np.ones(3), hessian_contribution=DiagonalSystem(np.ones(3))
            ),
            ineq=[BoundsConstraintLayer(lb=0.0, dimension=3)],
        )
        settings = OptimizationSettings(outer=OuterSettings(tolerance=1e-3))
        adapter = _BarrierAdapter(problem=problem, settings=settings)

        def scripted(
            x: npt.NDArray[np.float64],
            t: float,
            last_step: bool,
            outer_nit: int,
        ) -> _CenteringOutcome:
            if outer_nit == 1:
                return _result(10.0, float("nan"), 1.0)
            return _result(5.0, 4.9999, 2.0)

        with (
            patch.object(adapter, "initialize_barrier_parameter", new=_pinned_t0),
            patch.object(adapter, "centering_step", new=scripted),
        ):
            result = adapter.solve(x0=np.full(3, 0.5))

        gaps = result.barrier_diagnostics.duality_gaps
        self.assertGreater(gaps[0], 0.0)
        self.assertGreaterEqual(len(gaps), 2)
        np.testing.assert_allclose(result.dual_value, 4.9999, rtol=1e-6, atol=1e-12)
