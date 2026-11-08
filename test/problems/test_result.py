"""Test the unified SolveResult types."""

from __future__ import annotations

import dataclasses
import json
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from parameterized import parameterized

from cvxium.problems.result import (
    BarrierDiagnostics,
    Duals,
    HsdDiagnostics,
    InfeasibilityCertificate,
    NoDualsError,
    NoDualValueError,
    NoObjectiveError,
    ObjectiveOutcome,
    PrimalDualDiagnostics,
    SaddleBarrierDiagnostics,
    SolverDiagnostics,
    SolveResult,
    SolveStatus,
    TwoBlockSolveResult,
    WrongDiagnosticsError,
)


def _minimal(status: SolveStatus = SolveStatus.FEASIBLE) -> SolveResult:
    """A spine-only result: feasibility solve, no objective, no duals."""
    return SolveResult(
        solution=np.zeros(3),
        status=status,
        message="ok",
        solver="equality",
        n_iterations=0,
    )


def _agg_axes() -> Axes:
    """A fresh Axes on the headless Agg backend for plot tests."""
    _, ax = plt.subplots()
    return ax


class ResultTest(unittest.TestCase):
    def test_spine_fields_need_no_narrowing(self) -> None:
        res = _minimal()
        np.testing.assert_array_equal(res.solution, np.zeros(3))
        assert res.status is SolveStatus.FEASIBLE
        assert res.message == "ok"
        assert res.solver == "equality"
        assert res.n_iterations == 0

    def test_feasibility_result_has_no_objective_or_duals(self) -> None:
        res = _minimal()
        with self.assertRaises(NoObjectiveError):
            _ = res.objective_value
        with self.assertRaises(NoObjectiveError):
            _ = res.suboptimality
        with self.assertRaises(NoObjectiveError):
            _ = res.dual_value
        with self.assertRaises(NoDualsError):
            _ = res.equality_multipliers
        with self.assertRaises(NoDualsError):
            _ = res.inequality_multipliers

    def test_optimization_result_exposes_objective_and_duals(self) -> None:
        res = SolveResult(
            solution=np.ones(2),
            status=SolveStatus.OPTIMAL,
            message="converged",
            solver="barrier",
            n_iterations=7,
            objective=ObjectiveOutcome(value=4.0, suboptimality=1e-9, dual_value=3.5),
            duals=Duals(
                equality_multipliers=np.array([1.0, 2.0]),
                inequality_multipliers=np.array([0.5]),
            ),
        )
        assert res.objective_value == 4.0
        np.testing.assert_allclose(res.suboptimality, 1e-9, rtol=1e-6, atol=1e-12)
        assert res.dual_value == 3.5
        assert res.objective is not None
        duality_gap = res.objective.duality_gap
        assert duality_gap is not None
        self.assertAlmostEqual(duality_gap, 0.5)
        np.testing.assert_array_equal(res.equality_multipliers, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(res.inequality_multipliers, np.array([0.5]))

    def test_partial_duals_raise_only_for_the_absent_axis(self) -> None:
        res = dataclasses.replace(
            _minimal(), duals=Duals(inequality_multipliers=np.array([1.0]))
        )
        np.testing.assert_array_equal(res.inequality_multipliers, np.array([1.0]))
        with self.assertRaises(NoDualsError):
            _ = res.equality_multipliers

    def test_duality_gap_is_none_without_dual_value(self) -> None:
        outcome = ObjectiveOutcome(value=2.0, suboptimality=0.1)
        assert outcome.dual_value is None
        assert outcome.duality_gap is None

    def test_missing_dual_value_raises(self) -> None:
        res = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL),
            objective=ObjectiveOutcome(value=2.0, suboptimality=0.1),
        )
        assert res.objective_value == 2.0
        with self.assertRaises(NoDualValueError):
            _ = res.dual_value

    @parameterized.expand(
        [
            (SolveStatus.OPTIMAL, True),
            (SolveStatus.ACCEPTABLE, True),
            (SolveStatus.FEASIBLE, True),
            (SolveStatus.MAX_ITERATIONS, False),
            (SolveStatus.INFEASIBLE, False),
            (SolveStatus.UNBOUNDED, False),
        ]
    )
    def test_succeeded(self, status: SolveStatus, expected: bool) -> None:
        assert _minimal(status=status).succeeded is expected

    def test_results_are_frozen(self) -> None:
        res = _minimal()
        # A non-literal attribute name exercises the runtime frozen guard without
        # tripping mypy's static assignability check or flake8 B010 (both of which
        # fire on a direct/constant assignment).
        field_name = "message"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(res, field_name, "mutated")

    def test_barrier_diagnostics_carry_trajectories(self) -> None:
        diag = BarrierDiagnostics(
            duality_gaps=(1.0, 0.1),
            inner_nits=(2, 3),
            inner_suboptimalities=((1.0,), (0.2, 0.1)),
        )
        res = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL), diagnostics=diag
        )
        assert isinstance(res.diagnostics, BarrierDiagnostics)
        assert res.diagnostics.duality_gaps == (1.0, 0.1)
        assert res.diagnostics.inner_nits == (2, 3)
        assert res.diagnostics.inner_suboptimalities[1] == (0.2, 0.1)

    def test_diagnostics_accessors_narrow_to_the_engine(self) -> None:
        barrier = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL),
            diagnostics=BarrierDiagnostics(duality_gaps=(1.0, 0.1)),
        )
        primal_dual = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL),
            diagnostics=PrimalDualDiagnostics(surrogate_gaps=(2.0, 0.2)),
        )
        hsd = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL),
            diagnostics=HsdDiagnostics(),
        )
        # Each accessor returns the matching variant without a call-site narrow.
        assert barrier.barrier_diagnostics.duality_gaps == (1.0, 0.1)
        assert primal_dual.primal_dual_diagnostics.surrogate_gaps == (2.0, 0.2)
        assert isinstance(hsd.hsd_diagnostics, HsdDiagnostics)
        # The wrong engine (and the no-diagnostics spine) raise, even under -O.
        with self.assertRaises(WrongDiagnosticsError):
            _ = barrier.primal_dual_diagnostics
        with self.assertRaises(WrongDiagnosticsError):
            _ = primal_dual.barrier_diagnostics
        with self.assertRaises(WrongDiagnosticsError):
            _ = hsd.barrier_diagnostics
        with self.assertRaises(WrongDiagnosticsError):
            _ = _minimal().barrier_diagnostics

    def test_infeasibility_certificate(self) -> None:
        cert = InfeasibilityCertificate(
            kind="infeasible", dual_witness=np.array([1.0, -1.0])
        )
        res = dataclasses.replace(
            _minimal(status=SolveStatus.INFEASIBLE), certificate=cert
        )
        assert res.certificate is not None
        assert res.certificate.kind == "infeasible"
        np.testing.assert_array_equal(
            res.certificate.dual_witness, np.array([1.0, -1.0])
        )
        assert res.certificate.primal_witness is None
        assert not res.succeeded

    def test_barrier_plot_convergence_returns_axes(self) -> None:
        diag = BarrierDiagnostics(duality_gaps=(1.0, 0.1, 0.01))
        out = diag.plot_convergence(ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Duality Gap"

    def test_barrier_plot_newton_convergence_returns_axes_and_validates_it(
        self,
    ) -> None:
        diag = BarrierDiagnostics(
            inner_nits=(2, 3),
            inner_suboptimalities=((1.0, 0.1), (1.0, 0.2, 0.05)),
        )
        out = diag.plot_newton_convergence(2, ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Sub-Optimality"
        assert out.get_title() == "Convergence of Centering Step 2"
        with self.assertRaisesRegex(ValueError, "must be between 1 and 2, inclusive"):
            diag.plot_newton_convergence(0)
        with self.assertRaisesRegex(ValueError, "must be between 1 and 2, inclusive"):
            diag.plot_newton_convergence(3)

    def test_barrier_plot_newton_convergence_empty_trajectories_raises_clearly(
        self,
    ) -> None:
        empty = BarrierDiagnostics()
        with self.assertRaisesRegex(
            ValueError, "no inner-Newton trajectories recorded"
        ):
            empty.plot_newton_convergence(1)

    def test_primal_dual_plot_convergence_labels_surrogate(self) -> None:
        diag = PrimalDualDiagnostics(surrogate_gaps=(2.0, 0.2, 0.02))
        out = diag.plot_convergence(ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Surrogate Duality Gap"

    def test_primal_dual_plot_feasibility_returns_axes(self) -> None:
        diag = PrimalDualDiagnostics(
            primal_residual_norms=(1.0, 0.1),
            dual_residual_norms=(0.5, 0.05),
        )
        out = diag.plot_feasibility(ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Feasibility Residual"

    def test_hsd_plot_convergence_returns_axes(self) -> None:
        diag = HsdDiagnostics(duality_gaps=(1.0, 0.1, 0.01))
        out = diag.plot_convergence(ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Duality Gap"

    def test_hsd_plot_homogeneity_plots_tau_and_kappa(self) -> None:
        diag = HsdDiagnostics(
            tau_history=(1.0, 0.9, 0.8), kappa_history=(1.0, 0.5, 0.1)
        )
        out = diag.plot_homogeneity(ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Homogenizing Variable"
        # Two series (tau and kappa) with a legend.
        assert len(out.get_lines()) == 2
        legend = out.get_legend()
        assert legend is not None


class QuadraticPhaseStepsTest(unittest.TestCase):
    """The step trajectory exists so consumers can assert on convergence."""

    @staticmethod
    def _diagnostics() -> BarrierDiagnostics:
        """Two outer steps: the first reaches the quadratic phase, the second not."""
        return BarrierDiagnostics(
            duality_gaps=(1.0, 0.1),
            inner_nits=(4, 2),
            inner_suboptimalities=((1.0, 0.5, 0.2, 0.01), (0.9, 0.4)),
            inner_btls_steps=((0.25, 0.5, 1.0, 1.0), (0.5, 0.25)),
            inner_quadratic_from=(2, None),
        )

    def test_it_reports_only_the_steps_taken_in_the_quadratic_phase(self) -> None:
        """The damped-phase steps are legitimately small; including them
        would make the assertion consumers want impossible to write."""
        self.assertEqual(
            self._diagnostics().steps_past_quadratic_convergence(), (1.0, 1.0)
        )

    def test_an_outer_step_that_never_converged_contributes_nothing(self) -> None:
        """`None` means the threshold was never crossed, not "index 0"."""
        diagnostics = dataclasses.replace(
            self._diagnostics(), inner_quadratic_from=(None, None)
        )

        self.assertEqual(diagnostics.steps_past_quadratic_convergence(), ())

    def test_it_round_trips_through_json(self) -> None:
        """Otherwise a persisted result silently loses the trajectory."""
        original = self._diagnostics()

        restored = SolverDiagnostics.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )

        assert isinstance(restored, BarrierDiagnostics)
        self.assertEqual(restored.inner_btls_steps, original.inner_btls_steps)
        self.assertEqual(restored.inner_quadratic_from, original.inner_quadratic_from)
        self.assertEqual(
            restored.steps_past_quadratic_convergence(),
            original.steps_past_quadratic_convergence(),
        )

    def test_the_saddle_subclass_round_trips_them_too(self) -> None:
        """The robust solvers report `SaddleBarrierDiagnostics`.

        It overrides `to_dict`/`from_dict`, so inheriting the fields is not
        enough -- without naming them in both, a robust solve's trajectory
        survives in memory and vanishes the moment the result is persisted.

        """
        original = SaddleBarrierDiagnostics(
            duality_gaps=(1.0,),
            inner_nits=(3,),
            inner_suboptimalities=((1.0, 0.4, 0.01),),
            inner_btls_steps=((0.5, 1.0, 1.0),),
            inner_quadratic_from=(1,),
            upper_bounds=(2.0,),
            lower_bounds=(0.5,),
            certified_upper_bounds=(2.1,),
            certified_lower_bounds=(0.4,),
        )

        restored = SolverDiagnostics.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )

        assert isinstance(restored, SaddleBarrierDiagnostics)
        self.assertEqual(restored.inner_btls_steps, original.inner_btls_steps)
        self.assertEqual(restored.steps_past_quadratic_convergence(), (1.0, 1.0))

    def test_a_payload_written_before_these_existed_still_loads(self) -> None:
        """Old serialized results predate the fields; they must not fail to load."""
        d = self._diagnostics().to_dict()
        del d["inner_btls_steps"]
        del d["inner_quadratic_from"]

        restored = SolverDiagnostics.from_dict(d)

        assert isinstance(restored, BarrierDiagnostics)
        self.assertEqual(restored.steps_past_quadratic_convergence(), ())


class ComparableWorkAcrossSolversTest(unittest.TestCase):
    """`iteration_count` is the one number a performance guard can compare.

    The unit differs by method -- Newton steps for the barrier, outer
    iterations for the predictor-corrector solvers -- but a guard only needs
    it to be stable within one solver, and it has to exist for all of them or
    the guard has blind spots.

    """

    def test_the_barrier_counts_inner_newton_steps(self) -> None:
        diagnostics = BarrierDiagnostics(inner_nits=(4, 2, 3))

        self.assertEqual(diagnostics.iteration_count(), 9)

    def test_primal_dual_counts_outer_iterations(self) -> None:
        diagnostics = PrimalDualDiagnostics(surrogate_gaps=(1.0, 0.5, 0.1))

        self.assertEqual(diagnostics.iteration_count(), 3)

    def test_hsd_counts_outer_iterations(self) -> None:
        diagnostics = HsdDiagnostics(duality_gaps=(1.0, 0.2))

        self.assertEqual(diagnostics.iteration_count(), 2)

    def test_a_solver_that_does_not_say_reports_zero(self) -> None:
        """Better than raising: a guard can skip it rather than crash."""
        self.assertEqual(SolverDiagnostics().iteration_count(), 0)


class PrimalDualStepSizesTest(unittest.TestCase):
    """The predictor-corrector analogue of the barrier's quadratic phase."""

    def test_the_tail_is_what_convergence_shows_up_in(self) -> None:
        """A method still taking short steps at the end is held back.

        There is no threshold to cross as there is for Newton, so the honest
        claim is about the tail rather than a phase flag.

        """
        diagnostics = PrimalDualDiagnostics(
            step_sizes=(0.2, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0)
        )

        self.assertEqual(diagnostics.late_step_sizes(), (1.0, 1.0))
        self.assertEqual(
            diagnostics.late_step_sizes(fraction=0.5), (1.0, 1.0, 1.0, 1.0)
        )

    def test_a_solver_with_no_recorded_steps_yields_nothing(self) -> None:
        """Empty is "no evidence"; callers must not read it as "passed"."""
        self.assertEqual(PrimalDualDiagnostics().late_step_sizes(), ())

    def test_a_fraction_outside_the_unit_interval_is_refused(self) -> None:
        """Out-of-range fractions all have plausible-looking wrong answers.

        Without the check, ``0`` and any negative floor to a one-element tail
        and anything above ``1`` returns the whole trajectory -- each reads
        like a tail and silently is not one. A caller who means 25% and types
        ``25`` should get an error, not the entire solve.
        """
        diagnostics = PrimalDualDiagnostics(step_sizes=(0.2, 0.4, 0.7, 1.0))
        for bad in (0.0, -0.25, 1.5, 25.0):
            with self.assertRaisesRegex(ValueError, r"fraction must be in \(0, 1\]"):
                diagnostics.late_step_sizes(fraction=bad)

    def test_the_whole_trajectory_is_a_legal_request(self) -> None:
        """``fraction=1`` is the inclusive end of the range, not an off-by-one."""
        diagnostics = PrimalDualDiagnostics(step_sizes=(0.2, 0.4, 0.7, 1.0))
        self.assertEqual(
            diagnostics.late_step_sizes(fraction=1.0), (0.2, 0.4, 0.7, 1.0)
        )

    def test_an_empty_trajectory_is_still_validated(self) -> None:
        """The range check precedes the empty short-circuit.

        Otherwise a bad fraction would be silently accepted on exactly the
        solves that carry no evidence to contradict it.
        """
        with self.assertRaisesRegex(ValueError, r"fraction must be in \(0, 1\]"):
            PrimalDualDiagnostics().late_step_sizes(fraction=0.0)

    def test_it_round_trips_through_json(self) -> None:
        original = PrimalDualDiagnostics(
            surrogate_gaps=(1.0, 0.1),
            primal_residual_norms=(0.5, 0.05),
            dual_residual_norms=(0.4, 0.04),
            step_sizes=(0.6, 1.0),
        )

        restored = SolverDiagnostics.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )

        assert isinstance(restored, PrimalDualDiagnostics)
        self.assertEqual(restored.step_sizes, (0.6, 1.0))

    def test_a_payload_written_before_step_sizes_still_loads(self) -> None:
        d = PrimalDualDiagnostics(surrogate_gaps=(1.0,)).to_dict()
        del d["step_sizes"]

        restored = SolverDiagnostics.from_dict(d)

        assert isinstance(restored, PrimalDualDiagnostics)
        self.assertEqual(restored.step_sizes, ())


class SaddleBarrierDiagnosticsTest(unittest.TestCase):
    """`SaddleBarrierDiagnostics`: barrier trajectories plus saddle-value bracket."""

    def test_is_a_barrier_diagnostics_and_carries_bounds(self) -> None:
        diag = SaddleBarrierDiagnostics(
            duality_gaps=(1.0, 0.5),
            inner_nits=(2, 3),
            inner_suboptimalities=((1.0,), (0.2, 0.1)),
            upper_bounds=(5.0, 4.5),
            lower_bounds=(4.0, 4.0),
        )
        # A subclass, so the plain barrier lens still narrows to it.
        assert isinstance(diag, BarrierDiagnostics)
        assert diag.upper_bounds == (5.0, 4.5)
        assert diag.lower_bounds == (4.0, 4.0)
        # The bracket half-widths sum to the surrogate gap it inherits.
        assert diag.upper_bounds[1] - diag.lower_bounds[1] == diag.duality_gaps[1]

    def test_both_diagnostics_lenses_narrow_to_the_saddle_variant(self) -> None:
        res = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL),
            diagnostics=SaddleBarrierDiagnostics(
                duality_gaps=(1.0,), upper_bounds=(1.0,), lower_bounds=(0.0,)
            ),
        )
        # The saddle lens returns the rich type; the base barrier lens still
        # works because it is a subclass.
        assert res.saddle_barrier_diagnostics.upper_bounds == (1.0,)
        assert res.barrier_diagnostics.duality_gaps == (1.0,)

    def test_plain_barrier_diagnostics_is_not_a_saddle_solve(self) -> None:
        res = dataclasses.replace(
            _minimal(status=SolveStatus.OPTIMAL),
            diagnostics=BarrierDiagnostics(duality_gaps=(1.0,)),
        )
        with self.assertRaises(WrongDiagnosticsError):
            _ = res.saddle_barrier_diagnostics

    def test_plot_bounds_returns_axes_with_both_curves(self) -> None:
        diag = SaddleBarrierDiagnostics(
            upper_bounds=(5.0, 4.1, 4.01),
            lower_bounds=(4.0, 3.9, 3.99),
        )
        out = diag.plot_bounds(ax=_agg_axes())
        assert isinstance(out, Axes)
        assert out.get_ylabel() == "Saddle Value"
        # Two step curves (upper and lower) plus a legend.
        assert len(out.get_lines()) == 2
        assert out.get_legend() is not None

    def test_plot_bounds_length_mismatch_raises(self) -> None:
        diag = SaddleBarrierDiagnostics(upper_bounds=(1.0, 2.0), lower_bounds=(0.0,))
        with self.assertRaises(ValueError):
            diag.plot_bounds(ax=_agg_axes())

    def test_plot_bounds_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            SaddleBarrierDiagnostics().plot_bounds(ax=_agg_axes())

    def test_carries_certified_bounds_including_deferrals(self) -> None:
        diag = SaddleBarrierDiagnostics(
            upper_bounds=(5.0, 4.5),
            lower_bounds=(4.0, 4.0),
            certified_upper_bounds=(float("inf"), 4.4),
            certified_lower_bounds=(float("-inf"), 4.1),
        )
        # The certified suspenders default empty; here they carry a deferral
        # (+/-inf) at step 0 and a genuine guaranteed bracket at step 1.
        assert diag.certified_upper_bounds == (float("inf"), 4.4)
        assert diag.certified_lower_bounds == (float("-inf"), 4.1)
        # The finite certified bracket is tighter than the closed-form belt.
        assert diag.certified_upper_bounds[1] <= diag.upper_bounds[1]
        assert diag.certified_lower_bounds[1] >= diag.lower_bounds[1]

    def test_certified_bounds_default_empty(self) -> None:
        diag = SaddleBarrierDiagnostics(upper_bounds=(1.0,), lower_bounds=(0.0,))
        assert diag.certified_upper_bounds == ()
        assert diag.certified_lower_bounds == ()

    def test_plot_bounds_overlays_certified_curves(self) -> None:
        diag = SaddleBarrierDiagnostics(
            upper_bounds=(5.0, 4.1),
            lower_bounds=(4.0, 3.9),
            certified_upper_bounds=(4.9, 4.05),
            certified_lower_bounds=(4.05, 3.95),
        )
        out = diag.plot_bounds(ax=_agg_axes())
        # Two belt curves plus two dashed certified curves.
        assert len(out.get_lines()) == 4

    def test_plot_bounds_masks_certified_deferrals(self) -> None:
        # +/-inf deferrals must not raise; they are masked to NaN and skipped.
        diag = SaddleBarrierDiagnostics(
            upper_bounds=(5.0, 4.1),
            lower_bounds=(4.0, 3.9),
            certified_upper_bounds=(float("inf"), 4.05),
            certified_lower_bounds=(float("-inf"), 3.95),
        )
        out = diag.plot_bounds(ax=_agg_axes())
        assert len(out.get_lines()) == 4

    def test_plot_bounds_certified_length_mismatch_raises(self) -> None:
        diag = SaddleBarrierDiagnostics(
            upper_bounds=(5.0, 4.1),
            lower_bounds=(4.0, 3.9),
            certified_upper_bounds=(4.9,),
            certified_lower_bounds=(4.05,),
        )
        with self.assertRaises(ValueError):
            diag.plot_bounds(ax=_agg_axes())


class TwoBlockSolveResultTest(unittest.TestCase):
    """`TwoBlockSolveResult`: flat spine plus x-/y-block accessors."""

    def test_from_blocks_concatenates_and_splits(self) -> None:
        res = TwoBlockSolveResult.from_blocks(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([4.0, 5.0]),
            status=SolveStatus.OPTIMAL,
            message="converged",
            solver="barrier saddle",
            n_iterations=5,
        )
        # The spine stays a flat concatenation for generic consumers.
        np.testing.assert_array_equal(res.solution, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert res.x_dimension == 3
        np.testing.assert_array_equal(res.x, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(res.y, np.array([4.0, 5.0]))

    def test_is_a_solveresult_with_composed_parts(self) -> None:
        res = TwoBlockSolveResult.from_blocks(
            x=np.array([1.0]),
            y=np.array([2.0]),
            status=SolveStatus.OPTIMAL,
            message="converged",
            solver="primal-dual saddle",
            n_iterations=9,
            objective=ObjectiveOutcome(value=4.0, suboptimality=1e-9),
            duals=Duals(inequality_multipliers=np.array([0.5, 0.25])),
            diagnostics=PrimalDualDiagnostics(surrogate_gaps=(2.0, 0.2)),
        )
        # A TwoBlockSolveResult IS a SolveResult: every generic accessor works.
        assert isinstance(res, SolveResult)
        assert res.succeeded
        assert res.objective_value == 4.0
        np.testing.assert_array_equal(res.inequality_multipliers, np.array([0.5, 0.25]))
        assert res.primal_dual_diagnostics.surrogate_gaps == (2.0, 0.2)

    @parameterized.expand(
        [
            (0, np.array([]), np.array([1.0, 2.0])),
            (2, np.array([1.0, 2.0]), np.array([])),
        ]
    )
    def test_degenerate_split_points(
        self,
        x_dimension: int,
        expected_x: npt.NDArray[np.float64],
        expected_y: npt.NDArray[np.float64],
    ) -> None:
        res = TwoBlockSolveResult(
            solution=np.array([1.0, 2.0]),
            status=SolveStatus.OPTIMAL,
            message="ok",
            solver="barrier saddle",
            n_iterations=1,
            x_dimension=x_dimension,
        )
        np.testing.assert_array_equal(res.x, expected_x)
        np.testing.assert_array_equal(res.y, expected_y)

    def test_split_point_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"outside \[0, 2\]"):
            TwoBlockSolveResult(
                solution=np.array([1.0, 2.0]),
                status=SolveStatus.OPTIMAL,
                message="ok",
                solver="barrier saddle",
                n_iterations=1,
                x_dimension=3,
            )

    def test_is_frozen(self) -> None:
        res = TwoBlockSolveResult.from_blocks(
            x=np.array([1.0]),
            y=np.array([2.0]),
            status=SolveStatus.OPTIMAL,
            message="ok",
            solver="barrier saddle",
            n_iterations=1,
        )
        field_name = "x_dimension"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(res, field_name, 0)


class ObjectiveOutcomeNegationTest(unittest.TestCase):
    """`ObjectiveOutcome.negated` and the sense-invariant `duality_gap`."""

    def test_negated_flips_value_and_dual_bound(self) -> None:
        """Negating swaps the ``-g`` minimization view for the ``g`` maximization view."""
        # A -g minimization reporting value=-4.0 with a lower bound of -4.5.
        minimized = ObjectiveOutcome(value=-4.0, suboptimality=1e-9, dual_value=-4.5)
        maximized = minimized.negated()
        self.assertAlmostEqual(maximized.value, 4.0)  # g(w*)
        assert maximized.dual_value is not None
        self.assertAlmostEqual(maximized.dual_value, 4.5)  # upper bound on g

    def test_negated_preserves_suboptimality(self) -> None:
        """The nonnegative gap magnitude is sense-invariant and carries over."""
        minimized = ObjectiveOutcome(value=-4.0, suboptimality=2.5e-8, dual_value=-4.5)
        self.assertEqual(minimized.negated().suboptimality, 2.5e-8)

    def test_negated_without_dual_value_stays_none(self) -> None:
        """A solve with no dual value negates to one with no dual value."""
        maximized = ObjectiveOutcome(value=-2.0, suboptimality=0.1).negated()
        self.assertIsNone(maximized.dual_value)
        self.assertIsNone(maximized.duality_gap)

    def test_duality_gap_is_nonnegative_in_both_senses(self) -> None:
        """The gap reads as the same positive magnitude before and after negation."""
        minimized = ObjectiveOutcome(value=-4.0, suboptimality=1e-9, dual_value=-4.5)
        maximized = minimized.negated()
        assert minimized.duality_gap is not None
        assert maximized.duality_gap is not None
        self.assertAlmostEqual(minimized.duality_gap, 0.5)
        self.assertAlmostEqual(maximized.duality_gap, 0.5)

    def test_double_negation_is_identity(self) -> None:
        """Negating twice restores every reported quantity."""
        original = ObjectiveOutcome(value=-4.0, suboptimality=1e-9, dual_value=-4.5)
        round_trip = original.negated().negated()
        self.assertAlmostEqual(round_trip.value, original.value)
        assert round_trip.dual_value is not None
        assert original.dual_value is not None
        self.assertAlmostEqual(round_trip.dual_value, original.dual_value)
        self.assertEqual(round_trip.suboptimality, original.suboptimality)


def _round_trip(res: SolveResult) -> SolveResult:
    """Serialize a result to a JSON string and reconstruct it."""
    return SolveResult.deserialize(res.serialize())


class SerializationTest(unittest.TestCase):
    """Full serialize/deserialize round-trips over the SolveResult hierarchy.

    The master invariant is ``deserialize(serialize(r)).to_dict() == r.to_dict()``:
    ``to_dict`` produces a plain JSON-ready structure of ints, floats, strings and
    ``None``, so dict equality is an honest value comparison (JSON round-trips
    Python floats exactly, and ``inf == inf`` holds). Targeted asserts add back the
    checks dict-equality cannot make: concrete type preservation and array dtype.
    """

    def test_spine_only_round_trips_and_stays_base_type(self) -> None:
        res = _minimal(SolveStatus.FEASIBLE)
        back = _round_trip(res)
        self.assertIs(type(back), SolveResult)
        self.assertEqual(back.to_dict(), res.to_dict())
        self.assertEqual(back.status, SolveStatus.FEASIBLE)
        np.testing.assert_array_equal(back.solution, res.solution)
        self.assertIsNone(back.objective)
        self.assertIsNone(back.duals)
        self.assertIsNone(back.diagnostics)
        self.assertIsNone(back.certificate)
        self.assertIsNone(back.phase1)

    def test_full_result_round_trips_every_part(self) -> None:
        res = SolveResult(
            solution=np.array([1.0, 2.0, 3.0]),
            status=SolveStatus.OPTIMAL,
            message="done",
            solver="barrier",
            n_iterations=7,
            objective=ObjectiveOutcome(value=1.5, suboptimality=1e-8, dual_value=1.4),
            duals=Duals(
                equality_multipliers=np.array([0.5]),
                inequality_multipliers=np.array([0.1, 0.2]),
            ),
            diagnostics=BarrierDiagnostics(
                duality_gaps=(1.0, 0.1, 0.01),
                inner_nits=(3, 2),
                inner_suboptimalities=((1.0, 0.1), (0.5,)),
            ),
            phase1=_minimal(SolveStatus.FEASIBLE),
        )
        back = _round_trip(res)
        self.assertEqual(back.to_dict(), res.to_dict())
        self.assertIs(type(back), SolveResult)
        assert back.objective is not None
        self.assertEqual(back.objective.dual_value, 1.4)
        self.assertIsInstance(back.diagnostics, BarrierDiagnostics)
        np.testing.assert_array_equal(back.inequality_multipliers, np.array([0.1, 0.2]))
        self.assertEqual(back.inequality_multipliers.dtype, np.float64)
        assert back.phase1 is not None
        self.assertEqual(back.phase1.status, SolveStatus.FEASIBLE)

    @parameterized.expand(
        [
            ("base", SolverDiagnostics()),
            (
                "barrier",
                BarrierDiagnostics(
                    duality_gaps=(1.0, 0.1),
                    inner_nits=(2, 1),
                    inner_suboptimalities=((1.0, 0.5), (0.2,)),
                ),
            ),
            (
                "saddle",
                SaddleBarrierDiagnostics(
                    duality_gaps=(1.0,),
                    inner_nits=(2,),
                    inner_suboptimalities=((1.0, 0.5),),
                    upper_bounds=(2.0,),
                    lower_bounds=(1.0,),
                    certified_upper_bounds=(2.5,),
                    certified_lower_bounds=(0.5,),
                ),
            ),
            (
                "primal_dual",
                PrimalDualDiagnostics(
                    surrogate_gaps=(1.0, 0.1),
                    primal_residual_norms=(0.3, 0.2),
                    dual_residual_norms=(0.4, 0.1),
                ),
            ),
            (
                "hsd",
                HsdDiagnostics(
                    duality_gaps=(1.0, 0.1),
                    tau_history=(0.9, 0.95),
                    kappa_history=(0.1, 0.05),
                ),
            ),
        ]
    )
    def test_diagnostics_round_trip_preserves_type(
        self, _name: str, diag: SolverDiagnostics
    ) -> None:
        back = SolverDiagnostics.from_dict(diag.to_dict())
        self.assertIs(type(back), type(diag))
        self.assertEqual(back.to_dict(), diag.to_dict())

    def test_unknown_diagnostics_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            SolverDiagnostics.from_dict({"__type__": "NoSuchDiagnostics"})

    def test_saddle_infinite_deferrals_survive_json_round_trip(self) -> None:
        # +/-inf certified-bound deferrals must survive the JSON string round-trip
        # (json's default allow_nan=True encodes them as Infinity/-Infinity).
        res = TwoBlockSolveResult.from_blocks(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0]),
            status=SolveStatus.OPTIMAL,
            message="ok",
            solver="saddle_barrier",
            n_iterations=4,
            diagnostics=SaddleBarrierDiagnostics(
                duality_gaps=(1.0,),
                inner_nits=(2,),
                inner_suboptimalities=((1.0,),),
                upper_bounds=(2.0,),
                lower_bounds=(1.0,),
                certified_upper_bounds=(float("inf"),),
                certified_lower_bounds=(float("-inf"),),
            ),
        )
        back = _round_trip(res)
        self.assertEqual(back.to_dict(), res.to_dict())
        diag = back.saddle_barrier_diagnostics
        self.assertEqual(diag.certified_upper_bounds, (float("inf"),))
        self.assertEqual(diag.certified_lower_bounds, (float("-inf"),))

    def test_two_block_round_trips_as_subclass(self) -> None:
        res = TwoBlockSolveResult.from_blocks(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0, 5.0]),
            status=SolveStatus.OPTIMAL,
            message="ok",
            solver="primal_dual",
            n_iterations=5,
            diagnostics=PrimalDualDiagnostics(
                surrogate_gaps=(1.0, 0.1),
                primal_residual_norms=(0.2, 0.1),
                dual_residual_norms=(0.3, 0.1),
            ),
        )
        back = _round_trip(res)
        self.assertIsInstance(back, TwoBlockSolveResult)
        assert isinstance(back, TwoBlockSolveResult)
        self.assertEqual(back.x_dimension, 2)
        np.testing.assert_array_equal(back.x, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(back.y, np.array([3.0, 4.0, 5.0]))
        self.assertEqual(back.to_dict(), res.to_dict())

    def test_infeasibility_certificate_round_trips(self) -> None:
        res = SolveResult(
            solution=np.zeros(2),
            status=SolveStatus.INFEASIBLE,
            message="infeasible",
            solver="hsd",
            n_iterations=9,
            certificate=InfeasibilityCertificate(
                kind="infeasible", dual_witness=np.array([1.0, -1.0])
            ),
            diagnostics=HsdDiagnostics(
                duality_gaps=(1.0,), tau_history=(0.9,), kappa_history=(0.1,)
            ),
        )
        back = _round_trip(res)
        self.assertEqual(back.to_dict(), res.to_dict())
        assert back.certificate is not None
        self.assertEqual(back.certificate.kind, "infeasible")
        np.testing.assert_array_equal(
            back.certificate.dual_witness, np.array([1.0, -1.0])
        )
        self.assertIsNone(back.certificate.primal_witness)

    def test_unknown_certificate_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            InfeasibilityCertificate.from_dict({"__type__": "x", "kind": "bogus"})

    def test_partial_duals_round_trip(self) -> None:
        res = dataclasses.replace(
            _minimal(SolveStatus.OPTIMAL),
            duals=Duals(inequality_multipliers=np.array([0.3, 0.4])),
        )
        back = _round_trip(res)
        assert back.duals is not None
        self.assertIsNone(back.duals.equality_multipliers)
        np.testing.assert_array_equal(
            back.duals.inequality_multipliers, np.array([0.3, 0.4])
        )

    def test_objective_without_dual_value_round_trips(self) -> None:
        oo = ObjectiveOutcome(value=2.0, suboptimality=1e-7)
        back = ObjectiveOutcome.from_dict(oo.to_dict())
        self.assertEqual(back.value, 2.0)
        self.assertEqual(back.suboptimality, 1e-7)
        self.assertIsNone(back.dual_value)

    def test_unknown_solve_result_type_raises(self) -> None:
        d = _minimal().to_dict()
        d["__type__"] = "NoSuchResult"
        with self.assertRaises(ValueError):
            SolveResult.from_dict(d)

    def test_to_dict_is_json_serializable(self) -> None:
        # A direct guard that to_dict produces only JSON-native types.
        res = SolveResult(
            solution=np.array([1.0, 2.0]),
            status=SolveStatus.MAX_ITERATIONS,
            message="stopped",
            solver="barrier",
            n_iterations=3,
            objective=ObjectiveOutcome(value=1.0, suboptimality=1e-3),
        )
        reloaded = json.loads(json.dumps(res.to_dict()))
        self.assertEqual(reloaded["__type__"], "SolveResult")
        self.assertEqual(reloaded["status"], "max_iterations")
