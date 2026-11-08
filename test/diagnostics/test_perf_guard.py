"""Install the performance guard for this suite, and test the guard itself.

Importing this module is what arms the guard: both unittest and pytest import
every ``test_*.py`` before running any test, so an install at import time is
in place for the whole run regardless of runner.

"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from cvxium.diagnostics import perf_guard
from cvxium.problems.result import (
    PrimalDualDiagnostics,
    SolverDiagnostics,
    SolveResult,
    SolveStatus,
)

perf_guard.install(Path(__file__).parent.parent / "perf_baseline.json")


def _result(
    n_iterations: int,
    diagnostics: Any = None,
    phase1: SolveResult | None = None,
) -> SolveResult:
    """A spine-only result, with `diagnostics`/`phase1` left to the caller."""
    return SolveResult(
        solution=np.zeros(1),
        status=SolveStatus.FEASIBLE,
        message="ok",
        solver="test",
        n_iterations=n_iterations,
        diagnostics=diagnostics,
        phase1=phase1,
    )


class RecorderComparisonTest(unittest.TestCase):
    """The comparison, exercised directly rather than through a real solve."""

    @staticmethod
    def _recorder(baseline: dict[str, int], tmp: Path) -> perf_guard._Recorder:
        path = tmp / "baseline.json"
        path.write_text(json.dumps(baseline))
        return perf_guard._Recorder(path, tolerance=0.1, writing=False)

    def _check(self, baseline: dict[str, int], observed: int) -> str | None:
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder(baseline, Path(d))
            r.begin("T.test_x")
            r._running_total = observed
            return r.end()

    def test_a_count_inside_the_band_passes(self) -> None:
        self.assertIsNone(self._check({"T.test_x": 100}, 105))

    def test_a_material_rise_fails(self) -> None:
        message = self._check({"T.test_x": 100}, 120)

        self.assertIsNotNone(message)
        self.assertIn("more", str(message))

    def test_a_material_drop_also_fails(self) -> None:
        """A sudden drop is how a weakened fixture looks, not only good news."""
        message = self._check({"T.test_x": 100}, 80)

        self.assertIsNotNone(message)
        self.assertIn("fewer", str(message))

    def test_a_test_with_no_baseline_is_not_a_regression(self) -> None:
        """Adding a test should not fail on someone; recording is deliberate."""
        self.assertIsNone(self._check({}, 100))

    def test_tiny_counts_are_not_guarded(self) -> None:
        """A relative band around 3 iterations means nothing."""
        self.assertIsNone(self._check({"T.test_x": 3}, 6))

    def test_small_counts_tolerate_a_couple_of_iterations(self) -> None:
        """One iteration of cross-machine drift must not fail a build.

        At a baseline of 14 the relative band alone admits only 13-15, and a
        solve finishing just inside a tolerance can legitimately take an
        iteration more on a different machine. CI caught exactly this.

        """
        self.assertIsNone(self._check({"T.test_x": 14}, 16))

    def test_a_real_regression_still_fails_a_small_count(self) -> None:
        """The slack must not swallow the defect it was added around.

        The half-step bug moved one test from 9 to 16; that has to still fire.

        """
        self.assertIsNotNone(self._check({"T.test_x": 9}, 16))

    def test_a_big_drop_below_the_floor_still_fails(self) -> None:
        """The noise floor is about the baseline, not about the new count.

        A baseline of 100 that collapses to 3 is exactly the "silently
        loosened tolerance or a fixture that stopped exercising the solver"
        case the drop-detection exists for -- gating on the (small) observed
        count instead of the baseline let it through silently.

        """
        message = self._check({"T.test_x": 100}, 3)

        self.assertIsNotNone(message)
        self.assertIn("fewer", str(message))

    def test_a_small_baseline_is_not_guarded_even_if_total_grows(self) -> None:
        """A baseline recorded below the floor stays ungated either way."""
        self.assertIsNone(self._check({"T.test_x": 3}, 100))

    def test_an_ignored_test_is_not_guarded(self) -> None:
        """For counts that are bimodal across machines, not merely noisy."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": 100, "__ignore__": ["T.test_x"]}))
            r = perf_guard._Recorder(path, tolerance=0.1, writing=False)
            r.begin("T.test_x")
            r._running_total = 500

            self.assertIsNone(r.end())

    def test_the_message_says_how_to_regenerate(self) -> None:
        """A failure nobody knows how to resolve gets the guard deleted."""
        message = str(self._check({"T.test_x": 100}, 200))

        self.assertIn("CVXIUM_PERF_BASELINE_WRITE=1", message)

    def test_a_stringified_baseline_value_does_not_raise(self) -> None:
        """A stringified baseline value should parse and compare normally, not crash."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": "100"}))
            r = perf_guard._Recorder(path, tolerance=0.1, writing=False)
            r.begin("T.test_x")
            r._running_total = 105

            self.assertIsNone(r.end())

    def test_a_malformed_ignore_entry_raises_instead_of_silently_degrading(
        self,
    ) -> None:
        """A hand-edited `__ignore__` must fail loudly, not silently degrade.

        A bare string value for `__ignore__` would otherwise pass through
        `frozenset(...)` as a set of individual characters, so the intended
        test stays guarded with no warning; a non-iterable value would raise
        a bare `TypeError` that escapes `install_from_environment` and
        crashes the import of Cvxium for the whole run.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(
                json.dumps({"T.test_x": 100, perf_guard._IGNORE_KEY: "T.test_x"})
            )

            with self.assertRaises(ValueError):
                perf_guard._Recorder(path, tolerance=0.1, writing=False)


class PerTestToleranceTest(unittest.TestCase):
    """The object-form baseline entry, ``{"expected": N, "tolerance": T}``.

    Reserved for a test whose count is legitimately bimodal across machines
    (e.g. one that sits on the feasibility boundary) but not so wide that
    ``__ignore__`` -- which would also let a real regression through -- is the
    honest answer.
    """

    @staticmethod
    def _recorder(baseline: dict[str, object], tmp: Path) -> perf_guard._Recorder:
        path = tmp / "baseline.json"
        path.write_text(json.dumps(baseline))
        return perf_guard._Recorder(path, tolerance=0.1, writing=False)

    def _check(self, baseline: dict[str, object], observed: int) -> str | None:
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder(baseline, Path(d))
            r.begin("T.test_x")
            r._running_total = observed
            return r.end()

    def test_a_bare_int_entry_still_uses_the_default_tolerance(self) -> None:
        """Backward compatibility: every existing baseline entry is a bare int."""
        self.assertIsNone(self._check({"T.test_x": 100}, 109))
        self.assertIsNotNone(self._check({"T.test_x": 100}, 120))

    def test_an_object_entry_widens_the_band_past_the_default(self) -> None:
        """A move the default 10% band would reject, admitted by the override."""
        message = self._check({"T.test_x": 100}, 120)
        self.assertIsNotNone(message)

        widened = self._check({"T.test_x": {"expected": 100, "tolerance": 0.25}}, 120)
        self.assertIsNone(widened)

    def test_an_object_entry_can_also_narrow_the_band(self) -> None:
        """The override is not one-directional: it can tighten, not just widen.

        A delta of 2 sits inside the ``_ABS_SLACK`` floor (3), so this only
        distinguishes a real override from one silently clamped to the floor.
        """
        self.assertIsNone(self._check({"T.test_x": 100}, 102))

        message = self._check({"T.test_x": {"expected": 100, "tolerance": 0.01}}, 102)
        self.assertIsNotNone(message)

    def test_an_object_entry_can_narrow_below_the_floor(self) -> None:
        """The floor gates the default tolerance, not an explicit override.

        ``expected`` here sits below ``_MIN_ITERATIONS`` (8), where the
        default tolerance is deliberately not enforced -- but an override is
        a choice about this one test and must still be honored, or "narrow
        the band" is a documented lie for every sub-floor baseline.
        """
        self.assertIsNone(self._check({"T.test_x": 5}, 6))

        message = self._check({"T.test_x": {"expected": 5, "tolerance": 0.1}}, 6)
        self.assertIsNotNone(message)

    def test_a_real_regression_still_fails_past_the_widened_band(self) -> None:
        """The override must not swallow a move it was not sized for."""
        message = self._check({"T.test_x": {"expected": 82, "tolerance": 0.13}}, 150)
        self.assertIsNotNone(message)
        self.assertIn("more", str(message))

    def test_missing_tolerance_key_falls_back_to_the_suite_default(self) -> None:
        """An object entry may omit tolerance and rely on the general default."""
        self.assertIsNone(self._check({"T.test_x": {"expected": 100}}, 109))
        self.assertIsNotNone(self._check({"T.test_x": {"expected": 100}}, 120))

    def test_missing_expected_key_raises_a_clear_error_naming_the_test(self) -> None:
        """A malformed entry must surface an actionable error, not a bare KeyError."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": {"tolerance": 0.1}}))

            with self.assertRaises(ValueError) as ctx:
                perf_guard._Recorder(path, tolerance=0.1, writing=False)

            self.assertIn("T.test_x", str(ctx.exception))

    def test_a_non_integral_bare_entry_raises_instead_of_truncating(self) -> None:
        """15.9 must not silently become 15 and shift the comparison band."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": 15.9}))

            with self.assertRaises(ValueError) as ctx:
                perf_guard._Recorder(path, tolerance=0.1, writing=False)

            self.assertIn("T.test_x", str(ctx.exception))

    def test_a_non_integral_object_entry_raises_instead_of_truncating(self) -> None:
        """Same guard for the object form's ``expected`` key."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": {"expected": 15.9}}))

            with self.assertRaises(ValueError) as ctx:
                perf_guard._Recorder(path, tolerance=0.1, writing=False)

            self.assertIn("T.test_x", str(ctx.exception))

    def test_an_out_of_range_tolerance_override_raises_instead_of_silently_applying(
        self,
    ) -> None:
        """A typo'd `10` (meant `0.10`) must fail loudly, not widen the band 1000%."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(
                json.dumps({"T.test_x": {"expected": 100, "tolerance": 10}})
            )

            with self.assertRaises(ValueError) as ctx:
                perf_guard._Recorder(path, tolerance=0.1, writing=False)

            self.assertIn("T.test_x", str(ctx.exception))

    def test_a_negative_tolerance_override_raises(self) -> None:
        """A negative band is nonsensical, not merely a stricter one."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(
                json.dumps({"T.test_x": {"expected": 100, "tolerance": -0.1}})
            )

            with self.assertRaises(ValueError) as ctx:
                perf_guard._Recorder(path, tolerance=0.1, writing=False)

            self.assertIn("T.test_x", str(ctx.exception))

    def test_regenerating_preserves_the_tolerance_override(self) -> None:
        """Regeneration must not flatten an override back to a bare int.

        A hand-tuned tolerance is metadata about the test, not about the one
        run that happened to regenerate the file -- losing it silently on the
        next ``CVXIUM_PERF_BASELINE_WRITE=1`` would make the override a trap
        that only survives until someone next regenerates.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(
                json.dumps({"T.test_x": {"expected": 82, "tolerance": 0.13}})
            )
            out = Path(d) / "regenerated.json"
            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_x")
            r._running_total = 90
            r.end()

            with patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}):
                r.write_if_requested()

            self.assertEqual(
                json.loads(out.read_text()),
                {"T.test_x": {"total": 90, "tolerance": 0.13}},
            )


class AbandonedTallyTest(unittest.TestCase):
    """A tally the guard declines to judge must not be charged to anyone else."""

    @staticmethod
    def _recorder(baseline: dict[str, int], tmp: Path) -> perf_guard._Recorder:
        path = tmp / "baseline.json"
        path.write_text(json.dumps(baseline))
        return perf_guard._Recorder(path, tolerance=0.1, writing=False)

    def test_an_abandoned_tally_leaves_no_state_behind(self) -> None:
        """Abandon clears the tally outright, so nothing can inherit it.

        ``begin`` would reset it anyway, so this is the contract rather than a
        reproduction of a live bug -- it is what lets the skip path (which
        returns before ``doCleanups``, so the guard's cleanup never fires)
        leave nothing dangling when no test follows.
        """
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.test_next": 100}, Path(d))
            r.begin("T.test_skipped")
            r._running_total = 5000

            r.abandon()

            self.assertIsNone(r._current)
            self.assertEqual(r._running_total, 0)
            self.assertEqual(r._depth, 0)

    def test_abandoning_renders_no_verdict_on_the_abandoned_test(self) -> None:
        """Abandon is silence, not a pass and not a failure."""
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.test_x": 100}, Path(d))
            r.begin("T.test_x")
            r._running_total = 5000

            r.abandon()

            # Nothing to report, and nothing recorded for regeneration either.
            self.assertIsNone(r.end())
            self.assertNotIn("T.test_x", r._observed)

    def test_a_test_with_no_solves_is_not_recorded_for_regeneration(self) -> None:
        """A test that never solved anything must not gain a baseline entry.

        Otherwise a regeneration run (`CVXIUM_PERF_BASELINE_WRITE=1`) would
        merge a `"Class.test_method": 0` entry into the output for every
        non-solving test in the suite -- and the checked-in baseline has no
        such entries, so that is not what a real regeneration produces.
        """
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({}, Path(d))
            r.begin("T.test_no_solves")
            # r._running_total stays 0: the test never entered a solve.

            self.assertIsNone(r.end())
            self.assertNotIn("T.test_no_solves", r._observed)

    def test_a_baselined_test_dropping_to_zero_is_still_recorded_for_regeneration(
        self,
    ) -> None:
        """A regression to zero must be regeneratable, not a permanent trap.

        Unlike a brand-new zero-total test, this one already has a baseline
        entry, so it fails here with a "fewer iterations" message whose hint
        promises a regen will clear it. That promise is a lie unless the
        dropped-to-zero observation is recorded for `write_if_requested` to
        pick up -- otherwise the stale entry can only be removed by hand.
        """
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.test_x": 100}, Path(d))
            r.begin("T.test_x")
            # r._running_total stays 0: every solve raised, or the test was
            # reworked to stop solving.

            message = r.end()

            self.assertIsNotNone(message)
            self.assertIn("fewer", str(message))
            self.assertEqual(r._observed["T.test_x"], (0, []))

    @staticmethod
    def _fake_result(failures: int = 0, errors: int = 0, skipped: int = 0) -> Any:
        return type(
            "Result",
            (),
            {
                "failures": [object()] * failures,
                "errors": [object()] * errors,
                "skipped": [object()] * skipped,
            },
        )()

    @staticmethod
    def _fake_test(expected_failure_pending: bool = False) -> Any:
        """A stand-in for the ``unittest.TestCase`` ``_already_failing`` reads.

        Only ``_outcome.expectedFailure`` is read; a real ``TestCase`` is
        unnecessary for the pure-predicate tests below.
        """
        outcome = type(
            "Outcome",
            (),
            {"expectedFailure": object() if expected_failure_pending else None},
        )()
        return type("Test", (), {"_outcome": outcome})()

    def test_a_failing_test_gets_no_perf_verdict_on_top(self) -> None:
        """The wiring, not just the predicate.

        A tally wildly outside the band, on a test that has already failed,
        must produce silence -- otherwise a setUp error surfaces as two
        failures and the bogus one buries the real cause.
        """
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.test_x": 100}, Path(d))
            r.begin("T.test_x")
            r._running_total = 5000

            result = self._fake_result(errors=1)
            self.assertIsNone(perf_guard._verdict(r, 0, result, self._fake_test()))

    def test_a_passing_test_still_gets_its_verdict(self) -> None:
        """The excusal must not swallow real regressions on healthy tests."""
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.test_x": 100}, Path(d))
            r.begin("T.test_x")
            r._running_total = 5000

            message = perf_guard._verdict(r, 0, self._fake_result(), self._fake_test())

            self.assertIsNotNone(message)
            self.assertIn("more", str(message))

    def test_a_healthy_test_is_not_treated_as_already_failing(self) -> None:
        self.assertFalse(
            perf_guard._already_failing(0, self._fake_result(), self._fake_test())
        )

    def test_a_new_failure_since_baseline_is_detected(self) -> None:
        """A fault that happened during this test, not one already on the
        board before it started (the ``baseline_faults`` count excludes
        earlier tests' faults accumulated on the same shared ``result``).
        """
        test = self._fake_test()
        self.assertTrue(
            perf_guard._already_failing(0, self._fake_result(errors=1), test)
        )
        self.assertTrue(
            perf_guard._already_failing(0, self._fake_result(failures=1), test)
        )
        self.assertTrue(
            perf_guard._already_failing(0, self._fake_result(skipped=1), test)
        )

    @staticmethod
    def _fake_pytest_result(fault_count: int = 0) -> Any:
        """A stand-in for ``_pytest.unittest.TestCaseFunction``.

        That class has no ``failures``/``errors``/``skipped`` lists; every
        ``addError``/``addFailure``/``addSkip`` call instead appends to
        ``_excinfo`` (``None`` until the first fault). This mimics only the
        shape ``_fault_count`` reads.
        """
        return type(
            "PytestResult",
            (),
            {"_excinfo": None if fault_count == 0 else [object()] * fault_count},
        )()

    def test_a_pytest_unittest_bridge_result_is_counted_too(self) -> None:
        """PyRake runs its ``unittest.TestCase``\\ s under pytest, where
        ``result`` is a ``TestCaseFunction``, not a ``unittest.TestResult`` --
        it has no ``failures``/``errors``/``skipped`` lists at all, only
        ``_excinfo``. Reading the ``unittest.TestResult`` attributes
        unconditionally raises ``AttributeError`` on every fault-bearing
        ``unittest.TestCase`` PyRake runs, which is exactly what happened
        here before this was handled.
        """
        test = self._fake_test()
        self.assertFalse(
            perf_guard._already_failing(0, self._fake_pytest_result(), test)
        )
        self.assertTrue(
            perf_guard._already_failing(0, self._fake_pytest_result(1), test)
        )

    def test_a_pending_expected_failure_is_detected_too(self) -> None:
        """``result.expectedFailures`` cannot be read from a cleanup (see
        ``_expected_failure_pending``'s docstring for why), so this must be
        detected via ``test._outcome.expectedFailure`` instead -- with an
        otherwise-healthy ``result``.
        """
        self.assertTrue(
            perf_guard._already_failing(
                0, self._fake_result(), self._fake_test(expected_failure_pending=True)
            )
        )

    def test_a_fault_already_on_the_board_before_this_test_is_not_double_counted(
        self,
    ) -> None:
        """`baseline_faults` excludes an earlier test's fault on a shared result."""
        self.assertFalse(
            perf_guard._already_failing(
                1, self._fake_result(errors=1), self._fake_test()
            )
        )

    def test_a_real_failing_setup_gets_no_perf_verdict_on_top(self) -> None:
        """Drives an actual failing ``setUp`` through real ``TestCase.run``/
        ``doCleanups``, the only way to catch the bug this guards against:
        CPython's ``testPartExecutor`` resets ``_outcome.success`` to True on
        every entry, including the one wrapping the cleanup that used to read
        it, so a hand-built fake with a fixed ``.success`` attribute cannot
        exercise that reset and would stay green even if the predicate always
        returned False.
        """

        class DiesInSetup(unittest.TestCase):
            def setUp(self) -> None:
                raise RuntimeError("setup blew up")

            def test_it(self) -> None:
                pass

        key = f"{DiesInSetup.__module__}.{DiesInSetup.__name__}.test_it"
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({key: 100}, Path(d))
            wrapped_run = perf_guard._make_run(r, unittest.TestCase.run)
            result = unittest.TestResult()

            wrapped_run(DiesInSetup("test_it"), result)

            # Exactly the real setUp error -- not a second, bogus perf
            # failure piled on top of it.
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(len(result.failures), 0)
            self.assertIn("setup blew up", result.errors[0][1])

    def test_a_real_expected_failure_gets_no_perf_verdict_on_top(self) -> None:
        """Drives an actual ``@expectedFailure`` test through real
        ``TestCase.run``/``doCleanups``, the only way to catch the bug this
        guards against: a test body that raises under ``@expectedFailure``
        lands in ``result.expectedFailures``, not ``failures``/``errors``/
        ``skipped``, so a predicate that only checked that triad would still
        render a verdict here and the guard's ``AssertionError`` would land
        as a second, genuine-looking failure on a test that was supposed to
        fail.
        """

        class ExpectedToFail(unittest.TestCase):
            @unittest.expectedFailure
            def test_it(self) -> None:
                raise AssertionError("known broken")

        key = f"{ExpectedToFail.__module__}.{ExpectedToFail.__name__}.test_it"
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({key: 100}, Path(d))
            wrapped_run = perf_guard._make_run(r, unittest.TestCase.run)
            result = unittest.TestResult()

            wrapped_run(ExpectedToFail("test_it"), result)

            # Exactly the real expected failure -- not a second, bogus perf
            # failure piled on top of it.
            self.assertEqual(len(result.expectedFailures), 1)
            self.assertEqual(len(result.errors), 0)
            self.assertEqual(len(result.failures), 0)

    def test_a_real_passing_test_with_a_baseline_still_fails_through_run(
        self,
    ) -> None:
        """The excusal must not swallow a real regression driven through the
        actual wrapped ``run``, not just through ``_verdict`` directly.
        """

        class HealthySlowTest(unittest.TestCase):
            def test_it(self) -> None:
                recorder._running_total = 5000

        key = f"{HealthySlowTest.__module__}.{HealthySlowTest.__name__}.test_it"
        with tempfile.TemporaryDirectory() as d:
            recorder = self._recorder({key: 100}, Path(d))
            wrapped_run = perf_guard._make_run(recorder, unittest.TestCase.run)
            result = unittest.TestResult()

            wrapped_run(HealthySlowTest("test_it"), result)

            self.assertEqual(len(result.errors), 0)
            self.assertEqual(len(result.failures), 1)
            self.assertIn("more", result.failures[0][1])

    def test_calling_with_no_result_still_fires_start_and_stop_test_run(
        self,
    ) -> None:
        """`real_run` always sees a non-``None`` result here (the wrapper
        already substituted one), so it takes the branch that skips
        ``startTestRun``/``stopTestRun`` -- the wrapper must fire them itself
        when it is the one that built the result.
        """

        calls: list[str] = []

        class TrackingResult(unittest.TestResult):
            def startTestRun(self) -> None:
                calls.append("start")

            def stopTestRun(self) -> None:
                calls.append("stop")

        class TrackingTestCase(unittest.TestCase):
            def defaultTestResult(self) -> unittest.TestResult:
                return TrackingResult()

            def test_it(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as d:
            recorder = self._recorder({}, Path(d))
            wrapped_run = perf_guard._make_run(recorder, unittest.TestCase.run)

            wrapped_run(TrackingTestCase("test_it"))

            self.assertEqual(calls, ["start", "stop"])


class NestedRunTest(unittest.TestCase):
    """A test that drives another ``TestCase.run`` in-process (a sub-suite, a
    nested runner) re-enters ``begin`` before the outer test's own
    ``end``/``abandon`` -- the outer tally must survive that, not be
    clobbered or silently dropped.
    """

    @staticmethod
    def _recorder(baseline: dict[str, int], tmp: Path) -> perf_guard._Recorder:
        path = tmp / "baseline.json"
        path.write_text(json.dumps(baseline))
        return perf_guard._Recorder(path, tolerance=0.1, writing=False)

    def test_a_nested_begin_end_restores_the_outer_tally(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.outer": 100, "T.inner": 10}, Path(d))
            r.begin("T.outer")
            r._running_total = 40

            r.begin("T.inner")
            r._running_total = 999
            r.end()

            self.assertEqual(r._current, "T.outer")
            self.assertEqual(r._running_total, 40)

    def test_a_nested_begin_abandon_restores_the_outer_tally(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.outer": 100}, Path(d))
            r.begin("T.outer")
            r._running_total = 40

            r.begin("T.inner")
            r._running_total = 999
            r.abandon()

            self.assertEqual(r._current, "T.outer")
            self.assertEqual(r._running_total, 40)

    def test_the_outer_tally_still_renders_its_own_verdict_after_a_nested_run(
        self,
    ) -> None:
        """Without save/restore, the outer test's own ``end()`` sees ``None``
        left behind by the inner call and silently skips its verdict -- a
        real regression on the outer test would pass unnoticed.
        """
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({"T.outer": 100, "T.inner": 10}, Path(d))
            r.begin("T.outer")
            r._running_total = 40

            r.begin("T.inner")
            r._running_total = 10
            r.end()

            r._running_total += 160  # the outer test's own solves, post-nesting

            message = r.end()

            self.assertIsNotNone(message)
            self.assertIn("more", str(message))

    def test_a_real_nested_TestCase_run_does_not_double_pop_the_outer_tally(
        self,
    ) -> None:
        """Drive ``_make_run`` itself with a real nested ``TestCase.run``.

        The tests above exercise ``_Recorder.begin``/``end``/``abandon``
        directly and would stay green even if the wrapped ``run``'s own
        ``finally: recorder.abandon()`` popped an extra frame after the
        cleanup already popped one -- exactly the bug where an inner
        ``TestCase.run`` discards the outer test's tally. This one goes
        through ``_make_run`` so that regression is caught.
        """
        with tempfile.TemporaryDirectory() as d:

            def real_run(test: unittest.TestCase, result: Any = None) -> Any:
                getattr(test, test._testMethodName)()
                test.doCleanups()
                return result

            class Inner(unittest.TestCase):
                def test_it(self) -> None:
                    r._running_total = 10

            class Outer(unittest.TestCase):
                def test_it(self) -> None:
                    r._running_total = 40
                    wrapped_run(Inner("test_it"))
                    r._running_total += 160  # the outer's own solves

            # Keys are module-qualified, same as the wrapped ``run`` builds
            # them -- these classes are defined here, not imported, so there
            # is no fixed string to hardcode.
            outer_key = f"{Outer.__module__}.{Outer.__name__}.test_it"
            inner_key = f"{Inner.__module__}.{Inner.__name__}.test_it"
            r = self._recorder({outer_key: 100, inner_key: 10}, Path(d))
            wrapped_run = perf_guard._make_run(r, real_run)

            wrapped_run(Outer("test_it"))

            self.assertIsNone(r._current)
            # These fixtures set ``_running_total`` directly rather than
            # performing solves, so the observation carries no per-solve
            # detail -- `_encode_observation` renders that as `{"total": N}`,
            # never a fabricated one-element solve list (see its docstring).
            self.assertEqual(r._observed.get(outer_key), (200, []))

    def test_same_class_name_in_two_modules_does_not_collide(self) -> None:
        """The baseline key must be module-qualified, not just class + method.

        Two test classes named alike in different ``test_*.py`` files are a
        common pattern. Forging ``__module__`` on two same-named classes
        reproduces that without actually duplicating a class name across two
        real files.
        """
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder({}, Path(d))

            def real_run(test: unittest.TestCase, result: Any = None) -> Any:
                getattr(test, test._testMethodName)()
                test.doCleanups()
                return result

            class Dup(unittest.TestCase):
                def test_it(self) -> None:
                    r._running_total = 1

            DupA = type("Dup", (Dup,), {"__module__": "pkg.module_a"})
            DupB = type("Dup", (Dup,), {"__module__": "pkg.module_b"})

            wrapped_run = perf_guard._make_run(r, real_run)
            wrapped_run(DupA("test_it"))
            wrapped_run(DupB("test_it"))

            self.assertEqual(
                set(r._observed),
                {"pkg.module_a.Dup.test_it", "pkg.module_b.Dup.test_it"},
            )


class WriteIfRequestedTest(unittest.TestCase):
    """Regenerating the baseline must land on disk, not vanish silently."""

    def test_missing_out_env_writes_to_the_loaded_baseline(self) -> None:
        """Regeneration without an explicit output updates the baseline in place."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": 100}))
            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_x")
            # A total the merged write visibly changes the file to, so a
            # skipped write is not masked by the observed count happening to
            # match what is already on disk.
            r._running_total = 999
            r.end()

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(perf_guard._ENV_OUT, None)
                r.write_if_requested()

            self.assertEqual(json.loads(path.read_text()), {"T.test_x": {"total": 999}})

    def test_out_env_present_writes_there(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_x": 100}))
            out = Path(d) / "regenerated.json"
            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_x")
            r._running_total = 110
            r.end()

            with patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}):
                r.write_if_requested()

            self.assertEqual(json.loads(out.read_text()), {"T.test_x": {"total": 110}})

    def test_two_writers_to_the_same_path_both_survive(self) -> None:
        """Simulates two worker processes regenerating the same file.

        Each recorder loads the baseline once, at "process start", before
        either one writes -- exactly what happens when a parallel runner forks a process
        per test module. Without serializing the read-merge-write, the
        second writer's stale `self._baseline` snapshot clobbers the first
        writer's freshly observed entry, and the file still comes out with a
        full key count, so the loss is silent.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_a": 100, "T.test_b": 200}))
            out = Path(d) / "regenerated.json"

            r1 = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r2 = perf_guard._Recorder(path, tolerance=0.1, writing=True)

            r1.begin("T.test_a")
            r1._running_total = 105
            r1.end()

            r2.begin("T.test_b")
            r2._running_total = 210
            r2.end()

            with patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}):
                r1.write_if_requested()
                r2.write_if_requested()

            self.assertEqual(
                json.loads(out.read_text()),
                {"T.test_a": {"total": 105}, "T.test_b": {"total": 210}},
            )

    def test_a_test_missing_from_the_live_file_is_not_resurrected_from_a_stale_snapshot(
        self,
    ) -> None:
        """A process's own start-of-run snapshot must not leak into the merge.

        Simulates a test deleted from source: this recorder's own baseline
        snapshot (loaded at process start) still carries it, but another
        process's write already dropped it from the live file before this one
        runs. The stale snapshot must not bring it back.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_gone": 50, "T.test_a": 100}))
            out = Path(d) / "regenerated.json"
            out.write_text(json.dumps({"T.test_a": 100}))

            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_a")
            r._running_total = 105
            r.end()

            with patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}):
                r.write_if_requested()

            self.assertEqual(json.loads(out.read_text()), {"T.test_a": {"total": 105}})

    def test_a_malformed_live_file_raises_instead_of_crashing_in_atexit(
        self,
    ) -> None:
        """A truncated write or hand-edit can leave `out_path` as a JSON

        array, `null`, or a scalar rather than an object. Left unchecked,
        `{**current, **observed}` raises a bare `TypeError` from inside this
        atexit handler; this asserts the same loud, diagnosable `ValueError`
        the `__init__` load path already raises for the same shape of
        corruption.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({"T.test_a": 100}))
            out = Path(d) / "regenerated.json"
            out.write_text(json.dumps(["T.test_a", 100]))

            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_a")
            r._running_total = 105
            r.end()

            with (
                patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}),
                self.assertRaises(ValueError),
            ):
                r.write_if_requested()

    def test_a_recorder_with_no_baseline_does_not_wipe_the_live_ignore_list(
        self,
    ) -> None:
        """A recorder that never resolved a source baseline has no opinion.

        Simulates a process whose `baseline_path` did not resolve: `writing`
        is True but the path does not exist, so `self._ignored` comes out
        empty for a reason unrelated to what the source baseline actually
        says. That recorder must not then blank out an `__ignore__` entry
        another (correctly-loaded) process already wrote to the shared
        output file -- doing so would permanently re-arm a test that was
        deliberately exempted.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "does_not_exist.json"
            out = Path(d) / "regenerated.json"
            out.write_text(
                json.dumps({"T.test_a": 100, perf_guard._IGNORE_KEY: ["T.flaky"]})
            )

            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_a")
            r._running_total = 105
            r.end()

            with patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}):
                r.write_if_requested()

            self.assertEqual(
                json.loads(out.read_text()),
                {"T.test_a": {"total": 105}, perf_guard._IGNORE_KEY: ["T.flaky"]},
            )

    def test_a_recorder_with_no_baseline_preserves_an_existing_tolerance_override(
        self,
    ) -> None:
        """A recorder that never resolved a source baseline has no opinion on
        any test's tolerance either -- the same blind spot as the
        `__ignore__` case above. It must not flatten an existing
        `{"expected": N, "tolerance": T}` entry to a bare int just because
        its own (unloaded) `_per_test_tolerance` is empty.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "does_not_exist.json"
            out = Path(d) / "regenerated.json"
            out.write_text(
                json.dumps({"T.test_a": {"expected": 100, "tolerance": 0.5}})
            )

            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_a")
            r._running_total = 105
            r.end()

            with patch.dict(os.environ, {perf_guard._ENV_OUT: str(out)}):
                r.write_if_requested()

            self.assertEqual(
                json.loads(out.read_text()),
                {"T.test_a": {"total": 105, "tolerance": 0.5}},
            )


class IterationsTest(unittest.TestCase):
    """`_iterations` must not silently contribute 0 for an undiagnosed solve."""

    def test_diagnostics_present_uses_the_diagnostics_count(self) -> None:
        result = _result(
            n_iterations=99,
            diagnostics=PrimalDualDiagnostics(surrogate_gaps=(1.0, 2.0, 3.0)),
        )
        self.assertEqual(perf_guard._iterations(result), 3)

    def test_diagnostics_absent_falls_back_to_n_iterations(self) -> None:
        """A solver that attaches no diagnostics must still be counted.

        Before this fix, a `None` diagnostics contributed 0 regardless of how
        much work the solve actually did -- if every solve in a test took this
        path, the guard's total stayed 0, no baseline entry was recorded, and
        `end()` had nothing to compare. The guard would look armed while
        silently watching nothing.
        """
        result = _result(n_iterations=7, diagnostics=None)
        self.assertEqual(perf_guard._iterations(result), 7)

    def test_phase1_falls_back_independently_of_the_top_level(self) -> None:
        """Each node in the phase1 chain is judged on its own diagnostics."""
        phase1 = _result(n_iterations=5, diagnostics=None)
        top = _result(
            n_iterations=99,
            diagnostics=PrimalDualDiagnostics(surrogate_gaps=(1.0, 2.0)),
            phase1=phase1,
        )
        self.assertEqual(perf_guard._iterations(top), 2 + 5)

    def test_a_present_but_uninformative_diagnostics_still_falls_back(self) -> None:
        """A base ``SolverDiagnostics()`` reports 0 ("this solver does not
        say"), not "no work happened" -- the fallback must trigger whenever
        the diagnostics report 0, not only when ``diagnostics is None``, or a
        solve carrying this base instance silently contributes nothing.
        """
        result = _result(n_iterations=11, diagnostics=SolverDiagnostics())
        self.assertEqual(perf_guard._iterations(result), 11)


class PerSolveListTest(unittest.TestCase):
    r"""Recording each solve, not just the per-test total.

    The gap this closes: a total can sit comfortably inside its band while one
    solve underneath it has stopped converging. Four PyRake entries read
    ``1071`` -- ``1000`` from a solve burning its entire iteration budget plus
    ``71`` of healthy work -- and the guard was green on every one.
    """

    @staticmethod
    def _recorder(baseline: dict[str, Any], tmp: Path) -> perf_guard._Recorder:
        path = tmp / "baseline.json"
        path.write_text(json.dumps(baseline))
        return perf_guard._Recorder(path, tolerance=0.1, writing=False)

    def _check(self, baseline: dict[str, Any], solves: list[int]) -> str | None:
        with tempfile.TemporaryDirectory() as d:
            r = self._recorder(baseline, Path(d))
            r.begin("T.test_x")
            r._solves = list(solves)
            r._running_total = sum(solves)
            return r.end()

    def test_a_single_bad_solve_is_caught_behind_a_healthy_total(self) -> None:
        """The motivating case, in miniature.

        Baseline: 20 solves of ~50. Observed: one of them runs away to 500
        while the rest shrink to compensate, so the TOTAL lands inside its
        band. Only the worst-solve check can see this.
        """
        baseline = {"T.test_x": [50] * 20}  # total 1000, max 50
        solves = [500] + [26] * 19  # total 994 -- within 10% of 1000

        message = self._check(baseline, solves)

        self.assertIsNotNone(message)
        self.assertIn("worst single solve moved", str(message))
        self.assertIn("500", str(message))
        # And it says the total was fine, so nobody hunts for a total regression.
        self.assertIn("still inside its band", str(message))

    def test_a_healthy_spread_passes(self) -> None:
        baseline = {"T.test_x": [50, 48, 52, 47]}

        self.assertIsNone(self._check(baseline, [51, 49, 50, 48]))

    def test_the_total_check_still_fires_on_its_own(self) -> None:
        """The pre-existing behaviour must survive the format change."""
        baseline = {"T.test_x": [50] * 20}

        message = self._check(baseline, [90] * 20)

        self.assertIsNotNone(message)
        self.assertIn("solver work moved", str(message))

    def test_a_legacy_scalar_baseline_guards_the_total_only(self) -> None:
        """Old entries keep working; they simply cannot speak to worst-solve."""
        self.assertIsNone(self._check({"T.test_x": 1000}, [500] + [26] * 19))
        self.assertIsNotNone(self._check({"T.test_x": 1000}, [90] * 20))

    def test_a_tiny_worst_solve_is_not_guarded(self) -> None:
        """A relative band around 4 iterations is meaningless, as for totals."""
        baseline = {"T.test_x": [4] * 30}  # total 120, max 4

        self.assertIsNone(self._check(baseline, [7] * 17 + [1] * 13))


class EncodeObservationTest(unittest.TestCase):
    """Short tests keep a readable list; long ones collapse to a summary."""

    def test_a_short_run_keeps_every_solve(self) -> None:
        self.assertEqual(
            perf_guard._encode_observation(34, [9, 21, 4], None), [9, 21, 4]
        )

    def test_a_long_run_becomes_a_summary(self) -> None:
        solves = list(range(1, 101))  # 100 solves, well over the threshold

        encoded = perf_guard._encode_observation(sum(solves), solves, None)

        self.assertEqual(encoded, {"solves": 100, "total": 5050, "max": 100})

    def test_the_threshold_is_inclusive(self) -> None:
        """Exactly at the limit still lists; one more summarises.

        ``assertIsInstance`` rather than a bare ``assert``: the latter is
        stripped under ``python -O``, which would make this pass vacuously.
        """
        at = [1] * perf_guard._MAX_PER_SOLVE_LIST
        over = [1] * (perf_guard._MAX_PER_SOLVE_LIST + 1)

        self.assertIsInstance(perf_guard._encode_observation(20, at, None), list)
        self.assertIsInstance(perf_guard._encode_observation(21, over, None), dict)

    def test_a_tolerance_override_survives_both_forms(self) -> None:
        """Flattening it away would silently drop the override on regeneration."""
        short = perf_guard._encode_observation(13, [9, 4], 0.13)
        long = perf_guard._encode_observation(50, [1] * 50, 0.13)

        self.assertEqual(short, {"per_solve": [9, 4], "tolerance": 0.13})
        self.assertEqual(cast("dict[str, Any]", long)["tolerance"], 0.13)

    def test_a_total_with_no_per_solve_detail_does_not_fabricate_a_max(self) -> None:
        """``[N]`` would read back as "one solve of N", inventing a worst-solve.

        A later run carrying real per-solve detail would then measure its
        genuine max against that inflated figure, pass comfortably, and
        reinstate the very masking this format removes.
        """
        encoded = perf_guard._encode_observation(500, [], None)

        self.assertEqual(encoded, {"total": 500})
        self.assertIsNone(perf_guard._parse_baseline_entry(encoded).max_solve)

    def test_a_summary_missing_max_is_rejected(self) -> None:
        """Silently skipping the worst-solve check is the whole failure mode."""
        with self.assertRaisesRegex(ValueError, "no 'max'"):
            perf_guard._parse_baseline_entry({"solves": 30, "total": 1000})

    def test_every_written_form_reads_back_identically(self) -> None:
        """Round-trip: what regeneration writes, the next run must parse."""
        for solves, tol in (([9, 21, 4], None), ([9, 4], 0.13), ([7] * 50, None)):
            with self.subTest(n=len(solves), tol=tol):
                encoded = perf_guard._encode_observation(sum(solves), solves, tol)
                parsed = perf_guard._parse_baseline_entry(
                    json.loads(json.dumps(encoded))
                )

                self.assertEqual(parsed.total, sum(solves))
                self.assertEqual(parsed.max_solve, max(solves))
                self.assertEqual(parsed.solves, len(solves))
                self.assertEqual(parsed.tolerance, tol)


class PerSolveThroughTheRealPathTest(unittest.TestCase):
    """Record per-solve counts through the instrumented path, not by hand.

    The scenarios above set ``_solves`` directly, so they would stay green if
    the instrumentation that populates it regressed. This drives real
    ``enter``/``leave`` calls with real ``SolveResult`` objects, so the wiring
    from a solve to a recorded number is itself under test.
    """

    def test_each_solve_is_recorded_separately(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({}))
            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_x")
            for n in (47, 9, 21):
                r.enter()
                r.leave(_result(n))
            r.end()

            self.assertEqual(r._observed["T.test_x"], (77, [47, 9, 21]))

    def test_a_nested_solve_counts_once_at_the_outer_call(self) -> None:
        """``Problem.solve`` delegates to a solver's ``solve``; both are wrapped.

        One logical solve must contribute one entry, not two, or every list
        would double-count and the worst-solve figure would be meaningless.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            path.write_text(json.dumps({}))
            r = perf_guard._Recorder(path, tolerance=0.1, writing=True)
            r.begin("T.test_x")
            r.enter()  # Problem.solve
            r.enter()  # the solver's solve, underneath it
            r.leave(_result(30))
            r.leave(_result(30))
            r.end()

            self.assertEqual(r._observed["T.test_x"], (30, [30]))


class BaselineResolutionTest(unittest.TestCase):
    """The baseline a test target names is actually found."""

    def test_the_suites_own_baseline_resolves_from_any_directory(self) -> None:
        """A path that fails to resolve would disable the guard without a word.

        Run from an unrelated directory, so the path cannot be found as given and the
        module-location fallback is what resolves it.

        """
        path = os.environ.get("CVXIUM_PERF_BASELINE")
        if path is None:
            self.skipTest("CVXIUM_PERF_BASELINE is not set for this run.")
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            self.assertTrue(perf_guard._resolve_baseline_path(path).is_file())
