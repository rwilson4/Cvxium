"""Fail a test whose solver work moves materially against a recorded baseline.

Correctness tests do not notice a solver that still converges but now takes half again
as many iterations; a Phase I step-length regression once went unnoticed for months with
every suite green. This module records the iterations each test's solves take and
compares them against a checked-in baseline.

A drop fails too, not only a rise. A test that suddenly does much less work is usually
good news, but it is also what a silently loosened tolerance, or a fixture that stopped
exercising the solver, looks like, and neither should land unremarked. The fix for
either direction is the same: regenerate the baseline and say why in the commit message.

The guard is activated by the environment, not by a test module. A test runner may split
the suite across processes (pytest-xdist does), and installing from an arbitrary test
module would reach some processes and not others, so the guard would pass vacuously in
exactly the runs that skip that module. Instead the suite sets
   CVXIUM_PERF_BASELINE=<path to the baseline>
(``test/conftest.py`` does so by default) and Cvxium's ``__init__`` calls
``install_from_environment``, which installs when it sees the variable. Every process
that imports Cvxium, which is every process running one of these tests, gets the guard.

To regenerate, run the suite with
   CVXIUM_PERF_BASELINE_WRITE=1
The run then records instead of comparing, and writes to the baseline it loaded, or to
``CVXIUM_PERF_BASELINE_OUT`` when that is set.

"""

import atexit
import contextlib
import fcntl
import json
import os
import sys
import traceback
import unittest
from pathlib import Path
from typing import Any, NamedTuple

from ..problems.result import SolveResult

# Counts below this are dominated by a single iteration's worth of noise, and a
# relative band around them is meaningless: 2 -> 3 iterations is +50%. Guard
# only what is big enough for the ratio to mean something.
_MIN_ITERATIONS: int = 8

# Absolute slack, applied alongside the relative band. Iteration counts are
# reproducible on one machine but not always across machines: a solve that
# finishes just inside a tolerance can take an iteration more or fewer
# depending on floating-point details, and CI is not a developer's laptop.
# Without this a baseline of 14 admits only 13-15, which one iteration of drift
# breaks. A real regression clears both bounds comfortably -- the half-step bug
# moved one test from 9 to 16.
_ABS_SLACK: int = 3

# Tests whose counts are not reproducible across machines, listed in the
# baseline under this key. Reserve it for solves that terminate on a tolerance
# boundary, where the count is genuinely bimodal rather than merely noisy.
_IGNORE_KEY: str = "__ignore__"

# Above this many solves in one test, the baseline records a summary rather
# than the individual counts. Chosen from the measured shape of these suites:
# the median test performs 2-5 solves and ~88% sit under this threshold, so
# almost everything keeps a readable per-solve list, while the handful of
# frontier sweeps (one performs 6,629 solves) collapse to three numbers
# instead of adding thousands of integers to a file people have to review.
_MAX_PER_SOLVE_LIST: int = 20

# Set by a test target to arm the guard and name its baseline. Unset in
# production, where this module is never imported.
_ENV_BASELINE: str = "CVXIUM_PERF_BASELINE"

_ENV_WRITE: str = "CVXIUM_PERF_BASELINE_WRITE"

# Where a regeneration run writes, when somewhere other than the baseline it
# loaded (to compare a fresh recording against the checked-in one, say).
_ENV_OUT: str = "CVXIUM_PERF_BASELINE_OUT"

# Appended to every failure message, so the remedy travels with the complaint
# and nobody has to go hunting for the recipe.
_REGENERATE_HINT: str = (
    f" If this is intended, regenerate with {_ENV_WRITE}=1 and say why in "
    f"the commit message. If the count is not reproducible across machines, add the test "
    f'to "{_IGNORE_KEY}" in the baseline instead.'
)

# Sentinel guarding `install()` against a second, redundant install. A
# module-level global rather than an attribute hung off the function object,
# so no attribute-access suppression is needed to satisfy the type checker.
_installed: bool = False

# Classes already patched by `_wrap_solve`, so a class visited twice (e.g. by
# two modules that both install) is not wrapped twice. A module-level set
# rather than a marker attribute on the wrapper function, for the same reason
# as `_installed` above.
_wrapped_classes: set[type] = set()


class _Expected(NamedTuple):
    """Baseline expectation for one test.

    ``max_solve`` and ``solves`` are ``None`` for the scalar forms, which record only a
    per-test total and say nothing about the individual solves underneath it.

    The scalar form is also the deliberate opt-out from the worst-solve check, for a
    test whose per-solve split differs between machines while its total does not (for
    example, a long frontier sweep). Recording per-solve counts on one machine would arm a
    check against numbers CI does not reproduce. The ignore list (``__ignore__``) is a
    worse remedy, since ``_Recorder.end`` returns early for an ignored test and so drops
    the total check as well. The scalar form keeps the check that works and declines the
    one that does not; do not convert such an entry to the per-solve form without a CI
    measurement of the per-solve split.

    Fields
    ------
     total : int
        Expected total iterations across the test's solves.
     max_solve : int | None
        Expected iterations of the test's worst single solve, or ``None`` when the entry
        carries no per-solve detail.
     solves : int | None
        Expected number of solves, or ``None`` when the entry does not record it.
     tolerance : float | None
        Per-test relative band overriding the suite default, or ``None`` for the
        default.

    """

    total: int
    max_solve: int | None
    solves: int | None
    tolerance: float | None


def _parse_baseline_entry(value: Any) -> _Expected:
    """Parse one baseline entry, in any of the forms the file may hold.

    The forms are:
    - ``[47, 21, 9]``: the per-solve list, the default for a test with at most
      ``_MAX_PER_SOLVE_LIST`` solves (most tests perform only a few). The total, worst
      solve and solve count all derive from it, and the individual counts are visible
      in review. That is the point: a per-test total can hide one solve that burned its
      whole iteration budget behind the healthy ones beside it.
    - ``{"solves": N, "total": T, "max": M}``: the summary, for tests above that
      threshold, such as a frontier sweep of thousands of solves, whose full list would
      make every regeneration an unreviewable diff.
    - ``{"total": T}``: a total with no per-solve detail (see ``_encode_observation``).
    - ``{"expected": N, "tolerance": T}``: a scalar with a per-test band override.
    - ``12`` (or ``"12"``): a bare scalar.
    Any object form may carry ``"tolerance"``, and the list form may be wrapped as
    ``{"per_solve": [...], "tolerance": T}`` when it needs one. The caller wraps any
    error raised here with the offending test's name, since a bare error gives no clue
    which of several hundred entries is at fault.

    Parameters
    ----------
     value : Any
        The entry as loaded from JSON.

    Returns
    -------
     expected : _Expected
        The parsed expectation.

    Raises
    ------
     ValueError
        When the tolerance lies outside (0, 1), a summary has ``"solves"`` but no
        ``"max"``, ``"per_solve"`` is not a list, or a count is not integral or not
        numeric.
     KeyError
        When an object entry has none of ``"per_solve"``, ``"total"`` or ``"expected"``.
     TypeError
        When a count or tolerance has a type ``float`` cannot convert.

    """
    if isinstance(value, list):
        return _from_per_solve(value, None)
    if isinstance(value, dict):
        raw_tolerance = value.get("tolerance")
        tolerance = float(raw_tolerance) if raw_tolerance is not None else None
        if tolerance is not None and not 0 < tolerance < 1:
            # A typo'd `10` (meant `0.10`) or a negative value would otherwise
            # silently widen or invert the band instead of failing loudly like
            # every other malformed entry here. A band this wide belongs in
            # `_IGNORE_KEY` instead, which says so rather than pretending to
            # guard a test it cannot meaningfully bound.
            raise ValueError(
                f"tolerance override must be between 0 and 1 (exclusive), got "
                f"{tolerance!r}; a band this wide belongs in {_IGNORE_KEY!r} "
                "instead"
            )
        if "per_solve" in value:
            return _from_per_solve(value["per_solve"], tolerance)
        if "total" in value:
            # A summary -- anything claiming a solve count -- must carry
            # ``max``. Treating it as optional means a truncated or
            # hand-edited ``{"solves": 30, "total": 1000}`` loads happily,
            # silently skips the worst-solve comparison, and reinstates
            # exactly the total-only blind spot this format exists to close.
            # Fail loudly instead. ``{"total": N}`` on its own is the
            # deliberate max-less form (see ``_encode_observation``), for an
            # observation that carries no per-solve detail to summarise.
            if "solves" in value and "max" not in value:
                raise ValueError(
                    "summary entry has 'solves' but no 'max'; the worst-solve "
                    "check would be silently skipped. Regenerate the baseline, "
                    "or write {'total': N} if there is genuinely no per-solve "
                    "detail."
                )
            return _Expected(
                total=_strict_int(value["total"]),
                max_solve=_strict_int(value["max"]) if "max" in value else None,
                solves=_strict_int(value["solves"]) if "solves" in value else None,
                tolerance=tolerance,
            )
        return _Expected(_strict_int(value["expected"]), None, None, tolerance)
    return _Expected(_strict_int(value), None, None, None)


def _from_per_solve(raw: Any, tolerance: float | None) -> _Expected:
    """Derive the comparison figures from a per-solve list.

    Parameters
    ----------
     raw : Any
        The per-solve iteration counts, as loaded from JSON.
     tolerance : float | None
        Per-test band override to carry along, or ``None``.

    Returns
    -------
     expected : _Expected
        The total, worst solve (``None`` for an empty list) and solve count of ``raw``.

    Raises
    ------
     ValueError
        When ``raw`` is not a list, or, propagated from ``_strict_int``, when a count is
        not integral.
     TypeError
        Propagated from ``_strict_int`` when a count is not numeric.

    """
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of per-solve counts, got {raw!r}")
    solves = [_strict_int(v) for v in raw]
    return _Expected(
        total=sum(solves),
        max_solve=max(solves) if solves else None,
        solves=len(solves),
        tolerance=tolerance,
    )


def _encode_observation(total: int, solves: list[int], tolerance: float | None) -> Any:
    """Encode an observed test in the most informative form that stays readable.

    Writes a per-solve list for at most ``_MAX_PER_SOLVE_LIST`` solves, a summary above
    that, and either one wrapped in an object when a tolerance override must be kept.

    An empty ``solves`` means the total was recorded with no per-solve detail behind it.
    That is written as ``{"total": N}``, never as ``[N]``: a one-element list reads back
    as one solve of N iterations, fabricating a worst-solve figure equal to the whole
    total. A later run with real per-solve detail would then compare its true maximum
    against that inflated figure and pass comfortably, masking exactly the single-solve
    regressions this format exists to catch.

    Parameters
    ----------
     total : int
        Total iterations observed for the test.
     solves : list[int]
        Iterations of each solve, in order; empty when there is no per-solve detail.
     tolerance : float | None
        Per-test band override to preserve, or ``None``.

    Returns
    -------
     entry : Any
        A JSON-serializable list or dict in one of the forms ``_parse_baseline_entry``
        reads.

    """
    encoded: dict[str, Any]
    if not solves:
        encoded = {"total": total}
    elif len(solves) <= _MAX_PER_SOLVE_LIST:
        if tolerance is None:
            return solves
        return {"per_solve": solves, "tolerance": tolerance}
    else:
        encoded = {
            "solves": len(solves),
            "total": sum(solves),
            "max": max(solves),
        }
    if tolerance is not None:
        encoded["tolerance"] = tolerance
    return encoded


def _strict_int(value: Any) -> int:
    """Convert to int, rejecting a non-integral number instead of truncating it.

    A malformed baseline value such as ``15.9`` must fail loudly at load time, not
    silently become 15 and shift the comparison band by nearly a full iteration.

    Parameters
    ----------
     value : Any
        A number or numeric string.

    Returns
    -------
     n : int
        The integral value.

    Raises
    ------
     ValueError
        When ``value`` is not integral, is nan, or is a non-numeric string.
     TypeError
        When ``value`` has a type ``float`` cannot convert.
     OverflowError
        When ``value`` is infinite.

    """
    numeric = float(value)
    if numeric != int(numeric):
        raise ValueError(f"expected an integral iteration count, got {value!r}")
    return int(numeric)


def _iterations(result: SolveResult) -> int:
    """Count the total solver iterations in a result, Phase I sub-solves included.

    Phase I hangs off ``SolveResult.phase1`` rather than appearing in the top-level
    diagnostics, so the count walks that chain; a guard that skipped Phase I would miss
    regressions there.

    Each node contributes its diagnostics' ``iteration_count()``, falling back to
    ``n_iterations`` (a field every ``SolveResult`` has) whenever that count is 0. The
    diagnostics can be absent, and a base ``SolverDiagnostics`` reports 0 to mean "this
    solver does not say", not "no work happened"; counting either as 0 would disarm the
    guard for any test whose solves all take that path.

    Parameters
    ----------
     result : SolveResult
        The result of one solve.

    Returns
    -------
     n : int
        Iterations summed over ``result`` and its nested Phase I results.

    """
    total = 0
    node: SolveResult | None = result
    while node is not None:
        diagnostics = node.diagnostics
        count = diagnostics.iteration_count() if diagnostics is not None else 0
        if count == 0:
            count = node.n_iterations
        total += count
        node = node.phase1
    return total


def _fault_count(result: Any) -> int:
    """Count the failures, errors and skips recorded on a test result so far.

    The count is populated synchronously as each fault happens and never reset, so
    comparing it with a count captured before the test ran is stable across a cleanup's
    own ``testPartExecutor`` scope. Expected failures and unexpected successes are
    deliberately excluded; see ``_expected_failure_pending`` for why they cannot be read
    this way.

    ``result`` is not always a ``unittest.TestResult``. Pytest's unittest bridge
    (``_pytest.unittest.TestCaseFunction``, used when a ``unittest.TestCase`` is
    collected under a pytest target) implements the same ``addError``, ``addFailure``
    and ``addSkip`` callbacks, but has no ``failures``, ``errors`` or ``skipped`` lists;
    each call appends to ``_excinfo`` instead, which is ``None`` until the first fault.
    ``_excinfo`` is a class attribute present only on that shape, so ``hasattr`` tells
    the two apart, where a ``getattr`` default could not tell "pytest, no fault yet"
    from "no such attribute".

    Parameters
    ----------
     result : Any
        A ``unittest.TestResult`` or pytest's ``TestCaseFunction``.

    Returns
    -------
     n : int
        Number of faults recorded so far.

    """
    if hasattr(result, "_excinfo"):
        excinfo = result._excinfo
        return 0 if excinfo is None else len(excinfo)
    return len(result.failures) + len(result.errors) + len(result.skipped)


def _expected_failure_pending(test: unittest.TestCase) -> bool:
    """Check whether a test's body just raised under an active ``@expectedFailure``.

    Such a raise is captured into ``test._outcome.expectedFailure`` (an ``exc_info``
    tuple) while the test method's ``testPartExecutor`` scope is active, but
    ``TestCase.run`` moves it into ``result.expectedFailures`` only after ``doCleanups``
    returns, and only if ``outcome.success`` is still True then. A cleanup that raises,
    such as this guard's own ``AssertionError``, sets ``outcome.success`` to False
    first, which suppresses that bookkeeping entirely. So ``result.expectedFailures``
    cannot be compared with a baseline count from inside a cleanup.
    ``_outcome.expectedFailure`` is the one place the state survives long enough for a
    cleanup to read it, because, unlike ``success``, nothing resets it between the test
    body finishing and the cleanups running.

    Parameters
    ----------
     test : unittest.TestCase
        The test whose outcome is inspected.

    Returns
    -------
     pending : bool
        True when the body raised under ``@expectedFailure``.

    """
    outcome = getattr(test, "_outcome", None)
    return outcome is not None and outcome.expectedFailure is not None


def _already_failing(
    baseline_faults: int, result: Any, test: unittest.TestCase
) -> bool:
    """Check whether a test gained a new fault since a recorded count.

    Cleanups also run when ``setUp`` raised or the test body failed, and then the guard
    has nothing useful to say: a test that died before the solves it would have measured
    is not a performance regression, and raising anyway piles a misleading second
    failure on the real one. The same holds for a test that raised under
    ``@expectedFailure``, where an extra failure would also suppress the
    expected-failure bookkeeping (see ``_expected_failure_pending``).

    unittest's private ``_outcome.success`` cannot answer this from a cleanup:
    ``doCleanups`` runs each cleanup inside ``outcome.testPartExecutor``, which resets
    ``success`` to True on entry, so the flag always reads True by the time a cleanup
    sees it. ``_fault_count`` is stable across that scope, so comparing it with a count
    taken before the test ran works where ``success`` cannot.

    Parameters
    ----------
     baseline_faults : int
        The ``_fault_count`` of ``result`` captured before the test ran.
     result : Any
        The test result object, as accepted by ``_fault_count``.
     test : unittest.TestCase
        The running test.

    Returns
    -------
     failing : bool
        True when a fault was recorded during the test, or its body raised under
        ``@expectedFailure``.

    """
    return _fault_count(result) > baseline_faults or _expected_failure_pending(test)


def _verdict(
    recorder: "_Recorder", baseline_faults: int, result: Any, test: unittest.TestCase
) -> str | None:
    """Return the guard's verdict for the current test, or None to stay silent.

    An already-failing test is excused: its tally is abandoned and no verdict is given.
    Otherwise the recorder ends the test and compares it with the baseline. This is a
    module-level function rather than part of ``install``'s closure so that tests can
    reach the decision directly.

    Parameters
    ----------
     recorder : _Recorder
        The recorder holding the current test's tally.
     baseline_faults : int
        The ``_fault_count`` of ``result`` captured before the test ran.
     result : Any
        The test result object.
     test : unittest.TestCase
        The running test.

    Returns
    -------
     message : str | None
        A failure message, or ``None`` when the test passes the guard or is excused.

    """
    if _already_failing(baseline_faults, result, test):
        recorder.abandon()
        return None
    return recorder.end()


class _Recorder:
    """Recorder of per-test iteration counts, checked against a baseline on test exit.

    Loads the baseline at construction, accumulates the iterations of each test's
    outermost solves between ``begin`` and ``end``, and on ``end`` compares the test's
    total and worst solve with the baseline. In write mode it records instead, and
    ``write_if_requested`` merges the observations into the output file at exit.

    Parameters
    ----------
     baseline_path : Path
        The baseline JSON file. When it is missing outside write mode, a warning is
        printed and every test is treated as new. A file that is not a JSON object, an
        ``__ignore__`` entry that is not a list of names, or a malformed entry makes the
        constructor raise ``ValueError``.
     tolerance : float
        Default relative band, applied in both directions alongside the absolute slack
        ``_ABS_SLACK``.
     writing : bool | None
        Whether to record rather than compare; ``None`` reads the
        ``CVXIUM_PERF_BASELINE_WRITE`` environment variable.

    """

    def __init__(
        self, baseline_path: Path, tolerance: float, writing: bool | None = None
    ) -> None:
        self._path = baseline_path
        self._tolerance = tolerance
        # Explicit beats the environment, so the guard's own tests are not at
        # the mercy of the mode the suite happens to be running in.
        self._writing: bool = (
            os.environ.get(_ENV_WRITE) == "1" if writing is None else writing
        )
        self._baseline: dict[str, _Expected] = {}
        self._per_test_tolerance: dict[str, float] = {}
        self._ignored: frozenset[str] = frozenset()
        # Distinguishes "the source baseline genuinely has no ignores" from
        # "this process never resolved a source baseline at all" -- the two
        # look identical via `self._ignored` alone, but only the former is a
        # trustworthy answer for what `write_if_requested` should write back.
        self._loaded_baseline: bool = baseline_path.exists()
        if baseline_path.exists():
            raw = json.loads(baseline_path.read_text())
            if not isinstance(raw, dict):
                # A truncated write or hand-edit can leave the file as `null`,
                # a list, or a scalar. Left unchecked, the `.pop`/`.items()`
                # below raise a bare AttributeError that propagates through
                # `install_from_environment` and crashes the import of Cvxium
                # for every test process in the run.
                raise ValueError(
                    f"malformed baseline at {baseline_path}: expected a JSON "
                    f"object mapping test name to iteration count, got "
                    f"{type(raw).__name__}"
                )
            ignore_entry = raw.pop(_IGNORE_KEY, [])
            if not isinstance(ignore_entry, list) or not all(
                isinstance(name, str) for name in ignore_entry
            ):
                # A hand-edit that writes a bare string silently degrades to a
                # frozenset of characters; a non-iterable value raises a bare
                # TypeError from frozenset(...) that escapes
                # install_from_environment and crashes the import of Cvxium
                # for every test process in the run. Fail loudly instead,
                # matching every other malformed-entry check above.
                raise ValueError(
                    f"malformed {_IGNORE_KEY!r} entry in baseline at "
                    f"{baseline_path}: expected a list of test names, got "
                    f"{ignore_entry!r}"
                )
            self._ignored = frozenset(ignore_entry)
            for name, entry in raw.items():
                try:
                    parsed = _parse_baseline_entry(entry)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"malformed baseline entry for {name!r}: {entry!r} ({exc})"
                    ) from exc
                self._baseline[name] = parsed
                if parsed.tolerance is not None:
                    self._per_test_tolerance[name] = parsed.tolerance
        elif not self._writing:
            # Every comparison below reads as "no baseline", which end()
            # cannot distinguish from a legitimately new test -- so a path
            # that resolved wrong (layout change, missing resource) makes
            # the whole suite pass vacuously instead of failing loudly.
            print(
                f"[perf_guard] no baseline at {baseline_path}; every test "
                "in this run will be treated as new rather than guarded. "
                "If that is unexpected, check the target's resource layout.",
                file=sys.stderr,
            )
        self._observed: dict[str, tuple[int, list[int]]] = {}
        self._current: str | None = None
        self._running_total: int = 0
        # ``Problem.solve`` delegates to a solver's ``solve``, and both are
        # wrapped. Counting only the outermost call keeps one logical solve
        # worth one contribution instead of two.
        self._depth: int = 0
        # A test that drives ``TestCase.run`` on another instance in-process
        # (a sub-suite, a nested runner) re-enters ``begin`` before the outer
        # test's ``end``/``abandon``. Without saving the outer tally here, the
        # inner call's ``begin`` clobbers it and the inner call's
        # ``end``/``abandon`` clears it, leaving the outer test unguarded.
        self._stack: list[tuple[str | None, int, int, list[int]]] = []
        # Iterations for each solve this test performed, in order. The total is
        # kept alongside rather than derived, because a caller can set it
        # directly (the guard's own tests do) without any per-solve detail.
        self._solves: list[int] = []

    def enter(self) -> None:
        """Note entry into a wrapped ``solve`` call.

        Increments the nesting depth, so only the outermost of nested solves (for
        example ``Problem.solve`` delegating to a solver's ``solve``) is counted.

        """
        self._depth += 1

    def leave(self, result: SolveResult) -> None:
        """Note a wrapped ``solve`` call returning a result, counting it if outermost.

        Parameters
        ----------
         result : SolveResult
            The result the call returned; its iterations, Phase I included, are added to
            the current test's tally when the call was outermost and a test is running.

        """
        self._depth -= 1
        if self._depth == 0 and self._current is not None:
            count = _iterations(result)
            self._running_total += count
            self._solves.append(count)

    def begin(self, name: str) -> None:
        """Start the tally for a test.

        Saves the enclosing tally on a stack first, so a test that runs another
        ``TestCase.run`` in-process does not lose the outer test's tally.

        Parameters
        ----------
         name : str
            The test's baseline key, ``module.ClassName.test_method``.

        """
        self._stack.append(
            (self._current, self._running_total, self._depth, self._solves)
        )
        self._current = name
        self._running_total = 0
        self._depth = 0
        self._solves = []

    def unwind(self) -> None:
        """Leave a wrapped ``solve`` call that produced no countable result.

        Decrements the nesting depth without counting anything; used when the call
        raised or returned something other than a ``SolveResult``.

        """
        self._depth -= 1

    def abandon(self) -> None:
        """Drop the current test's tally without rendering a verdict.

        For tests the guard has nothing useful to say about: a skip, or a failure before
        the solves it would have measured. Restores the enclosing tally (or a clean
        slate), so nothing left open is charged to the next test.

        """
        if self._stack:
            self._current, self._running_total, self._depth, self._solves = (
                self._stack.pop()
            )
        else:
            self._current = None
            self._running_total = 0
            self._depth = 0
            self._solves = []

    def end(self) -> str | None:
        """Finish the current test and compare it with the baseline.

        Restores the enclosing tally, records the observation for a later regeneration,
        and compares the total and worst solve with the baseline (see ``_compare``). It
        returns no message for an ignored test, a test with no baseline entry (a new
        test is not a regression), a baseline total below ``_MIN_ITERATIONS`` without a
        per-test override, or a run in write mode.

        The observation is recorded before the floor check, so a sub-floor baseline that
        has grown past the floor still updates on regeneration. A brand-new test with
        zero iterations is not recorded, so regeneration does not add an entry for every
        test that performs no solves; a test with an existing entry is recorded even at
        zero, so that regeneration clears its "fewer iterations" failure.

        Returns
        -------
         message : str | None
            A failure message, or ``None`` when the test passes or is not checked.

        """
        name, total, solves = self._current, self._running_total, self._solves
        if self._stack:
            self._current, self._running_total, self._depth, self._solves = (
                self._stack.pop()
            )
        else:
            self._current = None
            self._running_total = 0
            self._depth = 0
            self._solves = []
        if name is None or name in self._ignored:
            return None

        expected = self._baseline.get(name)
        # Recorded before the floor check below, not after: a sub-floor
        # baseline that has genuinely grown past the floor must still show up
        # in a regeneration, or it stays pinned at its old sub-floor value
        # forever, "permanently ungated" even by an explicit regen. A brand
        # new zero-total test is excluded: it means the test performed no
        # solves at all, not a legitimately low count that could organically
        # grow, and recording it would seed a regeneration with an entry for
        # every non-solving test in the suite. A test that already has a
        # baseline entry is recorded even at zero, though: it failed above
        # with a "fewer iterations" message, and the regenerate hint in that
        # message promises a regen will clear the failure -- which it cannot
        # do if the dropped-to-zero observation is withheld here, leaving the
        # stale entry to be edited out by hand instead.
        if total > 0 or expected is not None:
            self._observed[name] = (total, solves)
        # A relative band is meaningless below this floor. Gate on the
        # baseline when one exists -- it is the stable number -- rather than
        # on `total`: a baseline of 100 that suddenly drops to 3 is exactly
        # the regression this guard exists to catch, and gating on the
        # (small) observed count let it through silently. Only a brand-new
        # test, with no baseline to check against, falls back to gating on
        # its own first observation.
        # An explicit per-test override is consulted before the floor: the
        # floor exists because a relative band is meaningless for a small
        # default tolerance, but an override is a deliberate choice about
        # this one test's band and must not be silently discarded just
        # because its baseline happens to sit below the floor.
        override = self._per_test_tolerance.get(name) if expected is not None else None
        if (
            expected is not None
            and override is None
            and expected.total < _MIN_ITERATIONS
        ):
            return None

        if self._writing:
            return None

        if expected is None:
            # A new test is not a regression. Recording it is a deliberate act,
            # so say so rather than failing someone who just added a test.
            return None

        return self._compare(expected, override, total, solves)

    @staticmethod
    def _format_allowance(value: float) -> str:
        # `.0f` reads fine for the common case (`_ABS_SLACK` floors it at 3
        # absent an override), but a per-test override can legitimately drive
        # this below 1 -- e.g. a baseline of 8 with a 5% override allows 0.4.
        # `.0f` would then print "+/-0", claiming zero tolerance for a
        # message whose whole point is reporting the tolerance actually
        # applied.
        """Format an allowed deviation for a failure message.

        Uses no decimals for an allowance of at least 1, and two significant digits
        below that, which a per-test override can produce (a baseline of 8 with a 5%
        override allows 0.4), where rounding to "+/-0" would misreport the band applied.

        Parameters
        ----------
         value : float
            The allowed deviation, in iterations.

        Returns
        -------
         text : str
            The formatted allowance.

        """
        return f"{value:.0f}" if value >= 1 else f"{value:.2g}"

    def _compare(
        self,
        expected: _Expected,
        override: float | None,
        total: int,
        solves: list[int],
    ) -> str | None:
        """Check a test's total and worst solve against their baseline bands.

        The allowance for a baseline figure v is
           max(``_ABS_SLACK``, tolerance × v)
        with the default tolerance, or exactly override × v with a per-test override,
        which is honored even below the absolute slack. The total is checked first. The
        worst single solve is then checked against the same band, but only when the
        baseline records a worst-solve figure of at least ``_MIN_ITERATIONS`` and the
        test performed solves; a total can sit inside its band while one solve
        underneath it has stopped converging.

        Parameters
        ----------
         expected : _Expected
            The baseline expectation.
         override : float | None
            Per-test tolerance override, or ``None`` for the default band.
         total : int
            Observed total iterations.
         solves : list[int]
            Observed iterations of each solve.

        Returns
        -------
         message : str | None
            A failure message naming the figure that moved, or ``None`` when both are
            within their bands.

        """
        if override is not None:
            # An explicit per-test override is a deliberate choice about this
            # one test, not a suite-wide default -- honor it exactly, even
            # below `_ABS_SLACK`, or "narrow the band" is a documented lie.
            tolerance = override
            band_desc = f"{tolerance:.0%}"

            def allowance(value: int) -> float:
                """Return the allowed deviation under the per-test override.

                Parameters
                ----------
                 value : int
                    The baseline figure, in iterations.

                Returns
                -------
                 allowed : float
                    value × override, with no absolute floor.

                """
                return value * tolerance

        else:
            tolerance = self._tolerance
            band_desc = f"the larger of {tolerance:.0%} and {_ABS_SLACK}"

            def allowance(value: int) -> float:
                """Return the allowed deviation under the default band.

                Parameters
                ----------
                 value : int
                    The baseline figure, in iterations.

                Returns
                -------
                 allowed : float
                    The larger of ``_ABS_SLACK`` and value × tolerance.

                """
                return max(float(_ABS_SLACK), value * tolerance)

        allowed = allowance(expected.total)
        if abs(total - expected.total) > allowed:
            direction = "more" if total > expected.total else "fewer"
            return (
                f"solver work moved: {total} iterations against a baseline of "
                f"{expected.total} ({direction}; allowed "
                f"+/-{self._format_allowance(allowed)}, "
                f"{band_desc}).{_REGENERATE_HINT}"
            )

        # The total can sit comfortably inside its band while one solve
        # underneath it has stopped converging -- the case that motivated
        # recording solves individually. Guarded only when the baseline
        # actually carries a worst-solve figure (the scalar forms do
        # not) and that figure is above the noise floor.
        if (
            expected.max_solve is None
            or expected.max_solve < _MIN_ITERATIONS
            or not solves
        ):
            return None
        observed_max = max(solves)
        max_allowed = allowance(expected.max_solve)
        if abs(observed_max - expected.max_solve) <= max_allowed:
            return None
        direction = "more" if observed_max > expected.max_solve else "fewer"
        return (
            f"worst single solve moved: {observed_max} iterations against a "
            f"baseline of {expected.max_solve} ({direction}; allowed "
            f"+/-{self._format_allowance(max_allowed)}, {band_desc}). "
            f"The per-test total "
            f"({total} vs {expected.total}) is still inside its band, so this "
            f"is one solve changing behaviour rather than the test as a "
            f"whole.{_REGENERATE_HINT}"
        )

    def write_if_requested(self) -> None:
        """Merge this process's observations into the output baseline, in write mode.

        Does nothing outside write mode. Writes to ``CVXIUM_PERF_BASELINE_OUT`` when it
        is set, else to the baseline this recorder loaded.

        A parallel runner (pytest-xdist, say) splits the suite across processes, each of
        which observes only the tests it ran. Under an exclusive lock on a separate ``.lock`` file, each process reads
        the file's current contents (which already hold earlier processes'
        contributions), overlays its own observations, restores the ignore list from its
        loaded baseline, and writes the result atomically. It never merges its own
        start-of-run snapshot, which would clobber other processes' fresh entries or
        resurrect deleted tests. The lock lives on its own path because the atomic write
        replaces the data file's inode.

        Raises
        ------
         ValueError
            Propagated from ``_load_current`` when the output file is not a JSON object
            or not valid JSON.
         OSError
            Propagated from the lock, read or write.

        """
        if not self._writing:
            return
        out_path = os.environ.get(_ENV_OUT) or str(self._path)
        # A parallel runner splits the suite across multiple processes. Each one's
        # `self._baseline` is a snapshot taken at process start, and each
        # `self._observed` only covers the tests that process ran, so writing
        # `{**self._baseline, **self._observed}` unconditionally makes the
        # last process to exit clobber every other process's freshly observed
        # entries with the stale pre-regeneration values it loaded -- silently,
        # since the file still comes out with a full key count. An exclusive
        # lock around a read-current/merge/write-back cycle closes that: each
        # process folds its own observations into whatever the file currently
        # holds (which already carries earlier processes' contributions)
        # rather than into the snapshot it started with. Deliberately excludes
        # this process's own start-of-run snapshot from the merge: a test
        # deleted from source (and so never in `observed`) could otherwise be
        # resurrected from that stale snapshot even after another process's
        # write had already dropped it from the live file.
        #
        # The lock lives on a dedicated `.lock` path rather than `out_path`
        # itself: the write below replaces `out_path` with a new inode, so a
        # lock held on an fd opened against the pre-replace inode would stop
        # protecting the file the moment the next process opens `out_path` by
        # name and locks the (different) inode now living there. A stable
        # lock path sidesteps that regardless of how the data file is written.
        lock_path = f"{out_path}.lock"
        with open(lock_path, "a") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            current = self._load_current(out_path)
            observed = self._build_observed(current)
            merged: dict[str, Any] = {**current, **observed}
            if self._loaded_baseline and self._ignored:
                merged[_IGNORE_KEY] = sorted(self._ignored)
            text = json.dumps(dict(sorted(merged.items())), indent=2) + "\n"
            self._atomic_write(out_path, text)
        # Excludes `_IGNORE_KEY`: it is not a baseline, and counting it here
        # would overstate how many tests this write actually recorded.
        written = len(merged) - (1 if _IGNORE_KEY in merged else 0)
        print(
            f"[perf_guard] wrote {written} baselines to {out_path}",
            file=sys.stderr,
        )

    def _load_current(self, out_path: str) -> dict[str, Any]:
        """Load the output baseline's current contents for merging.

        When this process loaded a baseline, the on-disk ignore list is dropped so the
        process's own (start-of-run) ignore list replaces it, rather than a union that
        would resurrect a name someone deliberately removed. A process whose baseline
        never loaded leaves the on-disk ignore list alone.

        Parameters
        ----------
         out_path : str
            The output baseline file; a missing or empty file reads as empty.

        Returns
        -------
         current : dict[str, Any]
            The file's entries, without ``__ignore__`` when this process loaded a
            baseline.

        Raises
        ------
         ValueError
            When the file is not valid JSON, or not a JSON object.

        """
        try:
            with open(out_path) as f:
                raw = f.read()
        except FileNotFoundError:
            raw = ""
        current_raw: Any = json.loads(raw) if raw else {}
        if not isinstance(current_raw, dict):
            # Same failure mode as the `__init__` load path: a truncated
            # write or hand-edit can leave the file as `null`, a list, or
            # a scalar. Left unchecked, `{**current, **observed}` below
            # raises a bare TypeError from inside this atexit handler.
            raise ValueError(
                f"malformed baseline at {out_path}: expected a JSON "
                f"object mapping test name to iteration count, got "
                f"{type(current_raw).__name__}"
            )
        current: dict[str, Any] = current_raw
        # Every process in a run reads `__ignore__` from the same source
        # baseline at start-of-run, so there is nothing to merge here the
        # way per-test entries are merged below: unioning with whatever
        # is still on disk (`current`'s own `__ignore__`) would silently
        # resurrect a name someone deliberately removed from the source
        # and then regenerated, the same failure mode the per-process
        # snapshot exclusion above exists to avoid for ordinary entries.
        # Gated on `self._loaded_baseline`: a process whose own baseline
        # never resolved has no trustworthy opinion on the ignore list
        # and must leave whatever is already in `current` alone, rather
        # than blanking it out because its own (unloaded) `self._ignored`
        # reads empty.
        if self._loaded_baseline:
            current.pop(_IGNORE_KEY, None)
        return current

    def _build_observed(self, current: dict[str, Any]) -> dict[str, Any]:
        # A test with a per-test tolerance override keeps the object form on
        # regeneration -- flattening it back to a bare int would silently
        # drop the override the next time this file is read.
        #
        # Regeneration only ever adds or updates entries for tests that
        # solved something in this run; it can never learn that a test
        # was deleted, renamed, or reworked to stop solving entirely,
        # because that looks identical -- from this one process's
        # partial view of a suite a parallel runner has sharded -- to a
        # test that simply runs in a different process. So a dead entry
        # can never be dropped by regenerating; it has to be deleted by
        # hand from the checked-in baseline.
        """Encode this process's observations as baseline entries.

        A test's tolerance override is kept: from this process's baseline when it has
        one, else from the entry already on disk. Regeneration only adds or updates
        entries for tests that ran in this process; it cannot tell a deleted test from
        one that ran in another process, so a dead entry has to be removed from the
        baseline by hand.

        Parameters
        ----------
         current : dict[str, Any]
            The output file's current entries, consulted for tolerance overrides.

        Returns
        -------
         observed : dict[str, Any]
            Encoded entries, keyed by test name.

        """
        observed: dict[str, Any] = {}
        for name, (total, solves) in self._observed.items():
            tolerance = self._per_test_tolerance.get(name)
            if tolerance is None:
                # This process has no opinion of its own (e.g. its
                # baseline never resolved) -- fall back to whatever
                # tolerance is already on disk rather than silently
                # dropping it, the same silent-loss failure mode the
                # `_loaded_baseline` gate above exists to prevent for the
                # ignore list.
                existing = current.get(name)
                if isinstance(existing, dict) and "tolerance" in existing:
                    tolerance = existing["tolerance"]
            observed[name] = _encode_observation(total, solves, tolerance)
        return observed

    def _atomic_write(self, out_path: str, text: str) -> None:
        # Written to a sibling temp file and swapped in with `os.replace`
        # rather than truncated in place: an in-place truncate leaves a
        # window, between the truncate and a completed write, where a
        # crash or an I/O error (ENOSPC, a killed process) leaves the
        # checked-in baseline empty or partial. The next read then hits
        # the malformed-baseline check in `__init__` and crashes Cvxium's
        # import for every test process. `os.replace` is atomic on the
        # same filesystem, so `out_path` always shows either the old
        # complete contents or the new complete contents, never a partial
        # write, still under the lock held by the caller.
        """Replace a file's contents atomically.

        Writes to a sibling temporary file, syncs it, and swaps it in with
        ``os.replace``, so the file always holds either the old or the new complete
        contents; an in-place truncate could leave a partial baseline that crashes every
        later import. The temporary file is removed when any step fails.

        Parameters
        ----------
         out_path : str
            The file to replace.
         text : str
            The new contents.

        Raises
        ------
         OSError
            Propagated from the write, sync or replace, after removing the temporary
            file.

        """
        tmp_path = f"{out_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as tmp_f:
                tmp_f.write(text)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, out_path)
        except BaseException:
            # A write/fsync/replace failure would otherwise leave this
            # process's scratch file behind under the checked-in test
            # directory -- untracked debris a future `sl status` (or
            # commit) picks up, on top of whatever raised.
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise


def _make_spy(recorder: "_Recorder", real: Any) -> Any:
    """Build a counting wrapper around a ``solve`` method.

    Parameters
    ----------
     recorder : _Recorder
        The recorder to notify.
     real : Any
        The original ``solve`` function.

    Returns
    -------
     spy : Any
        A function with ``real``'s calling convention that reports each call to
        ``recorder``.

    """

    def spy(self: Any, *a: Any, **k: Any) -> Any:
        """Call the wrapped ``solve`` and report it to the recorder.

        A call that raises, or returns something other than a ``SolveResult``, is
        unwound without being counted; a raising solve did consume iterations, but the
        result that would report them is gone.

        Parameters
        ----------
         *a : Any
            Positional arguments forwarded to ``solve``.
         **k : Any
            Keyword arguments forwarded to ``solve``.

        Returns
        -------
         out : Any
            Whatever the wrapped ``solve`` returned.

        Raises
        ------
         BaseException
            Whatever the wrapped ``solve`` raised, re-raised unchanged.

        """
        recorder.enter()
        try:
            out = real(self, *a, **k)
        except BaseException:
            # A raising solve still consumed iterations, but the result
            # that would report them is gone. Unwind the depth so the next
            # solve is counted from a clean slate.
            recorder.unwind()
            raise
        if isinstance(out, SolveResult):
            recorder.leave(out)
        else:
            recorder.unwind()
        return out

    return spy


def _wrap_solve(cls: type, recorder: "_Recorder") -> None:
    """Replace a class's ``solve`` with a counting wrapper, at most once per class.

    Does nothing when the class has no ``solve`` or was already wrapped.

    Parameters
    ----------
     cls : type
        The class to patch.
     recorder : _Recorder
        The recorder the wrapper reports to.

    """
    real = getattr(cls, "solve", None)
    if real is None or cls in _wrapped_classes:
        return

    cls.solve = _make_spy(recorder, real)  # type: ignore[attr-defined]
    _wrapped_classes.add(cls)


def _check(
    recorder: "_Recorder", baseline_faults: int, result: Any, test: unittest.TestCase
) -> None:
    """Raise the guard's verdict for the current test as a failure.

    Parameters
    ----------
     recorder : _Recorder
        The recorder holding the current test's tally.
     baseline_faults : int
        The ``_fault_count`` of ``result`` captured before the test ran.
     result : Any
        The test result object.
     test : unittest.TestCase
        The running test.

    Raises
    ------
     AssertionError
        When ``_verdict`` returns a failure message.

    """
    message = _verdict(recorder, baseline_faults, result, test)
    if message is not None:
        raise AssertionError(message)


def _make_run(recorder: "_Recorder", real_run: Any) -> Any:
    """Build a replacement for ``unittest.TestCase.run`` that guards each test.

    Parameters
    ----------
     recorder : _Recorder
        The recorder to drive.
     real_run : Any
        The original ``TestCase.run``.

    Returns
    -------
     run : Any
        The replacement ``run`` method.

    """

    def run(self: unittest.TestCase, result: Any = None) -> Any:
        # A caller-supplied ``None`` is substituted here, rather than left for
        # `real_run` to build internally, so this frame can see the same
        # result object `real_run` populates -- `_already_failing` needs its
        # fault counts, which are invisible if `real_run` builds and discards
        # its own. Because `real_run` then always sees a non-``None`` result,
        # it takes the branch that skips `startTestRun`/`stopTestRun` -- fire
        # those here ourselves, same as stock `TestCase.run` would have.
        """Run a test with its solves tallied and checked in a cleanup.

        Begins a tally under the test's module-qualified name (so same-named classes in
        different modules do not share a baseline entry), records the fault count, and
        registers a cleanup that renders the verdict. A cleanup, rather than a later
        ``addFailure`` on the result, runs inside the scope ``TestCase.run`` protects,
        so every runner, pytest included, attributes the failure to this test. A test
        skipped by decorator never reaches its cleanups, so its tally is abandoned
        afterwards.

        When ``result`` is ``None``, this method creates the default result itself, so
        it can read the fault counts, and fires ``startTestRun`` and ``stopTestRun`` as
        the stock ``run`` would.

        Parameters
        ----------
         result : Any
            The ``unittest.TestResult`` to report to, or ``None`` for the test's
            default.

        Returns
        -------
         out : Any
            Whatever the original ``run`` returns.

        """
        owns_result = result is None
        if owns_result:
            result = self.defaultTestResult()
            start_test_run = getattr(result, "startTestRun", None)
            if start_test_run is not None:
                start_test_run()
        # Module-qualified: two test classes sharing a name in different
        # test_*.py modules (a common pattern) would otherwise collapse onto
        # one baseline entry, each silently checked against the other's count.
        recorder.begin(
            f"{type(self).__module__}.{type(self).__name__}.{self._testMethodName}"
        )
        consumed = False
        # Captured before `real_run` executes `setUp`/the test body, so the
        # cleanup below can tell "a new fault happened during this test" from
        # "one was already on the board before this test started" -- see
        # `_already_failing`.
        baseline_faults = _fault_count(result)

        def _consume() -> None:
            """Mark the cleanup as fired and check the test against the baseline.

            Raises
            ------
             AssertionError
                Propagated from ``_check`` when the guard fails the test.

            """
            nonlocal consumed
            consumed = True
            _check(recorder, baseline_faults, result, self)

        # A cleanup, not a post-hoc ``addFailure`` on the result object. The
        # cleanup runs inside the scope ``TestCase.run`` protects, so the
        # failure is attributed to this test by every runner. Adding it to the
        # result afterwards works under plain unittest but is silently dropped
        # by some runners, which would let the guard pass under one runner
        # while failing under another.
        self.addCleanup(_consume)
        try:
            return real_run(self, result)
        finally:
            # A test skipped by decorator returns from ``TestCase.run`` before
            # it reaches ``doCleanups``, so the cleanup above never fires and
            # this frame's tally is left open -- abandon it here. When the
            # cleanup did fire (the normal path), ``end``/``abandon`` already
            # popped this frame from inside it; popping again here would take
            # the *next* frame down the stack instead, which is how a nested
            # ``TestCase.run`` used to discard the outer test's tally.
            if not consumed:
                recorder.abandon()
            if owns_result:
                stop_test_run = getattr(result, "stopTestRun", None)
                if stop_test_run is not None:
                    stop_test_run()

    return run


def _write_or_die(recorder: "_Recorder") -> None:
    """Write the baseline at exit, forcing a nonzero exit status on failure.

    ``write_if_requested`` raises plainly, so tests can call it directly and assert on
    the exception. But CPython reports an exception raised from an ``atexit`` callback
    as "Exception ignored" and leaves the exit status at 0 (``SystemExit`` is swallowed
    the same way), so a regeneration run whose write failed would still report success.
    This wrapper, registered with ``atexit`` instead of the method, prints the traceback
    and calls ``os._exit(1)``, the only way to force a nonzero status from an ``atexit``
    callback.

    Parameters
    ----------
     recorder : _Recorder
        The recorder whose observations to write.

    """
    try:
        recorder.write_if_requested()
    except BaseException:
        # there is nothing to re-raise to.
        traceback.print_exc()
        os._exit(1)


def install(baseline_path: Path | str, tolerance: float = 0.1) -> None:
    """Patch the solver entry points and the test runner to enforce a baseline.

    Wraps ``solve`` on ``Problem``, ``BarrierSolver``, ``HsdSolver``,
    ``PrimalDualSolver``, ``NewtonSolver`` and ``EqualitySolver``, replaces
    ``unittest.TestCase.run`` with a guarded version, and registers the baseline write
    at exit. Idempotent: a second call does nothing.

    Parameters
    ----------
     baseline_path : Path | str
        JSON file mapping ``module.ClassName.test_method`` to a baseline entry, in any
        form ``_parse_baseline_entry`` reads.
     tolerance : float
        Default relative band, in both directions: 0.1 lets a test move 10% either way,
        or ``_ABS_SLACK`` iterations if that is larger, before it fails.

    Raises
    ------
     ValueError
        Propagated from ``_Recorder`` when the baseline is malformed.

    """
    global _installed
    if _installed:
        return

    recorder = _Recorder(Path(baseline_path), tolerance)

    from ..problems.problem import Problem
    from ..solvers.barrier_solver import (
        BarrierSolver,
    )
    from ..solvers.hsd_solver import HsdSolver
    from ..solvers.newton_solver import (
        NewtonSolver,
    )
    from ..solvers.primal_dual_solver import (
        PrimalDualSolver,
    )
    from ..solvers.solver import EqualitySolver

    # `Problem.solve` delegates to one of these; `NewtonSolver` and
    # `EqualitySolver` are also driven directly by their own dedicated test
    # files (test_newton_solver.py, the `EqualitySolver` cases in
    # test_linear_programs.py). Unwrapped, those tests would accumulate a
    # total of 0 every run and take the never-fails new-test path, leaving a
    # regression in either unguarded. `enter`/`leave`'s
    # depth counter already keeps a nested call (`Problem.solve` calling one
    # of these internally) from double-counting.
    #
    # `StationaritySolver`/`PrimalDualStationaritySolver` are deliberately
    # not wrapped here: they are building blocks inside downstream consumers'
    # own solvers (e.g. a two-block saddle-point solver), so wrapping them
    # shifts what "the test's total" means for every baseline entry built on
    # top of them, well beyond this file. They have their own dedicated
    # line-search instrumentation instead (see `test_stationarity_solver.py`'s
    # line-search recording).
    for cls in (
        Problem,
        BarrierSolver,
        HsdSolver,
        PrimalDualSolver,
        NewtonSolver,
        EqualitySolver,
    ):
        _wrap_solve(cls, recorder)

    real_run = unittest.TestCase.run
    unittest.TestCase.run = _make_run(recorder, real_run)  # type: ignore[method-assign]
    atexit.register(_write_or_die, recorder)
    # Set only once every patch above has actually applied: raising midway
    # (an import error, say) with this set first would leave the guard
    # permanently -- and silently -- inactive for the rest of the process,
    # since the early return above treats `_installed` as proof the patching
    # happened.
    _installed = True


def install_from_environment() -> None:
    """Install the guard when the test suite asks for it via ``CVXIUM_PERF_BASELINE``.

    Cvxium's ``__init__`` calls this hook. When the variable is absent, which is every
    production import, it costs one environment lookup. A relative path that does not
    exist as given is resolved against the repository root, derived from this module's
    location.

    Raises
    ------
     ValueError
        Propagated from ``install`` when the baseline is malformed.

    """
    path = os.environ.get(_ENV_BASELINE)
    if not path:
        return
    install(_resolve_baseline_path(path))


def _resolve_baseline_path(path: str) -> Path:
    """Resolve the baseline path named in ``CVXIUM_PERF_BASELINE``.

    Parameters
    ----------
     path : str
        The path as given: absolute, or relative to the working directory or to the
        repository root.

    Returns
    -------
     baseline : Path
        The path as given when it is absolute or exists; otherwise the same path
        under the repository root.

    Notes
    -----
    The repository root is derived from this module's own location, so a suite run from
    another directory still finds a repo-relative baseline. A baseline that is not found
    disables the guard silently, so ``test_perf_guard`` checks that the suite's own
    baseline resolves.

    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    #   <root>/cvxium/diagnostics/perf_guard.py
    return Path(__file__).resolve().parents[2] / path
