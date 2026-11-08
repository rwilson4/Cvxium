"""Every Cvxium logger has to hang off the `cvxium` root."""

import importlib
import logging
import pkgutil
import tempfile
import unittest
from pathlib import Path

import cvxium
from cvxium.diagnostics import handlers


class LoggerNamespaceTest(unittest.TestCase):
    """The namespace is the whole configuration surface.

    `Cvxium/__init__.py` attaches a NullHandler to `cvxium` and callers
    configure that one name. A module that reaches for
    `getLogger(__name__)` instead lands under
    `<mod>`, which is a different
    tree: its records never reach that handler, and the failure is silent --
    the log simply has nothing from that module in it, which reads like the
    module was never exercised.

    """

    @staticmethod
    def _module_loggers() -> dict[str, list[str]]:
        """Module name -> names of the `logging.Logger` objects it holds."""
        found: dict[str, list[str]] = {}
        for info in pkgutil.walk_packages(
            cvxium.__path__, prefix=f"{cvxium.__name__}."
        ):
            if ".test" in info.name:
                continue
            module = importlib.import_module(info.name)
            names = [
                value.name
                for value in vars(module).values()
                if isinstance(value, logging.Logger)
            ]
            if names:
                found[info.name] = names
        return found

    def test_every_logger_is_under_the_cvxium_root(self) -> None:
        """Otherwise `getLogger("cvxium")` silently misses the module."""
        loggers = self._module_loggers()
        # Guard the guard: if the walk finds nothing, the assertion below is
        # vacuous and would keep passing while the namespace rotted.
        self.assertGreater(len(loggers), 3)

        stray = {
            module: [n for n in names if n != "cvxium" and not n.startswith("cvxium.")]
            for module, names in loggers.items()
        }
        stray = {module: names for module, names in stray.items() if names}

        self.assertEqual(
            stray,
            {},
            "These modules log outside the `cvxium` root, so their records do "
            "not reach a handler attached to it. Use "
            'logging.getLogger("cvxium.<module>").',
        )


def _reset_previous_namespaces() -> None:
    """Isolated version of the tearDown step that clears `_previous_namespaces`.

    Extracted so a test can check this specific reset without calling
    `tearDown` directly, which would run it twice per test (once here, once
    via the framework) -- harmless only as long as `tearDown` stays
    idempotent.
    """
    handlers._previous_namespaces = ()


class EnableLoggingTest(unittest.TestCase):
    """`enable_logging` is the twelve lines of boilerplate, written once."""

    def setUp(self) -> None:
        self._log = logging.getLogger("cvxium")
        self._saved = (list(self._log.handlers), self._log.level, self._log.propagate)

    def tearDown(self) -> None:
        saved_handlers, level, propagate = self._saved
        for handler in self._log.handlers:
            if handler not in saved_handlers:
                handler.close()
        self._log.handlers = saved_handlers
        self._log.setLevel(level)
        self._log.propagate = propagate
        # A test that passes multiple namespaces (e.g. ("cvxium", "other"))
        # otherwise leaks that tuple into the next test's stale-namespace
        # cleanup in `enable_logging`, making its behavior depend on
        # execution order.
        _reset_previous_namespaces()

    def test_records_reach_the_file(self) -> None:
        """The whole point: a solver's logger writes where it was told."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.log"
            cvxium.enable_logging(path)
            logging.getLogger("cvxium.optimization").info("hello from the solver")
            logging.getLogger("cvxium").handlers[0].flush()

            self.assertIn("hello from the solver", path.read_text())

    def test_one_handler_is_shared_so_the_file_is_chronological(self) -> None:
        """Separate trees interleave only because they share a destination.

        Two handlers on one file would each buffer independently, and the
        ordering that makes a combined log readable would be incidental.

        """
        with tempfile.TemporaryDirectory() as d:
            other = logging.getLogger("pretend_downstream")
            saved = (list(other.handlers), other.level, other.propagate)
            try:
                path = Path(d) / "out.log"
                cvxium.enable_logging(path, namespaces=("cvxium", other.name))

                self.assertIs(other.handlers[0], self._log.handlers[0])
            finally:
                for handler in other.handlers:
                    if handler not in saved[0]:
                        handler.close()
                other.handlers, other.level, other.propagate = saved

    def test_calling_it_twice_does_not_duplicate_every_line(self) -> None:
        """Re-running a notebook cell must not triple the log."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.log"
            cvxium.enable_logging(path)
            cvxium.enable_logging(path)

            self.assertEqual(len(self._log.handlers), 1)

    def test_it_stops_propagation_to_the_root_logger(self) -> None:
        """Otherwise a notebook shows every record twice."""
        with tempfile.TemporaryDirectory() as d:
            cvxium.enable_logging(Path(d) / "out.log")

            self.assertFalse(self._log.propagate)

    def test_a_level_set_before_import_survives_the_import(self) -> None:
        """Configuring by name does not require the library to be imported.

        `enable_logging` attaches to a logger NAME, so a caller can wire the
        namespaces up front and import the libraries afterwards -- which is
        the cheap path when the library costs seconds to import. An
        unconditional `setLevel` in the package body would then undo the
        level on import and the log would come back empty, which is the
        worst kind of failure: configured, silent, and no error.

        """
        import importlib

        with tempfile.TemporaryDirectory() as d:
            cvxium.enable_logging(Path(d) / "out.log", level=logging.DEBUG)
            self.assertEqual(self._log.level, logging.DEBUG)

            importlib.reload(cvxium)

            self.assertEqual(self._log.level, logging.DEBUG)

    def test_bare_str_namespaces_raises(self) -> None:
        """A bare str satisfies Sequence[str] and would iterate as characters.

        `enable_logging(path, namespaces="cvxium")` without the trailing
        comma must fail loudly rather than attach the handler to loggers
        named "c", "v", "x", ... .

        """
        with tempfile.TemporaryDirectory() as d, self.assertRaises(TypeError):
            cvxium.enable_logging(Path(d) / "out.log", namespaces="cvxium")

    def test_duplicate_namespaces_in_one_call_do_not_close_the_handler(
        self,
    ) -> None:
        """A repeated name must not re-close the handler it just attached.

        Without deduplication, the loop over `namespaces` visits `"cvxium"`
        twice: the second visit closes the handler the first visit just
        attached, then re-attaches that now-closed handler.

        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.log"
            cvxium.enable_logging(path, namespaces=("cvxium", "cvxium"))

            logging.getLogger("cvxium").info("still writable")
            self._log.handlers[0].flush()

            self.assertIn("still writable", path.read_text())

    def test_dropping_a_namespace_does_not_leave_a_closed_handler_attached(
        self,
    ) -> None:
        """A namespace excluded from a later call must not keep the old handler.

        Otherwise it holds a reference to the handler the later call closed,
        and any record still emitted through it hits a closed FileHandler.

        """
        other = logging.getLogger("pretend_downstream_dropped")
        saved = (list(other.handlers), other.level, other.propagate)
        try:
            with tempfile.TemporaryDirectory() as d:
                cvxium.enable_logging(
                    Path(d) / "first.log", namespaces=("cvxium", other.name)
                )
                first_handler = self._log.handlers[0]

                cvxium.enable_logging(Path(d) / "second.log")

                self.assertNotIn(first_handler, other.handlers)
        finally:
            for handler in other.handlers:
                if handler not in saved[0]:
                    handler.close()
            other.handlers, other.level, other.propagate = saved

    def test_dropping_a_namespace_restores_propagate_and_level(
        self,
    ) -> None:
        """A dropped namespace must not stay muted with no handler.

        Detaching only closes the handler; without also restoring
        `propagate` and `level`, the namespace is left with `propagate`
        False, a non-default level, and no handler, so any record emitted
        to it afterwards is silently swallowed instead of reaching the root
        logger.

        """
        other = logging.getLogger("pretend_downstream_muted")
        saved = (list(other.handlers), other.level, other.propagate)
        try:
            with tempfile.TemporaryDirectory() as d:
                cvxium.enable_logging(
                    Path(d) / "first.log", namespaces=("cvxium", other.name)
                )
                cvxium.enable_logging(Path(d) / "second.log")

                self.assertTrue(other.propagate)
                self.assertEqual(other.level, logging.NOTSET)
        finally:
            for handler in other.handlers:
                if handler not in saved[0]:
                    handler.close()
            other.handlers, other.level, other.propagate = saved

    def test_a_handler_attached_independently_survives_becoming_stale(
        self,
    ) -> None:
        """Dropping a namespace must not clobber a foreign handler on it.

        A namespace named in the current call is fully reconfigured -- every
        existing handler on it goes, deliberately, including one this module
        attached itself. But a namespace that merely falls out of a LATER
        call (goes stale) was never asked to give up anything it did not
        itself put there: the stale-namespace cleanup must remove only the
        handler this module attached, not a handler some other caller added
        to the same namespace afterward, independently of this module.

        """
        other = logging.getLogger("pretend_downstream_with_foreign_handler")
        saved = (list(other.handlers), other.level, other.propagate)
        try:
            with tempfile.TemporaryDirectory() as d:
                cvxium.enable_logging(
                    Path(d) / "first.log", namespaces=("cvxium", other.name)
                )
                # Attached after enable_logging configured `other`, not
                # before: this module's own full-clear on a namespace it is
                # actively (re)configuring is a separate, deliberate contract
                # covered by test_calling_it_twice_does_not_duplicate_every_line.
                foreign = logging.NullHandler()
                other.addHandler(foreign)

                cvxium.enable_logging(Path(d) / "second.log")

                self.assertIn(foreign, other.handlers)
        finally:
            foreign.close()
            for handler in other.handlers:
                if handler not in saved[0] and handler is not foreign:
                    handler.close()
            other.handlers, other.level, other.propagate = saved

    def test_mode_with_no_path_raises(self) -> None:
        """A `mode` that has no effect must not be accepted silently.

        `path=None` logs to stderr via a `StreamHandler`, which has no file
        mode, so a non-default `mode` would otherwise be silently ignored --
        a caller who passes `mode="a"` expecting accumulation gets no
        indication it did nothing.

        """
        with self.assertRaises(ValueError):
            cvxium.enable_logging(path=None, mode="a")

    def test_a_failed_open_does_not_tear_down_a_working_namespace(
        self,
    ) -> None:
        """A bad path in a later call must not silently kill a working namespace.

        If the stale-namespace cleanup ran before the new handler is built,
        a `FileHandler` that fails to open (here: a directory, not a file)
        would leave a previously-configured namespace detached -- propagate
        and level reset, handler closed -- with nothing to replace it, so
        logging for it silently stops instead of this call raising and
        leaving that namespace untouched.

        """
        other = logging.getLogger("pretend_downstream_survives_open_failure")
        saved = (list(other.handlers), other.level, other.propagate)
        try:
            with tempfile.TemporaryDirectory() as d:
                cvxium.enable_logging(
                    Path(d) / "first.log", namespaces=("cvxium", other.name)
                )
                working_handler = other.handlers[0]

                with self.assertRaises(IsADirectoryError):
                    cvxium.enable_logging(Path(d))  # a directory, not a file

                self.assertIn(working_handler, other.handlers)
                self.assertFalse(other.propagate)
        finally:
            for handler in other.handlers:
                if handler not in saved[0]:
                    handler.close()
            other.handlers, other.level, other.propagate = saved

    def test_resetting_previous_namespaces_clears_the_global(
        self,
    ) -> None:
        """The reset step `tearDown` relies on must actually clear the global.

        Without it, a namespace used only in one test would be treated as
        newly "stale" by the very next call to `enable_logging` -- in a
        later, unrelated test -- closing and clearing its handlers as a side
        effect of execution order rather than of any call that test itself
        made. Exercises the extracted `_reset_previous_namespaces` helper
        directly rather than `tearDown` itself, so this stays a test of one
        reset step instead of a second, redundant invocation of the whole
        teardown sequence.

        """
        other = logging.getLogger("pretend_leaked_namespace")
        saved = (list(other.handlers), other.level, other.propagate)
        try:
            with tempfile.TemporaryDirectory() as d:
                cvxium.enable_logging(
                    Path(d) / "out.log", namespaces=("cvxium", other.name)
                )
                self.assertEqual(handlers._previous_namespaces, ("cvxium", other.name))

            _reset_previous_namespaces()

            self.assertEqual(handlers._previous_namespaces, ())
        finally:
            for handler in other.handlers:
                if handler not in saved[0]:
                    handler.close()
            other.handlers, other.level, other.propagate = saved
