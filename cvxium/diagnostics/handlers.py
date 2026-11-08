"""Logging handlers and log configuration for Cvxium solvers.

Cvxium solvers narrate their progress through the standard ``logging`` module, attaching
structured fields (named with a ``cvx_`` prefix) to each record. This module provides
two ways to consume that output. ``IterationHandler`` collects the structured fields
into a list of dicts, or a pandas DataFrame, for analysis after a solve.
``enable_logging`` routes one or more logger namespaces to a single file or stderr
handler, replacing the usual handler boilerplate and making repeated calls idempotent.

"""

import logging
import os
from collections.abc import Sequence
from typing import Any


class IterationHandler(logging.Handler):
    """Logging handler that accumulates per-iteration solver events.

    Captures the structured data that Cvxium solvers attach to their log records into a
    list of dicts, one per record. Each dict holds the record's creation time and
    message, plus every recognized ``cvx_`` field present on the record, keyed without
    the prefix: the event type, the solver, iteration counters, timing, and the
    numerical diagnostics of that event. Call ``to_dataframe`` for a pandas DataFrame
    (requires pandas), or ``records`` for the raw list of dicts.

    Parameters
    ----------
     level : int
        Threshold passed to ``logging.Handler``; records below it are ignored.

    Examples
    --------
    Attach before solving, then inspect afterwards:
       import logging
       from cvxium import IterationHandler

       handler = IterationHandler()
       logging.getLogger("cvxium").addHandler(handler)
       logging.getLogger("cvxium").setLevel(logging.DEBUG)

       solver.solve()

       df = handler.to_dataframe()
       # df has columns: time, message, event, outer_nit, inner_nit, elapsed_ms, ...

    """

    _CVX_FIELDS = (
        "cvx_event",
        "cvx_solver",
        "cvx_outer_nit",
        "cvx_inner_nit",
        "cvx_elapsed_ms",
        "cvx_suboptimality",
        "cvx_t",
        "cvx_btls_s",
        "cvx_improvement",
        "cvx_expected_improvement",
        # HSD events (cvxium.hsd_solver logger).
        "cvx_nit",
        "cvx_mu",
        "cvx_rel_pres",
        "cvx_rel_dres",
        "cvx_tau",
        "cvx_kappa",
        "cvx_nits",
        "cvx_status",
        # PD events (cvxium.pd_solver logger).
        "cvx_eta",
        "cvx_dual_norm",
        "cvx_pri_norm",
        "cvx_residual_norm",
        "cvx_step",
    )

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self._records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store one log record as a dict.

        The dict holds the record's creation time (``time``) and formatted message
        (``message``), plus each field of ``_CVX_FIELDS`` that the record carries with a
        value other than ``None``, keyed without its ``cvx_`` prefix. Fields not listed
        in ``_CVX_FIELDS`` are dropped.

        Parameters
        ----------
         record : logging.LogRecord
            The record to store.

        """
        row: dict[str, Any] = {
            "time": record.created,
            "message": record.getMessage(),
        }
        for field in self._CVX_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                row[field.removeprefix("cvx_")] = value
        self._records.append(row)

    def records(self) -> list[dict[str, Any]]:
        """Return the accumulated records as a list of dicts.

        Returns
        -------
         records : list[dict[str, Any]]
            A shallow copy of the accumulated list, one dict per record, in arrival
            order. Appending to or removing from it does not affect the handler.

        """
        return list(self._records)

    def clear(self) -> None:
        """Discard every accumulated record."""
        self._records.clear()

    def to_dataframe(self) -> Any:
        """Convert the accumulated records to a pandas DataFrame.

        Requires pandas, which Cvxium does not otherwise depend on; it is imported only
        here.

        Returns
        -------
         df : pd.DataFrame
            One row per log record. Columns are the union of the keys across records:
            ``time`` and ``message`` always, plus fields such as ``event``, ``solver``,
            ``outer_nit``, ``inner_nit``, ``elapsed_ms``, ``suboptimality``, ``t``,
            ``btls_s``, ``improvement``, and ``expected_improvement``. A column with no
            data for a given event type holds ``NaN`` in that row.

        Raises
        ------
         ImportError
            When pandas is not installed.

        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required to call to_dataframe(). "
                "Install it with: pip install pandas"
            ) from e
        return pd.DataFrame(self._records)


_previous_namespaces: tuple[str, ...] = ()

# Namespace -> the handler this module itself attached there. Tracked so the
# stale-namespace cleanup below can remove exactly that handler and nothing
# else, leaving any handler a caller attached independently of this module.
_namespace_handlers: dict[str, logging.Handler] = {}


def _detach(namespace: str) -> None:
    """Undo this module's configuration of a namespace dropped since the last call.

    Used only by ``enable_logging`` for a namespace that an earlier call configured and
    the current call omits. A namespace named in the current call is instead fully
    reset, which is a deliberate hard reset; a namespace merely dropped between calls
    was never asked to give up a handler some other caller attached independently. So
    this removes and closes only the handler this module attached (if still present),
    rather than clearing the logger wholesale.

    It also restores the logger's ``propagate`` and level to the logging defaults
    (``True`` and ``logging.NOTSET``), so the dropped namespace is not left muted: with
    ``propagate`` ``False`` and no handler, its records would be silently swallowed
    instead of reaching the root logger.

    Parameters
    ----------
     namespace : str
        Logger name to detach.

    """
    log = logging.getLogger(namespace)
    old_handler = _namespace_handlers.pop(namespace, None)
    if old_handler is not None and old_handler in log.handlers:
        log.removeHandler(old_handler)
        old_handler.close()
    log.propagate = True
    log.setLevel(logging.NOTSET)


def enable_logging(
    path: str | os.PathLike[str] | None = None,
    level: int = logging.INFO,
    namespaces: Sequence[str] = ("cvxium",),
    mode: str = "w",
) -> logging.Handler:
    """Send one or more logger namespaces to a single handler.

    Replaces a dozen lines of handler boilerplate. The part worth centralizing is not
    the file handler but setting ``propagate`` to ``False`` and resetting the handlers:
    the first stops records from also reaching the root logger (which would duplicate
    every line in a notebook), and the second makes calling this twice idempotent rather
    than cumulative.

    When more than one namespace is passed, sharing one handler across them is the
    point. They are separate logger trees, so the only thing putting their records in
    order is that they arrive at the same place: the log is then chronological, and a
    solver's iterations sit underneath the line naming the problem being solved.

    Parameters
    ----------
     path : str | os.PathLike[str] | None
        Log file; ``~`` is expanded. ``None`` logs to stderr instead, which suits a
        terminal but not a notebook.
     level : int
        Threshold for both the handler and each namespace. ``logging.INFO`` gives the
        narration; ``logging.DEBUG`` adds per-iteration solver detail, which runs to
        thousands of lines on a large problem. Set unconditionally on every call, unlike
        the ``NOTSET``-guarded default in ``__init__.py``. That guard exists so a level
        set before import survives it; here the caller is asking for exactly this level,
        so a level from an earlier call does not survive a later one that omits
        ``level``.
     namespaces : Sequence[str]
        Logger names to attach to. Defaults to Cvxium alone; a library built on Cvxium
        passes its own name alongside the ones it depends on. Repeated names are
        dropped, keeping the first occurrence. Must not be a bare ``str``, which
        satisfies ``Sequence[str]`` but would iterate character by character; the
        function raises ``TypeError`` for one.
     mode : str
        File mode. ``"w"`` truncates on each call, usually what is wanted when the
        question is what happened this time; ``"a"`` appends. Must be left at ``"w"``
        when ``path`` is ``None``, since stderr has no file mode; any other value raises
        ``ValueError``.

    Returns
    -------
     handler : logging.Handler
        The attached handler, a ``logging.FileHandler`` or ``logging.StreamHandler``,
        for a caller that wants to remove it again.

    Raises
    ------
     TypeError
        When ``namespaces`` is a bare ``str`` rather than a sequence of names.
     ValueError
        When ``mode`` is not ``"w"`` while ``path`` is ``None``.
     OSError
        Propagated from ``logging.FileHandler`` when ``path`` cannot be opened. It is
        raised before any logger is touched, so the previous configuration stays intact.

    Notes
    -----
    Each call proceeds in three steps:
    1. Build the new handler, before tearing anything down, so that a bad path
       leaves the previous configuration in place.
    2. Detach (see ``_detach``) each namespace that the previous call configured
       and this call omits. Its handler is shared with namespaces that are in this
       call and is about to be closed, so leaving it attached would send the
       namespace's next record to a closed handler.
    3. For each namespace in this call, close and remove every existing handler
       (including one this module attached earlier, and the placeholder
       ``NullHandler`` that ``__init__.py`` puts on ``cvxium``), set ``propagate``
       to ``False``, set the level, and attach the new handler.

    Examples
    --------
    Narration only, to a file:
       Cvxium.enable_logging("~/cvxium.log")

    Every inner iteration, which runs to thousands of lines:
       Cvxium.enable_logging("~/cvxium.log", level=logging.DEBUG)

    """
    if isinstance(namespaces, str):
        raise TypeError(
            f"namespaces must be a sequence of strings, not a bare str: "
            f"{namespaces!r}. Did you forget the trailing comma, e.g. "
            f'namespaces=("{namespaces}",)?'
        )
    if path is None and mode != "w":
        # A StreamHandler has no file mode, so `mode` silently has no effect
        # when `path` is None. Reject it rather than let a caller believe
        # mode="a" is accumulating output that is actually going to stderr.
        raise ValueError(
            f"mode={mode!r} has no effect when path is None: logging goes "
            "to stderr via a StreamHandler, which has no file mode. Pass "
            "path to log to a file, or omit mode."
        )

    global _previous_namespaces
    # dict.fromkeys dedupes while preserving order. Without this, a repeated
    # name in `namespaces` re-visits the same logger in the loop below: the
    # second visit closes the handler the first visit just attached, then
    # re-attaches that now-closed handler.
    current = tuple(dict.fromkeys(namespaces))

    # Build the handler before tearing anything down: FileHandler can raise
    # (e.g. a bad path), and if that happens after the stale-namespace
    # detach loop below, previously-configured namespaces would already be
    # torn down with no replacement handler installed.
    if path is None:
        handler: logging.Handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(os.path.expanduser(os.fspath(path)), mode=mode)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    )
    handler.setLevel(level)

    # A namespace dropped from the current call still holds a reference to
    # the handler shared with namespaces that ARE in this call. That handler
    # gets closed below, so a dropped namespace left untouched would write
    # its next record to a closed FileHandler. Detach it here, before it goes
    # stale, even though it is otherwise untouched by this call.
    for stale in set(_previous_namespaces) - set(current):
        _detach(stale)

    for namespace in current:
        # Reset rather than append: calling this twice in one notebook session
        # would otherwise write every record two, then three times. A
        # namespace named in the current call is being fully (re)configured,
        # so every existing handler goes -- including one this module
        # attached earlier, and including `__init__.py`'s placeholder
        # NullHandler on "cvxium" -- unlike the gentler stale-namespace
        # cleanup above, which only removes this module's own handler from a
        # namespace no longer being configured at all.
        _namespace_handlers.pop(namespace, None)
        log = logging.getLogger(namespace)
        for old_handler in log.handlers:
            old_handler.close()
        log.handlers.clear()
        log.propagate = False
        log.setLevel(level)
        log.addHandler(handler)
        _namespace_handlers[namespace] = handler

    _previous_namespaces = current
    return handler
