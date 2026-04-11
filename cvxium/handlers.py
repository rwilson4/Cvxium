"""Custom logging handlers for Cvxium solvers."""

import logging
from typing import Any


class IterationHandler(logging.Handler):
    """Accumulates per-iteration solver events for post-hoc analysis.

    Captures structured data emitted by Cvxium solvers into a list of dicts.
    Each dict contains the event type, iteration counters, timing, and any
    other numerical data emitted at that log record.

    Call :meth:`to_dataframe` to convert to a pandas DataFrame (requires
    pandas), or :meth:`records` to get the raw list of dicts.

    Examples
    --------
    Attach before solving, then inspect afterwards::

        import logging
        from cvxium import IterationHandler

        handler = IterationHandler()
        logging.getLogger("cvxium").addHandler(handler)
        logging.getLogger("cvxium").setLevel(logging.DEBUG)

        solver.solve()

        df = handler.to_dataframe()
        # df has columns: event, outer_nit, inner_nit, elapsed_ms, suboptimality, ...

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
    )

    def __init__(self, level: int = logging.NOTSET) -> None:
        """Initialize handler."""
        super().__init__(level)
        self._records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the log record as a dict."""
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
        """Return accumulated records as a list of dicts."""
        return list(self._records)

    def clear(self) -> None:
        """Clear accumulated records."""
        self._records.clear()

    def to_dataframe(self):
        """Convert accumulated records to a pandas DataFrame.

        Requires pandas to be installed.

        Returns
        -------
        pd.DataFrame
            One row per log record. Columns include ``event``, ``outer_nit``,
            ``inner_nit``, ``elapsed_ms``, ``suboptimality``, ``t``,
            ``btls_s``, ``improvement``, ``expected_improvement``, and
            ``message``. Columns with no data for a given event type will
            contain ``NaN``.

        Raises
        ------
        ImportError
            If pandas is not installed.

        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required to call to_dataframe(). "
                "Install it with: pip install pandas"
            ) from e
        return pd.DataFrame(self._records)
