"""Narrow optional values for the type checker."""

from typing import TypeVar

T = TypeVar("T")


def none_throws(value: T | None, message: str = "Unexpected None") -> T:
    """Return ``value``, raising ``AssertionError`` if it is ``None``.

    Unlike a bare ``assert value is not None``, the check survives ``python -O``, and it
    reads as an expression where a statement would not fit.

    """
    if value is None:
        raise AssertionError(message)
    return value
