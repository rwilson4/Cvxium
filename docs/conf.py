"""Sphinx configuration for the Cvxium documentation."""

# For a full list of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from importlib.metadata import version as _get_version

# The repository root, so autodoc imports the checked-out cvxium.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -- Project information -----------------------------------------------------

project = "Cvxium"
copyright = "2026, Bob Wilson"
author = "Bob Wilson"

# The full version, including alpha/beta/rc tags
release = _get_version("cvxium")


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # `make doctest` executes the quickstart, so it cannot drift from the library.
    "sphinx.ext.doctest",
    # The usage guide is USAGE.md, included rather than copied.
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": True}
# Docstrings state every type, and the expanded numpy aliases in signatures
# (ndarray[tuple[Any, ...], dtype[float64]]) only obscure them.
autodoc_typehints = "none"

# Dataclass docstrings document their fields under "Fields".
napoleon_custom_sections = [("Fields", "params_style")]


# -- Plain-text docstrings ---------------------------------------------------
#
# Cvxium's docstrings are written to read well as plain text: display
# equations, ASCII-art matrices and lists sit directly against the surrounding
# prose, indented but without blank lines, and absolute values use |x|.
# reStructuredText reads those as block quotes, definition lists and
# substitution references. Rather than rewrite every docstring, translate at
# build time, before napoleon parses the NumPy sections:
# - an indented display block becomes a literal block (blank lines around it,
#   "::" on the line that introduces it), which keeps its line breaks and
#   alignment;
# - a list gets the blank lines around it that RST requires;
# - a bare "|" outside inline literals is escaped.

_FIELD_SECTIONS = {"Parameters", "Returns", "Raises", "Fields", "Yields"}
_LIST_ITEM = re.compile(r"(?:[-*]|\d+\.)\s+")


def _indent(line: str) -> int:
    """Count leading spaces."""
    return len(line) - len(line.lstrip(" "))


def _escape_pipes(line: str) -> str:
    """Escape "|" outside ``inline literals``."""
    parts = line.split("``")
    for k in range(0, len(parts), 2):  # even parts are outside literals
        parts[k] = parts[k].replace("|", r"\|")
    return "``".join(parts)


def _mark_literal(line: str) -> str:
    """End a line with "::", which RST renders as ":" or as nothing."""
    stripped = line.rstrip()
    if stripped.endswith("::"):
        return stripped
    if stripped.endswith(":"):
        return stripped + ":"
    return stripped + " ::"


def _translate_plain_text(lines: list[str]) -> list[str]:
    """Rewrite plain-text docstring lines as reStructuredText."""
    out: list[str] = []
    is_header = [
        k + 1 < len(lines)
        and bool(lines[k].strip())
        and re.fullmatch(r"\s*-{3,}\s*", lines[k + 1]) is not None
        for k in range(len(lines))
    ]
    para_indent = 0  # indent of the current paragraph's first line
    last_prose = -1  # index in `out` of the last prose line
    block_indent: int | None = None  # set while inside a literal block
    block_context = 0  # indent the block must exceed to continue
    list_indent: int | None = None  # text column of the current list item
    item_indent: int | None = None  # field-item indent inside a field section
    in_field = False
    prev_blank = True
    skip = 0
    for k, line in enumerate(lines):
        if skip:
            skip -= 1
            continue
        if not line.strip():
            out.append(line)
            prev_blank = True
            continue
        d = _indent(line)
        if is_header[k]:
            if not prev_blank:
                out.append("")
            out += [line, lines[k + 1]]
            skip = 1
            in_field = line.strip() in _FIELD_SECTIONS
            item_indent = None
            block_indent = list_indent = None
            para_indent, last_prose, prev_blank = d, -1, False
            continue
        if block_indent is not None:
            if d > block_context:  # still inside the block
                out.append(line)
                prev_blank = False
                continue
            block_indent = None
            if not prev_blank:
                out.append("")
            prev_blank = True
        if in_field and (item_indent is None or d <= item_indent):
            # A field item header such as " x : int"; napoleon parses it.
            item_indent = d
            out.append(_escape_pipes(line))
            para_indent, last_prose, list_indent, prev_blank = d, -1, None, False
            continue
        if in_field and para_indent == item_indent and d > item_indent:
            # First description line under a field item.
            out.append(_escape_pipes(line))
            para_indent, last_prose, prev_blank = d, len(out) - 1, False
            continue
        item = _LIST_ITEM.match(line.lstrip())
        if item is not None:
            if not prev_blank and list_indent is None:
                out.append("")
            list_indent = d + item.end()
            out.append(_escape_pipes(line))
            last_prose, prev_blank = len(out) - 1, False
            continue
        if list_indent is not None:
            if d == list_indent:
                out.append(_escape_pipes(line))  # item continuation
                last_prose, prev_blank = len(out) - 1, False
                continue
            if d < list_indent and not prev_blank and d <= para_indent:
                out.append("")  # the list ends
                prev_blank = True
            if d <= para_indent:
                list_indent = None
        context = list_indent if list_indent is not None else para_indent
        if d > context and last_prose >= 0:
            # A display block: make it a literal block.
            out[last_prose] = _mark_literal(out[last_prose])
            if not prev_blank:
                out.append("")
            block_indent, block_context = d, context
            out.append(line)
            prev_blank = False
            continue
        if prev_blank:
            para_indent = d
        out.append(_escape_pipes(line))
        last_prose, prev_blank = len(out) - 1, False
    return out


def _process_docstring(app, what, name, obj, options, lines) -> None:
    """Translate a docstring before napoleon sees it."""
    lines[:] = _translate_plain_text(lines)


def setup(app) -> None:
    # Before napoleon (priority 500), which needs the NumPy sections intact.
    app.connect("autodoc-process-docstring", _process_docstring, priority=400)


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"Cvxium {release}"
html_theme_options = {
    "github_url": "https://github.com/rwilson4/Cvxium",
    "show_prev_next": False,
}
