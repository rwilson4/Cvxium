"""Run the code examples in README.md, USAGE.md, and the user guides."""

import contextlib
import io
import re
import unittest
from pathlib import Path

import matplotlib

_ROOT = Path(__file__).resolve().parents[1]


def _python_blocks(path: Path) -> str:
    """Join a Markdown file's python code blocks into one script."""
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    return "\n".join(blocks)


class DocumentationExamplesTest(unittest.TestCase):
    """Each guide's python blocks run, in order, as one script."""

    @classmethod
    def setUpClass(cls) -> None:
        matplotlib.use("Agg")

    def _run(self, name: str, replacements: dict[str, str] | None = None) -> None:
        script = _python_blocks(_ROOT / name)
        self.assertTrue(script, f"{name} has no python blocks")
        for old, new in (replacements or {}).items():
            self.assertIn(old, script, f"{name} no longer contains {old!r}")
            script = script.replace(old, new)
        # Silence the guides' prints, including the whole of cvxium.usage().
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(script, name, "exec"), {"__name__": f"docs.{name}"})

    def test_readme_examples_run(self) -> None:
        self._run("README.md")

    def test_usage_guide_examples_run(self) -> None:
        self._run("USAGE.md")

    def test_linear_programs_guide_examples_run(self) -> None:
        self._run("docs/user_guides/linear_programs.md")

    def test_banded_quadratic_programs_guide_examples_run(self) -> None:
        # At the guide's M = 100,000, cvxpy's compilation allocates a dense
        # (M + 1) x (M + 1) array (75 GB), which Linux CI runners refuse.
        self._run(
            "docs/user_guides/banded_quadratic_programs.md",
            {"M, lam = 100_000, 50.0": "M, lam = 10_000, 50.0"},
        )


if __name__ == "__main__":
    unittest.main()
