"""Arm the solver-performance guard for the test suite."""

import os
from pathlib import Path

# Set before any test module imports cvxium, whose __init__ installs the guard when
# it sees this variable. setdefault, so a caller can point at another baseline or
# disable the guard with CVXIUM_PERF_BASELINE="".
os.environ.setdefault(
    "CVXIUM_PERF_BASELINE", str(Path(__file__).parent / "perf_baseline.json")
)
