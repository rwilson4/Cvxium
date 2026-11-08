"""Tests for the public solve_second_order_cone_program function.

Solves the SOCP `min f^T x  s.t. ||A_soc x + b_soc|| <= c_soc^T x + d_soc, F x = g, x
>= 0` via the Andersen-Roos-Terlaky homogeneous self-dual (HSD) embedding,
dispatched to `HsdSolver`.

"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np
from parameterized import parameterized

from cvxium.solvers.second_order_cone_programs import (
    solve_second_order_cone_program,
)
from cvxium.solvers.settings import (
    OptimizationSettings,
    OuterSettings,
)


def _make_feasible_socp(
    rng: np.random.Generator,
    n: int,
    m: int,
    p: int,
) -> dict[str, Any]:
    """Construct a random feasible SOCP instance.

    `x0 = ones(n)` is near the analytic center, the SOC margin gamma at x0 is
    `n^2`, the objective is positive (so the bound-barrier analytic center is
    well-defined and positive at every t), and entries are scaled to keep all
    matrix-vector products `O(1)` at x0.

    """
    A_soc = rng.standard_normal((m, n)) / math.sqrt(n)
    b_soc = rng.standard_normal(m) / math.sqrt(m)
    c_soc = rng.standard_normal(n) / math.sqrt(n)

    x0 = np.ones(n)
    v0 = A_soc @ x0 + b_soc
    norm_v0 = float(np.linalg.norm(v0))
    u_target = math.sqrt(n * n + norm_v0 * norm_v0)
    d_soc = u_target - float(c_soc @ x0)

    F = rng.standard_normal((p, n)) / math.sqrt(n)
    g_eq = F @ x0

    f = (0.5 + rng.random(n)) / math.sqrt(n)

    return {
        "f": f,
        "A_soc": A_soc,
        "b_soc": b_soc,
        "c_soc": c_soc,
        "d_soc": d_soc,
        "F": F,
        "g_eq": g_eq,
    }


class SolveSecondOrderConeProgramTest(unittest.TestCase):
    @parameterized.expand(
        [
            (4001, 5, 3, 1),
            (4002, 8, 4, 2),
            (4003, 12, 5, 3),
        ]
    )
    def test_converges_and_satisfies_constraints(
        self, seed: int, n: int, m: int, p: int
    ) -> None:
        """`solve_second_order_cone_program` returns a feasible iterate at a finite optimum."""
        rng = np.random.default_rng(seed)
        prob = _make_feasible_socp(rng, n, m, p)
        result = solve_second_order_cone_program(**prob)

        x = result.solution
        # Solution shape (HSD slacks stripped).
        assert x.shape == (n,)
        # Bounds.
        assert (x >= -1e-8).all()
        # Equality.
        np.testing.assert_allclose(prob["F"] @ x, prob["g_eq"], rtol=1e-4, atol=1e-4)
        # SOC.
        v = prob["A_soc"] @ x + prob["b_soc"]
        u = float(prob["c_soc"] @ x + prob["d_soc"])
        assert float(np.linalg.norm(v)) <= u + 1e-5

    def test_settings_are_forwarded(self) -> None:
        """User-supplied `settings` reach the solver."""
        rng = np.random.default_rng(4031)
        prob = _make_feasible_socp(rng, n=6, m=3, p=1)
        # Generous soft tolerance so neither solve raises -- this test is about
        # settings forwarding (more aspirational tolerance => at least as many outer
        # iterations), not about hitting an (unreachable) absolute gap.
        tight = OptimizationSettings(
            outer=OuterSettings(tolerance=1e-8, tolerance_soft=1.0)
        )
        loose = OptimizationSettings(
            outer=OuterSettings(tolerance=1e-2, tolerance_soft=1.0)
        )

        tight_result = solve_second_order_cone_program(settings=tight, **prob)
        loose_result = solve_second_order_cone_program(settings=loose, **prob)

        # Tight tolerance should require at least as many outer iterations as loose.
        assert tight_result.n_iterations >= loose_result.n_iterations

    @parameterized.expand(
        [
            (4101, 5, 3, 1),
            (4102, 8, 4, 2),
        ]
    )
    def test_hsd_embedding_returns_feasible_solution(
        self, seed: int, n: int, m: int, p: int
    ) -> None:
        """The HSD embedding returns an `n`-dim solution satisfying all constraints.

        The HSD path embeds the SOCP into `Nonneg(n) x SOC(m+1)` with slack
        variables and dispatches to `HsdSolver`. Validates the solution shape
        (slacks stripped) and feasibility on bounds, equality, and SOC.
        """
        rng = np.random.default_rng(seed)
        prob = _make_feasible_socp(rng, n, m, p)
        result = solve_second_order_cone_program(**prob)

        x = result.solution
        assert x.shape == (n,)
        assert (x >= -1e-6).all()
        np.testing.assert_allclose(prob["F"] @ x, prob["g_eq"], rtol=1e-4, atol=1e-4)
        v = prob["A_soc"] @ x + prob["b_soc"]
        u = float(prob["c_soc"] @ x + prob["d_soc"])
        assert float(np.linalg.norm(v)) <= u + 1e-5

    def test_a_soc_diag_hint_agrees_with_runtime_detection(self) -> None:
        """Passing `A_soc_diag` matches the runtime-detected structured path.

        Builds a SOCP with `A_soc = [diag(d) | 0]` so the runtime detection
        (`A_soc_diag=None`) and the explicit hint route the same structured
        embedding; the solutions must agree.
        """
        rng = np.random.default_rng(4021)
        n, m, p = 6, 4, 2
        diag = 0.5 + rng.random(m)
        A_soc = np.zeros((m, n))
        A_soc[:, :m] = np.diag(diag)
        b_soc = rng.standard_normal(m) / math.sqrt(m)
        c_soc = rng.standard_normal(n) / math.sqrt(n)

        x0 = np.ones(n)
        v0 = A_soc @ x0 + b_soc
        norm_v0 = float(np.linalg.norm(v0))
        u_target = math.sqrt(n * n + norm_v0 * norm_v0)
        d_soc = u_target - float(c_soc @ x0)

        F = rng.standard_normal((p, n)) / math.sqrt(n)
        g_eq = F @ x0
        f = (0.5 + rng.random(n)) / math.sqrt(n)

        detected = solve_second_order_cone_program(
            f=f,
            A_soc=A_soc,
            b_soc=b_soc,
            c_soc=c_soc,
            d_soc=d_soc,
            F=F,
            g_eq=g_eq,
        )
        hinted = solve_second_order_cone_program(
            f=f,
            A_soc=A_soc,
            b_soc=b_soc,
            c_soc=c_soc,
            d_soc=d_soc,
            F=F,
            g_eq=g_eq,
            A_soc_diag=diag,
        )
        np.testing.assert_allclose(
            hinted.solution, detected.solution, rtol=1e-5, atol=1e-5
        )
