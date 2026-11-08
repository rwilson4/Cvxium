"""Cvxium: a framework for fast interior-point solvers for convex optimization.

Introduction
------------
This package provides a framework for building efficient Interior Point Method solvers
for convex optimization problems. The "special sauce" for making such methods fast
includes:
- exploiting special structure in the Hessian to enable fast calculation of Newton
  steps
- initializing the barrier penalty intelligently
- early stopping for feasibility problems
- fast evaluation of gradients
- fast backtracking line search

An optimization problem has the form
   minimize    f0(x)
   subject to  hj(x) = 0,  j = 1, ..., m
               fi(x) ≤ 0,  i = 1, ..., n.

A convex optimization problem has fi convex, i = 0, ..., n, and hj affine. In that case
the equality constraints are equivalent to A x = b, so this library is built for
problems like
   minimize    f0(x)
   subject to  A x = b
               fi(x) ≤ 0,  i = 1, ..., n.

Interior point methods solve problems with inequality constraints by incorporating them
into the objective with a barrier function, typically a logarithmic barrier:
   minimize    ft(x) := t f0(x) - ∑_{i=1}^n log(-fi(x))
   subject to  A x = b.
The parameter t is called the barrier multiplier and ft the barrier objective. For large
values of t, the barrier problem is a good approximation to the desired problem, but it
is typically hard to solve directly. Instead, we solve the problem first for a small
value of t, then increase t gradually, iteratively, to a large value. This is the
essence of the Interior Point Method.

Solving such problems requires a strictly feasible point x0, satisfying
   A x0 = b,   fi(x0) < 0,  i = 1, ..., n.
Finding one is itself a convex optimization problem (Phase I), which the solvers build
and solve automatically when no such x0 is supplied.

Problems and solvers
--------------------
Cvxium separates the statement of a problem from the algorithm that solves it. A
``Problem`` holds an ``Objective`` (or ``None`` for a feasibility problem), a list of
inequality ``ConstraintLayer`` objects, and an optional ``EqualityConstraintLayer``
(A x = b). A ``Solver`` solves it: ``BarrierSolver`` (the barrier method),
``PrimalDualSolver`` (the primal-dual method), ``NewtonSolver`` (no inequalities), or
``HsdSolver`` (linear conic programs, by the homogeneous self-dual method). Each returns
a ``SolveResult``:
   problem = Problem.minimize(objective).subject_to(eq=eq, ineq=[bounds, layer])
   result = problem.solve(solver=BarrierSolver())
   result.solution, result.objective_value, result.inequality_multipliers

How structure is exploited
--------------------------
Each component reports its second derivatives as a ``MatrixSystem``, a structured
matrix that can solve and multiply itself: ``DiagonalSystem``, ``BandedSystem``,
``DenseSystem``, ``BlockDiagonalSystem``, ``PermutedSystem``, ``ArrowSystem``, any of
them carrying low-rank updates (``MatrixSystem.low_rank_update``). The objective reports
∇^2 f0 through ``hessian_contribution``; each layer reports its weighted curvature
atoms
   ∑ λ_i C_i,   C_i = ∇^2 fi - (1 / fi) ∇fi ∇fi^T,
through ``hessian_contribution`` and its curvature ∑ λ_i ∇^2 fi alone through
``hessian_curvature``. ``Problem.build_hessian_system`` sums the contributions with
``MatrixSystem.plus``, which keeps their structure (diagonal plus diagonal stays
diagonal, and low-rank updates ride on top of any base) or raises
``IncompatibleStructureError`` rather than silently densifying. One contract serves
both methods: the barrier method weights with t and λ_i = -1 / fi, giving ∇^2 ft; the
primal-dual method with 1 and its duals.

With equality constraints, each Newton step solves an indefinite KKT system, which a
``SaddleSystem`` solves through the Schur complement A H^{-1} A^T, touching H only
through ``H.solve``. The ``EqualityConstraintLayer``'s injected ``SaddleFactory``
chooses it: ``SchurComplementSaddleFactory`` (the default, for a dense A),
``SparseSaddleFactory`` (a sparse A, never densified), ``BandedSaddleFactory`` (rows
ordered so the Schur complement is banded), or ``ArrowSaddleFactory`` (a banded core
plus a few dense rows).

Extending Cvxium
----------------
Library components cover linear and quadratic objectives and bound, linear, imbalance,
and norm constraints. A new objective subclasses ``Objective`` and implements
``dimension``, ``evaluate``, ``gradient``, and ``hessian_contribution``. A new layer
subclasses ``SlackConstraintLayer`` (Phase I through a slack variable; works for any
layer) or ``DirectConstraintLayer`` (a single constraint that serves as its own Phase I
objective) and implements ``dimension``, ``num_constraints``, ``constraints``,
``grad_constraints``, ``hessian_contribution``, and ``hessian_curvature``; a
closed-form ``btls_max_step`` and structured Jacobian products make it fast. The
solvers evaluate the dual, build Phase I, and initialize the barrier parameter
(``OuterSettings.barrier_init``) without further code.

``usage()`` prints the full guide, USAGE.md, which works one example through from the
mathematics to a fast solve, and covers caching, logging, Pareto frontiers
(``MultiObjectiveOptimizer``), conic programs, and saddle-point problems
(``StationarityProblem``).

References
----------
- Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
  Press, 2004.

"""

import logging
import os
import pathlib
from importlib.resources import files as _resource_files

from .diagnostics.handlers import (
    IterationHandler,
    enable_logging,
)
from .diagnostics.trace import (
    DataFrameTrace,
    ListTrace,
    LoggingTrace,
    Trace,
)
from .exceptions import (
    BacktrackingLineSearchError,
    CenteringStepError,
    ConstraintBoundaryError,
    InteriorPointMethodError,
    InvalidDescentDirectionError,
    NewtonStepError,
    OptimizationError,
    ProblemInfeasibleError,
    SevereCurvatureError,
)
from .frontier.frontier_interpolation import (
    FrontierEnvelope,
    interpolate_frontier,
)
from .frontier.multi_objective import (
    FrontierPoint,
    FrontierResults,
    MultiObjectiveOptimizer,
)
from .linalg.numerical_helpers import (
    multiply_arrow_sparsity_pattern,
    multiply_banded,
    multiply_block,
    multiply_block_diagonal,
    multiply_block_plus_one,
    multiply_diagonal,
    multiply_rank_one_update,
    multiply_rank_p_update,
    solve_arrow_sparsity_pattern,
    solve_banded,
    solve_block_diagonal,
    solve_block_plus_one,
    solve_diagonal,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_one_update,
    solve_rank_p_update,
    solve_with_schur,
)
from .linalg.systems import (
    ArrowSaddleFactory,
    ArrowSaddleSystem,
    ArrowSystem,
    BandedSaddleFactory,
    BandedSchurSaddleSystem,
    BandedSystem,
    BlockDiagonalSystem,
    Definiteness,
    DenseSystem,
    DiagonalSystem,
    IncompatibleStructureError,
    MatrixSystem,
    PermutedSystem,
    SaddleFactory,
    SaddleSystem,
    SchurComplementSaddleFactory,
    SchurComplementSaddleSystem,
    SparseSaddleFactory,
    SparseSaddleSystem,
    ZeroSystem,
)
from .problems.cone import (
    Cone,
    DenseNtScaling,
    NtScaling,
    SecondOrderCone,
)
from .problems.constraint_layers import (
    ConstraintLayer,
    DirectConstraintLayer,
    SlackConstraintLayer,
)
from .problems.linear_constraints import (
    ArrowLinearInequalityLayer,
    BandedLinearInequalityLayer,
    BoundsConstraintLayer,
    EqualityConstraintLayer,
    ImbalanceConstraintLayer,
    LinearInequalityConstraintLayer,
)
from .problems.objective import (
    LiftedObjective,
    LinearObjective,
    NegatedObjective,
    Objective,
    ObjectivePoint,
    QuadraticObjective,
)
from .problems.problem import Problem, Sense
from .problems.quadratic_constraints import (
    NormConstraintLayer,
    ObjectiveConstraintLayer,
    SquaredNormConstraintLayer,
)
from .problems.result import (
    BarrierDiagnostics,
    Duals,
    HsdDiagnostics,
    InfeasibilityCertificate,
    NoDualsError,
    NoDualValueError,
    NoObjectiveError,
    ObjectiveOutcome,
    PrimalDualDiagnostics,
    SaddleBarrierDiagnostics,
    SolverDiagnostics,
    SolveResult,
    SolveStatus,
    TwoBlockSolveResult,
    WrongDiagnosticsError,
)
from .saddle.primal_dual_stationarity_solver import (
    PrimalDualStationaritySolver,
)
from .saddle.stationarity import (
    QuadraticSaddleObjective,
    StationarityConvergenceError,
    StationarityProblem,
    TwoBlockObjective,
    TwoBlockStationaritySolver,
)
from .saddle.stationarity_solver import (
    StationaritySolver,
)
from .solvers.barrier_solver import (
    BarrierSolver,
)
from .solvers.hsd_solver import HsdSolver
from .solvers.newton_solver import NewtonSolver
from .solvers.optimization import (
    EqualityConstrainedNewtonSolver,
    FeasibilityInteriorPointSolver,
    FeasibilitySolver,
    InteriorPointMethodSolver,
    Optimizer,
    ProblemCertifiablyInfeasibleError,
    ProblemMarginallyFeasibleError,
    TrivialFeasibilitySolver,
    UnconstrainedNewtonSolver,
)
from .solvers.phase1 import (
    LiftedConstraintLayer,
)
from .solvers.primal_dual_solver import (
    PrimalDualSolver,
)
from .solvers.second_order_cone_programs import (
    solve_second_order_cone_program,
)
from .solvers.settings import (
    ArmijoSettings,
    BarrierInitStrategy,
    InnerSettings,
    LevenbergSettings,
    LineSearchSettings,
    ObservabilitySettings,
    OptimizationSettings,
    OuterSettings,
    PrimalDualSettings,
)
from .solvers.solver import (
    EqualitySolver,
    Solver,
)

logging.getLogger("cvxium").addHandler(logging.NullHandler())
# Stdlib library convention: silent by default. Users opt in via
# OptimizationSettings(verbose=True) or by lowering this level themselves.
# Prevents INFO/DEBUG records from leaking through a host application's root
# handler (e.g., Jupyter notebooks).
#
# Only when nobody has chosen a level yet. A caller may configure the
# namespace BEFORE importing this module -- `enable_logging` only needs the
# logger's name, not the library -- and an unconditional setLevel here would
# silently undo that on import, turning the log they asked for back off.
if logging.getLogger("cvxium").level == logging.NOTSET:
    logging.getLogger("cvxium").setLevel(logging.WARNING)


def usage() -> None:
    """Print USAGE.md to stdout."""
    try:
        text = (
            _resource_files("cvxium").joinpath("USAGE.md").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        # A source checkout keeps USAGE.md at the repository root; the wheel
        # force-includes it inside the package.
        text = (pathlib.Path(__file__).parent.parent / "USAGE.md").read_text(
            encoding="utf-8"
        )
    print(text)


__all__ = [
    "ArmijoSettings",
    "ArrowLinearInequalityLayer",
    "ArrowSaddleFactory",
    "ArrowSaddleSystem",
    "ArrowSystem",
    "BacktrackingLineSearchError",
    "BandedLinearInequalityLayer",
    "BandedSaddleFactory",
    "BandedSchurSaddleSystem",
    "BandedSystem",
    "BarrierDiagnostics",
    "BarrierInitStrategy",
    "BarrierSolver",
    "BlockDiagonalSystem",
    "BoundsConstraintLayer",
    "CenteringStepError",
    "Cone",
    "ConstraintBoundaryError",
    "ConstraintLayer",
    "DataFrameTrace",
    "Definiteness",
    "DenseNtScaling",
    "DenseSystem",
    "DiagonalSystem",
    "DirectConstraintLayer",
    "Duals",
    "EqualityConstrainedNewtonSolver",
    "EqualityConstraintLayer",
    "EqualitySolver",
    "FeasibilityInteriorPointSolver",
    "FeasibilitySolver",
    "FrontierEnvelope",
    "FrontierPoint",
    "FrontierResults",
    "HsdDiagnostics",
    "HsdSolver",
    "ImbalanceConstraintLayer",
    "IncompatibleStructureError",
    "InfeasibilityCertificate",
    "InnerSettings",
    "InteriorPointMethodError",
    "InteriorPointMethodSolver",
    "InvalidDescentDirectionError",
    "IterationHandler",
    "LevenbergSettings",
    "LiftedConstraintLayer",
    "LiftedObjective",
    "LineSearchSettings",
    "LinearInequalityConstraintLayer",
    "LinearObjective",
    "ListTrace",
    "LoggingTrace",
    "MatrixSystem",
    "MultiObjectiveOptimizer",
    "NegatedObjective",
    "NewtonSolver",
    "NewtonStepError",
    "NoDualValueError",
    "NoDualsError",
    "NoObjectiveError",
    "NormConstraintLayer",
    "NtScaling",
    "Objective",
    "ObjectiveConstraintLayer",
    "ObjectiveOutcome",
    "ObjectivePoint",
    "ObservabilitySettings",
    "OptimizationError",
    "OptimizationSettings",
    "Optimizer",
    "OuterSettings",
    "PermutedSystem",
    "PrimalDualDiagnostics",
    "PrimalDualSettings",
    "PrimalDualSolver",
    "PrimalDualStationaritySolver",
    "Problem",
    "ProblemCertifiablyInfeasibleError",
    "ProblemInfeasibleError",
    "ProblemMarginallyFeasibleError",
    "QuadraticObjective",
    "QuadraticSaddleObjective",
    "SaddleBarrierDiagnostics",
    "SaddleFactory",
    "SaddleSystem",
    "SchurComplementSaddleFactory",
    "SchurComplementSaddleSystem",
    "SecondOrderCone",
    "Sense",
    "SevereCurvatureError",
    "SlackConstraintLayer",
    "SolveResult",
    "SolveStatus",
    "Solver",
    "SolverDiagnostics",
    "SparseSaddleFactory",
    "SparseSaddleSystem",
    "SquaredNormConstraintLayer",
    "StationarityConvergenceError",
    "StationarityProblem",
    "StationaritySolver",
    "Trace",
    "TrivialFeasibilitySolver",
    "TwoBlockObjective",
    "TwoBlockSolveResult",
    "TwoBlockStationaritySolver",
    "UnconstrainedNewtonSolver",
    "WrongDiagnosticsError",
    "ZeroSystem",
    "enable_logging",
    "interpolate_frontier",
    "multiply_arrow_sparsity_pattern",
    "multiply_banded",
    "multiply_block",
    "multiply_block_diagonal",
    "multiply_block_plus_one",
    "multiply_diagonal",
    "multiply_rank_one_update",
    "multiply_rank_p_update",
    "solve_arrow_sparsity_pattern",
    "solve_banded",
    "solve_block_diagonal",
    "solve_block_plus_one",
    "solve_diagonal",
    "solve_diagonal_eta_inverse",
    "solve_kkt_system",
    "solve_rank_one_update",
    "solve_rank_p_update",
    "solve_second_order_cone_program",
    "solve_with_schur",
    "usage",
]


# Arm the solver-performance guard when the test suite asks for it. A parallel
# runner may split test modules across processes, so a guard installed from a
# test module reaches only some of them; keying off the environment here reaches
# every process that imports Cvxium. The env lookup happens before importing
# perf_guard (rather than calling perf_guard.install_from_environment()
# directly), so a production import -- where the variable is unset -- costs
# only the dict lookup and does not also pull in perf_guard's unittest,
# fcntl, atexit and json imports. The name below must match perf_guard's
# `_ENV_BASELINE`.
if os.environ.get("CVXIUM_PERF_BASELINE"):
    from .diagnostics import (
        perf_guard as _perf_guard,
    )

    _perf_guard.install_from_environment()
