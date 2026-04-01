r"""Optimization utilities.

Introduction
------------
This module provides a framework for building efficient Interior Point Method solvers
for convex optimization problems. The "special sauce" for making such methods fast
include:
- exploiting special structure in the Hessian to enable fast calculation of Newton
  steps
- initializing the barrier penalty intelligently
- early stopping for feasibility problems
- fast evaluation of gradients
- fast backtracking line search

An optimization problem has the form:
    minimize    f0(x)
    subject to  hj(x) = 0, j=1, ..., m
                fi(x) <= 0, i=1, ..., n

A convex optimization problem has fi convex, i=0, ..., n, and hj linear. In that case
the equality constraints are equivalent to hj(x) = 0 <-> A * x = b. Thus, this library
is built for problems like:
    minimize    f0(x)
    subject to  A * x = b
                fi(x) <= 0, i=1, ..., n

Interior point methods solve problems with inequality constraints by incorporating them
into the objective with a barrier function, typically a logarithmic barrier function:
    minimize    ft(x) := t * f0(x) - \sum_{i=1}^n log(-fi(x))
    subject to  A * x = b.
The parameter t is called the barrier multiplier and ft the barrier objective. For large
values of t, the barrier problem is a good approximation to the desired problem, but
will typically be hard to solve directly. Instead, we solve the problem first for a
small value of t, then increase t gradually, iteratively, to a large value. This is the
essence of the Interior Point Method.

Solving such problems typically requires a strictly feasible point, x0, satisfying
A * x0 = b and fi(x0) < 0, i=1, ..., n. Finding a strictly feasible point is sometimes
called a Phase I method, which itself often involves solving a convex optimization
problem. The code defines an InteriorPointMethodSolver class and a
PhaseIInteriorPointMethodSolver class; the main difference is that once a
PhaseIInteriorPointMethodSolver finds a feasible point, we can stop without "optimizing
fully".

There is also an EqualityConstrainedInteriorPointMethodSolver intended to be used
whenever we have equality constraints. The main reason to do so is a custom
initialization of the barrier penalty, which can result in faster solution. If this
custom initialization is not desired, it may be better to inherit from
InteriorPointMethodSolver; that is, not all problems with equality constraints need
inherit from EqualityConstrainedInteriorPointMethodSolver.

For general information on convex optimization and interior point method solvers, see
(Boyd and Vandenberghe, 2004).

Usage
-----
To use this framework, create a class that inherits from one of these base classes. The
main methods you'll need to implement are:
- calculate_newton_step (most important for fast solvers, discussed more below)
- evaluate_objective
- hessian_multiply
- constraints
- grad_constraints
- evaluate_dual (discussed more below)

After you get an initial implementation, you likely also want to override
- grad_constraints_multiply
- grad_constraints_transpose_multiply
- btls_keep_feasible (discussed more below)
Base implementations are available, but custom implementations can make the code much
faster.

Have a look at the docstrings for these methods to figure out what they should do. A few
of these methods warrant further discussion.

- calculate_newton_step
  - This is the main "special sauce" that leads to an efficient solver. The matrix of
    second derivatives of the barrier objective is called the Hessian. The Hessian often
    has special structure that facilitates solving H * y = b. For example, when H is
    diagonal (with entries eta), we can calculate y as b / eta, which runs in linear
    time. Solving the system without exploiting this special structure generally takes
    cubic time, so exploiting special structure can achieve a quadratic speed up, which
    makes a material difference with large problem dimensions. It's not unusual to speed
    up solvers by a factor of thousands or millions depending on the problem dimension.
  - To facilitate efficient implementations, we provide some "numerical helpers":
    - solve_kkt_system (for problems with equality constraints)
    - solve_diagonal
    - solve_rank_p_update
    - solve_rank_one_update (special case of solve_rank_p_update with p=1)
    - solve_with_schur
    - solve_block_plus_one (special case of solve_with_schur with p=1)
    - solve_arrow_sparsity_pattern (special case of solve_block_plus_one)
  - These helpers are designed to be modular. A particular Hessian might be a low-rank
    update to an arrow sparsity pattern. By properly nesting these helpers, we can avoid
    cubic run times.
- evaluate_dual
  - This function is especially important for Phase I Solvers, since the dual problem
    can identify when a problem is infeasible. The Lagrangian is:
      L(x, nu, lambda) = f0(x) + \sum_{j=1}^m nu_j * hj(x)
                               + \sum_{i=1}^n lambda_i * fj(x).
  - The Lagrangian dual function, g(nu, lambda), is the infimum (over x) of the
    Lagrangian, evaluated at nu, lambda.
  - The dual function is always a lower bound on the problem objective, so it can often
    act as a "certificate" that the problem is infeasible. It can also be used to assess
    the precision of the solution.
- btls_keep_feasible
  - This function returns the largest step size that keeps the next iterate feasible. A
    base implementation is available using an iterative method, but we can often
    calculate the largest step size with some clever math. This will tend to result in
    more effective solvers.

References
----------
- Boyd, Stephen and Vandenberghe, Lieven, Convex Optimization, Cambridge University
  Press, 2004.

"""

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
from .linear_programs import (
    EqualitySolver,
    EqualityWithBoundsAndImbalanceConstraintSolver,
    EqualityWithBoundsSolver,
)
from .numerical_helpers import (
    solve_arrow_sparsity_pattern,
    solve_block_plus_one,
    solve_diagonal,
    solve_diagonal_eta_inverse,
    solve_kkt_system,
    solve_rank_one_update,
    solve_rank_p_update,
    solve_with_schur,
)
from .optimization import (
    EqualityConstrainedInteriorPointMethodSolver,
    InteriorPointMethodResult,
    InteriorPointMethodSolver,
    NewtonResult,
    OptimizationResult,
    OptimizationSettings,
    Optimizer,
    PhaseIInteriorPointSolver,
    PhaseISolver,
    ProblemCertifiablyInfeasibleError,
    ProblemMarginallyFeasibleError,
    UnconstrainedNewtonSolver,
)
from .quadratic_programs import (
    QuadraticNewtonSolver,
    QuadraticProgramEqualityBoundsSolver,
)

__all__ = [
    "BacktrackingLineSearchError",
    "CenteringStepError",
    "ConstraintBoundaryError",
    "EqualityConstrainedInteriorPointMethodSolver",
    "EqualitySolver",
    "EqualityWithBoundsAndImbalanceConstraintSolver",
    "EqualityWithBoundsSolver",
    "InteriorPointMethodError",
    "InteriorPointMethodResult",
    "InteriorPointMethodSolver",
    "InvalidDescentDirectionError",
    "NewtonResult",
    "NewtonStepError",
    "OptimizationError",
    "OptimizationResult",
    "OptimizationSettings",
    "Optimizer",
    "PhaseIInteriorPointSolver",
    "PhaseISolver",
    "ProblemCertifiablyInfeasibleError",
    "ProblemInfeasibleError",
    "ProblemMarginallyFeasibleError",
    "QuadraticNewtonSolver",
    "QuadraticProgramEqualityBoundsSolver",
    "SevereCurvatureError",
    "UnconstrainedNewtonSolver",
    "solve_arrow_sparsity_pattern",
    "solve_block_plus_one",
    "solve_diagonal",
    "solve_diagonal_eta_inverse",
    "solve_kkt_system",
    "solve_rank_one_update",
    "solve_rank_p_update",
    "solve_with_schur",
]
