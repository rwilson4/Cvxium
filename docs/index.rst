Cvxium
======

Cvxium (pronounced "calcium") builds fast interior-point solvers for
convex optimization problems of the form

.. math::

   \begin{array}{ll}
   \mbox{minimize}   & f_0(w) \\
   \mbox{subject to} & A w = b \\
                     & f_i(w) \leq 0, \quad i = 1, \ldots, n.
   \end{array}

A generic solver factors an :math:`M \times M` dense matrix at every
Newton step, at a cost of :math:`O(M^3)`. Most problems that arise in
practice have far more structure than that: a diagonal Hessian, a
diagonal plus a low-rank update, a band, or an arrow pattern. Cvxium
carries that structure through every Newton step, so the same
iteration can cost :math:`O(M)`, which is the difference between a
problem with a thousand variables and one with a million.

The library provides:

- **Composable problems**: a :class:`~cvxium.Problem` holds an
  :class:`~cvxium.Objective`, a list of inequality
  :class:`~cvxium.ConstraintLayer` objects, and optional equality
  constraints, built fluently with ``Problem.minimize(...).subject_to(...)``.
- **Interior-point solvers**: the barrier method
  (:class:`~cvxium.BarrierSolver`), the primal-dual method
  (:class:`~cvxium.PrimalDualSolver`), Newton's method
  (:class:`~cvxium.NewtonSolver`), and a homogeneous self-dual method
  for conic programs (:class:`~cvxium.HsdSolver`), each finding its
  own strictly feasible start (Phase I) when none is given.
- **Structured linear algebra**: every component reports its Hessian
  as a :class:`~cvxium.MatrixSystem` (diagonal, banded, low-rank
  updated, block diagonal, arrow), and the solver sums the
  contributions without losing structure. Equality constraints go
  through a Schur-complement :class:`~cvxium.SaddleSystem`, dense,
  sparse, or banded.
- **Results that know themselves**: a :class:`~cvxium.SolveResult`
  carries the solution, duality-gap bound, Lagrange multipliers, and
  per-solver diagnostics, and round-trips through JSON.
- **Pareto frontiers**: :class:`~cvxium.MultiObjectiveOptimizer`
  traces the tradeoff between competing objectives, finds its knee,
  and certifies an interpolant with the multipliers it gets for free.

The methods follow [BV04]_; the conic solver follows [ART03]_. Full
citations are on the :doc:`references` page.

Installation
------------

.. code-block:: bash

    pip install cvxium

Quickstart
----------

Find the minimum-norm point satisfying :math:`A w = b` and
:math:`w \geq 0.01`. The objective :math:`\frac{1}{2} \| w \|_2^2` has a
diagonal Hessian and the bounds add another diagonal, so every Newton
step costs time linear in the number of variables:

.. doctest::

    >>> import numpy as np
    >>> from cvxium import (
    ...     BarrierSolver,
    ...     BoundsConstraintLayer,
    ...     DiagonalSystem,
    ...     EqualityConstraintLayer,
    ...     Problem,
    ...     QuadraticObjective,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> M = 100
    >>> A = rng.standard_normal((20, M))
    >>> b = A @ (rng.random(M) + 0.1)
    >>> objective = QuadraticObjective(
    ...     c=np.zeros(M), hessian_contribution=DiagonalSystem(np.ones(M))
    ... )
    >>> result = (
    ...     Problem.minimize(objective)
    ...     .subject_to(
    ...         eq=EqualityConstraintLayer(A=A, b=b),
    ...         ineq=[BoundsConstraintLayer(lb=0.01, dimension=M)],
    ...     )
    ...     .solve(solver=BarrierSolver())
    ... )
    >>> print(result.status)
    SolveStatus.OPTIMAL
    >>> bool(np.allclose(A @ result.solution, b)), bool(np.all(result.solution > 0.01))
    (True, True)
    >>> print(f"{result.objective_value:.4f}")
    4.1942

When no starting point is given, the solver finds a strictly feasible
one itself. The :doc:`user_guides/usage` works one example, from the
mathematics to a fast solve, through writing a new objective and a new
constraint layer; ``cvxium.usage()`` prints the same guide.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guides/index
   api/index
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
