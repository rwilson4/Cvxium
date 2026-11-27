User Guides
===========

The usage guide works one example through from start to finish:
calibrating survey weights, from the mathematics and a scipy baseline
to a fast Cvxium solve. Along the way it writes a new objective and a
new constraint layer, explains how Cvxium assembles Hessian structure,
and covers barrier initialization, caching, logging, Pareto frontiers,
and conic and saddle-point problems.

The guide to large linear programs solves a media plan with 20,000
variables and 50 constraints, compares Cvxium with scipy's ``linprog``,
and explains the structure and the settings behind the speedup.

The guide to banded quadratic programs smooths 100,000 nonnegative
power readings with a tridiagonal quadratic, compares Cvxium with a
steel-manned cvxpy across five solvers, and shows how a band structure
carries through every Newton step.

The guide to unbounded problems shows how to read a recession-ray
certificate and explains which problems the solvers can certify.

The guide to radiotherapy QPs takes a clinical treatment-planning QP
from RadOptLIB, compares Cvxium with Clarabel, and shows how to
capture a problem's own Hessian structure in a new ``MatrixSystem``
and ``ConstraintLayer``, and what that structure buys.

.. toctree::
   :maxdepth: 1

   usage
   linear_programs
   banded_quadratic_programs
   unbounded_problems
   radiotherapy_qps
