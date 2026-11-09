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

.. toctree::
   :maxdepth: 1

   usage
   linear_programs
