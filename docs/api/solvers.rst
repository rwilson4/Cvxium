Solvers
=======

Solver
------

.. automodule:: cvxium.solvers.solver
   :members: Solver, EqualitySolver

Barrier Method
--------------

.. automodule:: cvxium.solvers.barrier_solver
   :members: BarrierSolver

Primal-Dual Method
------------------

.. automodule:: cvxium.solvers.primal_dual_solver
   :members: PrimalDualSolver

Newton's Method
---------------

.. automodule:: cvxium.solvers.newton_solver
   :members: NewtonSolver

Homogeneous Self-Dual Method
----------------------------

.. automodule:: cvxium.solvers.hsd_solver
   :members: HsdSolver

Second-Order Cone Programs
--------------------------

.. automodule:: cvxium.solvers.second_order_cone_programs
   :members: solve_second_order_cone_program

Settings
--------

.. automodule:: cvxium.solvers.settings
   :members: OptimizationSettings, OuterSettings, InnerSettings, ArmijoSettings, LineSearchSettings, PrimalDualSettings, LevenbergSettings, ObservabilitySettings, BarrierInitStrategy

Subclassing Framework
---------------------

.. automodule:: cvxium.solvers.optimization
   :members: Optimizer, FeasibilitySolver, TrivialFeasibilitySolver, InteriorPointMethodSolver, FeasibilityInteriorPointSolver, UnconstrainedNewtonSolver, EqualityConstrainedNewtonSolver
