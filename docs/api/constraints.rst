Constraint Layers
=================

Base Classes
------------

.. automodule:: cvxium.problems.constraint_layers
   :members: ConstraintLayer, SlackConstraintLayer, DirectConstraintLayer

Linear Constraints
------------------

.. automodule:: cvxium.problems.linear_constraints
   :members: EqualityConstraintLayer, BoundsConstraintLayer, LinearInequalityConstraintLayer, BandedLinearInequalityLayer, ArrowLinearInequalityLayer, ImbalanceConstraintLayer

Quadratic Constraints
---------------------

.. automodule:: cvxium.problems.quadratic_constraints
   :members: NormConstraintLayer, SquaredNormConstraintLayer, ObjectiveConstraintLayer

Lifted Constraints
------------------

.. automodule:: cvxium.solvers.phase1
   :members: LiftedConstraintLayer
