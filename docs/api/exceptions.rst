Exceptions
==========

Solver Exceptions
-----------------

.. automodule:: cvxium.exceptions
   :members: OptimizationError, CenteringStepError, InteriorPointMethodError, ProblemInfeasibleError, NewtonStepError, BacktrackingLineSearchError, ConstraintBoundaryError, InvalidDescentDirectionError, SevereCurvatureError

Infeasibility
-------------

.. currentmodule:: cvxium.solvers.optimization

.. autoexception:: ProblemCertifiablyInfeasibleError
   :members:

.. autoexception:: ProblemMarginallyFeasibleError
   :members:


Result Access
-------------

.. currentmodule:: cvxium.problems.result

.. autoexception:: NoObjectiveError
   :members:

.. autoexception:: NoDualsError
   :members:

.. autoexception:: NoDualValueError
   :members:

.. autoexception:: WrongDiagnosticsError
   :members:


Structure
---------

.. currentmodule:: cvxium.linalg.systems

.. autoexception:: IncompatibleStructureError
   :members:

