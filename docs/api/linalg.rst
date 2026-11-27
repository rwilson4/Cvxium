Structured Linear Algebra
=========================

Matrix Systems
--------------

.. automodule:: cvxium.linalg.systems
   :members: MatrixSystem, ZeroSystem, DiagonalSystem, BandedSystem, DenseSystem, BlockDiagonalSystem, PermutedSystem, ArrowSystem, Definiteness

Saddle Systems
--------------

.. currentmodule:: cvxium.linalg.systems

.. autoclass:: SaddleSystem
   :members:

.. autoclass:: SchurComplementSaddleSystem
   :members:

.. autoclass:: SparseSaddleSystem
   :members:

.. autoclass:: BandedSchurSaddleSystem
   :members:

.. autoclass:: ArrowSaddleSystem
   :members:

.. autoclass:: SaddleFactory
   :members:

.. autoclass:: SchurComplementSaddleFactory
   :members:

.. autoclass:: SparseSaddleFactory
   :members:

.. autoclass:: BandedSaddleFactory
   :members:

.. autoclass:: ArrowSaddleFactory
   :members:


Numerical Kernels
-----------------

.. automodule:: cvxium.linalg.numerical_helpers
   :members: solve_diagonal, solve_diagonal_eta_inverse, solve_banded, solve_block_diagonal, solve_rank_one_update, solve_rank_p_update, solve_with_schur, solve_block_plus_one, solve_arrow_sparsity_pattern, solve_kkt_system, multiply_diagonal, multiply_banded, multiply_block, multiply_block_diagonal, multiply_rank_one_update, multiply_rank_p_update, multiply_block_plus_one, multiply_arrow_sparsity_pattern
