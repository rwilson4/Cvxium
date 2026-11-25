# Detecting Unbounded Problems

A minimization problem is *unbounded below* if, for every finite
objective level, some feasible point has a lower objective value.
For example, a model that rewards buying more of an asset may be
unbounded if the model omits a budget or capacity limit. When Cvxium
verifies a descending ray, it returns a result with status
`SolveStatus.UNBOUNDED` and a certificate that we can inspect.

## Solve a quadratic problem and inspect its ray

*Example 1.* Let $x=(x_1,x_2)$ and minimize
$-x_1 + x_2^2/2$ over $x_1,x_2\ge 0$ with $|x_2|\le 2$.
Increasing $x_1$ reduces the objective and leaves the quadratic term
and the norm constraint unchanged. We expect a certificate along
$d=(1,0)$.

```python
import numpy as np

from cvxium import (
    BarrierSolver,
    BoundsConstraintLayer,
    DiagonalSystem,
    NormConstraintLayer,
    PrimalDualSolver,
    Problem,
    QuadraticObjective,
    SolveStatus,
)

Q = DiagonalSystem(np.array([0.0, 1.0]))
c = np.array([-1.0, 0.0])
B = np.array([[0.0, 1.0]])
problem = Problem(
    obj=QuadraticObjective(c=c, hessian_contribution=Q),
    ineq=[
        BoundsConstraintLayer(lb=0.0, dimension=2),
        NormConstraintLayer(B=B, c=np.zeros(1), phi=2.0),
    ],
)

for solver in (BarrierSolver(), PrimalDualSolver()):
    result = solver.solve(problem, np.ones(2))
    assert result.status is SolveStatus.UNBOUNDED
    assert not result.succeeded

    certificate = result.certificate
    assert certificate is not None and certificate.kind == "unbounded"
    d = certificate.primal_witness
    assert d is not None
    np.testing.assert_array_equal(Q.multiply(d), np.zeros(2))
    np.testing.assert_array_equal(B @ d, np.zeros(1))
    assert np.all(d >= 0.0)
    assert np.dot(c, d) < 0.0
    assert problem.is_feasible(result.solution)
```

For these two solvers, `result.solution` is the iterate where Cvxium
found the ray. Its objective value describes that point, while the
infimum over the whole feasible set is $-\infty$. The useful output
for diagnosing the model is `certificate.primal_witness`. Its
coordinates show which variables can move together indefinitely.
Check `result.status` before using `result.solution` as an optimizer.
A dual value of `-inf` by itself means the reported dual bound provides
no finite lower bound.

## How Cvxium finds and verifies a ray

A *recession direction* $d$ preserves feasibility along $x+td$ for
every $t\ge 0$ from a feasible point $x$. To certify unboundedness,
Cvxium also needs the objective to tend to $-\infty$ along that ray.
It checks the model's structure over the entire ray. A negative
directional derivative at the current iterate could turn positive
farther along the direction.

| Model component | Ray check |
| --- | --- |
| Affine equality $Ax=b$ | $Ad=0$ |
| Affine inequality $Gx\le h$ | $Gd\le 0$. For example, $d_i\ge 0$ at a lower-bounded coordinate and $d_i=0$ at a coordinate with both bounds. |
| Norm bound $\lVert Bx-a\rVert_2\le r$ | $Bd=0$. The residual stays fixed. A squared-norm bound on all coordinates permits only $d=0$. |
| Linear objective $c^Tx$ | $c^Td<0$ proves descent to $-\infty$. |
| Convex quadratic objective $x^TQx/2+c^Tx$ | $Qd=0$ and $c^Td<0$. Any positive curvature along the ray eventually dominates the linear decrease. |

Lifted objectives and constraints apply their checks to the
coordinates they use. The solvers perform the checks in floating
point. In particular, the Newton-step verifier allows a residual
tolerance for affine equalities.

`BarrierSolver` looks for a run of centering steps that grow the
iterate geometrically. `PrimalDualSolver` checks candidate Newton
directions from feasible iterates. For supported low-rank quadratic
objectives, both can also search for a ray with an auxiliary linear
program. The search combines the objective's nullspace equations
with affine constraints, bounds, and supported norm equations. A
candidate from that search goes through verification against the
original model before Cvxium returns `UNBOUNDED`.

## Conic problems

`HsdSolver` works with a linear objective $c^Tx$, affine equalities
$Ax=b$, and membership in a product of cones $K$. It looks for
certificates through a homogeneous self-dual embedding. An improving
direction satisfies $d\in K$, $Ad=0$, and $c^Td<0$. From any feasible
point, this direction preserves feasibility and drives the objective
to $-\infty$.

The conic solver reports this direction under
`SolveStatus.UNBOUNDED` in `certificate.primal_witness`. The status
uses the embedding's termination and objective sign tests. The solver
does not independently recheck the ray equations. To confirm primal unboundedness, check
$d\in K$, $Ad=0$, and $c^Td<0$ against the original conic model and
establish that the model has a feasible point. The conic result's
`solution` on this status comes from the embedding and need not
provide that point.

## When a solve ends without a descending ray

- **Flat directions.** If a barrier centering step follows a ray
  along which the objective is constant, the centering problem can
  lack a minimizer even when the original objective is bounded below.
  `BarrierSolver` can diagnose this with a
  `RecessionDirectionError`, usually chained under
  `InteriorPointMethodError`. For example, minimizing $x_1$ over
  $x_1,x_2\ge 0$ has optimal value zero even though $x_2$ can grow
  without bound.
- **Sublinear descent.** Minimizing $-\sqrt{x}$ over $x\ge 0$ has
  infimum $-\infty$, but the limiting slope along $d=1$ is zero.
  Cvxium's descending-ray check does not certify this behavior.
- **Curved escape.** Minimizing $-x$ subject to $y\ge x^2$ has
  infimum $-\infty$ along $(x,y)=(t,t^2)$. Every feasible straight
  recession ray keeps $x$ fixed, so the search has no descending
  ray to find.

A failed line search, a singular Newton system, or a run that reaches
its iteration limit leaves boundedness unresolved by itself. If
Cvxium returns a witness for a model we expected to be bounded, we
can inspect which variables move along it and check whether the
application calls for a missing constraint or cost.
