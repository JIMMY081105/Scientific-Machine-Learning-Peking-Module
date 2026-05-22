# 1.4 Solving Linear and Nonlinear Equations

**Problem.** Solve (a) linear systems A x = C and (b) single and coupled nonlinear equations f(x) = 0 - the root-finding machinery behind optimisation, since a minimum is a root of the derivative.

Split by equation type:

| File | Goal / strategy |
| --- | --- |
| `01_LinearEquations.jl` | Circuit system A i = C via backslash and LinearSolve. |
| `02_SingleNonlinearEquation.jl` | dA/dr = 0 by hand-coded Newton-Raphson and NonlinearSolve. |
| `03_NonlinearSystem.jl` | Two coupled equations via Newton-Raphson with the Jacobian and NonlinearSolve. |
