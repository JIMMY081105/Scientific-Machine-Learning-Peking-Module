# 4.4 Implicit Problems (Parameter Estimation): Linear Equations

**Problem.** Solve an inverse problem where the quantity of interest is defined implicitly by a linear system: find the input voltage V1 that makes the circuit current i5 equal zero. This needs derivatives taken THROUGH a linear solve.

This lesson is split by the two strategies, each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_CircuitSolveAndSweep.jl` | Assemble/solve the circuit system, then sweep V1 and plot i5 (and i5^2) vs V1. |
| `02_OptimizeThroughSolve.jl` | Minimise loss(V1) = i5^2 through the solve with ForwardDiff - hand-coded gradient descent, then Optimization.jl (Adam). |
