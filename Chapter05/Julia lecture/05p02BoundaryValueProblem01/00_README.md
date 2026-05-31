# 5.2 Boundary Value Problem (Another Example)

**Problem.** Solve a fuller linear BVP with both first- and second-derivative terms: y'' + (2/x) y' - (2/x^2) y = 0 on [1,2] with y(1)=5, y(2)=3, checked against the exact solution y = x + 4/x^2.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_DenseFiniteDifference.jl` | Assemble a dense finite-difference matrix and solve with backslash; compare to the exact solution. |
| `02_SparseSolveSpeedup.jl` | Rebuild A as a sparse matrix and solve with LinearSolve (UMFPACK) for the speed-up. |
