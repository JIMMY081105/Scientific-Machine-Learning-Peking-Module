# 4.3 Application of Forward Mode Automatic Differentiation

**Problem.** Use the hand-built Dual-number AD from 4.2 to solve real problems, so you trust it does the same job as the libraries. Each file repeats the extended `Dual` definition so it runs on its own.

This lesson is split by goal/strategy:

| File | Goal / strategy |
| --- | --- |
| `01_DualNewtonRaphson.jl` | Newton-Raphson for dA/dr = 0, taking value and derivative from one Dual evaluation. |
| `02_DualNonlinearSystem.jl` | Assemble a Jacobian from Dual evaluations and iterate Newton-Raphson on a 2x2 system. |
| `03_DualRegression.jl` | Fit a constant a0 to data by gradient descent, with dS/da0 from a Dual evaluation. |
