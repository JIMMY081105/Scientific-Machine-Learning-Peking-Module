# 2.5 Second Order Methods (Newton, coded from scratch)

**Problem.** Build Newton's method by hand using the second derivative (Hessian), see why it converges so fast, and understand its danger: it seeks any zero-gradient point, so it can converge to a maximum.

Split by dimensionality:

| File | Goal / strategy |
| --- | --- |
| `01_Newton1D_AndPitfall.jl` | 1-D Newton on (x^3 - x)^2: Taylor derivation, fast convergence vs Nesterov, and the maximum-seeking pitfall. |
| `02_NewtonND_QuadraticHimmelblau.jl` | N-D Newton (x <- x - inv(H)*g) on the Shewchuk quadratic and Himmelblau. |
