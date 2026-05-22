# 2.2 Basic Optimization in Julia Continued (second-order library methods)

**Problem.** Find minima with second-order library methods that use curvature (Hessian) information - Newton, BFGS, and LBFGS - which converge in very few iterations.

Split by test function (each file overlays all three solvers on that problem):

| File | Goal / strategy |
| --- | --- |
| `01_Sextic1D_NewtonBFGS.jl` | 1-D (x^3 - x)^2: Newton (~3 iters) vs BFGS vs LBFGS; different starts can find different minima. |
| `02_Quadratic2D_NewtonBFGS.jl` | 2-D Shewchuk quadratic with an AutoForwardDiff Hessian. |
| `03_Rosenbrock_NewtonBFGS.jl` | Rosenbrock: overlay paths and print sol.u/objective/retcode. |
