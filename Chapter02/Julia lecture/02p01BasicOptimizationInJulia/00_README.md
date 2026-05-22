# 2.1 Basic Optimization in Julia (first-order library methods)

**Problem.** Find minima using Optimization.jl with the four common first-order (gradient-only) optimisers - Gradient Descent, Momentum, Nesterov Momentum, and Adam - and compare how each converges.

Split by test function (each file overlays all optimisers on that problem):

| File | Goal / strategy |
| --- | --- |
| `01_Quadratic1D_Optimizers.jl` | 1-D quadratic f = x^2 - 2x + 0.6: GD vs Momentum vs Nesterov vs Adam. |
| `02_Quadratic2D_Optimizers.jl` | 2-D Shewchuk quadratic: GD vs Momentum vs Nesterov paths on a contour. |
| `03_Rosenbrock_Optimizers.jl` | Rosenbrock: GD vs Nesterov vs Adam, plus sol.u/objective/retcode. |
