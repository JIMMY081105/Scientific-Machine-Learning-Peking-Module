# 2.4 Multivariate Examples (coded from scratch, 2-D)

**Problem.** Extend the hand-coded first-order methods to functions of two variables, where the gradient is a 2-vector, and visualise why "follow the negative gradient" works.

Split by test function:

| File | Goal / strategy |
| --- | --- |
| `01_Bowl_GradientDescent.jl` | Bowl x1^2 + x2^2: contour + gradient arrows + a gradient-descent path to (0, 0). |
| `02_Quadratic_GDMomentumNesterov.jl` | Shewchuk quadratic: GD, Momentum, Nesterov in vector form, overlaid on the contour. |
