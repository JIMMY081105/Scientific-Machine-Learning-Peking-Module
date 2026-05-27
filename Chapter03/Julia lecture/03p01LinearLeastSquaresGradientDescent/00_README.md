# 3.1 Linear Regression with Gradient Descent

**Problem.** Given noisy (x, y) data, find the model coefficients that best fit it by minimising the sum-of-squared-errors loss S with gradient descent.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_ConstantModelGradientDescent.jl` | Constant model f = a0, hand-coded gradient descent using the analytic derivative. |
| `02_LineModelOptimization.jl` | Line model f = a0 + a1*x, loss-surface contour, minimised with Optimization.jl (Descent, AutoZygote). |
