# 4.2 Implementation of Automatic Differentiation (Forward Mode)

**Problem.** Demystify automatic differentiation by building forward mode yourself: carry each value together with its derivative (a "Dual number") and apply the chain rule at every elementary operation.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_DualNumbersUnivariate.jl` | Define `Dual`, overload +, -, *, ^, sin, and differentiate f(x) by seeding `Dual(x, 1.0)`; plot vs analytic. |
| `02_MultivariablePartials.jl` | Partial derivatives: seed the chosen variable with 1 and the rest with 0 (one pass per input). |
