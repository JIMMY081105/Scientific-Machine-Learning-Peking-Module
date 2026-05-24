# 2.3 Gradient Based Methods (coded from scratch, 1-D)

**Problem.** Understand the first-order optimisers by implementing their update equations yourself (no library) for 1-D functions with an explicit gradient.

Split by test function:

| File | Goal / strategy |
| --- | --- |
| `01_Quadratic_GDMomentumNesterov.jl` | f = x^2 - 2x + 0.6: hand-coded GD, Momentum, Nesterov compared. |
| `02_GaussianWell_SlowGradient.jl` | g = -exp(-x^2): the tiny far-field gradient makes plain GD crawl; momentum escapes faster. |
