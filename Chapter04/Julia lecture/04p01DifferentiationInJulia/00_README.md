# 4.1 Introduction to Differentiation

**Problem.** Understand the different ways to compute derivatives (symbolic, numerical, automatic) and use Julia's ForwardDiff and Zygote to get exact derivatives, gradients, and Jacobians without differentiating by hand.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_SymbolicDifferentiation.jl` | Exact symbolic derivative with the Symbolics package. |
| `02_NumericalDifferentiation.jl` | Forward/backward/central differences + complex step; error vs step size h. |
| `03_AutomaticDifferentiation.jl` | Machine-precision derivatives with ForwardDiff/Zygote, plotted against the analytic one. |
| `04_ADForRootFindingAndDescent.jl` | AD inside Newton-Raphson and gradient descent on Himmelblau. |
| `05_FunctionsWithParameters.jl` | Differentiating multi-argument functions: ForwardDiff (one closure at a time) vs Zygote (several at once). |
| `06_ParameterEstimationFit.jl` | Fit a1*(1 - exp(a2*x)) to data with Zygote gradients + gradient descent. |
| `07_JacobiansAndNonlinearSystems.jl` | Jacobians of vector functions and a Newton-Raphson system solve. |
