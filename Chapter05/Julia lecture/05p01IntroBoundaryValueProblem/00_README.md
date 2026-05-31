# 5.1 Introduction to Boundary Value Problems

**Problem.** Find the temperature distribution T(x) along a heated plate governed by k T'' = -q(x) on [0,1] with T(0)=5 and T'(1)=0. The unknown is a function, obtained by solving the ODE numerically with finite differences.

This lesson is split by goal/strategy so each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_UniformHeatingTridiagonal.jl` | Uniform heating: build the tridiagonal system A T = C, solve, and time the tridiagonal solver vs a general sparse solver. |
| `02_NonUniformHeating.jl` | Non-uniform heating q(x) = 1 + 2x: only C changes, so re-solve and overlay both profiles. |
