# 5.4 Introduction to Inverse Problems for BVPs

**Problem.** The forward BVP k T'' = -1 is known, but the material constant k is unknown. Given a measurement T(1)=3, find the k that makes the finite-difference solution reproduce it.

This lesson is split by the two strategies for finding k, each file runs on its own:

| File | Goal / strategy |
| --- | --- |
| `01_ParameterSweepScan.jl` | Brute force: solve for many trial k and plot T(1) and (T(1)-3)^2 vs k to locate the answer (~0.22). |
| `02_OptimizeWithAdam.jl` | Gradient based: minimise loss(k) = (T(1)-3)^2 with Optimization.jl (Adam) differentiating through the solve (AutoForwardDiff). |
