# =====================================================================================
# 5.4 - Inverse problem by gradient-based optimization
# =====================================================================================
# GOAL: find the k that makes T(x=1) = 3, but instead of scanning every value, minimise
#       a loss directly.
# STRATEGY: define loss(k) = (T_end(k) - 3)^2, which solves the linear system on every
#           evaluation, and minimise it with Optimization.jl (Adam), differentiating
#           THROUGH the solve with AutoForwardDiff. A callback logs the loss, and we
#           sanity-check with a ForwardDiff derivative near the solution.
# =====================================================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays

# Build the same tridiagonal matrix A (the load 1/k lives in C, set inside the loss).
Delta = 0.1
x = 0:Delta:1
ThermalConductivity = 0.1

n = length(x)
d = zeros(n)
dl = zeros(n - 1)
du = zeros(n - 1)

du[1] = 0.0
d[1] = 1.0
for i = 2:n-1
    dl[i-1] = (1.0 / Delta^2)
    d[i] = -(2.0 / Delta^2)
    du[i] = (1.0 / Delta^2)
end
dl[n-1] = -1
d[n] = 1.0

A = Tridiagonal(dl, d, du)

# Loss: build C from the trial k = p[1], solve, and square the error at the last point.
function loss(p, ydesired)
    C = ones(n) ./ p[1]
    C[1] = 5.0
    C[n] = 0.0
    prob = LinearProblem(A, C)
    y = LinearSolve.solve(prob)
    return sum((y[end] - ydesired[1]).^2)
end

function callback(state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

using Optimization
using Optimisers
using OptimizationOptimisers
using ForwardDiff

# Minimise the loss for k, differentiating through the linear solve with ForwardDiff.
losses = Float64[]
adtype = AutoForwardDiff()
optf = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optf, [0.1], [3])
@time sol = Optimization.solve(optprob, Adam(0.01f0), maxiters = 5000, callback = callback)

# Sanity check: the derivative of the loss near the solution k = 0.225.
df = ForwardDiff.derivative((p) -> loss(p, [3.0]), 0.225)
