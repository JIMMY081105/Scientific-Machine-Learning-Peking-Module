# =====================================================================================
# 4.4 - Finding V1 by optimizing through the linear solve
# =====================================================================================
# GOAL: find the input voltage V1 that makes i5 = 0, without scanning every value.
# STRATEGY: define loss(V1) that rebuilds C, solves A i = C, and returns i5^2 - so each
#           evaluation solves the system. ForwardDiff differentiates THROUGH the solve;
#           we minimise first with hand-coded gradient descent, then with Optimization.jl
#           (Adam) and a loss-logging callback.
# =====================================================================================

using LinearSolve
using CairoMakie
using Optimization
using Optimisers
using OptimizationOptimisers
using ForwardDiff

# Circuit constants and system matrix.
R₁ = 5.0
R₂ = 150.0
R₃ = 100.0
R₄ = 250.0
R₅ = 200.0
V₂ = 50.0

A = [R₁ R₂ 0 0 0;
    0 -R₂ R₃ R₄ 0;
    0 0 0 R₄ -R₅;
    1 -1 -1 0 0;
    0 0 1 -1 -1]

# Loss: for a trial V1 = p[1], solve the system and return i5^2.
function loss(p)
    C = [p[1], 0, V₂, 0, 0]
    prob = LinearProblem(A, C)
    i = LinearSolve.solve(prob)
    return sum(i[end].^2)
end

loss([91.0])

# ForwardDiff differentiates through the solve (sign matches the i5^2-vs-V1 curve).
ForwardDiff.gradient(loss, [91.0])
ForwardDiff.gradient(loss, [55.0])

# Minimise with hand-coded gradient descent.
iguess = 60
α = 100.00
for i = 1:50000
    iguess -= α * ForwardDiff.gradient(loss, [iguess])[1]
end
iguess

# Minimise with the Optimization.jl library (Adam), logging the loss via a callback.
function callback(state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

losses = Float64[]
optf = OptimizationFunction((p, x) -> loss(p), AutoForwardDiff())
optprob = OptimizationProblem(optf, [80.0])
sol = Optimization.solve(optprob, Adam(0.1f0), maxiters = 500, callback = callback)
#sol=Optimization.solve(optprob,Descent(0.1f0),maxiters=10000,callback=callback)
