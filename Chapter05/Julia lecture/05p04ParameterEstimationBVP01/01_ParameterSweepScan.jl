# =====================================================================================
# 5.4 - Inverse problem by brute-force parameter sweep
# =====================================================================================
# GOAL: the forward BVP k T'' = -1 is known but k is unknown; given the measurement
#       T(x=1) = 3, find roughly which k reproduces it.
# STRATEGY: solve the tridiagonal system for many trial values of k, plot T(x=1) and the
#           squared error (T(1) - 3)^2 against k, and read off where the target is met
#           (around k = 0.22). This is the "scan every value" approach.
# =====================================================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using BenchmarkTools

# Grid and constants.
Delta = 0.1
x = 0:Delta:1
α = 5.0
β = 3.0
ThermalConductivity = 0.1

n = length(x)
d = zeros(n)
dl = zeros(n - 1)
du = zeros(n - 1)
C = zeros(n)

# Same tridiagonal system as 5.1 (note the +1/k sign convention used here).
du[1] = 0.0
d[1] = 1.0
C[1] = 5.0
for i = 2:n-1
    dl[i-1] = (1.0 / Delta^2)
    d[i] = -(2.0 / Delta^2)
    du[i] = (1.0 / Delta^2)
    C[i] = 1.0 / ThermalConductivity
end
dl[n-1] = -1
d[n] = 1.0
C[n] = 0.0

A = Tridiagonal(dl, d, du)
A_sparse = sparse(A)

# Forward solve for the initial k, and read off the temperature at the last point.
prob = LinearProblem(A, C)
@time T = LinearSolve.solve(prob)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
lines!(ax, x, T, color = :blue, label = "Finite difference solution")
axislegend(ax, position = :rt)
fig

T[end]

# Sweep k over a range, re-solving each time.
ThermalCondVect = 0.1:0.02:1.0
EndTemperature = similar(ThermalCondVect)

for i = 1:length(ThermalCondVect)
    ThermalConductivity = ThermalCondVect[i]
    C[2:n-1] .= 1.0 / ThermalConductivity
    prob = LinearProblem(A, C)
    y = LinearSolve.solve(prob)
    EndTemperature[i] = y[end]
end

# T(x=1) vs k, and the squared error vs k (the minimum marks the answer, k ~ 0.22).
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"k", ylabel = L"T(x=1)")
lines!(ax1, ThermalCondVect, EndTemperature)

ax2 = Axis(fig[1, 2], xlabel = L"k", ylabel = L"(T(x=1)-3)^2")
lines!(ax2, ThermalCondVect, (EndTemperature .- 3).^2)
# save("../../figures/05p04ParameterEstimationBVP0101.svg", fig)   # (figure-save disabled in study file)
fig
