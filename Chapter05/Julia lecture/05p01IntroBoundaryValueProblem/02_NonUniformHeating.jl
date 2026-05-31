# =====================================================================================
# 5.1 - Heated plate with non-uniform heating
# =====================================================================================
# GOAL: repeat the heated-plate BVP but with position-dependent heating q(x) = 1 + 2x,
#       and overlay the two temperature profiles.
# STRATEGY: the matrix A is identical to the uniform case - only the right-hand side C
#           changes - so we rebuild the system once and re-solve with the new C.
# =====================================================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays

Delta = 0.2
x = 0:Delta:1
ThermalConductivity = 0.1

n = length(x)
d = zeros(n)
dl = zeros(n - 1)
du = zeros(n - 1)
C = zeros(n)

# Build A and the uniform-heating right-hand side first (for the comparison curve).
du[1] = 0.0
d[1] = 1.0
C[1] = 5.0
for i = 2:n-1
    dl[i-1] = (1.0 / Delta^2)
    d[i] = -(2.0 / Delta^2)
    du[i] = (1.0 / Delta^2)
    C[i] = -1.0 / ThermalConductivity
end
dl[n-1] = -1
d[n] = 1.0
C[n] = 0.0

A = Tridiagonal(dl, d, du)

prob = LinearProblem(A, C)
Tuniform = LinearSolve.solve(prob)

# Non-uniform heating: only the interior entries of C change to -(1 + 2x)/k.
for i = 2:n-1
    C[i] = -(1.0 + 2 * x[i]) / ThermalConductivity
end

prob = LinearProblem(A, C)
@time Tlinear = LinearSolve.solve(prob)

# Overlay the two temperature profiles.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)")
lines!(ax, x, Tuniform, color = :black, label = "Uniform Heating")
lines!(ax, x, Tlinear, color = :blue, label = "Linear Heating")
axislegend(ax, position = :lt)
fig
