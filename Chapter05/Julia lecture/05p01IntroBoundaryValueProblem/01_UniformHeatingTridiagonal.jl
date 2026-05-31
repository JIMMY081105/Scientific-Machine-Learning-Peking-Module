# =====================================================================================
# 5.1 - Heated plate with uniform heating (tridiagonal finite differences)
# =====================================================================================
# GOAL: find the temperature T(x) of a uniformly heated plate governed by
#       k T'' = -q on [0, 1] with T(0) = 5 and T'(1) = 0.
# STRATEGY: replace T'' with a central difference on a grid, assemble the linear system
#           A T = C as a Tridiagonal matrix, and solve it. We also time the specialised
#           tridiagonal solver against a general sparse solver.
# =====================================================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using BenchmarkTools

# Grid and physical constants.
Delta = 0.2
x = 0:Delta:1
α = 5.0
ThermalConductivity = 0.1

# Diagonals of the tridiagonal system and the right-hand side.
n = length(x)
d = zeros(n)
dl = zeros(n - 1)
du = zeros(n - 1)
C = zeros(n)

# Row 1: boundary condition T(0) = 5. Interior rows: the discretised ODE.
# Last row: the T'(1) = 0 condition.
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

# Construct the specialized structured matrix
A = Tridiagonal(dl, d, du)
A_sparse = sparse(A)

# Solve: tridiagonal solver vs general sparse solver (tridiagonal is faster).
prob = LinearProblem(A, C)
#@btime y=LinearSolve.solve(prob)
@time Tuniform = LinearSolve.solve(prob)

probSparse = LinearProblem(A_sparse, C)
@time y = LinearSolve.solve(probSparse, UMFPACKFactorization())

# Plot T(x).
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)")
lines!(ax, x, Tuniform, color = :black, label = "Uniform Heating")
scatter!(ax, x, Tuniform, color = :black, markersize = 20)
axislegend(ax, position = :lt)
# save("../../figures/05p01IntroBVP01.svg", fig)   # (figure-save disabled in study file)
fig
