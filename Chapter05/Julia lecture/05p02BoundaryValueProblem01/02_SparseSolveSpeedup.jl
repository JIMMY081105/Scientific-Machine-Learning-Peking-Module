# =====================================================================================
# 5.2 - The same BVP solved with a sparse matrix (speed-up)
# =====================================================================================
# GOAL: solve the same y'' + (2/x) y' - (2/x^2) y = 0 BVP, but exploit the fact that the
#       finite-difference matrix is mostly zeros.
# STRATEGY: convert A to a sparse matrix and solve with LinearSolve's UMFPACK factorisation;
#           the speed-up over the dense backslash grows with grid size.
# =====================================================================================

using CairoMakie
using LinearSolve
using SparseArrays

# Rebuild the same system as 01_DenseFiniteDifference.jl.
Delta = 0.001
x = 1:Delta:2

n = length(x)
A = zeros(n, n)
C = zeros(n)

A[1, 1] = 1.0
C[1] = 5.0
for i = 2:n-1
    A[i, i-1] = (1.0 / Delta^2) - (2 / (x[i])) * (1 / (2 * Delta))
    A[i, i] = -(2.0 / Delta^2) - (2 / (x[i] * x[i]))
    A[i, i+1] = (1.0 / Delta^2) + (2 / (x[i])) * (1 / (2 * Delta))
end
A[n, n] = 1.0
C[n] = 3.0

# Sparse version of the matrix.
ASparse = sparse(A)

# Solve with LinearSolve (UMFPACK) - much faster than the dense backslash on fine grids.
probSparse = LinearProblem(ASparse, C)
@time y2 = LinearSolve.solve(probSparse, UMFPACKFactorization())

# Confirm the sparse solution matches the exact one.
yexact(x) = x + 4 / x^2
xplot = 1:0.01:2
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
lines!(ax, xplot, yexact.(xplot), color = :black, label = "Analytical solution")
lines!(ax, x, y2, color = :blue, label = "Sparse finite difference solution")
axislegend(ax, position = :rt)
fig
