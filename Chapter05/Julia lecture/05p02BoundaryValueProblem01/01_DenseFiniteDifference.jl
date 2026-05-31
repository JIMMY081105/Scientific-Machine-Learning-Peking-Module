# =====================================================================================
# 5.2 - A fuller linear BVP, solved with a dense finite-difference matrix
# =====================================================================================
# GOAL: solve y'' + (2/x) y' - (2/x^2) y = 0 on [1, 2] with y(1) = 5, y(2) = 3, and
#       check it against the known exact solution y = x + 4/x^2.
# STRATEGY: central differences for BOTH y'' and y' give each interior point a 3-term
#           row; assemble a dense matrix A and solve with backslash.
# =====================================================================================

using CairoMakie

# Grid and boundary values.
Delta = 0.001
x = 1:Delta:2
α = 5.0
β = 3.0

n = length(x)
A = zeros(n, n)
C = zeros(n)

# Row 1: y(1) = 5. Interior rows: discretised ODE (three nonzero coefficients each).
# Last row: y(2) = 3.
A[1, 1] = 1.0
C[1] = 5.0
for i = 2:n-1
    A[i, i-1] = (1.0 / Delta^2) - (2 / (x[i])) * (1 / (2 * Delta))
    A[i, i] = -(2.0 / Delta^2) - (2 / (x[i] * x[i]))
    A[i, i+1] = (1.0 / Delta^2) + (2 / (x[i])) * (1 / (2 * Delta))
end
A[n, n] = 1.0
C[n] = 3.0

# Solve the dense system (slow for fine grids).
@time y = A \ C

# Compare against the exact solution.
yexact(x) = x + 4 / x^2

xplot = 1:0.01:2
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
lines!(ax, xplot, yexact.(xplot), color = :black, label = "Analytical solution")
lines!(ax, x, y, color = :blue, label = "Finite difference solution")
axislegend(ax, position = :rt)
fig
