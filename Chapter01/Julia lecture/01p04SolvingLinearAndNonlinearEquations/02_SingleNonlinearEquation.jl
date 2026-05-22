# =====================================================================================
# 1.4 - Solving a single nonlinear equation (Example 1.4.2)
# =====================================================================================
# GOAL: find the root of dA/dr = 0 for A(r) = 2*pi*r^2 + 2/r (the optimal radius).
# STRATEGY: Newton-Raphson x <- x - f(x)/f'(x) by hand in a loop, then the same thing
#           with the NonlinearSolve library; compare against the exact root.
# =====================================================================================

using NonlinearSolve
using CairoMakie

Area(x) = 2π * x^2 + 2 / x
f(x) = 4π * x - 2 / x^2     # dA/dx
dfdx(x) = 4π + 4 / x^3      # d^2A/dx^2

# Plot A(x) and its derivative.
xplot = 0.1:0.01:2
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"x", ylabel = L"A(x)", limits = (0, 2, 5, 25))
ax2 = Axis(fig[2, 1], xlabel = L"x", ylabel = L"dA/dx(x)", limits = (0, 2, -20, 20))
lines!(ax1, xplot, Area.(xplot); color = :black)
lines!(ax2, xplot, f.(xplot); color = :black)
# save("../../figures/01p04SolvingLinearAndNonlinearEquations02.svg", fig)   # (figure-save disabled in study file)
display(fig)

# Newton-Raphson by hand.
x = 1.0
for i = 1:10
    x = x - f(x) / dfdx(x)
    println(x)
end

# The exact root, for comparison.
(1 / (2π))^(1 / 3)

# The same solve via NonlinearSolve.
x0 = 1.0
p = []
prob = NonlinearProblem((x, p) -> f(x), x0, p)
sol = NonlinearSolve.solve(prob)

# TODO (Class Demo 02): Use Newton-Raphson to find x such that 5x^2 = 8 + x*cos(x).
