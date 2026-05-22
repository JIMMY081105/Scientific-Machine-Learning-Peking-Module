# =====================================================================================
# 1.4 - Solving a system of nonlinear equations (Example 1.4.3)
# =====================================================================================
# GOAL: find where two curves cross: u(x, y) = x^2 - y + 1 = 0 and v(x, y) = 3cos(x) - y = 0.
# STRATEGY: extend Newton-Raphson with the Jacobian matrix and iterate, then confirm with
#           NonlinearSolve; visualise the solution as the crossing of the two zero-contours.
# =====================================================================================

using NonlinearSolve
using CairoMakie

# The two curves y = x^2 + 1 and y = 3cos(x).
x₁ = 0:0.1:3
y₁ = x₁.^2 .+ 1
y₂ = 3 * cos.(x₁)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (0, 3, -2.0, 5.0), xlabelsize = 25, ylabelsize = 25)
lines!(ax, x₁, y₁; color = :black, label = L"u(x,y)")
lines!(ax, x₁, y₂; color = :blue, label = L"v(x,y)")
axislegend(ax, position = :rt)
display(fig)

# Newton-Raphson with the Jacobian.
function uandv(x, y)
    return [x^2 - y + 1, 3 * cos(x) - y]
end
function Jacobian(x, y)
    return [2*x -1; -3*sin(x) -1]
end

x = y = 10.0
for i = 1:10
    J = Jacobian(x, y)
    F = uandv(x, y)
    delta = -J \ F
    x += delta[1]
    y += delta[2]
    println("x=$x, y=$y")
end

# Visualise the solution as the crossing of the two zero-contours.
xrange = 0:0.02:3
yrange = -2:0.02:5
functionToPlot = [uandv(x, y) for x in xrange, y in yrange]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 25, ylabelsize = 25, limits = (0, 3.0, -2.0, 5.0))
contour!(ax, xrange, yrange, getindex.(functionToPlot, 1); levels = [0.0], color = :black, label = L"u(x,y)")
contour!(ax, xrange, yrange, getindex.(functionToPlot, 2); levels = [0.0], color = :blue, label = L"v(x,y)")
axislegend(ax, position = :rt)
display(fig)

# The same solve via NonlinearSolve, with the root marked.
function FunctionToSolve(x)
    func = [x[1]^2 - x[2] + 1, 3 * cos(x[1]) - x[2]]
    return func
end

prob = NonlinearProblem((x, p) -> FunctionToSolve(x), [1, 1], [])
sol = NonlinearSolve.solve(prob)
scatter!(ax, sol.u[1], sol.u[2]; color = :red, markersize = 20)
display(fig)

# TODO (Class Demo 03): Adapt this to find all solutions of x^2 + y^2 = 1 and 4x^2 + y^2/4 = 1.
