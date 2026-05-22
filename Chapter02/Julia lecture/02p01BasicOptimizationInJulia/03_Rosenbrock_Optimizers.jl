# =====================================================================================
# 2.1 - First-order optimisers on the Rosenbrock function (Example 2.1.3)
# =====================================================================================
# GOAL: minimise the Rosenbrock function (minimum at (1, 1)), a classic hard case with a
#       curved narrow valley.
# STRATEGY: run Gradient Descent, Nesterov Momentum, and Adam, overlay their paths, and
#           read the final answer/status from sol.u, sol.objective, sol.retcode.
# =====================================================================================

using Optimization, OptimizationOptimisers, Zygote, CairoMakie

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

function my_callback2D(state, l)
    push!(vec_points, Point2D(state.u..., l))
    return false # continues the solver
end

rosenbrock(x) = (1.0 - x[1])^2 + 1.0 * (x[2] - x[1]^2)^2

x1range = -5:0.02:5
x2range = -5:0.02:5
funcplot = [rosenbrock([x1, x2]) for x1 in x1range, x2 in x2range]

# Run each optimiser from the same start.
x0 = -1.0 * ones(2)
optf = OptimizationFunction((x, p) -> rosenbrock(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

vec_points = []
sol = solve(prob, Optimisers.Descent(0.1); maxiters = 50, callback = my_callback2D)
vec_points_gd = copy(vec_points)

vec_points = []
sol = solve(prob, Optimisers.Nesterov(0.1, 0.9); maxiters = 50, callback = my_callback2D)
vec_points_nm = copy(vec_points)

vec_points = []
sol = solve(prob, Optimisers.Adam(0.1, (0.9, 0.999)); maxiters = 50, callback = my_callback2D)
vec_points_ad = copy(vec_points)

# Overlay the three paths.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "rosenbrock", limits = (-5, 5.0, -5.0, 5.0))
levels = 10.0 .^ range(-2, 3.5; length = 10)
contourf!(ax, x1range, x2range, funcplot; levels, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; labels = true, levels, color = :black)
scatter!(ax, [1], [1]; markersize = 20, color = :black)
lines!(ax, getproperty.(vec_points_gd[:], :x1), getproperty.(vec_points_gd[:], :x2); color = :cyan, label = "Gradient Descent")
lines!(ax, getproperty.(vec_points_nm[:], :x1), getproperty.(vec_points_nm[:], :x2); color = :orange, label = "Nesterov Momentum")
lines!(ax, getproperty.(vec_points_ad[:], :x1), getproperty.(vec_points_ad[:], :x2); color = :red, label = "Adam")
axislegend(ax; position = :rt)
fig

# Final solution details.
println("Optimal parameters u:", sol.u)
println("Optimal objective value:", sol.objective)
println("Convergence status:", sol.retcode)

# TODO (Exercise Ex05): Generalise to f(x1, x2, p1, p2) = (p1 - x1)^2 + p2*(x2 - x1^2)^2 for varying p1, p2.
# TODO (Class Demo 02): Minimise the Himmelblau function h(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2.
