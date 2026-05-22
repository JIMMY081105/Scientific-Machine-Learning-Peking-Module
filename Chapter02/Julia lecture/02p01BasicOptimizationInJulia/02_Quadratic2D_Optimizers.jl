# =====================================================================================
# 2.1 - First-order optimisers on a 2-D quadratic (Example 2.1.2)
# =====================================================================================
# GOAL: minimise the Shewchuk quadratic (minimum at (2, -2)) with Optimization.jl.
# STRATEGY: run Gradient Descent, Momentum, and Nesterov Momentum, and overlay their
#           paths on the contour plot to compare how they approach the minimum.
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

quad(x) = (3.0 / 2.0) * x[1]^2 + 2 * x[1] * x[2] + 3 * x[2]^2 - 2 * x[1] + 8 * x[2]

x1range = -6:0.02:6
x2range = -6:0.02:6
funcplot = [quad([x1, x2]) for x1 in x1range, x2 in x2range]

# Base contour plot with the true minimum marked.
function base_plot()
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "Simple Quadratic function", limits = (-6, 6.0, -6.0, 6.0))
    contourf!(ax, x1range, x2range, funcplot; levels = 20, colormap = :bwr)
    contour!(ax, x1range, x2range, funcplot; labels = true, levels = 20, color = :black)
    scatter!(ax, [2], [-2]; markersize = 20, color = :black)
    return fig, ax
end

fig, ax = base_plot()
fig

# Run each optimiser from the same start.
x0 = 4.0 * ones(2)
optf = OptimizationFunction((x, p) -> quad(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

vec_points = []
sol = solve(prob, Optimisers.Descent(0.1); maxiters = 20, callback = my_callback2D)
vec_points_gd = copy(vec_points)

vec_points = []
sol = solve(prob, Optimisers.Momentum(0.1, 0.9); maxiters = 20, callback = my_callback2D)
vec_points_m = copy(vec_points)

vec_points = []
sol = solve(prob, Optimisers.Nesterov(0.1, 0.9); maxiters = 20, callback = my_callback2D)
vec_points_nm = copy(vec_points)

# Overlay the three paths on the contour.
fig, ax = base_plot()
lines!(ax, getproperty.(vec_points_gd[:], :x1), getproperty.(vec_points_gd[:], :x2); color = :cyan, label = "Gradient Descent")
lines!(ax, getproperty.(vec_points_m[:], :x1), getproperty.(vec_points_m[:], :x2); color = :yellow, label = "Momentum")
lines!(ax, getproperty.(vec_points_nm[:], :x1), getproperty.(vec_points_nm[:], :x2); color = :orange, label = "Nesterov Momentum")
axislegend(ax; position = :lt)
fig
