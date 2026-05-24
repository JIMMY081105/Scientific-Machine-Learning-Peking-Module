# =====================================================================================
# 2.2 - Second-order library methods on the Rosenbrock function (Example 2.2.3)
# =====================================================================================
# GOAL: minimise the Rosenbrock function (minimum at (1, 1)) with Newton, BFGS, LBFGS.
# STRATEGY: overlay the three Optim paths and print sol.u/objective/retcode; all three
#           converge within 20 iterations.
# =====================================================================================

using Optimization, OptimizationOptimJL, ForwardDiff, CairoMakie, Zygote, ADTypes

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

# Run all three from the same start.
x0 = -1.0 * ones(2)
optf = OptimizationFunction((x, p) -> rosenbrock(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

vec_points = []
sol = solve(prob, Optim.Newton(); maxiters = 20, callback = my_callback2D)
vec_points_new = copy(vec_points)

vec_points = []
sol = solve(prob, Optim.BFGS(); maxiters = 20, callback = my_callback2D)
vec_points_bfgs = copy(vec_points)

vec_points = []
sol = solve(prob, Optim.LBFGS(); maxiters = 20, callback = my_callback2D)
vec_points_lbfgs = copy(vec_points)

# Overlay the three paths.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "rosenbrock", limits = (-5, 5.0, -5.0, 5.0))
levels = 10.0 .^ range(-2, 3.5; length = 10)
contourf!(ax, x1range, x2range, funcplot; levels, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; labels = true, levels, color = :black)
scatter!(ax, [1], [1]; markersize = 20, color = :black)
lines!(ax, getproperty.(vec_points_new[:], :x1), getproperty.(vec_points_new[:], :x2); color = :black, label = "Newton")
lines!(ax, getproperty.(vec_points_bfgs[:], :x1), getproperty.(vec_points_bfgs[:], :x2); color = :green, label = "BFGS")
lines!(ax, getproperty.(vec_points_lbfgs[:], :x1), getproperty.(vec_points_lbfgs[:], :x2); color = :cyan, label = "LBFGS")
axislegend(ax; position = :rt, orientation = :vertical)
fig

println("Optimal parameters u:", sol.u)
println("Optimal objective value:", sol.objective)
println("Convergence status:", sol.retcode)
