# =====================================================================================
# 2.2 - Second-order library methods on a 2-D quadratic (Example 2.2.2)
# =====================================================================================
# GOAL: minimise the Shewchuk quadratic (minimum at (2, -2)) with Newton, BFGS, LBFGS.
# STRATEGY: same Optim solvers, now in 2-D with an AutoForwardDiff Hessian; overlay the
#           three paths on the contour plot (they all reach the minimum quickly).
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

quad(x) = (3.0 / 2.0) * x[1]^2 + 2 * x[1] * x[2] + 3 * x[2]^2 - 2 * x[1] + 8 * x[2]

x1range = -6:0.02:6
x2range = -6:0.02:6
funcplot = [quad([x1, x2]) for x1 in x1range, x2 in x2range]

# Run all three from the same start (ForwardDiff supplies the Hessian).
x0 = 4.0 * ones(2)
optf = OptimizationFunction((x, p) -> quad(x), ADTypes.AutoForwardDiff())
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

# Overlay the three paths on the contour.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Simple Quadratic function", limits = (-6, 6.0, -6.0, 6.0))
contourf!(ax, x1range, x2range, funcplot; levels = 20, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; labels = true, levels = 20, color = :black)
scatter!(ax, [2], [-2]; markersize = 20, color = :black)
lines!(ax, getproperty.(vec_points_new[:], :x1), getproperty.(vec_points_new[:], :x2); color = :black, label = "Newton")
lines!(ax, getproperty.(vec_points_bfgs[:], :x1), getproperty.(vec_points_bfgs[:], :x2); color = :green, label = "BFGS")
lines!(ax, getproperty.(vec_points_lbfgs[:], :x1), getproperty.(vec_points_lbfgs[:], :x2); color = :cyan, label = "LBFGS")
axislegend(ax; position = :lt)
fig
