# =====================================================================================
# 2.2 - Second-order library methods on a 1-D function (Example 2.2.1)
# =====================================================================================
# GOAL: minimise f(x) = (x^3 - x)^2 with curvature-aware optimisers.
# STRATEGY: Newton, BFGS, and LBFGS via Optim (through Optimization.jl). Newton has no
#           learning rate and converges in ~3 iterations; note Newton and BFGS from the
#           same start can land on different minima.
# =====================================================================================

using Optimization, OptimizationOptimJL, ForwardDiff, CairoMakie, Zygote, ADTypes

struct Point1D
    x::Float64
    fvalue::Float64
end

function my_callback1D(state, l)
    push!(vec_points, Point1D(state.u..., l))
    return false # continues the solver
end

f(x) = (x[1]^3 - x[1])^2
xplot = -2:0.01:2
yplot = [f(x) for x in xplot]

# Minima (black) and the maxima between them (magenta).
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", limits = (-2, 2, -0.01, 0.2))
lines!(ax, xplot, yplot)
scatter!(ax, [0, -1, 1], f.([0, -1, 1]); color = :black, markersize = 15)
scatter!(ax, [1 / sqrt(3), -1 / sqrt(3)], f.([1 / sqrt(3), -1 / sqrt(3)]); color = :magenta, markersize = 15)
fig

# Run all three optimisers from x0 = 1.1.
x0 = 1.1 * ones(1)
optf = OptimizationFunction((x, p) -> f(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

vec_points = []
sol = solve(prob, Optim.Newton(); maxiters = 20, callback = my_callback1D)
vec_points_new = copy(vec_points)
sol.retcode

vec_points = []
sol = solve(prob, Optim.BFGS(); maxiters = 20, callback = my_callback1D)
vec_points_bfgs = copy(vec_points)

vec_points = []
sol = solve(prob, Optim.LBFGS(); maxiters = 20, callback = my_callback1D)
vec_points_lbfgs = copy(vec_points)

# Overlay the three convergence paths.
fig = Figure(size = (1000, 400))
ax1 = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", limits = (-2, 2, -0.01, 0.2))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x", limits = (0, 20, -0.5, 1.5))
lines!(ax1, xplot, yplot)
scatter!(ax1, [1], [f(1)]; markersize = 20, color = :black)
hlines!(ax2, 1.0, color = :red, linestyle = :dash)

for (pts, c, name) in ((vec_points_new, :black, "Newton"),
    (vec_points_bfgs, :green, "BFGS"),
    (vec_points_lbfgs, :cyan, "LBFGS"))
    scatter!(ax1, getproperty.(pts, :x), getproperty.(pts, :fvalue); markersize = 10, color = c, label = name)
    scatter!(ax2, getproperty.(pts, :x); color = c)
    lines!(ax2, getproperty.(pts, :x); color = c)
end
axislegend(ax1; position = :rt)
fig

# TODO (Exercise Ex01): Try x0 = -0.5. Which minimum does Newton find? Check sol.retcode.
