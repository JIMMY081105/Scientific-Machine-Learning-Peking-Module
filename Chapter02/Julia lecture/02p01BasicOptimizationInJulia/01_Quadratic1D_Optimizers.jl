# =====================================================================================
# 2.1 - First-order optimisers on a 1-D quadratic (Example 2.1.1)
# =====================================================================================
# GOAL: minimise f(x) = x^2 - 2x + 0.6 (minimum at x = 1) with Optimization.jl.
# STRATEGY: run Gradient Descent, Momentum, Nesterov Momentum, and Adam in turn, each
#           logging its path through a callback, and overlay the convergence paths to
#           compare speed and overshoot.
# =====================================================================================

using Optimization, OptimizationOptimisers, Zygote, CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end
struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

function my_callback1D(state, l)
    push!(vec_points, Point1D(state.u..., l))
    return false # continues the solver
end

# The objective (the library expects f(x) with x a vector).
f(x) = x[1]^2 - 2 * x[1] + 0.6

xplot = 0:0.01:2
yplot = [f(x) for x in xplot]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", xlabelsize = 20, ylabelsize = 20, limits = (0, 2, -0.5, 1.0))
lines!(ax, xplot, yplot)
scatter!(ax, 1, f.(1); color = :black, markersize = 15)
# save("../../figures/02p01BasicOptimizationInJulia01.svg", fig)   # (figure-save disabled in study file)
display(fig)

# Optimization.jl wants objective(x, p); wrap f as (x,p)->f(x).
x0 = 1.8 * ones(1)
optf = OptimizationFunction((x, p) -> f(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

# --- Gradient Descent ---
vec_points = []
sol = solve(prob, Optimisers.Descent(0.1); maxiters = 20, callback = my_callback1D)
vec_points_gd = copy(vec_points)

# --- Momentum ---
vec_points = []
sol = solve(prob, Optimisers.Momentum(0.1, 0.9); maxiters = 20, callback = my_callback1D)
vec_points_m = copy(vec_points)

# --- Nesterov Momentum ---
vec_points = []
sol = solve(prob, Optimisers.Nesterov(0.1, 0.9); maxiters = 20, callback = my_callback1D)
vec_points_nm = copy(vec_points)

# --- Adam ---
vec_points = []
sol = solve(prob, Optimisers.Adam(0.1, (0.9, 0.99)); maxiters = 20, callback = my_callback1D)
vec_points_ad = copy(vec_points)

# Overlay every method's path (left: on f(x); right: x vs iteration).
fig = Figure(size = (1000, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)", limits = (0, 2, -0.5, 1.0))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x", limits = (0, 20, 0.0, 2.0))
lines!(ax1, xplot, yplot; color = :black)
hlines!(ax2, 1.0, color = :red, linestyle = :dash)

for (pts, c, name) in ((vec_points_gd, :blue, "Gradient Descent"),
    (vec_points_m, :green, "Momentum"),
    (vec_points_nm, :black, "Nesterov Momentum"),
    (vec_points_ad, :red, "Adam"))
    scatter!(ax1, getproperty.(pts, :x), getproperty.(pts, :fvalue); markersize = 10, color = c)
    lines!(ax2, getproperty.(pts, :x); color = c, label = name)
    scatter!(ax2, getproperty.(pts, :x); markersize = 10, color = c)
end
axislegend(ax2; position = :rt)
fig

# TODO (Exercise Ex02): Repeat with learning rates alpha in [0.01, 0.8] and discuss small vs large alpha.
# TODO (Class Demo 01): Use one of these methods to minimise A(r) = 2*pi*r^2 + 2/r.
