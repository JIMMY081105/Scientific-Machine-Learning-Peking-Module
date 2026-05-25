# =====================================================================================
# 2.4 - Hand-coded first-order methods on a 2-D quadratic (Example 2.4.2)
# =====================================================================================
# GOAL: minimise the Shewchuk quadratic (minimum at (2, -2)) in vector form.
# STRATEGY: hand-coded Gradient Descent, Momentum, and Nesterov Momentum (each a 2-vector
#           update), overlaid on the contour plot with negative-gradient arrows.
# =====================================================================================

using CairoMakie
using LinearAlgebra

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

quad(x) = (3.0 / 2.0) * x[1]^2 + 2 * x[1] * x[2] + 3 * x[2]^2 - 2 * x[1] + 8 * x[2]
function Gradient_quad(x)
    d1 = 3 * x[1] + 2 * x[2] - 2 # df/dx1
    d2 = 6 * x[2] + 2 * x[1] + 8 # df/dx2
    return [d1, d2]
end

x1range = -6:0.02:6
x2range = -6:0.02:6
funcplot = [quad([x1, x2]) for x1 in x1range, x2 in x2range]

# Contour with negative-gradient arrows.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "Quadratic function", limits = (-6, 6.0, -6.0, 6.0))
contour!(ax, x1range, x2range, funcplot; labels = true, levels = 20, color = :black, linewidth = 5)
contourf!(ax, x1range, x2range, funcplot; levels = 50, colormap = :bwr)
x1range_arrow = -6:1:6
x2range_arrow = -6:1:6
arrowplot = [Gradient_quad([x1, x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax, x1range_arrow, x2range_arrow, -getindex.(arrowplot, 1), -getindex.(arrowplot, 2); color = (:black, 0.2), lengthscale = 0.1)
scatter!(ax, [2], [-2], color = :black, markersize = 20)

# --- Gradient Descent ---
x0 = [4.0, 4.0]
alpha = 0.1
data = [Point2D(x0..., quad(x0))]
x = x0
for i in 1:50
    x -= alpha * Gradient_quad(x)
    push!(data, Point2D(x..., quad(x)))
end
lines!(ax, getproperty.(data, :x1), getproperty.(data, :x2); linewidth = 3, color = :grey, label = "Gradient Descent")
scatter!(ax, getproperty.(data, :x1), getproperty.(data, :x2); color = :grey, markersize = 20, strokecolor = :black, strokewidth = 2)

# --- Momentum ---
alpha = 0.1
beta = 0.4
v = [0.0, 0.0]
data_m = [Point2D(x0..., quad(x0))]
x = x0
for i in 1:20
    v = beta * v .- alpha * Gradient_quad(x)
    x += v
    push!(data_m, Point2D(x..., quad(x)))
end
lines!(ax, getproperty.(data_m, :x1), getproperty.(data_m, :x2), linewidth = 3, color = :green, label = "Momentum")
scatter!(ax, getproperty.(data_m, :x1), getproperty.(data_m, :x2); color = :green, markersize = 20, strokecolor = :black, strokewidth = 2)

# --- Nesterov Momentum ---
beta = 0.4
v = [0.0, 0.0]
x = x0
data_nm = [Point2D(x0..., quad(x0))]
for i in 1:20
    v = beta * v - alpha * Gradient_quad(x + beta * v)
    x += v
    push!(data_nm, Point2D(x..., quad(x)))
end
lines!(ax, getproperty.(data_nm, :x1), getproperty.(data_nm, :x2), linewidth = 5, color = :red, label = "Nesterov Momentum")
axislegend(ax; position = :lt)
fig

# TODO (Class Demo 01): Implement the momentum method for this same quadratic.
