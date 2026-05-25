# =====================================================================================
# 2.4 - Gradient descent on a 2-D bowl (Example 2.4.1)
# =====================================================================================
# GOAL: minimise bowl(x) = x1^2 + x2^2 with hand-coded gradient descent, where the
#       gradient is a 2-vector.
# STRATEGY: draw a filled contour with negative-gradient arrows (they all point to the
#           minimum), then step x <- x - alpha*grad and plot the path to (0, 0).
# =====================================================================================

using CairoMakie
using LinearAlgebra

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

x1range = -5:0.02:5
x2range = -5:0.02:5
bowl(x) = x[1]^2 + x[2]^2

function Gradient_bowl(x)
    d1 = 2 * x[1]
    d2 = 2 * x[2]
    return [d1, d2]
end

funcplot = [bowl([x1, x2]) for x1 in x1range, x2 in x2range]

# Contour with negative-gradient arrows.
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "Bowl function", limits = (-5, 5.0, -5.0, 5.0))
levels = range(0.01, 50; length = 6)
contourf!(ax, x1range, x2range, funcplot; levels = 50, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; labels = true, levels, color = :black, linewidth = 5)
scatter!(ax, [0], [0], color = :black, markersize = 20)

x1range_arrow = -5:1:5
x2range_arrow = -5:1:5
arrowplot = [Gradient_bowl([x1, x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax, x1range_arrow, x2range_arrow, -getindex.(arrowplot, 1), -getindex.(arrowplot, 2); color = (:black, 0.2), lengthscale = 0.1)
display(fig)

# Gradient descent from (1.5, 2.5).
x0 = [1.5, 2.5]
alpha = 0.1
data = [Point2D(x0..., bowl(x0))]
x = x0
for i in 1:20
    x -= alpha * Gradient_bowl(x)
    push!(data, Point2D(x..., bowl(x)))
end
lines!(ax, getproperty.(data, :x1), getproperty.(data, :x2); linewidth = 3, color = :grey)
scatter!(ax, x0[1], x0[2], color = :blue, markersize = 20)
scatter!(ax, getproperty.(data, :x1), getproperty.(data, :x2); color = :grey, markersize = 20, strokecolor = :black, strokewidth = 2)
# save("../../figures/my_plot_02.png", fig)   # (figure-save disabled in study file)
display(fig)
