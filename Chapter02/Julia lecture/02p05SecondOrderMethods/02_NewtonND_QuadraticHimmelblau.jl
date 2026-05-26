# =====================================================================================
# 2.5 - Newton's method in N dimensions
# =====================================================================================
# GOAL: generalise Newton to several variables, where the update becomes
#       x <- x - inv(H)*g using the Hessian MATRIX.
# STRATEGY: code the analytic gradient and Hessian, apply Newton to the Shewchuk
#           quadratic and to the Himmelblau function (comparing against gradient descent).
# =====================================================================================

using CairoMakie

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

# --- Shewchuk quadratic ---
quad(x) = (3.0 / 2.0) * x[1]^2 + 2 * x[1] * x[2] + 3 * x[2]^2 - 2 * x[1] + 8 * x[2]
function Gradient_quad(x)
    d1 = 3 * x[1] + 2 * x[2] - 2
    d2 = 6 * x[2] + 2 * x[1] + 8
    return [d1, d2]
end
Hessian_quad(x) = [3 2; 2 6]

x0 = [4.0, 4.0]
data_Newton = [Point2D(x0..., quad(x0))]
x = x0
for i in 1:20
    x -= Hessian_quad(x) \ Gradient_quad(x)
    push!(data_Newton, Point2D(x..., quad(x)))
end
data_Newton

# --- Himmelblau function ---
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
function Gradient_himmelblau(x)
    d1 = 4 * x[1] * (x[1]^2 + x[2] - 11) + 2 * (x[1] + x[2]^2 - 7)
    d2 = 2 * (x[1]^2 + x[2] - 11) + 4 * x[2] * (x[1] + x[2]^2 - 7)
    return [d1, d2]
end
function Hessian_himmelblau(x)
    return [12 * x[1]^2 + 4 * x[2] - 42 4 * x[1] + 4 * x[2]; 4 * x[1] + 4 * x[2] 12 * x[2]^2 + 4 * x[1] - 26]
end

x1range = -6:0.02:6
x2range = -6:0.02:6
funcplot = [himmelblau([x1, x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Himelblau function", limits = (-6, 6.0, -6.0, 6.0))
levels = 10.0 .^ range(-2, 4; length = 50)
contourf!(ax, x1range, x2range, funcplot; levels, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; labels = true, levels, color = :black)
scatter!(ax, [-0.270845], [-0.923039]; color = :magenta, markersize = 15)
scatter!(ax, [3.0, -2.805118, -3.779310, 3.584428], [2.0, 3.13132, -3.283136, -1.848126], color = :black, markersize = 15)
x1range_arrow = -6:1:6
x2range_arrow = -6:1:6
arrowplot = [Gradient_himmelblau([x1, x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax, x1range_arrow, x2range_arrow, -getindex.(arrowplot, 1), -getindex.(arrowplot, 2); color = (:grey, 0.5), lengthscale = 0.001)
display(fig)

# Gradient descent (slow) vs Newton (fast) on Himmelblau from (-2, -2).
x0 = [-2.0, -2.0]
alpha = 0.01
data = [Point2D(x0..., himmelblau(x0))]
x = x0
for i in 1:20
    x -= alpha * Gradient_himmelblau(x)
    push!(data, Point2D(x..., himmelblau(x)))
end
scatter!(ax, x0[1], x0[2], color = :blue, markersize = 20)
lines!(ax, getproperty.(data, :x1), getproperty.(data, :x2), color = :cyan, linewidth = 3, label = "Gradient Descent")

x0 = [-2.0, -2.0]
data_Newton = [Point2D(x0..., himmelblau(x0))]
x = x0
for i in 1:50
    x -= Hessian_himmelblau(x) \ Gradient_himmelblau(x)
    push!(data_Newton, Point2D(x..., himmelblau(x)))
end
lines!(ax, getproperty.(data_Newton, :x1), getproperty.(data_Newton, :x2), linewidth = 3, color = :green, label = "Newton")
fig

# TODO (Exercise Ex02): Rerun Newton from (-0.5, -1.0). Can you explain what happens?
