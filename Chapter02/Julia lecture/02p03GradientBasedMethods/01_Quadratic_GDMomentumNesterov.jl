# =====================================================================================
# 2.3 - Hand-coded first-order methods on a 1-D quadratic (Example 2.3.1)
# =====================================================================================
# GOAL: implement Gradient Descent, Momentum, and Nesterov Momentum yourself (no library)
#       for f(x) = x^2 - 2x + 0.6 with an explicit gradient.
# STRATEGY: GD steps x <- x - alpha*g; Momentum keeps a velocity v <- beta*v - alpha*g,
#           x <- x + v; Nesterov evaluates the gradient at the look-ahead x + beta*v.
#           Overlay the three convergence curves.
# =====================================================================================

using CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end

f(x) = x^2 - 2x + 0.6
∇f(x) = 2x - 2

xrange = -1:0.01:3

# --- Standard Gradient Descent ---
x0 = 2.5
alpha = 0.1
data = [Point1D(x0, f(x0))]
x = x0
for i in 1:20
    x -= alpha * ∇f(x)
    push!(data, Point1D(x, f(x)))
end

# --- Momentum ---
alpha = 0.1
beta = 0.6
v = 0.0
x = x0
data_m = [Point1D(x0, f(x0))]
for i in 1:20
    v = beta * v - alpha * ∇f(x)
    x += v
    push!(data_m, Point1D(x, f(x)))
end

# --- Nesterov Momentum (gradient at the look-ahead point) ---
v = 0.0
x = x0
data_nm = [Point1D(x0, f(x0))]
for i in 1:20
    v = beta * v - alpha * ∇f(x + beta * v)
    x += v
    push!(data_nm, Point1D(x, f(x)))
end

# Overlay the three convergence curves (x vs iteration).
fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x")
for (d, c, name) in ((data, :blue, "Gradient Descent"), (data_m, :orange, "Momentum"), (data_nm, :green, "Nesterov Momentum"))
    lines!(ax_conv, getproperty.(d, :x); label = name, color = c)
    scatter!(ax_conv, getproperty.(d, :x); color = 1:length(d), strokewidth = 1, strokecolor = :black, markersize = 15)
end
hlines!(ax_conv, 1.0; color = :red, linewidth = 5.0)
axislegend(ax_conv, position = :rt, orientation = :vertical)
# save("../../figures/02p03GradientBasedMethods03.svg", fig_conv)   # (figure-save disabled in study file)
display(fig_conv)

# TODO (Exercise Ex02): Rerun with alpha = 0.01, 0.5, 0.8 and comment on convergence.
# TODO (Exercise Ex03): Change alpha and beta and see what happens.
