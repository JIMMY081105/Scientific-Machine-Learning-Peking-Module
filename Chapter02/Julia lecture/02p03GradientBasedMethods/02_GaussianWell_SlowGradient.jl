# =====================================================================================
# 2.3 - Where plain gradient descent struggles (Example 2.3.2)
# =====================================================================================
# GOAL: minimise g(x) = -exp(-x^2), whose gradient is tiny far from the minimum, so plain
#       gradient descent crawls.
# STRATEGY: run the same hand-coded GD, Momentum, and Nesterov (100 iterations) and see
#           that the momentum methods escape the flat region far faster.
# =====================================================================================

using CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end

g(x) = -exp(-x^2)
∇g(x) = 2 * x * exp(-x^2)

xplot = -5:0.01:5
yplot = [g(x) for x in xplot]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = L"$g(x)$", limits = (-5, 5, -1.5, 0.5))
lines!(ax, xplot, yplot)
fig

# --- Standard Gradient Descent (slow: the far-field gradient is tiny) ---
x0 = 2.0
alpha = 0.1
data = [Point1D(x0, g(x0))]
x = x0
for i in 1:100
    x -= alpha * ∇g(x)
    push!(data, Point1D(x, g(x)))
end

# --- Momentum ---
beta = 0.6
v = 0.0
x = x0
data_m = [Point1D(x0, g(x0))]
for i in 1:100
    v = beta * v - alpha * ∇g(x)
    x += v
    push!(data_m, Point1D(x, g(x)))
end

# --- Nesterov Momentum ---
v = 0.0
x = x0
data_nm = [Point1D(x0, g(x0))]
for i in 1:100
    v = beta * v - alpha * ∇g(x + beta * v)
    x += v
    push!(data_nm, Point1D(x, g(x)))
end

# Overlay the three convergence curves.
fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")
for (d, c, name) in ((data, :blue, "Gradient Descent"), (data_m, :orange, "Momentum"), (data_nm, :green, "Nesterov Momentum"))
    lines!(ax_conv, getproperty.(d, :x); label = name, color = c)
    scatter!(ax_conv, getproperty.(d, :x); color = 1:length(d), strokewidth = 1, strokecolor = :black, markersize = 15)
end
hlines!(ax_conv, 0.0; color = :red, linewidth = 5.0)
axislegend(ax_conv, position = :rt, orientation = :vertical)
fig_conv

# TODO (Exercise Ex04): Minimise h(x) = (x^3 - x)^2 and plot the convergence.
# TODO (Class Demo 01): Use gradient descent to minimise A(r) = 2*pi*r^2 + 2/r.
