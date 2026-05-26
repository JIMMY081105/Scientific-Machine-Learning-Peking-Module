# =====================================================================================
# 2.5 - Newton's method from scratch in 1-D, and its pitfall (Example 2.5.1)
# =====================================================================================
# GOAL: build Newton's method by hand from the second derivative (Hessian), see why it
#       converges so fast, and see its danger: it seeks any point where the gradient is
#       zero, so it can converge to a MAXIMUM.
# STRATEGY: derive Newton from a 2nd-order Taylor fit (x <- x - g/H), run it on
#           h(x) = (x^3 - x)^2, compare against Nesterov Momentum, then start near a
#           maximum and watch Newton climb to it while Nesterov still finds the minimum.
# =====================================================================================

using CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end

h(x) = (x^3 - x)^2
∇h(x) = 2 * (x^3 - x) * (3x^2 - 1)
Hessianh(x) = 12 * x * (x^3 - x) + 2 * (3 * x^2 - 1)^2 # second derivative
xplot = -2:0.01:2
yplot = [h(x) for x in xplot]

# One Newton step, illustrated: fit a parabola at x0 and jump to its minimum.
fig = Figure(backgroundcolor = :transparent, size = (800, 600))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = L"$h(x)$", limits = (-2, 2, -0.1, 0.2), backgroundcolor = :transparent)
lines!(ax, xplot, yplot; color = :blue, label = "True function")
x0 = 0.9
f2ndOrderTaylor(x, x0) = h(x0) + ∇h(x0) * (x - x0) + (Hessianh(x0) / 2.0) * (x - x0)^2
lines!(ax, xplot, f2ndOrderTaylor.(xplot, x0); color = :green, label = "Taylors Second order polynomial")
xnewton = x0 - ∇h(x0) / Hessianh(x0)
scatter!(ax, xnewton, f2ndOrderTaylor(xnewton, x0); color = :green, markersize = 15)
scatter!(ax, x0, h(x0); color = :red, markersize = 15)
scatter!(ax, [0, -1, 1], h.([0, -1, 1]); color = :black, markersize = 15)
scatter!(ax, [1 / sqrt(3), -1 / sqrt(3)], h.([1 / sqrt(3), -1 / sqrt(3)]); color = :magenta, markersize = 15)
axislegend(ax, position = :lt)
display(fig)

# Newton iteration: x <- x - g/H.
x0 = 0.9
x = x0
data_Newton = [Point1D(x0, h(x0))]
for i in 1:20
    x -= Hessianh(x) \ ∇h(x)
    push!(data_Newton, Point1D(x, h(x)))
end

# Compare Newton against Nesterov Momentum (Newton converges in ~3 iterations).
x0 = 0.9
alpha = 0.05
beta = 0.9
v = 0.0
x = x0
data_nm = [Point1D(x0, h(x0))]
for i in 1:20
    v = beta * v - alpha * ∇h(x + beta * v)
    x += v
    push!(data_nm, Point1D(x, h(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")
lines!(ax_conv, getproperty.(data_Newton, :x), label = "Newton")
scatter!(ax_conv, getproperty.(data_Newton, :x); color = 1:length(data_Newton), colormap = :Blues, strokewidth = 1, strokecolor = :black, markersize = 15)
lines!(ax_conv, getproperty.(data_nm, :x), label = "Nesterov Momentum", color = :orange)
scatter!(ax_conv, getproperty.(data_nm, :x); color = 1:length(data_nm), colormap = :Greens, strokewidth = 1, strokecolor = :black, markersize = 15)
axislegend(ax_conv, position = :rt, orientation = :vertical)
hlines!(ax_conv, 1.0; color = :red, linewidth = 5.0)
fig_conv

# --- The pitfall: start near a maximum (x0 = 0.4) ---
# Newton chases the zero-gradient point and converges to the MAXIMUM at 1/sqrt(3).
x0 = 0.4
data_Newton = [Point1D(x0, h(x0))]
for i in 1:20
    x0 -= Hessianh(x0) \ ∇h(x0)
    push!(data_Newton, Point1D(x0, h(x0)))
end

# Nesterov from the same start still reaches the true minimum.
x0 = 0.4
alpha = 0.05
beta = 0.9
v = 0.0
x = x0
data_nm = [Point1D(x0, h(x0))]
for i in 1:20
    v = beta * v - alpha * ∇h(x + beta * v)
    x += v
    push!(data_nm, Point1D(x, h(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x")
lines!(ax_conv, getproperty.(data_Newton, :x), color = :blue, label = "Newton")
scatter!(ax_conv, getproperty.(data_Newton, :x); color = 1:length(data_Newton), colormap = :Blues, strokewidth = 1, strokecolor = :black, markersize = 15)
lines!(ax_conv, getproperty.(data_nm, :x), color = :green, label = "Nesterov Momentum")
scatter!(ax_conv, getproperty.(data_nm, :x); color = 1:length(data_nm), colormap = :Greens, strokewidth = 1, strokecolor = :black, markersize = 15)
axislegend(ax_conv, position = :rt, orientation = :vertical)
hlines!(ax_conv, sqrt(1.0 / 3.0); color = :red, linewidth = 2.0) # the maximum Newton is drawn to
hlines!(ax_conv, 0.0; color = :red, linewidth = 2.0)             # the true minimum
fig_conv

# TODO (Class Demo 01): Implement Newton's method to minimise A(r) = 2*pi*r^2 + 2/r.
