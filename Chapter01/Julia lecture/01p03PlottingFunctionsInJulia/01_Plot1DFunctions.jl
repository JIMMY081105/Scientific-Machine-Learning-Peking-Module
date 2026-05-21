# =====================================================================================
# 1.3 - Plotting 1-D test functions
# =====================================================================================
# GOAL: plot the 1-D functions later optimisers will run on and mark their minima/maxima.
# STRATEGY: broadcast the function over a range with the dot, then lines!/scatter! with
#           CairoMakie.
# =====================================================================================

using CairoMakie

# A simple quadratic with its minimum marked.
f(x) = x^2 - 2x + 0.6
xplot = 0:0.01:2
Array(xplot)
f.(xplot)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", limits = (0, 2, -0.5, 1.0))
lines!(ax, xplot, f.(xplot))
hlines!(ax, [0], color = :black, linewidth = 1.0)
scatter!(ax, 1, f.(1); color = :black, markersize = 15)
# save("../../figures/01p03PlottingFunctionsInJulia01.svg", fig)   # (figure-save disabled in study file)
display(fig)

# TODO (Exercise Ex01): Show this has minima at x = 0, +/-1 and maxima at x = +/-1/sqrt(3).

# A harder function with all minima (black) and maxima (magenta) marked.
f(x) = (x^3 - x)^2
xplot = -2.0:0.01:2.0
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", limits = (-2, 2, -0.01, 0.2))
lines!(ax, xplot, f.(xplot))
scatter!(ax, [0, -1, 1], f.([0, -1, 1]); color = :black, markersize = 15)
scatter!(ax, [1 / sqrt(3), -1 / sqrt(3)], f.([1 / sqrt(3), -1 / sqrt(3)]); color = :magenta, markersize = 15)
display(fig)

# TODO (Class Demo 01): Plot A(r) = 2*pi*r^2 + 2/r.
