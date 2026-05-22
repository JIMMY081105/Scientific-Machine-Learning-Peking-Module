# =====================================================================================
# 1.3 - Plotting 2-D test functions as contours
# =====================================================================================
# GOAL: draw filled contour plots of the 2-D functions used to test optimisers.
# STRATEGY: build a grid with an array comprehension, then contourf!/contour! with
#           CairoMakie; mark the known minima and maxima.
# =====================================================================================

using CairoMakie

# --- Himmelblau function (4 minima, 1 maximum) ---
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

x1range = -6:0.02:6
x2range = -6:0.02:6
funcplot = [himmelblau([x1, x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 20, ylabelsize = 20, title = "Himelblau function", limits = (-6, 6.0, -6.0, 6.0))
levels = 10.0 .^ range(0, 3.5; length = 10)
contourf!(ax, x1range, x2range, funcplot; levels, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; levels, label = true, color = :black)
scatter!(ax, -0.270845, -0.923039; markersize = 20, color = :magenta) # maximum
scatter!(ax, 3.0, 2.0, markersize = 20, color = :grey)                # minima
scatter!(ax, -2.805118, 3.13132, markersize = 20, color = :grey)
scatter!(ax, -3.779310, -3.283136, markersize = 20, color = :grey)
scatter!(ax, 3.584428, -1.848126, markersize = 20, color = :grey)
# save("../../figures/01p03PlottingFunctionsInJulia02.svg", fig)   # (figure-save disabled in study file)
display(fig)

# --- Rosenbrock function (minimum in a long narrow valley) ---
x1range = -5:0.02:5
x2range = -5:0.02:5
rosenbrock(x) = (1.0 - x[1])^2 + 1.0 * (x[2] - x[1]^2)^2
funcplot = [rosenbrock([x1, x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 20, ylabelsize = 20, title = "Rosenbrock function", limits = (-5, 5.0, -5.0, 5.0))
levels = 10.0 .^ range(-2, 3.5; length = 20)
contourf!(ax, x1range, x2range, funcplot; levels, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; levels, color = :black)
scatter!(ax, [1], [1]; color = :grey, markersize = 15)
display(fig)

# TODO (Exercise Ex03): Plot Rosenbrock for different p1, p2 and comment on the changes.
# TODO (Class Demo 02): Contour-plot f(x) = sin(x1)*cos(3*x2).
