########################################################################################
# 1.3 Plotting Functions in Julia
########################################################################################
#
# PROBLEM
# Learn to plot the standard test functions that later optimisation algorithms will
# be run on, so you can see minima and maxima before trying to compute them.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Plot a simple 1-D quadratic and mark its minimum with a scatter point.
#   2. Plot the harder 1-D function (x^3 - x)^2 and mark all its minima/maxima.
#   3. Build a 2-D grid with an array comprehension and draw filled contour plots of
#      the Himmelblau function (4 minima, 1 maximum).
#   4. Do the same for the Rosenbrock function, whose minimum sits in a long narrow
#      valley that is hard for algorithms to find.
#
# This file is notebook "01p03PlottingFunctionsInJulia" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################

# In this course I will also show you how you can use the `CairoMakie` (which is part
# of the Makie.jl) plotting package to plot graphs.

using CairoMakie
# In many engineering problems, we will be required to find the minimum of a
# function.

f(x)=x^2-2x+0.6;

xplot=0:0.01:2

Array(xplot)
# We will evaluate function `f` at every element of `xplot` so that we can plot the
# graph

f.(xplot)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
lines!(ax,xplot,f.(xplot)) #main code to plot f(x)
hlines!(ax,[0],color=:black,linewidth=1.0)
scatter!(ax,1,f.(1);color=:black,markersize=15)
# save("../figures/01p03PlottingFunctionsInJulia01.svg", fig) #save figure as svg file   # (figure-save disabled in study file)
display(fig)

# TODO (Exercise Ex01): Show that this function has a minimum at x=0,\pm 1 and maximum at x=0,\pm
#   1/\sqrt{3}
# We will now introduce a more complex function which we will use later on in the
# course

f(x)=(x^3-x)^2
xplot=-2.0:0.01:2.0
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-2,2,-0.01,0.2))
lines!(ax,xplot,f.(xplot))
scatter!(ax,[0, -1, 1],f.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax,[1/sqrt(3), -1/sqrt(3)],f.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)
display(fig)

# TODO (Class Demo 01): Write a Julia program to plot the function A(r)=2\pi r^2+\frac{2}{r}

himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2 #the input of himmelblau is x which is a vector of two elements.

x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [himmelblau([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x_1", 
    ylabel = L"x_2",
    xlabelsize = 20, 
    ylabelsize = 20,
    title = "Himelblau function",
    limits=(-6,6.0,-6.0,6.0))
levels = 10.0.^range(0, 3.5; length=10)
contourf!(ax,x1range,x2range,funcplot; levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; levels,label=true,color=:black)
# save("../figures/01p03PlottingFunctionsInJulia02.svg", fig) #save figure as svg file   # (figure-save disabled in study file)

scatter!(ax,-0.270845,-0.923039;markersize=20,color=:magenta)
scatter!(ax,3.0,2.0,markersize=20,color=:grey)
scatter!(ax,-2.805118,3.13132,markersize=20,color=:grey)
scatter!(ax,-3.779310,-3.283136,markersize=20,color=:grey)
scatter!(ax,3.584428,-1.848126,markersize=20,color=:grey)


display(fig)
# A few points to note

# =========================
# Rosenbrock function
# =========================

# TODO (Exercise Ex02): Show that the Rosenbrock function has a local minimum at (x_1,x_2)=(p_1,p_1^2).
#   For simplicity, we will use the value p_1=1 and p_2=1. For these values of p_1
#   and p_2, f(x_1,x_2) will have a local minimum at (x_1,x_2)=(1,1). Let's plot the
#   Rosenbrock function and see how it looks like
# Another common function to test optimization algorithm is the Rosenbrock function

x1range=-5:0.02:5;
x2range=-5:0.02:5;

rosenbrock(x)=(1.0 - x[1])^2 + 1.0 * (x[2] - x[1]^2)^2
funcplot = [rosenbrock([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x_1", 
    ylabel = L"x_2",
    xlabelsize = 20, 
    ylabelsize = 20,
    title = "Rosenbrock function",limits=(-5,5.0,-5.0,5.0))
levels = 10.0.^range(-2, 3.5; length=20)
contourf!(ax,x1range,x2range,funcplot; levels, colormap=:bwr)
contour!(ax,x1range,x2range,funcplot;  levels, color=:black)
scatter!(ax,[1],[1];color=:grey,markersize=15)
display(fig)

# TODO (Exercise Ex03): Plot the Rosenbrock function for different values of p_1 and p_2 and comment on
#   how the function changes.

# TODO (Class Demo 02): . Write a Julia program to do a contour plot of the function
#   f(\vec{x})=\sin(x_1)\cos(3x_2)
