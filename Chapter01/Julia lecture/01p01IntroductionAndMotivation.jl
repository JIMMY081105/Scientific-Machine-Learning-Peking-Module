########################################################################################
# 1.1 Introduction & Motivation
########################################################################################
#
# PROBLEM
# Design a cylinder that holds a fixed volume of water using the least material. This
# means minimising the surface area A(r) = 2*pi*r^2 + 2/r. It introduces the core
# idea of the course: many engineering goals become 'find the value that minimises a
# function'.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Load a plotting package (CairoMakie).
#   2. Write the surface-area function A(r) as a plain Julia function.
#   3. Sweep r over a range and plot A(r) to eyeball where the minimum sits (around r
#      = 0.5 m).
#   4. Generalise to A(r, V) when the volume V is also free, and plot one curve per V
#      to see how the shape shifts.
#   5. Conclude: reading a minimum off a graph works but is crude; later notebooks
#      compute it exactly.
#
# This file is notebook "01p01IntroductionAndMotivation" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################

# In this notebook, we will discuss the following points
# Import package to plot graphs in Julia

using CairoMakie

# ====================================================================
# Why do we need to find the minimum (or maximum) of a function?
# ====================================================================
# There are many engineering applications where we need to find the minimum of a
# function.

# TODO (Exercise Ex01): You want to make a cylinder that can store up to 1 m ^3 of water. Show that the
#   area of the cylinder is given by A(r)=2\pi r^2+\frac{2}{r} What is the radius r
#   such that you minimize the surface area of the cylinder (so that you need the
#   least amount of material to make this cylinder)?

A(r)=2π*r^2+2/r # Define function A(r)
rplot=0.1:0.01:2 #define the range of the plot
fig = Figure() #set up a figure
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)") #Define axis in the figure
lines!(ax,rplot,A.(rplot);color=:black) #Plot line
# save("../figures/01p01IntroductionAndMotivation01.svg", fig) #save figure as svg file   # (figure-save disabled in study file)
fig #show the figure

# TODO (Exercise Ex02): Show that the minimum of A(r) occurs at r=\sqrt[3]{\frac{1}{2\pi}}

# TODO (Exercise Ex03): Go through the exercise above again and show that A(r,V)=2\pi r^2+\frac{2V}{r}

A(r,V)=2π*r^2+2*V/r # Define function A(r,V)
Vplot=0.5:0.5:2
rplot=0.1:0.01:2 #define the range of the plot
fig = Figure() #set up a figure
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)") #Define axis in the figure
for i=1:length(Vplot)
    lines!(ax,rplot,A.(rplot,Vplot[i]),label=L"V=%$(Vplot[i])") 
end
axislegend(ax,position=:rt) #add legend to the axis
# save("../figures/01p01IntroductionAndMotivation02.svg", fig) #save figure as svg file   # (figure-save disabled in study file)
fig #show the figure

# ====================
# Why Julia?
# ====================
# Julia, on the other hand is quite a new language designed to be fast.

# ======================================
# Why Scientific Machine Learning?
# ======================================
