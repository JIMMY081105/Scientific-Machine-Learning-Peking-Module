########################################################################################
# 2.4 Multivariate Examples (coded from scratch, 2-D)
########################################################################################
#
# PROBLEM
# Extend the hand-coded first-order methods to functions of two variables, where the
# gradient is a 2-vector, and visualise why 'follow the negative gradient' works.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Write each 2-D function together with a Gradient function returning [df/dx1,
#      df/dx2].
#   2. Draw a filled contour plot and overlay arrows of the negative gradient to show
#      they all point toward the minimum.
#   3. Example 2.4.1 (bowl x1^2+x2^2): run gradient descent from a start point and
#      plot the path converging to (0,0).
#   4. Example 2.4.2 (Shewchuk quadratic): run Gradient Descent, then Momentum, then
#      Nesterov Momentum (all in vector form) and compare their paths on the contour
#      plot.
#
# This file is notebook "02p04MultivariateExamples" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ===============================
# 2.4 Multivariate Examples
# ===============================
# In the previous notebook, we used gradient methods to find minimum of function of
# one variable.

using CairoMakie
using LinearAlgebra
# Declare structs to store the store the intermediate solutions

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

# =======================
# **Example 2.4.1**
# =======================

x1range=-5:0.02:5;
x2range=-5:0.02:5;
bowl(x)=x[1]^2+x[2]^2

function Gradient_bowl(x)
    d1=2*x[1]
    d2=2*x[2]
    return [d1,d2]
end

funcplot = [bowl([x1,x2]) for x1 in x1range, x2 in x2range]

x = [(x1i, x2i) for x1i in x1range, x2i in x2range]
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Bowl function",limits=(-5,5.0,-5.0,5.0))
levels = range(0.01, 50; length=6)
contourf!(ax,x1range,x2range,funcplot; levels=50, colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels, color=:black, linewidth = 5)
scatter!(ax,[0],[0],color=:black,markersize=20)

x1range_arrow=-5:1:5;
x2range_arrow=-5:1:5;
arrowplot = [Gradient_bowl([x1,x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax,x1range_arrow,x2range_arrow,-getindex.(arrowplot,1),-getindex.(arrowplot,2);color=(:black, 0.2),lengthscale = 0.1)

# Save as PNG
# save("../figures/my_plot.png", fig)   # (figure-save disabled in study file)
display(fig)

# ==================================================
# Example 2.4.1 with Standard Gradient Descent
# ==================================================
# Try standard gradient descent method with update equations given by
# The code below uses standard gradient descent to find the local minimum.

x0=[1.5,2.5] #initial guess x0.  Since this is a two-dimensional function, you need two guesses, one for x1 and another for x2
alpha=0.1 #learning rate
data=[Point2D(x0...,bowl(x0))]
x=x0
for i in 1 : 20
    x-=alpha*Gradient_bowl(x)
    push!(data,Point2D(x...,bowl(x)))
end
lines!(ax,getproperty.(data, :x1),getproperty.(data, :x2);linewidth=3,color=:grey) 
scatter!(ax,x0[1],x0[2],color=:blue,markersize=20)
scatter!(ax,getproperty.(data, :x1),getproperty.(data, :x2);color=:grey,markersize=20,strokecolor = :black,strokewidth = 2,) 

# Save as PNG
# save("../figures/my_plot_02.png", fig)   # (figure-save disabled in study file)
display(fig)

# =======================
# **Example 2.4.2**
# =======================

# TODO (Exercise Ex01): Compute \partial f/\partial x_1 and \partial f/\partial x_2 and set them to zero
#   to show that the minimum of this function occurs at (x_1,x_2)=(2,-2).

quad(x)=(3.0/2.0)*x[1]^2+2*x[1]*x[2]+3*x[2]^2-2*x[1]+8*x[2] # Define the function
function Gradient_quad(x)
    d1=3*x[1]+2*x[2]-2 #df/dx1
    d2=6*x[2]+2*x[1]+8 #df/dx2
    return [d1,d2]
end


function Hessian_quad(x)
    return [3 2;2 6] # This is the Hessian of the function which we will use later.
end

x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [quad([x1,x2]) for x1 in x1range, x2 in x2range]


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Quadratic function",limits=(-6,6.0,-6.0,6.0))
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black,linewidth = 5)
contourf!(ax,x1range,x2range,funcplot;  levels=50,colormap=:bwr)

x1range_arrow=-6:1:6;
x2range_arrow=-6:1:6;
arrowplot = [Gradient_quad([x1,x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax,x1range_arrow,x2range_arrow,-getindex.(arrowplot,1),-getindex.(arrowplot,2);color=(:black, 0.2),lengthscale = 0.1)

scatter!(ax,[2],[-2],color=:black,markersize=20)

# Save as PNG
# save("../figures/my_plot_03.png", fig)   # (figure-save disabled in study file)
display(fig)

# ==================================================
# Example 2.4.2 with Standard Gradient Descent
# ==================================================
# The code below uses the standard gradient descent method to find the local minimum
# of the quadratic function defined above.

x0=[4.0,4.0] #initial guess of xm
alpha=0.1 #learning rate
data=[Point2D(x0...,quad(x0))]

x=x0

for i in 1 : 50
    x-=alpha*Gradient_quad(x)
    push!(data,Point2D(x...,quad(x)))
end

lines!(ax,getproperty.(data, :x1),getproperty.(data, :x2);linewidth=3,color=:grey,label="Gradient Descent") 
scatter!(ax,getproperty.(data, :x1),getproperty.(data, :x2);color=:grey,markersize=20,strokecolor = :black,strokewidth = 2,) 

#scatter!(ax,x0[1],x0[2],color=:blue,markersize=20)
fig

# =================================
# Example 2.4.2 with Momentum
# =================================
# The update equations for the momentum method for a function of two variables are
# The Julia code to implement the momentum equations is given below

x0=[4.0,4.0] #initial guess of x0
alpha=0.1 #learning rate
beta=0.4 #momentum
v=[0.0,0.0]
data_m=[Point2D(x0...,quad(x0))]
x=x0
for i in 1 : 20
    v=beta*v.-alpha*Gradient_quad(x)
    x+=v
    push!(data_m,Point2D(x...,quad(x)))
end


lines!(ax,getproperty.(data_m, :x1),getproperty.(data_m, :x2),linewidth=3,color=:green,label="Momentum") 
scatter!(ax,getproperty.(data_m, :x1),getproperty.(data_m, :x2);color=:green,markersize=20,strokecolor = :black,strokewidth = 2,) 

axislegend(ax;position = :lt)

# Save as PNG
# save("../figures/my_plot_04.png", fig)   # (figure-save disabled in study file)
display(fig)

# TODO (Class Demo 01): Write a Julia program that implements the momentum method to find the minimum of
#   the two dimensional function f(x_1,x_2)=\frac{3}{2}x_1^2+2x_1x_2+3x_2^2-2x_1+8x_2

# ===========================================
# Example 2.4.2 with Nesterov Momentum.
# ===========================================
# Let's use Nesterov momentum.
# The Julia code for the Nesterov momentum method is shown below.

x0=[4.0,4.0] #initial guess of xm
alpha=0.1 #learning rate
beta=0.4
v=[0.0,0.0]
x=x0
data_nm=[Point2D(x0...,quad(x0))]
for i in 1 : 20
    v =beta*v-alpha*Gradient_quad(x+beta*v)
    x+=v
    push!(data_nm,Point2D(x...,quad(x)))
end

lines!(ax,getproperty.(data_nm, :x1),getproperty.(data_nm, :x2),linewidth=5,color=:red,label="Nesterov Momentum") 
axislegend(ax;position = :lt)
fig
# As you can see from the example above, the Newton method converges very quickly for
# this method.
