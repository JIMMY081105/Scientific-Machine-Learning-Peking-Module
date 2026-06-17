########################################################################################
# 2.3 Gradient Based Methods (coded from scratch, 1-D)
########################################################################################
#
# PROBLEM
# Understand the first-order optimisers by implementing their update equations
# yourself (no library) for 1-D functions, using an explicitly written gradient.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Write the function f(x) and its gradient df/dx by hand.
#   2. Gradient Descent: loop x <- x - alpha*g and store each guess; watch it slow
#      down near the minimum as the gradient shrinks.
#   3. Momentum: keep a velocity v <- beta*v - alpha*g, then x <- x + v, so past
#      gradients are remembered (beta=0 recovers plain gradient descent).
#   4. Nesterov Momentum: same as Momentum but evaluate the gradient at the
#      look-ahead point x + beta*v, which reduces overshoot.
#   5. Overlay the three convergence curves; then repeat on g(x) = -exp(-x^2) where
#      the tiny far-field gradient makes plain gradient descent very slow.
#
# This file is notebook "02p03GradientBasedMethods" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ================================
# 2.3 Gradient Based Methods
# ================================
# Load the libraries needed for this notebook

using CairoMakie
# Define a struct to store data during iterations

struct Point1D
    x::Float64 #guess locations of the minimum
    fvalue::Float64 # value of the function at the guess locations
end

# =======================
# **Example 2.3.1**
# =======================

# TODO (Exercise Ex01): Prove to yourself that this function has a local minimum at x=1.

f(x)=x^2-2x+0.6 #Define the function
∇f(x)=2x-2 #Define the gradient of the function


xrange=-1:0.01:3
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-1,3,-0.5,2.0))
lines!(ax,xrange,f.(xrange))
scatter!(ax,1.0,f.(1.0);color=:black,markersize=15)
display(fig)

# ======================================
# Standard Gradient Descent method
# ======================================

x0=2.5 #initial guess of x0
alpha=0.1 #learning rate
data=[Point1D(x0,f(x0))] #Store the initial value of the guess and the function value into a struct called data
x=x0
for i in 1 : 20
    x-=alpha*∇f(x) #standard gradient descent update equation
    push!(data,Point1D(x,f(x))) #store x and f(x) into data
end

fig_conv = Figure(size=(1000, 400))

ax_conv1 = Axis(fig_conv[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-1,3,-0.5,2.0))
lines!(ax_conv1,xrange,f.(xrange))
scatter!(ax_conv1,getproperty.(data,:x),getproperty.(data,:fvalue);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)


ax_conv2 = Axis(fig_conv[1, 2], xlabel = "iter", ylabel = L"x")
lines!(ax_conv2,getproperty.(data, :x),label="Gradient Descent")
scatter!(ax_conv2,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)
hlines!(ax_conv2,1.0;color=:red)
# save("../figures/02p03GradientBasedMethods01.svg",fig_conv)   # (figure-save disabled in study file)
display(fig_conv)

# TODO (Exercise Ex02): Run the code above again with \alpha=0.01, 0.5 and 0.8. Comment on the convergece
#   of the solution for small and large values of \alpha.

# ========================================
# Example 2.3.1 with Momentum method
# ========================================

x0=2.5 #initial guess of x0
alpha=0.1 #learning rate
beta=0.6 #momentum
v=0.0 # setting the initial velocity to be zero.  
x=x0
data_m=[Point1D(x0,f(x0))]
for i in 1 : 20
    v=beta*v-alpha*∇f(x) #update equation for the momentum method. 
    x+=v #update equation for the momentum method. 
    push!(data_m,Point1D(x,f(x)))
end

# =================================================
# Example 2.3.1 with Nesterov Momentum method
# =================================================

x0=2.5 #initial guess of xm
alpha=0.1 #learning rate
beta=0.6
v=0.0
x=x0
data_nm=[Point1D(x0,f(x0))]
for i in 1 : 20
    v =beta*v-alpha*∇f(x+beta*v) #update equations for the Nesterov Momentum 
    x+=v #update equations for the Nesterov Momentum 
    push!(data_nm,Point1D(x,f(x)))
end
# Now we plot the convence to analyse the properties of the different methods.

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x")

lines!(ax_conv,getproperty.(data, :x),label="Gradient Descent")
scatter!(ax_conv,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

lines!(ax_conv,getproperty.(data_m, :x),label="Momentum",color=:orange)
scatter!(ax_conv,getproperty.(data_m,:x);
    color=1:length(data_m),
    strokewidth=1,
    strokecolor=:black,
    colormap = :OrRd,
    markersize=15)

lines!(ax_conv,getproperty.(data_nm, :x),label="Nesterov Momentum",color=:green)
scatter!(ax_conv,getproperty.(data_nm,:x);
    color=1:length(data_nm),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Greens,
    markersize=15)

axislegend(ax_conv,  position = :rt,orientation = :vertical)
hlines!(ax_conv,1.0;color=:red,linewidth=5.0)
# save("../figures/02p03GradientBasedMethods03.svg",fig_conv)   # (figure-save disabled in study file)
display(fig_conv)

# TODO (Exercise Ex03): Change the values of \alpha and \beta and see what happens to the convergence.

# =======================
# **Example 2.3.2**
# =======================

g(x)=-exp(-x^2)
∇g(x)=2*x*exp(-x^2)
xplot=-5:0.01:5
yplot=[g(x) for x in xplot];
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = L"$g(x)$",limits=(-5,5,-1.5,0.5))
lines!(ax,xplot,yplot)
fig

# ==================================================
# Example 2.3.2 with Standard Gradient Descent
# ==================================================

x0=2.0 #initial guess of xm
alpha=0.1 #learning rate
data=[Point1D(x0,g(x0))]
x=x0
for i in 1 : 100
    x-=alpha*∇g(x)
    push!(data,Point1D(x,g(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")
lines!(ax_conv,getproperty.(data, :x);label="Gradient Descent")
axislegend(ax_conv,  position = :rt,orientation = :vertical)

scatter!(ax_conv,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

hlines!(ax_conv,0.0;color=:red,linewidth=5.0)
fig_conv

# =================================
# Example 2.3.2 with Momentum
# =================================

x0=2.0 #initial guess of x0
alpha=0.1 #learning rate
beta=0.6 #momentum
v=0.0
x=x0
data_m=[Point1D(x0,g(x0))]
for i in 1 : 100
    v=beta*v-alpha*∇g(x)
    x+=v
    push!(data_m,Point1D(x,g(x)))
end

# ==========================================
# Example 2.3.2 with Nesterov Momentum
# ==========================================

x0=2.0 #initial guess of x0
alpha=0.1 #learning rate
beta=0.6
v=0.0
x=x0
data_nm=[Point1D(x0,g(x0))]
for i in 1 : 100
    v =beta*v-alpha*∇g(x+beta*v)
    x+=v
    push!(data_nm,Point1D(x,g(x)))
end
# Plotting the convergence and analysing the rate of convergence for all 3 methods.

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")
lines!(ax_conv,getproperty.(data, :x);label="Gradient Descent")


scatter!(ax_conv,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

hlines!(ax_conv,0.0;color=:red,linewidth=5.0)


lines!(ax_conv,getproperty.(data_m, :x),label="Momentum",color=:orange)
scatter!(ax_conv,getproperty.(data_m,:x);
    color=1:length(data_m),
    strokewidth=1,
    strokecolor=:black,
    colormap = :OrRd,
    markersize=15)

lines!(ax_conv,getproperty.(data_nm, :x),label="Nesterov Momentum",color=:green)
scatter!(ax_conv,getproperty.(data_nm,:x);
    color=1:length(data_nm),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Greens,
    markersize=15)

axislegend(ax_conv,  position = :rt,orientation = :vertical)
fig_conv

# TODO (Exercise Ex04): Write a Julia code to find the minimum of the function h(x)=(x^3-x)^2. Plot the
#   convergence of the scheme that you have chosen to use.

# TODO (Class Demo 01): Write a Julia code that uses Gradient Descent to find the minimum of the function
#   A(r)=2\pi r^2+2/r
# Write a Julia code that uses Gradient Descent to find the minimum of the function
