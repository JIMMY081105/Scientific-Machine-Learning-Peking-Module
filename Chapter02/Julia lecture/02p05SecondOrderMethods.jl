########################################################################################
# 2.5 Second Order Methods (Newton, coded from scratch)
########################################################################################
#
# PROBLEM
# Build the Newton method by hand using the second derivative (Hessian), see why it
# converges so fast, and understand its key danger: it seeks any point where the
# gradient is zero, so it can converge to a maximum.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Write h(x), its gradient, and its Hessian (second derivative) explicitly.
#   2. Derive Newton from a 2nd-order Taylor fit: x <- x - g/H; illustrate one step
#      graphically (fit a parabola at x0, jump to its minimum).
#   3. Run Newton on (x^3-x)^2 and compare its ~3-iteration convergence against
#      Nesterov Momentum.
#   4. Show the pitfall: starting near a maximum, Newton converges to the maximum
#      while Nesterov Momentum still reaches the true minimum.
#   5. Generalise to N-D: the update becomes x <- x - inv(H)*g using the Hessian
#      matrix; apply it to the Shewchuk quadratic and the Himmelblau function (with
#      analytic gradient and Hessian).
#
# This file is notebook "02p05SecondOrderMethods" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ==============================
# 2.5 Second Order Methods
# ==============================
# Load plotting library and define structure to store intermediate data

using CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end

# =======================
# **Example 2.5.1**
# =======================

h(x)=(x^3-x)^2
∇h(x)=2*(x^3-x)*(3x^2-1);
Hessianh(x)=12*x*(x^3-x)+2*(3*x^2-1)^2; # Second derivative
xplot=-2:0.01:2
yplot=[h(x) for x in xplot];
fig = Figure(backgroundcolor = :transparent,size=(800,600))
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"$h(x)$",limits=(-2,2,-0.1,0.2),backgroundcolor = :transparent)
lines!(ax,xplot,yplot;color=:blue)
scatter!(ax,[0, -1, 1],h.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax,[1/sqrt(3), -1/sqrt(3)],h.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)

# Save the figure
# save("../figures/ManyMinimums01.png", fig)   # (figure-save disabled in study file)

display(fig)
# I will now graphically illustrate the Newton method.

fig = Figure(backgroundcolor = :transparent,size=(800,600))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = L"$h(x)$",limits=(-2,2,-0.1,0.2),backgroundcolor = :transparent)
lines!(ax,xplot,yplot;color=:blue,label="True function")


x0=0.9
f2ndOrderTaylor(x,x0)=h(x0)+∇h(x0)*(x-x0)+(Hessianh(x0)/2.0)*(x-x0)^2
lines!(ax,xplot,f2ndOrderTaylor.(xplot,x0);color=:green, label="Taylors Second order polynomial")

xnewton=x0-∇h(x0)/Hessianh(x0)

scatter!(ax,xnewton,f2ndOrderTaylor(xnewton,x0);color=:green,markersize=15)
scatter!(ax,x0,h(x0);color=:red,markersize=15)

scatter!(ax,[0, -1, 1],h.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax,[1/sqrt(3), -1/sqrt(3)],h.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)

axislegend(ax,position=:lt)


# Save the figure
# save("../figures/ManyMinimums02.png", fig)   # (figure-save disabled in study file)

display(fig)
# We can code up the update equations for the Newton method

x0=0.9 #initial guess of x0
x=x0
data_Newton=[Point1D(x0,h(x0))]
for i in 1 : 20
    x-=Hessianh(x)\∇h(x) #update equation for the Newton method
    push!(data_Newton,Point1D(x,h(x)))
end

data_Newton
# We will now plot the convergence to see how fast the Newton method converges

fig_conv = Figure()

ax_conv1 = Axis(fig_conv[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-2,2,-0.01,0.2))
lines!(ax_conv1,xplot,h.(xplot))
scatter!(ax_conv1,getproperty.(data_Newton,:x),getproperty.(data_Newton,:fvalue);
    color=1:length(data_Newton),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)


ax_conv2 = Axis(fig_conv[1, 2], xlabel = "iter", ylabel = L"x")
lines!(ax_conv2,getproperty.(data_Newton, :x),label="Gradient Descent")
scatter!(ax_conv2,getproperty.(data_Newton,:x);
    color=1:length(data_Newton),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)
hlines!(ax_conv2,1.0;color=:red)

scatter!(ax_conv1,[0, -1, 1],h.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax_conv1,[1/sqrt(3), -1/sqrt(3)],h.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)
fig_conv

# Save the figure
# save("../figures/ManyMinimums02.png", fig_conv)   # (figure-save disabled in study file)

display(fig_conv)
# Let's try to compute the same solution using the Nesterov Momentum method and
# compare the rate of convergence with Newton method to see how fast it converges.

x0=0.9 #initial guess of xm
alpha=0.05 #learning rate
beta=0.9
v=0.0
x=x0
data_nm=[Point1D(x0,h(x0))]
for i in 1 : 20
    v =beta*v-alpha*∇h(x+beta*v) #update equations for the Nesterov Momentum 
    x+=v #update equations for the Nesterov Momentum 
    push!(data_nm,Point1D(x,h(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")

lines!(ax_conv,getproperty.(data_Newton, :x),label="Newton")
scatter!(ax_conv,getproperty.(data_Newton,:x);
    color=1:length(data_Newton),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

lines!(ax_conv,getproperty.(data_nm, :x),label="Nesterov Momentum",color=:orange)
scatter!(ax_conv,getproperty.(data_nm,:x);
    color=1:length(data_nm),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Greens,
    markersize=15)

axislegend(ax_conv,  position = :rt,orientation = :vertical)
hlines!(ax_conv,1.0;color=:red,linewidth=5.0)
fig_conv

# ===============================================
# Beware some pitfalls of the Newton Method
# ===============================================

x0=0.4 #initial guess of x

data_Newton=[Point1D(x0,h(x0))]
for i in 1 : 20
    x0-=Hessianh(x0)\∇h(x0)
    push!(data_Newton,Point1D(x0,h(x0)))
end
# Let's plot the convergence of Newton method

fig_conv = Figure()

ax_conv1 = Axis(fig_conv[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-2,2,-0.01,0.2))
lines!(ax_conv1,xplot,h.(xplot))
scatter!(ax_conv1,getproperty.(data_Newton,:x),getproperty.(data_Newton,:fvalue);
    color=1:length(data_Newton),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)


ax_conv2 = Axis(fig_conv[1, 2], xlabel = "iter", ylabel = L"x_m")
lines!(ax_conv2,getproperty.(data_Newton, :x),label="Gradient Descent")
scatter!(ax_conv2,getproperty.(data_Newton,:x);
    color=1:length(data_Newton),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)
hlines!(ax_conv2,+sqrt(1/3);color=:red)

scatter!(ax_conv1,[0, -1, 1],h.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax_conv1,[1/sqrt(3), -1/sqrt(3)],h.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)
fig_conv

x0=0.4 #initial guess of x0
alpha=0.05 #learning rate
beta=0.9
v=0.0
x=x0
data_nm=[Point1D(x0,h(x0))]
for i in 1 : 20
    v =beta*v-alpha*∇h(x+beta*v) #update equations for the Nesterov Momentum 
    x+=v #update equations for the Nesterov Momentum 
    push!(data_nm,Point1D(x,h(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x")

lines!(ax_conv,getproperty.(data_Newton, :x),color=:blue,label="Newton")
scatter!(ax_conv,getproperty.(data_Newton,:x);
    color=1:length(data_Newton),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

lines!(ax_conv,getproperty.(data_nm, :x),color=:green,label="Nesterov Momentum",)
scatter!(ax_conv,getproperty.(data_nm,:x);
    color=1:length(data_nm),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Greens,
    markersize=15)

axislegend(ax_conv,  position = :rt,orientation = :vertical)
hlines!(ax_conv,sqrt(1.0/3.0);color=:red,linewidth=2.0)
hlines!(ax_conv,0.0;color=:red,linewidth=2.0)
fig_conv

# TODO (Class Demo 01): Write a Julia program that implements the Newton's method to find the minimum of
#   the function A(r)=2\pi r^2+\frac{2}{r}

# ===============================================
# Newton Method For Multivariable Functions
# ===============================================
# Differentiate the 2nd order Taylor series expansion and set to zero gives

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

quad(x)=(3.0/2.0)*x[1]^2+2*x[1]*x[2]+3*x[2]^2-2*x[1]+8*x[2] # Define the function
function Gradient_quad(x)
    d1=3*x[1]+2*x[2]-2 #df/dx1
    d2=6*x[2]+2*x[1]+8 #df/dx2
    return [d1,d2]
end


function Hessian_quad(x)
    return [3 2;2 6] # This is the Hessian of the function which we will use later.
end

x0=[4.0,4.0]#initial guess of xm

data_Newton=[Point2D(x0...,quad(x0))]
x=x0
for i in 1 : 20
    x-=Hessian_quad(x)\Gradient_quad(x)
    push!(data_Newton,Point2D(x...,quad(x)))
end

data_Newton
# Let's now try the Himmelblau Function
# Define the Himmelblau function, its gradients and Hessian and plotting it to see
# how it looks like.

himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2


function Gradient_himmelblau(x)
    d1=4*x[1]*(x[1]^2+x[2]-11)+2*(x[1]+x[2]^2-7)
    d2=2*(x[1]^2+x[2]-11)+4*x[2]*(x[1]+x[2]^2-7)
    return [d1,d2]
end

function Hessian_himmelblau(x)
    return [12*x[1]^2+4*x[2]-42 4*x[1]+4*x[2];4*x[1]+4*x[2] 12*x[2]^2+4*x[1]-26]
end

x1range=-6:0.02:6;
x2range=-6:0.02:6;

funcplot = [himmelblau([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y",title = "Himelblau function",limits=(-6,6.0,-6.0,6.0))
levels = 10.0.^range(-2, 4; length=50)
contourf!(ax,x1range,x2range,funcplot; levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels,color=:black)
scatter!(ax,[-0.270845],[-0.923039];color=:magenta,markersize=15)
scatter!(ax,[3.0,-2.805118,-3.779310,3.584428],[2.0,3.13132,-3.283136,-1.848126],color=:black,markersize=15)

x1range_arrow=-6:1:6;
x2range_arrow=-6:1:6;
arrowplot = [Gradient_himmelblau([x1,x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax,x1range_arrow,x2range_arrow,-getindex.(arrowplot,1),-getindex.(arrowplot,2);color=(:grey, 0.5),lengthscale = 0.001)

display(fig)
# Using Standard Gradient Descent

x0=[-2.0,-2.0] #initial guess of xm
alpha=0.01 #learning rate
data=[Point2D(x0...,himmelblau(x0))]
x=x0
for i in 1 : 20
    x-=alpha*Gradient_himmelblau(x)
    push!(data,Point2D(x...,himmelblau(x)))
end
scatter!(ax,x0[1],x0[2],color=:blue,markersize=20)
lines!(ax,getproperty.(data, :x1),getproperty.(data, :x2),color=:cyan,linewidth=3,label="Gradient Descent") 
fig

x0=[-2.0,-2.0]#initial guess of x0

data_Newton=[Point2D(x0...,himmelblau(x0))]

x=x0
for i in 1 : 50
    x-=Hessian_himmelblau(x)\Gradient_himmelblau(x)
    push!(data_Newton,Point2D(x...,himmelblau(x)))
end

lines!(ax,getproperty.(data_Newton, :x1),getproperty.(data_Newton, :x2),linewidth=3,color=:green,label="Newton") 
fig

data_Newton

# TODO (Exercise Ex02): Run Newton method again but use the initial guess (x_1,x_2)=(-0.5,-1.0). Can you
#   explain what happens?
