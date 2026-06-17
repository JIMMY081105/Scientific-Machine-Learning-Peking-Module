########################################################################################
# 2.1 Basic Optimization in Julia (first-order library methods)
########################################################################################
#
# PROBLEM
# Find the minimum of functions using Julia's Optimization.jl library with the four
# common first-order (gradient-only) optimisers: Gradient Descent, Momentum, Nesterov
# Momentum, and Adam. Compare how each one converges.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Load Optimization + OptimizationOptimisers + Zygote (for automatic gradients)
#      + CairoMakie.
#   2. Define Point1D/Point2D structs and callback functions that record every
#      intermediate guess so the convergence path can be plotted.
#   3. The library expects the objective as objective(x, p) with x a vector; since
#      our functions have no parameters p, wrap them as (x,p)->f(x).
#   4. Standard three-step recipe: OptimizationFunction (function + AutoZygote
#      gradient) -> OptimizationProblem (with initial guess x0) -> solve(prob,
#      optimiser; maxiters, callback).
#   5. Run Example 2.1.1 (1-D quadratic) with Descent, Momentum, Nesterov, and Adam
#      in turn, overlaying each convergence path to compare speed and overshoot.
#   6. Repeat for Example 2.1.2 (2-D Shewchuk quadratic) and Example 2.1.3
#      (Rosenbrock), reading the final answer from sol.u and status from sol.retcode.
#
# This file is notebook "02p01BasicOptimizationInJulia" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# =====================================
# 2.1 Basic Optimization In Julia
# =====================================

using Optimization,OptimizationOptimisers,Zygote,CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end
struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

function my_callback1D(state, l)
    push!(vec_points,Point1D(state.u...,l))
    return false # continues the solver
end

function my_callback2D(state, l)

    push!(vec_points,Point2D(state.u...,l))
    return false # continues the solver
end;

# =======================
# **Example 2.1.1**
# =======================

# TODO (Exercise Ex01): Show that the minimum of this function occur at x=1.

f(x)=x[1]^2-2*x[1]+0.6;  #define the function to be optimized. 

                        #NOTE that in order to use the optimisation functions, 
                        # Julia expects the input of the function to be a vector, 
                        # even if the function is univariate

xplot=0:0.01:2 #define the range to plot the function
yplot=[f(x) for x in xplot];
fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x", 
    ylabel = L"f(x)",
    xlabelsize = 20, 
    ylabelsize = 20,
    limits=(0,2,-0.5,1.0))
lines!(ax,xplot,yplot)
scatter!(ax,1,f.(1);color=:black,markersize=15)
# save("../figures/02p01BasicOptimizationInJulia01.svg", fig)   # (figure-save disabled in study file)
display(fig)
# To use the `Optimization.jl` library to find the minimum of the function above, you
# need to first

# =========================================
# Example 2.1.1 with Gradient Descent
# =========================================

x0=1.8*ones(1)

temp=(x,p)->f(x)    #create a temporary function.  This function takes in two inputs x and p, but only uses x.  
                    #It will have the same output as f(x).
println(temp(4,3))
println(f(4))

x0=1.8*ones(1) #Define initial guess
optf = OptimizationFunction((x,p)->f(x), ADTypes.AutoZygote()) #define function to optimize and its derivative which is given by ADTypes.AutoZygote()
#optf = OptimizationFunction(temp, ADTypes.AutoZygote()) 

prob = OptimizationProblem(optf, x0) #define initial guess x=1.8
sol = solve(prob, Optimisers.Descent(0.1);maxiters=20) #solve the problem and take 20 iterations. Optimisers.Descent(0.1) asks solve to use the gradient descent optimiser with learning rate =0.1.

sol.u

vec_points=[] #Define empty vector to store intermediate solutions

optf = OptimizationFunction((x,p)->f(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.1);maxiters=20,callback=my_callback1D); #the 0.1 is called the learning rate and it is a parameter you can tune in the `solve()' function

vec_points_gd=copy(vec_points);  #Copy information about the intermediate solution to vec_points_gd
vec_points
# A few points to note with the numbers above

fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x",limits=(0,20,0.0,2.0))
lines!(ax1,xplot,yplot;color=:black)
scatter!(ax1,getproperty.(vec_points[:],:x),getproperty.(vec_points[:],:fvalue);markersize=15,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black,label="Gradient Descent")
scatter!(ax2,getproperty.(vec_points[:],:x);markersize=15,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points[:],:x);color=:blue,label="Gradient Descent")
hlines!(ax2, 1.0, color=:red, linestyle=:dash) 
axislegend(ax2;position = :rt)
# save("../figures/02p01BasicOptimizationInJulia02.svg", fig)   # (figure-save disabled in study file)
display(fig)

# TODO (Exercise Ex02): Repeat the example above with different values of learning rate \alpha. Can you
#   think of the advantages/disadvantages of using small/large values of \alpha? Try
#   with \alpha \in [0.01,0.8]

# ========================================
# Example 2.1.1 with Momentum method
# ========================================
# We will plot the solution over the standard gradient descent solution.

x0=1.8*ones(1) #Define initial guess
vec_points=[]
sol = solve(prob, Optimisers.Momentum(0.1,0.9);maxiters=20,callback=my_callback1D) #solve the problem and take 10 iterations. 

vec_points_m=copy(vec_points)

fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x",limits=(0,20,0.0,2.0))
lines!(ax1,xplot,yplot;color=:black)
scatter!(ax1,getproperty.(vec_points_gd[:],:x),getproperty.(vec_points_gd[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black,label="Gradient Descent")
scatter!(ax2,getproperty.(vec_points_gd[:],:x);markersize=10,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_gd[:],:x);color=:blue,label="Gradient Descent")
hlines!(ax2, 1.0, color=:red, linestyle=:dash) 

scatter!(ax1,getproperty.(vec_points_m[:],:x),getproperty.(vec_points_m[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:BuGn,strokewidth=1,strokecolor=:black,label="Momentum")
scatter!(ax2,getproperty.(vec_points_m[:],:x);markersize=10,color=1:length(vec_points),colormap=:Greens_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points[:],:x);color=:green,label="Momentum")

axislegend(ax2;position = :rt)

fig

# TODO (Exercise Ex03): Repeat the example above with different values of momentum decay \beta. Try using
#   \beta \in [0.0,0.5]. What happens to the intermediate solutions as you increase
#   \beta?

# =================================================
# Example 2.1.1 with Nesterov Momentum method
# =================================================

x0=1.8*ones(1) #Define initial guess
vec_points=[]
sol = solve(prob, Optimisers.Nesterov(0.1,0.9);maxiters=20,callback=my_callback1D) #solve the problem and take 10 iterations. 


vec_points_nm=copy(vec_points)

###
#The code below plots the intermediate solutions for all methods
#####

fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x_m",limits=(0,20,0.0,2.0))
lines!(ax1,xplot,yplot;color=:black)
scatter!(ax1,getproperty.(vec_points_gd[:],:x),getproperty.(vec_points_gd[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black,label="Gradient Descent")
scatter!(ax2,getproperty.(vec_points_gd[:],:x);markersize=10,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_gd[:],:x);color=:blue,label="Gradient Descent")
hlines!(ax2, 1.0, color=:red, linestyle=:dash) 

scatter!(ax1,getproperty.(vec_points_m[:],:x),getproperty.(vec_points_m[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:BuGn,strokewidth=1,strokecolor=:black,label="Momentum")
scatter!(ax2,getproperty.(vec_points_m[:],:x);markersize=10,color=1:length(vec_points),colormap=:Greens_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_m[:],:x);color=:green,label="Momentum")

scatter!(ax1,getproperty.(vec_points_nm[:],:x),getproperty.(vec_points_nm[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:Greys,strokewidth=1,strokecolor=:black,label="Nesterov Momentum")
scatter!(ax2,getproperty.(vec_points_nm[:],:x);markersize=10,color=1:length(vec_points),colormap=:Greys,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_nm[:],:x);color=:black,label="Nesterov Momentum")
axislegend(ax2;position = :rt)

fig
# Note that the Nesterov momentum method is less oscillatory than the momentum
# method.

# ====================================
# Example 2.1.1 with Adam method
# ====================================

x0=1.8*ones(1) #Define initial guess
vec_points=[]
sol = solve(prob, Optimisers.Adam(0.1,(0.9,0.99));maxiters=20,callback=my_callback1D) #solve the problem and take 10 iterations. 

vec_points_ad=copy(vec_points)

###
#The code below plots the intermediate solutions for all methods
#####

fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x_m",limits=(0,20,0.0,2.0))
lines!(ax1,xplot,yplot;color=:black)
scatter!(ax1,getproperty.(vec_points_gd[:],:x),getproperty.(vec_points_gd[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black,label="Gradient Descent")
scatter!(ax2,getproperty.(vec_points_gd[:],:x);markersize=10,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_gd[:],:x);color=:blue,label="Gradient Descent")
hlines!(ax2, 1.0, color=:red, linestyle=:dash) 

scatter!(ax1,getproperty.(vec_points_m[:],:x),getproperty.(vec_points_m[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:BuGn,strokewidth=1,strokecolor=:black,label="Momentum")
scatter!(ax2,getproperty.(vec_points_m[:],:x);markersize=10,color=1:length(vec_points),colormap=:Greens_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_m[:],:x);color=:green,label="Momentum")

scatter!(ax1,getproperty.(vec_points_nm[:],:x),getproperty.(vec_points_nm[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:Greys,strokewidth=1,strokecolor=:black,label="Nesterov Momentum")
scatter!(ax2,getproperty.(vec_points_nm[:],:x);markersize=10,color=1:length(vec_points),colormap=:Greys,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_nm[:],:x);color=:black,label="Nesterov Momentum")
axislegend(ax2;position = :rt)

scatter!(ax1,getproperty.(vec_points_ad[:],:x),getproperty.(vec_points_ad[:],:fvalue);markersize=10,color=1:length(vec_points),colormap=:OrRd,strokewidth=1,strokecolor=:black,label="Adam")
scatter!(ax2,getproperty.(vec_points_ad[:],:x);markersize=10,color=1:length(vec_points),colormap=:OrRd,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points_ad[:],:x);color=:red,label="Adam")

axislegend(ax2;position = :rt)
fig

# TODO (Class Demo 01): . Use one of the methods above to find the value of r that minimizes the function
#   A(r)=2\pi r^2+\frac{2}{r}.

# =======================
# **Example 2.1.2**
# =======================

# TODO (Exercise Ex04): Show that the minimum of this function is at (x_1,x_2)=(2,-2).

quad(x)=(3.0/2.0)*x[1]^2+2*x[1]*x[2]+3*x[2]^2-2*x[1]+8*x[2]

x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [quad([x1,x2]) for x1 in x1range, x2 in x2range]

x = [(x1i, x2i) for x1i in x1range, x2i in x2range]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
scatter!(ax,[2],[-2];markersize=20,color=:black)
fig

# =========================================================
# Example 2.1.2 with Standard Gradient Descent method
# =========================================================

x0=4.0*ones(2)
vec_points=[] #Define empty vector
optf = OptimizationFunction((x,p)->quad(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.1);maxiters=20,callback=my_callback2D)

vec_points_gd=copy(vec_points)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
scatter!(ax,[2],[-2];markersize=20,color=:black)

lines!(ax,getproperty.(vec_points_gd[:],:x1),getproperty.(vec_points_gd[:],:x2);color=:cyan,label="Gradient Descent")
fig

# ========================================
# Example 1.6.2 with Momentum method
# ========================================

vec_points=[] #Define empty vector
optf = OptimizationFunction((x,p)->quad(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Momentum(0.1,0.9);maxiters=20,callback=my_callback2D)

vec_points_m=copy(vec_points)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
#contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,colormap=:hsv)
scatter!(ax,[2],[-2];markersize=20,color=:black)

lines!(ax,getproperty.(vec_points_gd[:],:x1),getproperty.(vec_points_gd[:],:x2);color=:cyan,label="Gradient Descent")
lines!(ax,getproperty.(vec_points_m[:],:x1),getproperty.(vec_points_m[:],:x2);color=:yellow,label="Momentum")
fig

# =================================================
# Example 2.1.2 with Nesterov Momentum method
# =================================================

vec_points=[] #Define empty vector
optf = OptimizationFunction((x,p)->quad(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Nesterov(0.1,0.9);maxiters=20,callback=my_callback2D)

vec_points_nm=copy(vec_points)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
#contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,colormap=:hsv)
scatter!(ax,[2],[-2];markersize=20,color=:black)

lines!(ax,getproperty.(vec_points_gd[:],:x1),getproperty.(vec_points_gd[:],:x2);color=:cyan,label="Gradient Descent")
lines!(ax,getproperty.(vec_points_m[:],:x1),getproperty.(vec_points_m[:],:x2);color=:yellow,label="Momentum")
lines!(ax,getproperty.(vec_points_nm[:],:x1),getproperty.(vec_points_nm[:],:x2);color=:orange,label="Nesterov Momentum")
fig

# =======================
# **Example 2.1.3**
# =======================
# In this example, we will use the same methods to find the minimum of the Rosenbrock
# function.

rosenbrock(x) = (1.0 - x[1])^2 + 1.0* (x[2] - x[1]^2)^2

x1range=-5:0.02:5;
x2range=-5:0.02:5;
funcplot = [rosenbrock([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "rosenbrock",limits=(-5,5.0,-5.0,5.0))
levels = 10.0.^range(-2, 3.5; length=20)
contourf!(ax,x1range,x2range,funcplot;  levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels,color=:black)
scatter!(ax,[1],[1];markersize=20,color=:black)
fig
# We will now use standard gradient descend, Nesterov momentum and Adam to try to
# find the minimum of this function

x0=-1.0*ones(2)
vec_points=[] #Define empty vector
optf = OptimizationFunction((x,p)->rosenbrock(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

sol = solve(prob, Optimisers.Descent(0.1);maxiters=50,callback=my_callback2D)
vec_points_gd=copy(vec_points)

vec_points=[]
sol = solve(prob, Optimisers.Nesterov(0.1,0.9);maxiters=50,callback=my_callback2D)
vec_points_nm=copy(vec_points)

vec_points=[]
sol = solve(prob, Optimisers.Adam(0.1,(0.9,0.999));maxiters=50,callback=my_callback2D)
vec_points_ad=copy(vec_points)


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_x", ylabel = L"x_2",title = "rosenbrock",limits=(-5,5.0,-5.0,5.0))
levels = 10.0.^range(-2, 3.5; length=10)
contourf!(ax,x1range,x2range,funcplot;  levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels,color=:black)
scatter!(ax,[1],[1];markersize=20,color=:black)

lines!(ax,getproperty.(vec_points_gd[:],:x1),getproperty.(vec_points_gd[:],:x2);color=:cyan,label="Gradient Descent")
lines!(ax,getproperty.(vec_points_nm[:],:x1),getproperty.(vec_points_nm[:],:x2);color=:orange,label="Nesterov Momentum")
lines!(ax,getproperty.(vec_points_ad[:],:x1),getproperty.(vec_points_ad[:],:x2);color=:red,label="Adam")
fig

vec_points_ad
# We can print out more information about the final solution if we like

println("Optimal parameters u:", sol.u)
println("Optimal objective value:", sol.objective)
println("Convergence status:",sol.retcode)

# TODO (Exercise Ex05): Generalize the code above such that you can find the minimum of the rosenbrock
#   function f(x_1,x_2,p_1,p_2)=(p_1-x_1)^2+p_2(x_2-x_1^2)^2 for different values of
#   p_1 and p_2.

# TODO (Class Demo 02): .Use the Julia functions to find the minimum of the Himmelblau function
#   h(x_1,x_2)=(x_1^2+x_2-11)^2+(x_1+x_2^2-7)^2.
