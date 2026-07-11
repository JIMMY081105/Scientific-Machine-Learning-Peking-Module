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
save("../figures/02p01BasicOptimizationInJulia01.svg", fig)
display(fig)

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

fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x",limits=(0,20,0.0,2.0))
lines!(ax1,xplot,yplot;color=:black)
scatter!(ax1,getproperty.(vec_points[:],:x),getproperty.(vec_points[:],:fvalue);markersize=15,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black,label="Gradient Descent")
scatter!(ax2,getproperty.(vec_points[:],:x);markersize=15,color=1:length(vec_points),colormap=:Blues_3,strokewidth=1,strokecolor=:black)
lines!(ax2,getproperty.(vec_points[:],:x);color=:blue,label="Gradient Descent")
hlines!(ax2, 1.0, color=:red, linestyle=:dash) 
axislegend(ax2;position = :rt)
save("../figures/02p01BasicOptimizationInJulia02.svg", fig)
display(fig)

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

println("Optimal parameters u:", sol.u)
println("Optimal objective value:", sol.objective)
println("Convergence status:",sol.retcode)
