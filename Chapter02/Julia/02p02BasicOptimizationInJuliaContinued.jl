using Optimization,OptimizationOptimJL,ForwardDiff,CairoMakie,Zygote,ADTypes

struct Point1D
    x::Float64 # value of x
    fvalue::Float64 #value of the function at x
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
end

f(x)=(x[1]^3-x[1])^2;
xplot=-2:0.01:2
yplot=[f(x) for x in xplot];
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-2,2,-0.01,0.2))
lines!(ax,xplot,yplot)
scatter!(ax,[0, -1, 1],f.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax,[1/sqrt(3), -1/sqrt(3)],f.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)
fig

x0=1.1*ones(1) #Define initial guess at x=1.1

optf = OptimizationFunction((x,p)->f(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0) #first argument is the optimization function, second is the guess value and third is the parameter for optf()

vec_points=[]
sol = solve(prob,Optim.Newton();maxiters=20,callback=my_callback1D)
vec_points_new=copy(vec_points);

vec_points

sol.retcode

fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-2,2,-0.01,0.2))
ax2 = Axis(fig[1, 2], xlabel = "iter", ylabel = L"x",limits=(0,20,-0.5,1.5))
lines!(ax1,xplot,yplot)
scatter!(ax1,[1],[f(1)];markersize=20,color=:black)


scatter!(ax1,getproperty.(vec_points_new[:],:x),getproperty.(vec_points_new[:],:fvalue);markersize=10,color=:black,label="Newton")
scatter!(ax2,getproperty.(vec_points_new[:],:x);markersize=10,color=:black,label="Newton")
lines!(ax2,getproperty.(vec_points_new[:],:x);color=:black,label="")
hlines!(ax2,1.0, color=:red, linestyle=:dash) 
axislegend(ax1;position = :rt)

fig

vec_points

vec_points=[]
sol = solve(prob, Optim.BFGS();maxiters=20,callback=my_callback1D) #solve the problem and take 10 iterations. 
vec_points_bfgs=copy(vec_points);


scatter!(ax1,getproperty.(vec_points[:],:x),getproperty.(vec_points[:],:fvalue);markersize=10,color=:green,label="BFGS")
scatter!(ax2,getproperty.(vec_points[:],:x);color=:green,label="BFGS")
lines!(ax2,getproperty.(vec_points[:],:x);color=:green)
axislegend(ax1;position = :rt)

fig

vec_points=[]
sol = solve(prob, Optim.LBFGS();maxiters=20,callback=my_callback1D) #solve the problem and take 10 iterations. 



scatter!(ax1,getproperty.(vec_points[:],:x),getproperty.(vec_points[:],:fvalue);markersize=10,color=:cyan,label="LBFGS")
scatter!(ax2,getproperty.(vec_points[:],:x);color=:cyan,label="LBFGS")
lines!(ax2,getproperty.(vec_points[:],:x);color=:cyan)
axislegend(ax1;position = :rt)

fig

quad(x)=(3.0/2.0)*x[1]^2+2*x[1]*x[2]+3*x[2]^2-2*x[1]+8*x[2]

x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [quad([x1,x2]) for x1 in x1range, x2 in x2range]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
scatter!(ax,[2],[-2];markersize=20,color=:black)
fig

x0=4.0*ones(2) #Define initial guess

optf = OptimizationFunction((x,p)->quad(x), ADTypes.AutoForwardDiff())
prob = OptimizationProblem(optf, x0) #first argument is the optimization function, second is the guess value and third is the parameter for optf()

vec_points=[] #Define empty vector
sol = solve(prob,Optim.Newton();maxiters=20,callback=my_callback2D)
vec_points_new=copy(vec_points)

lines!(ax,getproperty.(vec_points[:],:x1),getproperty.(vec_points[:],:x2);color=:black,label="Newton")
fig

vec_points=[] #Define empty vector
sol = solve(prob,Optim.BFGS();maxiters=20,callback=my_callback2D)
vec_points_bfgs=copy(vec_points)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
scatter!(ax,[2],[-2];markersize=20,color=:black)


lines!(ax,getproperty.(vec_points_new[:],:x1),getproperty.(vec_points_new[:],:x2);color=:black,label="Newton")
lines!(ax,getproperty.(vec_points_bfgs[:],:x1),getproperty.(vec_points_bfgs[:],:x2);color=:Green,label="BFGS")
fig

vec_points=[] #Define empty vector
sol = solve(prob,Optim.LBFGS();maxiters=20,callback=my_callback2D)
vec_points_lbfgs=copy(vec_points)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot;  levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
scatter!(ax,[2],[-2];markersize=20,color=:black)


lines!(ax,getproperty.(vec_points_new[:],:x1),getproperty.(vec_points_new[:],:x2);color=:black,label="Newton")
lines!(ax,getproperty.(vec_points_bfgs[:],:x1),getproperty.(vec_points_bfgs[:],:x2);color=:Green,label="BFGS")
lines!(ax,getproperty.(vec_points_lbfgs[:],:x1),getproperty.(vec_points_lbfgs[:],:x2);color=:cyan,label="LBFGS")
fig

rosenbrock(x) = (1.0 - x[1])^2 + 1.0* (x[2] - x[1]^2)^2

x1range=-5:0.02:5;
x2range=-5:0.02:5;
funcplot = [rosenbrock([x1,x2]) for x1 in x1range, x2 in x2range]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "rosenbrock",limits=(-5,5.0,-5.0,5.0))
levels = 10.0.^range(-2, 3.5; length=10)
contourf!(ax,x1range,x2range,funcplot; levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels,color=:black)
scatter!(ax,[1],[1];markersize=20,color=:black)
fig

x0=-1.0*ones(2)
optf = OptimizationFunction((x,p)->rosenbrock(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)

vec_points=[] #Define empty vector
sol = solve(prob, Optim.Newton();maxiters=20,callback=my_callback2D)
vec_points_new=copy(vec_points)

vec_points=[] #Define empty vector
sol = solve(prob, Optim.BFGS();maxiters=20,callback=my_callback2D)
vec_points_bfgs=copy(vec_points)

vec_points=[] #Define empty vector
sol = solve(prob, Optim.LBFGS();maxiters=20,callback=my_callback2D)
vec_points_lbfgs=copy(vec_points)






fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "rosenbrock",limits=(-5,5.0,-5.0,5.0))
levels = 10.0.^range(-2, 3.5; length=10)
contourf!(ax,x1range,x2range,funcplot;  levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels,color=:black)
scatter!(ax,[1],[1];markersize=20,color=:black)

lines!(ax,getproperty.(vec_points_new[:],:x1),getproperty.(vec_points_new[:],:x2);color=:black,label="Newton")
lines!(ax,getproperty.(vec_points_bfgs[:],:x1),getproperty.(vec_points_bfgs[:],:x2);color=:Green,label="BFGS")
lines!(ax,getproperty.(vec_points_lbfgs[:],:x1),getproperty.(vec_points_lbfgs[:],:x2);color=:cyan,label="LBFGS")

axislegend(ax,  position = :rt,orientation = :vertical)

fig

println("Optimal parameters u:", sol.u)
println("Optimal objective value:", sol.objective)
println("Convergence status:",sol.retcode)


