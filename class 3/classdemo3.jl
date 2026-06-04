using Optimization,OptimizationOptimisers,Zygote,CairoMakie

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

function my_callback2D(state, l)

    push!(vec_points,Point2D(state.u...,l))
    return false # continues the solver
end;

himmeblau(x)=(3.0/2.0)*x[1]^2+2*x[1]*x[2]+3*x[2]^2-2*x[1]+8*x[2]

x0=4.0*ones(2)
vec_points=[] #Define empty vector
optf = OptimizationFunction((x,p)->himmeblau(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.1);maxiters=20,callback=my_callback2D)

vec_points_gd=copy(vec_points)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Simple Quadratic function",limits=(-6,6.0,-6.0,6.0))
contourf!(ax,x1range,x2range,funcplot; levels=20,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black)
scatter!(ax,[2],[-2];markersize=20,color=:black)

lines!(ax,getproperty.(vec_points_gd[:],:x1),getproperty.(vec_points_gd[:],:x2);color=:cyan,label="Gradient Descent")
fig