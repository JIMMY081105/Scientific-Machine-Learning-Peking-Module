using Optimization,OptimizationOptimisers,Zygote,CairoMakie


struct Point1D
    x::Float64
    fvalue::Float64
end

function my_callback1D(state, l)
    push!(vec_points,Point1D(state.u...,l))
    return false # continues the solver
end

f(x)=2*π*x[1]^2+2/x[1];  

vec_points=[] 

x0=[1.8]
optf = OptimizationFunction((x,p)->f(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.001);maxiters=1000,callback=my_callback1D);

xvec=getproperty.(vec_points[:],:x)
fvec=getproperty.(vec_points[:],:fvalue)
fig = Figure() #set up a figure
ax = Axis(fig[1, 1], xlabel = L"iter", ylabel = L"x") #Define axis in the figure
lines!(ax,xvec;color=:black) #Plot line
display(fig)