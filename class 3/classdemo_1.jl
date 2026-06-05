using Optimization,OptimizationOptimisers,Zygote,CairoMakie

struct Point1D
    x::Float64
    fvalue::Float64
end

function my_callback1D(state, loss)
    push!(vec_points, Point1D(state.u..., loss))
    return false # continues the solver
end

vec_points = Point1D[]

f(x) = 2pi * x[1]^2 + 2 / x[1]

x0=[1.8] #Define initial guess
optf = OptimizationFunction((x,p)->f(x), ADTypes.AutoZygote())
#optf = OptimizationFunction(temp, ADTypes.AutoZygote()) 
prob = OptimizationProblem(optf, x0) #define initial guess x=1.8
sol = solve(prob, Optimisers.Descent(0.001);maxiters=20,callback=my_callback1D)

