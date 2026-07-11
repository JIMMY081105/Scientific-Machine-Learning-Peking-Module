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


himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2 

x0=ones(2)

vec_points=[] #Define empty vector
optf = OptimizationFunction((x,p)->himmelblau(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.000001);maxiters=20,callback=my_callback2D)