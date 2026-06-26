using Lux
using Random
using CairoMakie

model01 = Chain(Dense(1 => 1))
rng = Random.default_rng()
parameters, layer_states = Lux.setup(rng, model01)

parameters.layer_1.weight.=Float32(3.0)
parameters.layer_1.bias.=Float32(4.0)

xgrid=Array(-1:0.1:1)

ygrid,_=model01(xgrid',parameters,layer_states)


fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"NN(x)")
lines!(ax,xgrid,ygrid[:];color=:blue,label="prediction",linewidth=5)

fig


xgrid=collect(-3.0:0.01:3.0) 
N_SAMPLES=length(xgrid)

#y_prediction,_=model01(reshape(xgrid,(1,N_SAMPLES)),parameters, layer_states)
ygrid,_=model01(xgrid',parameters, layer_states)
ydata=xgrid.+3+0.1*randn(N_SAMPLES)

function loss_fn(p, ls)
    y_prediction, _= model01(xgrid', p, ls)
    loss = 0.5 * mean((y_prediction[:] .- ydata).^2)
    return loss
end

callback = function (p, l)
    if length(loss_history) % 100 == 0
        println("Iteration: $(p.iter), Loss: $l")
    end
    push!(loss_history, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_fn, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters),layer_states)

# Initialize a vector to store loss values
loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 100)

parameters03=neural_network.u