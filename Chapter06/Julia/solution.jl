using Zygote
using DataFrames
using CSV
using Lux
using Random
using Optimisers
using Optimization
using OptimizationOptimisers
using Statistics

df = CSV.read("Chapter06/Data05.csv", DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
scatter!(ax,xdata,ydata,color=:black,label="data")
axislegend(ax,position=:ct)
display(fig)

model01 = Chain(Dense(1 =>70,tanh),
                Dense(70=>1))
rng = Random.default_rng()
parameters, layer_states = Lux.setup(rng, model01)

function loss_fn(p, ls)
    y_prediction, _= model01(xdata', p, ls)
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

using ComponentArrays

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_fn, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters),layer_states)

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 1000)

optprob = Optimization.OptimizationProblem(optf, ComponentArray(neural_network.u),layer_states)
loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 1000)

xgrid=collect(0.0:0.01:5.0)
ygrid,_=model01(xgrid',neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"NN(x)")
scatter!(ax,xdata,ydata,color=:black)
lines!(ax,xgrid,ygrid[:];color=:blue,label="prediction",linewidth=5)

fig
