using CSV
using DataFrames
using CairoMakie
using Statistics
using Random
using Lux
using Zygote
using Optimization
using OptimizationOptimisers
using ComponentArrays

df = CSV.read(joinpath(@__DIR__, "DataHomeworkDay12.csv"), DataFrame)

x1data = df[:, :x1data]
x2data = df[:, :x2data]
ydata = df[:, :ydata]

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "Data")
scatter!(ax, x1data, x2data; color = ydata, colormap = :bwr, strokewidth = 1, strokecolor = :black)
fig

t_input = [x1data'; x2data']

model = Chain(
    Dense(2 => 16, tanh),
    Dense(16 => 16, tanh),
    Dense(16 => 1)
)

rng = Xoshiro(1)
parameters, layer_states = Lux.setup(rng, model)

function loss_fn(p, ls)
    y_prediction, new_ls = model(t_input, p, ls)
    loss = 0.5 * mean((y_prediction .- ydata') .^ 2)
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
optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters), layer_states)

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 10000)

y_prediction, _ = model(t_input, neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss", yscale = log10)
lines!(ax, loss_history; color = :blue)
fig

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "Neural Network prediction")
tricontourf!(ax, x1data, x2data,y_prediction[:]; colormap = :bwr)
scatter!(ax, x1data, x2data; color = ydata, colormap = :bwr, strokewidth = 1, strokecolor = :black)
fig