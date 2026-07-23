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

df = CSV.read(joinpath(@__DIR__, "DataHomeworkDay11.csv"), DataFrame; select = ["xdata", "ydata"])
xdata = df.xdata
ydata = df.ydata
N_SAMPLES = length(xdata)
xplot = collect(0:0.01:5)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax, xdata, ydata; color = :black, label = "data")
axislegend(ax; position = :rc)
display(fig)

degree = 8
V = hcat([xdata .^ k for k in 0:degree]...)
a = V'V \ (V'ydata)
Vplot = hcat([xplot .^ k for k in 0:degree]...)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (0, 5, -2, 7))
scatter!(ax, xdata, ydata; color = :black, label = "data")
lines!(ax, xplot, Vplot * a; color = :green, label = "polynomial")
axislegend(ax; position = :rc)
display(fig)

model_c = Chain(Dense(1 => 1, tanh), Dense(1 => 1))
rng = Random.default_rng()
ps, ls = Lux.setup(rng, model_c)

function loss_c(p, ls)
    y_prediction, _ = model_c(xdata', p, ls)
    return 0.5 * mean((y_prediction[:] .- ydata) .^ 2)
end

loss_history = Float64[]
callback = function (p, l)
    push!(loss_history, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_c, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ps), ls)
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.02); callback, maxiters = 40000)
ps = neural_network.u

y_prediction, _ = model_c(xplot', ps, ls)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (0, 5, -2, 7))
scatter!(ax, xdata, ydata; color = :black, label = "data")
lines!(ax, xplot, y_prediction[:]; color = :red, label = "1 node")
axislegend(ax; position = :rc)
display(fig)

println("Part c final loss: ", loss_history[end])

model_d = Chain(Dense(1 => 8, tanh), Dense(8 => 1))
ps, ls = Lux.setup(rng, model_d)

function loss_d(p, ls)
    y_prediction, _ = model_d(xdata', p, ls)
    return 0.5 * mean((y_prediction[:] .- ydata) .^ 2)
end

loss_history = Float64[]
callback = function (p, l)
    push!(loss_history, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_d, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ps), ls)
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.02); callback, maxiters = 40000)
ps = neural_network.u

y_prediction, _ = model_d(xplot', ps, ls)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (0, 5, -2, 7))
scatter!(ax, xdata, ydata; color = :black, label = "data")
lines!(ax, xplot, y_prediction[:]; color = :blue, label = "8 nodes")
axislegend(ax; position = :rc)
display(fig)

println("Part d final loss: ", loss_history[end])
