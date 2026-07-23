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


df = CSV.read(joinpath(@__DIR__, "..", "..", "Chapter06", "Data07.csv"), DataFrame)

tdata=df[:, :tdata]
xdata=df[:, :xdata]
xzero=df[:, :xzero]

rng = Xoshiro(1)
t_input=[xzero';tdata']
model = Chain(
    Dense(2 => 5),
    Dense(5 => 5,tanh),
    Dense(5 => 1)
)
parameters, layer_states = Lux.setup(rng, model)
x_prediction, _= model(t_input,parameters, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata,xdata;color=:black,label="Data")
lines!(ax,tdata,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(position=:lt)
fig

function loss_fn(p, ls)
    x_prediction, new_ls= model(t_input, p, ls)
    loss = 0.5 * mean((x_prediction .- xdata').^2)
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
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

x_prediction, new_ls= model(t_input,neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata,xdata;color=:black,label="Data")
lines!(ax,tdata,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :rb,orientation = :vertical)
fig

df = CSV.read(joinpath(@__DIR__, "..", "..", "Chapter06", "Data08.csv"), DataFrame)
tdata=df[:, :tdata]
xdata=df[:, :xdata]
xzero=df[:, :xzero]
t_input=[xzero';tdata']
x_prediction, new_ls= model(t_input,neural_network.u, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata,xdata;color=:red,label="Data")
lines!(ax,tdata,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :lb,orientation = :vertical)
fig


df1 = CSV.read(joinpath(@__DIR__, "..", "..", "Chapter06", "Data07.csv"), DataFrame)
tdata1=df1[:, :tdata]
xdata1=df1[:, :xdata]
xzero1=df1[:, :xzero]

df2 = CSV.read(joinpath(@__DIR__, "..", "..", "Chapter06", "Data08.csv"), DataFrame)
tdata2=df2[:, :tdata]
xdata2=df2[:, :xdata]
xzero2=df2[:, :xzero]

t_input = [xzero2';tdata2']

x_prediction, new_ls= model(t_input,neural_network.u, layer_states)

fig = Figure()
ax1 = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax1,tdata1,xdata1;color=:black,label="Data for x(0)=0.0")
scatter!(ax1,tdata2,xdata2;color=:red,label="Data for x(0)=1.0")
axislegend(ax1,  position = :lb,orientation = :vertical)

fig


t_input=vcat(hcat(xzero1', xzero2'),hcat(tdata1', tdata2'))
xdata_combined=hcat(xdata1', xdata2')

model = Chain(
    Dense(2 => 16),
    Dense(16 => 16,tanh),
    Dense(16 => 1)
)
parameters, layer_states = Lux.setup(rng, model)

function loss_fn(p, ls)
    x_prediction, new_ls= model(t_input, p, ls)
    loss = 0.5 * mean((x_prediction .- xdata_combined).^2)
    return loss
end

x_prediction, new_ls= model(t_input, parameters, layer_states)
x_prediction.- xdata_combined
loss_fn(parameters, layer_states)

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
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

t_input1=[xzero1';tdata1']
x_prediction, new_ls= model(t_input1,neural_network.u, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata1,xdata1;color=:black,label="Data")
lines!(ax,tdata1,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :rb,orientation = :vertical)
fig

t_input2=[xzero2';tdata2']
x_prediction, new_ls= model(t_input2,neural_network.u, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata2,xdata2;color=:red,label="Data")
lines!(ax,tdata2,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :lb,orientation = :vertical)
fig


df3 = CSV.read(joinpath(@__DIR__, "..", "..", "Chapter06", "Data09.csv"), DataFrame)
tdata3=df3[:, :tdata]
xdata3=df3[:, :xdata]
xzero3=df3[:, :xzero]

df4 = CSV.read(joinpath(@__DIR__, "..", "..", "Chapter06", "Data10.csv"), DataFrame)
tdata4=df4[:, :tdata]
xdata4=df4[:, :xdata]
xzero4=df4[:, :xzero]

t_input3=[xzero3';tdata3']
x_prediction, new_ls= model(t_input3,neural_network.u, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata3,xdata3;color=:black,label="Data09")
lines!(ax,tdata3,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :rb,orientation = :vertical)
fig

t_input4=[xzero4';tdata4']
x_prediction, new_ls= model(t_input4,neural_network.u, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata4,xdata4;color=:red,label="Data10")
lines!(ax,tdata4,x_prediction[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :lb,orientation = :vertical)
fig

