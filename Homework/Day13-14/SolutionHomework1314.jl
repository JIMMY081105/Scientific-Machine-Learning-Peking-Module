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

#xt=0
dfa = CSV.read(joinpath(@__DIR__,  "DataHomeworkDay13a.csv"), DataFrame)

atdata=dfa[:, :tdata]
axdata=dfa[:, :xdata]
axzero=dfa[:, :xzero]

#xt=1
dfb = CSV.read(joinpath(@__DIR__,  "DataHomeworkDay13b.csv"), DataFrame)

btdata=dfb[:, :tdata]
bxdata=dfb[:, :xdata]
bxzero=dfb[:, :xzero]

#xt=2
dfc = CSV.read(joinpath(@__DIR__,  "DataHomeworkDay13c.csv"), DataFrame)

ctdata=dfc[:, :tdata]
cxdata=dfc[:, :xdata]
cxzero=dfc[:, :xzero]

#xt=0.5
dfd = CSV.read(joinpath(@__DIR__,  "DataHomeworkDay13d.csv"), DataFrame)

dtdata=dfd[:, :tdata]
dxdata=dfd[:, :xdata]
dxzero=dfd[:, :xzero]

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,atdata,axdata;color=:black,label="x(t)=0")
scatter!(ax,btdata,bxdata;color=:blue,label="x(t)=1")
scatter!(ax,ctdata,cxdata;color=:green,label="x(t)=2")
scatter!(ax,dtdata,dxdata;color=:red,label="x(t)=0.5")
axislegend(position=:rt)
fig

#train with x(0)=0
rng = Xoshiro(1)
t_input=[axzero';atdata']
model = Chain(
    Dense(2 => 16,tanh),
    Dense(16 => 16,tanh),
    Dense(16 => 1)
)
parameters, layer_states = Lux.setup(rng, model)
x0_prediction, _= model(t_input,parameters, layer_states)

function loss_fn(p, ls)
    x0_prediction, new_ls= model(t_input, p, ls)
    loss = 0.5 * mean((x0_prediction .- axdata').^2)
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

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

t_input = [dxzero'; dtdata']
x0_prediction, new_ls = model(t_input, neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,dtdata,dxdata;color=:black,label="Data")
lines!(ax,dtdata,x0_prediction[:];color=:blue,label="NN trained on x(0)=0")
axislegend(ax,  position = :rb,orientation = :vertical)
fig

#train with x(0)=0 and x(0)=1
rng = Xoshiro(1)
abxzero = [axzero' bxzero']
abtdata = [atdata' btdata']
abxdata = [axdata' bxdata']
t_input=[abxzero ; abtdata]
model = Chain(
    Dense(2 => 16,tanh),
    Dense(16 => 16,tanh),
    Dense(16 => 1)
)
parameters, layer_states = Lux.setup(rng, model)
x1_prediction, _= model(t_input,parameters, layer_states)

function loss_fn(p, ls)
    x1_prediction, new_ls= model(t_input, p, ls)
    loss = 0.5 * mean((x1_prediction .- abxdata).^2)
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

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

t_input = [dxzero'; dtdata']
x1_prediction, new_ls= model(t_input,neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,dtdata,dxdata;color=:black,label="Data")
lines!(ax,dtdata,x1_prediction[:];color=:blue,label="NN trained on x(0)=0,1")
axislegend(ax,  position = :lb,orientation = :vertical)
fig

#train with x(0)=0, x(0)=1 and x(0)=2
rng = Xoshiro(1)
abczero = [axzero' bxzero' cxzero']
abctdata = [atdata' btdata' ctdata']
abcdata = [axdata' bxdata' cxdata']
t_input=[abczero ; abctdata]
model = Chain(
    Dense(2 => 16,tanh),
    Dense(16 => 16,tanh),
    Dense(16 => 1)
)
parameters, layer_states = Lux.setup(rng, model)
x2_prediction, _= model(t_input,parameters, layer_states)

function loss_fn(p, ls)
    x2_prediction, new_ls= model(t_input, p, ls)
    loss = 0.5 * mean((x2_prediction .- abcdata).^2)
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

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

t_input = [dxzero'; dtdata']
x2_prediction, new_ls= model(t_input,neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,dtdata,dxdata;color=:black,label="Data")
lines!(ax,dtdata,x2_prediction[:];color=:blue,label="NN trained on x(0)=0,1,2")
axislegend(ax, position = :rb,orientation = :vertical)
fig

#train with x(0)=0, x(0)=1, x(0)=2 and x(0)=0.5
rng = Xoshiro(1)
abcdzero = [axzero' bxzero' cxzero' dxzero']
abcdtdata = [atdata' btdata' ctdata' dtdata']
abcddata = [axdata' bxdata' cxdata' dxdata']
t_input=[abcdzero ; abcdtdata]
model = Chain(
    Dense(2 => 16,tanh),
    Dense(16 => 16,tanh),
    Dense(16 => 1)
)
parameters, layer_states = Lux.setup(rng, model)
x3_prediction, _= model(t_input,parameters, layer_states)

function loss_fn(p, ls)
    x3_prediction, new_ls= model(t_input, p, ls)
    loss = 0.5 * mean((x3_prediction .- abcddata).^2)
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

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

t_input = [dxzero'; dtdata']
x3_prediction, new_ls= model(t_input,neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,dtdata,dxdata;color=:black,label="Data")
lines!(ax,dtdata,x3_prediction[:];color=:blue,label="NN trained on x(0)=0,1,2,0.5")
axislegend(ax, position = :rb,orientation = :vertical)
fig
