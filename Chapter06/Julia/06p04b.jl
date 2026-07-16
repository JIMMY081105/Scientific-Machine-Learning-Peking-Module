using CSV
using DataFrames


df = CSV.read(joinpath(@__DIR__, "..", "Data07.csv"), DataFrame)
tdata=df[:, :tdata]
xdata=df[:, :xdata]
xzero=df[:, :xzero]

t_input=[xzero';tdata']
model = Chain(
    Dense(2 => 5),
    Dense(5 => 5,tanh),
    Dense(5 => 1)
)
parameters, layer_states = Lux.setup(rng, model)

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


df2 = CSV.read(joinpath(@__DIR__, "..", "Data08.csv"), DataFrame)
tdata2=df2[:, :tdata]
xdata2=df2[:, :xdata]
xzero2=df2[:, :xzero]

t_input2=[xzero2';tdata2']


x_prediction2, new_ls= model(t_input2,neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)")
scatter!(ax,tdata2,xdata2;color=:black,label="Data")
lines!(ax,tdata2,x_prediction2[:];color=:blue,label="Neural Network prediction")
axislegend(ax,  position = :rb,orientation = :vertical)
fig