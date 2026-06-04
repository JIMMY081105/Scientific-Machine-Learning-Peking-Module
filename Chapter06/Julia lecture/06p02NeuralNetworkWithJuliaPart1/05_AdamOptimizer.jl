# =====================================================================================
# 6.2 - Training with the Optimization.jl library (Adam)
# =====================================================================================
# GOAL: fit the same hidden-layer network to g(x) = 2*sin(pi*x/5), but let a real
#       optimiser do the work instead of hand-coded gradient descent.
# STRATEGY: wrap the loss in an OptimizationProblem, differentiate with AutoZygote, and
#           solve with the Adam optimiser; a callback logs the loss each iteration.
# =====================================================================================

using Lux
using Random
using Zygote
using Statistics
using CairoMakie
using Optimization
using OptimizationOptimisers
using ComponentArrays

# Hidden-layer network and its parameters.
model03 = Chain(Dense(1 => 1, tanh),
    Dense(1 => 1))
rng = Random.default_rng()
parameters03, layer_states03 = Lux.setup(rng, model03)

# Data from g(x) = 2*sin(pi*x/5).
xgrid = collect(-3.0:0.01:3.0)
N_SAMPLES = length(xgrid)
g(x) = 2.0sin(pi * x / 5)
data03 = g.(xgrid)

# Loss with the (parameters, layer_state) signature the optimiser expects.
function loss_fn03(p, ls)
    y_prediction, _ = model03(xgrid', p, ls)
    loss = 0.5 * mean((y_prediction[:] .- data03).^2)
    return loss
end

# A callback logs the loss; layer_states03 is passed as the problem's fixed parameter.
callback = function (p, l)
    if length(loss_history) % 100 == 0
        println("Iteration: $(p.iter), Loss: $l")
    end
    push!(loss_history, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_fn03, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters03), layer_states03)

# Initialize a vector to store loss values
loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 1000)

parameters03 = neural_network.u

# Loss for the current set of parameters.
loss_fn03(parameters03, layer_states03)

# Plot to see if the new model matches the data.
y_prediction, _ = model03(reshape(xgrid, (1, N_SAMPLES)), parameters03, layer_states03)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
scatter!(ax, xgrid, data03; color = :black, label = "Data")
lines!(ax, xgrid, y_prediction[:]; color = :blue, label = "Neural Network prediction")
axislegend(ax, position = :rb, orientation = :vertical)
fig

# TODO (Class Demo 02): Construct a single-input single-output Neural Network using Lux
#   in Julia to fit the data in "Data05.csv".
