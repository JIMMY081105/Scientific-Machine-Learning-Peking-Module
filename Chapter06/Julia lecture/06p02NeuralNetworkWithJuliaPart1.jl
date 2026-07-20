########################################################################################
# 6.2 Neural Network with Julia (Part 1)
########################################################################################
#
# PROBLEM
# Rebuild the neural-network workflow from 6.1 using Julia's Lux.jl library, which
# handles the network definition, parameter initialisation and evaluation for you,
# then train these Lux networks to fit data.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Define the simplest Lux network NN(x) = Wx + b with Chain(Dense(1 => 1));
#      initialise its parameters and state randomly with Lux.setup, optionally
#      overwriting W and b by hand (note Lux defaults to Float32).
#   2. Evaluate and plot the network over a grid; generate noisy data from y = x + 3
#      and compare the untrained prediction against it.
#   3. Define a mean-squared-error loss, get its gradient w.r.t. the Lux parameters
#      with Zygote, and train by hand-coded gradient descent; read back the recovered
#      W and b.
#   4. Add a tanh activation (Dense(1 => 1, tanh)), then a hidden-layer network
#      NN(x) = W2*tanh(W1*x + b1) + b2 (a Chain of two Dense layers); fit it to
#      g(x) = 2*sin(pi*x/5) with hand-coded gradient descent.
#   5. Retrain the same hidden-layer network with the Optimization.jl library (Adam,
#      AutoZygote) instead of hand-coded descent, logging the loss through a callback.
#
# This file is notebook "06p02NeuralNetworkWithJuliaPart1" with every code cell joined
# in order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


using Zygote
using Lux
using Random
using Distributions
using CairoMakie
using Optimisers
using Optimization
using OptimizationOptimisers
using ComponentArrays

# ==========================================
# A simple Lux network: NN(x) = Wx + b
# ==========================================
# One input, one output. Lux defines the structure; you supply parameters and state.

model01 = Chain(Dense(1 => 1))

# Initialize the model weights and state (a random generator is preferred).
rng = Random.default_rng()
parameters, layer_states = Lux.setup(rng, model01)

# You can also set W and b explicitly. Lux parameters default to Float32.
parameters.layer_1.weight .= Float32(2.0)
parameters.layer_1.bias   .= Float32(1.0)

# Plot the network over a grid.
xgrid = -3.0:0.01:3.0
y_prediction, _ = model01(xgrid', parameters, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"NN(x)")
lines!(ax, xgrid, y_prediction[:]; color = :blue, label = "prediction", linewidth = 5)
# save("../figures/06p02NeuraNetworkWithJulia01.svg", fig)   # (figure-save disabled in study file)
fig

# TODO (Class Demo 01): Define and plot the Neural Network NN(x) = sin(3x + 4) for x in [-2, 3].

# ==========================================
# Generate data and compare the prediction
# ==========================================
# Artificial (noisy) data from y(x) = x + 3.

xgrid = collect(-3.0:0.01:3.0)
N_SAMPLES = length(xgrid)

#y_prediction,_=model01(reshape(xgrid,(1,N_SAMPLES)),parameters, layer_states)
y_prediction, _ = model01(xgrid', parameters, layer_states)
y_data = xgrid .+ 3 + 0.1 * randn(N_SAMPLES)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
lines!(ax, xgrid, y_prediction[:]; color = :blue, label = "prediction", linewidth = 5)
scatter!(ax, xgrid, y_data[:]; color = :black, label = "data")
fig

# ==========================================
# Train by hand-coded gradient descent
# ==========================================
# Define the loss, then differentiate it w.r.t. the Lux parameters with Zygote.

function loss_fn(p, ls)
    y_prediction, _ = model01(xgrid', p, ls)
    loss = 0.5 * mean((y_prediction[:] .- y_data).^2)
    return loss
end

#taking f_loss with respect to loss_fn with only ps 
f_loss = (ps) -> loss_fn(ps, layer_states)

Zygote.gradient(f_loss, parameters)[1]

# Standard gradient descent on the (scalar) weight and bias.
α = 0.01

new_weight = parameters.layer_1.weight[1]
new_bias   = parameters.layer_1.bias[1]

for i = 1:5000
    grads = Zygote.gradient(f_loss, parameters)[1]
    new_weight -= α * grads.layer_1.weight[1]
    new_bias   -= α * grads.layer_1.bias[1]

    parameters.layer_1.weight .= Float32(new_weight)
    parameters.layer_1.bias   .= Float32(new_bias)
end

# Plot and compare after training.
y_prediction, _ = model01(xgrid', parameters, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
lines!(ax, xgrid[:], y_prediction[:]; color = :blue, label = "prediction", linewidth = 5)
scatter!(ax, xgrid[:], y_data[:]; color = :black, label = "data")
fig

# The recovered weight and bias are close to the expected 1.0 and 3.0.
parameters.layer_1.bias[1]
parameters.layer_1.weight[1]

# ==========================================
# Adding a tanh activation: NN(x) = tanh(Wx + b)
# ==========================================
model02 = Chain(Dense(1 => 1, tanh))
parameters.layer_1.weight .= Float32(1.0) # Change the weight to 1.0
parameters.layer_1.bias   .= Float32(0.0) # Change the bias to 0.0

y_prediction, _ = model02(xgrid', parameters, layer_states) # predict the output of the Neural network

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"NN(x)")
lines!(ax, xgrid[:], y_prediction[:]; color = :blue, label = "prediction")
fig

# ==========================================
# A hidden layer: NN(x) = W2*tanh(W1*x + b1) + b2
# ==========================================
model03 = Chain(Dense(1 => 1, tanh),
    Dense(1 => 1))

# Initialize the model weights and state.
parameters03, layer_states03 = Lux.setup(rng, model03)

# Optional: set W1 = W2 = 1, b1 = b2 = 0.
parameters03.layer_1.weight .= Float32(1.0)
parameters03.layer_2.weight .= Float32(1.0)
parameters03.layer_1.bias   .= Float32(0.0)
parameters03.layer_2.bias   .= Float32(0.0)

# Generate data from g(x) = 2*sin(pi*x/5) and view the untrained fit.
g(x) = 2.0sin(pi * x / 5)
data03 = g.(xgrid)

y_prediction, _ = model03(xgrid', parameters03, layer_states03) # predict the output of the Neural network

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
scatter!(ax, xgrid, data03; color = :black, label = "Data")
lines!(ax, xgrid, y_prediction[:]; linewidth = 5, color = :blue, label = "prediction")
axislegend(ax, position = :rb, orientation = :vertical)
fig

# Loss function for the hidden-layer network.
function loss_fn03(p, ls)
    y_prediction, _ = model03(xgrid', p, ls)
    loss = 0.5 * mean((y_prediction[:] .- data03).^2)
    return loss
end

f03_loss = (ps) -> loss_fn03(ps, layer_states03)[1]

f03_loss(parameters03)

# Gradient of the loss via Zygote.
grads = Zygote.gradient(f03_loss, parameters03)[1]

# Hand-coded gradient descent over all four parameters.
new_weight_l1 = parameters03.layer_1.weight[1]
new_bias_l1   = parameters03.layer_1.bias[1]
new_weight_l2 = parameters03.layer_2.weight[1]
new_bias_l2   = parameters03.layer_2.bias[1]

α = 0.2

for i = 1:500
    grads = Zygote.gradient(f03_loss, parameters03)[1]
    new_weight_l1 -= α * grads.layer_1.weight[1]
    new_bias_l1   -= α * grads.layer_1.bias[1]
    new_weight_l2 -= α * grads.layer_2.weight[1]
    new_bias_l2   -= α * grads.layer_2.bias[1]

    parameters03.layer_1.weight .= Float32(new_weight_l1)
    parameters03.layer_1.bias   .= Float32(new_bias_l1)
    parameters03.layer_2.weight .= Float32(new_weight_l2)
    parameters03.layer_2.bias   .= Float32(new_bias_l2)

    println(f03_loss(parameters03))
end

y_prediction, _ = model03(xgrid', parameters03, layer_states03)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
lines!(ax, xgrid[:], y_prediction[:]; color = :blue, label = "prediction", linewidth = 5)
scatter!(ax, xgrid[:], data03[:]; color = :black, label = "data")
fig

f03_loss(parameters03)

# ==========================================
# Train with the Optimization.jl library (Adam)
# ==========================================
# Same network, but let the optimiser drive training; a callback logs the loss.
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
