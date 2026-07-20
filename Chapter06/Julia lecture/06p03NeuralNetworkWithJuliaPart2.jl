########################################################################################
# 6.3 Neural Network with Julia (Part 2)
########################################################################################
#
# PROBLEM
# Build deeper and wider networks with Lux, and study generalisation: split data into
# training and validation sets, train only on part of it, and test how well the
# network predicts unseen data - both by interpolation (a random split) and by
# extrapolation (train on x < x0, predict beyond x0).
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Show how to construct multi-node / multi-layer networks with Chain(Dense(...)):
#      a 1-hidden-layer (2 nodes) network and a 2-hidden-layer (4 and 6 nodes) network.
#   2. Generate noisy data from h(x) = 2*cos(pi*x)*exp(-x^2/5) on [0, 5].
#   3. Interpolation test: randomly pick 50% of the points as the training set, train a
#      2-hidden-layer (5, 5) network with Optimization.jl (Adam, AutoZygote), plot the
#      prediction over all data, and compute the mean squared error on the held-out
#      validation set (prediction inside the sampled region is good).
#   4. Extrapolation test: retrain the same architecture using only data with x < x0 = 2,
#      then predict for x > x0 and compute the validation MSE - showing the network
#      interpolates well but extrapolates poorly.
#
# This file is notebook "06p03NeuralNetworkWithJuliaPart2" with every code cell joined
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
# Constructing deeper / wider networks
# ==========================================
# 1 input, 1 output, 1 hidden layer with 2 nodes.
model = Chain(Dense(1 => 2, tanh),
    Dense(2 => 1))

# 1 input, 1 output, 2 hidden layers with 4 and 6 nodes.
model = Chain(Dense(1 => 4, tanh),
    Dense(4 => 6, tanh),
    Dense(6 => 1))

# ==========================================
# Training vs validation data
# ==========================================
# The point of a model is to predict where we have no data. Generate noisy data from
# h(x) = 2*cos(pi*x)*exp(-x^2/5) on [0, 5], then hold some of it back for validation.
N_SAMPLES = 100
xdata = 5 * rand(N_SAMPLES)
h(x) = 2.0cos(pi * x) * exp(-x^2 / 5)
ydata = h.(xdata) .+ 0.05 * randn(N_SAMPLES)

# View the data.
fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "data")
# save("../figures/06p03NeuralNetworkWithJuliaPart2.png", fig)   # (figure-save disabled in study file)
fig

# ==========================================
# Interpolation: random 50% training split
# ==========================================
#training_indices=rand(eachindex(xdata),round(Int,0.5*N_SAMPLES))
training_indices = shuffle(eachindex(xdata))[1:round(Int, 0.5 * N_SAMPLES)]
xtraining = xdata[training_indices]
ytraining = ydata[training_indices]

# Training data overlaid on the full dataset.
fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
axislegend(ax, location = :rt)
# save("../figures/06p03NeuralNetworkWithJuliaPart204.png", fig)   # (figure-save disabled in study file)
fig

# A 2-hidden-layer (5, 5) network.
model_training = Chain(Dense(1 => 5, tanh),
    Dense(5 => 5, tanh),
    Dense(5 => 1))

# Initialize the model weights and state
rng = Random.default_rng()
ps, ls = Lux.setup(rng, model_training)

# Loss and training via Optimization.jl (Adam, AutoZygote), logged through a callback.
function loss_training(ps, ytraining)
    ypred, _ = model_training(xtraining', ps, ls)
    loss = 0.5 * mean((ypred .- ytraining').^2)
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
optf = Optimization.OptimizationFunction(loss_training, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ps), ytraining)

# Initialize a vector to store loss values
loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 10000)

ps = neural_network.u

# Prediction over the full dataset.
fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
scatter!(ax, xdata, model_training(xdata', ps, ls)[1][:]; color = :blue, label = "predicted data")
axislegend(ax, location = :rt)
# save("../figures/06p03NeuralNetworkWithJuliaPart205.png", fig)   # (figure-save disabled in study file)
fig

# Mean squared error on the held-out validation data (good, since we interpolate).
validation_indices = setdiff(1:N_SAMPLES, training_indices)
ypred, _ = model_training(xdata[validation_indices]', ps, ls)
mean((ypred .- ydata[validation_indices]').^2)

# ==========================================
# Extrapolation: train on x < x0, predict beyond
# ==========================================
# Train only where x < x0 = 2, then see how the network predicts for x > x0.
training_region = 2.0
training_indices = findall(xdata .< training_region)
xtraining = xdata[training_indices]
ytraining = ydata[training_indices]

fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
axislegend(ax, location = :rt)
# save("../figures/06p03NeuralNetworkWithJuliaPart202.png", fig)   # (figure-save disabled in study file)
fig

model_training = Chain(Dense(1 => 5, tanh),
    Dense(5 => 5, tanh),
    Dense(5 => 1))

# Initialize the model weights and state
rng = Random.default_rng()
ps, ls = Lux.setup(rng, model_training)

# The loss function
function loss_training(ps, ytraining)
    ypred, _ = model_training(xtraining', ps, ls)
    loss = 0.5 * mean((ypred .- ytraining').^2)
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
optf = Optimization.OptimizationFunction(loss_training, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ps), ytraining)

# Initialize a vector to store loss values
loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 10000)

ps = neural_network.u

fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
scatter!(ax, xdata, model_training(xdata', ps, ls)[1][:]; color = :blue, label = "predicted data")
axislegend(ax, location = :rt)
#save("../figures/06p03NeuralNetworkWithJuliaPart202.png",fig)
fig

# Mean squared error on the validation data (worse, since we now extrapolate).
validation_indices = setdiff(1:N_SAMPLES, training_indices)
ypred, _ = model_training(xdata[validation_indices]', ps, ls)
mean((ypred .- ydata[validation_indices]').^2)

# TODO (Class Demo 01): Construct a Neural Network using Lux in Julia to fit a limited
#   section of the data in "Data06.csv", then use it to predict the data beyond the
#   training region.
