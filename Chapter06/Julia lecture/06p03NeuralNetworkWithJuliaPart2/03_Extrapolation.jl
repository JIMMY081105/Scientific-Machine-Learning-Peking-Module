# =====================================================================================
# 6.3 - Predicting outside the training region (extrapolation)
# =====================================================================================
# GOAL: train only where x < x0 = 2, then ask the network to predict for x > x0, and see
#       that a neural network extrapolates poorly (unlike the interpolation case).
# STRATEGY: same (5, 5) network and Adam training as the interpolation example, but the
#           training set is now one contiguous region instead of a random sample.
# =====================================================================================

using Lux
using Random
using Zygote
using Statistics
using CairoMakie
using Optimization
using OptimizationOptimisers
using ComponentArrays

# Noisy data from h(x) = 2*cos(pi*x)*exp(-x^2/5) on [0, 5].
N_SAMPLES = 100
xdata = 5 * rand(N_SAMPLES)
h(x) = 2.0cos(pi * x) * exp(-x^2 / 5)
ydata = h.(xdata) .+ 0.05 * randn(N_SAMPLES)

# Training set = only the points with x < x0.
training_region = 2.0
training_indices = findall(xdata .< training_region)
xtraining = xdata[training_indices]
ytraining = ydata[training_indices]

fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
axislegend(ax, location = :rt)
# save("../../figures/06p03NeuralNetworkWithJuliaPart202.png", fig)   # (figure-save disabled in study file)
fig

# Same (5, 5) architecture.
model_training = Chain(Dense(1 => 5, tanh),
    Dense(5 => 5, tanh),
    Dense(5 => 1))

rng = Random.default_rng()
ps, ls = Lux.setup(rng, model_training)

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

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 10000)

ps = neural_network.u

# The prediction tracks the data for x < x0 but drifts away beyond it.
fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
scatter!(ax, xdata, model_training(xdata', ps, ls)[1][:]; color = :blue, label = "predicted data")
axislegend(ax, location = :rt)
# save("../../figures/06p03NeuralNetworkWithJuliaPart202.png", fig)   # (figure-save disabled in study file)
fig

# Validation MSE (larger than the interpolation case - extrapolation is hard).
validation_indices = setdiff(1:N_SAMPLES, training_indices)
ypred, _ = model_training(xdata[validation_indices]', ps, ls)
mean((ypred .- ydata[validation_indices]').^2)

# TODO (Class Demo 01): Construct a Neural Network using Lux in Julia to fit a limited
#   section of the data in "Data06.csv", then use it to predict the data beyond the
#   training region.
