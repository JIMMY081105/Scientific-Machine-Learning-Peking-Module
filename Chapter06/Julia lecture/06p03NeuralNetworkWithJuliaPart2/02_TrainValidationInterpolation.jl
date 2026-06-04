# =====================================================================================
# 6.3 - Training / validation split (interpolation)
# =====================================================================================
# GOAL: train only on part of the data and measure how well the network predicts the
#       part it never saw - here the training points are scattered across the whole
#       range, so the test is interpolation.
# STRATEGY: randomly pick 50% of the points as the training set, train a (5, 5) network
#           with Optimization.jl (Adam, AutoZygote), then compute the mean squared error
#           on the held-out validation points.
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

# View the data.
fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "data")
# save("../../figures/06p03NeuralNetworkWithJuliaPart2.png", fig)   # (figure-save disabled in study file)
fig

# Randomly select 50% of the points for training (scattered across the whole range).
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
# save("../../figures/06p03NeuralNetworkWithJuliaPart204.png", fig)   # (figure-save disabled in study file)
fig

# A 2-hidden-layer (5, 5) network.
model_training = Chain(Dense(1 => 5, tanh),
    Dense(5 => 5, tanh),
    Dense(5 => 1))

rng = Random.default_rng()
ps, ls = Lux.setup(rng, model_training)

# Loss on the training set, trained with Adam via Optimization.jl.
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

# Prediction over the full dataset (interpolation is good here).
fig = Figure(backgroundcolor = :transparent)
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent, limits = (0, 5, -2, 2))
scatter!(ax, xdata, ydata; color = :black, label = "all data")
scatter!(ax, xtraining, ytraining; color = :red, label = "training data")
scatter!(ax, xdata, model_training(xdata', ps, ls)[1][:]; color = :blue, label = "predicted data")
axislegend(ax, location = :rt)
# save("../../figures/06p03NeuralNetworkWithJuliaPart205.png", fig)   # (figure-save disabled in study file)
fig

# Mean squared error on the held-out validation points.
validation_indices = setdiff(1:N_SAMPLES, training_indices)
ypred, _ = model_training(xdata[validation_indices]', ps, ls)
mean((ypred .- ydata[validation_indices]').^2)
