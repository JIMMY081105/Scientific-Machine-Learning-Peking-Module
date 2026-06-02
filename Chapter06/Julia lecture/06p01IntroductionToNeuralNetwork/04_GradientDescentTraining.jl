# =====================================================================================
# 6.1 - Training the hidden-layer network by hand-coded gradient descent
# =====================================================================================
# GOAL: automatically tune the four parameters of NN(x) = W2*tanh(W1*x + b1) + b2 so it
#       fits the target g(x) = 2*sin(pi*x/5).
# STRATEGY: define a squared-error loss, get dLoss/dW and dLoss/db from Zygote automatic
#           differentiation, and step the parameters downhill (plain gradient descent).
# =====================================================================================

using CairoMakie
using Zygote

# The hidden-layer network from 03_HiddenLayerModel.jl (redefined so this file runs alone).
function NN03(x, W, b)
    return W[2] * tanh.(W[1] * x .+ b[1]) .+ b[2]
end

# Target function and the data we want to fit.
g(x) = 2 * sin(pi * x / 5)
xdata = Array(-3:0.1:3)
data = g.(xdata)

# Untrained network on top of the data.
W = [1.0, 1.0]
b = [0.0, 0.0]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
scatter!(ax, xdata, data; color = :black, label = "Data")
lines!(ax, xdata, NN03(xdata, W, b); color = :blue, linewidth = 5, label = "Neural Network")
axislegend(ax, position = :rb, orientation = :vertical)
fig

# Loss (error) function: sum of squared errors between model and data.
function Loss(xdata, W, b)
    return sum((data - NN03(xdata, W, b)).^2)
end

# Freeze the data argument so the loss is a function of (W, b) only.
ff = (W, b) -> Loss(xdata, W, b)

# Hand-coded gradient descent.
W = [1.0, 1.0]
b = [0.0, 0.0]
α = 0.01
loss_values = [Loss(xdata, W, b)]

for i = 1:100
    grads = Zygote.gradient(ff, W, b)
    W -= α * grads[1]
    b -= α * grads[2]
    push!(loss_values, Loss(xdata, W, b))
    println(ff(W, b))
end

# Loss history (log-log).
fig = Figure()
ax = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "iter", ylabel = "loss")
lines!(ax, loss_values; color = :blue)
fig

# Trained network against the data.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", title = "Neural Network")
scatter!(ax, xdata, data; color = :black, label = "Data")
lines!(ax, xdata, NN03(xdata, W, b); color = :blue, linewidth = 5, label = "Neural Network")
fig
