# =====================================================================================
# 6.2 - Hidden-layer Lux network, trained by hand-coded gradient descent
# =====================================================================================
# GOAL: fit NN(x) = W2*tanh(W1*x + b1) + b2 (a Chain of two Dense layers) to the target
#       g(x) = 2*sin(pi*x/5).
# STRATEGY: same hand-coded gradient descent as before, but now stepping all four
#           parameters across the two layers using Zygote gradients.
# =====================================================================================

using Lux
using Random
using Zygote
using Statistics
using CairoMakie

# Hidden layer: one tanh layer feeding a linear layer.
model03 = Chain(Dense(1 => 1, tanh),
    Dense(1 => 1))

rng = Random.default_rng()
parameters03, layer_states03 = Lux.setup(rng, model03)

# Optional: start from W1 = W2 = 1, b1 = b2 = 0.
parameters03.layer_1.weight .= Float32(1.0)
parameters03.layer_2.weight .= Float32(1.0)
parameters03.layer_1.bias   .= Float32(0.0)
parameters03.layer_2.bias   .= Float32(0.0)

# Data from g(x) = 2*sin(pi*x/5) and the untrained fit.
xgrid = collect(-3.0:0.01:3.0)
g(x) = 2.0sin(pi * x / 5)
data03 = g.(xgrid)

y_prediction, _ = model03(xgrid', parameters03, layer_states03)
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
