# =====================================================================================
# 6.2 - Training a Lux network by hand-coded gradient descent
# =====================================================================================
# GOAL: fit the linear Lux network NN(x) = Wx + b to noisy data generated from y = x + 3.
# STRATEGY: define a mean-squared-error loss on the Lux parameters, get its gradient with
#           Zygote, and update W and b by plain gradient descent (no optimiser library).
# =====================================================================================

using Lux
using Random
using Zygote
using Statistics
using CairoMakie

# Model and its (random) parameters.
model01 = Chain(Dense(1 => 1))
rng = Random.default_rng()
parameters, layer_states = Lux.setup(rng, model01)

# Artificial (noisy) data from y(x) = x + 3.
xgrid = collect(-3.0:0.01:3.0)
N_SAMPLES = length(xgrid)
y_data = xgrid .+ 3 + 0.1 * randn(N_SAMPLES)

# Untrained prediction vs data.
y_prediction, _ = model01(xgrid', parameters, layer_states)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
lines!(ax, xgrid, y_prediction[:]; color = :blue, label = "prediction", linewidth = 5)
scatter!(ax, xgrid, y_data[:]; color = :black, label = "data")
fig

# Loss: mean squared error between prediction and data.
function loss_fn(p, ls)
    y_prediction, _ = model01(xgrid', p, ls)
    loss = 0.5 * mean((y_prediction[:] .- y_data).^2)
    return loss
end

# Freeze the layer state so the loss is a function of the parameters only.
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
