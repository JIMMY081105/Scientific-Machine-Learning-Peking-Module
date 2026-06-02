# =====================================================================================
# 6.2 - Adding a tanh activation
# =====================================================================================
# GOAL: turn the linear Lux unit into the nonlinear NN(x) = tanh(Wx + b) by passing an
#       activation function to Dense, and plot the resulting curve.
# STRATEGY: Chain(Dense(1 => 1, tanh)); set W = 1, b = 0 to see the plain tanh shape.
# =====================================================================================

using Lux
using Random
using CairoMakie

model02 = Chain(Dense(1 => 1, tanh))

rng = Random.default_rng()
parameters, layer_states = Lux.setup(rng, model02)
parameters.layer_1.weight .= Float32(1.0) # Change the weight to 1.0
parameters.layer_1.bias   .= Float32(0.0) # Change the bias to 0.0

xgrid = -3.0:0.01:3.0
y_prediction, _ = model02(xgrid', parameters, layer_states) # predict the output of the Neural network

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"NN(x)")
lines!(ax, xgrid[:], y_prediction[:]; color = :blue, label = "prediction")
fig
