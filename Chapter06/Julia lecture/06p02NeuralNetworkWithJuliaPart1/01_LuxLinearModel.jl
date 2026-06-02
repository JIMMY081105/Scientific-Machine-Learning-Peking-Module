# =====================================================================================
# 6.2 - Defining a network with Lux.jl
# =====================================================================================
# GOAL: build the simplest network NN(x) = Wx + b with the Lux.jl library, which stores
#       the structure separately from its parameters and state, and plot it.
# STRATEGY: Chain(Dense(1 => 1)) for the structure; Lux.setup for random parameters
#           (Lux defaults to Float32); optionally overwrite W and b by hand.
# =====================================================================================

using Lux
using Random
using CairoMakie

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
# save("../../figures/06p02NeuraNetworkWithJulia01.svg", fig)   # (figure-save disabled in study file)
fig

# TODO (Class Demo 01): Define and plot the Neural Network NN(x) = sin(3x + 4) for x in [-2, 3].
