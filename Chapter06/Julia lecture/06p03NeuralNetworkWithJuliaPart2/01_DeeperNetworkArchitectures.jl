# =====================================================================================
# 6.3 - Constructing deeper and wider networks
# =====================================================================================
# GOAL: see how to describe multi-node, multi-layer networks with Lux's Chain/Dense
#       syntax - the widths and depths are just arguments.
# =====================================================================================

using Lux

# 1 input, 1 output, 1 hidden layer with 2 nodes.
model = Chain(Dense(1 => 2, tanh),
    Dense(2 => 1))

# 1 input, 1 output, 2 hidden layers with 4 and 6 nodes.
model = Chain(Dense(1 => 4, tanh),
    Dense(4 => 6, tanh),
    Dense(6 => 1))
