# =====================================================================================
# 6.1 - Activation functions
# =====================================================================================
# GOAL: plot the three activation functions a neuron can use - identity, sigmoid and
#       tanh - and see how each one shapes its input s = Wx + b.
# =====================================================================================

using CairoMakie

xplot = -3:0.01:3

function sigmoid(s)
    return 1.0 / (1.0 + exp(-s))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"s", ylabel = L"\sigma(s)", limits = (-3, 3, -3.0, 3.0), title = "Identity")
lines!(ax, xplot, xplot)

ax = Axis(fig[1, 2], xlabel = L"s", ylabel = L"\sigma(s)", limits = (-3, 3, -3.0, 3.0), title = "Sigmoid")
lines!(ax, xplot, sigmoid.(xplot))

ax = Axis(fig[1, 3], xlabel = L"s", ylabel = L"\sigma(s)", limits = (-3, 3, -3.0, 3.0), title = "Tanh")
lines!(ax, xplot, tanh.(xplot))

# save("../../figures/ActivationFunction.svg", fig)   # (figure-save disabled in study file)
fig
