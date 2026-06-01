# =====================================================================================
# 6.1 - Adding a hidden layer
# =====================================================================================
# GOAL: build the two-layer network NN(x) = W2*tanh(W1*x + b1) + b2, which has four
#       tunable parameters, and see how much more flexible its shape is against data.
# STRATEGY: still tuned by hand - here we just look at the richer function a hidden
#           layer produces before we start training it in 04_GradientDescentTraining.jl.
# =====================================================================================

using CairoMakie

xplot = Array(-3:0.01:3)
xdata = Array(-3:0.1:3)

W = [0.8, 2]
b = [0, 0]

# NN(x) = W2 * tanh(W1*x + b1) + b2
function NN03(x, W, b)
    return W[2] * tanh.(W[1] * x .+ b[1]) .+ b[2]
end

fig = Figure(backgroundcolor = :transparent)
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent)
lines!(ax, xplot, NN03(xplot, W, b); color = :blue, linewidth = 5, label = "Neural Network model")
scatter!(ax, xdata, 2 * sin.(π * xdata / 5.0); color = :black, label = "Data") # fictitious data
# save("../../figures/06p01IntroductionToNeuralNetwork02.png", fig)   # (figure-save disabled in study file)
axislegend(ax, position = :rb, orientation = :vertical)
fig
