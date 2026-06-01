# =====================================================================================
# 6.1 - A single neuron, tuned by hand
# =====================================================================================
# GOAL: see how the weight W and bias b control the output of a single neuron - first
#       the linear unit NN(x) = Wx + b, then the nonlinear unit NN(x) = tanh(Wx + b) -
#       and try to match some data by adjusting W and b yourself.
# STRATEGY: no optimisation yet; you tune the parameters manually and watch the curve.
# =====================================================================================

using CairoMakie

# ------------------------------------------------------------------
# Linear unit: NN(x) = Wx + b  (this is just a straight line)
# ------------------------------------------------------------------
xplot = Array(-3:0.01:3)
W = 2
b = 1
function NN01(x, W, b)
    return W * x + b
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", title = "Neural Network")
lines!(ax, xplot, NN01.(xplot, W, b))
fig

# TODO (Exercise Ex01): Change the values of W and b and comment on how they change f(x).

# ------------------------------------------------------------------
# Nonlinear unit: NN(x) = tanh(Wx + b)
# ------------------------------------------------------------------
xplot = Array(-3:0.01:3)
W = 1
b = 0
function NN02(x, W, b)
    return tanh(W * x + b)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", title = "Neural Network")
lines!(ax, xplot, NN02.(xplot, W, b))
fig

# TODO (Exercise Ex02): Change W and b and comment on how they change f(x).

# ------------------------------------------------------------------
# Try to match some (fictitious) data by hand
# ------------------------------------------------------------------
xplot = Array(-3:0.01:3)
xdata = Array(-3:0.1:3)
W = 1
b = 0.1

fig = Figure(backgroundcolor = :transparent)
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", backgroundcolor = :transparent)
#lines!(ax,xplot,NN02.(xplot,W,b);label="NN model",color=:blue,linewidth=5)
scatter!(ax, xdata, 1.5 * sin.(π * xdata / 5.0); color = :black, label = "Data") # fictitious data
# save("../../figures/06p01IntroductionToNeuralNetwork01.png", fig)   # (figure-save disabled in study file)
axislegend(ax, position = :rb, orientation = :vertical)
fig

# TODO (Exercise Ex03): Uncomment the NN02 line above, then change W and b and see if you
#   can make NN a closer match to the black dots (data). Can you get a good match with a
#   single neuron for this data?
