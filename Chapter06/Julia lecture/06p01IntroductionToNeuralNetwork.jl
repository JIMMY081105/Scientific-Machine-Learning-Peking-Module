########################################################################################
# 6.1 Introduction to Neural Networks
########################################################################################
#
# PROBLEM
# Understand what a neural network is (a parameterised function that can approximate
# any function) and learn to train one: fit the weights and biases so the network
# matches data, using a loss function and gradient descent with
# automatically-computed gradients.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Start with one input, one output: NN(x; W, b) = sigma(Wx + b); plot the
#      activation functions (identity, sigmoid, tanh).
#   2. See a plain linear unit (sigma = identity) is just a line, then add
#      nonlinearity with tanh; adjust W, b by hand and watch the curve change against
#      some data.
#   3. Add a hidden layer: NN(x) = W2*tanh(W1*x + b1) + b2, giving four tunable
#      parameters and a far more flexible shape.
#   4. Train it: define Loss(W, b) = sum of squared errors to a target g(x), get
#      dLoss/dW and dLoss/db from Zygote.gradient, and run hand-coded gradient
#      descent until the network fits the data.
#   5. Go wider and higher-dimensional: hidden layers with several nodes (more
#      parameters), and a network with two inputs visualised as a contour plot.
#
# This file is notebook "06p01IntroductionToNeuralNetwork" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ========================================
# 6.1 Introduction to Neural Network
# ========================================

using CairoMakie
using Zygote

# ==========================
# One input one output
# ==========================
# Let's plot the functions and see how they look like

xplot=-3:0.01:3
function sigmoid(s)
    return 1.0 / (1.0 + exp(-s))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"s", ylabel = L"\sigma(s)",limits=(-3,3,-3.0,3.0),title="Identity")
lines!(ax,xplot,xplot)


ax = Axis(fig[1, 2], xlabel = L"s", ylabel = L"\sigma(s)",limits=(-3,3,-3.0,3.0),title="Sigmoid")
lines!(ax,xplot,sigmoid.(xplot))

ax = Axis(fig[1, 3], xlabel = L"s", ylabel = L"\sigma(s)",limits=(-3,3,-3.0,3.0),title="Tanh")
lines!(ax,xplot,tanh.(xplot))

# save("../figures/ActivationFunction.svg",fig)   # (figure-save disabled in study file)

fig

xplot=Array(-3:0.01:3)
W=2
b=1
function NN01(x,W,b)
    return W*x+b
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",title="Neural Network")
lines!(ax,xplot,NN01.(xplot,W,b))

fig

# TODO (Exercise Ex01): Change the values of W and b and comment on how they change f(x).

xplot=Array(-3:0.01:3)
W=1
b=0
function NN02(x,W,b)
    return tanh(W*x+b)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",title="Neural Network")
lines!(ax,xplot,NN02.(xplot,W,b))

fig

# TODO (Exercise Ex02): Change W and b and comment on how they change f(x).

xplot=Array(-3:0.01:3)
xdata=Array(-3:0.1:3)
W=1
b=0.1
function NN02(x,W,b)
    return tanh(W*x+b)
end

fig = Figure(backgroundcolor = :transparent)
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",backgroundcolor = :transparent)
#lines!(ax,xplot,NN02.(xplot,W,b);label="NN model",color=:blue,linewidth=5)
scatter!(ax,xdata,1.5*sin.(π*xdata/5.0);color=:black,label="Data") #fictitious data

# save("../figures/06p01IntroductionToNeuralNetwork01.png",fig)   # (figure-save disabled in study file)

axislegend(ax,  position = :rb,orientation = :vertical)
fig

# TODO (Exercise Ex03): Change W and b and see if you can make NN a closer match to the black dots
#   (data). Can you get a good match with the data for this Neural Network?

# ====================
# Hidden layers
# ====================
# This Neural Network can be visualised below.

xplot=Array(-3:0.01:3)
xdata=Array(-3:0.1:3)

W=[0.8,2]
b=[0,0]

function NN03(x,W,b)
    return W[2]*tanh.(W[1]*x.+b[1]).+b[2]
end

fig = Figure(backgroundcolor=:transparent)
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",backgroundcolor=:transparent)
lines!(ax,xplot,NN03(xplot,W,b);color=:blue,linewidth=5,label="Neural Network model")
scatter!(ax,xdata,2*sin.(π*xdata/5.0);color=:black,label="Data") #fictitious data
# #save("../figures/06p01IntroductionToNeuralNetwork02.png",fig)   # (figure-save disabled in study file)
axislegend(ax,  position = :rb,orientation = :vertical)

fig

# ===============================================
# Using optimisers to tune a Neural Network
# ===============================================

g(x)=2*sin(pi *x/5 )
xplot=Array(-3:0.1:3)
data=g.(xplot)

W=[1.0,1.0]
b=[0.0,0.0]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
scatter!(ax,xplot,data;color=:black,label="Data")
lines!(ax,xplot,NN03(xplot,W,b);color=:blue,linewidth=5,label="Neural Network")

axislegend(ax,  position = :rb,orientation = :vertical)
fig
# To optimise the fit, we first need to define a loss (error) function to quantify
# the error between your model and your data.

function Loss(xdata,W,b)
    return sum((data-NN03(xdata,W,b)).^2)
end

ff=(W,b)->Loss(xdata,W,b)

W=[1.0,1.0]
b=[0.0,0.0]
α=0.01
loss_values=[Loss(xdata,W,b)]
   
for i=1:100
    grads=Zygote.gradient(ff,W,b)
    W-=α*grads[1]
    b-=α*grads[2]
    push!(loss_values,Loss(xdata,W,b))
    println(ff(W,b))
end

fig = Figure()
ax = Axis(fig[1, 1], xscale=log10,yscale=log10,xlabel = "iter", ylabel = "loss")
lines!(ax,loss_values;color=:blue)
fig
# Plot the data and prediction to see if we can get a good fit

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",title="Neural Network")
scatter!(ax,xplot,data;color=:black, label="Data")
lines!(ax,xplot,NN03(xplot,W,b);color=:blue,linewidth=5,label="Neural Network")
fig
# The plot above now shows that you have a good match with the data.

# ====================================
# A bit more about hidden layers
# ====================================
# (the figure above was made using https://alexlenail.me/NN-SVG/)

# =====================
# Multiple Inputs
# =====================
# (the figure above was made using https://alexlenail.me/NN-SVG/)

x1range=-5:0.02:5;
x2range=-5:0.02:5;

W=[0.3 1.0]
b=[1]
function f(x)
    return tanh.(W*x+b)
end


x = [[x1i, x2i] for x1i in x1range, x2i in x2range]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Neural Network",limits=(-5.0,5.0,-5.0,5.0))
contour!(ax,x1range,x2range,only.(f.(x)); labels=true,  colormap=:hsv,linewidth = 5)
fig
