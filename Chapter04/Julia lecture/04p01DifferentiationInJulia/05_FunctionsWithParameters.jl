# =====================================================================================
# 4.1 - Differentiating functions with parameters
# =====================================================================================
# GOAL: differentiate NN(x, W, b) = sin(Wx + b) with respect to each of its arguments.
# STRATEGY: ForwardDiff wants a single (vector) argument, so you differentiate one
#           closure at a time; Zygote can differentiate w.r.t. several arguments at once.
#           We also chain Zygote to get a second derivative and plot both.
# =====================================================================================

using CairoMakie
using ForwardDiff
using Zygote

function NN(x, W, b)
    sin(W * x + b)
end

# ForwardDiff on a multi-argument function: differentiate one closed-over variable.
ForwardDiff.derivative((x) -> NN(x, 2, 3), 1.0)
ForwardDiff.derivative((W) -> NN(1, W, 3), 2.0)
ForwardDiff.derivative((b) -> NN(1, 2, b), 3.0)

# Zygote can take the gradient w.r.t. all three arguments at once.
Zygote.gradient(NN, 1, 2, 3)

# First and second derivative in x (nesting Zygote), plotted against the analytic ones.
W = 1
b = 0
NNd1 = (x) -> Zygote.gradient(NN, x, W, b)[1]
NNd2 = (x) -> Zygote.gradient(NNd1, x)[1]

xplot = Array(0:0.01:2π)
xdata = Array(0:0.1:2π)
W = 1.0
b = 0.0
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = L"f(x)")
lines!(ax, xplot, NNd1.(xplot); color = :blue)
lines!(ax, xplot, NNd2.(xplot); color = :black)
scatter!(ax, xdata, W * cos.(W * xdata .+ b); color = :blue)
scatter!(ax, xdata, -W * W * sin.(W * xdata .+ b); color = :black)
fig
