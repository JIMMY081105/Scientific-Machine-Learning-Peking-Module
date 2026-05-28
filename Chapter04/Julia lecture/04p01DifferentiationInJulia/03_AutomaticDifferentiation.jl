# =====================================================================================
# 4.1 - Automatic differentiation (ForwardDiff and Zygote)
# =====================================================================================
# GOAL: get machine-precision derivatives of f(x) = sin(x^2) with no hand derivatives
#       and no step-size error.
# STRATEGY: call ForwardDiff.derivative and Zygote.gradient, then plot both against the
#           analytic derivative to confirm they land exactly on it.
# =====================================================================================

using CairoMakie
using ForwardDiff
using Zygote

f(x) = sin(x^2)
x = π / 2
d_true = π * cos(π^2 / 4)

# Both packages give the derivative to machine precision.
ForwardDiff.derivative(f, π / 2)
println("The true answer is ", d_true, " ForwardDiff gives ", ForwardDiff.derivative(f, x))

Zygote.gradient(f, x)
println("The true answer is ", d_true, " Zygote gives ", Zygote.gradient(f, x)[1])

# Wrap them as derivative functions and plot against the ground truth.
dfForwardDiff(x) = ForwardDiff.derivative(f, x)
dfZygote(x) = Zygote.gradient(f, x)[1]

xplot = -1:0.01:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "df/dx", limits = (-1.0, 1.0, -2.0, 2.0))
lines!(xplot, 2 * xplot .* cos.(xplot.^2); color = :black, label = "Ground truth")
lines!(xplot, dfForwardDiff.(xplot); color = :blue, label = "ForwardDiff")
lines!(xplot, dfZygote.(xplot); color = :green, label = "Zygote")
axislegend(ax, position = :rb)
fig
