# =====================================================================================
# 4.1 - Numerical differentiation
# =====================================================================================
# GOAL: approximate f'(x) for f(x) = sin(x^2) with finite differences and the
#       complex-step trick, and understand their accuracy.
# STRATEGY: compare forward / backward / central differences and the complex step, then
#           plot error vs step size h to watch truncation error and round-off error
#           fight each other (central and complex step win).
# =====================================================================================

using CairoMakie

f(x) = sin(x^2)
dfdx(x) = 2 * x * cos(x^2)

# Single-point comparison of the four schemes against the exact value.
h = 1e-3
xi = π / 2.0
println("forward_diff=$( (f(xi+h)-f(xi))/h )")
println("backward_diff=$( (f(xi)-f(xi-h))/h )")
println("central_diff=$( (f(xi+h)-f(xi-h))/(2*h) )")
println("complex_step=$( imag(f(xi+im*h))/h )")
println("exact_diff=$( dfdx(xi) )")

# The four schemes as functions of the step size h.
diff_forward(f, x; h = sqrt(eps(Float64))) = (f(x + h) - f(x)) / h
diff_central(f, x; h = cbrt(eps(Float64))) = (f(x + h / 2) - f(x - h / 2)) / h
diff_backward(f, x; h = sqrt(eps(Float64))) = (f(x) - f(x - h)) / h
diff_complex(f, x; h = 1e-20) = imag(f(x + h * im)) / h

# Error vs h, swept from 10^-17 to 10^1.
x = π / 2
d_true = π * cos(π^2 / 4) # true value of the derivative

arr_h = collect(10 .^ range(-17, stop = 1, length = 101))
arr_forward = [abs(d_true - diff_forward(f, x, h = h)) for h in arr_h]
arr_central = [abs(d_true - diff_central(f, x, h = h)) for h in arr_h]
arr_complex = [abs(d_true - diff_complex(f, x, h = h)) for h in arr_h]

fig1 = Figure()
ax1 = Axis(fig1[1, 1], xscale = log10, yscale = log10, xlabel = "step size h", ylabel = "Absolute Error", title = "Error vs h")
lines!(ax1, arr_h, arr_forward; color = :red, label = "Forward")
lines!(ax1, arr_h, arr_central; color = :blue, label = "Central Difference")
lines!(ax1, arr_h, arr_complex; color = :green, label = "Complex Step")
axislegend(position = :lb)
fig1
