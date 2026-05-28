# =====================================================================================
# 4.1 - Using AD in place of hand derivatives
# =====================================================================================
# GOAL: replace the hand-derived derivatives in two familiar algorithms with AD.
# STRATEGY: Newton-Raphson using ForwardDiff.derivative for f'(x), and gradient descent
#           on the Himmelblau function using ForwardDiff.gradient.
# =====================================================================================

using ForwardDiff

# --- Newton-Raphson root finding (f'(x) from ForwardDiff) ---
f2(x) = 4π * x - 2 / x^2
x = 1.0
for i = 1:10
    x -= f2(x) / ForwardDiff.derivative(f2, x)
    println(x)
end

# --- Gradient descent on the Himmelblau function (gradient from ForwardDiff) ---
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

x0 = [0.0, 0.0]
ForwardDiff.gradient(himmelblau, x0)

# TODO (Exercise Ex02): Use Zygote.gradient to get the gradient of himmelblau at (0, 0).

# Record the descent path as a list of points.
struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

x0 = [-2.0, -2.0] # initial guess
alpha = 0.01
data_gd = [Point2D(x0..., himmelblau(x0))]

x = x0
for i in 1:50
    x -= alpha * ForwardDiff.gradient(himmelblau, x)
    push!(data_gd, Point2D(x..., himmelblau(x)))
end

data_gd
