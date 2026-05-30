# =====================================================================================
# 4.2 - Dual numbers for multivariable partial derivatives
# =====================================================================================
# GOAL: get partial derivatives of a function of several variables with the same Dual
#       machinery.
# STRATEGY: to get df/dx_i, seed variable i with derivative 1 and all the others with 0,
#           then run the function. This needs one pass per input - the limitation that
#           reverse-mode AD later removes.
# =====================================================================================

# Same Dual number and operations as 01_DualNumbersUnivariate.jl (repeated so this runs alone).
struct Dual <: Number
    v::Float64
    ∂::Float64
end

import Base: +, -, *, sin, ^
+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
+(a::Dual, b::Number) = Dual(a.v + b, a.∂)
-(a::Dual, b::Dual) = Dual(a.v - b.v, a.∂ - b.∂)
*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.∂ + b.v * a.∂)
*(a::Number, b::Dual) = Dual(a * b.v, a * b.∂)
^(a::Dual, n::Integer) = Dual(a.v^n, n * a.v^(n - 1) * a.∂)
sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.∂)

quad(x, y) = (3.0 / 2.0) * x^2 + 2 * x * y + 3 * y^2 - 2 * x + 8 * y

# df/dx at (1, 1): seed x with 1, y with 0. The result's ∂ field is the partial.
xp = Dual(1.0, 1.0)
yp = Dual(1.0, 0.0)
quad(xp, yp)

# df/dy at (1, 1): seed y with 1, x with 0.
xp = Dual(1.0, 0.0)
yp = Dual(1.0, 1.0)
quad(xp, yp)

# TODO (Exercise Ex03): What do you type to find df/dy at (x, y) = (3.0, 1.0)?
