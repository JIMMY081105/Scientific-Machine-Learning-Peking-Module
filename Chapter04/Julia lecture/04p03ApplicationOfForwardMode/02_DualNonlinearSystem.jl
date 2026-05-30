# =====================================================================================
# 4.3 - Nonlinear system solved with a Dual-number Jacobian
# =====================================================================================
# GOAL: solve a 2x2 nonlinear system with Newton-Raphson, building the Jacobian from our
#       own Dual numbers.
# STRATEGY: evaluate the vector function with one variable seeded at a time (∂ = 1 for
#           that variable, 0 for the rest); stacking the ∂ fields gives each column of
#           the Jacobian.
# =====================================================================================

# Extended Dual number (mutable) and its operations (repeated so this runs alone).
mutable struct Dual <: Number
    v::Float64
    ∂::Float64
end

import Base: +, -, *, sin, ^, /, cos
+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
+(a::Dual, b::Number) = Dual(a.v + b, a.∂)
+(a::Number, b::Dual) = Dual(a + b.v, b.∂)
-(a::Dual, b::Dual) = Dual(a.v - b.v, a.∂ - b.∂)
-(a::Number, b::Dual) = Dual(a - b.v, -b.∂)
-(a::Dual, b::Number) = Dual(a.v - b, a.∂)
*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.∂ + b.v * a.∂)
*(a::Number, b::Dual) = Dual(a * b.v, a * b.∂)
/(a::Dual, b::Dual) = Dual(a.v / b.v, (b.v * a.∂ - a.v * b.∂) / (b.v)^2)
/(a::Number, b::Dual) = Dual(a / b.v, (-a * b.∂) / (b.v)^2)
^(a::Dual, n::Integer) = Dual(a.v^n, n * a.v^(n - 1) * a.∂)
sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.∂)
cos(a::Dual) = Dual(cos(a.v), -sin(a.v) * a.∂)

VectorFunc(x) = [x[1]^2 - x[2] + 1,
    3 * cos(x[1]) - x[2]]

# Partial derivatives by seeding one variable at a time.
df1dx1 = VectorFunc([Dual(1.0, 1.0), Dual(1.0, 0.0)])[1].∂
df1dx1 = VectorFunc([Dual(1.0, 1.0), Dual(1.0, 0.0)])[2].∂

# Assemble the full Jacobian: column 1 seeds x1, column 2 seeds x2.
xvals = [1, 1]
J = hcat(getfield.(VectorFunc([Dual(xvals[1], 1.0), Dual(xvals[2], 0.0)])[:], :∂),
    getfield.(VectorFunc([Dual(xvals[1], 0.0), Dual(xvals[1], 1.0)])[:], :∂))

# Newton-Raphson iteration on the system.
xvals = [1, 1]
for i = 1:10
    F = VectorFunc(xvals)
    J = hcat(getfield.(VectorFunc([Dual(xvals[1], 1.0), Dual(xvals[2], 0.0)])[:], :∂),
        getfield.(VectorFunc([Dual(xvals[1], 0.0), Dual(xvals[1], 1.0)])[:], :∂))
    delta = -J \ F
    xvals += delta
    println(xvals)
end
