# =====================================================================================
# 4.3 - Newton-Raphson with hand-built Dual numbers
# =====================================================================================
# GOAL: find the radius r where dA/dr = 0 for the cylinder problem, using our own
#       forward-mode AD instead of a library.
# STRATEGY: one Dual evaluation of f(r) gives both f(r).v (the value) and f(r).∂ (the
#           derivative), which is exactly what Newton-Raphson needs each step.
# =====================================================================================

# Extended Dual number (mutable) and its operations (repeated so this runs alone).
mutable struct Dual <: Number
    v::Float64 # value of the function
    ∂::Float64 # its derivative
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

# f(r) = dA/dr for the cylinder; the root is the optimal radius.
f(r) = 4π * r - 2.0(1 / r) * (1 / r)

f(Dual(0.2, 1.0))

# Newton-Raphson: value and derivative both come from one Dual evaluation.
r = Dual(1.0, 1.0)
for i = 1:50
    println(r)
    r = r - f(r).v / f(r).∂
end
