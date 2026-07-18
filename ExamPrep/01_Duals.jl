# =====================================================================================
# DUAL NUMBERS (hand-built forward-mode AD) — Level 1 (no packages at all)
# =====================================================================================
# The idea: a Dual carries a value v and its derivative ∂. Every operation updates
# both using the chain rule. Seed the variable you differentiate with ∂ = 1.0.
# Sources: Chapter04 04p02/04p03, Homework/Day06, Homework/Day07.
# =====================================================================================

struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end

# Overload ONLY the operations your function needs (add more in the same pattern).
import Base: +, -, *, /, ^, exp, cos, sin

+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
-(a::Dual) = Dual(-a.v, -a.∂)
-(a::Dual, b::Dual) = Dual(a.v - b.v, a.∂ - b.∂)
*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.∂ + b.v * a.∂)
/(a::Dual, b::Dual) = Dual(a.v / b.v, (a.∂ * b.v - a.v * b.∂) / b.v^2)
^(a::Dual, n::Integer) = Dual(a.v^n, n * a.v^(n - 1) * a.∂)
exp(a::Dual) = Dual(exp(a.v), exp(a.v) * a.∂)
cos(a::Dual) = Dual(cos(a.v), -sin(a.v) * a.∂)
sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.∂)

# Mixing Duals with plain numbers (needed for things like 3*x2 or x+1)
+(a::Number, b::Dual) = Dual(a + b.v, b.∂)
+(a::Dual, b::Number) = Dual(a.v + b, a.∂)
-(a::Number, b::Dual) = Dual(a - b.v, -b.∂)
-(a::Dual, b::Number) = Dual(a.v - b, a.∂)
*(a::Number, b::Dual) = Dual(a * b.v, a * b.∂)
*(a::Dual, b::Number) = Dual(a.v * b, a.∂ * b)
/(a::Dual, b::Number) = Dual(a.v / b, a.∂ / b)
/(a::Number, b::Dual) = Dual(a / b.v, -a * b.∂ / b.v^2)

# -------------------------------------------------------------------------------------
# 1) Derivative of a 1-variable function: seed with ∂ = 1.0, read .∂ of the result
# -------------------------------------------------------------------------------------
f(x) = exp(-x^2) * cos(x)

x = 2.0
result = f(Dual(x, 1.0))
println("f($x)     = ", result.v)
println("df/dx($x) = ", result.∂)
# check against the analytical answer: exp(-x^2)*(-2x*cos(x)-sin(x))
println("analytic  = ", exp(-x^2) * (-2 * x * cos(x) - sin(x)))

# -------------------------------------------------------------------------------------
# 2) Partial derivatives of a 2-variable function:
#    seed the variable you want with ∂=1.0, the OTHER with ∂=0.0. One pass per input.
# -------------------------------------------------------------------------------------
g(x1, x2) = exp(-x1^2) * cos(3 * x2)

# df/dx1 at (2,3): seed x1
x1 = Dual(2.0, 1.0)
x2 = Dual(3.0, 0.0)
println("df/dx1 = ", g(x1, x2).∂)

# df/dx2 at (2,3): seed x2
x1 = Dual(2.0, 0.0)
x2 = Dual(3.0, 1.0)
println("df/dx2 = ", g(x1, x2).∂)

# -------------------------------------------------------------------------------------
# 3) Newton-Raphson using Duals (value AND derivative from ONE evaluation)
#    Solve h(x) = 0. Update: x <- x - h(x)/h'(x)
# -------------------------------------------------------------------------------------
h(x) = 4pi * x - 2 / x^2   # example: dA/dr of the cylinder; replace with your function

x = 1.0
for i = 1:10
    r = h(Dual(x, 1.0))
    global x = x - r.v / r.∂
    println("iteration $i: x = $x")
end

# -------------------------------------------------------------------------------------
# 4) Gradient descent with a Dual-computed derivative (fit a constant a0 to data)
#    Loss S(a0) = 0.5*sum((yi-a0)^2); dS/da0 comes from evaluating S on a Dual.
# -------------------------------------------------------------------------------------
yi = [2.2, 2.4, 2.3, 2.35, 2.25]   # replace with the question's data
S(a0) = 0.5 * sum((yi .- a0) .^ 2)

a0 = 3.0
α = 0.01
for i = 1:1000
    dS = S(Dual(a0, 1.0)).∂
    global a0 -= α * dS
end
println("fitted a0 = ", a0)
