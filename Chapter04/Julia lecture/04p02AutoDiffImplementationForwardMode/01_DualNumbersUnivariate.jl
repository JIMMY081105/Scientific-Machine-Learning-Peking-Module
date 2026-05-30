# =====================================================================================
# 4.2 - Forward-mode AD from scratch: Dual numbers (univariate)
# =====================================================================================
# GOAL: build forward-mode automatic differentiation yourself for functions of one
#       variable.
# STRATEGY: a Dual number carries a value AND its derivative; overload each elementary
#           operation to apply the chain rule, then seed x as Dual(x, 1.0) - the
#           result's derivative field is df/dx, obtained just by running the function.
# =====================================================================================

using CairoMakie

# A Dual number: value v and derivative d(=∂).
struct Dual <: Number
    v::Float64 # value of the function
    ∂::Float64 # its derivative
end

# Overload the elementary operations so each returns the chain-rule derivative.
import Base: +, -, *, sin, ^
+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
+(a::Dual, b::Number) = Dual(a.v + b, a.∂)
-(a::Dual, b::Dual) = Dual(a.v - b.v, a.∂ - b.∂)
*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v * b.∂ + b.v * a.∂)
*(a::Number, b::Dual) = Dual(a * b.v, a * b.∂)
^(a::Dual, n::Integer) = Dual(a.v^n, n * a.v^(n - 1) * a.∂)
sin(a::Dual) = Dual(sin(a.v), cos(a.v) * a.∂)

# Seeding x with derivative 1.0 means "differentiate with respect to x".
x = Dual(8.0, 1.0)
x^2

# A more complicated example: f(x) = 3x + sin(x^2) at x = pi/2.
x = Dual(π / 2.0, 1.0)
f(x::Dual) = 3 * x + sin(x^2)
f(x)

g(x::Dual) = sin(x^2) + x^3
g(x)
π * cos(π^2 / 4) + 3 * (π / 2)^2  # the analytic derivative of g, for comparison

# Confirm the Dual derivative matches the analytic derivative across a range.
xplot = -1:0.001:1.0
f(x::Dual) = sin(x^2 + 3.0) + x^3
dfdx(x) = 2x * cos(x^2 + 3) + 3x^2
dfAuto = [f(Dual(x, 1.0)).∂ for x in xplot] # AD derivative at every x
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"df/dx", title = "Derivative")
lines!(ax, xplot, dfAuto, color = :black, label = "Automatic Differentiation")
scatter!(ax, xplot, dfdx.(xplot), color = :blue, label = "Symbolic differentiation")
axislegend(ax, position = :rt)
fig
