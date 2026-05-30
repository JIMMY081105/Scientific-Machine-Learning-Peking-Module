# =====================================================================================
# 4.3 - Regression by gradient descent with Dual numbers
# =====================================================================================
# GOAL: fit a constant model a0 to noisy data by minimising S(a0) = 0.5*sum((yi - a0)^2).
# STRATEGY: get dS/da0 from the derivative field of a single Dual evaluation of the loss,
#           and step a0 downhill. (a0 is a mutable Dual so we can update a0.v in place.)
# =====================================================================================

using CairoMakie
using Distributions
using Random

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

# Noisy data scattered around y = 2.3.
N_SAMPLES = 10
rng = Xoshiro(1)
xi = rand(rng, Uniform(-1, 1), N_SAMPLES)
yi = rand(rng, Normal(2.3, 0.1), N_SAMPLES)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 25, ylabelsize = 25, limits = (-1, 1, 0, 5), backgroundcolor = :transparent)
scatter!(ax, xi, yi; label = "Data", color = :black, markersize = 15)
axislegend(ax; position = :rt)
# save("../../figures/04p03ApplicationOfForwardMode.svg", fig)   # (figure-save disabled in study file)
display(fig)

# Loss as a function of the single parameter a0.
S(a0) = 0.5 * sum((yi .- a0).^2)

fig = Figure()
a0vec = 0:0.5:5
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"S(a_0)", xlabelsize = 20, ylabelsize = 20, backgroundcolor = :transparent)
lines!(ax, a0vec, S.(a0vec), color = :black)
# save("../../figures/04p03ApplicationOfForwardMode02.svg", fig)   # (figure-save disabled in study file)
display(fig)

# dS/da0 straight from the derivative field of a Dual evaluation.
a0 = Dual(1.0, 1.0)
S(a0).∂

a0 = Dual(4.0, 1.0)
S(a0).∂

# Gradient descent, updating a0.v in place.
a0 = Dual(1.0, 1.0)
α = 0.01
for i = 1:500
    a0.v = a0.v - α * S(a0).∂
    println(a0.v)
end
