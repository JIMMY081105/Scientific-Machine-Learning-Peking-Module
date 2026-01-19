using CairoMakie
using Zygote

f(x1, x2) = exp(−x[1]^2*cos(3*x^2))
