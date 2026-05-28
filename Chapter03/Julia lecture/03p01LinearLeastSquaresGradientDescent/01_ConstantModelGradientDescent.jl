# =====================================================================================
# 3.1 - Constant model, fitted by hand-coded gradient descent
# =====================================================================================
# GOAL: fit the constant model f = a0 to noisy data by minimising S(a0) = 0.5*sum((yi - a0)^2).
# STRATEGY: use the ANALYTIC derivative dS/da0 and hand-code gradient descent
#           a0 <- a0 - alpha*dS/da0. The recovered a0 lands near the data mean (~2.3).
# =====================================================================================

using Random
using Distributions
using CairoMakie

# Fictitious data scattered around y = 2.3.
N_SAMPLES = 5
rng = Xoshiro(1)
xi = rand(rng, Uniform(-1, 1), N_SAMPLES)
yi = rand(rng, Normal(2.3, 0.1), N_SAMPLES)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 25, ylabelsize = 25, limits = (-1, 1, 1, 5), backgroundcolor = :transparent)
scatter!(ax, xi, yi; label = "Data", color = :black, markersize = 15)
axislegend(ax; position = :rt)
# save("../../figures/03p01LinearLeastSquaresGradientDescent01.png", fig)   # (figure-save disabled in study file)
display(fig)

# Constant model and its loss.
f_model(a0, xi) = a0
S(a0) = (1 / 2) * sum((yi .- f_model.(a0, xi)).^2)

avec = 0:0.5:5
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"S(a_0)")
lines!(ax, avec, S.(avec); label = "Data", color = :black)
axislegend(ax; position = :rt)
fig

# Analytic derivative dS/da0.
dSda(a0) = -sum(yi .- f_model.(a0, xi))

# Hand-coded gradient descent.
a0 = 3.0
α = 0.001
for i = 1:1000
    a0 -= α * dSda(a0)
end
a0

# The fitted constant over the data.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", xlabelsize = 25, ylabelsize = 25, limits = (-1, 1, 1, 5), backgroundcolor = :transparent)
scatter!(ax, xi, yi; label = "Data", color = :black, markersize = 15)
lines!(ax, xi, f_model.(a0, xi))
axislegend(ax; position = :rt)
display(fig)
