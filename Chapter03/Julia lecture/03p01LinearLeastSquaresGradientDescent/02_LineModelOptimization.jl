# =====================================================================================
# 3.1 - Line model, fitted with Optimization.jl
# =====================================================================================
# GOAL: fit the straight line f = a0 + a1*x to noisy data generated from y = 2x + 3.
# STRATEGY: view the loss S over the (a0, a1) plane as a contour (one minimum at (3, 2)),
#           then minimise it with the Optimization.jl library (Descent optimiser,
#           AutoZygote gradients) instead of hand-coding the update.
# =====================================================================================

using Random
using Distributions
using CairoMakie
using Zygote
using Optimization
using OptimizationOptimisers
using Optimisers
using ADTypes

# Fictitious line data: y = 2x + 3 + noise.
N_SAMPLES = 50
rng = Xoshiro(1)
x_samples = rand(rng, Uniform(-1, 1), N_SAMPLES)
y_noise = rand(rng, Normal(0.0, 0.1), N_SAMPLES)
y_samples = 2.0 .* (x_samples) .+ 3.0 .+ y_noise

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (-1, 1, 1, 5))
scatter!(ax, x_samples, y_samples; label = "Data", color = :black)
axislegend(ax; position = :rt)
fig

# Line model (a[1] + a[2]*x, since Julia is 1-indexed) and the untrained fit.
fmodel(a, x) = a[1] * x.^0 + a[2] .* x
a = [1.0, 1.0]
xplot = Array(-1:0.001:1)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (-1, 1, 0, 5))
scatter!(ax, x_samples, y_samples; label = "Data", color = :black)
lines!(ax, xplot, fmodel(a, xplot); label = "model", color = :blue)
axislegend(ax; position = :rt)
fig

# Loss as a function of the parameters.
function S(parameters, x_samples)
    ŷ = fmodel(parameters, x_samples)
    sum(0.5 * (y_samples .- ŷ).^2)
end
S(a, x_samples)

# Loss surface over (a0, a1): a single minimum at (3, 2).
a0range = 0:0.1:5
a1range = 0:0.1:5
loss_values = [S([a0, a1], x_samples) for a0 in a0range, a1 in a1range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"a_1", limits = (0, 5.0, 0.0, 5.0))
contourf!(ax, a0range, a1range, loss_values; levels = 50, colormap = :bwr)
contour!(ax, a0range, a1range, loss_values; levels = 50, labels = true, color = :black)
scatter!(ax, [3], [2]; markersize = 20, color = :black)
fig

# Minimise with Optimization.jl (Descent optimiser, AutoZygote gradients).
a0 = [1.0; 1.0]
optf = OptimizationFunction(S, ADTypes.AutoZygote())
prob = OptimizationProblem(optf, a0, x_samples)
sol = solve(prob, Optimisers.Descent(0.01), maxiters = 50_000)

# Fitted line over the data.
a = [sol.u[1], sol.u[2]]
xplot = Array(-1:0.001:1)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (-1, 1, 0, 5))
scatter!(ax, x_samples, y_samples; label = "Data", color = :black)
lines!(ax, xplot, fmodel(a, xplot); label = "model", color = :blue)
axislegend(ax; position = :rt)
fig

# TODO (Class Demo 01): Generate data from y = 2x^2 - 10x + 3 + noise on x in [0, 2*pi] and fit
#   f_model(x) = a0 + a1*x + a2*x^2 by writing your own gradient descent.
