# =====================================================================================
# 4.1 - Parameter estimation by gradient descent (Zygote)
# =====================================================================================
# GOAL: fit the model f_model(a, x) = a1*(1 - exp(a2*x)) to noisy data.
# STRATEGY: define a squared-error loss, get dLoss/da from Zygote.gradient (a is a
#           2-vector), and run gradient descent on the parameters.
# =====================================================================================

using CairoMakie
using Random
using Distributions
using Zygote

# Generate noisy data from a1 = 4, a2 = -1.8.
N_SAMPLES = 50
rng = Xoshiro(1)
x_samples = rand(rng, Uniform(0, 5), N_SAMPLES)
y_noise = rand(rng, Normal(0.0, 0.1), N_SAMPLES)
y_samples = 4 * (1.0 .- exp.(-1.8 * x_samples)) + y_noise

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (0, 5, 0, 5))
scatter!(ax, x_samples, y_samples; label = "Data", color = :black)
axislegend(ax; position = :rt)
fig

# Model and loss (a[1] = a, a[2] = b).
fmodel(a, x) = a[1] * (1.0 .- exp.(a[2] * x))
function custom_loss(parameters, x_samples)
    ŷ = fmodel(parameters, x_samples)
    0.5 * sum((y_samples .- ŷ).^2)
end

a = [1.0, -1.0] # initial guess of a0, a1
S = (a) -> custom_loss(a, x_samples)
Zygote.gradient(S, a)[1]

# Gradient descent on the parameters.
α = 0.01 # learning rate
for i = 1:500
    a -= α * Zygote.gradient(S, a)[1]
    println(a)
end

# Fitted model over the data.
xplot = Array(0:0.01:5)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (0, 5, 0, 5), xlabelsize = 20, ylabelsize = 20)
scatter!(ax, x_samples, y_samples; label = "Data", color = :black)
lines!(ax, xplot, fmodel(a, xplot); label = "model", color = :blue)
axislegend(ax; position = :rb)
fig
