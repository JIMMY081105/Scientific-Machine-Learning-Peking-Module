# 3.2 Linear Least Squares Normal Equations
# Extracted from 03p02LinearLeastSquaresNormalEquations.ipynb.

using Random
using CairoMakie
using Distributions

# -----------------------------------------------------------------------------
# Section 1: Generate the fictitious data
# y = 0.5 + x + 3x^2 + epsilon
# -----------------------------------------------------------------------------

N_SAMPLES = 50
rng = Xoshiro(1)

x_samples = rand(rng, Uniform(-1, 1), N_SAMPLES)
y_noise = rand(rng, Normal(0.0, 0.1), N_SAMPLES)
y_samples = 3 .* x_samples .^ 2 .+ x_samples .+ 0.5 .+ y_noise

fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", limits=(-1, 1, 0, 5))
scatter!(ax, x_samples, y_samples; label="Data", color=:black, markersize=15)
axislegend(ax; position=:lt)
display(fig)

# -----------------------------------------------------------------------------
# Section 2: Plot an initial guess
# a[1], a[2], and a[3] represent a0, a1, and a2, respectively.
# -----------------------------------------------------------------------------

a = [1.0, 1.0, 1.0]
xplot = -1:0.01:1
yplot = a[1] .+ a[2] .* xplot .+ a[3] .* xplot .^ 2

fig = Figure()
ax = Axis(
    fig[1, 1];
    xlabel=L"x",
    ylabel=L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1, 1, 0, 5),
)
scatter!(ax, x_samples, y_samples; label="Data", color=:black, markersize=15)
lines!(ax, xplot, yplot; label="Model", color=:blue)
axislegend(ax; position=:lt)

# The notebook used this relative path. Uncomment it if the directory exists.
# save("../figures/03p01LinearLeastSquaresRegression01.png", fig)
display(fig)

# -----------------------------------------------------------------------------
# Section 3: Original notebook method -- normal equations
# (V'V)a = V'y
# -----------------------------------------------------------------------------

V = hcat(ones(N_SAMPLES), x_samples, x_samples .^ 2)
a = (V' * V) \ (V' * y_samples)
println("Normal-equation coefficients [a0, a1, a2] = ", a)

fmodel(a, x) = a[1] .* x .^ 0 .+ a[2] .* x .+ a[3] .* x .^ 2

fig = Figure()
xplot = -1:0.01:1
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", limits=(-1, 1, 0, 5))
scatter!(ax, x_samples, y_samples; label="Data", color=:black)
lines!(ax, xplot, fmodel(a, xplot); label="Model", color=:blue)
axislegend(ax; position=:lt)
display(fig)

# -----------------------------------------------------------------------------
# Alternative A: Solve the least-squares problem directly (recommended)
# This avoids explicitly forming V'V and is generally more numerically stable.
# -----------------------------------------------------------------------------

a_direct = V \ y_samples
println("Direct least-squares coefficients = ", a_direct)

# -----------------------------------------------------------------------------
# Alternative B: Explicit QR factorization
# -----------------------------------------------------------------------------

a_qr = qr(V) \ y_samples
println("QR coefficients = ", a_qr)

# -----------------------------------------------------------------------------
# Optional Class Demo 01
# Fit y = 2x^2 - 10x + 3 + epsilon on x in [0, 2pi].
# -----------------------------------------------------------------------------

function class_demo_01(; n_samples=50, seed=1, noise_std=0.1)
    demo_rng = Xoshiro(seed)
    x = rand(demo_rng, Uniform(0, 2pi), n_samples)
    noise = rand(demo_rng, Normal(0.0, noise_std), n_samples)
    y = 2 .* x .^ 2 .- 10 .* x .+ 3 .+ noise
    design = hcat(ones(n_samples), x, x .^ 2)
    coefficients = (design' * design) \ (design' * y)

    x_plot = range(0, 2pi; length=500)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
    scatter!(ax, x, y; label="Data", color=:black)
    lines!(ax, x_plot, fmodel(coefficients, x_plot); label="Model", color=:blue)
    axislegend(ax)
    display(fig)

    println("Class Demo 01 coefficients [a0, a1, a2] = ", coefficients)
    return coefficients
end

# Run with:
# class_demo_01()

# -----------------------------------------------------------------------------
# Optional Class Demo 02
# Read Data.csv and fit a third-order polynomial.
# Requires CSV.jl and DataFrames.jl. The first two columns are treated as x and y.
# -----------------------------------------------------------------------------

function class_demo_02(csv_path="Data.csv")
    # Load optional packages only when this demo is run.
    @eval using CSV
    @eval using DataFrames

    data = CSV.read(csv_path, DataFrame)
    x = Float64.(data[:, 1])
    y = Float64.(data[:, 2])
    design = hcat(ones(length(x)), x, x .^ 2, x .^ 3)
    coefficients = design \ y

    x_plot = range(minimum(x), maximum(x); length=500)
    y_plot = (coefficients[1] .+ coefficients[2] .* x_plot .+
              coefficients[3] .* x_plot .^ 2 .+ coefficients[4] .* x_plot .^ 3)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
    scatter!(ax, x, y; label="Data", color=:black)
    lines!(ax, x_plot, y_plot; label="Cubic model", color=:blue)
    axislegend(ax)
    display(fig)

    println("Class Demo 02 coefficients [a0, a1, a2, a3] = ", coefficients)
    return coefficients
end

# Run with:
# class_demo_02(joinpath(@__DIR__, "Data.csv"))

