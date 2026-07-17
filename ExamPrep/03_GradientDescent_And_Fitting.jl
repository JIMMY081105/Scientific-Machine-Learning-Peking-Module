# =====================================================================================
# GRADIENT DESCENT + CURVE FITTING — organized by exam level
# =====================================================================================
# GD:        x <- x - α*∇f(x)
# Momentum:  v <- β*v - α*∇f(x);        x <- x + v
# Nesterov:  v <- β*v - α*∇f(x + β*v);  x <- x + v
# Newton:    x <- x - H\g   (uses the Hessian; can converge to a MAXIMUM — check!)
# Fitting = minimising a loss S(a) = squared error between model(a,x) and data.
# Sources: Chapter02, Chapter03, Homework/Day03, Homework/Day04-05, Assignment 1.
# =====================================================================================

# =====================================================================================
# LEVEL 1 — hand-coded loops with analytic gradients
# =====================================================================================

# ---- 1a) 1-D gradient descent ----
f(x) = 2 * π * x^2 + 2 / x
d(x) = 4 * π * x - 2 / x^2      # hand derivative

x = 1.8
α = 0.02
for i = 1:5000
    global x -= α * d(x)
end
println("1-D GD minimum: ", x)

# ---- 1b) Momentum (remembers past gradients; escapes flat regions faster) ----
x = 1.8
β = 0.8
v = 0.0
for i = 1:5000
    global v = β * v - α * d(x)
    global x += v
end
println("Momentum minimum: ", x)

# ---- 1c) Nesterov (gradient at the look-ahead point; less overshoot) ----
x = 1.8
v = 0.0
for i = 1:5000
    global v = β * v - α * d(x + β * v)
    global x += v
end
println("Nesterov minimum: ", x)

# ---- 1d) 2-D gradient descent (gradient is a vector now) ----
f2(x) = -x[1] * x[2] * exp(-(x[1]^2 + x[2]^2) / 2)
function Gradient_f2(x)
    d1 = x[2] * (x[1]^2 - 1) * exp(-(x[1]^2 + x[2]^2) / 2)
    d2 = x[1] * (x[2]^2 - 1) * exp(-(x[1]^2 + x[2]^2) / 2)
    return [d1, d2]
end

x = [0.5, 0.5]         # WHICH minimum you find depends on the initial guess —
α2 = 0.1               # if asked, run from several starts and compare (Assignment 1 Q2)
for i = 1:100
    global x -= α2 * Gradient_f2(x)
end
println("2-D GD minimum: ", x)

# ---- 1e) Newton's method for OPTIMIZATION (1-D: x <- x - g/H, N-D: x <- x - H\g) ----
# Fast (~3 iterations) but seeks ANY zero-gradient point — may land on a maximum.
h(x) = (x^3 - x)^2
gh(x) = 2 * (x^3 - x) * (3 * x^2 - 1)
Hh(x) = 2 * (3 * x^2 - 1)^2 + 2 * (x^3 - x) * 6 * x

x = 2.0
for i = 1:10
    global x -= gh(x) / Hh(x)
end
println("Newton optimization: ", x, "  (check f'' > 0 to confirm it is a minimum)")

# =====================================================================================
# LEVEL 1 FITTING — no iteration needed when the model is LINEAR in coefficients!
# =====================================================================================
# Normal equations: build Vandermonde V, solve once. This is Level 1 legal (`\` only).
xdata = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]        # replace with question data
ydata = [0.6, 1.2, 2.1, 2.9, 4.4, 5.9, 8.1]

degree = 2
N = length(xdata)
V = hcat([xdata .^ k for k in 0:degree]...)          # columns: 1, x, x^2, ... x^degree
a = V'V \ (V'ydata)
println("polynomial coefficients: ", a)
# For a non-polynomial linear basis (e.g. 1, sin(x), exp(-x)): V = hcat(ones(N), sin.(xdata), exp.(-xdata))

# ---- Fitting a NONLINEAR model at Level 1: hand GD on the loss (Assignment 1 Q3) ----
# model T(x) = 10 + 30*exp(-k*x); unknown k; dS/dk written by hand via chain rule
model_exp(k, x) = 10 + 30 * exp(-k * x)
k = 1.0
αm = 0.01
for i in 1:20_000
    y_model = model_exp.(k, xdata)
    dS_dk = sum((y_model .- ydata) .* (-30 .* xdata .* exp.(-k .* xdata))) / N
    global k -= αm * dS_dk
end
println("fitted k = ", k)

# =====================================================================================
# LEVEL 2 — same loops, gradients from ForwardDiff / Zygote
# =====================================================================================
using ForwardDiff
using Zygote

himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

x = [0.0, 0.0]
for i = 1:100
    global x -= 0.01 * ForwardDiff.gradient(himmelblau, x)   # no hand gradient needed
end
println("Level 2 GD on Himmelblau: ", x)

# Zygote version of a fitting loss gradient (loss must take the coefficient vector):
S(a) = sum((a[1] .+ a[2] .* xdata .+ a[3] .* xdata .^ 2 .- ydata) .^ 2) / N
a = zeros(3)
for i = 1:20_000
    global a -= 0.001 * Zygote.gradient(S, a)[1]
end
println("Level 2 Zygote fit: ", a)

# =====================================================================================
# LEVEL 3 — Optimization.jl does the loop for you
# =====================================================================================
using Optimization, OptimizationOptimisers

# The three-step recipe. Objective MUST be (x, p) with x a VECTOR (even in 1-D).
optf = OptimizationFunction((x, p) -> himmelblau(x), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, [0.0, 0.0])
sol = solve(prob, Optimisers.Descent(0.01); maxiters = 1000)
println("Optimization.jl: ", sol.u, "  objective: ", sol.objective)

# Swap the optimiser without changing anything else:
sol_mom = solve(prob, Optimisers.Momentum(0.01, 0.9); maxiters = 1000)
sol_adam = solve(prob, Optimisers.Adam(0.05); maxiters = 1000)
println("Momentum: ", sol_mom.u, "   Adam: ", sol_adam.u)

# Fitting with Optimization.jl (Homework Day04-05 pattern): loss S(a,p), same recipe
Sfit(a, p) = sum((a[1] .+ a[2] .* xdata .+ a[3] .* xdata .^ 2 .- ydata) .^ 2) / N
optf2 = OptimizationFunction(Sfit, ADTypes.AutoZygote())
prob2 = OptimizationProblem(optf2, zeros(3))
sol2 = solve(prob2, Optimisers.Adam(0.05); maxiters = 20_000)
println("Level 3 fit: ", sol2.u)

# Recording the convergence path (for "plot the iterations" questions):
losses = Float64[]
callback = function (state, l)
    push!(losses, l)
    return false           # false = keep going
end
sol3 = solve(prob2, Optimisers.Adam(0.05); maxiters = 5000, callback)
println("final loss: ", losses[end])   # plot `losses` with lines!(ax, losses)

# Second-order methods (Newton/BFGS), if asked: using OptimizationOptimJL, then
# solve(prob, Optim.Newton()) / Optim.BFGS() / Optim.LBFGS() — see Chapter02 02p02.
