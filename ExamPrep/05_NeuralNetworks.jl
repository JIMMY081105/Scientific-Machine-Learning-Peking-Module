# =====================================================================================
# NEURAL NETWORKS (Lux) — Level 3
# =====================================================================================
# The ONE training recipe (memorize this shape, everything else is variations):
#   1. data -> input matrix (features × samples) and target
#   2. model = Chain(Dense(...))  ;  parameters, layer_states = Lux.setup(rng, model)
#   3. loss_fn compares model output to TARGET y (never to an input!)
#   4. OptimizationFunction -> OptimizationProblem -> solve(Adam) with callback
#   5. predict: model(new_input, neural_network.u, layer_states)
# Sources: Chapter06, Homework/Day10-11, Day12, Day13-14, Assignment 2 Q2-Q3.
# =====================================================================================

using CSV
using DataFrames
using CairoMakie
using Statistics
using Random
using Lux
using Zygote
using Optimization
using OptimizationOptimisers
using ComponentArrays

# =====================================================================================
# 0) DATA — in the exam you'll read a CSV; template (adjust column names to the file!):
#    df = CSV.read(joinpath(@__DIR__, "Data.csv"), DataFrame)
#    xdata = df[:, :xdata];  ydata = df[:, :ydata]
# Here we generate data so this file runs standalone:
# =====================================================================================
rng = Xoshiro(1)
N_SAMPLES = 60
xdata = sort(5 .* rand(rng, N_SAMPLES))
ydata = 2 .* cos.(pi .* xdata) .* exp.(-xdata .^ 2 ./ 5) .+ 0.1 .* randn(rng, N_SAMPLES)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax, xdata, ydata; color = :black, label = "Data")
axislegend(ax; position = :rt)
display(fig)

# =====================================================================================
# 1) ONE-INPUT NETWORK, full recipe
# =====================================================================================
model = Chain(
    Dense(1 => 16, tanh),
    Dense(16 => 16, tanh),
    Dense(16 => 1)
)
parameters, layer_states = Lux.setup(rng, model)

t_input = xdata'                      # 1×N: input must be features × samples

function loss_fn(p, ls)
    y_prediction, new_ls = model(t_input, p, ls)
    loss = 0.5 * mean((y_prediction[:] .- ydata) .^ 2)    # <- compare to ydata!
    return loss
end

callback = function (p, l)
    if length(loss_history) % 100 == 0
        println("Iteration: $(p.iter), Loss: $l")
    end
    push!(loss_history, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_fn, adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters), layer_states)

loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 5000)

# ALWAYS look at the loss curve first (loss down but plot wrong = plotting bug):
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss", yscale = log10)
lines!(ax, loss_history; color = :blue)
display(fig)

# Predict on a smooth grid and overlay on the data:
xplot = collect(0:0.01:5)
y_prediction, _ = model(xplot', neural_network.u, layer_states)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax, xdata, ydata; color = :black, label = "Data")
lines!(ax, xplot, y_prediction[:]; color = :blue, label = "NN prediction")
axislegend(ax; position = :rt)
display(fig)

# =====================================================================================
# 2) POLYNOMIAL COMPARISON (Day10-11): normal equations on the SAME data — one line
# =====================================================================================
degree = 8
V = hcat([xdata .^ k for k in 0:degree]...)
a = V'V \ (V'ydata)
Vplot = hcat([xplot .^ k for k in 0:degree]...)
# plot with: lines!(ax, xplot, Vplot * a)

# =====================================================================================
# 3) TRAIN/VALIDATION SPLIT (06p03): interpolation vs extrapolation
# =====================================================================================
# Interpolation (random 50%):
training_indices = shuffle(rng, eachindex(xdata))[1:round(Int, 0.5 * N_SAMPLES)]
# Extrapolation (train only below a threshold):
# training_indices = findall(xdata .< 2.0)
x_train = xdata[training_indices]
y_train = ydata[training_indices]
validation_indices = setdiff(eachindex(xdata), training_indices)
# Train on x_train'/y_train with the same recipe, then validation MSE:
#   pred_val, _ = model(xdata[validation_indices]', neural_network.u, layer_states)
#   mse = mean((pred_val[:] .- ydata[validation_indices]).^2)

# =====================================================================================
# 4) TWO-INPUT NETWORK + MULTIPLE DATASETS (Day13-14 / Day12 — most likely exam form!)
# =====================================================================================
# Each dataset has columns xzero (initial condition), tdata, xdata.
# Input = 2×N matrix [xzero'; tdata'], first Dense layer must be Dense(2 => ...).
#
#   t_input = [axzero'; atdata']                          # ONE dataset
#
# COMBINING datasets: hcat blocks side by side with SPACES (not commas!):
#   abxzero = [axzero' bxzero']
#   abtdata = [atdata' btdata']            # <- time row is tdata, NOT xdata!
#   abxdata = [axdata' bxdata']            # <- target (1×2N), used ONLY in the loss
#   t_input = [abxzero ; abtdata]
#   loss: 0.5 * mean((x_prediction .- abxdata).^2)
#
# PREDICTING for a new initial condition (e.g. x(0)=0.5 stored in dataset d):
#   t_input = [dxzero'; dtdata']           # after training! reuse trained weights:
#   x_prediction, _ = model(t_input, neural_network.u, layer_states)
#   ...then plot against dtdata (the input you predicted at), not the training time.
#
# Full corrected worked example: ../Homework/Day13-14/SolutionHomework1314.jl

# =====================================================================================
# 5) 2-D SCATTER DATA -> use tricontourf!, NOT lines! (Day12 solutionday12.jl)
# =====================================================================================
#   fig = Figure()
#   ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2")
#   tricontourf!(ax, x1data, x2data, y_prediction[:]; colormap = :bwr)
#   scatter!(ax, x1data, x2data; color = ydata, colormap = :bwr, strokewidth = 1, strokecolor = :black)

# =====================================================================================
# 6) NN + PHYSICS SOLVE (Assignment 2 Q3 / 0605.jl) — the network output feeds a
#    Tridiagonal BVP solve, and the LOSS compares the SOLVED result to data:
# =====================================================================================
#   function loss_nn(p, Tdata)
#       qmodel, _ = QModel(xgrid, p, layer_states)     # NN evaluates q(x) on the grid
#       C = -qmodel[:] ./ k                            # NN output -> right-hand side
#       C[1] = 5.0;  C[n] = 0.0                        # boundary rows
#       Tpred = LinearSolve.solve(LinearProblem(A, C)) # solve the physics
#       return sum((Tpred .- Tdata).^2)                # compare SOLVED result to data
#   end
#   Use adtype = Optimization.AutoForwardDiff() here (differentiating THROUGH the solve).

# =====================================================================================
# CHECKLIST before you submit (L-I-S-S-C):
#   L — Loss compares prediction to the TARGET y (never an input x)
#   I — plot matches Input dimension (2 inputs -> tricontourf!/coloured scatter, not lines!)
#   S — lines! needs Sorted, single-valued x (else scatter! or a sorted xplot grid)
#   S — Shapes: input features×samples (use '), output 1×N (use [:] to flatten)
#   C — Check the loss curve first; also check the dfX in each extraction line matches
#       the CSV.read right above it (copy-paste bug from Day13-14!)
# =====================================================================================
