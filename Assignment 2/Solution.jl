using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using CSV
using DataFrames
using Lux
using Random
using ComponentArrays
using Statistics

using Optimization
using Optimisers
using OptimizationOptimisers
using ForwardDiff
using Zygote

# Question 1
# d^2y/dx^2 + p*y(x) = cos(x), x in [0,2], dy/dx(0)=0, y(2)=5

# b)

function solve_q1(p; n = 101)
    Delta = 2.0 / (n - 1)
    x = 0:Delta:2
    T = typeof(p)

    d = zeros(T, n)
    dl = zeros(T, n - 1)
    du = zeros(T, n - 1)
    C = zeros(T, n)

    d[1] = -1.0
    du[1] = 1.0
    C[1] = 0.0
    for i = 2:n-1
        dl[i-1] = 1.0 / Delta^2
        d[i] = -2.0 / Delta^2 + p
        du[i] = 1.0 / Delta^2
        C[i] = cos(x[i])
    end
    dl[n-1] = 0.0
    d[n] = 1.0
    C[n] = 5.0

    A = Tridiagonal(dl, d, du)
    prob = LinearProblem(A, C)
    y = LinearSolve.solve(prob)
    return x, y
end

p = 4.0
x_q1, y_q1 = solve_q1(p)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)", title = "Question 1b: p = 4")
lines!(ax, x_q1, y_q1, color = :blue, label = "Finite difference solution")
axislegend(ax, position = :rt)
fig

println("Question 1b: y(x=0) = $(y_q1[1]) for p = 4")

# c) y(x=0) as a function of p has resonances near p~0.6 and p~5.8 (the
# system goes ill-conditioned there), so start below the first one.

y0(p) = solve_q1(p)[2][1]
loss_q1(p) = (y0(p) - 1.0)^2

p_gd = 0.0
alpha = 0.05
loss_history_q1 = Float64[]
for i in 1:300
    grad = ForwardDiff.derivative(loss_q1, p_gd)
    global p_gd -= alpha * grad
    push!(loss_history_q1, loss_q1(p_gd))
end

println("Question 1c: hand-coded gradient descent gives p = $p_gd, y(x=0) = $(y0(p_gd))")

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss", yscale = log10, title = "Question 1c: gradient descent convergence")
lines!(ax, loss_history_q1, color = :black)
fig

# d)

function loss_q1_opt(u, ytarget)
    return (y0(u[1]) - ytarget[1])^2
end

adtype = Optimization.AutoForwardDiff()
optf_q1 = Optimization.OptimizationFunction(loss_q1_opt, adtype)
optprob_q1 = Optimization.OptimizationProblem(optf_q1, [0.0], [1.0])
sol_q1 = Optimization.solve(optprob_q1, OptimizationOptimisers.Adam(0.05), maxiters = 2000)

println("Question 1d: Optimization.jl gives p = $(sol_q1.u[1]), y(x=0) = $(y0(sol_q1.u[1]))")

# Question 2
# T(x,k): Data01 (k=0.1), Data02 (k=0.5) are training; Data03 (k=0.3), Data04 (k=1.5) are testing.

dir_q2 = @__DIR__
df1 = CSV.read(joinpath(dir_q2, "Data01.csv"), DataFrame)
df2 = CSV.read(joinpath(dir_q2, "Data02.csv"), DataFrame)
df3 = CSV.read(joinpath(dir_q2, "Data03.csv"), DataFrame)
df4 = CSV.read(joinpath(dir_q2, "Data04.csv"), DataFrame)

xdata1, Tdata1, k1 = df1[:, :xdata], df1[:, :ydata], 0.1
xdata2, Tdata2, k2 = df2[:, :xdata], df2[:, :ydata], 0.5
xdata3, Tdata3, k3 = df3[:, :xdata], df3[:, :ydata], 0.3
xdata4, Tdata4, k4 = df4[:, :xdata], df4[:, :ydata], 1.5

# a)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)", title = "Question 2a: Data01 (k=0.1)")
scatter!(ax, xdata1, Tdata1, color = :black, label = "Data")
axislegend(ax, position = :rb)
fig

function make_input(xd, k)
    return Float32.([xd'; fill(k, 1, length(xd))])
end

function train_Txk_model(x_input, y_target; maxiters = 5000)
    rng = Xoshiro(1)
    model = Chain(
        Dense(2 => 10, tanh),
        Dense(10 => 10, tanh),
        Dense(10 => 1)
    )
    parameters, layer_states = Lux.setup(rng, model)

    function loss_fn(p, ls)
        y_prediction, _ = model(x_input, p, ls)
        loss = 0.5 * mean((y_prediction .- y_target).^2)
        return loss
    end

    loss_history = Float64[]
    callback = function (p, l)
        if length(loss_history) % 200 == 0
            println("Iteration: $(p.iter), Loss: $l")
        end
        push!(loss_history, l)
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(loss_fn, adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters), layer_states)
    neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = maxiters)

    return model, neural_network.u, layer_states, loss_history
end

function plot_Txk_predictions(model, ps, ls, title)
    fig = Figure(size = (900, 700))
    datasets = ((xdata1, Tdata1, k1, "k=0.1 (train)"), (xdata2, Tdata2, k2, "k=0.5 (train)"),
                (xdata3, Tdata3, k3, "k=0.3 (test)"), (xdata4, Tdata4, k4, "k=1.5 (test)"))
    for (idx, (xd, Td, k, name)) in enumerate(datasets)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = CairoMakie.Axis(fig[row, col], xlabel = L"x", ylabel = L"T(x)", title = name)
        pred, _ = model(make_input(xd, k), ps, ls)
        scatter!(ax, xd, Td, color = :black, label = "Data")
        lines!(ax, xd, pred[:], color = :blue, label = "NN prediction")
        axislegend(ax, position = :rb)
    end
    Label(fig[0, :], title, fontsize = 18)
    return fig
end

# b)
x_input_b = make_input(xdata1, k1)
y_target_b = Float32.(Tdata1')
model_b, ps_b, ls_b, losses_b = train_Txk_model(x_input_b, y_target_b)
fig = plot_Txk_predictions(model_b, ps_b, ls_b, "Question 2b: trained on k=0.1 only")
fig

# c)
x_input_c = [make_input(xdata1, k1) make_input(xdata2, k2)]
y_target_c = Float32.([Tdata1; Tdata2]')
model_c, ps_c, ls_c, losses_c = train_Txk_model(x_input_c, y_target_c)
fig = plot_Txk_predictions(model_c, ps_c, ls_c, "Question 2c: trained on k=0.1 and k=0.5")
fig

# Question 3
# k*d^2T/dx^2 = -q(x). Reuses Q2's data: Data01 (k=0.1) trains q(x);
# Data02/03/04 test whether it generalises to other conductivities.

k_train_q3 = k1
Delta_q3 = xdata1[2] - xdata1[1]
n_q3 = length(xdata1)

# Matrix A only depends on the grid (not on k, which only enters the RHS).
dl_q3 = zeros(n_q3 - 1)
d_q3 = zeros(n_q3)
du_q3 = zeros(n_q3 - 1)

du_q3[1] = 0.0
d_q3[1] = 1.0
for i = 2:n_q3-1
    dl_q3[i-1] = 1.0 / Delta_q3^2
    d_q3[i] = -2.0 / Delta_q3^2
    du_q3[i] = 1.0 / Delta_q3^2
end
dl_q3[n_q3-1] = -1.0
d_q3[n_q3] = 1.0

A_q3 = Tridiagonal(dl_q3, d_q3, du_q3)

# a)
rng_q3 = Xoshiro(1)
QModel = Chain(
    Dense(1 => 16, tanh),
    Dense(16 => 16, tanh),
    Dense(16 => 1)
)
parameters_q3, layer_states_q3 = Lux.setup(rng_q3, QModel)

xgrid_q3 = Float32.(collect(xdata1)')

function solve_T(qvalues, k, ydata_dirichlet)
    C = -qvalues ./ k
    C[1] = ydata_dirichlet
    C[n_q3] = 0.0
    prob = LinearProblem(A_q3, C)
    return LinearSolve.solve(prob)
end

function loss_q3(p, Tdata)
    qmodel, _ = QModel(xgrid_q3, p, layer_states_q3)
    Tpred = solve_T(qmodel[:], k_train_q3, 5.0)
    return sum((Tpred .- Tdata).^2)
end

loss_q3(ComponentArray(parameters_q3), Tdata1)

losses_q3 = Float64[]
callback_q3 = function (state, l)
    push!(losses_q3, l)
    if length(losses_q3) % 200 == 0
        println("Question 3 - iteration $(length(losses_q3)): loss = $(losses_q3[end])")
    end
    return false
end

adtype_q3 = Optimization.AutoForwardDiff()
optf_q3 = Optimization.OptimizationFunction(loss_q3, adtype_q3)
optprob_q3 = Optimization.OptimizationProblem(optf_q3, ComponentArray(parameters_q3), Tdata1)
sol_q3 = Optimization.solve(optprob_q3, OptimizationOptimisers.Adam(0.01), maxiters = 5000, callback = callback_q3)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss", yscale = log10, title = "Question 3a: training loss")
lines!(ax, losses_q3, color = :black)
fig

qmodel_trained, _ = QModel(xgrid_q3, sol_q3.u, layer_states_q3)

# b)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"q(x)", title = "Question 3b: Neural Network approximation of q(x)")
lines!(ax, xdata1[2:end-1], qmodel_trained[2:end-1], color = :red, label = "NN q(x)")
axislegend(ax, position = :rt)
fig

Tpred_train_q3 = solve_T(qmodel_trained[:], k_train_q3, 5.0)
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)", title = "Question 3: fit to k=0.1 training data")
scatter!(ax, xdata1, Tdata1, color = :black, label = "Data (k=0.1)")
lines!(ax, xdata1, Tpred_train_q3, color = :blue, label = "Model")
axislegend(ax, position = :rb)
fig

# c) fixed q(x); only C changes with k, A stays the same
fig = Figure(size = (900, 350))
datasets_q3 = ((xdata2, Tdata2, k2, "k=0.5"), (xdata3, Tdata3, k3, "k=0.3"), (xdata4, Tdata4, k4, "k=1.5"))
for (col, (xd, Td, k, name)) in enumerate(datasets_q3)
    Tpred = solve_T(qmodel_trained[:], k, 5.0)
    local ax = CairoMakie.Axis(fig[1, col], xlabel = L"x", ylabel = L"T(x)", title = name)
    scatter!(ax, xd, Td, color = :black, label = "Data")
    lines!(ax, xd, Tpred, color = :blue, label = "Model")
    axislegend(ax, position = :rb)
    println("Question 3c ($name): mean squared error = $(mean((Tpred .- Td).^2))")
end
fig
