using Lux
using Random
using Zygote
using Optimisers
using CairoMakie
using Distributions
using Statistics
using Printf
using Optimization
using OptimizationOptimisers
using ComponentArrays

N_SAMPLES = 1_000

# Our Pseudo-Random Number Generator
rng = Xoshiro(42)

x1_samples = rand(
    rng,
    Uniform(0.0, 2 * π),
    (1,N_SAMPLES),
)

x2_samples = rand(
    rng,
    Uniform(0.0, 2 * π),
    (1,N_SAMPLES),
)


y_noise = rand(
    rng,
    Normal(0.0, 0.01),
    (1,N_SAMPLES),
)

y_samples = Float32.(sin.(x1_samples.*x2_samples) .+ 0.01*y_noise)

V=hcat(ones(N_SAMPLES),x1_samples',x2_samples',x1_samples[:].*x2_samples[:],x1_samples[:].^2,x2_samples[:].^2)
a=V'V\(V'*y_samples')

ypred=a[1].+a[2]*x1_samples[:].+a[3]*x2_samples[:].+a[4]*x1_samples[:].*x2_samples[:].+a[5]*x1_samples[:].^2+a[6]*x2_samples[:].^2

fig = Figure(backgroundcolor=:transparent)
ax1 = CairoMakie.Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",limits=(0,2π,0,2π),xlabelsize = 20,ylabelsize = 20,backgroundcolor=:transparent)
ax2 = CairoMakie.Axis(fig[1, 2], xlabel = L"x_1", ylabel = L"x_2",limits=(0,2π,0,2π),xlabelsize = 20,ylabelsize = 20,backgroundcolor=:transparent)
tricontourf!(ax1,x1_samples[:],x2_samples[:],y_samples[:],colormap=:bwr)
tricontourf!(ax2,x1_samples[:],x2_samples[:],ypred[:],colormap=:bwr)
display(fig)

x_samples=[Float32.(x1_samples);Float32.(x2_samples)]

model = Chain(
    Dense(2 => 10),
    Dense(10 => 10,tanh),
    Dense(10 => 10,tanh),
    Dense(10 => 10,tanh),
    Dense(10 => 1)
)
parameters, layer_states = Lux.setup(rng, model)

function loss_fn(p, ls)
    y_prediction, _= model(x_samples, p, ls)
    loss = 0.5 * mean((y_prediction .- y_samples).^2)
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
optprob = Optimization.OptimizationProblem(optf, ComponentArray(parameters),layer_states)
#optprob = Optimization.OptimizationProblem(optf, ComponentArray(neural_network.u),layer_states)

# Initialize a vector to store loss values
loss_history = Float64[]
neural_network = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback, maxiters = 10000)

# predict using Neural Network
y_NN,_=model(x_samples,neural_network.u, layer_states) #note that model expects the input as a row vector
#y_NN,_=model(x_samples,parameters, layer_states) #note that model expects the input as a row vector


fig = Figure(backgroundcolor=:transparent)
ax1 = CairoMakie.Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",limits=(0,2π,0,2π),xlabelsize = 20,ylabelsize = 20,backgroundcolor=:transparent)
ax2 = CairoMakie.Axis(fig[1, 2], xlabel = L"x_1", ylabel = L"x_2",limits=(0,2π,0,2π),xlabelsize = 20,ylabelsize = 20,backgroundcolor=:transparent)
tricontourf!(ax1,x_samples[1,:],x_samples[2,:],y_samples[:],colormap=:bwr)
tricontourf!(ax2,x_samples[1,:],x_samples[2,:],y_NN[:],colormap=:bwr)
display(fig)