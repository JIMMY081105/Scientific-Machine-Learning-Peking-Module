using Random
using Zygote
#using Optimisers
using Optimization
using OptimizationOptimisers
using CairoMakie
using Distributions
using Statistics

N_SAMPLES = 1000 #number of samples
LEARNING_RATE = 0.1
N_EPOCHS = 1_000 #epochs is the number of times the algorithm "sees" all your data

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

y_samples = 2.0*x1_samples.*x2_samples+x2_samples.^2

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",limits=(0,2π,0,2π))
tricontourf!(ax,x1_samples[:],x2_samples[:],y_samples[:],colormap=:bwr)
figdisplay(fig)
  
fmodel(a,x)=(a[1].+a[2]*x[1,:]+a[3]*x[2,:]+a[4]*x[1,:].*x[2,:]+a[5]*x[1,:].^2+a[6]*x[2,:].^2)'

a=[1.0,1.0,1.0,1.0,1.0,1.0] #by memory from memory
input_matrix=vcat(x1_samples[:]',x2_samples[:]') 
y_prediction=fmodel(a,input_matrix)

# The forward function
function loss_fn(a,input_matrix)
    y_prediction = fmodel(a,input_matrix)
    loss = 0.5 * mean((y_prediction .- y_samples).^2)
    return loss
end

function my_callback(state, l)
    push!(loss_history, l)
    return false # continues the solver
end

optf = OptimizationFunction(loss_fn, ADTypes.AutoZygote())
prob = OptimizationProblem(optf,a,input_matrix)
loss_history = []
sol = solve(prob, Optimisers.Adam(0.001,(0.9,0.99));maxiters=20000,callback=my_callback);
sol.u