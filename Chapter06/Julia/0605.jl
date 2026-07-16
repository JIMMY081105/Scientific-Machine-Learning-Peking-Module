using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using BenchmarkTools
using ComponentArrays
using CSV
using DataFrames
using Lux
using Random
using Distributions

using Optimization
using Optimisers    
using OptimizationOptimisers
using ForwardDiff

df = CSV.read(joinpath(@__DIR__, "..", "Data07.csv"), DataFrame)

tdata=df[:, :tdata]
xdata=df[:, :xdata]
xzero=df[:, :xzero]

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)",limits=(0, 1, -0.5, 1.5))
scatter!(ax,tdata,xdata,color=:black,label="Data")
#save("../figures/06p05NeuralNetworkScientificModels01.png",fig)
axislegend(ax,position=:lt)
fig

RModel=Chain(
    Dense(1 => 16, tanh),
    Dense(16 => 16, tanh),
    Dense(16 => 1)
)

rng = Xoshiro(1)
parameters, layer_states = Lux.setup(rng, model)

Delta=0.02
tdata=0:Delta:1

n=length(tdata)
d=zeros(n)
dl=zeros(n-1)
du=zeros(n-1)
C_true=zeros(n)
C_model=zeros(n)

rmodel,_= RModel(tdata', parameters, layer_states)
du[1]=0.0
d[1]=1.0
C_true[1]=0.0
for i=2:n-1
    dl[i-1]=(1.0/Delta^2)
    d[i]=-(2.0/Delta^2)+4.0
    du[i]=(1.0/Delta^2)
    C_true[i]=2*exp(-tdata[i]*tdata[i])
end
dl[n-1]=-1.0
d[n]=1.0
C_true[n]=0.0

# Construct the specialized structured matrix
A = Tridiagonal(dl, d, du)
prob=LinearProblem(A,C_true)
#@time xdata=LinearSolve.solve(prob)


C_model=rmodel[:]
C_model[1]=0.0 #boundary condition at t=0
C_model[n]=0.0 #boundary condition at t=1
prob=LinearProblem(A,C_model)
xpred=LinearSolve.solve(prob)

fig = Figure()

ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)",limits=(0, 1, -0.5, 1.5))
#ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)",backgroundcolor=:transparent)
scatter!(ax,tdata,xdata,color=:black,label="Data")
lines!(ax,tdata,xpred,color=:blue,label="Prediction")
#save("../figures/06p05NeuralNetworkScientificModels02.png",fig)
axislegend(ax,position=:lt)
fig

function loss(p,xdata)
    rmodel,_= RModel(tdata', p, layer_states)
    C_model=rmodel[:]
    C_model[1]=0.0 #boundary condition at t=0
    C_model[n]=0.0 #boundary condition at t=1
    prob=LinearProblem(A,C_model)
    x=LinearSolve.solve(prob)
    return sum((x.-xdata).^2)
end

loss(ComponentArray(parameters),Float32.(xdata))

function callback(state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

losses = Float64[]
adtype = AutoForwardDiff()
optf = OptimizationFunction(loss, adtype)

optprob = OptimizationProblem(optf,ComponentArray(parameters),Float32.(xdata))
sol=Optimization.solve(optprob,Adam(0.01f0),maxiters=20000, callback=callback)


fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = "iter", ylabel = "losses")
lines!(ax,log.(losses),color=:black)
fig

rmodel,_= RModel(tdata', sol.u, layer_states)
C_model=rmodel[:]
C_model[1]=1.0
C_model[n]=0.0
prob=LinearProblem(A,C_model)
xmodel=LinearSolve.solve(prob)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)",limits=(0, 1, -2.0, 1.5))
scatter!(ax,tdata,xdata,color=:black,label="data")
lines!(ax,tdata,xmodel.u,color=:blue,label="Model")
#save("../figures/06p05NeuralNetworkScientificModels03.png",fig)
axislegend(ax,position=:lt)
fig

df2 = CSV.read(joinpath(@__DIR__, "..", "Data08.csv"), DataFrame)

tdata2=df2[:, :tdata]
xdata2=df2[:, :xdata]
xzero2=df2[:, :xzero]

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"t", ylabel = L"x(t)",limits=(0, 1, -2.0, 1.5))
scatter!(ax,tdata2,xdata2,color=:black,label="Data")
lines!(ax,tdata,xmodel.u,color=:blue,label="Model")
#save("../figures/06p05NeuralNetworkScientificModels01.png",fig)
axislegend(ax,position=:lt)
fig