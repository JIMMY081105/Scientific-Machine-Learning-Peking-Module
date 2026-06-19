########################################################################################
# 5.4 Introduction to Inverse Problems for BVPs
########################################################################################
#
# PROBLEM
# The forward BVP k T'' = -1 is known, but the material constant k is unknown. Given
# a measurement T(1)=3, find the k that makes the finite-difference solution
# reproduce it - an inverse problem.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Build the same tridiagonal finite-difference system as in 5.1 for a trial k
#      and solve for T; read off T at the last point.
#   2. Sweep k over a range, re-solving each time, and plot T(x=1) versus k (and
#      (T(1)-3)^2 versus k) to see roughly where the target is met (k about 0.22).
#   3. Define loss(p) = (T_end(k=p) - 3)^2, which must solve the linear system every
#      evaluation.
#   4. Minimise the loss for k with the Optimization.jl library (Adam) using
#      AutoForwardDiff to differentiate through the solve; a callback logs the loss.
#   5. Sanity-check by evaluating ForwardDiff.derivative of the loss near the
#      solution.
#
# This file is notebook "05p04ParameterEstimationBVP01" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ====================================================================
# 5.4 Introduction to Inverse Problem for Boundary Value Problem
# ====================================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using BenchmarkTools
# In this example, we are going to be looking at the problem below again but from a
# slightly different perspective.

Delta=0.1
x=0:Delta:1
α=5.0
β=3.0
ThermalConductivity=0.1

n=length(x)
d=zeros(n)
dl=zeros(n-1)
du=zeros(n-1)
C=zeros(n)

du[1]=0.0
d[1]=1.0
C[1]=5.0
for i=2:n-1
    dl[i-1]=(1.0/Delta^2)
    d[i]=-(2.0/Delta^2)
    du[i]=(1.0/Delta^2)
    C[i]=1.0/ThermalConductivity
end
dl[n-1]=-1
d[n]=1.0
C[n]=0.0

# Construct the specialized structured matrix
A = Tridiagonal(dl, d, du)

A_sparse = sparse(A)

prob=LinearProblem(A,C)
@time T=LinearSolve.solve(prob)
# Note that tridag solver is faster since it is specific for tridiagonal systems.

xplot=1:0.01:2
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
lines!(ax,x,T,color=:blue,label="Finite difference solution")
axislegend(ax,position=:rt)
fig

T[end]

ThermalCondVect=0.1:0.02:1.0

EndTemperature=similar(ThermalCondVect)

for i=1:length(ThermalCondVect)
    ThermalConductivity=ThermalCondVect[i]
    C[2:n-1].=1.0/ThermalConductivity
    prob=LinearProblem(A,C)
    #prob=LinearProblem(A_sparse,C)
    #y=LinearSolve.solve(prob,KrylovJL_GMRES())
    y=LinearSolve.solve(prob)
    EndTemperature[i]=y[end]
end

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"k", ylabel = L"T(x=1)")
lines!(ax1,ThermalCondVect, EndTemperature)

ax2 = Axis(fig[1, 2], xlabel = L"k", ylabel = L"(T(x=1)-3)^2")
lines!(ax2,ThermalCondVect, (EndTemperature.-3).^2)
# save("../figures/05p04ParameterEstimationBVP0101.svg",fig)   # (figure-save disabled in study file)
fig

function loss(p,ydesired)

    C=ones(n)./p[1]
    C[1]=5.0
    C[n]=0.0
    prob=LinearProblem(A,C)
    y=LinearSolve.solve(prob)

    return sum((y[end]-ydesired[1]).^2)
end

function callback(state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

using Optimization
using Optimisers
using OptimizationOptimisers
using ForwardDiff

losses = Float64[]
adtype = AutoForwardDiff()
optf = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optf,[0.1],[3])
@time sol=Optimization.solve(optprob,Adam(0.01f0),maxiters=5000, callback=callback)

# Evaluate the derivative at k=0.22....
df = ForwardDiff.derivative((p)->loss(p,[3.0]),0.225)
