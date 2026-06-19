########################################################################################
# 5.5 Another Inverse Problem for a BVP
########################################################################################
#
# PROBLEM
# Given measured beam-deflection data (x_i, y_i), find the unknown load distribution
# w(x) = a0 + a1*x in the beam equation y'' + 4y = w(x) so the model's deflection
# best fits the data. The model is defined implicitly through solving the BVP.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Read the (x, y) data from Data01.csv and plot it.
#   2. Discretise y'' + 4y = w(x) into a tridiagonal system A y = C, where C holds
#      the load model a0 + a1*x at each interior point and 0 at the boundaries.
#   3. Define loss(p, ydata): build C from p = [a0, a1], solve A y = C, and return
#      the squared error between the solved deflection and the data - so each
#      evaluation solves the BVP.
#   4. Minimise the loss over (a0, a1) with Optimization.jl (Adam, AutoForwardDiff),
#      logging losses via a callback.
#   5. Plot the fitted deflection over the data and the recovered load w(x) from the
#      optimal coefficients.
#
# This file is notebook "05p05ParameterEstimationBVP02" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ============================================================
# 5.5 Another Inverse Problem for Boundary Value Problem
# ============================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using BenchmarkTools
using CSV
using DataFrames
# In this notebook, you are given that some data is taken from a beam deflection

df = CSV.read("../Data01.csv", DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
scatter!(ax,xdata,ydata,color=:black,label="data")
axislegend(ax,position=:ct)
# save("../figures/05p05ParameterEstimationBVP0201.svg",fig)   # (figure-save disabled in study file)
fig
# the governing equation for beam deflection becomes
# For a domain of 5 intervals, our matrix equations will look like

Delta=xdata[2]-xdata[1]
n=length(xdata)
d=zeros(n)
dl=zeros(n-1)
du=zeros(n-1)
C=zeros(n)
du[1]=0.0
d[1]=1.0
C[1]=0.0
for i=2:n-1
    dl[i-1]=(1.0/Delta^2)
    d[i]=-(2.0/Delta^2)+4.0
    du[i]=(1.0/Delta^2)
end
dl[n-1]=0.0
d[n]=1.0
C[n]=0.0

# Construct the specialized structured matrix
A = Tridiagonal(dl, d, du)

function loss(p,ydata)
    
    C=(p[1].+p[2]*xdata) #this is the model for  w(x)
    C[1]=0.0
    C[n]=0.0
    prob=LinearProblem(A,C)
    fmodel=LinearSolve.solve(prob)

    return sum(0.5*(fmodel.-ydata).^2)
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
optprob = OptimizationProblem(optf,[1.0,1.0],ydata)
sol=Optimization.solve(optprob,Adam(0.001f0),maxiters=50000, callback=callback)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"iter", ylabel = L"losses")
lines!(ax,log10.(losses),color=:black,label="losses")
axislegend(ax,position=:lt)
fig

sol.u

    C=(sol.u[1].+sol.u[2]*xdata)
    C[1]=0.0
    C[n]=0.0
    prob=LinearProblem(A,C)
    ymodel=LinearSolve.solve(prob)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)")
scatter!(ax,xdata,ydata,color=:black,label="data")
lines!(ax,xdata,ymodel,color=:black,label="Model")
axislegend(ax,position=:lt)
fig

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"w(x)")
scatter!(ax,xdata[2:end-1],C[2:end-1],color=:black,label="Predicted Loading")
axislegend(ax,position=:lt)
fig

sol.u
