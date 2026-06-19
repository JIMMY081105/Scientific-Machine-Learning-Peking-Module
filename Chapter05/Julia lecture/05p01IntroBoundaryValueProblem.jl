########################################################################################
# 5.1 Introduction to Boundary Value Problems
########################################################################################
#
# PROBLEM
# Find the temperature distribution T(x) along a heated plate governed by k T'' =
# -q(x) on [0,1] with T(0)=5 and T'(1)=0. The unknown is a function, obtained by
# solving the ODE numerically.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Discretise: put a grid on [0,1] and replace T'' with the central-difference
#      formula (T[i-1]-2T[i]+T[i+1])/Delta^2.
#   2. Write one algebraic equation per interior grid point, plus rows for the two
#      boundary conditions, giving a linear system A T = C.
#   3. Build A as a Tridiagonal matrix (and its sparse form) and the right-hand side
#      C.
#   4. Solve with LinearSolve; compare the tridiagonal solver against a general
#      sparse solver (tridiagonal is faster) via @time.
#   5. Plot T(x); then redo it for non-uniform heating q(x) = 1 + 2x by only changing
#      C, and overlay both temperature profiles.
#
# This file is notebook "05p01IntroBoundaryValueProblem" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ================================================
# 5.3 Introduction to Boundary Value Problem
# ================================================

using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using BenchmarkTools
# In this notebook, we are going to find the temperature distribution of a uniformly
# heated plate.
# Using the finite difference approximation
# We will now split our domain into 5 sub intervals and apply the equation above onto
# each of the interior points

Delta=0.2
x=0:Delta:1
α=5.0
ThermalConductivity=0.1
# We can put this in matrix form as

n=length(x)
d=zeros(n)
dl=zeros(n-1)
du=zeros(n-1)
C=zeros(n)
# The code below constructs the matrix

du[1]=0.0
d[1]=1.0
C[1]=5.0
for i=2:n-1
    dl[i-1]=(1.0/Delta^2)
    d[i]=-(2.0/Delta^2)
    du[i]=(1.0/Delta^2)
    C[i]=-1.0/ThermalConductivity
end
dl[n-1]=-1
d[n]=1.0
C[n]=0.0

# Construct the specialized structured matrix
A = Tridiagonal(dl, d, du)

A_sparse = sparse(A)

prob=LinearProblem(A,C)
#@btime y=LinearSolve.solve(prob)
@time Tuniform=LinearSolve.solve(prob)

probSparse=LinearProblem(A_sparse,C)
@time y=LinearSolve.solve(probSparse, UMFPACKFactorization())
# Note that tridag solver is faster since it is specific for tridiagonal systems.

xplot=0:0.01:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)")
lines!(ax,x,Tuniform,color=:black,label="Uniform Heating")
scatter!(ax,x,Tuniform,color=:black,markersize=20)
axislegend(ax,position=:lt)
# save("figures/05p01IntroBVP01.svg")   # (figure-save disabled in study file)
fig

y[end]
# The governing equation is

for i=2:n-1
    C[i]=-(1.0+2*x[i])/ThermalConductivity
end

prob=LinearProblem(A,C)
#@btime y=LinearSolve.solve(prob)
@time Tlinear=LinearSolve.solve(prob)

xplot=0:0.01:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)")
lines!(ax,x,Tuniform,color=:black,label="Uniform Heating")
lines!(ax,x,Tlinear,color=:blue,label="Linear Heating")
axislegend(ax,position=:lt)
fig
