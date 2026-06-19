########################################################################################
# 5.2 Boundary Value Problem (Another Example)
########################################################################################
#
# PROBLEM
# Solve a fuller linear BVP that has both first- and second-derivative terms: y'' +
# (2/x) y' - (2/x^2) y = 0 on [1,2] with y(1)=5, y(2)=3, and check against the known
# exact solution y = x + 4/x^2.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Discretise both derivatives: central difference for y'' and for y', evaluated
#      at each grid point.
#   2. Substitute into the ODE so each interior point gives a row with three nonzero
#      coefficients (i-1, i, i+1), plus two Dirichlet boundary rows, forming A y = C.
#   3. Fill the dense matrix A and vector C in a loop and solve with backslash.
#   4. Plot the finite-difference solution on top of the exact y = x + 4/x^2 to
#      confirm agreement.
#   5. Rebuild A as a sparse matrix and solve with LinearSolve (UMFPACK) to show the
#      speed-up that grows with grid size.
#
# This file is notebook "05p02BoundaryValueProblem01" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


using CairoMakie
# Specify parameters of the problem

Delta=0.001
x=1:Delta:2
α=5.0
β=3.0

n=length(x)
A=zeros(n,n)
C=zeros(n)
# Set up the matrix system
# If we now sub everything back into the original equation, and assume we have a grid
# with 5 intervals (6 grid points), we can put into matrix form as

A[1,1]=1.0
C[1]=5.0
for i=2:n-1
    A[i,i-1]=(1.0/Delta^2)-(2/(x[i]))*(1/(2*Delta))
    A[i,i]=-(2.0/Delta^2)-(2/(x[i]*x[i]))
    A[i,i+1]=(1.0/Delta^2)+(2/(x[i]))*(1/(2*Delta))
end
A[n,n]=1.0
C[n]=3.0

C
# Solve the equation to get

@time y=A\C

yexact(x)=x+4/x^2

xplot=1:0.01:2
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
lines!(ax,xplot,yexact.(xplot),color=:black,label="Analytical solution")
lines!(ax,x,y,color=:blue,label="Finite difference solution")
axislegend(ax,position=:rt)
fig

collect(x)
# It takes a long time to solve this problem.

using LinearSolve
using SparseArrays
# Define sparse Matrix

ASparse=sparse(A)


probSparse=LinearProblem(ASparse,C)
@time y2=LinearSolve.solve(probSparse, UMFPACKFactorization())
