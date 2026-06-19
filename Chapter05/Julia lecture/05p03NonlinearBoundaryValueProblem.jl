########################################################################################
# 5.3 Nonlinear Boundary Value Problem
########################################################################################
#
# PROBLEM
# Solve a radiation heat-transfer BVP where the equation is nonlinear in T (a d*(T^4
# - TR^4) term), so it cannot be written as a matrix system and must be solved
# iteratively.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Discretise the derivatives on a grid as usual, but keep the nonlinear T^4 term
#      as-is.
#   2. Write the residual as a vector function HeatedDiskExample(T, p): each entry is
#      'left side minus right side' at a grid point (should be zero at the solution),
#      with the two boundary conditions as the first and last entries.
#   3. Pack the physical constants (R, d, TR, Delta) into a parameter vector p.
#   4. Set up a NonlinearProblem with an initial guess and solve it with
#      NonlinearSolve (which internally does Newton-Raphson using an
#      automatically-computed Jacobian).
#   5. Plot the resulting temperature profile T(r).
#
# This file is notebook "05p03NonlinearBoundaryValueProblem" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


using NonlinearSolve
using CairoMakie
# Again you use the finite diference approximation

d=0.1;TR=10;R=5;
Delta=0.01;
r=0:Delta:R;
# Set up parameters of the problem

N=length(r); #n is the number gridpoints
T=zeros(N);
p=[R,d,TR,Delta]


HeatedDiskExample(T,p)
# Define the nonlinear vector functions

function HeatedDiskExample(T,p)
    R=p[1]
    d=p[2]
    TR=p[3]
    Delta=p[4]

    func=vcat( T[1]-T[2],
        ((T[3:N]-2*T[2:N-1]+T[1:N-2])/Delta^2.0).+(1/(2*Delta*r[2:N-1]))*(T[3:N].-T[1:N-2]).-d*(T[2:N-1].^4.0.-TR^4).+(r[2:N-1].-R).^2,
       T[N]-TR)

    return func
end   

HeatedDiskExample(T,p)
# Use Julia code to solve the problem.

x0=ones(length(r)) #initial guess
prob = NonlinearProblem(HeatedDiskExample,x0,p)
sol = NonlinearSolve.solve(prob)

fig = Figure() #set up a figure
ax1 = Axis(fig[1, 1], xlabel = L"r", ylabel = L"T(r)") #Define axis in the figure
lines!(ax1,r,sol.u;color=:black) #Plot i_5 vs V_1
display(fig)
