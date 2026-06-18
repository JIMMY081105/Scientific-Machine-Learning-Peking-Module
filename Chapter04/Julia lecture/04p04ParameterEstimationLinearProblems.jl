########################################################################################
# 4.4 Implicit Problems (Parameter Estimation): Linear Equations
########################################################################################
#
# PROBLEM
# Solve an inverse problem where the quantity of interest is defined implicitly (only
# by solving a linear system): find the input voltage V1 that makes the circuit
# current i5 equal zero. This needs derivatives taken THROUGH a linear solve.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Recall the circuit linear system A i = C and solve it (backslash and
#      LinearSolve).
#   2. Sweep V1 over a range, re-solving each time, and plot i5 (and i5^2) versus V1
#      to locate roughly where i5 = 0.
#   3. Define a loss(p) that, for a trial V1 = p, rebuilds C, solves the system, and
#      returns i5^2 - so evaluating the loss requires solving the linear system.
#   4. Confirm ForwardDiff.gradient can differentiate the loss through the solve
#      (positive/negative gradient matches the plot).
#   5. Minimise the loss for V1 with hand-coded gradient descent, then with the
#      Optimization.jl library (Adam) and a callback that logs the loss.
#
# This file is notebook "04p04ParameterEstimationLinearProblems" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


using LinearSolve
using CairoMakie
using Optimization
using Optimisers
using OptimizationOptimisers
using ForwardDiff
# So far, in this course, we have looked at modelling data with mathematical models
# where the output of the models can be evaluated explicitly.

# ======================
# Linear Equations
# ======================
# Example of linear set of equations is given in the circuit diagram below
# Applying Kirchoff’s voltage law gives

R₁=5.0
R₂=150.0
R₃=100.0
R₄=250.0
R₅=200.0

V₁=100.0
V₂=50.0

A=[R₁ R₂ 0 0 0;
  0 -R₂ R₃ R₄ 0;
  0 0 0 R₄ -R₅;
  1.0 -1.0 -1.0 0 0;
  0 0 1.0 -1.0 -1.0]

C₁=[V₁, 0, V₂, 0, 0] #C1 is a (5x1) column vector
C₂=[V₁  0  V₂  0  0] #C2 is a (1x5) row matrix

@time i=A\C₁

A\C₂'

# Define the linear problem
prob = LinearProblem(A,C₁)

# Solve the problem using defaults
@time sol = solve(prob)

i=sol.u

A=[R₁ R₂ 0 0 0;
  0 -R₂ R₃ R₄ 0;
  0 0 0 R₄ -R₅;
  1 -1 -1 0 0;
  0 0 1 -1 -1]

C₁=[V₁, 0, V₂, 0, 0] #C1 is a (5x1) column vector
V₁vec=50:1:100
i₅vec=zeros(length(V₁vec))
for i=1:length(V₁vec)
    C₁[1]=V₁vec[i]
    prob = LinearProblem(A,C₁)
    sol = solve(prob)
    i₅vec[i]=sol.u[5]
end

i₅vec

fig = Figure() #set up a figure
ax1 = Axis(fig[1, 1], xlabel = L"V_1", ylabel = L"i_5",limits=(50,100,-0.07,0.07)) #Define axis in the figure
lines!(ax1,V₁vec,i₅vec;color=:black) #Plot i_5 vs V_1
# save("../figures/04p04ParameterEstimationLinearProblems03.svg",fig)   # (figure-save disabled in study file)
display(fig)

fig = Figure() #set up a figure
ax2 = Axis(fig[1, 1], xlabel = L"V_1", ylabel = L"i_5^2") #Define axis in the figure
lines!(ax2,V₁vec,i₅vec.^2;color=:black) #Plot i_5^2 vs V_1
# save("../figures/04p04ParameterEstimationLinearProblems04.svg",fig)   # (figure-save disabled in study file)
display(fig)

function loss(p)
    C=[p[1],0,V₂,0,0]
    prob = LinearProblem(A,C)
    i = LinearSolve.solve(prob)
    return sum(i[end].^2)
end

loss([91.0])
# Check the calculation of the gradient to see if all makes sense

ForwardDiff.gradient(loss,[91.0])

ForwardDiff.gradient(loss,[55.0])
# Now we will use gradient descent to find the minimum of the loss function

iguess=60
α=100.00
for i=1:50000
    iguess-=α*ForwardDiff.gradient(loss,[iguess])[1]
end
iguess

function callback(state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

losses = Float64[]
optf = OptimizationFunction((p,x)->loss(p), AutoForwardDiff())
optprob = OptimizationProblem(optf,[80.0])
sol=Optimization.solve(optprob,Adam(0.1f0),maxiters=500,callback=callback)
#sol=Optimization.solve(optprob,Descent(0.1f0),maxiters=10000,callback=callback)
