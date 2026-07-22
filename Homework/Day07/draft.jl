using CairoMakie
using NonlinearSolve
using ForwardDiff



function FunctionToSolve(x,p)
 func=[x[1]^2-x[2]+p[1],3*cos(x[1])-x[2]]
 return func
end
function loss(b,x)
 prob = NonlinearProblem(FunctionToSolve,x,b)
 sol = NonlinearSolve.solve(prob)
 return ((sol.u[1]-0.6).^2) #trying to minimise x_1=0.6
end
f=(b)->loss(b,[1.0,1.0]) # We want to use ForwardDiff, only one argument allowed
bguess=1.0
α=0.1
for i=1:10
 bguess-=α*ForwardDiff.gradient(f,[bguess])[1]
 println(loss(bguess,[1.0,1.0]))
end
bguess