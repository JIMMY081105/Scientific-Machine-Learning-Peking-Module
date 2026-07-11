using LinearSolve
using NonlinearSolve
using CairoMakie

R₁=5
R₂=150
R₃=100
R₄=250
R₅=200

V₁=100.0
V₂=50.0

A=[R₁ R₂ 0 0 0;
  0 -R₂ R₃ R₄ 0;
  0 0 0 R₄ -R₅;
  1 -1 -1 0 0;
  0 0 1 -1 -1]

C₁=[V₁, 0, V₂, 0, 0] #C1 is a (5x1) column vector
C₂=[V₁  0  V₂  0  0] #C2 is a (1x5) row matrix


@time i=A\C₁

A\C₂'

# Define the linear problem
prob = LinearProblem(A,C₁)

# Solve the problem using defaults
@time sol = solve(prob)

i=sol.u

sol.retcode

Area(x)=2π*x^2+2/x
f(x)=4π*x-2/x^2
dfdx(x)=4π+4/x^3

xplot=0.1:0.01:2
fig = Figure() #set up a figure
ax1 = Axis(fig[1, 1], xlabel = L"x", ylabel = L"A(x)",limits=(0,2,5,25)) #Define axis in the figure
ax2 = Axis(fig[2, 1], xlabel = L"x", ylabel = L"dA/dx(x)",limits=(0,2,-20,20)) #Define axis in the figure
lines!(ax1,xplot,Area.(xplot);color=:black) #Plot A(x) vs x
lines!(ax2,xplot,f.(xplot);color=:black) #Plot dA/dx(x) vs x
save("../figures/01p04SolvingLinearAndNonlinearEquations02.svg", fig)
display(fig) #show the figure

x=1.0 #initial guess
for i=1:10
    x=x-f(x)/dfdx(x)
    println(x)
end

(1/(2π))^(1/3)

x0=1.0 #initial guess
p=[] #no parameters
prob = NonlinearProblem((x,p)->f(x),x0,p)
sol = NonlinearSolve.solve(prob)

x₁=0:0.1:3
y₁=x₁.^2 .+1
y₂=3*cos.(x₁);

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    limits=(0,3,-2.0,5.0),
    xlabelsize = 25, 
    ylabelsize = 25)
lines!(ax,x₁,y₁;color=:black,label=L"u(x,y)") 
lines!(ax,x₁,y₂;color=:blue,label=L"v(x,y)")
axislegend(ax, position = :rt) # Options: :rt, :lt, :lb, :rb, etc.
save("../figures/01p04SolvingLinearAndNonlinearEquations03.svg", fig)
display(fig)

function uandv(x,y)
    return [x^2-y+1,3*cos(x)-y]
end

function Jacobian(x,y)
    return [2*x -1; -3*sin(x) -1]
end

x=y=10.0

for i=1:10
    J=Jacobian(x,y)
    F=uandv(x,y)
    delta=-J\F
    x+=delta[1]
    y+=delta[2]
    println("x=$x, y=$y")
end

function FunctionOnly(x,y)
    func=[x[1]^2-x[2]+1,3*cos(x[1])-x[2]]
    return func
end

xrange=0:0.02:3;
yrange=-2:0.02:5;
functionToPlot = [uandv(x,y) for x in xrange, y in yrange]

getindex.(functionToPlot,1)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize = 25, 
    ylabelsize = 25,
    limits=(0,3.0,-2.0,5.0))

contour!(ax,xrange,yrange,getindex.(functionToPlot,1);  levels = [0.0],color=:black, label=L"u(x,y)")
contour!(ax,xrange,yrange,getindex.(functionToPlot,2);  levels = [0.0],color=:blue, label=L"v(x,y)")

axislegend(ax, position = :rt) # Options: :rt, :lt, :lb, :rb, etc.

display(fig)

function FunctionToSolve(x)
    func=[x[1]^2-x[2]+1,3*cos(x[1])-x[2]]
    return func
end

prob = NonlinearProblem((x,p)->FunctionToSolve(x),[1,1],[])
sol = NonlinearSolve.solve(prob)

scatter!(ax,sol.u[1],sol.u[2];  color=:red,markersize=20)
display(fig)
