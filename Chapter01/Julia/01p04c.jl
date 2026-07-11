using CairoMakie

function uandv2(x,y)
    return [x^2+y^2-1,4*x^2+y^2/4-1]
end

function Jacobian2(x,y)
    return [2*x 2*y; 8*x y/2]
end

xrange=-1:0.02:1;
yrange=-2:0.02:2;
functionToPlot = [uandv2(x,y) for x in xrange, y in yrange]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize = 25, 
    ylabelsize = 25,
    limits=(-1.0,1.0,-2.0,2.0))

contour!(ax,xrange,yrange,getindex.(functionToPlot,1);  levels = [0.0],color=:black, label=L"u(x,y)")
contour!(ax,xrange,yrange,getindex.(functionToPlot,2);  levels = [0.0],color=:blue, label=L"v(x,y)")

axislegend(ax, position = :rt) # Options: :rt, :lt, :lb, :rb, etc.

display(fig)

x,y=0.5,-1.0

for i=1:50
    J=Jacobian2(x,y)
    F=uandv2(x,y)
    delta=-J\F
    x+=delta[1]
    y+=delta[2]
    println("x=$x, y=$y")
end

function FunctionToSolve2(x)
    func=[x[1]^2+x[2]^2-1,4*x[1]^2+x[2]^2/4-1]
    return func
end

prob = NonlinearProblem((x,p)->FunctionToSolve2(x),[0.5,-1.0],[])
sol = NonlinearSolve.solve(prob)