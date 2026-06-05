#import linear solving packages
using LinearSolve
#import non-linear solving packages
using NonlinearSolve
#import plotting tools
using CairoMakie

#define the function here 
function FunctionStore(x,y)
    func = [x^3+y^3-1, x^2+y^2-8]
    return func
end

# derivative matrix of the equation system
function Jacobian(x,y)
    return [3*x^2 3*y^2;
            2*x   2*y]
end

#define range of calculation for Julia
xrange=-10.0:0.5:10.0;
yrange=-10.0:0.5:10.0;  
#samples equation on the plane
functionToPlot = [FunctionStore(x,y) for x in xrange, y in yrange]

#get the figure plotted
fig = Figure()
#label x and y
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    # define label sizes
    xlabelsize = 15, 
    ylabelsize = 15,
    #define the limits of the graph
    limits=(-4.0,4.0,-4.0,4.0),
    #get gridlines
    xticks = -4.0:1.0:4.0,
    yticks = -4.0:1.0:4.0)

#plot out the 2 equation with diff colors
contour!(ax,xrange,yrange,getindex.(functionToPlot,1);  levels = [0.0],color=:black, label=L"u(x,y)")
contour!(ax,xrange,yrange,getindex.(functionToPlot,2);  levels = [0.0],color=:blue, label=L"v(x,y)")

axislegend(ax, position = :rt) 

display(fig)

#initial guess for the value of x y 
x=2
y=-2


# loop Newton-Raphson to solve it
for i=1:10
    J=Jacobian(x,y)
    F=FunctionStore(x,y)
    delta=-J\F
    x+=delta[1]
    y+=delta[2]
    println("x=$x, y=$y")
end
