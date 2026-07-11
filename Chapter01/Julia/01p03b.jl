using CairoMakie

x1range=-5:0.02:5;
x2range=-5:0.02:5;

MyRandomFunction(x)=sin(x[1])*cos(3*x[2])
funcplot = [MyRandomFunction([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x_1", 
    ylabel = L"x_2",
    xlabelsize = 20, 
    ylabelsize = 20,
    title = "Random function",limits=(-5,5.0,-5.0,5.0))

contour!(ax,x1range,x2range,funcplot;  levels=30, color=:black)
contourf!(ax,x1range,x2range,funcplot; levels=30, colormap=:bwr)
display(fig)