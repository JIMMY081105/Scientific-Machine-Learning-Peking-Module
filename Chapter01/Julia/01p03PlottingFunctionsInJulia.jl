using CairoMakie

f(x)=x^2-2x+0.6;

xplot=0:0.01:2

Array(xplot)

f.(xplot)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(0,2,-0.5,1.0))
lines!(ax,xplot,f.(xplot)) #main code to plot f(x)
hlines!(ax,[0],color=:black,linewidth=1.0)
scatter!(ax,1,f.(1);color=:black,markersize=15)
save("../figures/01p03PlottingFunctionsInJulia01.svg", fig) #save figure as svg file
display(fig)

f(x)=(x^3-x)^2
xplot=-2.0:0.01:2.0
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-2,2,-0.01,0.2))
lines!(ax,xplot,f.(xplot))
scatter!(ax,[0, -1, 1],f.([0, -1, 1]);color=:black,markersize=15)
scatter!(ax,[1/sqrt(3), -1/sqrt(3)],f.([1/sqrt(3), -1/sqrt(3)]);color=:magenta,markersize=15)
display(fig)

himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2 #the input of himmelblau is x which is a vector of two elements.

x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [himmelblau([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x_1", 
    ylabel = L"x_2",
    xlabelsize = 20, 
    ylabelsize = 20,
    title = "Himelblau function",
    limits=(-6,6.0,-6.0,6.0))
levels = 10.0.^range(0, 3.5; length=10)
contourf!(ax,x1range,x2range,funcplot; levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; levels,label=true,color=:black)
save("../figures/01p03PlottingFunctionsInJulia02.svg", fig) #save figure as svg file

scatter!(ax,-0.270845,-0.923039;markersize=20,color=:magenta)
scatter!(ax,3.0,2.0,markersize=20,color=:grey)
scatter!(ax,-2.805118,3.13132,markersize=20,color=:grey)
scatter!(ax,-3.779310,-3.283136,markersize=20,color=:grey)
scatter!(ax,3.584428,-1.848126,markersize=20,color=:grey)


display(fig)

x1range=-5:0.02:5;
x2range=-5:0.02:5;

rosenbrock(x)=(1.0 - x[1])^2 + 1.0 * (x[2] - x[1]^2)^2
funcplot = [rosenbrock([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x_1", 
    ylabel = L"x_2",
    xlabelsize = 20, 
    ylabelsize = 20,
    title = "Rosenbrock function",limits=(-5,5.0,-5.0,5.0))
levels = 10.0.^range(-2, 3.5; length=20)
contourf!(ax,x1range,x2range,funcplot; levels, colormap=:bwr)
contour!(ax,x1range,x2range,funcplot;  levels, color=:black)
scatter!(ax,[1],[1];color=:grey,markersize=15)
display(fig)
