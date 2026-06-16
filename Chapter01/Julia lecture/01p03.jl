using CairoMakie



A(r)=(2*π*r^2)+(2/r)

range_r=0.0:0.1:100.0

fig = Figure()
ax = Axis(fig[1, 1],
         xlabel = L"x", 
         ylabel = L"f(x)",
         limits=(0,5,-0.5,30),
         yticks=0.0:3.0:30.0,)
lines!(ax,range_r,A.(range_r)) 
display(fig)


f(x)=sin(x[1])+cos(3*x[2])
x1range=-10:0.2:10;
x2range=-10:0.2:10;
funcplot = [f([x1,x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x_1", 
    ylabel = L"x_2",
    xlabelsize = 20, 
    ylabelsize = 20,
    title = "sin cos function",limits=(-10.0,10.0,-10.0,10.0))
levels = 10.0.^range(-2, 3.5; length=20)
contourf!(ax,x1range,x2range,funcplot; levels, colormap=:bwr)
contour!(ax,x1range,x2range,funcplot;  levels, color=:black)
display(fig)