using CairoMakie
using LinearAlgebra

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

x1range=-5:0.02:5;
x2range=-5:0.02:5;
bowl(x)=x[1]^2+x[2]^2

function Gradient_bowl(x)
    d1=2*x[1]
    d2=2*x[2]
    return [d1,d2]
end

funcplot = [bowl([x1,x2]) for x1 in x1range, x2 in x2range]

x = [(x1i, x2i) for x1i in x1range, x2i in x2range]
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Bowl function",limits=(-5,5.0,-5.0,5.0))
levels = range(0.01, 50; length=6)
contourf!(ax,x1range,x2range,funcplot; levels=50, colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; labels=true, levels, color=:black, linewidth = 5)
scatter!(ax,[0],[0],color=:black,markersize=20)

x1range_arrow=-5:1:5;
x2range_arrow=-5:1:5;
arrowplot = [Gradient_bowl([x1,x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax,x1range_arrow,x2range_arrow,-getindex.(arrowplot,1),-getindex.(arrowplot,2);color=(:black, 0.2),lengthscale = 0.1)

# Save as PNG
save("../figures/my_plot.png", fig)
display(fig)

x0=[1.5,2.5] #initial guess x0.  Since this is a two-dimensional function, you need two guesses, one for x1 and another for x2
alpha=0.1 #learning rate
data=[Point2D(x0...,bowl(x0))]
x=x0
for i in 1 : 20
    x-=alpha*Gradient_bowl(x)
    push!(data,Point2D(x...,bowl(x)))
end
lines!(ax,getproperty.(data, :x1),getproperty.(data, :x2);linewidth=3,color=:grey) 
scatter!(ax,x0[1],x0[2],color=:blue,markersize=20)
scatter!(ax,getproperty.(data, :x1),getproperty.(data, :x2);color=:grey,markersize=20,strokecolor = :black,strokewidth = 2,) 

# Save as PNG
save("../figures/my_plot_02.png", fig)
display(fig)

quad(x)=(3.0/2.0)*x[1]^2+2*x[1]*x[2]+3*x[2]^2-2*x[1]+8*x[2] # Define the function
function Gradient_quad(x)
    d1=3*x[1]+2*x[2]-2 #df/dx1
    d2=6*x[2]+2*x[1]+8 #df/dx2
    return [d1,d2]
end


function Hessian_quad(x)
    return [3 2;2 6] # This is the Hessian of the function which we will use later.
end

x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [quad([x1,x2]) for x1 in x1range, x2 in x2range]


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2",title = "Quadratic function",limits=(-6,6.0,-6.0,6.0))
contour!(ax,x1range,x2range,funcplot; labels=true, levels=20,color=:black,linewidth = 5)
contourf!(ax,x1range,x2range,funcplot;  levels=50,colormap=:bwr)

x1range_arrow=-6:1:6;
x2range_arrow=-6:1:6;
arrowplot = [Gradient_quad([x1,x2]) for x1 in x1range_arrow, x2 in x2range_arrow]
arrows2d!(ax,x1range_arrow,x2range_arrow,-getindex.(arrowplot,1),-getindex.(arrowplot,2);color=(:black, 0.2),lengthscale = 0.1)

scatter!(ax,[2],[-2],color=:black,markersize=20)

# Save as PNG
save("../figures/my_plot_03.png", fig)
display(fig)

x0=[4.0,4.0] #initial guess of xm
alpha=0.1 #learning rate
data=[Point2D(x0...,quad(x0))]

x=x0

for i in 1 : 50
    x-=alpha*Gradient_quad(x)
    push!(data,Point2D(x...,quad(x)))
end

lines!(ax,getproperty.(data, :x1),getproperty.(data, :x2);linewidth=3,color=:grey,label="Gradient Descent") 
scatter!(ax,getproperty.(data, :x1),getproperty.(data, :x2);color=:grey,markersize=20,strokecolor = :black,strokewidth = 2,) 

#scatter!(ax,x0[1],x0[2],color=:blue,markersize=20)
fig

x0=[4.0,4.0] #initial guess of x0
alpha=0.1 #learning rate
beta=0.4 #momentum
v=[0.0,0.0]
data_m=[Point2D(x0...,quad(x0))]
x=x0
for i in 1 : 20
    v=beta*v.-alpha*Gradient_quad(x)
    x+=v
    push!(data_m,Point2D(x...,quad(x)))
end


lines!(ax,getproperty.(data_m, :x1),getproperty.(data_m, :x2),linewidth=3,color=:green,label="Momentum") 
scatter!(ax,getproperty.(data_m, :x1),getproperty.(data_m, :x2);color=:green,markersize=20,strokecolor = :black,strokewidth = 2,) 

axislegend(ax;position = :lt)

# Save as PNG
save("../figures/my_plot_04.png", fig)
display(fig)

x0=[4.0,4.0] #initial guess of xm
alpha=0.1 #learning rate
beta=0.4
v=[0.0,0.0]
x=x0
data_nm=[Point2D(x0...,quad(x0))]
for i in 1 : 20
    v =beta*v-alpha*Gradient_quad(x+beta*v)
    x+=v
    push!(data_nm,Point2D(x...,quad(x)))
end

lines!(ax,getproperty.(data_nm, :x1),getproperty.(data_nm, :x2),linewidth=5,color=:red,label="Nesterov Momentum") 
axislegend(ax;position = :lt)
fig
