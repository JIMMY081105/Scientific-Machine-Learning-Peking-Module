using CairoMakie

struct Point1D
    x::Float64 #guess locations of the minimum
    fvalue::Float64 # value of the function at the guess locations
end

f(x)=x^2-2x+0.6 #Define the function
∇f(x)=2x-2 #Define the gradient of the function


xrange=-1:0.01:3
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-1,3,-0.5,2.0))
lines!(ax,xrange,f.(xrange))
scatter!(ax,1.0,f.(1.0);color=:black,markersize=15)
display(fig)

x0=2.5 #initial guess of x0
alpha=0.1 #learning rate
data=[Point1D(x0,f(x0))] #Store the initial value of the guess and the function value into a struct called data
x=x0
for i in 1 : 20
    x-=alpha*∇f(x) #standard gradient descent update equation
    push!(data,Point1D(x,f(x))) #store x and f(x) into data
end

fig_conv = Figure(size=(1000, 400))

ax_conv1 = Axis(fig_conv[1, 1], xlabel = L"x", ylabel = L"f(x)",limits=(-1,3,-0.5,2.0))
lines!(ax_conv1,xrange,f.(xrange))
scatter!(ax_conv1,getproperty.(data,:x),getproperty.(data,:fvalue);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)


ax_conv2 = Axis(fig_conv[1, 2], xlabel = "iter", ylabel = L"x")
lines!(ax_conv2,getproperty.(data, :x),label="Gradient Descent")
scatter!(ax_conv2,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)
hlines!(ax_conv2,1.0;color=:red)
save("../figures/02p03GradientBasedMethods01.svg",fig_conv)
display(fig_conv)


x0=2.5 #initial guess of x0
alpha=0.1 #learning rate
beta=0.6 #momentum
v=0.0 # setting the initial velocity to be zero.  
x=x0
data_m=[Point1D(x0,f(x0))]
for i in 1 : 20
    v=beta*v-alpha*∇f(x) #update equation for the momentum method. 
    x+=v #update equation for the momentum method. 
    push!(data_m,Point1D(x,f(x)))
end

x0=2.5 #initial guess of xm
alpha=0.1 #learning rate
beta=0.6
v=0.0
x=x0
data_nm=[Point1D(x0,f(x0))]
for i in 1 : 20
    v =beta*v-alpha*∇f(x+beta*v) #update equations for the Nesterov Momentum 
    x+=v #update equations for the Nesterov Momentum 
    push!(data_nm,Point1D(x,f(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x")

lines!(ax_conv,getproperty.(data, :x),label="Gradient Descent")
scatter!(ax_conv,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

lines!(ax_conv,getproperty.(data_m, :x),label="Momentum",color=:orange)
scatter!(ax_conv,getproperty.(data_m,:x);
    color=1:length(data_m),
    strokewidth=1,
    strokecolor=:black,
    colormap = :OrRd,
    markersize=15)

lines!(ax_conv,getproperty.(data_nm, :x),label="Nesterov Momentum",color=:green)
scatter!(ax_conv,getproperty.(data_nm,:x);
    color=1:length(data_nm),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Greens,
    markersize=15)

axislegend(ax_conv,  position = :rt,orientation = :vertical)
hlines!(ax_conv,1.0;color=:red,linewidth=5.0)
save("../figures/02p03GradientBasedMethods03.svg",fig_conv)
display(fig_conv)

g(x)=-exp(-x^2)
∇g(x)=2*x*exp(-x^2)
xplot=-5:0.01:5
yplot=[g(x) for x in xplot];
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = L"$g(x)$",limits=(-5,5,-1.5,0.5))
lines!(ax,xplot,yplot)
fig

x0=2.0 #initial guess of xm
alpha=0.1 #learning rate
data=[Point1D(x0,g(x0))]
x=x0
for i in 1 : 100
    x-=alpha*∇g(x)
    push!(data,Point1D(x,g(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")
lines!(ax_conv,getproperty.(data, :x);label="Gradient Descent")
axislegend(ax_conv,  position = :rt,orientation = :vertical)

scatter!(ax_conv,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

hlines!(ax_conv,0.0;color=:red,linewidth=5.0)
fig_conv

x0=2.0 #initial guess of x0
alpha=0.1 #learning rate
beta=0.6 #momentum
v=0.0
x=x0
data_m=[Point1D(x0,g(x0))]
for i in 1 : 100
    v=beta*v-alpha*∇g(x)
    x+=v
    push!(data_m,Point1D(x,g(x)))
end

x0=2.0 #initial guess of x0
alpha=0.1 #learning rate
beta=0.6
v=0.0
x=x0
data_nm=[Point1D(x0,g(x0))]
for i in 1 : 100
    v =beta*v-alpha*∇g(x+beta*v)
    x+=v
    push!(data_nm,Point1D(x,g(x)))
end

fig_conv = Figure()
ax_conv = Axis(fig_conv[1, 1], xlabel = "iter", ylabel = L"x_m")
lines!(ax_conv,getproperty.(data, :x);label="Gradient Descent")


scatter!(ax_conv,getproperty.(data,:x);
    color=1:length(data),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Blues,
    markersize=15)

hlines!(ax_conv,0.0;color=:red,linewidth=5.0)


lines!(ax_conv,getproperty.(data_m, :x),label="Momentum",color=:orange)
scatter!(ax_conv,getproperty.(data_m,:x);
    color=1:length(data_m),
    strokewidth=1,
    strokecolor=:black,
    colormap = :OrRd,
    markersize=15)

lines!(ax_conv,getproperty.(data_nm, :x),label="Nesterov Momentum",color=:green)
scatter!(ax_conv,getproperty.(data_nm,:x);
    color=1:length(data_nm),
    strokewidth=1,
    strokecolor=:black,
    colormap = :Greens,
    markersize=15)

axislegend(ax_conv,  position = :rt,orientation = :vertical)
fig_conv
