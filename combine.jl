using Random
using CairoMakie
using Distributions

N_SAMPLES=50 #50 data points
rng=Xoshiro(1) #my random number genrator

x_samples=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=3*x_samples.^2.0.+(x_samples).+0.5.+y_noise; #y=3x^2+x+0.5+epsilon

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
axislegend(ax;position = :lt)
display(fig)

a=[1.0,1.0,1.0] #initial assumptions of the values of  a0, a1, a2.  Remember that the indexing in Julia
                # starts from 1, not 0. So a[1] is a0, a[2] is a1 and a[3] is a2.
xplot=-1:0.01:1
yplot=a[1].+a[2]*xplot.+a[3]*xplot.^2.0

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x", 
    ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
lines!(ax,xplot,yplot;label="Model",color=:blue)
axislegend(ax;position = :lt)
save("../figures/03p01LinearLeastSquaresRegression01.png",fig)
display(fig)

V=hcat(ones(N_SAMPLES),x_samples,x_samples.^2)
a=V'V\(V'y_samples)


fmodel(a,x)=a[1]*x.^0+a[2].*x.+a[3].*x.^2;

fig = Figure()
xplot=-1:0.01:1
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="Model",color=:blue)
axislegend(ax;position = :lt)
fig