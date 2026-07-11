using Random
using Zygote
using Optimization
using CairoMakie
using Distributions
using Enzyme
using OptimizationOptimisers

N_SAMPLES=5
rng=Xoshiro(1) #my random number generator

xi=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
yi=rand(rng,Normal(2.3,0.1),N_SAMPLES)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1,1,1,5),
    backgroundcolor = :transparent)
scatter!(ax,xi,yi;label="Data",color=:black,markersize=15)
axislegend(ax;position = :rt)
save("../figures/03p01LinearLeastSquaresGradientDescent01.png", fig)
display(fig)


f_model(a0,xi)=a0
S(a0)=(1/2)*sum((yi.-f_model.(a0,xi)).^2)

avec=0:0.5:5
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"S(a_0)")
lines!(ax,avec,S.(avec);label="Data",color=:black)
axislegend(ax;position = :rt)
fig


dSda(a0)=-sum(yi.-f_model.(a0,xi))


a0=3.0
α=0.001

for i=1:1000
    a0-=α*dSda(a0)
end

a0

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1,1,1,5),
    backgroundcolor = :transparent)
scatter!(ax,xi,yi;label="Data",color=:black,markersize=15)
lines!(ax,xi,f_model.(a0,xi))
axislegend(ax;position = :rt)
display(fig)

N_SAMPLES=50 #50 data points
rng=Xoshiro(1) #my random number generator

x_samples=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=2.0.*(x_samples).+3.0.+y_noise; #y=2x+3+epsilon

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,1,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
axislegend(ax;position = :rt)
fig

fmodel(a,x)=a[1]*x.^0+a[2].*x;

a=[1.0,1.0] #initial assumptions of the values of  a0, a1 
xplot=Array(-1:0.001:1)


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="model",color=:blue)
axislegend(ax;position = :rt)
fig

function S(parameters,x_samples)
           ŷ = fmodel(parameters, x_samples)
           sum(0.5*(y_samples .- ŷ).^2)
end

S(a,x_samples)

a0range=0:0.1:5
a1range=0:0.1:5
loss_values=[S([a0,a1],x_samples) for a0 in a0range, a1 in a1range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"a_1",limits=(0,5.0,0.0,5.0))
contourf!(ax,a0range,a1range,loss_values; levels=50, colormap=:bwr)
contour!(ax,a0range,a1range,loss_values; levels=50,labels=true, color=:black)
scatter!(ax,[3],[2];markersize=20,color=:black)
fig

a0=[1.0;1.0] #initial guess  of the values of  a0, a1.
optf = OptimizationFunction(S, ADTypes.AutoZygote())
prob = OptimizationProblem(optf,a0,x_samples) #first argument is the optimization function, second is the guess value and third is the parameter for optf()
sol= solve(prob,Optimisers.Descent(0.01),maxiters=50_000)

a=[sol.u[1],sol.u[2]] #initial assumptions of the values of  a0, a1 
xplot=Array(-1:0.001:1)


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="model",color=:blue)
axislegend(ax;position = :rt)
fig
