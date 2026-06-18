########################################################################################
# 3.1 Linear Regression with Gradient Descent
########################################################################################
#
# PROBLEM
# Given noisy (x, y) data, find the model coefficients that best fit it by minimising
# the sum-of-squared-errors loss S with gradient descent. Done twice: a constant
# model f = a0, then a straight-line model f = a0 + a1*x.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Generate fictitious data from a known function plus noise (so the 'right'
#      answer is known) and plot it.
#   2. Example 1 (constant model f = a0): write the loss S(a0) = (1/2) sum (yi -
#      a0)^2 and its analytic derivative dS/da0, then hand-code gradient descent a0
#      <- a0 - alpha*dS/da0 for 1000 steps; recover a0 approximately 2.3.
#   3. Example 2 (line f = a0 + a1*x): write the model as a[1] + a[2]*x (Julia is
#      1-indexed) and the loss S(a, x_samples).
#   4. Visualise the loss surface S over (a0, a1) as a contour plot to see the single
#      minimum at (3, 2).
#   5. Minimise S with the Optimization.jl library (Descent optimiser, AutoZygote
#      gradient), then plot the fitted line over the data.
#
# This file is notebook "03p01LinearLeastSquaresGradientDescent" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


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
# save("../figures/03p01LinearLeastSquaresGradientDescent01.png", fig)   # (figure-save disabled in study file)
display(fig)

f_model(a0,xi)=a0
S(a0)=(1/2)*sum((yi.-f_model.(a0,xi)).^2)

avec=0:0.5:5
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"S(a_0)")
lines!(ax,avec,S.(avec);label="Data",color=:black)
axislegend(ax;position = :rt)
fig
# Formula for the derivative

dSda(a0)=-sum(yi.-f_model.(a0,xi))

a0=3.0
α=0.001

for i=1:1000
    a0-=α*dSda(a0)
end

a0
# Now let's plot our model with our original data

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
# As can be seen above, the model fits the data pretty well!
# In this next example, we will try to find a model that fits the fictitious data
# from the function
# Generating the fictious data

N_SAMPLES=50 #50 data points
rng=Xoshiro(1) #my random number generator

x_samples=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=2.0.*(x_samples).+3.0.+y_noise; #y=2x+3+epsilon
# Let's plot the data and see how it looks like

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,1,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
axislegend(ax;position = :rt)
fig
# We will define a model of the same form as the data.

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

# TODO (Class Demo 01): Generate a fictitious data set from the function y=2x^2-10x+3+\epsilon and use
#   f_{model}(x)=a_0+a_1 x+a_2 x^2 \epsilon is a small amount of random noise. Can
#   you find a set of a_0, a_1 and a_2 that make f_{model}(x) looks like y_i? Use the
#   domain x\in[0,2\pi]. Here we will write our own Gradient Descent code to solve
#   this problem. Prepared by Andrew Ooi 8th July 2026
