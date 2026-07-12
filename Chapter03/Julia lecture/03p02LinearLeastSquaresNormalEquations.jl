########################################################################################
# 3.2 Linear Least Squares - Normal Equations
########################################################################################
#
# PROBLEM
# Fit a quadratic y = a0 + a1*x + a2*x^2 to noisy data WITHOUT iterating. Because the
# model is linear in the coefficients, minimising the squared-error loss gives a
# single linear system (the normal equations) that is solved in one step.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Generate quadratic-plus-noise data and show that a hand-guessed set of
#      coefficients fits poorly.
#   2. Derive the normal equations: set dS/da0 = dS/da1 = dS/da2 = 0, which yields a
#      3x3 linear system in the coefficients.
#   3. Recognise that system compactly as V'V a = V'y, where V (the Vandermonde
#      matrix) has columns [1, x, x^2].
#   4. Build V with hcat(ones, x, x.^2) and solve a = V'V \ (V'y) in a single line -
#      no iteration.
#   5. Plot the fitted quadratic; the recovered coefficients (approximately 0.5, 1.0,
#      3.0) match the data-generating function.
#
# This file is notebook "03p02LinearLeastSquaresNormalEquations" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ===============================================
# 3.2 Linear Least Squares Normal Equations
# ===============================================

using Random
using CairoMakie
using Distributions
# Linear least squares regression is a method of fitting a function with linear
# coefficients to some data points.

N_SAMPLES=50 #50 data points
rng=Xoshiro(1) #my random number genrator

x_samples=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=3*x_samples.^2.0.+(x_samples).+0.5.+y_noise; #y=3x^2+x+0.5+epsilon
# Let's plot the data and see how it looks like

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
axislegend(ax;position = :lt)
display(fig)
# Let's try to fit the function

a=[1.0,1.0,1.0] #initial assumptions of the values of  a0, a1, a2.  Remember that the indexing in Julia
                # starts from 1, not 0. So a[1] is a0, a[2] is a1 and a[3] is a2.
xplot=-1:0.01:1
yplot=a[1].+a[2]*xplot.+a[3]*xplot.^2.0

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], 
    xlabel = L"x", 
    ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
lines!(ax,xplot,yplot;label="Model",color=:blue)
axislegend(ax;position = :lt)
# save("../figures/03p01LinearLeastSquaresRegression01.png",fig)   # (figure-save disabled in study file)
display(fig)
# The overall idea is to define an error and then try to minimise the error between
# the actual data and the polynomial fit.
# Writing this in Matrix form gives
# Convince yourself that the matrix can be written as

V=hcat(ones(N_SAMPLES),x_samples,x_samples.^2)
a=V'V\(V'y_samples)
# Note that the solution you have found is

fmodel(a,x)=a[1]*x.^0+a[2].*x.+a[3].*x.^2;

fig = Figure()
xplot=-1:0.01:1
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="Model",color=:blue)
axislegend(ax;position = :lt)
fig

# TODO (Class Demo 01): Generate a fictitious data set from the function y=2x^2-10x+3+\epsilon and use
#   f_{model}(x)=a_0+a_1 x+a_2 x^2 \epsilon is a small amount of random noise. Can
#   you find a set of a_0, a_1 and a_2 that make f_{model}(x) looks like y_i? Use the
#   domain x\in[0,2\pi]. Here, find a_0, a_1 and a_2 using the "Normal equations"
#   method.

NUMOF_SAMPLES=50 #50 data points
#set seed to 1, so generation is consistent 
rng=Xoshiro(1)

x_samples=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=2*x_samples.^2.0.+10*(x_samples).+3.0.+y_noise;

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-3,3,-6,25))
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
axislegend(ax;position = :lt)
display(fig)

a=[1.0,1.0,1.0] #initial assumptions of the values of  a0, a1, a2.  Remember that the indexing in Julia
                # starts from 1, not 0. So a[1] is a0, a[2] is a1 and a[3] is a2.
xplot=-1:0.01:1
yplot=a[1].+a[2]*xplot.+a[3]*xplot.^2.0

V=hcat(ones(N_SAMPLES),x_samples,x_samples.^2)
a=V'V\(V'y_samples)
# Note that the solution you have found is

fmodel(a,x)=a[1]*x.^0+a[2].*x.+a[3].*x.^2;

fig = Figure()
xplot=-1:0.01:1
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="Model",color=:blue)
axislegend(ax;position = :lt)
fig

# TODO (Class Demo 02): Read the data from the file "Data.csv" and try to see if you can get a good fit
#   with a third order polynomial


