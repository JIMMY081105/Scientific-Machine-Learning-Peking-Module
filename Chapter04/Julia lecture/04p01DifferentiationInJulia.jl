########################################################################################
# 4.1 Introduction to Differentiation
########################################################################################
#
# PROBLEM
# Understand the different ways to compute derivatives (manual, symbolic, numerical,
# automatic) and learn to use Julia's ForwardDiff and Zygote packages to get exact
# derivatives, gradients, and Jacobians without differentiating by hand.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Symbolic differentiation with the Symbolics package (good for small problems
#      only).
#   2. Numerical differentiation: forward/backward/central differences plus the
#      complex-step trick; plot error vs step size h to see truncation and round-off
#      error fight each other.
#   3. Automatic differentiation with ForwardDiff.derivative / Zygote.gradient -
#      machine-precision derivatives; plot them against the analytic derivative to
#      confirm.
#   4. Apply AD in place of hand derivatives: Newton-Raphson using ForwardDiff for
#      f'(x), and gradient descent on Himmelblau using ForwardDiff.gradient.
#   5. Functions with parameters: note ForwardDiff needs a single (vector) argument
#      while Zygote can differentiate w.r.t. several; fit f_model = a1(1 - exp(a2*x))
#      to data with Zygote gradients.
#   6. Vector-valued functions: compute Jacobians with ForwardDiff.jacobian /
#      Zygote.jacobian, and see why Zygote rejects in-place array mutation.
#
# This file is notebook "04p01DifferentiationInJulia" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


using CairoMakie
using ForwardDiff
using Zygote
using Random
using Distributions
# Consider the function

f(x)=sin(x^2);
xplot=-1:0.01:1
lines(xplot,f.(xplot))
# Manual diffentiation is we apply the rules differentiation we learn in high school
# to get the derivative

using Symbolics

# Define  x as a symbolic variable
@variables x

#Define the function.  Note that we cannot define f here because f has been defined as a function prior in the cell above
f1 = sin(x^2)

#Define a differential operator with respect to x
D=Differential(x)

#Expand the expression
expand_derivatives(D(f1))

# TODO (Exercise Ex01): Change code above to calculate the symbolic derivative \sin(x^2)+\exp(-\cos(x)).
# The code segment below implements the forward, central and complex step methods to
# calculate the derivative for the function

dfdx(x)=2*x*cos(x^2)
h=1e-3
xi=π/2.0
println("forward_diff=$( (f(xi+h)-f(xi))/h )")
println("backward_diff=$( (f(xi)-f(xi-h))/h )")
println("central_diff=$( (f(xi+h)-f(xi-h))/(2*h) )")
println("complex_step=$( imag(f(xi+im*h))/h )")
println("exact_diff=$( dfdx(xi) )")
# You can see from the numbers that they produce, the central and complex step
# methods produce comparable accuracy.

#Algorithms for forward, backward, central and complex step method for computing derivatives
diff_forward(f, x; h=sqrt(eps(Float64))) = (f(x+h) - f(x))/h
diff_central(f, x; h=cbrt(eps(Float64))) = (f(x+h/2) - f(x-h/2))/h
diff_backward(f, x; h=sqrt(eps(Float64))) = (f(x) - f(x-h))/h
diff_complex(f, x; h=1e-20) = imag(f(x + h*im)) / h

#Compute the derivative at x=π
x = π/2
d_true = π*cos(π^2/4) #true value of the derivative

#Compute the derivatives using numerical approximation for h=10^(-17) to 10^1
arr_h = collect(10 .^ range(-17, stop=1, length=101))
arr_forward = [abs(d_true - diff_forward(f, x, h=h))
					for h in arr_h];
arr_central = [abs(d_true - diff_central(f, x, h=h))
						for h in arr_h];
arr_complex = [abs(d_true - diff_complex(f, x, h=h))
						for h in arr_h];

fig1 = Figure()
ax1 = Axis(fig1[1, 1],xscale=log10,yscale=log10,xlabel="step size h",ylabel="Absolute Error",title="Error vs h")
lines!(ax1,arr_h, arr_forward;color=:red,label="Forward")
lines!(ax1,arr_h, arr_central;color=:blue,label="Central Difference")
lines!(ax1,arr_h, arr_complex;color=:green,label="Complex Step")
axislegend( position = :lb)

fig1

# ================================================
# Univariate Function without any parameters
# ================================================

ForwardDiff.derivative(f, π/2)
println("The true answer is ",d_true," ForwardDiff gives ",ForwardDiff.derivative(f, x))
# Note above thata the ForwardDiff.derivative() gives a value of the derivative up to
# machine precision

Zygote.gradient(f, x)
println("The true answer is ",d_true," Zygote gives ",Zygote.gradient(f, x)[1])
# All the derivatives given by ForwardDiff.jl and Zygote.jl are accurate down to
# machine precision.
# Define derivative functions for all two packages

dfForwardDiff(x)=ForwardDiff.derivative(f, x)
dfZygote(x)=Zygote.gradient(f, x)[1]

xplot=-1:0.01:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "df/dx",limits=(-1.0,1.0,-2.0,2.0))
lines!(xplot,2*xplot.*cos.(xplot.^2);color=:black,label="Ground truth")
lines!(xplot,dfForwardDiff.(xplot);color=:blue,label="ForwardDiff")
lines!(xplot,dfZygote.(xplot);color=:green,label="Zygote")

axislegend(ax,position = :rb)
fig
# In the first week of this course, we used the Newton-Raphson method to find the
# zero of the function

f2(x)=4π*x-2/x^2
x=1.0
for i=1:10
    x-=f2(x)/ForwardDiff.derivative(f2,x)
    println(x)
end

# ==================================================
# Multivariate Function without any parameters
# ==================================================
# We will now use Enzyme to calculate the gradients of the Himmelblau function
# Define the Himmelblau function

himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2

x0 = [0.0, 0.0]
ForwardDiff.gradient(himmelblau,x0)

# TODO (Exercise Ex02): Try using the `Zygote.gradient()` to calculate the gradient of the himmelblau
#   function at (x_1,x_2)=(0,0)

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

x0=[-2.0,-2.0]#initial guess of x0
alpha=0.01
data_gd=[Point2D(x0...,himmelblau(x0))]

x=x0
for i in 1 : 50
    x-=alpha*ForwardDiff.gradient(himmelblau,x)
    push!(data_gd,Point2D(x...,himmelblau(x)))
end

data_gd

# ==============================
# Function with parameters
# ==============================

function NN(x,W,b)
    sin(W*x+b)
end

ForwardDiff.gradient(NN,1,2,3)

ForwardDiff.derivative((x)->NN(x,2,3),1.0)

ForwardDiff.derivative((W)->NN(1,W,3),2.0)

ForwardDiff.derivative((b)->NN(1,2,b),3.0)
# Or you can just use the `Zygote` package

Zygote.gradient(NN,1,2,3)

W=1
b=0
NNd1=(x)->Zygote.gradient(NN,x,W,b)[1]
NNd2=(x)->Zygote.gradient(NNd1,x)[1]

xplot=Array(0:0.01:2π)

xdata=Array(0:0.1:2π)
W=1.0
b=0.0
fig = Figure()
ax = Axis(fig[1, 1],xlabel = "x", ylabel = L"f(x)")
lines!(ax,xplot,NNd1.(xplot);color=:blue)
lines!(ax,xplot,NNd2.(xplot);color=:black)

scatter!(ax,xdata,W*cos.(W*xdata.+b);color=:blue)
scatter!(ax,xdata,-W*W*sin.(W*xdata.+b);color=:black)

fig
# In Notebook 3.5, we used the `Optimisation()` package in Julia to find the
# parameters of the model function

N_SAMPLES=50 #50 data points
rng=Xoshiro(1) #my random number generator

x_samples=rand(rng,Uniform(0,5),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=4*(1.0.-exp.(-1.8*x_samples))+y_noise; #y=2x+3+epsilon

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(0,5,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
axislegend(ax;position = :rt)
fig
# We will define the model and the loss function.

fmodel(a,x)=a[1]*(1.0.-exp.(a[2]*x)); # for this model a[1]=a and a[2]=b.
function custom_loss(parameters,x_samples)
           ŷ = fmodel(parameters, x_samples)
           0.5*sum((y_samples .- ŷ).^2)
end
a=[1.0,-1.0] #initial assumptions of the values of  a0, a1 

S=(a)->custom_loss(a,x_samples)
Zygote.gradient(S, a)[1]

α=0.01 #learning rate
for i=1:500
    a-=α * Zygote.gradient(S,a)[1]
    println(a)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(0,5,0,5),
    xlabelsize = 20, # Set X label size
    ylabelsize = 20) 
scatter!(ax,x_samples,y_samples;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="model",color=:blue)
axislegend(ax;position = :rb)
fig

# ================================================
# Vector (vector-valued) functions of vector
# ================================================

function fvec1(x)
    df=similar(x)
    df[1]=x[1]-1
    for i=2:length(x)-1
        df[i]=x[i]-x[i-1]^2
    end
    df[end]=x[end]-x[end-1]
    return df
end

fvec2(x)=[x[1]-1;
x[2:end-1].-x[1:end-2].^2;
x[end]-x[end-1]]

N=5
x=2*ones(N)
fvec1(x)

fvec1(x).-fvec2(x)

Zygote.jacobian(fvec1,x)[1]

Zygote.jacobian(fvec2,x)[1]

ForwardDiff.jacobian(fvec1,x)

ForwardDiff.jacobian(fvec2,x)
# Recall in the first week that we used `NonlinearSolve()` to find the solution to
# the two equations below

VectorFunc(x)=[x[1]^2-x[2]+1,
                3*cos(x[1])-x[2]]

ForwardDiff.jacobian(VectorFunc,[1,1])
# Now using Newton-Raphson formula

x=[1,1]
for i=1:5
    F=VectorFunc(x)
    J=ForwardDiff.jacobian(VectorFunc,x)
    delta=-J\F
    x+=delta
    println(x)
end
