########################################################################################
# 4.3 Application of Forward Mode Automatic Differentiation
########################################################################################
#
# PROBLEM
# Use the hand-built Dual-number AD from the previous notebook to solve real
# problems, so you trust it does the same job as the libraries.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Extend the Dual type with the extra operations needed (/, cos, and more Number
#      combinations).
#   2. Newton-Raphson via Dual numbers: find r where dA/dr = 0 for the cylinder,
#      taking f(r).v and f(r).d straight from one Dual evaluation.
#   3. Nonlinear system: assemble the Jacobian of a vector function from Dual
#      evaluations (seeding one variable at a time) and iterate Newton-Raphson to the
#      solution.
#   4. Regression: fit a constant model a0 to noisy data by gradient descent, taking
#      dS/da0 from the derivative field of a Dual evaluation of the loss.
#
# This file is notebook "04p03ApplicationOfForwardMode" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


using CairoMakie
using Distributions
using Random
# Define dual number and the corresponding operations

mutable struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end

import Base:+,-,*,sin,^,/,cos
+(a::Dual,b::Dual)=Dual(a.v+b.v,a.∂+b.∂) 
+(a::Dual,b::Number)=Dual(a.v+b,a.∂) 
+(a::Number,b::Dual)=Dual(a+b.v,b.∂) 
-(a::Dual,b::Dual)=Dual(a.v-b.v,a.∂-b.∂) 
-(a::Number,b::Dual)=Dual(a-b.v,-b.∂) 
-(a::Dual,b::Number)=Dual( a.v-b,a.∂)
*(a::Dual,b::Dual)=Dual(a.v*b.v,a.v*b.∂+b.v*a.∂) 
*(a::Number,b::Dual)=Dual(a*b.v,a*b.∂) 
/(a::Dual,b::Dual)=Dual(a.v/b.v,(b.v*a.∂-a.v*b.∂)/(b.v)^2) 
/(a::Number,b::Dual)=Dual(a/b.v,(-a*b.∂)/(b.v)^2) 
#*(a::Float64,b::Dual)=Dual(a*b.v,a*b.∂) 
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂) 
sin(a::Dual)=Dual(sin(a.v),cos(a.v)*a.∂) 
cos(a::Dual)=Dual(cos(a.v),-sin(a.v)*a.∂)

f(r)=4π*r-2.0(1/r)*(1/r)

f(Dual(0.2,1.0))

r=Dual(1.0,1.0)

for i=1:50
        println(r)
        r=r-f(r).v/f(r).∂
end

# =====================================================
# Dual Number Example with Multivariate functions
# =====================================================
# We have to iterate using the following formula
# Define the vector function

VectorFunc(x)=[x[1]^2-x[2]+1,
                3*cos(x[1])-x[2]]

df1dx1=VectorFunc([Dual(1.0,1.0),Dual(1.0,0.0)])[1].∂

df1dx1=VectorFunc([Dual(1.0,1.0),Dual(1.0,0.0)])[2].∂
# To calculate the Jacobian

xvals=[1,1]
J=hcat(getfield.(VectorFunc([Dual(xvals[1],1.0),Dual(xvals[2],0.0)])[:],:∂),
getfield.(VectorFunc([Dual(xvals[1],0.0),Dual(xvals[1],1.0)])[:],:∂))

xvals=[1,1]
for i=1:10
    F=VectorFunc(xvals)
    J=hcat(getfield.(VectorFunc([Dual(xvals[1],1.0),Dual(xvals[2],0.0)])[:],:∂),
getfield.(VectorFunc([Dual(xvals[1],0.0),Dual(xvals[1],1.0)])[:],:∂))
    delta=-J\F
    xvals+=delta
    println(xvals)
end
# Let's see how we can use our own Dual mode Forward Mode automatic differentiation
# to solve a very easy regression problem.

N_SAMPLES=10
rng=Xoshiro(1) #my random number generator

xi=rand(rng,Uniform(-1,1),N_SAMPLES) #generate random values of x between -1 and 1
yi=rand(rng,Normal(2.3,0.1),N_SAMPLES)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1,1,0,5),
    backgroundcolor = :transparent)
scatter!(ax,xi,yi;label="Data",color=:black,markersize=15)
axislegend(ax;position = :rt)
# save("../figures/04p03ApplicationOfForwardMode.svg",fig)   # (figure-save disabled in study file)
display(fig)

S(a0)=0.5*sum((yi.-a0).^2)

fig = Figure()
a0vec=0:0.5:5
ax = Axis(fig[1, 1], xlabel = L"a_0", ylabel = L"S(a_0)",
    xlabelsize=20,
    ylabelsize=20,
    backgroundcolor = :transparent)
lines!(ax,a0vec,S.(a0vec),color=:black)
# save("../figures/04p03ApplicationOfForwardMode02.svg",fig)   # (figure-save disabled in study file)
display(fig)

a0=Dual(1.0,1.0)
S(a0).∂

a0=Dual(4.0,1.0)
S(a0).∂

a0=Dual(1.0,1.0)

α=0.01

for i=1:500
    a0.v=a0.v-α*S(a0).∂
    println(a0.v)
end
