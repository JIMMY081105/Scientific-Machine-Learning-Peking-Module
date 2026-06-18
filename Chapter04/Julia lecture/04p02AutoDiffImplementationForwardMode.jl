########################################################################################
# 4.2 Implementation of Automatic Differentiation (Forward Mode)
########################################################################################
#
# PROBLEM
# Demystify automatic differentiation by building forward-mode AD yourself. The idea:
# break a function into elementary operations and carry each value together with its
# derivative (a 'Dual number'), applying the chain rule at every step.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Motivate with a worked table: decompose f(x) = 3x + sin(x^2) into steps v1..v5
#      and propagate the derivative dot(v) alongside each value.
#   2. Define a Dual struct with fields v (value) and d (derivative).
#   3. Overload the elementary operations (+, -, *, ^, sin, ...) on Dual so each
#      returns a new Dual with the correct chain-rule derivative.
#   4. Seed a variable as Dual(x, 1.0) and evaluate a function - the result's
#      derivative field IS df/dx, obtained just by running the function.
#   5. Plot the Dual-number derivative against the analytic one to confirm they
#      match.
#   6. Multivariate case: to get a partial derivative, seed the chosen variable with
#      dot = 1 and the others with dot = 0; note this needs one pass per input (the
#      limitation reverse mode fixes).
#
# This file is notebook "04p02AutoDiffImplementationForwardMode" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# ======================================================
# 4.2 Implementation of Automatic Differentiation.
# ======================================================

# ==========================
# Univariate functions
# ==========================

# TODO (Exercise Ex01): Go through the exercise above and calculate the value of df/dx(x=\pi). Herein
#   shows the main difference between symbolic and automatic differentiation. For
#   symbolic differentiation, you will get an expression for the derivative. In the
#   example above, the symbolic derivative is the expression 3+2x\cos(x^2). If you
#   want to calculate the derivative using symbolic differentiation, all you will
#   need to just substitute the numerical value of x and evaluate the expression. For
#   automatic differentiation, everytime you want to calculate the value of the
#   derivative (for a different value of x), you will need to run through the numbers
#   in the table above from top to bottom, i.e. calculate the numerical values v_i's
#   and \dot{v}_i's. For this simple problem above, you can easily perform symbolic
#   differentiation. However, when you have a complex function, such as a Neural
#   Network (see later in the course), it would not be possible to find the symbolic
#   expression for the derivative. For these complex functions, you should use
#   automatic differentiation.

using CairoMakie

struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end

import Base:+,-,*,sin,^
+(a::Dual,b::Dual)=Dual(a.v+b.v,a.∂+b.∂) 
+(a::Dual,b::Number)=Dual(a.v+b,a.∂) 
-(a::Dual,b::Dual)=Dual(a.v-b.v,a.∂-b.∂) 
*(a::Dual,b::Dual)=Dual(a.v*b.v,a.v*b.∂+b.v*a.∂) 
*(a::Number,b::Dual)=Dual(a*b.v,a*b.∂) 
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂) 
sin(a::Dual)=Dual(sin(a.v),cos(a.v)*a.∂) 

x=Dual(8.0,1.0)

x^2
# Let's try a more complicated example

x=Dual(π/2.0,1.0)  #the value of x=pi/2 and dx/dx=1.0
f(x::Dual)=3*x+sin(x^2) #define the function f(x)=3x+sin(x^2)

f(x)

g(x::Dual)=sin(x^2)+x^3

g(x)

π*cos(π^2/4)+3*(π/2)^2
# Now let's do another example

xplot=-1:0.001:1.0
f(x::Dual)=sin(x^2+3.0)+x^3
dfdx(x)=2x*cos(x^2+3)+3x^2
dfAuto=[f(Dual(x,1.0)).∂ for x in xplot]; #Automatic differentiation for all values of x in xplot
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"df/dx",title = "Derivative")
lines!(ax,xplot,dfAuto,color=:black,label="Automatic Differentiation")
scatter!(ax,xplot,dfdx.(xplot),color=:blue,label="Symbolic differentiation")
axislegend(ax,position=:rt)
fig
# The two lines fall on top of each other.

# ====================================
# Multivariable function example
# ====================================

# TODO (Exercise Ex02): Try calculating \frac{\partial f}{\partial x_2} (x_1=\pi/2,x_2=\pi/2). What
#   values should we use for \dot{v}_1 and \dot{v}_2?
# The Dual number approach also works with functions of more than 1 variable.
# Now we try to find the partial derivatives that make up the 2d function

quad(x,y)=(3.0/2.0)*x^2+2*x*y+3*y^2-2*x+8*y

xp=Dual(1.0,1.0)
yp=Dual(1.0,0.0)
quad(xp,yp)

xp=Dual(1.0,0.0)
yp=Dual(1.0,1.0)
quad(xp,yp)

# TODO (Exercise Ex03): What do you need to type in to find \frac{\partial f}{\partial y} at
#   (x,y)=(3.0,1.0)
