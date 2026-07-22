using ForwardDiff

# Homework Day 6
# Differentiate f(x)=exp(-x^2)cos(x) using three methods

f(x)=exp(-x^2)*cos(x)

# a) Analytical differentiation
# Using the product rule and chain rule:
# df/dx=exp(-x^2)*(-2x*cos(x)-sin(x))
dfdx(x)=exp(-x^2)*(-2*x*cos(x)-sin(x))


# b) ForwardDiff package
dfForwardDiff(x)=ForwardDiff.derivative(f,x)


# c) Dual numbers introduced in lectures
struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end

# Only define the operators needed to evaluate exp(-x^2)*cos(x)
import Base:-,*,^,exp,cos
-(a::Dual)=Dual(-a.v,-a.∂)
*(a::Dual,b::Dual)=Dual(a.v*b.v,a.v*b.∂+b.v*a.∂)
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂)
exp(a::Dual)=Dual(exp(a.v),exp(a.v)*a.∂)
cos(a::Dual)=Dual(cos(a.v),-sin(a.v)*a.∂)


# Compare the three methods at x=2.0
x=2.0
analytical_answer=dfdx(x)
forwarddiff_answer=dfForwardDiff(x)
dual_answer=f(Dual(x,1.0)).∂

println("At x=$x:")
println("Analytical derivative = $analytical_answer")
println("ForwardDiff derivative = $forwarddiff_answer")
println("Dual number derivative = $dual_answer")
