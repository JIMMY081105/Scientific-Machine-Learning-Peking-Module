using CairoMakie
using Zygote

f(x1, x2) = exp(−x1^2)*cos(3*x2)

#df/dx1 with respect to x1 hence only need to foces on exp
#df/dx1 = -2x1*exp(-x1^2)*cos(3*x2)

#df/dx2 with respect to x2 hence only need to foces on cos
#df/dx2 = -3*e^(-x1^2)*sin(3*x2)

FunctionJacobian(x1,x2) = -2x1*exp(-x1^2)*cos(3*x2),
                        -3*e^(-x1^2)*sin(3*x2)

#results for a is approximately (0.06676, -0.02265)

G = Zygote.gradient(f,2.0,3.0)

#results here is (0.06675173142911019, -0.02264464006518042)

#define the structure of the dual 
mutable struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end

#import the right ones that are needed
import Base:-,*,^,exp,cos

#exp
exp(a::Dual)=Dual(exp(a.v),exp(a.v)*a.∂)
#substraction, use this for both duals
-(a::Dual) = Dual(-a.v, -a.∂) 
#power
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂) 
#cosine 
cos(a::Dual)=Dual(cos(a.v),-sin(a.v)*a.∂)
# 3*x2
*(a::Number, b::Dual) =Dual(a*b.v, a*b.∂)
# exp(...) * cos(...)
*(a::Dual, b::Dual) =Dual(a.v*b.v, a.v*b.∂ + b.v*a.∂)

x1 = Dual(2.0,1.0)
x2 = Dual(3.0,0.0)

exp(−x1^2)*cos(3*x2)
#the value here for f(x1,x2) is and df/dx1 is (-0.016687932857277547, 0.06675173142911019)

x1 = Dual(2.0,0.0)
x2 = Dual(3.0,1.0)

exp(−x1^2)*cos(3*x2)
#the value here for f(x1,x2) is and df/dx2 is (-0.016687932857277547, -0.02264464006518042)