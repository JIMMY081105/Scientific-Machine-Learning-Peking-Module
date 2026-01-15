struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end

x=Dual(1,1)
x.∂

import Base:^,sin,*,+
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂) 
sin(a::Dual)=Dual(sin(a.v),cos(a.v)*a.∂)
*(a::Number,b::Dual)=Dual(a*b.v,a*b.∂)
+(a::Dual,b::Dual)=Dual(a.v+b.v,a.∂+b.∂)  

x^2

x^3

sin(x^2)

3*x

3*x+sin(x^2)