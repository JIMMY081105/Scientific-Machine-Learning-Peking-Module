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
-(a::Dual,b::Number)=Dual(a.v-b,a.∂) 
*(a::Dual,b::Dual)=Dual(a.v*b.v,a.v*b.∂+b.v*a.∂) 
*(a::Number,b::Dual)=Dual(a*b.v,a*b.∂) 
/(a::Dual,b::Dual)=Dual(a.v/b.v,(b.v*a.∂-a.v*b.∂)/(b.v)^2) 
/(a::Number,b::Dual)=Dual(a/b.v,(-a*b.∂)/(b.v)^2) 
#*(a::Float64,b::Dual)=Dual(a*b.v,a*b.∂) 
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂) 
sin(a::Dual)=Dual(sin(a.v),cos(a.v)*a.∂) 
cos(a::Dual)=Dual(cos(a.v),-sin(a.v)*a.∂) 


f(r)=4*π*r-2.0*(1/r)*(1/r)
r=Dual(1.0,1.0)
f(r).∂

for i=1:10
    r-=f(r).v/f(r).∂
end

fvec(x)=[x[1]^2-x[2]+1,3*cos(x[1])-x[2]]

xvec=[Dual(1.0,1.0),Dual(2.0,0.0 )]
a=fvec(xvec)
xvec=[Dual(1.0,0.0),Dual(2.0,1.0 )]
b=fvec(xvec)

J=hcat(getfield.(VectorFunc([Dual(xvals[1],1.0),Dual(xvals[2],0.0)])[:],:∂),
getfield.(VectorFunc([Dual(xvals[1],0.0),Dual(xvals[1],1.0)])[:],:∂))
   