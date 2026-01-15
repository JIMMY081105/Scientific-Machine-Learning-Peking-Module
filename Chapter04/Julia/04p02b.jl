struct Dual <: Number
    v::Float64 #value of the function
    ∂::Float64 #its derivative
end



import Base:^,sin,*,+
^(a::Dual,n::Integer)=Dual(a.v^n,n*a.v^(n-1)*a.∂) 
sin(a::Dual)=Dual(sin(a.v),cos(a.v)*a.∂)
*(a::Number,b::Dual)=Dual(a*b.v,a*b.∂)
+(a::Dual,b::Dual)=Dual(a.v+b.v,a.∂+b.∂)  

x=Dual(2,1 )
y=Dual(2, 0)

g(x,y)=x^2+y^3

g(x,y)

x=Dual(1,0)
y=Dual(2,1)