using ForwardDiff
using Zygote

f(x)=x^2*exp(-x^2)+cos(x^3)
f(x)=x^2
ForwardDiff.derivative(f,2.0)
a=Zygote.gradient(f,2)
typeof(a[1])

A(x)=4*π*x-2/x^2
x=1.0
for i=1:100
    x-=A(x)/ForwardDiff.derivative(A,x)
end


himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2

ForwardDiff.gradient(himmelblau,[1,2])

Zygote.gradient(himmelblau,[1,2])

xvec=[1,1]
Jimmy=0.01
for i=1:100
    xvec-=Jimmy*ForwardDiff.gradient(himmelblau,xvec)
    println(xvec)
end

NN(x,W,b)=sin(W*x+b)
Zygote.gradient(NN,1,2,3)

ForwardDiff.gradient(NN,1,2,3)[3]
Lukas(x)=sin(x[2]*x[1]+x[3])
ForwardDiff.gradient(Lukas,[1,2,3])
