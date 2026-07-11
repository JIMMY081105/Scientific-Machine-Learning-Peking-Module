A(x)=2*π*x^2+2/x
∇A(x)=4*π*x-2/x^2;
Hessianh(x)=4*π+4/x^3;

x0=10.0 #initial guess of x0
x=x0
for i in 1 : 5
    x-=Hessianh(x)\∇A(x)
    println(x)
end
