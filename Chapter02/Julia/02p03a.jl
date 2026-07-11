f(x)=2*π*x^2+2/x;  

df(x)=4*π*x-2/x^2

α=0.001
x=1.8

for i=1:10
    x-=α*df(x)
end