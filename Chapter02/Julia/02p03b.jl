f(x)=2*π*x^2+2/x;  

df(x)=4*π*x-2/x^2

α=0.01
β=0.8
x=1.8
v=0
for i=1:50
    v=β*v-α*df(x)
    x+=v
end