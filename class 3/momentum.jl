f(x)=2*π*x^2+2/x
#claculating df is the hardest problem in complex funciton 
d(x)=4*π*x-2/x^2

x=1.8
β=0.8
v=0
α=0.2

for i=1:5000
    v=β*v-α*d(x)
    x+=v
end
