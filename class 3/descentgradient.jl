using Optimization,OptimizationOptimisers,Zygote,CairoMakie


#fundamental of machine learning


f(x)=2*π*x^2+2/x
#claculating df is the hardest problem in complex funciton 
d(x)=4*π*x-2/x^2

x=1.8

α=0.2

for i=1:5000
    x = x -α*d(x)
end


