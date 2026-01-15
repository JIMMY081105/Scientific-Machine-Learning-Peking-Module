using CairoMakie
using ForwardDiff
using Zygote
using Random
using Distributions

N_SAMPLES=50 #50 data points
rng=Xoshiro(1) #my random number generator

x_samples=rand(rng,Uniform(0,5),N_SAMPLES) #generate random values of x between -1 and 1
y_noise=rand(rng,Normal(0.0,0.1),N_SAMPLES) #generate some noise, normal distribution with standard deviation of 0.1
y_samples=4*(1.0.-exp.(-1.8*x_samples))+y_noise;

x_samples

fmodel(a,x)=a[1]*(1.0.-exp.(a[2]*x)); # for this model a[1]=a and a[2]=b.
function custom_loss(parameters,x_samples)
           ŷ = fmodel(parameters, x_samples)
           0.5*sum((y_samples .- ŷ).^2)
end
a=[1.0,-1.0]
custom_loss(a,x_samples)

S=(a)->custom_loss(a,x_samples)
Zygote.gradient(S, a)[1]

α=0.01 #learning rate
for i=1:500
    a-=α * Zygote.gradient(S,a)[1]
    println(a)
end