using Random
using Zygote
using Optimization
using CairoMakie
using Distributions
using Enzyme
using OptimizationOptimisers


#HERE IS QUESTION A


#number of samples i need 
num_of_samples = 70
#random number generator
ran_num_gen=Xoshiro(1)

#random generation of x between -2 and 2 
x_generation = rand(ran_num_gen,Uniform(-2,2),num_of_samples)
#make noises of a difference 0.1
y_make_noise = rand(ran_num_gen,Normal(-0.1,0.1),num_of_samples)
#the y axis generation follows the function sin(x) with noise
y_generation = sin.(x_generation) .+ y_make_noise


#HERE IS QUESTION B


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-2,2,-1,1),
    backgroundcolor = :transparent)
scatter!(ax,x_generation,y_generation;label="Data",color=:black,markersize=15)
axislegend(ax;position = :rt)
output_dir = joinpath(dirname(@__DIR__), "figures")
mkpath(output_dir)
save(joinpath(output_dir, "03p01LinearLeastSquaresGradientDescent01.png"), fig)
display(fig)






