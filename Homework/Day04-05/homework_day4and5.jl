using Random
using Zygote
using Optimization
using CairoMakie
using OptimizationOptimisers


#HERE IS QUESTION A


#number of samples i need
num_of_samples = 70
#random number generator
ran_num_gen = Xoshiro(1)

#random generation of x between 0 and 2pi
x_generation = sort(2pi.*rand(ran_num_gen,num_of_samples))
#make noise with a standard deviation of 0.1
y_make_noise = 0.1.*randn(ran_num_gen,num_of_samples)
#the y axis generation follows the function sin(x) with noise
y_generation = sin.(x_generation) .+ y_make_noise


#HERE IS QUESTION B


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(0,2pi,-1.5,1.5),
    backgroundcolor = :transparent)
scatter!(ax,x_generation,y_generation;label="Data",color=:black,markersize=15)
axislegend(ax;position = :rt)
output_dir = joinpath(dirname(@__DIR__), "figures")
mkpath(output_dir)
save(joinpath(output_dir, "homework_day4and5_data.png"), fig)
display(fig)


#HERE IS QUESTION C


#define model with the corresponding coefficients
Regmodel(a,x) = a[1] .+ a[2].*x .+ a[3].*x.^2 .+ a[4].*x.^3

#mean squared error between the generated data and the model
S(a,p) = sum((Regmodel(a,x_generation) .- y_generation).^2)/num_of_samples

#initial values of a0, a1, a2 and a3
c = zeros(4)

optf = OptimizationFunction(S, ADTypes.AutoZygote())
prob = OptimizationProblem(optf,c)
sol_descent = solve(prob,Optimisers.Descent(0.00001);maxiters=50_000)

#plot the polynomial obtained using standard gradient descent
x_plot = range(0,2pi,length=500)

fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(0,2pi,-1.5,1.5),
    backgroundcolor = :transparent)
scatter!(ax2,x_generation,y_generation;label="Data",color=:black,markersize=12)
lines!(ax2,x_plot,Regmodel(sol_descent.u,x_plot);label="Gradient descent",color=:blue,linewidth=3)
axislegend(ax2;position = :rt)
save(joinpath(output_dir, "homework_day4and5_gradient_descent.png"), fig2)
display(fig2)


#HERE IS QUESTION D


#try momentum and Adam with different hyperparameters
sol_momentum = solve(prob,Optimisers.Momentum(0.00001,0.9);maxiters=50_000)
sol_adam = solve(prob,Optimisers.Adam(0.01,(0.9,0.999));maxiters=50_000)

#compare the fitted models and their final errors
descent_error = S(sol_descent.u,nothing)
momentum_error = S(sol_momentum.u,nothing)
adam_error = S(sol_adam.u,nothing)

println("Gradient descent coefficients = ",sol_descent.u)
println("Momentum coefficients = ",sol_momentum.u)
println("Adam coefficients = ",sol_adam.u)
println("Gradient descent mean squared error = ",descent_error)
println("Momentum mean squared error = ",momentum_error)
println("Adam mean squared error = ",adam_error)

fig3 = Figure()
ax3 = Axis(fig3[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(0,2pi,-1.5,1.5),
    backgroundcolor = :transparent)
scatter!(ax3,x_generation,y_generation;label="Data",color=:black,markersize=12)
lines!(ax3,x_plot,Regmodel(sol_descent.u,x_plot);label="Gradient descent",color=:blue,linewidth=3)
lines!(ax3,x_plot,Regmodel(sol_momentum.u,x_plot);label="Momentum",color=:green,linewidth=3,linestyle=:dash)
lines!(ax3,x_plot,Regmodel(sol_adam.u,x_plot);label="Adam",color=:red,linewidth=3,linestyle=:dot)
axislegend(ax3;position = :rt)
save(joinpath(output_dir, "homework_day4and5_optimisers.png"), fig3)
display(fig3)
