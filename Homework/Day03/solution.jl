using Random
using Zygote
using Optimization
using CairoMakie
using OptimizationOptimisers

f(x) = x * exp(-x^2)

#question a: plot the function
xplot = -3:0.01:3

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",
    xlabelsize = 15,
    ylabelsize = 15,
    limits = (-3, 3, -0.6, 0.6))
lines!(ax, xplot, f.(xplot), color = :blue)
display(fig)

#from the plot there is ONE local minimum at x = -1/sqrt(2) ≈ -0.707
#and ONE local maximum at x = +1/sqrt(2) ≈ +0.707
#the function flattens to 0 as x goes to ±∞

#question b: optimization package
#Optimization.jl needs the input as a vector and the objective as (x,p)
x0 = [-2.0]
optf = OptimizationFunction((x, p) -> f(x[1]), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.1); maxiters = 200)

println("package answer: x = ", sol.u[1])
println("exact answer:   x = ", -1 / sqrt(2))

#question c: standard gradient descent by hand
#df/dx = exp(-x^2)*(1-2x^2) by product rule
∇f(x) = exp(-x^2) * (1 - 2 * x^2)

x = -2.0
α = 0.1
for i = 1:200
    global x -= α * ∇f(x)
end

println("hand GD answer: x = ", x)
println("difference with (b): ", abs(sol.u[1] - x))

#both give x ≈ -0.707 = -1/sqrt(2), they agree
#note: if the initial guess is to the RIGHT of the local maximum (e.g. x0 = 2.0),
#gradient descent runs off to +∞ where the function is flat and never finds the minimum
