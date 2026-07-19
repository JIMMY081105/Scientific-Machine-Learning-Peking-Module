using Optimization, OptimizationOptimisers, Zygote, CairoMakie

f(x) = x * exp(-x^2)          # function to be optimized
∇f(x) = exp(-x^2) * (1 - 2x^2) # analytical gradient df/dx (used in part c)

# struct to store the intermediate solutions during the iterations
struct Point1D
    x::Float64      # guess location of the minimum
    fvalue::Float64 # value of the function at the guess location
end

xplot = -3:0.01:3
yplot = [f(x) for x in xplot]

fig = Figure()
ax = Axis(fig[1, 1],
    xlabel = L"x",
    ylabel = L"f(x)",
    xlabelsize = 20,
    ylabelsize = 20,
    title = L"f(x) = x e^{-x^2}",
    limits = (-3, 3, -0.6, 0.6))
lines!(ax, xplot, yplot)

# Setting f'(x) = exp(-x^2)*(1-2x^2) = 0 gives x = ±1/sqrt(2)
scatter!(ax, [-1/sqrt(2)], [f(-1/sqrt(2))]; color = :red, markersize = 15,
    label = "local minimum")
scatter!(ax, [1/sqrt(2)], [f(1/sqrt(2))]; color = :green, markersize = 15,
    label = "local maximum")
axislegend(ax; position = :rb)
save(joinpath(@__DIR__, "one_dimensional_optimization_a.png"), fig)
display(fig)

# part a:
# From the plot, f(x) has exactly ONE local minimum and ONE local maximum.
#   - one local minimum at x = -1/sqrt(2) ≈ -0.7071 where f ≈ -0.4289
#   - one local maximum at x = +1/sqrt(2) ≈ +0.7071 where f ≈ +0.4289
# Away from these two turning points the function flattens out and
# approaches 0 as x -> ±∞.


# Note that Optimization.jl expects the input of the function to be a vector,
# even if the function is univariate, and the objective must take two
# arguments (x,p) where p are (constant) parameters (not used here).
x0 = -2.0 * ones(1) # initial guess

optf = OptimizationFunction((x, p) -> f(x[1]), ADTypes.AutoZygote())
prob = OptimizationProblem(optf, x0)
sol = solve(prob, Optimisers.Descent(0.1); maxiters = 200)

println("Part b) Optimization.jl with gradient descent, Optimisers.Descent(0.1)")
println("   local minimum found at x = ", sol.u[1])
println("   objective value  f(x)    = ", sol.objective)
println("   exact answer     x       = ", -1/sqrt(2))
println()


# The update equation for the standard gradient descent is
#       x^(k+1) = x^(k) - alpha*g^(k),   g^(k) = df/dx evaluated at x^(k)
x0c = -2.0   # initial guess (same as part b so the answers can be compared)
alpha = 0.1  # learning rate
niter = 200  # number of iterations

data = [Point1D(x0c, f(x0c))] # store the initial guess and function value
x = x0c
for i in 1:niter
    global x -= alpha * ∇f(x)          # standard gradient descent update
    push!(data, Point1D(x, f(x)))      # store x and f(x)
end

println("Part c) hand written standard gradient descent")
println("   local minimum found at x = ", x)
println("   objective value  f(x)    = ", f(x))
println()
println("Check: |x_partb - x_partc| = ", abs(sol.u[1] - x))

# Plot the convergence of x^(k)
fig_conv = Figure(size = (1000, 400))
ax_conv1 = Axis(fig_conv[1, 1], xlabel = L"x", ylabel = L"f(x)",
    limits = (-3, 3, -0.6, 0.6))
lines!(ax_conv1, xplot, yplot)
scatter!(ax_conv1, getproperty.(data, :x), getproperty.(data, :fvalue);
    color = 1:length(data),
    strokewidth = 1,
    strokecolor = :black,
    colormap = :Blues,
    markersize = 15)

ax_conv2 = Axis(fig_conv[1, 2], xlabel = "iter", ylabel = L"x")
lines!(ax_conv2, getproperty.(data, :x), label = "Gradient Descent")
scatter!(ax_conv2, getproperty.(data, :x);
    color = 1:length(data),
    strokewidth = 1,
    strokecolor = :black,
    colormap = :Blues,
    markersize = 15)
hlines!(ax_conv2, -1/sqrt(2); color = :red, linestyle = :dash)
axislegend(ax_conv2; position = :rt)
save(joinpath(@__DIR__, "one_dimensional_optimization_c.png"), fig_conv)
display(fig_conv)

# part c:
# Starting from x0 = -2.0 with learning rate alpha = 0.1, the hand written
# gradient descent converges to x ≈ -0.7071 = -1/sqrt(2), which agrees with
# the answer obtained with the Optimization.jl package in part b) and with
# the exact analytical answer.
#
# because f(x) flattens out for
# large |x| (the gradient goes to zero) and has a local MAXIMUM at
# x = +1/sqrt(2), gradient descent only converges to the minimum if the
# initial guess is to the left of the local maximum. If we start at, say,
# x0 = 2.0, the method pushes x to larger and larger values where the
# function is flat and never finds the minimum.
