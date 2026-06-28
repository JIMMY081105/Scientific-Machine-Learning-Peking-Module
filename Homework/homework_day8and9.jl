using NonlinearSolve
using CairoMakie

f(x, b) = x^3 * exp(-(x / b)^2) - 1.0


xplot = 0:0.02:14
bvalues = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x;b)",
    xlabelsize = 25,
    ylabelsize = 25,
    limits = (0, 14, -1.5, 15))
for b in bvalues
    lines!(ax, xplot, [f(x, b) for x in xplot]; label = "b = $b")
end
lines!(ax, xplot, zeros(length(xplot)); color = :black) 
axislegend(ax, position = :rt)
display(fig)


x = 10.0
g(b)    = f(x, b)                             
dgdb(b) = x^3 * exp(-(x / b)^2) * (2 * x^2 / b^3) 

b = 3.5
for i = 1:10
    global b = b - g(b) / dgdb(b)
    println(b)
end

10 / sqrt(log(1000))


b0 = 3.5 
p = []     
prob = NonlinearProblem((b, p) -> f(10.0, b), b0, p)   # solve f(10; b) = 0 for b
sol = NonlinearSolve.solve(prob)

sol.u
