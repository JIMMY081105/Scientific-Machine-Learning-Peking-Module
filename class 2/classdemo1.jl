using CairoMakie


A(r) = 2pi * r^2 + 2 / r
rplot = -0.1:0.01:2.0
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)", limits = (0, 2, 0.0, 25))
lines!(ax, rplot, A.(rplot), color = :red)
display(fig)
