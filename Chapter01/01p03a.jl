
using CairoMakie
A(r)=2*π*r^2+2/r
rplot=0.1:0.01:2.0
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)")
lines!(ax,rplot,A.(rplot))
display(fig)
