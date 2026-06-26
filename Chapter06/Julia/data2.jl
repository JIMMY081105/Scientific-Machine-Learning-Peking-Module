using CairoMakie
using Zygote
using DataFrames
using CSV

df = CSV.read(joinpath(@__DIR__, "..", "Data02.csv"), DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
scatter!(ax,xdata,ydata,color=:black,label="data")
axislegend(ax,position=:ct)
display(fig)

function NN01(x,W,b)
    return sin.(W*x.+b)
end

function Loss(xdata,W,b)
    return sum((ydata-NN01(xdata,W,b)).^2)
end

#u take the thing that only u want to optimize which is w and b here 
ff01=(W,b)->Loss(xdata,W,b)

Zygote.gradient(ff01,1,1)

α=0.01
W=15
b=15
for i=1:100
    grads=Zygote.gradient(ff01,W,b)
    W-=α*grads[1]
    b-=α*grads[2]
    println(ff01(W,b))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",title="Neural Network")
scatter!(ax,xdata,ydata,color=:black,label="data")
lines!(ax,xdata,NN01(xdata,W,b);color=:blue,linewidth=1,label="Neural Network")
fig