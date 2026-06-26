using CairoMakie
using Zygote
using DataFrames
using CSV

df = CSV.read(joinpath(@__DIR__, "..", "Data04.csv"), DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]


fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
scatter!(ax,xdata,ydata,color=:black,label="data")
axislegend(ax,position=:ct)
display(fig)

function NN01(x,W1,W2,b1,b2)
    return W1 .* tanh.(W2 .* x .+ b1) .+ b2
end

function Loss(W1,W2,b1,b2)
    return sum((ydata .- NN01(xdata,W1,W2,b1,b2)).^2)
end

α=0.001
W1=-1.0
W2=1.0
b1=0.0
b2=0.0
for i=1:5000
    grads=Zygote.gradient(Loss,W1,W2,b1,b2)
    W1-=α*grads[1]
    W2-=α*grads[2]
    b1-=α*grads[3]
    b2-=α*grads[4]
    println(Loss(W1,W2,b1,b2))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",title="Neural Network")
scatter!(ax,xdata,ydata,color=:black,label="data")
lines!(ax,xdata,NN01(xdata,W1,W2,b1,b2);color=:blue,linewidth=2,label="Neural Network")
axislegend(ax,position=:rt)
fig

Zygote.gradient(Loss,W1,W2,b1,b2)