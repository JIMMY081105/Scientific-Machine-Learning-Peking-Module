using CSV
using DataFrames
using CairoMakie
using Zygote

df = CSV.read(joinpath(@__DIR__, "..", "Data02.csv"), DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
scatter!(ax,xdata,ydata;color=:black,label="Data")
fig

function NN01(x,W,b)
    return sin.(W*x.+b)
end

function Loss01(xdata,W,b)
    return 0.5*sum((ydata-NN01(xdata,W,b)).^2)
end

Loss01(xdata,1,1)

ff01=(W,b)->Loss01(xdata,W,b)

Zygote.gradient(ff01,1,1)
α=0.01

W=15
b=1
for i=1:50
    grads=Zygote.gradient(ff01,W,b)
    W-=α*grads[1]
    b-=α*grads[2]
    println(ff01(W,b))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax,xdata,ydata;label="Data",color=:black)
lines!(ax,xdata,NN01(xdata,W,b);label="Model",color=:blue)
axislegend(ax;position = :lt)
fig

n=length(xdata)
V=hcat(ones(n),xdata,xdata.^2,xdata.^3)
a=V'V\(V'ydata)

fmodel(a,x)=a[1]*x.^0+a[2].*x+a[3].*x.^2+a[4].*x.^3;
fig = Figure()
xplot=-1:0.01:1
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax,xdata,ydata;label="Data",color=:black)
lines!(ax,xdata,fmodel(a,xdata);label="Model",color=:blue)
axislegend(ax;position = :lt)
fig