using CairoMakie
using LinearAlgebra
using LinearSolve
using SparseArrays
using CSV
using DataFrames

df = CSV.read(joinpath(@__DIR__, "..", "Data05.csv"), DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y(x)")
scatter!(ax,xdata,ydata,color=:black,label="data")
axislegend(ax,position=:ct)
display(fig)

N_SAMPLES=length(xdata)
xplot=-1:0.01:1

V=hcat(ones(N_SAMPLES),xdata,xdata.^2,xdata.^3)
a=V'V\(V'ydata)

fmodel(a,x)=a[1]*x.^0+a[2].*x.+a[3].*x.^2+a[4].*x.^3;

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax,xdata,ydata;label="Data",color=:black)
lines!(ax,xplot,fmodel(a,xplot);label="Model",color=:blue)
axislegend(ax;position = :ct)
display(fig)

S=sum((ydata.-fmodel(a,xdata)).^2)
V4=hcat(ones(N_SAMPLES),xdata,xdata.^2,xdata.^3,xdata.^4)
a4=V4'V4\(V4'ydata)

fmodel4(a,x)=a[1]*x.^0+a[2].*x.+a[3].*x.^2+a[4].*x.^3+a[5].*x.^4;

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",limits=(0,1,-0.05,0.0))
scatter!(ax,xdata,ydata;label="Data",color=:black)
lines!(ax,xplot,fmodel4(a4,xplot);label="Model",color=:blue)
axislegend(ax;position = :ct)
display(fig)

S4=sum((ydata.-fmodel4(a4,xdata)).^2)
