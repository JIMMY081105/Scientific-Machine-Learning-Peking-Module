using Random
using CairoMakie
using CSV
using DataFrames


data_path = joinpath(@__DIR__, "Datasets", "class 3", "Data02.csv")
data2 = CSV.read(data_path, DataFrame)

x_samples = data2.xdata
y_samples = data2.ydata

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(0,1,-10,10),
    backgroundcolor = :transparent)
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
axislegend(ax;position = :rt)

display(fig)
a=[1.0,1.0,1.0] #initial assumptions of the values of  a0, a1, a2.  Remember that the indexing in Julia
#fmodel starts here            # starts from 1, not 0. So a[1] is a0, a[2] is a1 and a[3] is a2.
xplot=-1:0.01:1
yplot=a[1].+a[2]*xplot.+a[3]*xplot.^2.0

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = L"x", 
    ylabel = L"y",
    xlabelsize=25,
    ylabelsize=25,
    limits=(-1,1,0,5))
scatter!(ax,x_samples,y_samples;label="Data",color=:black,markersize=15)
lines!(ax,xplot,yplot;label="Model",color=:blue)
axislegend(ax;position = :lt)
save("../figures/03p01LinearLeastSquaresRegression01.png",fig)
display(fig)


V=hcat(ones(N_SAMPLES),x_samples,x_samples.^2)
a=V'V\(V'y_samples)