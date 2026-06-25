using CSV
using DataFrames
using CairoMakie
using Zygote

df = CSV.read(joinpath(@__DIR__, "..", "Data03.csv"), DataFrame;
select=["xdata", "ydata"])
xdata=df[:, :xdata]
ydata=df[:, :ydata]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)")
scatter!(ax,xdata,ydata;color=:black,label="Data")
fig

function NN03(x,W,b)
    return W[2]*tanh.(W[1]*x.+b[1]).+b[2]
end

function Loss03(xdata,W,b)
    return 0.5*sum((ydata-NN03(xdata,W,b)).^2)
end

ff03=(W,b)->Loss03(xdata,W,b)


α=0.01
W=[1,1]
b=[1,1]

function GradientDescent(W,b)
    for i=1:1000
        grads=Zygote.gradient(ff03,W,b)
        W-=α*grads[1]
        b-=α*grads[2]

    #println(ff03(W,b))
    end
    return W,b
end

W,b=GradientDescent(W,b)



fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
scatter!(ax,xdata,ydata;label="Data",color=:black)
lines!(ax,xdata,NN03(xdata,W,b);label="Model",color=:blue)
axislegend(ax;position = :lt)
fig

#n=length(xdata)
#V=hcat(ones(n),xdata,xdata.^2,xdata.^3)
#a=V'V\(V'ydata)

#fmodel(a,x)=a[1]*x.^0+a[2].*x+a[3].*x.^2+a[4].*x.^3;
#fig = Figure()
#xplot=-1:0.01:1
#ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y")
#scatter!(ax,xdata,ydata;label="Data",color=:black)
#lines!(ax,xdata,fmodel(a,xdata);label="Model",color=:blue)
#axislegend(ax;position = :lt)
#fig