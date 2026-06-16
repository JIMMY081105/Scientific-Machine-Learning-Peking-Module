using CairoMakie

# Since the volume is 1 m^3, the surface area of the cylinder works out to
#   A(r) = 2*pi*r^2 + 2/r
A(r)  = 2*π*r^2 + 2/r
dA(r) = 4*π*r - 2/r^2      # derivative dA/dr. NOTE: use dA, not A' ('  means transpose
                          # in Julia). And d/dr(2/r) = -2/r^2, not -2/r^3.

range_r = 0.1:0.1:5       # start at 0.1, because A(0) = 2/0 = Inf
fig = Figure()
ax  = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)")
lines!(ax, range_r, A.(range_r); color=:black)   # use range_r (you had rplot, undefined)
display(fig)              # your figure is `fig`, not the type `Figure`

r=(1/(2*π))^(1/3)

dA(r)

A(r,V)=2π*r^2+2*V/r # Define function A(r,V)
Vplot=0.5:0.5:2
rplot=0.1:0.01:2 #define the range of the plot
fig = Figure() #set up a figure
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)") #Define axis in the figure
for i=1:length(Vplot)
    lines!(ax,rplot,A.(rplot,Vplot[i]),label=L"V=%$(Vplot[i])") 
end
axislegend(ax,position=:rt) #add legend to the axis
fig #show the figure