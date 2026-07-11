using CairoMakie

A(r)=2π*r^2+2/r # Define function A(r)
rplot=0.1:0.01:2 #define the range of the plot
fig = Figure() #set up a figure
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)") #Define axis in the figure
lines!(ax,rplot,A.(rplot);color=:black) #Plot line
save("../figures/01p01IntroductionAndMotivation01.svg", fig) #save figure as svg file
fig #show the figure


A(r,V)=2π*r^2+2*V/r # Define function A(r,V)
Vplot=0.5:0.5:2
rplot=0.1:0.01:2 #define the range of the plot
fig = Figure() #set up a figure
ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"A(r)") #Define axis in the figure
for i=1:length(Vplot)
    lines!(ax,rplot,A.(rplot,Vplot[i]),label=L"V=%$(Vplot[i])") 
end
axislegend(ax,position=:rt) #add legend to the axis
save("../figures/01p01IntroductionAndMotivation02.svg", fig) #save figure as svg file
fig #show the figure
