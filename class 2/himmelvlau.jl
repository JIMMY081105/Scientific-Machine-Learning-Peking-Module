using CairoMakie


himmelblau(x)=(x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2 #the input of himmelbl
x1range=-6:0.02:6;
x2range=-6:0.02:6;
funcplot = [himmelblau([x1,x2]) for x1 in x1range, x2 in x2range]
fig = Figure()
ax = Axis(fig[1, 1],
 xlabel = L"x_1",
 ylabel = L"x_2",
 xlabelsize = 20,
 ylabelsize = 20,
 title = "Himmelblau function",
 limits=(-6,6.0,-6.0,6.0))
levels = 10.0.^range(0, 3.5; length=10)
contourf!(ax,x1range,x2range,funcplot; levels,colormap=:bwr)
contour!(ax,x1range,x2range,funcplot; levels,label=true,color=:black)
scatter!(ax,-0.270845,-0.923039;markersize=20,color=:magenta)
scatter!(ax,3.0,2.0,markersize=20,color=:grey)
scatter!(ax,-2.805118,3.13132,markersize=20,color=:grey)

output_path = joinpath("docs", "assets", "himmelblau-contours.svg")
mkpath(dirname(output_path))
save(output_path, fig)
