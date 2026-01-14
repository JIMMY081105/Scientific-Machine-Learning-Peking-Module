using CairoMakie
using CSV
using DataFrames
using LinearAlgebra
using LinearSolve
using NonlinearSolve

#Equation 1:
y1top(x) = 5 + sqrt(8 - (x + 1)^2)
y1bot(x) = 5 - sqrt(8 - (x + 1)^2)
x1 = range(-1 - sqrt(8) + 0.001, -1 + sqrt(8) - 0.001, length = 400)

#Equation 2:
y2top(x) = 4 + sqrt(-x^3)
y2bot(x) = 4 - sqrt(-x^3)
x2 = range(-5, 0, length = 400)

fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "x",
    ylabel = "y",
    title = "Solutions of the Two Equations",
    limits = (-5, 2, 0, 9),
    xticks = -5:0.5:2,
    yticks = 0:0.5:9
)

lines!(ax, x1, y1top.(x1), color = :blue, label = "Equation 1")
lines!(ax, x1, y1bot.(x1), color = :blue)

lines!(ax, x2, y2top.(x2), color = :green, label = "Equation 2")
lines!(ax, x2, y2bot.(x2), color = :green)

axislegend(ax)

display(fig)

#the 2 intersection points looks like 
#(-1.5,2.5)
#(-2.25,7.5)

function uandv(x,y)
    return [(x+1)^2+(y-5)^2-8,(y-4)^2+x^3]
end

# Jacobian: 2x2 matrix of partial derivatives
# row 1: partial derivatives of u with u = (x+1)^2 + (y-5)^2 - 8
# row 2: partial derivatives of v with v = (y-4)^2 + x^3
function Jacobian(x,y)
    return [2*(x+1) 2*(y-5); 3*x^2 2*(y-4)]
end

#Newton-Raphson from each initial guess found in part b)
for (x0, y0) in [(-1.5, 2.5), (-2.25, 7.5)]
    x = x0
    y = y0
    println("Newton-Raphson starting from ($x0, $y0):")
    for i = 1:10
        J = Jacobian(x, y)
        F = uandv(x, y)
        delta = -J\F
        x += delta[1]
        y += delta[2]
        println("  i=$i: x=$x, y=$y")
    end
end

#Check the answers with the NonlinearSolve package
function FunctionToSolve(x)
    return [(x[1]+1)^2 + (x[2]-5)^2 - 8, (x[2]-4)^2 + x[1]^3]
end

for x0 in ([-1.5, 2.5], [-2.25, 7.5])
    prob = NonlinearProblem((x, p) -> FunctionToSolve(x), x0, [])
    sol = NonlinearSolve.solve(prob)
    println("NonlinearSolve check from $x0: x=$(sol.u[1]), y=$(sol.u[2])")
    scatter!(ax, sol.u[1], sol.u[2]; color = :red, markersize = 15)
end

display(fig)

# Question 2

struct Point2D
    x1::Float64
    x2::Float64
    fvalue::Float64
end

# a) Define and plot the function
f(x)=-x[1]*x[2]*exp(-(x[1]^2+x[2]^2)/2)

function Gradient_f(x)
    d1=x[2]*(x[1]^2-1)*exp(-(x[1]^2+x[2]^2)/2) #df/dx1
    d2=x[1]*(x[2]^2-1)*exp(-(x[1]^2+x[2]^2)/2) #df/dx2
    return [d1,d2]
end

x1range=-3:0.02:3;
x2range=-3:0.02:3;
funcplot=[f([x1,x2]) for x1 in x1range, x2 in x2range]

fig2=Figure()
ax2=Axis(fig2[1,1],xlabel=L"x_1",ylabel=L"x_2",
    title="Contour plot of f(x1,x2)",limits=(-3,3,-3,3))
contourf!(ax2,x1range,x2range,funcplot;levels=50,colormap=:bwr)
contour!(ax2,x1range,x2range,funcplot;labels=true,levels=15,color=:black)
display(fig2)

# b) From the contour plot, the local minima are approximately
# (1,1) and (-1,-1).
scatter!(ax2,[1,-1],[1,-1],color=:black,markersize=20)
display(fig2)

# c) Standard gradient descent from an initial guess near (1,1)
x0=[0.5,0.5] #initial guess
alpha=0.1 #learning rate
data=[Point2D(x0...,f(x0))]
x=x0

for i in 1:100
    global x-=alpha*Gradient_f(x) #standard gradient descent update equation
    push!(data,Point2D(x...,f(x)))
end

println("Local minimum from $x0 is approximately $x")
lines!(ax2,getproperty.(data,:x1),getproperty.(data,:x2);
    linewidth=3,color=:grey,label="Gradient Descent")
scatter!(ax2,getproperty.(data,:x1),getproperty.(data,:x2);
    color=:grey,markersize=10,strokecolor=:black,strokewidth=1)

# Use another initial guess to find the second local minimum
x0=[-0.5,-0.5]
data2=[Point2D(x0...,f(x0))]
x=x0

for i in 1:100
    global x-=alpha*Gradient_f(x)
    push!(data2,Point2D(x...,f(x)))
end

println("Local minimum from $x0 is approximately $x")
lines!(ax2,getproperty.(data2,:x1),getproperty.(data2,:x2);
    linewidth=3,color=:green,label="Gradient Descent")
scatter!(ax2,getproperty.(data2,:x1),getproperty.(data2,:x2);
    color=:green,markersize=10,strokecolor=:black,strokewidth=1)
axislegend(ax2;position=:lt)
display(fig2)

# d) Try the required initial guess x0=(-2,2)
x0=[-2.0,2.0]
data3=[Point2D(x0...,f(x0))]
x=x0

for i in 1:100
    global x-=alpha*Gradient_f(x)
    push!(data3,Point2D(x...,f(x)))
end

println("Starting from $x0, the final point after 100 iterations is $x")
println("No local minimum is found because the points move away from the origin and f(x) approaches zero.")

lines!(ax2,getproperty.(data3,:x1),getproperty.(data3,:x2);
    linewidth=3,color=:orange,label="Start at (-2,2)")
scatter!(ax2,getproperty.(data3,:x1),getproperty.(data3,:x2);
    color=:orange,markersize=10,strokecolor=:black,strokewidth=1)
axislegend(ax2;position=:lt)
display(fig2)

# e) Try the initial guess x0=(-2,-2)
x0=[-2.0,-2.0]
data4=[Point2D(x0...,f(x0))]
x=x0

for i in 1:100
    global x-=alpha*Gradient_f(x)
    push!(data4,Point2D(x...,f(x)))
end

println("Starting from $x0, the local minimum is approximately $x")

# f) Compare the paths from (-2,2) and (-2,-2)
fig3=Figure()
ax3=Axis(fig3[1,1],xlabel=L"x_1",ylabel=L"x_2",
    title="Effect of the initial guess",limits=(-3,3,-3,3))
contourf!(ax3,x1range,x2range,funcplot;levels=50,colormap=:bwr)
contour!(ax3,x1range,x2range,funcplot;levels=15,color=:black)
lines!(ax3,getproperty.(data3,:x1),getproperty.(data3,:x2);
    linewidth=3,color=:orange,label="Start at (-2,2)")
scatter!(ax3,getproperty.(data3,:x1),getproperty.(data3,:x2);
    color=:orange,markersize=10)
lines!(ax3,getproperty.(data4,:x1),getproperty.(data4,:x2);
    linewidth=3,color=:green,label="Start at (-2,-2)")
scatter!(ax3,getproperty.(data4,:x1),getproperty.(data4,:x2);
    color=:green,markersize=10)
axislegend(ax3;position=:lt)
display(fig3)

# At (-2,2), x1 and x2 have opposite signs.  Gradient descent moves along
# x2=-x1 and away from the origin, where f approaches zero.  It therefore
# does not find a finite local minimum.  At (-2,-2), both coordinates have
# the same sign and the iterates move towards the local minimum (-1,-1).


# Question 3

# Read Q3Data.csv relative to the folder containing this solution file.
data_path=joinpath(@__DIR__,"Q3Data.csv")
df=CSV.read(data_path,DataFrame;
    select=["xQ1a","yQ1a","xQ1b","yQ1b","xQ1c","yQ1c"])

x_a=df[:,:xQ1a]
y_a=df[:,:yQ1a]
x_b=df[:,:xQ1b]
y_b=df[:,:yQ1b]
x_c=df[:,:xQ1c]
y_c=df[:,:yQ1c]

# a) Plot the three sets of temperature data
fig4=Figure(size=(1200,400))
ax4a=Axis(fig4[1,1],xlabel="x (m)",ylabel="Temperature (degrees C)",
    title="Heating condition A",limits=(0,10,0,42))
ax4b=Axis(fig4[1,2],xlabel="x (m)",ylabel="Temperature (degrees C)",
    title="Heating condition B",limits=(0,10,0,42))
ax4c=Axis(fig4[1,3],xlabel="x (m)",ylabel="Temperature (degrees C)",
    title="Heating condition C",limits=(0,10,0,42))
scatter!(ax4a,x_a,y_a;color=:black,label="Data")
scatter!(ax4b,x_b,y_b;color=:black,label="Data")
scatter!(ax4c,x_c,y_c;color=:black,label="Data")
display(fig4)

# b) Conditions A and B show exponential decay towards 10 degrees C:
#        T(x)=10+30*exp(-k*x)
# Condition C shows a damped oscillation towards 10 degrees C:
#        T(x)=10+30*exp(-k*x)*cos(omega*x)
# These forms also satisfy T(0)=40 degrees C.

model_exp(k,x)=10+30*exp(-k*x)
model_osc(k,omega,x)=10+30*exp(-k*x)*cos(omega*x)

# c) The models are nonlinear in k and omega, so use standard gradient
# descent to minimise the mean squared errors.

# Heating condition A
k_a=1.0 #initial guess
alpha_model=0.01 #learning rate

for i in 1:20_000
    y_model=model_exp.(k_a,x_a)
    dS_dk=sum((y_model.-y_a).*(-30 .*x_a.*exp.(-k_a.*x_a)))/length(x_a)
    global k_a-=alpha_model*dS_dk
end

# Heating condition B
k_b=1.0 #initial guess

for i in 1:20_000
    y_model=model_exp.(k_b,x_b)
    dS_dk=sum((y_model.-y_b).*(-30 .*x_b.*exp.(-k_b.*x_b)))/length(x_b)
    global k_b-=alpha_model*dS_dk
end

# Heating condition C
k_c=0.5 #initial guess for k
omega=1.5 #initial guess for omega
alpha_osc=0.001 #learning rate

for i in 1:50_000
    exponential=exp.(-k_c.*x_c)
    y_model=10 .+30 .*exponential.*cos.(omega.*x_c)
    dS_dk=sum((y_model.-y_c).*(-30 .*x_c.*exponential.*cos.(omega.*x_c)))/length(x_c)
    dS_domega=sum((y_model.-y_c).*(-30 .*x_c.*exponential.*sin.(omega.*x_c)))/length(x_c)
    global k_c-=alpha_osc*dS_dk
    global omega-=alpha_osc*dS_domega
end

println("\nQuestion 3 fitted models:")
println("  A: T(x)=10+30exp(-$k_a*x)")
println("  B: T(x)=10+30exp(-$k_b*x)")
println("  C: T(x)=10+30exp(-$k_c*x)cos($omega*x)")

# Plot the fitted models together with the data
xplot=0:0.01:10
lines!(ax4a,xplot,model_exp.(k_a,xplot);color=:blue,linewidth=3,label="Model")
lines!(ax4b,xplot,model_exp.(k_b,xplot);color=:blue,linewidth=3,label="Model")
lines!(ax4c,xplot,model_osc.(k_c,omega,xplot);color=:blue,linewidth=3,label="Model")
axislegend(ax4a;position=:rt)
axislegend(ax4b;position=:rt)
axislegend(ax4c;position=:rt)
display(fig4)

