using LinearSolve
R₁=5
R₂=150
R₃=100
R₄=250
R₅=200

V₁=100.0
V₂=50.0

A=[R₁ R₂ 0 0 0;
  0 -R₂ R₃ R₄ 0;
  0 0 0 R₄ -R₅;
  1 -1 -1 0 0;
  0 0 1 -1 -1]

C₁=[V₁, 0, V₂, 0, 0] #C1 is a (5x1) column vector
V₁vec=50:1:100
i₅vec=zeros(length(V₁vec))
for i=1:length(V₁vec)
    C₁[1]=V₁vec[i]
    prob = LinearProblem(A,C₁)
    sol = solve(prob)
    i₅vec[i]=sol.u[5]
end

fig = Figure() #set up a figure
ax1 = Axis(fig[1, 1], xlabel = L"V_1", ylabel = L"i_5",limits=(50,100,-0.07,0.07)) #Define axis in the figure
lines!(ax1,V₁vec,i₅vec;color=:black) #Plot i_5 vs V_1
display(fig)