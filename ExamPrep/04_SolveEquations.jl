# =====================================================================================
# SOLVING EQUATIONS: linear systems, nonlinear equations/systems, contours, BVPs
# =====================================================================================
# Sources: Chapter01 01p04, class 2 (Himmelblau/nonlinear), Chapter05, Assignment 2 Q1.
# =====================================================================================

# =====================================================================================
# LINEAR SYSTEM A x = C  (allowed at EVERY level — "linear solvers")
# =====================================================================================
using LinearAlgebra
using LinearSolve

A = [5.0 2.0 1.0;
     2.0 7.0 3.0;
     1.0 3.0 9.0]
C = [10.0, 5.0, 3.0]

# Way 1: backslash (always fine)
x = A \ C
println("backslash: ", x)

# Way 2: LinearSolve (faster for big systems; gives retcode)
prob = LinearProblem(A, C)
sol = LinearSolve.solve(prob)
println("LinearSolve: ", sol.u)

# =====================================================================================
# PLOTTING TO LOCATE SOLUTIONS FIRST (do this BEFORE any Newton loop!)
# =====================================================================================
using CairoMakie

# ---- Zero-contours of a 2-equation system: solutions = where the curves CROSS ----
FunctionStore(x, y) = [x^3 + y^3 - 1, x^2 + y^2 - 8]

xrange = -4.0:0.05:4.0
yrange = -4.0:0.05:4.0
functionToPlot = [FunctionStore(x, y) for x in xrange, y in yrange]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", limits = (-4, 4, -4, 4))
contour!(ax, xrange, yrange, getindex.(functionToPlot, 1); levels = [0.0], color = :black, label = L"u(x,y)")
contour!(ax, xrange, yrange, getindex.(functionToPlot, 2); levels = [0.0], color = :blue, label = L"v(x,y)")
axislegend(ax, position = :rt)
display(fig)
# Read the crossing points off this plot -> those are your Newton initial guesses.

# ---- Contour plot of a 2-D function (Himmelblau) to LOCATE minima/maxima ----
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
x1range = -6:0.02:6
x2range = -6:0.02:6
funcplot = [himmelblau([x1, x2]) for x1 in x1range, x2 in x2range]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", limits = (-6, 6, -6, 6))
contourf!(ax, x1range, x2range, funcplot; levels = 50, colormap = :bwr)
contour!(ax, x1range, x2range, funcplot; labels = true, levels = 15, color = :black)
display(fig)
# 4 minima visible -> run gradient descent (file 03) from a start near each one.

# =====================================================================================
# NONLINEAR EQUATIONS — hand Newton is in 02_NewtonRaphson.jl; Level 3 version here
# =====================================================================================
using NonlinearSolve

f(x) = 4pi * x - 2 / x^2
prob_nl = NonlinearProblem((x, p) -> f(x), 1.0, [])
sol_nl = NonlinearSolve.solve(prob_nl)
println("single nonlinear: ", sol_nl.u)

prob_sys = NonlinearProblem((x, p) -> FunctionStore(x[1], x[2]), [2.0, -2.0], [])
sol_sys = NonlinearSolve.solve(prob_sys)
println("nonlinear system: ", sol_sys.u)

# =====================================================================================
# BOUNDARY VALUE PROBLEM by finite differences (Level 1/2 legal — only `\` needed)
# =====================================================================================
# Example: k*T'' = -q on [0,1], T(0)=5 (Dirichlet), T'(1)=0 (Neumann).
# Recipe: grid -> replace T'' with (T[i-1]-2T[i]+T[i+1])/Δ² -> one equation per point
# -> boundary rows -> Tridiagonal A, solve A*T = C.

Delta = 0.02
xgrid = 0:Delta:1
n = length(xgrid)
k = 0.1
q = 1.0

d = zeros(n)
dl = zeros(n - 1)
du = zeros(n - 1)
C2 = zeros(n)

# Row 1 — LEFT boundary:
#   Dirichlet T(0)=value:  d[1]=1.0;             C2[1]=value
#   Neumann  T'(0)=0:      d[1]=-1.0; du[1]=1.0; C2[1]=0.0
du[1] = 0.0
d[1] = 1.0
C2[1] = 5.0

# Interior rows — the discretised ODE. For k*T'' = -q:
#   (1/Δ²)T[i-1] + (-2/Δ²)T[i] + (1/Δ²)T[i+1] = -q/k
# If the ODE has an extra c*T term (e.g. T'' + p*T = RHS), it goes into d[i]: -2/Δ² + p
for i = 2:n-1
    dl[i-1] = 1.0 / Delta^2
    d[i] = -2.0 / Delta^2
    du[i] = 1.0 / Delta^2
    C2[i] = -q / k              # non-uniform heating q(x): just make this depend on xgrid[i]
end

# Row n — RIGHT boundary:
#   Dirichlet T(1)=value:  d[n]=1.0;               C2[n]=value
#   Neumann  T'(1)=0:      dl[n-1]=-1.0; d[n]=1.0; C2[n]=0.0
dl[n-1] = -1.0
d[n] = 1.0
C2[n] = 0.0

A2 = Tridiagonal(dl, d, du)
probBVP = LinearProblem(A2, C2)
T = LinearSolve.solve(probBVP)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"T(x)")
lines!(ax, xgrid, T.u, color = :black)
display(fig)

# =====================================================================================
# NONLINEAR BVP (e.g. a T^4 term) — can't be a matrix; write a residual, Level 3
# =====================================================================================
# Each residual entry = "left side minus right side" at a grid point (0 at solution);
# first/last entries are the boundary conditions. Then NonlinearProblem.
# Full worked example: Chapter05/Julia lecture/05p03NonlinearBoundaryValueProblem.jl

# =====================================================================================
# INVERSE PROBLEM through a solve (Assignment 2 Q1 pattern) — find p so the
# solution hits a target. Loss rebuilds + solves the system each evaluation.
# =====================================================================================
using ForwardDiff

function solve_bvp(p; n = 101)
    Δ = 2.0 / (n - 1)
    xg = 0:Δ:2
    Tt = typeof(p)                      # IMPORTANT: zeros(typeof(p), n) so ForwardDiff
    dd = zeros(Tt, n)                   # can push its Dual numbers through the solve
    ddl = zeros(Tt, n - 1)
    ddu = zeros(Tt, n - 1)
    CC = zeros(Tt, n)
    dd[1] = -1.0; ddu[1] = 1.0; CC[1] = 0.0
    for i = 2:n-1
        ddl[i-1] = 1.0 / Δ^2
        dd[i] = -2.0 / Δ^2 + p
        ddu[i] = 1.0 / Δ^2
        CC[i] = cos(xg[i])
    end
    ddl[n-1] = 0.0; dd[n] = 1.0; CC[n] = 5.0
    AA = Tridiagonal(ddl, dd, ddu)
    return LinearSolve.solve(LinearProblem(AA, CC))
end

loss(p) = (solve_bvp(p)[1] - 1.0)^2     # target: y(x=0) = 1

p_gd = 0.0
αp = 0.05
for i = 1:300
    global p_gd -= αp * ForwardDiff.derivative(loss, p_gd)
end
println("inverse problem: p = ", p_gd)
