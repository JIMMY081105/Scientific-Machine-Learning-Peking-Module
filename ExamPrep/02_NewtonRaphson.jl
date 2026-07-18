# =====================================================================================
# NEWTON-RAPHSON (root finding) — organized by exam level
# =====================================================================================
# Update rule (1 equation):  x <- x - f(x)/f'(x)
# Update rule (system):      x <- x + delta,  where J*delta = -F  (delta = -J\F)
# Sources: Chapter01 01p04, Homework/Day02, Homework/Day08-09, Assignment 1 Q1.
# =====================================================================================

# =====================================================================================
# LEVEL 1 — hand-coded, analytic derivative
# =====================================================================================

# ---- 1a) One equation f(x) = 0 ----
f(x) = 4pi * x - 2 / x^2
dfdx(x) = 4pi + 4 / x^3      # differentiate by hand

x = 1.0                      # initial guess (read it off a plot if given data)
for i = 1:10
    global x = x - f(x) / dfdx(x)
    println(x)
end

# ---- 1b) System of 2 equations u(x,y)=0, v(x,y)=0 ----
function FunctionStore(x, y)
    return [x^3 + y^3 - 1, x^2 + y^2 - 8]     # replace with the question's equations
end

# Jacobian by hand: row 1 = [du/dx du/dy], row 2 = [dv/dx dv/dy]
function Jacobian(x, y)
    return [3*x^2 3*y^2;
            2*x   2*y]
end

x = 2.0                      # initial guesses: read off the zero-contour plot (see
y = -2.0                     # 04_SolveEquations.jl for how to draw it)
for i = 1:10
    J = Jacobian(x, y)
    F = FunctionStore(x, y)
    delta = -J \ F           # the linear solver — allowed at Level 1
    global x += delta[1]
    global y += delta[2]
    println("x=$x, y=$y")
end

# ---- 1c) Solving for a PARAMETER instead of x (Homework Day08-09 pattern) ----
# Given f(x, b) and a fixed x, find b such that f(x, b) = 0. Same loop, the
# unknown is just b now — differentiate f with respect to b.
fb(x, b) = x^3 * exp(-(x / b)^2) - 1.0
x_fixed = 10.0
g(b) = fb(x_fixed, b)
dgdb(b) = x_fixed^3 * exp(-(x_fixed / b)^2) * (2 * x_fixed^2 / b^3)

b = 3.5
for i = 1:10
    global b = b - g(b) / dgdb(b)
    println(b)
end

# =====================================================================================
# LEVEL 2 — same loops, derivative from ForwardDiff (no more hand differentiation)
# =====================================================================================
using ForwardDiff

# ---- 2a) One equation ----
f2(x) = 4pi * x - 2 / x^2
x = 1.0
for i = 1:10
    global x -= f2(x) / ForwardDiff.derivative(f2, x)
end
println("Level 2 root: ", x)

# ---- 2b) System: Jacobian from ForwardDiff (vector input, vector output) ----
VectorFunc(x) = [x[1]^3 + x[2]^3 - 1, x[1]^2 + x[2]^2 - 8]

x = [2.0, -2.0]
for i = 1:10
    J = ForwardDiff.jacobian(VectorFunc, x)
    F = VectorFunc(x)
    global x += -J \ F
end
println("Level 2 system solution: ", x)

# =====================================================================================
# LEVEL 3 — NonlinearSolve does everything
# =====================================================================================
using NonlinearSolve

# ---- 3a) One equation ----
x0 = 1.0
p = []
prob = NonlinearProblem((x, p) -> f2(x), x0, p)
sol = NonlinearSolve.solve(prob)
println("NonlinearSolve root: ", sol.u)

# ---- 3b) System ----
prob = NonlinearProblem((x, p) -> VectorFunc(x), [2.0, -2.0], [])
sol = NonlinearSolve.solve(prob)
println("NonlinearSolve system: ", sol.u)     # also check sol.retcode
