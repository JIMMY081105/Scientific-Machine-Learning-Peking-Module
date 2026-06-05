using NonlinearSolve

f(x) = 4pi * x - 2 / x^2


x0 = 1.0 # initial guess
p = [] # no parameters
prob = NonlinearProblem((x, p) -> f(x), x0, p)
sol = NonlinearSolve.solve(prob)
