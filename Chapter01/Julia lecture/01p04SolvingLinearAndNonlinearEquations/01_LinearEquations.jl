# =====================================================================================
# 1.4 - Solving a linear system (Example 1.4.1)
# =====================================================================================
# GOAL: solve the circuit linear system A i = C for the currents.
# STRATEGY: build A and C, solve with the backslash operator, and again with the
#           LinearSolve library (faster for big systems); read sol.u and sol.retcode.
# =====================================================================================

using LinearSolve

R₁ = 5
R₂ = 150
R₃ = 100
R₄ = 250
R₅ = 200

V₁ = 100.0
V₂ = 50.0

A = [R₁ R₂ 0 0 0;
    0 -R₂ R₃ R₄ 0;
    0 0 0 R₄ -R₅;
    1 -1 -1 0 0;
    0 0 1 -1 -1]

C₁ = [V₁, 0, V₂, 0, 0] # C1 is a (5x1) column vector
C₂ = [V₁ 0 V₂ 0 0]     # C2 is a (1x5) row matrix

# Backslash solve.
@time i = A \ C₁
A \ C₂'

# The same solve via LinearSolve.
prob = LinearProblem(A, C₁)
@time sol = solve(prob)
i = sol.u
sol.retcode

# TODO (Class Demo 01): Solve for i5 over V1 in [50, 100] and plot i5 vs V1.
