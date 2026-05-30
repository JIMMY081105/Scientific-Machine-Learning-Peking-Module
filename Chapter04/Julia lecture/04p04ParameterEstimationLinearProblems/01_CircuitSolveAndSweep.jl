# =====================================================================================
# 4.4 - Circuit linear system: solve and sweep
# =====================================================================================
# GOAL: for a resistor circuit whose currents solve A i = C, find roughly which input
#       voltage V1 makes the current i5 equal zero.
# STRATEGY: assemble and solve the linear system (backslash and LinearSolve), then sweep
#           V1 over a range, re-solving each time, and plot i5 (and i5^2) vs V1 to locate
#           the crossing. This is the brute-force scan; 02 does it with optimization.
# =====================================================================================

using LinearSolve
using CairoMakie

# Resistances and source voltages.
R₁ = 5.0
R₂ = 150.0
R₃ = 100.0
R₄ = 250.0
R₅ = 200.0

V₁ = 100.0
V₂ = 50.0

# Kirchhoff's laws give the system A i = C.
A = [R₁ R₂ 0 0 0;
    0 -R₂ R₃ R₄ 0;
    0 0 0 R₄ -R₅;
    1.0 -1.0 -1.0 0 0;
    0 0 1.0 -1.0 -1.0]

C₁ = [V₁, 0, V₂, 0, 0] # C1 is a (5x1) column vector
C₂ = [V₁ 0 V₂ 0 0]     # C2 is a (1x5) row matrix

@time i = A \ C₁
A \ C₂'

# The same solve via LinearSolve.
prob = LinearProblem(A, C₁)
@time sol = solve(prob)
i = sol.u

# Sweep V1 from 50 to 100 and record i5 each time.
A = [R₁ R₂ 0 0 0;
    0 -R₂ R₃ R₄ 0;
    0 0 0 R₄ -R₅;
    1 -1 -1 0 0;
    0 0 1 -1 -1]

C₁ = [V₁, 0, V₂, 0, 0]
V₁vec = 50:1:100
i₅vec = zeros(length(V₁vec))
for i = 1:length(V₁vec)
    C₁[1] = V₁vec[i]
    prob = LinearProblem(A, C₁)
    sol = solve(prob)
    i₅vec[i] = sol.u[5]
end
i₅vec

# i5 vs V1 (crosses zero near V1 = 91).
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"V_1", ylabel = L"i_5", limits = (50, 100, -0.07, 0.07))
lines!(ax1, V₁vec, i₅vec; color = :black)
# save("../../figures/04p04ParameterEstimationLinearProblems03.svg", fig)   # (figure-save disabled in study file)
display(fig)

# i5^2 vs V1 (its minimum marks the target).
fig = Figure()
ax2 = Axis(fig[1, 1], xlabel = L"V_1", ylabel = L"i_5^2")
lines!(ax2, V₁vec, i₅vec.^2; color = :black)
# save("../../figures/04p04ParameterEstimationLinearProblems04.svg", fig)   # (figure-save disabled in study file)
display(fig)
