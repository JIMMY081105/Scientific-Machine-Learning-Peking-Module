# =====================================================================================
# 4.1 - Symbolic differentiation
# =====================================================================================
# GOAL: get an exact symbolic expression for the derivative of f(x) = sin(x^2).
# STRATEGY: use the Symbolics package - great for small problems, but the expression
#           blows up for complicated functions (the reason we move on to AD).
# =====================================================================================

using CairoMakie
using Symbolics

# The function we differentiate throughout 4.1.
f(x) = sin(x^2)
xplot = -1:0.01:1
lines(xplot, f.(xplot))

# Symbolic derivative with Symbolics.
@variables x

# Note: this is a symbolic expression, not the function f defined above.
f1 = sin(x^2)

# Differential operator w.r.t. x, then expand.
D = Differential(x)
expand_derivatives(D(f1))

# TODO (Exercise Ex01): Change the code above to get the symbolic derivative of sin(x^2) + exp(-cos(x)).
