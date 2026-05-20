# =====================================================================================
# 1.2 - Operators
# =====================================================================================
# GOAL: scalar arithmetic, the difference between / and \, matrix*vector, elementwise
#       (.) operators, and solving A x = C with the backslash operator.
# =====================================================================================

# Scalar arithmetic.
a = 1.0 + 2.0 # addition
b = 1.0 - 2.0 # subtraction
c = 2.0 * 3.0 # multiplication
d = 3.0^3     # power
(a, b, c, d)

# / divides left by right; \ divides right by left.
1.0 / 2.0
1.0 \ 2.0

# Matrix times vector.
a = [1, 2, 3]
b = [2.0 3.0 4.0; 8.0 9.0 4.0]
b * a

# TODO (Exercise Ex04): Does a*b or a*a work? Why?

# Elementwise operations: precede the operator with a dot.
a .* a
b .* b
a .* b[1, :]

# Solve A x = C (backslash is A^{-1} C).
A = [1 2 3; 4 5 6; 7 8 9]
C = [2, 3, 4]
x = A \ C
A * x - C
