# =====================================================================================
# 1.2 - Functions
# =====================================================================================
# GOAL: one-line and full "function" definitions, broadcasting a function with ., and
#       anonymous (x, p) -> ... functions used later by optimisation libraries.
# =====================================================================================

a = [1, 2, 3]

# One-line function.
f(x) = x^3
typeof(f(1)), typeof(f(2.0))

# Broadcast f over every element of a with the dot.
f.(a)

# TODO (Exercise Ex06): What does f(a) give you? Why?

# Full definition with the function keyword.
function my_first_function(x, y)
    mean = 0.5 * (x + y)
end
my_first_function(1.0, 2.0)

# Anonymous function (this (x, p) -> ... form is what the optimisation libraries expect).
dummy_func = (x, p) -> f(x)
dummy_func(3, 10)
