# =====================================================================================
# 4.1 - Jacobians of vector-valued functions
# =====================================================================================
# GOAL: compute Jacobians of functions that return a vector, and use one to solve a
#       nonlinear system with Newton-Raphson.
# STRATEGY: ForwardDiff.jacobian / Zygote.jacobian; note Zygote rejects in-place array
#           mutation (fvec1 mutates, fvec2 is allocation-free), so prefer the
#           non-mutating style with Zygote.
# =====================================================================================

using ForwardDiff
using Zygote

# Same function written two ways: fvec1 mutates an array, fvec2 does not.
function fvec1(x)
    df = similar(x)
    df[1] = x[1] - 1
    for i = 2:length(x)-1
        df[i] = x[i] - x[i-1]^2
    end
    df[end] = x[end] - x[end-1]
    return df
end

fvec2(x) = [x[1] - 1;
    x[2:end-1] .- x[1:end-2].^2;
    x[end] - x[end-1]]

N = 5
x = 2 * ones(N)
fvec1(x)
fvec1(x) .- fvec2(x)

# Zygote works on the non-mutating fvec2; ForwardDiff handles both.
Zygote.jacobian(fvec2, x)[1]
ForwardDiff.jacobian(fvec1, x)
ForwardDiff.jacobian(fvec2, x)

# Nonlinear system: Jacobian drives Newton-Raphson.
VectorFunc(x) = [x[1]^2 - x[2] + 1,
    3 * cos(x[1]) - x[2]]

ForwardDiff.jacobian(VectorFunc, [1, 1])

x = [1, 1]
for i = 1:5
    F = VectorFunc(x)
    J = ForwardDiff.jacobian(VectorFunc, x)
    delta = -J \ F
    x += delta
    println(x)
end
