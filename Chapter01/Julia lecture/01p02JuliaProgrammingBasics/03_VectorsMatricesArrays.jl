# =====================================================================================
# 1.2 - Vectors, matrices, and arrays
# =====================================================================================
# GOAL: build column vectors and matrices, slice rows/columns, transpose, make ranges,
#       and build arrays with comprehensions.
# =====================================================================================

# Uninitialised (undef) vector and matrix.
Vector{Float64}(undef, 4)
Matrix{Float64}(undef, 2, 4)

# Column vector: [] with comma-separated elements.
a = [1.0, 2.0, 3.0]

# TODO (Exercise Ex02): Type a = [1 2 3 4]. What does Julia give you?

# Matrix: rows separated by semicolons.
b = [2.0 3.0 4.0; 8.0 9.0 4.0]

# Slice rows/columns with : (slicing gives a vector).
b[1, :]
b[:, 2]

# TODO (Exercise Ex03): Type b[:]. What does Julia give you?

# Transpose / adjoint.
transpose(b)
b'

# Ranges (start:step:stop) and turning one into an Array.
x = 0.0:0.1:1.0
Array(x)

# Array comprehension over two ranges.
xrange = -1:1:1
yrange = 1:1:3
loss_values = [x + y^2 for x in xrange, y in yrange]
