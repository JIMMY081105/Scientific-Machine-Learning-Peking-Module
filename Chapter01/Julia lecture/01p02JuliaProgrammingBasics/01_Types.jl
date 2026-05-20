# =====================================================================================
# 1.2 - Types
# =====================================================================================
# GOAL: assign values and inspect their types with typeof; see Int64 vs Float64 vs
#       Float32 vs String.
# =====================================================================================

x = 1
y = 2.0
z = "Here is a string"

# typeof reports the type. x defaults to Int64, y to Float64.
typeof(x)

# TODO (Exercise Ex01): Use typeof() on y and z.

# Force single-precision floating point.
p = 1.0f0
q = Float32(1.0)
typeof(p), typeof(q)
