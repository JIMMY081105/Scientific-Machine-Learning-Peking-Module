# =====================================================================================
# 1.2 - Grouping values: tuples and structs
# =====================================================================================
# GOAL: bundle related values together with a tuple, an (immutable) struct, and a
#       mutable struct.
# =====================================================================================

x = 1
y = 2.0
z = "Here is a string"

# A tuple groups values of different types; index with [].
t = (x, y, z)
typeof(t)
t[3]

# A struct is a named group of fields; access with the dot.
struct House
    Address::String
    Size::Float32
    Construction::String
end

MyHouse = House("Australia", 100.0, "Brick")
MyHouse.Address

# Structs are immutable by default, so the next line intentionally FAILS:
# MyHouse.Size = 200   # ERROR: cannot set field of an immutable struct

# Use a mutable struct when you need to change fields.
mutable struct Age
    Years::Int
end
MyAge = Age(70)
MyAge.Years = 60
MyAge.Years
