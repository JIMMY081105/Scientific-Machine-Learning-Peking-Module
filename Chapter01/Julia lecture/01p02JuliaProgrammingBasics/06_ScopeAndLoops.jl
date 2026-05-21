# =====================================================================================
# 1.2 - Scope of variables and loops
# =====================================================================================
# GOAL: see how variables inside functions/loops stay local, and how for loops repeat
#       calculations.
# =====================================================================================

# A function argument named x is local; it does not change the outer x.
x = 2
function my_print(x)
    println(x)
end
my_print(3)

# Here the function has no x of its own, so it reads the outer x.
x = 2
function my_print2(b)
    println(x)
end
my_print2(10)

# A for loop repeats a calculation.
x = 7.0
for i = 1:5
    x -= 1.0
end
x

# A variable declared INSIDE a loop is only visible inside it.
xoutsideloop = 3.0
for i = 1:3
    xoutsideloop += 1.0
    xinsideloop = xoutsideloop + 1.0
end
xoutsideloop

# The next line intentionally FAILS: xinsideloop was local to the loop.
# xinsideloop   # ERROR: xinsideloop is not defined outside the loop
