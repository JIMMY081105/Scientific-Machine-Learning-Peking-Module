########################################################################################
# 1.2 Julia Programming Basics
########################################################################################
#
# PROBLEM
# Learn the minimum Julia needed to read every program in this course: variable
# types, tuples and structs, vectors/matrices/arrays, operators, functions, variable
# scope, and loops.
#
# PROCEDURE (how this file solves it, top to bottom)
#   1. Types: assign values and inspect them with typeof; see Int64 vs Float64 vs
#      Float32 vs String.
#   2. Grouping: bundle values with an (immutable) tuple, then with a struct and a
#      mutable struct.
#   3. Arrays: build column vectors and matrices, slice rows/columns,
#      transpose/adjoint, ranges, and array comprehensions.
#   4. Operators: scalar arithmetic, the difference between / and \, matrix*vector,
#      elementwise '.' operators, and solving A x = C with the backslash operator.
#   5. Functions: one-line and full 'function' definitions, broadcasting a function
#      with '.', and anonymous (x,p)->... functions used later by optimisation
#      libraries.
#   6. Scope and loops: how variables inside functions/loops stay local, and how
#      'for' loops repeat calculations.
#
# This file is notebook "01p02JuliaProgrammingBasics" with every code cell joined in
# order, so you can read it straight through. Comments are light and
# purpose-focused. Figure-saving lines are disabled; plots still display.
########################################################################################


# =================================
# 1.2 Julia Programming Basic
# =================================

# ====================
# Types
# ====================

x=1
y=2.0
z="Here is a string"
# You can find out the type of the variables by using the `typeof()` function in
# Julia

typeof(x)

# TODO (Exercise Ex01): Use `typeof()` to find out the type of variables `y` and `z`
# Note that when we set `x` to the value of 1, Julia by default set x to be an
# integer of type *Int64*
# You can force a variable to be single precision floating point with the code below

p=1.0f0
q=Float32(1.0)
typeof(p),typeof(q)
# You can group variables together by using a `tuple` in Julia

t=(x,y,z)
# Here the tuple `t` has three elements an integer, float and a string.

typeof(t)

t[3]
# Another way of grouping different variables together is by using a `struct`.

struct House
    Address::String
    Size::Float32
    Construction::String
end

MyHouse=House("Australia",100.0,"Brick")
# You can access each element of the struct by using the .

MyHouse.Address
# By default structs are immutable, that means you cannot change the variables in
# struct.

MyHouse.Size=200
# If you want to be able to change the elements inside struct, you will need to use a
# mutable struct

# Define a mutable struct called Counter
mutable struct Age
    Years::Int
end
MyAge=Age(70)

MyAge.Years=60

MyAge.Years

# =============================
# Vectors, Matrix, Arrays
# =============================
# You can make a undefined vector and matrix of 4 and 2x4 elements respectively

Vector{Float64}(undef, 4)

Matrix{Float64}(undef, 2, 4)
# You can specify a column vector in Julia by using a [] bracket and separating each
# element by a comma

a=[1.0, 2.0 ,3.0]
# a is a column vector with 3 elements

# TODO (Exercise Ex02): Type in `a=[1 2 3 4]`. What does Julia give you?
# Matrices can be defined using [], with each row of the matrices separated by a
# semicolon.

b=[2.0 3.0 4.0;8.0 9.0 4.0]
# You can slice rows and columns of a matrix by using :

b[1,:]

b[:,2]
# Note that after slicing the b matrix, you will get vectors.

# TODO (Exercise Ex03): Type in `b[:]`. What does Julia give you?
# The transpose of a matrix can be obtained by using the `transpose()` function

transpose(b)

b'
# You can define an array from 0 to 1 in steps of 0.1 by using a `range` using the
# following syntax

x=0.0:0.1:1.0

Array(x)

xrange=-1:1:1
yrange=1:1:3
loss_values=[x+y^2 for x in xrange, y in yrange]

# ====================
# Operators
# ====================
# The normal operators apply in Julia if the left and right arguments are scalar

a=1.0+2.0 # addition
b=1.0-2.0 #subtraction
c=2.0*3.0 #multiplication
d=3.0^3 #power

(a,b,c,d)

1.0/2.0

1.0\2.0 
# Let's now see what happens if the arguments to the operators are vectors/matrices.

a=[1,2,3]
b=[2.0 3.0 4.0;8.0 9.0 4.0]
b*a
# You can multiply matrix with a vector

b*a

# TODO (Exercise Ex04): Does `ab` or `aa` work? Why?
# We can perform elementwise operations by preceding the opearator with a .

a.*a

b.*b

a.*b[1,:]

# TODO (Exercise Ex05): What does `a.b[1,:]'` give you? Can you explain why?

A=[1 2 3;4 5 6;7 8 9]
C=[2, 3, 4]
x=A\C #equivalent to A^{-1}*C
A*x-C

# ====================
# Functions
# ====================
# Functions are used to group codes that is repeatedly used.

f(x)=x^3

typeof(f(1)),typeof(f(2.0))
# You can make f act on every element of a by using the .

f.(a)

# TODO (Exercise Ex06): What does `f(a)` give you? Why?
# More complex functions can be defined by using the `function` keyword.

function my_first_function(x,y)
    mean=0.5*(x+y)
end

my_first_function(1.0,2.0)

dummy_func=(x,p)->f(x)
dummy_func(3,10) 

# ========================
# Scope of variables
# ========================
# Usually variables inside functions are local

x=2
function my_print(x)
    println(x)
end

my_print(3)

x=2
function my_print(b)
    println(x)
end

my_print(10)

# ====================
# Loops
# ====================
# We usually use loops to do repeated calculations.

x=7.0
for i=1:5
    x-=1.0
end
x
# If you declare a variable inside the loop, then it is only available inside the
# loop.

xoutsideloop=3.0

for i=1:3
    xoutsideloop+=1.0
    xinsideloop=xoutsideloop+1.0
end

xoutsideloop
# Once you finish executing the loop, the program outside the loop does not know
# anything about `xinsideloop` which was declared inside the for loop.

xinsideloop
