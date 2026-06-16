# Chapter 1 - Introduction, Julia Basics, Plotting, and Solving Equations

This chapter sets up the whole course. The recurring engineering question is: how do we find the minimum (or maximum) of a function, because that is what optimisation, curve-fitting and machine learning all reduce to. Notebook 1.1 motivates why minimisation matters (least material, least cost, best model fit) with a cylinder-design example. Notebook 1.2 teaches just enough Julia syntax to read every later program. Notebook 1.3 shows how to plot 1-D functions and 2-D test functions (Himmelblau, Rosenbrock) that later optimisation algorithms will be tested on. Notebook 1.4 revises how to solve linear systems and nonlinear equations (Newton-Raphson), which is the root-finding machinery behind optimisation (a minimum is where the derivative equals zero).

## Notebooks in this chapter

### 1.1 Introduction & Motivation  →  `01p01IntroductionAndMotivation.jl`

**Problem.** Design a cylinder that holds a fixed volume of water using the least material. This means minimising the surface area A(r) = 2*pi*r^2 + 2/r. It introduces the core idea of the course: many engineering goals become 'find the value that minimises a function'.

**Procedure.**

1. Load a plotting package (CairoMakie).
2. Write the surface-area function A(r) as a plain Julia function.
3. Sweep r over a range and plot A(r) to eyeball where the minimum sits (around r = 0.5 m).
4. Generalise to A(r, V) when the volume V is also free, and plot one curve per V to see how the shape shifts.
5. Conclude: reading a minimum off a graph works but is crude; later notebooks compute it exactly.

### 1.2 Julia Programming Basics  →  `01p02JuliaProgrammingBasics.jl`

**Problem.** Learn the minimum Julia needed to read every program in this course: variable types, tuples and structs, vectors/matrices/arrays, operators, functions, variable scope, and loops.

**Procedure.**

1. Types: assign values and inspect them with typeof; see Int64 vs Float64 vs Float32 vs String.
2. Grouping: bundle values with an (immutable) tuple, then with a struct and a mutable struct.
3. Arrays: build column vectors and matrices, slice rows/columns, transpose/adjoint, ranges, and array comprehensions.
4. Operators: scalar arithmetic, the difference between / and \, matrix*vector, elementwise '.' operators, and solving A x = C with the backslash operator.
5. Functions: one-line and full 'function' definitions, broadcasting a function with '.', and anonymous (x,p)->... functions used later by optimisation libraries.
6. Scope and loops: how variables inside functions/loops stay local, and how 'for' loops repeat calculations.

### 1.3 Plotting Functions in Julia  →  `01p03PlottingFunctionsInJulia.jl`

**Problem.** Learn to plot the standard test functions that later optimisation algorithms will be run on, so you can see minima and maxima before trying to compute them.

**Procedure.**

1. Plot a simple 1-D quadratic and mark its minimum with a scatter point.
2. Plot the harder 1-D function (x^3 - x)^2 and mark all its minima/maxima.
3. Build a 2-D grid with an array comprehension and draw filled contour plots of the Himmelblau function (4 minima, 1 maximum).
4. Do the same for the Rosenbrock function, whose minimum sits in a long narrow valley that is hard for algorithms to find.

### 1.4 Solving Linear and Nonlinear Equations  →  `01p04SolvingLinearAndNonlinearEquations.jl`

**Problem.** Revise how to solve (a) linear systems A x = C, and (b) single and coupled nonlinear equations f(x) = 0. This is the root-finding machinery behind optimisation, since a minimum is a root of the derivative.

**Procedure.**

1. Linear system: build the circuit matrix A and right-hand side C, solve with the backslash operator, and again with the LinearSolve library (faster for big systems); read the solution from sol.u and check sol.retcode.
2. Newton-Raphson idea: derive x_{i+1} = x_i - f(x_i)/f'(x_i) from the truncated Taylor series.
3. Single nonlinear equation: apply Newton-Raphson by hand in a loop to find where dA/dr = 0, then solve the same thing with the NonlinearSolve library.
4. System of nonlinear equations: extend Newton-Raphson using the Jacobian matrix, iterate to the intersection of two curves, and confirm with NonlinearSolve; visualise the solution as the crossing of two zero-contours.
