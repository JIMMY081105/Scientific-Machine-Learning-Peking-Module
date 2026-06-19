# Chapter 5 - Boundary Value Problems and Inverse Problems

This chapter applies everything so far to differential equations, where the unknown is a whole function T(x) or y(x) defined by an ODE plus boundary conditions. The tool is finite differences: replace derivatives with difference formulas on a grid, turning the ODE into a system of algebraic equations at the grid points. Notebooks 5.1 and 5.2 solve LINEAR boundary value problems this way (the grid equations form a matrix system A T = C solved in one shot); 5.1 also introduces tridiagonal/sparse solvers for speed. Notebook 5.3 handles a NONLINEAR BVP (a T^4 radiation term), which cannot be written as a matrix and instead needs NonlinearSolve. Notebooks 5.4 and 5.5 are INVERSE problems: instead of solving the BVP, find an unknown input (a material constant k, or a load model w(x)) so the BVP's solution matches a measurement - by differentiating through the linear solve and minimising a loss, exactly the pattern from 4.4.

## Notebooks in this chapter

### 5.1 Introduction to Boundary Value Problems  →  `05p01IntroBoundaryValueProblem.jl`

**Problem.** Find the temperature distribution T(x) along a heated plate governed by k T'' = -q(x) on [0,1] with T(0)=5 and T'(1)=0. The unknown is a function, obtained by solving the ODE numerically.

**Procedure.**

1. Discretise: put a grid on [0,1] and replace T'' with the central-difference formula (T[i-1]-2T[i]+T[i+1])/Delta^2.
2. Write one algebraic equation per interior grid point, plus rows for the two boundary conditions, giving a linear system A T = C.
3. Build A as a Tridiagonal matrix (and its sparse form) and the right-hand side C.
4. Solve with LinearSolve; compare the tridiagonal solver against a general sparse solver (tridiagonal is faster) via @time.
5. Plot T(x); then redo it for non-uniform heating q(x) = 1 + 2x by only changing C, and overlay both temperature profiles.

### 5.2 Boundary Value Problem (Another Example)  →  `05p02BoundaryValueProblem01.jl`

**Problem.** Solve a fuller linear BVP that has both first- and second-derivative terms: y'' + (2/x) y' - (2/x^2) y = 0 on [1,2] with y(1)=5, y(2)=3, and check against the known exact solution y = x + 4/x^2.

**Procedure.**

1. Discretise both derivatives: central difference for y'' and for y', evaluated at each grid point.
2. Substitute into the ODE so each interior point gives a row with three nonzero coefficients (i-1, i, i+1), plus two Dirichlet boundary rows, forming A y = C.
3. Fill the dense matrix A and vector C in a loop and solve with backslash.
4. Plot the finite-difference solution on top of the exact y = x + 4/x^2 to confirm agreement.
5. Rebuild A as a sparse matrix and solve with LinearSolve (UMFPACK) to show the speed-up that grows with grid size.

### 5.3 Nonlinear Boundary Value Problem  →  `05p03NonlinearBoundaryValueProblem.jl`

**Problem.** Solve a radiation heat-transfer BVP where the equation is nonlinear in T (a d*(T^4 - TR^4) term), so it cannot be written as a matrix system and must be solved iteratively.

**Procedure.**

1. Discretise the derivatives on a grid as usual, but keep the nonlinear T^4 term as-is.
2. Write the residual as a vector function HeatedDiskExample(T, p): each entry is 'left side minus right side' at a grid point (should be zero at the solution), with the two boundary conditions as the first and last entries.
3. Pack the physical constants (R, d, TR, Delta) into a parameter vector p.
4. Set up a NonlinearProblem with an initial guess and solve it with NonlinearSolve (which internally does Newton-Raphson using an automatically-computed Jacobian).
5. Plot the resulting temperature profile T(r).

### 5.4 Introduction to Inverse Problems for BVPs  →  `05p04ParameterEstimationBVP01.jl`

**Problem.** The forward BVP k T'' = -1 is known, but the material constant k is unknown. Given a measurement T(1)=3, find the k that makes the finite-difference solution reproduce it - an inverse problem.

**Procedure.**

1. Build the same tridiagonal finite-difference system as in 5.1 for a trial k and solve for T; read off T at the last point.
2. Sweep k over a range, re-solving each time, and plot T(x=1) versus k (and (T(1)-3)^2 versus k) to see roughly where the target is met (k about 0.22).
3. Define loss(p) = (T_end(k=p) - 3)^2, which must solve the linear system every evaluation.
4. Minimise the loss for k with the Optimization.jl library (Adam) using AutoForwardDiff to differentiate through the solve; a callback logs the loss.
5. Sanity-check by evaluating ForwardDiff.derivative of the loss near the solution.

### 5.5 Another Inverse Problem for a BVP  →  `05p05ParameterEstimationBVP02.jl`

**Problem.** Given measured beam-deflection data (x_i, y_i), find the unknown load distribution w(x) = a0 + a1*x in the beam equation y'' + 4y = w(x) so the model's deflection best fits the data. The model is defined implicitly through solving the BVP.

**Procedure.**

1. Read the (x, y) data from Data01.csv and plot it.
2. Discretise y'' + 4y = w(x) into a tridiagonal system A y = C, where C holds the load model a0 + a1*x at each interior point and 0 at the boundaries.
3. Define loss(p, ydata): build C from p = [a0, a1], solve A y = C, and return the squared error between the solved deflection and the data - so each evaluation solves the BVP.
4. Minimise the loss over (a0, a1) with Optimization.jl (Adam, AutoForwardDiff), logging losses via a callback.
5. Plot the fitted deflection over the data and the recovered load w(x) from the optimal coefficients.
