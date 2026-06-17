# Chapter 2 - Optimisation (Finding the Minimum of a Function)

This chapter is all about finding the minimum of a function, first by calling Julia's optimisation libraries, then by coding the same algorithms by hand so you understand what the libraries do. Notebooks 2.1 and 2.2 use the Optimization.jl / Optim libraries: 2.1 covers FIRST-order methods that only need the gradient (Gradient Descent, Momentum, Nesterov Momentum, Adam), and 2.2 covers SECOND-order methods that also use curvature (Newton, BFGS, LBFGS). Notebooks 2.3-2.5 then re-implement these update rules from scratch with explicit gradients: 2.3 does the first-order methods on 1-D functions, 2.4 extends them to 2-D functions (with gradient-arrow plots), and 2.5 builds the Newton (second-order) method and shows its danger of converging to a maximum. The same test functions recur throughout: a simple quadratic, (x^3-x)^2, the Shewchuk quadratic, Rosenbrock, and Himmelblau.

## Notebooks in this chapter

### 2.1 Basic Optimization in Julia (first-order library methods)  →  `02p01BasicOptimizationInJulia.jl`

**Problem.** Find the minimum of functions using Julia's Optimization.jl library with the four common first-order (gradient-only) optimisers: Gradient Descent, Momentum, Nesterov Momentum, and Adam. Compare how each one converges.

**Procedure.**

1. Load Optimization + OptimizationOptimisers + Zygote (for automatic gradients) + CairoMakie.
2. Define Point1D/Point2D structs and callback functions that record every intermediate guess so the convergence path can be plotted.
3. The library expects the objective as objective(x, p) with x a vector; since our functions have no parameters p, wrap them as (x,p)->f(x).
4. Standard three-step recipe: OptimizationFunction (function + AutoZygote gradient) -> OptimizationProblem (with initial guess x0) -> solve(prob, optimiser; maxiters, callback).
5. Run Example 2.1.1 (1-D quadratic) with Descent, Momentum, Nesterov, and Adam in turn, overlaying each convergence path to compare speed and overshoot.
6. Repeat for Example 2.1.2 (2-D Shewchuk quadratic) and Example 2.1.3 (Rosenbrock), reading the final answer from sol.u and status from sol.retcode.

### 2.2 Basic Optimization in Julia Continued (second-order library methods)  →  `02p02BasicOptimizationInJuliaContinued.jl`

**Problem.** Find the minimum using second-order library methods that use curvature (Hessian) information: Newton, BFGS, and LBFGS. These usually converge in very few iterations but need second-derivative information.

**Procedure.**

1. Load Optimization + OptimizationOptimJL + ForwardDiff (and Zygote) for derivatives, plus CairoMakie.
2. Reuse the same Point/callback machinery and the same OptimizationFunction -> OptimizationProblem -> solve recipe, now passing Optim.Newton(), Optim.BFGS(), Optim.LBFGS() as the solver.
3. Example 2.2.1 on (x^3-x)^2: Newton has no tunable learning rate and converges in ~3 iterations; note that Newton and BFGS from the same start can land on different minima.
4. Example 2.2.2 (Shewchuk quadratic) and 2.2.3 (Rosenbrock): overlay Newton/BFGS/LBFGS paths and print sol.u, sol.objective, sol.retcode.

### 2.3 Gradient Based Methods (coded from scratch, 1-D)  →  `02p03GradientBasedMethods.jl`

**Problem.** Understand the first-order optimisers by implementing their update equations yourself (no library) for 1-D functions, using an explicitly written gradient.

**Procedure.**

1. Write the function f(x) and its gradient df/dx by hand.
2. Gradient Descent: loop x <- x - alpha*g and store each guess; watch it slow down near the minimum as the gradient shrinks.
3. Momentum: keep a velocity v <- beta*v - alpha*g, then x <- x + v, so past gradients are remembered (beta=0 recovers plain gradient descent).
4. Nesterov Momentum: same as Momentum but evaluate the gradient at the look-ahead point x + beta*v, which reduces overshoot.
5. Overlay the three convergence curves; then repeat on g(x) = -exp(-x^2) where the tiny far-field gradient makes plain gradient descent very slow.

### 2.4 Multivariate Examples (coded from scratch, 2-D)  →  `02p04MultivariateExamples.jl`

**Problem.** Extend the hand-coded first-order methods to functions of two variables, where the gradient is a 2-vector, and visualise why 'follow the negative gradient' works.

**Procedure.**

1. Write each 2-D function together with a Gradient function returning [df/dx1, df/dx2].
2. Draw a filled contour plot and overlay arrows of the negative gradient to show they all point toward the minimum.
3. Example 2.4.1 (bowl x1^2+x2^2): run gradient descent from a start point and plot the path converging to (0,0).
4. Example 2.4.2 (Shewchuk quadratic): run Gradient Descent, then Momentum, then Nesterov Momentum (all in vector form) and compare their paths on the contour plot.

### 2.5 Second Order Methods (Newton, coded from scratch)  →  `02p05SecondOrderMethods.jl`

**Problem.** Build the Newton method by hand using the second derivative (Hessian), see why it converges so fast, and understand its key danger: it seeks any point where the gradient is zero, so it can converge to a maximum.

**Procedure.**

1. Write h(x), its gradient, and its Hessian (second derivative) explicitly.
2. Derive Newton from a 2nd-order Taylor fit: x <- x - g/H; illustrate one step graphically (fit a parabola at x0, jump to its minimum).
3. Run Newton on (x^3-x)^2 and compare its ~3-iteration convergence against Nesterov Momentum.
4. Show the pitfall: starting near a maximum, Newton converges to the maximum while Nesterov Momentum still reaches the true minimum.
5. Generalise to N-D: the update becomes x <- x - inv(H)*g using the Hessian matrix; apply it to the Shewchuk quadratic and the Himmelblau function (with analytic gradient and Hessian).
